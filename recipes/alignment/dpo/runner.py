# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""DPO workflow using global pair batches and a fixed initial reference."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

import torch

from cortex_training.client import DEBUG_OPTIONS_ENV
from recipes.alignment.dpo.task import (
    PreferencePair,
    PreferenceTask,
    require_disjoint_prompts,
)
from recipes.logging import setup_logging
from recipes.utils import (
    build_renderer,
    collate_preference_pairs,
    compute_dpo_loss,
    forward_logprobs_payload,
    load_job_body,
    make_client,
    masked_row_logprob_sums,
    save_checkpoint,
    source_checkpoint_info,
    dpo_payload,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)

_RECIPE_DIR = Path(__file__).resolve().parent
_CONFIG_SEARCH_DIRS = (_RECIPE_DIR, _RECIPE_DIR / "configs")
_TERMINAL_JOB_STATES = {"failed", "done", "cancelled", "canceled", "terminated"}


class Config(Protocol):
    config: str
    dataset: str
    dataset_split: str
    eval_dataset: str | None
    eval_split: str
    max_eval_pairs: int
    dpo_beta: float
    max_steps: int | None
    n_train_pairs: int | None
    epochs: int
    pairs_per_batch: int
    data_seed: int
    dataset_revision: str | None
    eval_every: int
    lr_schedule: str
    warmup_ratio: float
    tokenizer_revision: str | None
    source_job_id: str | None
    checkpoint_id: str | None
    resume_state: str | None
    log_path: str
    min_eval_accuracy: float
    initial_loss_tolerance: float
    renderer_name: str | None
    enable_thinking: bool
    job_config: str
    debug_image_tag: str | None
    max_total_seconds: float


def job_body(config: Config) -> dict[str, Any]:
    body = load_job_body(config.job_config, search_dirs=_CONFIG_SEARCH_DIRS)
    validate_job_body(body)
    if config.debug_image_tag:
        body["debug"] = {"job": {"image_tag": config.debug_image_tag}}
    return body


def validate_job_body(body: Mapping[str, Any]) -> None:
    """Validate required data-parallel DPO batching constraints."""
    sub_jobs = body.get("sub_job_configs")
    if not isinstance(sub_jobs, list) or len(sub_jobs) != 1:
        raise ValueError("DPO requires exactly one training sub-job")
    sub_job = sub_jobs[0]
    if not isinstance(sub_job, Mapping) or sub_job.get("job_type") != "training":
        raise ValueError("DPO supports one training sub-job and no other sub-jobs")
    training = sub_job.get("training_config")
    if not isinstance(training, Mapping):
        raise ValueError("DPO training sub-job needs training_config")
    for name in ("n_gpus", "train_batch_size"):
        value = training.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"DPO {name} must be a positive integer")
    if training.get("sp_size") not in (None, 1) or training.get("ep_size") not in (
        None,
        0,
    ):
        raise ValueError("DPO supports pure data parallelism only")
    if (training.get("ds_config") or {}).get("gradient_accumulation_steps", 1) != 1:
        raise ValueError(
            "DPO packed accumulation requires DeepSpeed gradient_accumulation_steps=1"
        )
    max_seq_len = training.get("max_seq_len")
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len < 2
    ):
        raise ValueError("DPO max_seq_len must be an integer of at least 2")
    mb_spec = training.get("mb_spec")
    max_tokens_per_mb = (
        mb_spec.get("max_tokens_per_mb") if isinstance(mb_spec, Mapping) else None
    )
    if (
        isinstance(max_tokens_per_mb, bool)
        or not isinstance(max_tokens_per_mb, int)
        or max_tokens_per_mb < max_seq_len
    ):
        raise ValueError("DPO max_tokens_per_mb must fit one complete sequence")


def _extract_logprobs(
    result: Mapping[str, Any],
    expected_shape: tuple[int, int],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    candidates: list[Any] = [result.get("logprobs")]
    for container_name in ("batch", "post_process_outputs", "loss_fn_outputs"):
        container = result.get(container_name)
        if isinstance(container, Mapping):
            candidates.append(container.get("logprobs"))
    value = next((candidate for candidate in candidates if candidate is not None), None)
    if value is None:
        raise RuntimeError("operation result did not contain per-token logprobs")
    if not torch.is_tensor(value) and attention_mask is not None:
        if len(value) != expected_shape[0]:
            raise RuntimeError("operation logprobs row count does not match input")
        rows = []
        for row, length in zip(value, attention_mask.sum(dim=1).tolist(), strict=True):
            row_tensor = torch.as_tensor(row, dtype=torch.float32).reshape(-1)
            if row_tensor.numel() not in {
                int(length) - 1,
                int(length),
                expected_shape[1] - 1,
                expected_shape[1],
            }:
                raise RuntimeError("operation logprobs row length does not match input")
            rows.append(
                torch.nn.functional.pad(
                    row_tensor, (0, expected_shape[1] - row_tensor.numel())
                )
            )
        tensor = torch.stack(rows)
    else:
        tensor = (
            value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
        )
    tensor = tensor.to(torch.float32)
    if tensor.shape == (expected_shape[0], expected_shape[1] - 1):
        tensor = torch.nn.functional.pad(tensor, (0, 1))
    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"operation logprobs shape {tuple(tensor.shape)} does not match {expected_shape}"
        )
    return tensor


def _poll_submitted(client: Any, job_id: str, submitted: Any) -> Mapping[str, Any]:
    request_id = (
        submitted.get("request_id") if isinstance(submitted, Mapping) else submitted
    )
    result = (
        client.poll_request(job_id, request_id)
        if isinstance(request_id, str) and request_id
        else submitted
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("operation returned no result")
    return result


def _forward_logprobs(
    client: Any,
    job_id: str,
    kwargs: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    *,
    pad_token_id: int,
) -> torch.Tensor:
    submitted = client.forward(
        job_id,
        forward_logprobs_payload(
            kwargs,
            context,
            pad_token_id=pad_token_id,
        ),
        sub_job_id=f"{job_id}:training:0",
        sub_job_type="training",
    )
    result = _poll_submitted(client, job_id, submitted)
    _require_finite_result(result, "forward", allow_masked_logprob_nonfinite=True)
    return _extract_logprobs(
        result, tuple(kwargs["input_ids"].shape), kwargs["attention_mask"]
    )


def _require_finite_result(
    result: Mapping[str, Any],
    operation: str,
    *,
    allow_masked_logprob_nonfinite: bool = False,
) -> None:
    def visit(value: Any, name: str = "") -> None:
        if allow_masked_logprob_nonfinite and name == "logprobs":
            return
        if torch.is_tensor(value):
            if not torch.isfinite(value).all().item():
                raise RuntimeError(f"{operation} returned a non-finite tensor")
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"{operation} returned a non-finite value")
        elif isinstance(value, Mapping):
            for nested_name, nested in value.items():
                visit(nested, str(nested_name))
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, name)

    visit(result)


def _wait_for_terminal_job(client: Any, job_id: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    delay = max(float(client.poll_interval), 0.1)
    while time.monotonic() < deadline:
        job = client.get_job(job_id)
        status = client._normalize_job_status(job.get("status"))
        if status in _TERMINAL_JOB_STATES:
            logger.info("Job %s reached terminal state %s", job_id, status)
            return
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * float(client.poll_backoff_multiplier), 10.0)
    raise TimeoutError(
        f"job {job_id} did not reach a terminal state after cancellation"
    )


@contextlib.contextmanager
def running_dpo_job(client: Any, body: dict[str, Any]) -> Iterator[str]:
    job_id = client.create_job_from_body(body)["job_id"]
    logger.info("Recorded DPO job id %s", job_id)
    try:
        client.wait_for_job(job_id)
        logger.info("Job %s is running", job_id)
        yield job_id
    finally:
        try:
            client.cancel_job(job_id)
            logger.info("Requested cancellation for job %s", job_id)
        except Exception:
            job = client.get_job(job_id)
            status = client._normalize_job_status(job.get("status"))
            if status not in _TERMINAL_JOB_STATES:
                logger.exception("Failed to cancel non-terminal job %s", job_id)
                raise
        _wait_for_terminal_job(client, job_id)


def _check_wall_time(started_at: float, limit: float) -> None:
    elapsed = time.monotonic() - started_at
    if elapsed > limit:
        raise TimeoutError(f"DPO run exceeded max_total_seconds={limit:g}")


def _batch_indices(indices: list[int], batch_size: int, n_gpus: int) -> list[list[int]]:
    if batch_size <= 0 or min(len(indices), batch_size) < n_gpus:
        raise ValueError("each DPO batch must supply at least one pair per GPU")
    batches = [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]
    min_pairs = n_gpus
    if len(indices) < len(batches) * min_pairs:
        raise ValueError(
            "cannot supply one pair per GPU without exceeding pairs_per_batch"
        )
    if len(batches[-1]) < min_pairs:
        sizes = [len(batch) for batch in batches]
        deficit = min_pairs - sizes[-1]
        # Shift boundaries rather than merge rows: retain input order and the
        # configured ceiling, borrowing from earlier batches only as needed.
        for index in range(len(sizes) - 2, -1, -1):
            moved = min(deficit, sizes[index] - min_pairs)
            sizes[index] -= moved
            sizes[-1] += moved
            deficit -= moved
            if not deficit:
                break
        batches = []
        start = 0
        for size in sizes:
            batches.append(indices[start : start + size])
            start += size
    return batches


def _batches(pairs, *, pad_token_id, max_seq_len, pairs_per_batch=1, n_gpus=1):
    for indices in _batch_indices(list(range(len(pairs))), pairs_per_batch, n_gpus):
        yield collate_preference_pairs(
            [pairs[index].as_tuple() for index in indices],
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
        )


def _score_pairs(
    client,
    job_id,
    pairs,
    *,
    pad_token_id,
    max_seq_len,
    pairs_per_batch,
    n_gpus,
    deadline=None,
):
    sums = []
    for kwargs, context in _batches(
        pairs,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
        pairs_per_batch=pairs_per_batch,
        n_gpus=n_gpus,
    ):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("DPO run exceeded its wall-time limit")
        logprobs = _forward_logprobs(
            client, job_id, kwargs, context, pad_token_id=pad_token_id
        )
        sums.append(masked_row_logprob_sums(logprobs, context["loss_mask"]))
    return torch.cat(sums)


def _evaluate(
    client,
    job_id,
    pairs,
    reference_sums,
    *,
    beta,
    pad_token_id,
    max_seq_len,
    pairs_per_batch,
    n_gpus,
    deadline=None,
):
    totals = {
        name: 0.0
        for name in (
            "loss",
            "accuracy",
            "margin",
            "chosen_logprob",
            "rejected_logprob",
            "chosen_tokens",
            "rejected_tokens",
        )
    }
    for indices in _batch_indices(list(range(len(pairs))), pairs_per_batch, n_gpus):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("DPO run exceeded its wall-time limit")
        kwargs, context = collate_preference_pairs(
            [pairs[index].as_tuple() for index in indices],
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
        )
        row_indices = [row for index in indices for row in (2 * index, 2 * index + 1)]
        logprobs = _forward_logprobs(
            client, job_id, kwargs, context, pad_token_id=pad_token_id
        )
        loss, metrics = compute_dpo_loss(
            logprobs, reference_sums[row_indices], context["loss_mask"], beta=beta
        )
        sums = masked_row_logprob_sums(logprobs, context["loss_mask"])
        totals["loss"] += float(loss) * len(indices)
        totals["accuracy"] += metrics["dpo/accuracy"] * len(indices)
        totals["margin"] += metrics["dpo/margin"] * len(indices)
        totals["chosen_logprob"] += float(sums[0::2].sum())
        totals["rejected_logprob"] += float(sums[1::2].sum())
        totals["chosen_tokens"] += float(context["loss_mask"][0::2].sum())
        totals["rejected_tokens"] += float(context["loss_mask"][1::2].sum())
    metrics = {f"eval/{name}": value / len(pairs) for name, value in totals.items()}
    metrics["eval/chosen_nll_per_token"] = (
        -totals["chosen_logprob"] / totals["chosen_tokens"]
    )
    metrics["eval/chosen_logprob_change"] = metrics["eval/chosen_logprob"] - float(
        reference_sums[0::2].mean()
    )
    metrics["eval/rejected_logprob_change"] = metrics["eval/rejected_logprob"] - float(
        reference_sums[1::2].mean()
    )
    metrics["eval/unscaled_margin"] = metrics["eval/margin"] / beta
    return metrics


def _learning_rate(
    base_lr: float, step: int, total_steps: int, schedule: str, warmup_ratio: float
) -> float:
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return base_lr
    # Use the peak on the first post-warmup update and reach zero after the run.
    return base_lr * (total_steps - step) / max(1, total_steps - warmup_steps)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _data_identity(pairs: list[PreferencePair]) -> str:
    rows = [
        (
            pair.row_id,
            pair.prompt_fingerprint,
            pair.chosen.input_ids,
            pair.chosen.loss_mask,
            pair.rejected.input_ids,
            pair.rejected.loss_mask,
        )
        for pair in pairs
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _update(
    client,
    job_id,
    kwargs,
    context,
    reference_sums,
    *,
    config,
    pad_token_id,
    learning_rate,
):
    backward = _poll_submitted(
        client,
        job_id,
        client.forward_backward(
            job_id,
            dpo_payload(
                kwargs,
                context,
                reference_sums,
                beta=config.dpo_beta,
                pad_token_id=pad_token_id,
            ),
        ),
    )
    _require_finite_result(backward, "backward")
    loss = backward.get("avg_loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
    ):
        raise RuntimeError("backward did not report a finite scalar DPO loss")
    stepped = _poll_submitted(
        client, job_id, client.step(job_id, learning_rate=learning_rate)
    )
    _require_finite_result(stepped, "optimizer")
    acknowledgement = stepped.get("update_successful")
    if type(acknowledgement) not in (int, float) or acknowledgement != 1:
        raise RuntimeError("optimizer did not report a successful update")
    result = {
        "dpo/loss": float(loss),
        "train/learning_rate": learning_rate,
        "train/pairs": float(kwargs["input_ids"].shape[0] // 2),
    }
    grad_norm = stepped.get("grad_norm")
    if grad_norm is not None:
        result["train/grad_norm"] = float(grad_norm)
    return result


def _validate_saved_state(state: Any) -> None:
    """Reject incomplete recovery boundaries before allocating a replacement job."""
    if not isinstance(state, Mapping):
        raise ValueError("resume state must be a complete saved object")
    if not isinstance(state.get("signature"), Mapping) or not isinstance(
        state.get("references"), Mapping
    ):
        raise ValueError("resume state requires signature and reference scores")
    next_step = state.get("next_step")
    if isinstance(next_step, bool) or not isinstance(next_step, int):
        raise ValueError("resume next_step must be an integer")
    for name in ("initial_snapshot", "resumable", "best_checkpoint", "best_resumable"):
        checkpoint = state.get(name)
        if not isinstance(checkpoint, Mapping) or any(
            not isinstance(checkpoint.get(key), str) or not checkpoint[key]
            for key in ("source_job_id", "checkpoint_id")
        ):
            raise ValueError(
                f"resume state requires complete {name} checkpoint identity"
            )
    for name in ("initial_loss", "best_loss"):
        value = state.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"resume state requires finite {name}")


def train(config: Config, task: PreferenceTask) -> dict[str, Any]:
    """Train paired DPO, checkpoint complete updates, and release the owned job."""
    quality = config.n_train_pairs is not None
    n_train = config.n_train_pairs if quality else config.max_steps
    if n_train is None or n_train <= 0 or config.max_eval_pairs <= 0:
        raise ValueError(
            "n_train_pairs (or smoke max_steps) and max_eval_pairs must be positive"
        )
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps must be positive or None")
    if config.epochs <= 0 or config.eval_every <= 0 or config.pairs_per_batch <= 0:
        raise ValueError("epochs, eval_every, and pairs_per_batch must be positive")
    if config.lr_schedule not in {"constant", "linear"}:
        raise ValueError("lr_schedule must be constant or linear")
    if not math.isfinite(config.warmup_ratio) or not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    for name in (
        "dpo_beta",
        "initial_loss_tolerance",
        "max_total_seconds",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not math.isfinite(config.min_eval_accuracy)
        or not 0 <= config.min_eval_accuracy < 1
    ):
        raise ValueError("min_eval_accuracy must be in [0, 1)")
    if config.resume_state and (config.source_job_id or config.checkpoint_id):
        raise ValueError("resume_state and warm-start source are mutually exclusive")
    state_path = (
        Path(config.resume_state).expanduser()
        if config.resume_state
        else Path(config.log_path).expanduser() / "dpo-state.json"
    )
    state = json.loads(state_path.read_text()) if config.resume_state else None
    if state is not None:
        _validate_saved_state(state)
    if config.debug_image_tag:
        os.environ[DEBUG_OPTIONS_ENV] = "1"
    body = job_body(config)
    sub_job = body["sub_job_configs"][0]
    if state is not None and "source_checkpoint_info" not in sub_job:
        initial_source = state["signature"]["job"]["sub_job_configs"][0].get(
            "source_checkpoint_info"
        )
        if initial_source:
            sub_job["source_checkpoint_info"] = initial_source
    source = source_checkpoint_info(config.source_job_id, config.checkpoint_id)
    if source:
        if "source_checkpoint_info" in sub_job:
            raise ValueError("specify the warm-start source in either CLI or job JSON")
        sub_job["source_checkpoint_info"] = source
    training = sub_job["training_config"]
    if "source_checkpoint_info" in sub_job:
        # Starting a new DPO stage restores weights but never the old optimizer.
        training["load_optimizer_states"] = False
    max_seq_len, n_gpus = int(training["max_seq_len"]), int(training["n_gpus"])
    base_lr = float(training["optimizer"]["lr"])
    tokenizer, renderer, resolved_renderer = build_renderer(
        sub_job["model_name"],
        renderer_name=config.renderer_name,
        enable_thinking=config.enable_thinking,
        tokenizer_revision=config.tokenizer_revision,
    )
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_token_id is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id")

    records = {}
    load_options = dict(
        renderer=renderer, max_seq_len=max_seq_len, seed=config.data_seed
    )
    test_revision = (
        config.dataset_revision
        if config.eval_dataset in (None, config.dataset)
        else None
    )
    test_pairs = task.load_pairs(
        config.eval_dataset or config.dataset,
        config.eval_split,
        n_pairs=config.max_eval_pairs,
        revision=test_revision,
        **load_options,
    )
    records["test"] = dict(task.load_record)
    reserved = {pair.prompt_fingerprint for pair in test_pairs}
    if quality:
        dev_pairs = task.load_pairs(
            config.dataset,
            config.dataset_split,
            n_pairs=config.max_eval_pairs,
            exclude_prompts=reserved,
            revision=config.dataset_revision,
            **load_options,
        )
        records["development"] = dict(task.load_record)
        reserved.update(pair.prompt_fingerprint for pair in dev_pairs)
    else:
        dev_pairs = test_pairs
    train_pairs = task.load_pairs(
        config.dataset,
        config.dataset_split,
        n_pairs=n_train,
        exclude_prompts=reserved,
        revision=config.dataset_revision,
        **load_options,
    )
    records["train"] = dict(task.load_record)
    require_disjoint_prompts(train_pairs, dev_pairs)
    require_disjoint_prompts(train_pairs, test_pairs)
    pair_budget = training["mb_spec"]["max_tokens_per_mb"]
    if any(
        len(pair.chosen.input_ids) + len(pair.rejected.input_ids) > pair_budget
        for pair in train_pairs
    ):
        raise ValueError("a complete preference pair exceeds max_tokens_per_mb")
    updates = []
    for epoch in range(config.epochs):
        indices = list(range(len(train_pairs)))
        if config.dataset != "smoke":
            random.Random(config.data_seed + epoch).shuffle(indices)
        updates.extend(_batch_indices(indices, config.pairs_per_batch, n_gpus))
    if config.max_steps is not None:
        updates = updates[: config.max_steps]
    for pairs in (dev_pairs, test_pairs):
        _batch_indices(list(range(len(pairs))), config.pairs_per_batch, n_gpus)
    signature = {
        "job": deepcopy(body),
        "data": {
            name: _data_identity(pairs)
            for name, pairs in (
                ("train", train_pairs),
                ("development", dev_pairs),
                ("test", test_pairs),
            )
        },
        "dataset_records": records,
        "renderer": resolved_renderer,
        "tokenizer_revision": config.tokenizer_revision,
        "epochs": config.epochs,
        "pairs_per_batch": config.pairs_per_batch,
        "eval_every": config.eval_every,
        "data_seed": config.data_seed,
        "steps": len(updates),
        "update_plan_sha256": hashlib.sha256(json.dumps(updates).encode()).hexdigest(),
        "beta": config.dpo_beta,
        "lr_schedule": config.lr_schedule,
        "warmup_ratio": config.warmup_ratio,
    }
    if state is not None:
        if state["signature"] != signature:
            raise ValueError(
                "resume data/configuration/GPU count does not match the saved state"
            )
        if not 0 <= state["next_step"] <= len(updates):
            raise ValueError("resume next_step is outside the training schedule")
        if (
            state["next_step"] != len(updates)
            and state["next_step"] % config.eval_every
        ):
            raise ValueError("resume next_step is not a complete saved boundary")
        for name, pairs in (
            ("train", train_pairs),
            ("development", dev_pairs),
            ("test", test_pairs),
        ):
            scores = torch.tensor(state["references"][name], dtype=torch.float32)
            if scores.shape != (2 * len(pairs),) or not torch.isfinite(scores).all():
                raise ValueError("resume reference scores are incomplete or non-finite")
        sub_job["source_checkpoint_info"] = state["resumable"]
        training["load_optimizer_states"] = True
    _write_json(Path(config.log_path).expanduser() / "run.json", signature)
    client = make_client(config.config, poll_timeout=600.0)
    started = time.monotonic()
    options = dict(
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
        pairs_per_batch=config.pairs_per_batch,
        n_gpus=n_gpus,
        deadline=started + config.max_total_seconds,
    )
    with running_dpo_job(client, body) as job_id:
        ml_logger = setup_logging(config, client=client, job_id=job_id)
        try:
            if state is None:
                initial = save_checkpoint(client, job_id)
                if not initial["checkpoint_id"]:
                    raise RuntimeError("initial weights snapshot has no checkpoint ID")
                references = {}
                for name, pairs in (
                    ("train", train_pairs),
                    ("development", dev_pairs),
                    ("test", test_pairs),
                ):
                    _check_wall_time(started, config.max_total_seconds)
                    references[name] = _score_pairs(
                        client, job_id, pairs, **options
                    ).tolist()
                state = {
                    "signature": signature,
                    "next_step": 0,
                    "references": references,
                    "initial_snapshot": {
                        "source_job_id": job_id,
                        "checkpoint_id": initial["checkpoint_id"],
                    },
                    "best_loss": None,
                    "best_checkpoint": None,
                }
                # Check the complete development objective before training.
                initial_metrics = _evaluate(
                    client,
                    job_id,
                    dev_pairs,
                    torch.tensor(references["development"]),
                    beta=config.dpo_beta,
                    **options,
                )
                state["initial_loss"] = initial_metrics["eval/loss"]
                if (
                    abs(state["initial_loss"] - math.log(2))
                    > config.initial_loss_tolerance
                ):
                    raise RuntimeError("step-zero DPO loss failed ln(2) invariant")
                ml_logger.log_metrics(metrics=initial_metrics, step=0)
                initial_recovery = save_checkpoint(
                    client, job_id, checkpoint_type="resumable"
                )
                if not initial_recovery["checkpoint_id"]:
                    raise RuntimeError(
                        "initial recovery checkpoint has no checkpoint ID"
                    )
                state["resumable"] = {
                    "source_job_id": job_id,
                    "checkpoint_id": initial_recovery["checkpoint_id"],
                }
                state["best_loss"] = initial_metrics["eval/loss"]
                state["best_checkpoint"] = dict(state["initial_snapshot"])
                state["best_resumable"] = dict(state["resumable"])
                _write_json(state_path, state)
            references = {
                name: torch.tensor(values, dtype=torch.float32)
                for name, values in state["references"].items()
            }

            for step in range(state["next_step"], len(updates)):
                _check_wall_time(started, config.max_total_seconds)
                indices = updates[step]
                kwargs, context = collate_preference_pairs(
                    [train_pairs[index].as_tuple() for index in indices],
                    pad_token_id=pad_token_id,
                    max_seq_len=max_seq_len,
                )
                rows = [row for index in indices for row in (2 * index, 2 * index + 1)]
                metrics = _update(
                    client,
                    job_id,
                    kwargs,
                    context,
                    references["train"][rows],
                    config=config,
                    pad_token_id=pad_token_id,
                    learning_rate=_learning_rate(
                        base_lr,
                        step,
                        len(updates),
                        config.lr_schedule,
                        config.warmup_ratio,
                    ),
                )
                ml_logger.log_metrics(
                    metrics={
                        **metrics,
                        "progress/step": float(step + 1),
                        "time/total": time.monotonic() - started,
                    },
                    step=step + 1,
                )
                if (step + 1) % config.eval_every == 0 or step + 1 == len(updates):
                    development = _evaluate(
                        client,
                        job_id,
                        dev_pairs,
                        references["development"],
                        beta=config.dpo_beta,
                        **options,
                    )
                    ml_logger.log_metrics(metrics=development, step=step + 1)
                    weights = save_checkpoint(client, job_id)
                    resumable = save_checkpoint(
                        client, job_id, checkpoint_type="resumable"
                    )
                    if not weights["checkpoint_id"] or not resumable["checkpoint_id"]:
                        raise RuntimeError(
                            "checkpoint save did not return a checkpoint ID"
                        )
                    state["next_step"] = step + 1
                    state["resumable"] = {
                        "source_job_id": job_id,
                        "checkpoint_id": resumable["checkpoint_id"],
                    }
                    current_checkpoint = {
                        "source_job_id": job_id,
                        "checkpoint_id": weights["checkpoint_id"],
                    }
                    if not quality or development["eval/loss"] < state["best_loss"]:
                        state["best_loss"] = development["eval/loss"]
                        state["best_checkpoint"] = current_checkpoint
                        state["best_resumable"] = dict(state["resumable"])
                    _write_json(state_path, state)

            # Smoke gates measure the same first pair before/after all updates.
            if not quality:
                first_kwargs, first_context = collate_preference_pairs(
                    [train_pairs[0].as_tuple()],
                    pad_token_id=pad_token_id,
                    max_seq_len=max_seq_len,
                )
                final = _forward_logprobs(
                    client,
                    job_id,
                    first_kwargs,
                    first_context,
                    pad_token_id=pad_token_id,
                )
                final_loss, _ = compute_dpo_loss(
                    final,
                    references["train"][:2],
                    first_context["loss_mask"],
                    beta=config.dpo_beta,
                )
                if float(final_loss) >= state["initial_loss"]:
                    raise RuntimeError("DPO loss did not decrease")
            else:
                final_loss = state["best_loss"]
                # Select using development only, then score the final test once.
                selected = state["best_resumable"]
                loaded = _poll_submitted(
                    client,
                    job_id,
                    client.load(
                        job_id,
                        checkpoint_id=selected["checkpoint_id"],
                        source_job_id=selected["source_job_id"],
                    ),
                )
                _require_finite_result(loaded, "load selected checkpoint")
            evaluation = _evaluate(
                client,
                job_id,
                test_pairs,
                references["test"],
                beta=config.dpo_beta,
                **options,
            )
            if not quality and evaluation["eval/accuracy"] <= config.min_eval_accuracy:
                raise RuntimeError("held-out preference accuracy threshold failed")
            ml_logger.log_metrics(
                metrics={
                    name.replace("eval/", "test/"): value
                    for name, value in evaluation.items()
                },
                step=len(updates),
            )
            result = {
                "job_id": job_id,
                "checkpoint_id": state["best_checkpoint"]["checkpoint_id"],
                "checkpoint_source_job_id": state["best_checkpoint"]["source_job_id"],
                "initial_snapshot": state["initial_snapshot"],
                "resume_state": str(state_path),
                "initial_loss": state["initial_loss"],
                "final_loss": float(final_loss),
                "held_out_accuracy": evaluation["eval/accuracy"],
                "test_metrics": evaluation,
                "steps": len(updates),
                "finite": True,
            }
            _write_json(Path(config.log_path).expanduser() / "result.json", result)
            return result
        finally:
            ml_logger.close()
