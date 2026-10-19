import copy
import json
import math
from pathlib import Path

import pytest
import torch

dpo_runner = pytest.importorskip("recipes.alignment.dpo.runner", exc_type=ImportError)
dpo_task = pytest.importorskip("recipes.alignment.dpo.task", exc_type=ImportError)
recipe_utils = pytest.importorskip("recipes.utils", exc_type=ImportError)
wire = pytest.importorskip("cortex_training.wire")

REPO_ROOT = Path(__file__).resolve().parents[1]
DPO_CONFIG_DIR = REPO_ROOT / "recipes" / "alignment" / "dpo" / "configs"


class _Tokens:
    def __init__(self, values):
        self._values = values

    def to_ints(self):
        return self._values


class _Renderer:
    def build_supervised_example(self, messages, train_on_what):
        del train_on_what
        final = str(messages[-1]["content"])
        suffix = 7 if final == "chosen" else 9
        return _Tokens([1, 2, suffix]), torch.tensor([0.0, 0.0, 1.0])


def test_dpo_math_pairing_masking_and_gradient_direction(monkeypatch):
    logprobs = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0], [-1.5, -2.5, -3.5, -4.5]],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    reference_sums = recipe_utils.masked_row_logprob_sums(logprobs, loss_mask)
    loss, metrics = recipe_utils.compute_dpo_loss(
        logprobs,
        reference_sums,
        loss_mask,
        beta=0.1,
    )
    assert float(loss) == pytest.approx(math.log(2.0), abs=1e-7)
    assert metrics["dpo/loss"] == pytest.approx(math.log(2.0), abs=1e-7)

    row = {
        "score_chosen": 2.0,
        "score_rejected": 1.0,
        "chosen": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "chosen"},
        ],
        "rejected": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "rejected"},
        ],
    }
    pair = dpo_task.PreferenceTask().build_pair(row, _Renderer(), max_seq_len=8)
    kwargs, context = recipe_utils.collate_preference_pairs(
        [pair.as_tuple()],
        pad_token_id=0,
        max_seq_len=8,
    )
    assert kwargs["input_ids"].tolist() == [[1, 2, 7], [1, 2, 9]]
    # loss_mask[i] scores input_ids[i + 1], so only each final assistant token
    # is trained and the terminal roll-around position remains masked.
    assert context["loss_mask"].tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

    mismatched = copy.deepcopy(row)
    mismatched["rejected"][0]["content"] = "different prompt"
    with pytest.raises(ValueError, match="share the same prompt"):
        dpo_task.PreferenceTask().build_pair(
            mismatched,
            _Renderer(),
            max_seq_len=8,
        )
    with pytest.raises(ValueError, match="genuinely separate split"):
        dpo_task.require_disjoint_prompts([pair], [pair])

    tied = copy.deepcopy(row)
    tied["rejected"][-1] = copy.deepcopy(tied["chosen"][-1])
    with pytest.raises(ValueError, match="responses must differ"):
        dpo_task.PreferenceTask().build_pair(tied, _Renderer(), max_seq_len=8)
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: [tied, row],
    )
    loaded = dpo_task.PreferenceTask().load_pairs(
        "ultrafeedback",
        "test",
        n_pairs=1,
        renderer=_Renderer(),
        max_seq_len=8,
    )
    assert loaded[0].as_tuple() == pair.as_tuple()
    assert loaded[0].row_id


def test_direct_loss_frame_and_one_gpu_recipe_contracts():
    kwargs = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]], dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "position_ids": torch.arange(3, dtype=torch.long).expand(2, -1),
        "use_cache": False,
    }
    context = {
        "input_ids": kwargs["input_ids"],
        "loss_mask": torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    }
    references = torch.tensor([-2.0, -3.0], requires_grad=True)
    decoded = wire.loads(
        recipe_utils.dpo_payload(
            kwargs, context, references, beta=0.2, pad_token_id=151643
        )
    )
    forward_decoded = wire.loads(
        recipe_utils.forward_logprobs_payload(kwargs, context, pad_token_id=151643)
    )
    assert decoded["processing"] == {
        "loss_fn": "dpo",
        "post": ["compute_logprobs"],
        "config": {"beta": 0.2},
    }
    ctx = decoded["context"]
    assert ctx["dpo_pair_ids"].dtype == torch.int64
    assert ctx["dpo_pair_ids"].tolist() == [0, 0]
    assert ctx["dpo_is_chosen"].dtype == torch.bool
    assert ctx["dpo_is_chosen"].tolist() == [True, False]
    assert ctx["loss_mask"].dtype == torch.bool
    assert ctx["labels"].dtype == torch.int64
    assert ctx["labels"].tolist() == [[2, 3, -100], [2, 4, -100]]
    assert ctx["dpo_reference_logps"].tolist() == [-2.0, -3.0]
    assert not ctx["dpo_reference_logps"].requires_grad
    assert "cu_seqlens" not in ctx and "logprob_weights" not in ctx
    assert (
        forward_decoded["context"]["labels"][:, :-1].tolist()
        == ctx["labels"][:, :-1].tolist()
    )

    configs = {
        path.name: json.loads(path.read_text())
        for path in sorted(DPO_CONFIG_DIR.glob("*.json"))
    }
    assert {
        "qwen3_0_6b_lora.json",
        "qwen3_8b_lora.json",
        "qwen3_8b_full.json",
    } <= configs.keys()
    for body in configs.values():
        dpo_runner.validate_job_body(body)
        assert len(body["sub_job_configs"]) == 1
        assert body["sub_job_configs"][0]["training_config"]["n_gpus"] in (1, 4)
    qwen8_lora = configs["qwen3_8b_lora.json"]
    qwen8_full = configs["qwen3_8b_full.json"]
    assert qwen8_lora["sub_job_configs"][0]["model_name"] == "Qwen/Qwen3-8B"
    assert "peft_config" in qwen8_lora["sub_job_configs"][0]["training_config"]
    assert "peft_config" not in qwen8_full["sub_job_configs"][0]["training_config"]

    unrestricted = copy.deepcopy(qwen8_lora)
    unrestricted["sub_job_configs"][0]["model_name"] = "example/custom-model"
    unrestricted["sub_job_configs"][0]["training_config"]["max_seq_len"] = 32768
    unrestricted["sub_job_configs"][0]["training_config"]["mb_spec"][
        "max_tokens_per_mb"
    ] = 65536
    unrestricted["sub_job_configs"][0]["training_config"]["peft_config"][
        "lora_dropout"
    ] = 0.1
    dpo_runner.validate_job_body(unrestricted)

    invalid = copy.deepcopy(qwen8_lora)
    invalid["sub_job_configs"][0]["training_config"]["n_gpus"] = 2
    dpo_runner.validate_job_body(invalid)
    invalid["sub_job_configs"][0]["training_config"]["n_gpus"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        dpo_runner.validate_job_body(invalid)
    invalid = copy.deepcopy(qwen8_lora)
    invalid["sub_job_configs"][0]["training_config"]["sp_size"] = 2
    with pytest.raises(ValueError, match="pure data parallelism"):
        dpo_runner.validate_job_body(invalid)
    invalid = copy.deepcopy(qwen8_lora)
    invalid["sub_job_configs"][0]["training_config"]["ds_config"][
        "gradient_accumulation_steps"
    ] = 2
    with pytest.raises(ValueError, match="packed accumulation"):
        dpo_runner.validate_job_body(invalid)
    invalid = copy.deepcopy(qwen8_lora)
    invalid["sub_job_configs"].append(copy.deepcopy(invalid["sub_job_configs"][0]))
    with pytest.raises(ValueError, match="exactly one"):
        dpo_runner.validate_job_body(invalid)

    # The compatibility catalog only models generic RL/SFT profiles and requires
    # production profiles to use GPU counts in multiples of eight. DPO therefore
    # remains represented by its validated one-GPU recipe metadata/configs.
    schema = json.loads((REPO_ROOT / "model-catalog" / "schema.json").read_text())
    schema_text = json.dumps(schema)
    assert '"dpo"' not in schema_text.lower()
    recipe_metadata = (
        REPO_ROOT / "recipes" / "alignment" / "dpo" / "recipe.yaml"
    ).read_text()
    assert "id: dpo-one-gpu" in recipe_metadata


def _preference_row(prompt):
    return {
        "score_chosen": 3.0,
        "score_rejected": 1.0,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "chosen"},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "rejected"},
        ],
    }


def test_public_data_labels_seed_and_prompt_partitions(monkeypatch):
    task = dpo_task.PreferenceTask()
    assert (
        task.normalize_row(
            {**_preference_row("tie"), "score_rejected": 3}, "ultrafeedback"
        )
        is None
    )
    argilla = {
        "instruction": "prompt",
        "chosen_response": "chosen",
        "rejected_response": "rejected",
        "chosen_avg_rating": 4,
        "rejected_avg_rating": 2,
    }
    assert task.normalize_row(argilla, "argilla")["chosen"][-1]["content"] == "chosen"
    assert task.normalize_row({**argilla, "chosen_avg_rating": 2}, "argilla") is None
    context = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "followup"},
    ]
    human = {"context": context, "response1": "one", "response2": "two"}
    for sign, chosen in [(-2, "one"), (2, "two")]:
        normalized = task.normalize_row(
            {**human, "overall_preference": sign}, "helpsteer3"
        )
        assert normalized["chosen"][:-1] == context
        assert normalized["chosen"][-1]["content"] == chosen
    assert task.normalize_row({**human, "overall_preference": 0}, "helpsteer3") is None
    with pytest.raises(ValueError, match="finite"):
        task.normalize_row({**human, "overall_preference": float("nan")}, "helpsteer3")
    rows = [_preference_row(f"prompt-{i}") for i in range(12)]
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return rows

    monkeypatch.setattr("datasets.load_dataset", load)
    options = dict(n_pairs=3, renderer=_Renderer(), max_seq_len=8, seed=17)
    first = task.load_pairs("ultrafeedback", "train", **options)
    assert task.load_pairs("ultrafeedback", "train", **options) == first
    second = task.load_pairs(
        "ultrafeedback",
        "train",
        exclude_prompts={p.prompt_fingerprint for p in first},
        **options,
    )
    dpo_task.require_disjoint_prompts(first, second)
    assert calls[0][1]["revision"] == dpo_task.DATASETS["ultrafeedback"][1]
    assert calls[0][1]["split"] == "train_prefs"
    assert (
        task.load_record["exclusions_before_selection"]["duplicate_or_reserved_prompt"]
        == 3
    )


def test_multiturn_masks_and_whole_pair_length_filter():
    class ConversationRenderer:
        def build_supervised_example(self, messages, train_on_what):
            assert (
                train_on_what == dpo_task.renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            return _Tokens([1, 2, 3, 4, 5, 6]), torch.tensor(
                [0, 0, 0, 0, 1, 1], dtype=torch.float32
            )

    context = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "followup"},
    ]
    row = {
        "chosen": context + [{"role": "assistant", "content": "chosen"}],
        "rejected": context + [{"role": "assistant", "content": "rejected"}],
    }
    pair = dpo_task.PreferenceTask().build_pair(
        row, ConversationRenderer(), max_seq_len=6
    )
    assert pair.chosen.loss_mask == [0, 0, 0, 1, 1, 0]
    with pytest.raises(OverflowError):
        dpo_task.PreferenceTask().build_pair(row, ConversationRenderer(), max_seq_len=5)


def test_partial_batches_schedules_and_ragged_logprobs():
    with pytest.raises(ValueError, match="exceeding pairs_per_batch"):
        dpo_runner._batch_indices(list(range(5)), 2, 2)
    with pytest.raises(ValueError, match="one pair per GPU"):
        dpo_runner._batch_indices([0], 1, 4)
    assert dpo_runner._learning_rate(1.0, 0, 128, "linear", 0.03) == 0.25
    assert dpo_runner._learning_rate(1.0, 4, 128, "linear", 0.03) == 1.0
    assert dpo_runner._learning_rate(1.0, 127, 128, "linear", 0.03) == pytest.approx(
        1 / 124
    )
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    result = {"post_process_outputs": {"logprobs": [[-1, -2, -3], [-4]]}}
    extracted = dpo_runner._extract_logprobs(result, (2, 4), mask)
    assert extracted.tolist() == [[-1, -2, -3, 0], [-4, 0, 0, 0]]
    with pytest.raises(RuntimeError, match="row length"):
        dpo_runner._extract_logprobs({"logprobs": [[-1], [-2]]}, (2, 4), mask)


class _TrainingClient:
    """In-memory differentiable policy and real optimizer/checkpoint state."""

    poll_interval = 0.01
    poll_backoff_multiplier = 1.0

    def __init__(self):
        self.saved = {}
        self.job_count = 0
        self.active = False
        self.interrupt_after_step = None
        self._initialize()

    def _initialize(self):
        self.parameter = torch.nn.Parameter(torch.zeros(16))
        self.optimizer = torch.optim.Adam([self.parameter], betas=(0.9, 0.95))
        self.steps = 0

    def list_jobs(self, **kwargs):
        return [{"job_id": self.job_id, "status": "RUNNING"}] if self.active else []

    def create_job_from_body(self, body):
        self._initialize()
        self.created_body = copy.deepcopy(body)
        self.job_count += 1
        self.job_id = f"test-job-{self.job_count}"
        self.active = True
        source = body["sub_job_configs"][0].get("source_checkpoint_info")
        if source:
            self.load(self.job_id, **source)
            if not body["sub_job_configs"][0]["training_config"].get(
                "load_optimizer_states", True
            ):
                self.optimizer = torch.optim.Adam([self.parameter], betas=(0.9, 0.95))
        return {"job_id": self.job_id}

    def wait_for_job(self, job_id):
        pass

    def poll_request(self, job_id, submitted):
        return submitted

    def cancel_job(self, job_id):
        self.active = False

    def get_job(self, job_id):
        return {"status": "running" if self.active else "cancelled"}

    def _normalize_job_status(self, status):
        return status.lower()

    def _logprobs(self, kwargs):
        return self.parameter.log_softmax(dim=0)[kwargs["input_ids"].roll(-1, dims=1)]

    def forward(self, job_id, data, **kwargs):
        frame = wire.loads(data)
        return {"logprobs": self._logprobs(frame["kwargs"]).detach()}

    def forward_backward(self, job_id, data):
        frame = wire.loads(data)
        logprobs = self._logprobs(frame["kwargs"])
        assert frame["processing"]["loss_fn"] == "dpo"
        loss, _ = recipe_utils.compute_dpo_loss(
            logprobs,
            frame["context"]["dpo_reference_logps"],
            frame["context"]["loss_mask"],
            beta=frame["processing"]["config"]["beta"],
        )
        loss.backward()
        return {"avg_loss": float(loss.detach()), "logprobs": logprobs.detach()}

    def step(self, job_id, learning_rate):
        norm = float(self.parameter.grad.norm())
        self.optimizer.param_groups[0]["lr"] = learning_rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.steps += 1
        if self.interrupt_after_step == self.steps:
            self.interrupt_after_step = None
            raise RuntimeError("simulated lost optimizer response")
        return {"global_steps": self.steps, "grad_norm": norm, "update_successful": 1.0}

    def save(self, job_id, checkpoint_type="weights-only"):
        checkpoint_id = f"cp_test_{len(self.saved)}"
        self.saved[checkpoint_id] = {
            "weights": self.parameter.detach().clone(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "steps": self.steps,
        }
        return {"stage_path": f"/checkpoints/{checkpoint_id}/global_step{self.steps}/"}

    def load(self, job_id, checkpoint_id, source_job_id):
        saved = self.saved[checkpoint_id]
        with torch.no_grad():
            self.parameter.copy_(saved["weights"])
        self.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        self.steps = saved["steps"]
        return {"global_steps": self.steps}


class _MetricLogger:
    def __init__(self):
        self.records = []

    def log_metrics(self, metrics, step):
        self.records.append((step, metrics))

    def close(self):
        pass


class _DiverseRenderer:
    def build_supervised_example(self, messages, train_on_what):
        index = int(messages[0]["content"].rsplit("-", 1)[-1])
        suffix = (7 if messages[-1]["content"] == "chosen" else 10) + index % 3
        return _Tokens([1, 2, suffix]), torch.tensor([0.0, 0.0, 1.0])


def test_full_runner_resume_preserves_reference_optimizer_and_schedule(
    monkeypatch, tmp_path
):
    from recipes.alignment.dpo.train import Config

    def load(*args, **kwargs):
        prefix, count = ("test", 3) if kwargs["split"] == "test_prefs" else ("train", 9)
        return [_preference_row(f"{prefix}-{index}") for index in range(count)]

    monkeypatch.setattr("datasets.load_dataset", load)
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()
    monkeypatch.setattr(
        dpo_runner,
        "build_renderer",
        lambda *args, **kwargs: (tokenizer, _DiverseRenderer(), "test_renderer"),
    )
    logger = _MetricLogger()
    monkeypatch.setattr(dpo_runner, "setup_logging", lambda *args, **kwargs: logger)
    baseline = _TrainingClient()
    monkeypatch.setattr(dpo_runner, "make_client", lambda *args, **kwargs: baseline)
    options = dict(
        config="unused",
        dataset="ultrafeedback",
        n_train_pairs=6,
        max_eval_pairs=3,
        max_steps=None,
        epochs=2,
        pairs_per_batch=3,
        eval_every=1,
        lr_schedule="linear",
        warmup_ratio=0.25,
    )
    result = dpo_runner.train(
        Config(**options, log_path=str(tmp_path / "baseline")),
        dpo_task.PreferenceTask(),
    )
    expected = baseline.saved[result["checkpoint_id"]]["weights"]
    assert result["steps"] == 4
    assert result["final_loss"] < math.log(2)
    assert not baseline.active
    expected_lrs = [
        metrics["train/learning_rate"]
        for _, metrics in logger.records
        if "train/learning_rate" in metrics
    ]
    logger.records.clear()
    interrupted = _TrainingClient()
    interrupted.interrupt_after_step = 2
    monkeypatch.setattr(dpo_runner, "make_client", lambda *args, **kwargs: interrupted)
    log_path = tmp_path / "recovery"
    with pytest.raises(RuntimeError, match="lost optimizer response"):
        dpo_runner.train(
            Config(**options, log_path=str(log_path)), dpo_task.PreferenceTask()
        )
    assert not interrupted.active
    state_path = log_path / "dpo-state.json"
    before = json.loads(state_path.read_text())
    assert before["next_step"] == 1
    resumed = dpo_runner.train(
        Config(**options, log_path=str(log_path), resume_state=str(state_path)),
        dpo_task.PreferenceTask(),
    )
    after = json.loads(state_path.read_text())
    assert after["references"] == before["references"]
    assert after["initial_snapshot"] == before["initial_snapshot"]
    torch.testing.assert_close(
        interrupted.saved[resumed["checkpoint_id"]]["weights"], expected, rtol=0, atol=0
    )
    assert resumed["test_metrics"] == result["test_metrics"]
    assert [
        metrics["train/learning_rate"]
        for _, metrics in logger.records
        if "train/learning_rate" in metrics
    ] == expected_lrs
    with pytest.raises(ValueError, match="does not match"):
        dpo_runner.train(
            Config(
                **{**options, "data_seed": 99},
                log_path=str(log_path),
                resume_state=str(state_path),
            ),
            dpo_task.PreferenceTask(),
        )
    job_count = interrupted.job_count
    for field in (
        "initial_snapshot",
        "resumable",
        "best_checkpoint",
        "best_resumable",
        "initial_loss",
        "best_loss",
    ):
        incomplete = copy.deepcopy(after)
        incomplete.pop(field)
        state_path.write_text(json.dumps(incomplete))
        with pytest.raises(ValueError, match="resume state requires"):
            dpo_runner.train(
                Config(**options, log_path=str(log_path), resume_state=str(state_path)),
                dpo_task.PreferenceTask(),
            )
        assert interrupted.job_count == job_count
    for value in (True, 0.5):
        incomplete = {**after, "next_step": value}
        state_path.write_text(json.dumps(incomplete))
        with pytest.raises(ValueError, match="next_step must be an integer"):
            dpo_runner.train(
                Config(**options, log_path=str(log_path), resume_state=str(state_path)),
                dpo_task.PreferenceTask(),
            )
        assert interrupted.job_count == job_count
    old_schedule = copy.deepcopy(after)
    old_schedule["signature"].pop("update_plan_sha256")
    state_path.write_text(json.dumps(old_schedule))
    with pytest.raises(ValueError, match="does not match"):
        dpo_runner.train(
            Config(**options, log_path=str(log_path), resume_state=str(state_path)),
            dpo_task.PreferenceTask(),
        )
    assert interrupted.job_count == job_count
    after["references"]["train"] = []
    state_path.write_text(json.dumps(after))
    job_count = interrupted.job_count
    with pytest.raises(ValueError, match="reference scores"):
        dpo_runner.train(
            Config(**options, log_path=str(log_path), resume_state=str(state_path)),
            dpo_task.PreferenceTask(),
        )
    assert interrupted.job_count == job_count
    monkeypatch.setattr(dpo_runner, "make_client", lambda *args, **kwargs: baseline)
    warm_path = tmp_path / "warm-start"
    warm = dpo_runner.train(
        Config(
            **{**options, "max_steps": 1},
            log_path=str(warm_path),
            source_job_id=result["checkpoint_source_job_id"],
            checkpoint_id=result["checkpoint_id"],
        ),
        dpo_task.PreferenceTask(),
    )
    assert warm["initial_loss"] == pytest.approx(math.log(2), abs=1e-7)
    assert (
        baseline.created_body["sub_job_configs"][0]["training_config"][
            "load_optimizer_states"
        ]
        is False
    )
    warm_state = json.loads((warm_path / "dpo-state.json").read_text())
    assert warm_state["references"] != before["references"]


def test_evaluation_weights_partial_batches_and_chosen_tokens(monkeypatch):
    pairs = [
        dpo_task.PreferenceTask().build_pair(
            _preference_row(str(i)), _Renderer(), max_seq_len=8
        )
        for i in range(3)
    ]
    values = torch.tensor(
        [
            [-1.0, -2.0, 0.0],
            [-1.0, -4.0, 0.0],
            [-1.0, -6.0, 0.0],
            [-1.0, -8.0, 0.0],
            [-1.0, -10.0, 0.0],
            [-1.0, -12.0, 0.0],
        ]
    )
    offset = 0

    def forward(client, job_id, kwargs, context, **options):
        nonlocal offset
        rows = len(kwargs["input_ids"])
        result = values[offset : offset + rows]
        offset += rows
        return result

    monkeypatch.setattr(dpo_runner, "_forward_logprobs", forward)
    reference = torch.zeros(6)
    result = dpo_runner._evaluate(
        None,
        "job",
        pairs,
        reference,
        beta=0.1,
        pad_token_id=0,
        max_seq_len=8,
        pairs_per_batch=2,
        n_gpus=1,
    )
    assert result["eval/chosen_logprob"] == -6
    assert result["eval/rejected_logprob"] == -8
    assert result["eval/chosen_nll_per_token"] == 6
    assert result["eval/unscaled_margin"] == pytest.approx(2)


def test_legacy_smoke_keeps_direction_control_and_cleanup(monkeypatch, tmp_path):
    from recipes.alignment.dpo.train import Config

    def load(*args, **kwargs):
        prefix, count = ("test", 8) if kwargs["split"] == "test" else ("train", 3)
        return [_preference_row(f"{prefix}-{i}") for i in range(count)]

    monkeypatch.setattr("datasets.load_dataset", load)
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()
    monkeypatch.setattr(
        dpo_runner,
        "build_renderer",
        lambda *a, **k: (tokenizer, _Renderer(), "test_renderer"),
    )
    client = _TrainingClient()
    monkeypatch.setattr(dpo_runner, "make_client", lambda *a, **k: client)
    monkeypatch.setattr(dpo_runner, "setup_logging", lambda *a, **k: _MetricLogger())
    result = dpo_runner.train(
        Config(config="unused", log_path=str(tmp_path)),
        dpo_task.PreferenceTask(),
    )
    assert result["final_loss"] < result["initial_loss"]
    assert result["steps"] == 3
    assert not client.active


def test_owned_job_cleanup_allows_unrelated_active_jobs(monkeypatch):
    client = _TrainingClient()
    monkeypatch.setattr(
        client, "list_jobs", lambda: [{"job_id": "other-job", "status": "RUNNING"}]
    )
    with dpo_runner.running_dpo_job(client, {"sub_job_configs": [{}]}):
        assert client.active
    assert client.job_count == 1
    assert not client.active


def test_quality_selection_can_keep_initialization_and_scores_test_once(
    monkeypatch, tmp_path
):
    from recipes.alignment.dpo.train import Config

    def load(*args, **kwargs):
        prefix, count = ("test", 3) if kwargs["split"] == "test_prefs" else ("train", 9)
        return [_preference_row(f"{prefix}-{index}") for index in range(count)]

    monkeypatch.setattr("datasets.load_dataset", load)
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()
    monkeypatch.setattr(
        dpo_runner,
        "build_renderer",
        lambda *args, **kwargs: (tokenizer, _DiverseRenderer(), "test_renderer"),
    )
    client = _TrainingClient()
    monkeypatch.setattr(dpo_runner, "make_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        dpo_runner, "setup_logging", lambda *args, **kwargs: _MetricLogger()
    )
    evaluate = dpo_runner._evaluate
    observed_steps = []

    def worsening_development(*args, **kwargs):
        observed_steps.append(client.steps)
        metrics = evaluate(*args, **kwargs)
        # Simulate a development set that worsens after the update. Final test
        # is evaluated only after loading the selected initialization boundary.
        if client.steps:
            metrics["eval/loss"] += 1
        return metrics

    monkeypatch.setattr(dpo_runner, "_evaluate", worsening_development)
    result = dpo_runner.train(
        Config(
            config="unused",
            dataset="ultrafeedback",
            n_train_pairs=6,
            max_eval_pairs=3,
            max_steps=1,
            pairs_per_batch=3,
            eval_every=1,
            log_path=str(tmp_path),
        ),
        dpo_task.PreferenceTask(),
    )
    state = json.loads((tmp_path / "dpo-state.json").read_text())
    assert state["next_step"] == 1
    assert state["best_checkpoint"] == state["initial_snapshot"]
    assert result["checkpoint_id"] == state["initial_snapshot"]["checkpoint_id"]
    assert result["final_loss"] == pytest.approx(math.log(2), abs=1e-7)
    assert observed_steps == [0, 1, 0]
    assert not client.active


@pytest.mark.parametrize(
    ("count", "batch_size", "n_gpus", "expected_sizes"),
    [
        (33, 32, 4, [29, 4]),
        (65, 32, 4, [32, 29, 4]),
        (16, 5, 4, [4, 4, 4, 4]),
        (34, 32, 4, [30, 4]),
        (5, 3, 2, [3, 2]),
    ],
)
def test_short_tail_keeps_batch_ceiling_and_order(
    count, batch_size, n_gpus, expected_sizes
):
    indices = list(range(count))
    batches = dpo_runner._batch_indices(indices, batch_size, n_gpus)
    assert [len(batch) for batch in batches] == expected_sizes
    assert all(len(batch) >= n_gpus and len(batch) <= batch_size for batch in batches)
    assert [index for batch in batches for index in batch] == indices


@pytest.mark.parametrize("norm_response", ["missing", "null", "zero", "actual"])
def test_optional_optimizer_norm_preserves_update_and_actual_metric(norm_response):
    from recipes.alignment.dpo.train import Config

    def update(response):
        client = _TrainingClient()
        task = dpo_task.PreferenceTask()
        pairs = [
            task.build_pair(
                _preference_row(f"train-{i}"), _DiverseRenderer(), max_seq_len=8
            )
            for i in range(2)
        ]
        kwargs, context = recipe_utils.collate_preference_pairs(
            [pair.as_tuple() for pair in pairs], pad_token_id=0, max_seq_len=8
        )
        references = recipe_utils.masked_row_logprob_sums(
            client._logprobs(kwargs).detach(), context["loss_mask"]
        )
        with torch.no_grad():
            client.parameter[7] = 0.25
        before = client.parameter.detach().clone()
        step = client.step
        actual_norm = []

        def optional_norm_step(job_id, learning_rate):
            result = step(job_id, learning_rate)
            actual_norm.append(result["grad_norm"])
            if response == "missing":
                del result["grad_norm"]
            elif response == "null":
                result["grad_norm"] = None
            elif response == "zero":
                result["grad_norm"] = 0
            return result

        client.step = optional_norm_step
        metrics = dpo_runner._update(
            client,
            "test-job",
            kwargs,
            context,
            references,
            config=Config(config="unused"),
            pad_token_id=0,
            learning_rate=0.01,
        )
        assert client.steps == 1
        assert len(actual_norm) == 1 and actual_norm[0] > 0
        assert not torch.equal(client.parameter, before)
        return client, metrics, actual_norm[0]

    expected, expected_metrics, expected_norm = update("actual")
    actual, metrics, actual_norm = update(norm_response)
    assert actual_norm == expected_norm
    if norm_response in {"missing", "null"}:
        assert "train/grad_norm" not in metrics
    else:
        assert metrics["train/grad_norm"] == (
            0.0 if norm_response == "zero" else actual_norm
        )
    assert {
        key: value for key, value in metrics.items() if key != "train/grad_norm"
    } == {
        key: value
        for key, value in expected_metrics.items()
        if key != "train/grad_norm"
    }
    assert torch.equal(actual.parameter, expected.parameter)
    actual_state = actual.optimizer.state_dict()
    expected_state = expected.optimizer.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]
    assert actual_state["state"].keys() == expected_state["state"].keys()
    for parameter, state in actual_state["state"].items():
        assert state.keys() == expected_state["state"][parameter].keys()
        for key, value in state.items():
            other = expected_state["state"][parameter][key]
            if torch.is_tensor(value):
                assert torch.equal(value, other)
            else:
                assert value == other


@pytest.mark.parametrize(
    "step_result",
    [
        {},
        {"global_steps": 1},
        {"update_successful": None},
        {"update_successful": True},
        {"update_successful": "1"},
        {"update_successful": [1]},
        {"update_successful": {"success": 1}},
        {"update_successful": 0},
        {"update_successful": float("nan")},
    ],
)
def test_uncertain_step_does_not_save_or_advance_boundary(
    monkeypatch, tmp_path, step_result
):
    from recipes.alignment.dpo.train import Config

    def load(*args, **kwargs):
        prefix, count = ("test", 3) if kwargs["split"] == "test_prefs" else ("train", 9)
        return [_preference_row(f"{prefix}-{index}") for index in range(count)]

    monkeypatch.setattr("datasets.load_dataset", load)
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()
    monkeypatch.setattr(
        dpo_runner,
        "build_renderer",
        lambda *args, **kwargs: (tokenizer, _DiverseRenderer(), "test_renderer"),
    )
    client = _TrainingClient()
    step = client.step

    def uncertain_step(job_id, learning_rate):
        step(job_id, learning_rate)
        return step_result

    monkeypatch.setattr(client, "step", uncertain_step)
    monkeypatch.setattr(dpo_runner, "make_client", lambda *args, **kwargs: client)
    logger = _MetricLogger()
    logger.closed = False
    monkeypatch.setattr(logger, "close", lambda: setattr(logger, "closed", True))
    monkeypatch.setattr(dpo_runner, "setup_logging", lambda *args, **kwargs: logger)
    with pytest.raises(RuntimeError, match="optimizer"):
        dpo_runner.train(
            Config(
                config="unused",
                dataset="ultrafeedback",
                n_train_pairs=6,
                max_eval_pairs=3,
                max_steps=1,
                pairs_per_batch=3,
                eval_every=1,
                log_path=str(tmp_path),
            ),
            dpo_task.PreferenceTask(),
        )
    state = json.loads((tmp_path / "dpo-state.json").read_text())
    assert state["next_step"] == 0
    assert len(client.saved) == 2  # Only initialization weights and optimizer.
    assert all(saved["steps"] == 0 for saved in client.saved.values())
    assert not any("progress/step" in metrics for _, metrics in logger.records)
    assert client.steps == 1  # Simulate an update whose acknowledgement was lost.
    assert not client.active
    assert logger.closed
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("result", [{"avg_loss": float("nan")}, {}, {"avg_loss": True}])
def test_failed_native_backward_does_not_step(result):
    from recipes.alignment.dpo.train import Config

    client = _TrainingClient()
    client.forward_backward = lambda *args: result
    pair = dpo_task.PreferenceTask().build_pair(
        _preference_row("train"), _Renderer(), max_seq_len=8
    )
    kwargs, context = recipe_utils.collate_preference_pairs(
        [pair.as_tuple()], pad_token_id=0, max_seq_len=8
    )
    with pytest.raises(RuntimeError):
        dpo_runner._update(
            client,
            "job",
            kwargs,
            context,
            torch.zeros(2),
            config=Config(config="unused"),
            pad_token_id=0,
            learning_rate=0.01,
        )
    assert client.steps == 0


def test_8k_preset_budget_fits_complete_long_pair():
    class LongRenderer:
        def build_supervised_example(self, messages, train_on_what):
            return _Tokens([1] * 5000), torch.tensor([0.0] + [1.0] * 4999)

    pair = dpo_task.PreferenceTask().build_pair(
        _preference_row("long"), LongRenderer(), max_seq_len=8192
    )
    body = json.loads((DPO_CONFIG_DIR / "qwen3_8b_lora_4gpu_8k.json").read_text())
    training = body["sub_job_configs"][0]["training_config"]
    assert len(pair.chosen.input_ids) <= training["max_seq_len"]
    assert len(pair.rejected.input_ids) <= training["max_seq_len"]
    assert (
        len(pair.chosen.input_ids) + len(pair.rejected.input_ids)
        <= training["mb_spec"]["max_tokens_per_mb"]
    )
    assert training["mb_spec"]["max_tokens_per_mb"] >= 2 * training["max_seq_len"]


def test_custom_pair_budget_is_checked_before_job_creation(monkeypatch, tmp_path):
    from recipes.alignment.dpo.train import Config

    class LongRenderer:
        def build_supervised_example(self, messages, train_on_what):
            return _Tokens([1] * 5000), torch.tensor([0.0] + [1.0] * 4999)

    body = json.loads((DPO_CONFIG_DIR / "qwen3_8b_lora.json").read_text())
    training = body["sub_job_configs"][0]["training_config"]
    training["max_seq_len"] = 8192
    training["mb_spec"]["max_tokens_per_mb"] = 8192
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps(body))
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    monkeypatch.setattr(
        dpo_runner,
        "build_renderer",
        lambda *a, **k: (tokenizer, LongRenderer(), "test"),
    )
    client = _TrainingClient()
    monkeypatch.setattr(dpo_runner, "make_client", lambda *a, **k: client)
    with pytest.raises(ValueError, match="complete preference pair exceeds"):
        dpo_runner.train(
            Config(
                config="unused", job_config=str(config_path), log_path=str(tmp_path)
            ),
            dpo_task.PreferenceTask(),
        )
    assert client.job_count == 0
