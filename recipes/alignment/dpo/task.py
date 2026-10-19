# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chosen/rejected preference dataset component for DPO."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook import renderers

from recipes.utils import TrainSequence, sequence_from_conversation

logger = logging.getLogger(__name__)

_RECIPE_DIR = Path(__file__).resolve().parent
_SMOKE_TRAIN = _RECIPE_DIR / "data" / "smoke.jsonl"
_SMOKE_TEST = _RECIPE_DIR / "data" / "smoke_eval.jsonl"
_ULTRAFEEDBACK_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
_ARGILLA_DATASET = "argilla/ultrafeedback-binarized-preferences"
_HELPSTEER_DATASET = "nvidia/HelpSteer3"
DATASETS = {
    "ultrafeedback": (
        _ULTRAFEEDBACK_DATASET,
        "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
    ),
    "argilla": (_ARGILLA_DATASET, "737bab1bd42de611f3887e9d3e9866edf5df2780"),
    "helpsteer3": (_HELPSTEER_DATASET, "f6d145777bcbde96137596340fab89793acd1031"),
}
DATASETS.update({name: (name, revision) for name, revision in list(DATASETS.values())})


@dataclass(frozen=True)
class PreferencePair:
    chosen: TrainSequence
    rejected: TrainSequence
    prompt_fingerprint: str
    row_id: str = ""

    def as_tuple(self) -> tuple[TrainSequence, TrainSequence]:
        return self.chosen, self.rejected


def _validate_messages(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty message array")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"{field}[{index}] must be a message object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{field}[{index}].role must be a non-empty string")
        if "content" not in message:
            raise ValueError(f"{field}[{index}] needs content")
        messages.append(dict(message))
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{field} must end with an assistant response")
    return messages


def _prompt_fingerprint(messages: list[dict[str, Any]]) -> str:
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PreferenceTask:
    """Load and render adjacent preference pairs without losing pair identity."""

    name = "chosen/rejected preferences"

    def __init__(self) -> None:
        self.load_record: dict[str, Any] = {}

    @staticmethod
    def normalize_row(row: Mapping[str, Any], dataset: str) -> dict[str, Any] | None:
        """Map each source's strict preference label into paired conversations."""
        name = DATASETS.get(dataset, (dataset, None))[0]
        if name == _ULTRAFEEDBACK_DATASET:
            scores = row["score_chosen"], row["score_rejected"]
        elif name == _ARGILLA_DATASET:
            scores = row["chosen_avg_rating"], row["rejected_avg_rating"]
        elif name == _HELPSTEER_DATASET:
            preference = float(row["overall_preference"])
            if not math.isfinite(preference):
                raise ValueError("preference must be finite")
            if preference == 0:
                return None
            chosen, rejected = (
                (row["response1"], row["response2"])
                if preference < 0
                else (row["response2"], row["response1"])
            )
            prompt = row["context"]
            return {
                "chosen": prompt + [{"role": "assistant", "content": chosen}],
                "rejected": prompt + [{"role": "assistant", "content": rejected}],
            }
        else:
            return dict(row)
        chosen_score, rejected_score = map(float, scores)
        if not math.isfinite(chosen_score) or not math.isfinite(rejected_score):
            raise ValueError("preference scores must be finite")
        if chosen_score <= rejected_score:
            return None
        if name == _ARGILLA_DATASET:
            prompt = [{"role": "user", "content": row["instruction"]}]
            return {
                "chosen": prompt
                + [{"role": "assistant", "content": row["chosen_response"]}],
                "rejected": prompt
                + [{"role": "assistant", "content": row["rejected_response"]}],
            }
        return dict(row)

    def build_pair(
        self,
        row: Mapping[str, Any],
        renderer: Any,
        *,
        max_seq_len: int,
    ) -> PreferencePair:
        chosen_messages = _validate_messages(row.get("chosen"), "chosen")
        rejected_messages = _validate_messages(row.get("rejected"), "rejected")
        prompt = chosen_messages[:-1]
        if prompt != rejected_messages[:-1]:
            raise ValueError("chosen and rejected must share the same prompt messages")
        if chosen_messages[-1] == rejected_messages[-1]:
            raise ValueError("chosen and rejected assistant responses must differ")

        train_on = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
        chosen = sequence_from_conversation(chosen_messages, renderer, train_on)
        rejected = sequence_from_conversation(rejected_messages, renderer, train_on)
        if len(chosen.input_ids) > max_seq_len or len(rejected.input_ids) > max_seq_len:
            raise OverflowError("preference pair exceeds max_seq_len")
        if not any(chosen.loss_mask) or not any(rejected.loss_mask):
            raise ValueError(
                "both final assistant responses must contain trainable tokens"
            )
        return PreferencePair(
            chosen=chosen,
            rejected=rejected,
            prompt_fingerprint=_prompt_fingerprint(prompt),
        )

    def load_pairs(
        self,
        dataset: str,
        split: str,
        *,
        n_pairs: int,
        renderer: Any,
        max_seq_len: int,
        seed: int = 0,
        revision: str | None = None,
        exclude_prompts: set[str] | None = None,
    ) -> list[PreferencePair]:
        if n_pairs <= 0:
            raise ValueError("n_pairs must be greater than zero")
        import datasets

        dataset_name, pinned_revision = DATASETS.get(dataset, (dataset, None))
        revision = revision or pinned_revision
        dataset_split = split
        if dataset_name == _ULTRAFEEDBACK_DATASET:
            dataset_split = {"train": "train_prefs", "test": "test_prefs"}.get(
                split, split
            )
        elif dataset_name == _ARGILLA_DATASET:
            dataset_split = "train"
        elif dataset_name == _HELPSTEER_DATASET:
            dataset_split = {"test": "validation"}.get(split, split)
        skip_invalid_pairs = dataset_name in DATASETS
        if dataset_name == "smoke":
            loaded = datasets.load_dataset(
                "json",
                data_files={
                    "train": str(_SMOKE_TRAIN),
                    "test": str(_SMOKE_TEST),
                },
                split=dataset_split,
            )
        else:
            loaded = datasets.load_dataset(
                dataset_name,
                **(
                    {"name": "preference"} if dataset_name == _HELPSTEER_DATASET else {}
                ),
                split=dataset_split,
                revision=revision,
            )
        indices = list(range(len(loaded)))
        if dataset != "smoke":
            random.Random(seed).shuffle(indices)
        pairs: list[PreferencePair] = []
        counts = {
            "length": 0,
            "invalid": 0,
            "ambiguous_label": 0,
            "duplicate_or_reserved_prompt": 0,
        }
        seen = set(exclude_prompts or ())
        for index in indices:
            row = loaded[index]
            try:
                normalized = self.normalize_row(row, dataset)
                if normalized is None:
                    counts["ambiguous_label"] += 1
                    continue
                pair = self.build_pair(normalized, renderer, max_seq_len=max_seq_len)
            except OverflowError:
                counts["length"] += 1
                continue
            except (ValueError, KeyError, TypeError):
                if not skip_invalid_pairs:
                    raise
                counts["invalid"] += 1
                continue
            if pair.prompt_fingerprint in seen:
                counts["duplicate_or_reserved_prompt"] += 1
                continue
            seen.add(pair.prompt_fingerprint)
            pair = PreferencePair(
                pair.chosen,
                pair.rejected,
                pair.prompt_fingerprint,
                f"{dataset_split}:{index}",
            )
            pairs.append(pair)
            if len(pairs) == n_pairs:
                break
        self.load_record = {
            "dataset": dataset_name,
            "revision": revision,
            "split": dataset_split,
            "seed": seed,
            "selected_row_ids": [pair.row_id for pair in pairs],
            "exclusions_before_selection": counts,
        }
        logger.info(
            "Selected %d pairs from %s/%s; exclusions before selection: %s",
            len(pairs),
            dataset_name,
            dataset_split,
            counts,
        )
        if len(pairs) < n_pairs:
            raise ValueError(
                f"dataset supplied {len(pairs)} usable pairs for requested n_pairs={n_pairs}"
            )
        return pairs


def require_disjoint_prompts(
    train_pairs: list[PreferencePair],
    eval_pairs: list[PreferencePair],
) -> None:
    overlap = {pair.prompt_fingerprint for pair in train_pairs} & {
        pair.prompt_fingerprint for pair in eval_pairs
    }
    if overlap:
        raise ValueError(
            "held-out evaluation prompts overlap the training preferences; "
            "use a genuinely separate split"
        )
