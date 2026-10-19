# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin entry point for Direct Preference Optimization."""

from __future__ import annotations

import logging

import chz

from recipes.alignment.dpo.runner import train
from recipes.alignment.dpo.task import PreferenceTask


@chz.chz
class Config:
    config: str

    dataset: str = "smoke"
    dataset_split: str = "train"
    eval_dataset: str | None = None
    eval_split: str = "test"
    max_eval_pairs: int = 8
    dpo_beta: float = 0.1
    max_steps: int | None = 3
    n_train_pairs: int | None = None
    epochs: int = 1
    pairs_per_batch: int = 1
    data_seed: int = 0
    dataset_revision: str | None = None
    eval_every: int = 32
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.0
    tokenizer_revision: str | None = None
    source_job_id: str | None = None
    checkpoint_id: str | None = None
    resume_state: str | None = None
    min_eval_accuracy: float = 0.5
    initial_loss_tolerance: float = 1e-3

    renderer_name: str | None = None
    enable_thinking: bool = False

    log_path: str = "/tmp/cortex-training-examples/dpo-one-gpu"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False

    job_config: str = "configs/qwen3_8b_lora.json"
    debug_image_tag: str | None = None
    max_total_seconds: float = 3600.0


def main(config: Config) -> dict:
    logging.basicConfig(level=logging.INFO)
    return train(config, PreferenceTask())


if __name__ == "__main__":
    chz.nested_entrypoint(main)
