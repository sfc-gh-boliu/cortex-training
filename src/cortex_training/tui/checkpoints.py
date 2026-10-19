# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TUI's Checkpoints tab: a job's saved checkpoints (``list_checkpoints``) for display.

A checkpoint record, as ``GET /{job_id}/checkpoints`` lists it::

    {"checkpoint_id": "global_step12", "global_steps": 12, "avg_loss": 0.83,
     "created_at": "2026-07-20T18:05:00Z", "checkpoint_type": "resumable"}

A ``weights-only`` checkpoint can start a sampling job; a ``resumable`` one also
holds the optimizer state, to resume training from. Fields other than these are
shown as the record has them. The records reach this module through
``printable_client``, so their text is safe to draw.

Pure stdlib, NO ``textual`` import, like ``views.py``.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

from cortex_training.tui import views
from cortex_training.tui.format import local_time
from cortex_training.tui.format import parse_ts

WEIGHTS_ONLY = "weights-only"
RESUMABLE = "resumable"
# Shown in their own columns, or in the detail's first lines.
_KNOWN = ("checkpoint_id", "checkpoint_type", "global_steps", "created_at")
_STEP_IN_ID = re.compile(r"global_step(\d+)")


def checkpoint_type(record: dict) -> str:
    """``weights-only``, ``resumable``, else the type as the record spells it (``—`` without one)."""
    raw = str(record.get("checkpoint_type") or "").strip().lower().replace("_", "-")
    return {"weightsonly": WEIGHTS_ONLY}.get(raw, raw) or "—"


def checkpoint_step(record: dict) -> int | None:
    """``global_steps``, else the step a ``global_stepN`` id names."""
    steps = record.get("global_steps")
    if isinstance(steps, (int, float)) and not isinstance(steps, bool) and math.isfinite(steps) and steps >= 0:
        return int(steps)
    match = _STEP_IN_ID.search(str(record.get("checkpoint_id") or ""))
    return int(match.group(1)) if match else None


def saved_at(record: dict) -> datetime | None:
    return parse_ts(record.get("created_at"))


def sort_checkpoints(records: Any) -> list[dict]:
    """The records, newest first: by ``created_at``, then by step; a record without either last."""
    rows = [record for record in records or [] if isinstance(record, dict)]

    def key(record: dict) -> tuple:
        moment = saved_at(record)
        step = checkpoint_step(record)
        return (moment is not None, moment.timestamp() if moment else 0.0, step is not None, step or 0)

    return sorted(rows, key=key, reverse=True)


def metric_columns(records: list[dict]) -> list[str]:
    """The numeric fields beside the known ones (``avg_loss``), loss first, at most three."""
    names: dict[str, None] = {}
    for record in records:
        for name, value in record.items():
            if name not in _KNOWN and isinstance(value, (int, float)) and not isinstance(value, bool):
                names.setdefault(str(name))
    return sorted(names, key=lambda name: ("loss" not in name, name))[:3]


def format_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—" if value is None else str(value)
    if value == int(value) and abs(value) < 1e6:
        return f"{int(value):,}"
    size = abs(value)
    return f"{value:.2e}" if size >= 1e5 or size < 1e-3 else f"{value:.4g}"


def table_row(record: dict, metrics: list[str], now: datetime) -> tuple[str, ...]:
    """The table's cells: id, type, step, each metric column, saved (local time) and its age."""
    moment = saved_at(record)
    step = checkpoint_step(record)
    return (
        views.collapse(str(record.get("checkpoint_id") or "—")),
        checkpoint_type(record),
        "—" if step is None else f"{step:,}",
        *(format_number(record.get(name)) for name in metrics),
        local_time(moment) if moment else "—",
        views.fmt_ago(moment, now) if moment else "—",
    )


def summary_text(records: list[dict], now: datetime) -> str:
    """The tab's header: how many, and the newest."""
    if not records:
        return "no checkpoints saved"
    newest = records[0]
    count = f"{len(records)} checkpoint{'s' if len(records) != 1 else ''}"
    moment = saved_at(newest)
    when = f", saved {views.fmt_ago(moment, now)}" if moment else ""
    return f"{count} · newest {views.collapse(str(newest.get('checkpoint_id') or '—'))} ({checkpoint_type(newest)}){when}"


def no_checkpoints_text(job: dict) -> str:
    """Why a job lists no checkpoints."""
    if not any(views.sub_job_kind(sj) == "training" for sj in views.sub_jobs(job)):
        return "this job has no training sub-job, so it saves no checkpoints"
    if views.is_live(job):
        return ("no checkpoints saved yet: a training job saves one when its client calls save "
                "(the recipes save at their checkpoint interval and at the end)")
    return "this job saved no checkpoints"


def detail_lines(record: dict, job_id: str) -> list[tuple[str, list[str]]]:
    """(title, lines) sections for the selected checkpoint: its fields, then how to use it."""
    checkpoint_id = str(record.get("checkpoint_id") or "CHECKPOINT_ID")
    kind = checkpoint_type(record)
    step = checkpoint_step(record)
    moment = saved_at(record)
    fields = [
        f"id       {checkpoint_id}",
        f"type     {kind}",
        f"step     {'—' if step is None else f'{step:,}'}",
        f"saved    {local_time(moment, '%Y-%m-%d %H:%M:%S') if moment else '—'}",
    ]
    fields += [f"{name}  {format_number(value) if isinstance(value, (int, float)) else views.collapse(str(value))}"
               for name, value in record.items() if name not in _KNOWN]
    source = json.dumps({"source_checkpoint_info": {"checkpoint_id": checkpoint_id, "source_job_id": job_id}})
    sections = [("Checkpoint", fields)]
    if kind == WEIGHTS_ONLY:
        sections.append(("Sample from it", [
            "Create a sampling job whose sampling sub-job has",
            f"  {source}",
        ]))
    else:
        sections.append(("Resume training from it", [
            "Create a training job whose training sub-job has",
            f"  {source}",
            "With a different n_gpus, also set training_config.load_optimizer_states to false.",
            "Or load it into a running training job:",
            f"  cortex-training --job-id RUNNING_JOB_ID load {checkpoint_id} --source-job-id {job_id}",
        ]))
        if kind == RESUMABLE:
            sections[-1][1].append("Sampling needs a weights-only checkpoint.")
    sections.append(("Download its files", [
        "Short-lived URLs for each file (Python):",
        f"  client.export_checkpoint({job_id!r}, {checkpoint_id!r})[\"files\"]",
        f"List them all: cortex-training checkpoints {job_id}",
    ]))
    return sections
