# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TUI's Checkpoints tab helpers (``tui/checkpoints.py``)."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock

from cortex_training.tui import checkpoints
from cortex_training.tui.untrusted import printable_client

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
JOB_ID = "JOB_ID"


def _cp(cid, *, steps=None, at=None, kind="resumable", **extra):
    record = {"checkpoint_id": cid, "checkpoint_type": kind, **extra}
    if steps is not None:
        record["global_steps"] = steps
    if at is not None:
        record["created_at"] = at
    return record


def test_type_and_step():
    assert checkpoints.checkpoint_type({"checkpoint_type": "WEIGHTS_ONLY"}) == "weights-only"
    assert checkpoints.checkpoint_type({"checkpoint_type": "resumable"}) == "resumable"
    assert checkpoints.checkpoint_type({}) == "—"
    assert checkpoints.checkpoint_step(_cp("x", steps=12)) == 12
    assert checkpoints.checkpoint_step(_cp("global_step40")) == 40  # from the id
    assert checkpoints.checkpoint_step(_cp("cp_tag", steps=True)) is None


def test_newest_first_and_records_without_a_time_last():
    records = [
        _cp("global_step10", steps=10, at="2026-10-19T10:00:00Z"),
        _cp("untimed", steps=99),
        _cp("global_step30", steps=30, at="2026-10-19T11:00:00Z"),
        _cp("global_step20", steps=20, at="2026-10-19T11:00:00Z"),
        "not a record",
    ]
    assert [r["checkpoint_id"] for r in checkpoints.sort_checkpoints(records)] == \
        ["global_step30", "global_step20", "global_step10", "untimed"]
    assert checkpoints.sort_checkpoints(None) == []


def test_table_row_and_summary():
    records = checkpoints.sort_checkpoints([
        _cp("global_step12", steps=12, at="2026-10-19T11:55:00Z", avg_loss=0.83),
        _cp("global_step6", steps=6, at="2026-10-19T11:00:00Z", avg_loss=1.2, kind="weights-only"),
    ])
    metrics = checkpoints.metric_columns(records)
    assert metrics == ["avg_loss"]  # global_steps has its own column
    row = checkpoints.table_row(records[0], metrics, NOW)
    assert row[:4] == ("global_step12", "resumable", "12", "0.83")
    assert row[-1] == "5m 00s ago"
    assert checkpoints.summary_text(records, NOW) == \
        "2 checkpoints · newest global_step12 (resumable), saved 5m 00s ago"
    assert checkpoints.summary_text([], NOW) == "no checkpoints saved"


def test_metric_columns_put_loss_first_and_keep_three():
    records = [{"checkpoint_id": "a", "z": 1, "eval_loss": 2.0, "a": 3, "b": 4, "flag": True}]
    assert checkpoints.metric_columns(records) == ["eval_loss", "a", "b"]


def test_no_checkpoints_text():
    assert "no training sub-job" in checkpoints.no_checkpoints_text(
        {"status": "RUNNING", "sub_jobs": [{"sub_job_id": "j:sampling:0", "job_type": "sampling"}]})
    training = [{"sub_job_id": "j:training:0", "job_type": "training"}]
    assert checkpoints.no_checkpoints_text({"status": "RUNNING", "sub_jobs": training}).startswith("no checkpoints saved yet")
    assert checkpoints.no_checkpoints_text({"status": "SUCCEEDED", "sub_jobs": training}) == "this job saved no checkpoints"


def _lines(sections):
    return {title: lines for title, lines in sections}


def test_detail_says_how_to_use_each_type():
    weights = _lines(checkpoints.detail_lines(_cp("global_step6", steps=6, kind="weights-only"), JOB_ID))
    assert list(weights) == ["Checkpoint", "Sample from it", "Download its files"]
    assert any('"source_checkpoint_info": {"checkpoint_id": "global_step6", "source_job_id": "JOB_ID"}' in line
               for line in weights["Sample from it"])

    resumable = _lines(checkpoints.detail_lines(_cp("global_step12", steps=12, avg_loss=0.83), JOB_ID))
    assert list(resumable) == ["Checkpoint", "Resume training from it", "Download its files"]
    assert "  cortex-training --job-id RUNNING_JOB_ID load global_step12 --source-job-id JOB_ID" in \
        resumable["Resume training from it"]
    assert resumable["Resume training from it"][-1] == "Sampling needs a weights-only checkpoint."
    assert "avg_loss  0.83" in resumable["Checkpoint"]
    assert "  client.export_checkpoint('JOB_ID', 'global_step12')[\"files\"]" in resumable["Download its files"]


def test_checkpoint_reads_come_back_printable():
    client = MagicMock()
    client.list_checkpoints.return_value = [{"checkpoint_id": "global_step1\x1b]52;c;eA==\x07", "checkpoint_type": "x"}]
    records = printable_client(client).list_checkpoints(JOB_ID)
    assert "\x1b" not in records[0]["checkpoint_id"] and records[0]["checkpoint_id"].startswith("global_step1")
