# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The details summary line fits its row (the name and models give way, the
reason stays whole), never ends a row with a separator, and names a sub-job
once."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402

JOB = "00000000-0000-4000-8000-000000000003"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _fit(segments, width):
    from cortex_training.tui.screens import fit_line

    return fit_line(segments, width).plain


def test_a_line_that_fits_is_left_alone():
    segments = [("my job", "bold", 12), ("JOB", "", None), ("training", "", None), ("reason [b]x[/b]", "red", None)]
    assert _fit(segments, 80) == "my job  ·  JOB  ·  training  ·  reason [b]x[/b]"


def test_the_name_then_the_models_give_way_and_the_reason_stays_whole():
    name, models = "long-context sft · Qwen3-8B 32k", "Qwen/Qwen3-8B, Qwen/Qwen3-0.6B"
    segments = [(name, "bold", 12), (JOB, "", None), ("all sub-jobs", "", None), ("training", "", None),
                (models, "", 12), ("sub_job_failed", "red", None)]
    full = "  ·  ".join(text for text, _style, _least in segments)
    for width in (cell_len(full) - 5, cell_len(full) - 25, cell_len(full) - 36):
        line = _fit(segments, width)
        assert "\n" not in line and cell_len(line) <= width, (width, line)
        assert line.endswith("  ·  sub_job_failed") and JOB in line and "all sub-jobs" in line
    assert _fit(segments, cell_len(full) - 5).startswith("long-context sft · Qwen3-…  ·  ")  # the name first
    assert f"  ·  {models}  ·  " in _fit(segments, cell_len(full) - 5)  # the models untouched


def test_a_line_too_long_even_cut_wraps_between_segments():
    segments = [("name", "bold", 12), (JOB, "", None), ("all sub-jobs", "", None), ("train+sample", "", None),
                ("Qwen/Qwen3-8B", "", 12), ("quota_filled — needs 8 GPUs; 64 currently active", "yellow", None)]
    rows = _fit(segments, 70).split("\n")
    assert len(rows) > 1
    assert all(not row.rstrip().endswith("·") and not row.startswith(" ·") for row in rows), rows
    assert all(cell_len(row) <= 70 for row in rows)
    assert rows[-1].endswith("quota_filled — needs 8 GPUs; 64 currently active")
    assert "Qwen/Qwen3-8B" in "\n".join(rows), "wrapping anyway: nothing is cut"


def _job(**extra):
    job = {"job_id": JOB, "status": "FAILED", "reason": "sub_job_failed", "comment": "long-context sft · Qwen3-8B 32k",
           "owner_role": "ML", "hardware": "H200",
           "sub_jobs": [{"sub_job_id": f"{JOB}:training:0", "job_type": "TRAINING", "status": "FAILED",
                         "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}}]}
    job.update(extra)
    return job


def _client(job):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "C1", "eof": False}
    c.download_stdout_logs.return_value = []
    c.fetch_execution_logs.return_value = []
    return c


async def _wait(pilot, predicate, tries=300):
    for _ in range(tries):
        await pilot.pause(0.01)
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _summary(screen) -> list:
    return str(screen.query_one("#summary", Static).render()).split("\n")


@pytest.mark.parametrize("scope", [None, f"{JOB}:training:0"])
def test_the_details_summary_fits_one_row_at_140_columns(scope):
    job = _job()
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            app.push_screen(JobDetailScreen(c, JOB, job=job, sub_job_id=scope, initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and c.get_job.called)
            d = app.screen
            assert await _wait(pilot, lambda: "sub_job_failed" in "".join(_summary(d)))
            for _ in range(5):
                await pilot.pause()
            rows = _summary(d)
            box = d.query_one("#summary", Static)
            assert len(rows) == 1 and box.size.height == 1, rows  # no soft wrap either
            assert rows[0].endswith("  ·  sub_job_failed")
            assert rows[0].count(JOB) == 1, "a sub-job's id already names its job"
            assert (f"sub-job {JOB}:training:0" in rows[0]) == bool(scope)
            assert ("all sub-jobs" in rows[0]) == (not scope)
            # Narrower: the name and the models give way, then the line wraps between segments.
            await pilot.resize_terminal(90, 40)
            assert await _wait(pilot, lambda: all(cell_len(row) <= box.size.width for row in _summary(d)))
            assert all(not row.rstrip().endswith("·") for row in _summary(d)), _summary(d)
            assert _summary(d)[-1].endswith("sub_job_failed")
            assert "long-context sft · Qwen3-8B 32k" in "\n".join(_summary(d)), "wrapping anyway: the name is whole"
            await _settle(app, pilot)

    asyncio.run(run())
