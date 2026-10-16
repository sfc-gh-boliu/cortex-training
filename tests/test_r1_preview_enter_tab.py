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

"""Enter on a sub-job row (the preview's or the job list's) opens Logs; the preview says so."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402

JOB = "fake-job-enter"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job():
    return {"job_id": JOB, "status": "RUNNING", "comment": "enter probe", "owner_role": "ML",
            "sub_jobs": [{"sub_job_id": f"{JOB}:{kind}:0", "job_type": kind.upper(), "status": "RUNNING",
                          "model_name": "Qwen/Qwen3-8B", f"{'training' if kind == 'training' else 'inference'}_config":
                          {"n_gpus": 4}} for kind in ("training", "sampling")]}


def _client(job):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "C1", "eof": False}
    c.download_metrics.return_value = []
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


def test_enter_on_a_preview_sub_job_row_opens_logs():
    job = _job()
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            app.push_screen(JobSummaryScreen(c, JOB, job=job))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            p = app.screen
            assert await _wait(pilot, lambda: p._fetched_at is not None)
            title = str(p.query_one("#subjobs-title", Static).render())
            assert title == "Sub-jobs  (Enter opens Logs for the highlighted row)"
            table = p.query_one("#subjobs", DataTable)
            table.move_cursor(row=table.get_row_index("sj-1"))
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
            d = app.screen
            assert d._active_tab == "logs" and d._scope == f"{JOB}:sampling:0"
            await _settle(app, pilot)

    asyncio.run(run())


def test_enter_on_a_list_sub_job_row_opens_logs():
    job = _job()
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(180, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._sub_by_item)
            s = app.screen
            item = next(key for key, (_jid, sid) in s._sub_by_item.items() if sid == f"{JOB}:training:0")
            table = s.query_one("#jobs", DataTable)
            s.set_focus(table)
            row = table.get_row_index(item)
            # The list places its cursor after its first layout: move it once that is done.
            for _ in range(20):
                table.move_cursor(row=row)
                await pilot.pause(0.01)
                if table.cursor_row == row:
                    break
            assert table.cursor_row == row
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
            assert app.screen._active_tab == "logs"
            assert app.screen._scope == f"{JOB}:training:0"
            await _settle(app, pilot)

    asyncio.run(run())
