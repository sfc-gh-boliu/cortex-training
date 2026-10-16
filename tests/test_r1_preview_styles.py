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

"""Every style in the preview's lines is one Textual draws: an ended job's status
and its health line are grey there too, as they are in the tables, and so is the
reason on the details summary line."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.style import Style  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import ENDED_STYLE  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402

JOB = "fake-job-0003"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(status, reason):
    job = {"job_id": JOB, "status": status, "comment": "a short name", "hardware": "H200",
           "owner_role": "TRAINING_ROLE", "created_at": "2026-10-14T17:39:00Z", "updated_at": "2026-10-14T18:39:00Z",
           "sub_jobs": [{"sub_job_id": f"{JOB}:training:0", "job_type": "TRAINING", "status": status,
                         "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 4}}]}
    if reason:
        job["reason"] = reason
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


def _spans(screen, names) -> list:
    out = []
    for name in names:
        content = screen.query_one(name, Static).render()
        out += [(name, content.plain[span.start:span.end], str(span.style)) for span in content.spans]
    return out


@pytest.mark.parametrize("status, reason", [("CANCELLED", "cancelled_by_client"), ("TERMINATED", "idle_timeout"),
                                            ("SUCCEEDED", None), ("FAILED", "sub_job_failed"), ("PENDING", None)])
def test_every_style_in_the_preview_is_one_textual_draws(status, reason):
    job = _job(status, reason)
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(120, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            app.push_screen(JobSummaryScreen(c, JOB, job=job))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            p = app.screen
            assert await _wait(pilot, lambda: not p._loading)
            spans = _spans(p, ("#preview", "#fetch", "#health", "#subjobs-title", "#glance"))
            assert spans
            for _name, _text, style in spans:
                Style.parse(style)  # raises on a name only Rich knows, such as grey62
            if status in ("CANCELLED", "TERMINATED", "SUCCEEDED"):
                styles = {text: style for _name, text, style in spans}
                assert styles[status] == ENDED_STYLE, spans  # the Status field
                assert any(text.startswith("■ ") and style == ENDED_STYLE for _n, text, style in spans), spans
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_ended_job_s_reason_on_the_details_line_is_grey():
    job = _job("CANCELLED", "cancelled_by_client")
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            app.push_screen(JobDetailScreen(c, JOB, job=job, initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and c.get_job.called)
            d = app.screen
            assert await _wait(pilot, lambda: "cancelled_by_client" in str(d.query_one("#summary", Static).render()))
            spans = _spans(d, ("#summary",))
            for _name, _text, style in spans:
                Style.parse(style)
            assert ("#summary", "cancelled_by_client", ENDED_STYLE) in spans, spans
            await _settle(app, pilot)

    asyncio.run(run())
