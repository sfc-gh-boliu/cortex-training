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

"""The preview's rows: the comment gets exactly the rows the rest leaves (no row
wasted, none short), and a refresh that makes the preview scroll takes no width
from its text, so nothing re-wraps at the edge."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.containers import VerticalScroll  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.untrusted import unwrapped  # noqa: E402

JOB = "fake-job-0002"
COMMENT = "\n".join(f"line {i}: nightly eval of the support-chat checkpoints against the frozen March prompt set "
                    "(1,200 prompts)" for i in range(1, 11))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(**extra):
    job = {"job_id": JOB, "status": "CANCELLED", "reason": "cancelled_by_client", "comment": "a short name",
           "hardware": "H200", "owner_role": "TRAINING_ROLE", "submitted_by": "TEST_USER_A",
           "created_at": "2026-10-14T17:39:00Z", "updated_at": "2026-10-14T18:39:00Z",
           "state_changed_at": "2026-10-14T18:39:00Z",
           "sub_jobs": [{"sub_job_id": f"{JOB}:training:0", "job_type": "TRAINING", "status": "CANCELLED",
                         "model_name": "Qwen/Qwen3-8B",
                         "training_config": {"n_gpus": 4, "optimizer": {"name": "fused_adam", "lr": 1e-05}}}]}
    job.update(extra)
    return job


def _client(job):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
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


async def _open(app, pilot, job):
    c = app._client  # the app's printable wrapper; unwrapped(c) is the mock
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    app.push_screen(JobSummaryScreen(c, JOB, job=job))
    assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and unwrapped(c).get_job.called)
    p = app.screen
    assert await _wait(pilot, lambda: p._fetched_at is not None and not p._loading)
    for _ in range(10):
        await pilot.pause()
    return p


def _lines(p) -> list:
    return str(p.query_one("#preview", Static).render()).split("\n")


def _spare_rows(p) -> int:
    """The body's rows below its last line (its virtual height never goes below its own)."""
    body = p.query_one("#summary-body", VerticalScroll)
    return body.scrollable_content_region.bottom - p.query_one("#glance", Static).region.bottom


def test_the_comment_gets_exactly_the_rows_left():
    """The shortest screen that shows the comment whole fills it to its last row; one
    row less cuts the comment; and neither scrolls."""
    job = _job(comment=COMMENT)

    async def run():
        app = CortexTrainingLogTUI(_client(job), poll_interval=0.01)
        async with app.run_test(size=(140, 50)) as pilot:
            p = await _open(app, pilot, job)
            body = p.query_one("#summary-body", VerticalScroll)
            seen = {}
            for rows in range(46, 32, -1):
                await pilot.resize_terminal(140, rows)
                assert await _wait(pilot, lambda: p.size.height == rows)
                for _ in range(10):
                    await pilot.pause()
                whole = "e shows the whole comment" not in "\n".join(_lines(p))
                seen[rows] = (whole, _spare_rows(p), body.max_scroll_y)
            assert all(scroll == 0 for _whole, _spare, scroll in seen.values()), seen
            least_whole = min(rows for rows, (whole, _s, _y) in seen.items() if whole)
            assert not seen[least_whole - 1][0], seen
            assert seen[least_whole][1] == 0, f"one row wasted at {least_whole} rows: {seen}"
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_late_refresh_that_makes_the_preview_scroll_rewraps_nothing():
    job = _job()
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(100, 60)) as pilot:
            p = await _open(app, pilot, job)
            body = p.query_one("#summary-body", VerticalScroll)
            # A screen the preview just fits, then a failed refresh whose error makes it scroll.
            await pilot.resize_terminal(100, app.size.height - _spare_rows(p))
            for _ in range(20):
                await pilot.pause()
            assert body.max_scroll_y == 0 and _spare_rows(p) == 0
            width = p.query_one("#preview", Static).size.width
            c.get_job.side_effect = RuntimeError(
                "the job could not be read: " + "the connection timed out; " * 8)
            await pilot.press("r")
            assert await _wait(pilot, lambda: p._fetch_error and body.max_scroll_y > 0)
            for _ in range(20):
                await pilot.pause()
            assert p.query_one("#preview", Static).size.width == width, "the scrollbar takes no width from the text"
            for name in ("#preview", "#fetch", "#health", "#subjobs-title", "#glance"):
                box = p.query_one(name, Static)
                assert box.size.height == len(str(box.render()).split("\n")), f"{name} soft-wraps a row"
            await _settle(app, pilot)

    asyncio.run(run())
