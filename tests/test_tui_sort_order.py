# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The job list's sort order: o reverses it, and the list says which jobs come first."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(jid: str, status: str, minutes_ago: float) -> dict:
    return {"job_id": jid, "status": status, "created_at": _iso(minutes_ago), "updated_at": _iso(minutes_ago),
            "comment": f"job {jid}", "submitted_by": "TEST_USER_A", "hardware": "H200",
            "sub_jobs": [{"sub_job_id": f"{jid}:training:0", "job_type": "TRAINING", "status": status,
                          "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8.0}}]}


def _client(jobs: list) -> MagicMock:
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.side_effect = lambda **_: [dict(j) for j in jobs]
    c.get_job.side_effect = lambda jid, **_: {}
    return c


async def _wait(pilot, predicate, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.02)
    return bool(predicate())


def _order(screen) -> list:
    table = screen.query_one("#jobs", DataTable)
    return [screen._job_by_item[row.key.value] for row in table.ordered_rows if row.key.value in screen._job_by_item]


def _info(screen) -> str:
    return " ".join(str(screen.query_one("#listinfo", Static).render()).split())


@pytest.mark.parametrize("key, descending, words", [
    ("created", True, "newest first"),
    ("created", False, "oldest first"),
    ("updated", True, "newest first"),
    ("updated", False, "oldest first"),
    ("status", True, "running first"),
    ("status", False, "ended first"),
])
def test_the_direction_words_match_what_comes_first(key, descending, words):
    assert views.sort_direction(key, descending) == words
    jobs = [_job("old-done", "CANCELLED", 300), _job("new-running", "RUNNING", 10)]
    first = views.sort_job_list(jobs, key, descending)[0]["job_id"]
    newest_or_running_first = descending
    assert first == ("new-running" if newest_or_running_first else "old-done")


def test_o_is_in_the_footer_and_reverses_the_order_the_list_names():
    # Ended jobs only, so LIVE and HISTORY don't split the rows: the table order is the sort order.
    jobs = [_job(f"j{i}", "CANCELLED", minutes_ago=10 + i) for i in range(5)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(200, 30)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item
                               and not app.screen._refreshing)
            s = app.screen
            shown = {key: ab.binding.description for key, ab in s.active_bindings.items() if ab.binding.show}
            assert shown.get("o") == "Order"
            assert "o order" in str(app.sub_title)
            assert "sort: created ↓ newest first (o reverses)" in _info(s)
            assert _order(s) == ["j0", "j1", "j2", "j3", "j4"]

            await pilot.press("o")
            assert await _wait(pilot, lambda: _order(s) == ["j4", "j3", "j2", "j1", "j0"])
            assert "sort: created ↑ oldest first (o reverses)" in _info(s)

            await pilot.press("s", "s")  # created -> updated -> status, still ascending
            assert await _wait(pilot, lambda: s._sort == "status")
            assert "sort: status ↑ ended first (o reverses)" in _info(s)
            await pilot.press("o")
            assert await _wait(pilot, lambda: "sort: status ↓ running first (o reverses)" in _info(s))
            app.workers.cancel_all()
            await pilot.pause()

    asyncio.run(run())
