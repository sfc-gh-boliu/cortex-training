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

"""The job list's sort: what its words say comes first does, the line above the list never
moves the list when the sort changes, and rows without the sorted time stay last."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training.tui import app as app_module  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from tests.v2_joblist_support import client  # noqa: E402
from tests.v2_joblist_support import isolated_tui_files  # noqa: E402,F401
from tests.v2_joblist_support import job  # noqa: E402
from tests.v2_joblist_support import settle  # noqa: E402
from tests.v2_joblist_support import wait  # noqa: E402


async def _list(app, pilot) -> JobListScreen:
    assert await wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item
                      and not app.screen._refreshing)
    return app.screen


def _order(screen) -> list:
    table = screen.query_one("#jobs", DataTable)
    return [screen._job_by_item[row.key.value] for row in table.ordered_rows if row.key.value in screen._job_by_item]


def _rows(screen) -> list:
    return [row.key.value for row in screen.query_one("#jobs", DataTable).ordered_rows]


def _info(screen) -> str:
    return " ".join(str(screen.query_one("#listinfo", Static).render()).split())


def _sort_to(screen, key: str, descending: bool) -> None:
    while screen._sort != key:
        screen.action_cycle_sort()
    if screen._descending != descending:
        screen.action_toggle_order()


# ─── the status label says what shows first ──────────────────────────────────


def test_ascending_status_shows_the_ended_jobs_first_as_its_label_says():
    rows = [job("run", "RUNNING", minutes_ago=5), job("queue", "PENDING", minutes_ago=6),
            job("fail", "FAILED", minutes_ago=7), job("gone", "CANCELLED", minutes_ago=8)]

    async def run():
        app = CortexTrainingLogTUI(client(rows), poll_interval=0.01)
        async with app.run_test(size=(200, 30)) as pilot:
            s = await _list(app, pilot)
            _sort_to(s, "status", True)
            assert await wait(pilot, lambda: "sort: status ↓ running first" in _info(s))
            assert _rows(s)[0] == "hdr-live" and _order(s) == ["run", "queue", "fail", "gone"]
            await pilot.press("o")
            assert await wait(pilot, lambda: "sort: status ↑ ended first" in _info(s))
            # The ended jobs come first: HISTORY is drawn above LIVE, and the cursor is on its first job.
            assert _rows(s)[0] == "hdr-history" and _order(s) == ["gone", "fail", "queue", "run"]
            table = s.query_one("#jobs", DataTable)
            assert s._job_by_item.get(table.ordered_rows[table.cursor_row].key.value) == "gone"
            await pilot.press("s")  # created ↑ oldest first: LIVE above HISTORY again
            assert await wait(pilot, lambda: _rows(s)[0] == "hdr-live")
            await settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("key", views.SORT_KEYS)
def test_live_comes_first_but_under_ended_first(key):
    assert views.live_first(key, True)
    assert views.live_first(key, False) == (key != "status")
    assert views.sort_direction(key, False) == ("ended first" if key == "status" else "oldest first")


# ─── rows without the sorted time stay last ──────────────────────────────────


@pytest.mark.parametrize("key, field", [("created", "created_at"), ("updated", "updated_at")])
@pytest.mark.parametrize("descending", [True, False])
def test_rows_without_the_sorted_time_stay_after_the_timed_ones(key, field, descending):
    timed = [job("old", minutes_ago=300), job("new", minutes_ago=10), job("mid", minutes_ago=100)]
    # No time, a placeholder for one (Go's zero time), and an unparseable one: all shown as "—".
    untimed = [{**job("none-a", minutes_ago=50), field: None}, {**job("zero", minutes_ago=40),
                                                               field: "0001-01-01T00:00:00Z"},
               {**job("junk", minutes_ago=20), field: "not a time"}]
    ordered = [j["job_id"] for j in views.sort_job_list(untimed[:1] + timed + untimed[1:], key, descending)]
    expected = ["new", "mid", "old"] if descending else ["old", "mid", "new"]
    assert ordered[:3] == expected
    if key == "updated":
        assert ordered[3:] == ["junk", "zero", "none-a"]  # by created time, newest first, either way
    else:
        assert ordered[3:] == ["none-a", "zero", "junk"]  # no created time to go by: as they came


def test_in_the_list_an_unknown_row_without_times_stays_last_under_oldest_first():
    rows = [job("a", minutes_ago=10), job("b", minutes_ago=20),
            {"job_id": "u", "status": "UNKNOWN", "reason": "Job status unavailable: job not found on Control Plane",
             "comment": "never dated", "sub_jobs": []}]

    async def run():
        app = CortexTrainingLogTUI(client(rows), poll_interval=0.01)
        async with app.run_test(size=(200, 30)) as pilot:
            s = await _list(app, pilot)
            assert _order(s) == ["a", "b", "u"]
            await pilot.press("o")
            assert await wait(pilot, lambda: "oldest first" in _info(s))
            assert _order(s) == ["b", "a", "u"]
            await pilot.press("s")  # updated, oldest first
            assert await wait(pilot, lambda: s._sort == "updated")
            assert _order(s) == ["b", "a", "u"]
            await settle(app, pilot)

    asyncio.run(run())


# ─── the sort never moves the list ───────────────────────────────────────────


def test_every_sort_part_is_drawn_as_wide_as_the_widest():
    widths = {app_module._sort_part(key, descending).cell_length
              for key in views.SORT_KEYS for descending in (True, False)}
    assert app_module._SORT_PART_CELLS == max(widths) and len(widths) > 1  # "ended first" is the short one


def test_changing_the_sort_never_moves_the_list_at_any_width():
    """At about 104 columns the shorter "status ↑ ended first" fit on the first line where
    "status ↓ running first" didn't, so o moved the whole list by a row."""
    from cortex_training.tui.demo import DemoClient

    demo = DemoClient()

    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        app._lister.account_wide = True  # the demo's every-location list: its line is the one that moved
        async with app.run_test(size=(140, 40)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            seen = set()
            for width in range(96, 116):
                await pilot.resize_terminal(width, 40)
                assert await wait(pilot, lambda: s.size.width == width)
                tops = []
                for key, descending in [("status", True), ("status", False), ("status", True), ("created", True),
                                        ("created", False), ("updated", True), ("updated", False)]:
                    _sort_to(s, key, descending)
                    for _ in range(3):
                        await pilot.pause()
                    tops.append(table.region.y)
                assert len(set(tops)) == 1, (width, tops)
                seen.add(tops[0])
            assert len(seen) == 2  # the line above the list does wrap at some widths; the sort never decides it
            await settle(app, pilot)

    asyncio.run(run())
