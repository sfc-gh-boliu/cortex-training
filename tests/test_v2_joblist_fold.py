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

"""Unfolding and folding a job's sub-jobs: a click on its ▸/▾, → and ←, and e; space does nothing."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from tests.v2_joblist_support import client  # noqa: E402
from tests.v2_joblist_support import footer  # noqa: E402
from tests.v2_joblist_support import isolated_tui_files  # noqa: E402,F401
from tests.v2_joblist_support import job  # noqa: E402
from tests.v2_joblist_support import settle  # noqa: E402
from tests.v2_joblist_support import wait  # noqa: E402

# Newest first: the live job (unfolded, two sub-jobs), then a folded ended job with
# sub-jobs, then one without any (nothing to unfold: no marker).
ROWS = [job("live", "RUNNING", minutes_ago=5, subs=2), job("ended", minutes_ago=30, subs=2),
        job("bare", minutes_ago=50, subs=0)]


async def _list(app, pilot) -> JobListScreen:
    assert await wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item
                      and not app.screen._refreshing)
    screen = app.screen
    screen.set_focus(screen.query_one("#jobs", DataTable))
    await pilot.pause()
    return screen


def _key(screen, jid: str) -> str:
    return next(item for item, job_id in screen._job_by_item.items() if job_id == jid)


def _keys(screen) -> list:
    return [row.key.value for row in screen.query_one("#jobs", DataTable).ordered_rows]


def _subs(screen, jid: str) -> list:
    return [key for key, (job_id, _sid) in screen._sub_by_item.items() if job_id == jid]


def _name(screen, jid: str) -> str:
    table = screen.query_one("#jobs", DataTable)
    return str(table.get_cell(_key(screen, jid), "Name"))


def _cursor_key(screen):
    table = screen.query_one("#jobs", DataTable)
    return table.ordered_rows[table.cursor_row].key.value


def _offset(screen, key: str, x_in_name: int) -> tuple:
    """Where ``x_in_name`` cells into the Name column of row ``key`` is, relative to the table
    (``table.cell_padding`` is the ▸/▾ itself)."""
    table = screen.query_one("#jobs", DataTable)
    columns = table.ordered_columns
    at = [column.key.value for column in columns].index("Name")
    x = sum(column.get_render_width(table) for column in columns[:at]) + x_in_name
    y = (1 if table.show_header else 0) + table.get_row_index(key) - round(table.scroll_y)
    return x, y


def _move_to(screen, key: str) -> None:
    table = screen.query_one("#jobs", DataTable)
    table.move_cursor(row=table.get_row_index(key), animate=False)


def _run(rows, test, size=(160, 30)):
    async def run():
        app = CortexTrainingLogTUI(client(rows), poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            await test(app, pilot, await _list(app, pilot))
            assert app.return_code is None
            await settle(app, pilot)

    asyncio.run(run())


# ─── space is gone ───────────────────────────────────────────────────────────


def test_space_does_nothing_and_no_hint_offers_it():
    async def test(app, pilot, s):
        assert "space" not in s.active_bindings and "Sub-jobs" not in footer(s).values()
        assert "space" not in app.sub_title and "→/← sub-jobs" in app.sub_title
        for jid in ("live", "ended"):  # one unfolded, one folded
            _move_to(s, _key(s, jid))
            before = _keys(s)
            await pilot.press("space")
            for _ in range(3):
                await pilot.pause()
            assert _keys(s) == before and _cursor_key(s) == _key(s, jid)
        assert not isinstance(app.screen, JobSummaryScreen)

    _run(ROWS, test)


# ─── a click on the ▸/▾ ──────────────────────────────────────────────────────


def test_a_click_on_the_arrow_unfolds_and_folds_the_job_and_moves_the_cursor_to_it():
    async def test(app, pilot, s):
        table = s.query_one("#jobs", DataTable)
        assert _name(s, "live").startswith("▾ ") and _name(s, "ended").startswith("▸ ")
        assert _cursor_key(s) == _key(s, "live") and not _subs(s, "ended")
        arrow = table.cell_padding
        await pilot.click(table, offset=_offset(s, _key(s, "ended"), arrow))
        assert await wait(pilot, lambda: len(_subs(s, "ended")) == 2)
        assert _name(s, "ended").startswith("▾ ") and _cursor_key(s) == _key(s, "ended")
        assert app.screen is s  # not opened
        await pilot.click(table, offset=_offset(s, _key(s, "ended"), arrow))  # the highlighted row: still a toggle
        assert await wait(pilot, lambda: not _subs(s, "ended"))
        assert _name(s, "ended").startswith("▸ ") and app.screen is s
        # Only the arrow toggles: a click on the blank after the live job's ▾ highlights the row, as any click...
        await pilot.click(table, offset=_offset(s, _key(s, "live"), arrow + 1))
        assert await wait(pilot, lambda: _cursor_key(s) == _key(s, "live"))
        assert len(_subs(s, "live")) == 2 and app.screen is s
        # ...and its arrow folds it, though the row is highlighted (a second click on its name would open it).
        await pilot.click(table, offset=_offset(s, _key(s, "live"), arrow))
        assert await wait(pilot, lambda: not _subs(s, "live"))
        assert _cursor_key(s) == _key(s, "live") and app.screen is s
        # The cell's padding before the arrow is the row's too.
        await pilot.click(table, offset=_offset(s, _key(s, "ended"), arrow - 1))
        assert await wait(pilot, lambda: _cursor_key(s) == _key(s, "ended"))
        assert not _subs(s, "ended") and app.screen is s

    _run(ROWS, test)


def test_a_click_elsewhere_on_the_row_highlights_it_and_a_second_click_opens_it():
    async def test(app, pilot, s):
        table = s.query_one("#jobs", DataTable)
        name_text = table.cell_padding + 2  # the first letter of the name, after "▸ "
        await pilot.click(table, offset=_offset(s, _key(s, "ended"), name_text))
        assert await wait(pilot, lambda: _cursor_key(s) == _key(s, "ended"))
        assert not _subs(s, "ended") and app.screen is s  # highlighted, not unfolded
        await pilot.click(table, offset=_offset(s, _key(s, "ended"), name_text))
        assert await wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
        assert app.screen._job_id == "ended"

    _run(ROWS, test)


def test_a_click_on_the_status_column_of_a_row_is_the_tables_own():
    async def test(app, pilot, s):
        table = s.query_one("#jobs", DataTable)
        # The marker column (●/○), left of the Name column: never a toggle.
        await pilot.click(table, offset=_offset(s, _key(s, "ended"), -2))
        assert await wait(pilot, lambda: _cursor_key(s) == _key(s, "ended"))
        assert not _subs(s, "ended") and app.screen is s

    _run(ROWS, test)


def test_a_click_where_a_job_without_sub_jobs_has_no_arrow_does_nothing_special():
    async def test(app, pilot, s):
        table = s.query_one("#jobs", DataTable)
        assert not _name(s, "bare").startswith(("▸", "▾"))
        before = _keys(s)
        await pilot.click(table, offset=_offset(s, _key(s, "bare"), table.cell_padding))
        assert await wait(pilot, lambda: _cursor_key(s) == _key(s, "bare"))  # highlighted, as any click
        assert _keys(s) == before and app.screen is s

    _run(ROWS, test)


# ─── → and ← ─────────────────────────────────────────────────────────────────


def test_right_unfolds_left_folds_and_left_on_a_sub_job_row_folds_its_job_and_moves_to_it():
    async def test(app, pilot, s):
        _move_to(s, _key(s, "ended"))
        await pilot.press("right")
        assert await wait(pilot, lambda: len(_subs(s, "ended")) == 2)
        assert _cursor_key(s) == _key(s, "ended")  # the cursor stays on the job
        await pilot.press("left")
        assert await wait(pilot, lambda: not _subs(s, "ended"))
        assert _cursor_key(s) == _key(s, "ended")
        # From a sub-job row of the live job: its job folds, and the cursor goes to the job's row.
        _move_to(s, _subs(s, "live")[1])
        await pilot.press("left")
        assert await wait(pilot, lambda: not _subs(s, "live"))
        assert _cursor_key(s) == _key(s, "live")
        await pilot.press("right")
        assert await wait(pilot, lambda: len(_subs(s, "live")) == 2)
        assert _cursor_key(s) == _key(s, "live")

    _run(ROWS, test)


def test_where_nothing_unfolds_or_folds_the_arrows_scroll_sideways_on_a_narrow_terminal():
    async def test(app, pilot, s):
        table = s.query_one("#jobs", DataTable)
        assert await wait(pilot, lambda: table.max_scroll_x > 0)
        assert table.scroll_x == 0
        # An unfolded job: → has nothing to unfold, so the table scrolls right.
        _move_to(s, _key(s, "live"))
        await pilot.press("right")
        assert await wait(pilot, lambda: table.scroll_x == 1)
        assert len(_subs(s, "live")) == 2
        # A sub-job row: → scrolls too.
        _move_to(s, _subs(s, "live")[0])
        await pilot.press("right")
        assert await wait(pilot, lambda: table.scroll_x == 2)
        # A folded job: ← has nothing to fold, so it scrolls back.
        _move_to(s, _key(s, "ended"))
        await pilot.press("left")
        assert await wait(pilot, lambda: table.scroll_x == 1)
        # A job without sub-jobs, and a LIVE/HISTORY row: both directions scroll.
        _move_to(s, _key(s, "bare"))
        await pilot.press("right")
        assert await wait(pilot, lambda: table.scroll_x == 2)
        _move_to(s, "hdr-history")
        await pilot.press("left", "left")
        assert await wait(pilot, lambda: table.scroll_x == 0)
        assert not _subs(s, "ended") and len(_subs(s, "live")) == 2  # nothing unfolded or folded

    _run(ROWS, test, size=(80, 24))


def test_the_keys_panel_names_what_the_arrows_do():
    async def test(app, pilot, s):
        bindings = s.active_bindings
        assert bindings["right"].binding.description == "Unfold sub-jobs, else scroll right"
        assert bindings["left"].binding.description == "Fold sub-jobs, else scroll left"

    _run(ROWS, test)


def test_e_still_unfolds_and_folds_every_job():
    async def test(app, pilot, s):
        await pilot.press("e")  # not all unfolded: unfold all
        assert await wait(pilot, lambda: len(_subs(s, "ended")) == 2 and len(_subs(s, "live")) == 2)
        await pilot.press("e")  # all unfolded: fold all, the live ones too
        assert await wait(pilot, lambda: not s._sub_by_item)
        assert _name(s, "live").startswith("▸ ")

    _run(ROWS, test)


def test_the_arrow_stays_clickable_when_the_table_is_scrolled_down_and_sideways():
    rows = [job(f"j{i:02d}", minutes_ago=10 + i, subs=1) for i in range(30)]  # folded: they have ended

    async def test(app, pilot, s):
        table = s.query_one("#jobs", DataTable)
        assert await wait(pilot, lambda: table.max_scroll_x > 0 and table.max_scroll_y > 0)
        await pilot.press("end")  # the last job: the table scrolls down
        assert await wait(pilot, lambda: table.scroll_y > 0)
        _move_to(s, _key(s, "j29"))
        await pilot.press("right", "right")  # unfolds it, then scrolls the columns
        assert await wait(pilot, lambda: table.scroll_x == 1 and len(_subs(s, "j29")) == 1)
        # The marker and the name stay in place while the other columns scroll.
        await pilot.click(table, offset=_offset(s, _key(s, "j28"), table.cell_padding))
        assert await wait(pilot, lambda: len(_subs(s, "j28")) == 1)
        assert _cursor_key(s) == _key(s, "j28")
        # Unfolding and folding keep the columns where they were scrolled to.
        for _ in range(5):
            await pilot.pause()
        assert table.scroll_x == 1
        await pilot.press("left")  # folds j28: the columns stay
        assert await wait(pilot, lambda: not _subs(s, "j28"))
        for _ in range(5):
            await pilot.pause()
        assert table.scroll_x == 1

    _run(rows, test, size=(80, 24))
