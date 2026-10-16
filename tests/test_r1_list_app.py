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

"""The job list screen: drawing only what changed, the lines around the table, columns, refresh."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Input  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.screens import PromptScreen  # noqa: E402
from cortex_training.tui.stores import AliasStore  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_tui_files(tmp_path, monkeypatch):
    # As conftest does for test_tui_* files: no writes to the real cache or home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(jid: str, status: str = "CANCELLED", *, minutes_ago: float = 60, comment=None, **extra) -> dict:
    return {"job_id": jid, "status": status, "created_at": _iso(minutes_ago), "updated_at": _iso(minutes_ago / 2),
            "comment": f"job {jid}" if comment is None else comment, "submitted_by": "TEST_USER_A",
            "owner_role": "TRAINING_ROLE", "hardware": "H200",
            "sub_jobs": [{"sub_job_id": f"{jid}:training:0", "job_type": "TRAINING", "status": status,
                          "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8.0}}], **extra}


def _client(jobs: list) -> MagicMock:
    """The configured schema's list (the account-wide one isn't deployed); get_job answers the listed row."""
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.side_effect = lambda **_: [dict(j) for j in jobs]
    by_id = {j["job_id"]: j for j in jobs}
    c.get_job.side_effect = lambda jid, **_: dict(by_id.get(jid, {}))
    return c


async def _wait(pilot, predicate, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.02)
    return bool(predicate())


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


async def _list(app, pilot) -> JobListScreen:
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item
                       and not app.screen._refreshing)
    await pilot.pause()
    return app.screen


def _text(screen, selector) -> str:
    widget = screen.query_one(selector, Static)
    return str(widget.render()) if widget.display else ""


def _labels(screen) -> list:
    return [str(column.label) for column in screen.query_one("#jobs", DataTable).columns.values()]


def _cells(screen, jid) -> dict:
    key = next(item for item, job_id in screen._job_by_item.items() if job_id == jid)
    return dict(zip(_labels(screen), (str(cell) for cell in screen.query_one("#jobs", DataTable).get_row(key))))


def _order(screen) -> list:
    """Job ids in the order the table shows them."""
    table = screen.query_one("#jobs", DataTable)
    return [screen._job_by_item[row.key.value] for row in table.ordered_rows if row.key.value in screen._job_by_item]


def _count_fills(table: DataTable) -> list:
    """Record each time the table is filled: a replace_rows call, or an add_row outside one."""
    fills = []
    real_replace, real_add = table.replace_rows, table.add_row
    filling = []

    def replace_rows(rows):
        fills.append(("replace", len(rows)))
        filling.append(True)
        try:
            return real_replace(rows)
        finally:
            filling.pop()

    def add_row(*cells, **kwargs):
        if not filling:
            fills.append(("add", kwargs.get("key")))
        return real_add(*cells, **kwargs)

    table.replace_rows, table.add_row = replace_rows, add_row
    return fills


def _run(coro):
    asyncio.run(coro)


# ─── what didn't change isn't drawn again ────────────────────────────────────


def test_a_refresh_that_reads_the_same_list_draws_nothing_again():
    jobs = [_job(f"j{i}", "RUNNING" if i < 3 else "CANCELLED", minutes_ago=10 + i) for i in range(40)]
    c = _client(jobs)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            fills = _count_fills(table)
            before = {key: table.get_row(key) for key in s._job_by_item}
            await pilot.press("r")
            assert await _wait(pilot, lambda: c.list_jobs.call_count == 2 and not s._refreshing)
            await pilot.pause()
            assert fills == [], "the same list filled the table again"
            after = {key: table.get_row(key) for key in s._job_by_item}
            assert after.keys() == before.keys()
            assert all(a is b for key in before for a, b in zip(before[key], after[key])), "cells were built again"
            # A job that changed is drawn again, and only then is the table filled.
            jobs[5] = {**jobs[5], "status": "FAILED"}
            c.list_jobs.side_effect = lambda **_: [dict(j) for j in jobs]
            await pilot.press("r")
            assert await _wait(pilot, lambda: _cells(s, "j5")["Status"] == "FAILED")
            assert fills, "a changed job was not drawn"
            changed = {key: table.get_row(key) for key in s._job_by_item}
            j5 = next(key for key, jid in s._job_by_item.items() if jid == "j5")
            assert all(a is b for key in before if key != j5 for a, b in zip(before[key], changed[key])), (
                "the jobs that didn't change were built again")
            await _settle(app, pilot)

    _run(run())


def test_back_from_a_job_the_list_draws_nothing_again():
    jobs = [_job(f"j{i}", "RUNNING" if i < 3 else "CANCELLED", minutes_ago=10 + i) for i in range(40)]
    c = _client(jobs)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            s.set_focus(table)
            item = next(key for key, jid in s._job_by_item.items() if jid == "j7")
            table.move_cursor(row=table.get_row_index(item))
            await pilot.pause()
            jid = s._highlighted()[1]
            assert jid == "j7"
            fills = _count_fills(table)
            before = {key: table.get_row(key) for key in s._job_by_item}
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            await pilot.pause()
            await pilot.press("b")
            assert await _wait(pilot, lambda: app.screen is s)
            for _ in range(3):
                await pilot.pause()
            assert fills == [], "coming back filled the table again"
            after = {key: table.get_row(key) for key in s._job_by_item}
            assert all(a is b for key in before for a, b in zip(before[key], after[key])), "cells were built again"
            assert s._highlighted()[1] == jid, "the cursor stays on the job that was opened"
            await _settle(app, pilot)

    _run(run())


def test_a_new_order_moves_the_rows_without_adding_them_again():
    jobs = [_job(f"j{i}", minutes_ago=10 + i, updated_at=_iso(100 - i)) for i in range(30)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(220, 40)) as pilot:  # every column fits: no column moves
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            assert _order(s) == [f"j{i}" for i in range(30)]  # newest created first
            fills = _count_fills(table)
            await pilot.press("o")
            assert await _wait(pilot, lambda: _order(s) == [f"j{i}" for i in reversed(range(30))])
            await pilot.press("o", "s")  # by updated, newest first: the reverse of created
            assert await _wait(pilot, lambda: s._sort == "updated" and _order(s)[0] == "j29")
            assert fills == [], "rows that only moved were added again"
            await _settle(app, pilot)

    _run(run())


def test_unfolding_and_folding_every_job_builds_each_job_row_once(monkeypatch):
    jobs = [_job(f"j{i}", "RUNNING" if i < 2 else "CANCELLED", minutes_ago=10 + i) for i in range(20)]
    built = {"cells": 0, "subs": 0}
    real_cells, real_subs = JobListScreen._job_cells, JobListScreen._sub_rows

    def job_cells(self, job, compact):
        built["cells"] += 1
        return real_cells(self, job, compact)

    def sub_rows(self, job):
        built["subs"] += 1
        return real_subs(self, job)

    monkeypatch.setattr(JobListScreen, "_job_cells", job_cells)
    monkeypatch.setattr(JobListScreen, "_sub_rows", sub_rows)

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            assert built == {"cells": 20, "subs": 2}  # the live jobs start unfolded
            keys = lambda: [row.key.value for row in table.ordered_rows]  # noqa: E731
            subs = lambda: [key for key in keys() if key.startswith("sub-")]  # noqa: E731
            assert len(subs()) == 2
            await pilot.press("e")  # every job unfolds
            assert await _wait(pilot, lambda: len(subs()) == 20)
            assert built == {"cells": 20, "subs": 20}, "unfolding built more than the sub-job rows"
            item = {jid: key for key, jid in s._job_by_item.items()}
            assert keys().index(f"sub-{item['j7']}-0") == keys().index(item["j7"]) + 1
            assert _cells(s, "j7")["Name"].startswith("▾ ")
            await pilot.press("e")  # every job folds
            assert await _wait(pilot, lambda: not subs())
            assert _cells(s, "j7")["Name"].startswith("▸ ") and _cells(s, "j0")["Name"].startswith("▸ ")
            await pilot.press("e")
            assert await _wait(pilot, lambda: len(subs()) == 20)
            assert built == {"cells": 20, "subs": 20}, "folding and unfolding again built rows again"
            await _settle(app, pilot)

    _run(run())


def test_the_fast_fill_leaves_the_table_as_clear_and_add_row_do():
    from textual.app import App
    from textual.widgets.data_table import DuplicateKey

    from cortex_training.tui.app import _cell
    from cortex_training.tui.app import _JobTable

    columns = [("Name", 20), ("Status", 10), ("Type", 8)]

    class Tables(App):
        def compose(self):
            yield _JobTable(id="fast", cursor_type="row")
            yield _JobTable(id="slow", cursor_type="row")

    async def run():
        app = Tables()
        async with app.run_test(size=(80, 30)) as pilot:
            fast, slow = app.query_one("#fast", _JobTable), app.query_one("#slow", _JobTable)
            for table in (fast, slow):
                for label, width in columns:
                    table.add_column(label, width=width, key=label)
            assert fast._fills_fast()
            for size in (12, 30, 5):
                rows = [(f"r{i}", (_cell(f"job {i}", "bold"), _cell("RUNNING"), _cell("train"))) for i in range(size)]
                fast.replace_rows(rows)
                slow.clear()
                for key, cells in rows:
                    slow.add_row(*cells, key=key)
                await pilot.pause()
                await pilot.pause()
                assert [row.key.value for row in fast.ordered_rows] == [row.key.value for row in slow.ordered_rows]
                assert all(fast.get_row(key) == slow.get_row(key) for key, _cells in rows)
                assert (fast.row_count, fast.virtual_size, fast.cursor_coordinate) == (
                    slow.row_count, slow.virtual_size, slow.cursor_coordinate)
                assert fast.get_row_index("r3") == 3
            fast.move_cursor(row=4)
            await pilot.pause()
            assert fast.coordinate_to_cell_key(fast.cursor_coordinate).row_key.value == "r4"
            # Rendered as add_row's rows are.
            assert "".join(seg.text for seg in fast.render_line(2)) == "".join(seg.text for seg in slow.render_line(2))
            with pytest.raises(DuplicateKey):  # and the table is left as it was
                fast.replace_rows([("a", ()), ("a", ())])
            assert fast.row_count == 5 and fast.get_row_index("r4") == 4
            # A column that sizes itself to its cells: the rows go in one at a time, as add_row puts them.
            fast.clear(columns=True)
            fast.add_column("Auto", key="Auto")
            assert not fast._fills_fast()
            fast.replace_rows([("w", (_cell("a cell"),)), ("x", (_cell("another"),))])
            assert fast.row_count == 2 and str(fast.get_row("x")[0]) == "another"

    _run(run())


def test_a_search_on_a_member_key_reads_the_jobs_once_per_list(monkeypatch):
    from cortex_training.tui import views

    jobs = [_job(f"j{i}", minutes_ago=10 + i) for i in range(10)]
    jobs[3]["sub_jobs"][0]["training_config"]["ep_size"] = 4
    c = _client(jobs)
    resolved = []
    real = views.resolve_query

    def resolve_query(query, all_jobs):
        resolved.append(len(list(all_jobs)) if not isinstance(all_jobs, list) else len(all_jobs))
        return real(query, all_jobs)

    monkeypatch.setattr(views, "resolve_query", resolve_query)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            s.query_one("#jobfilter", Input).value = "ep_size>1"
            assert await _wait(pilot, lambda: set(s._job_by_item.values()) == {"j3"})
            count = len(resolved)
            s.set_focus(s.query_one("#jobs", DataTable))
            await pilot.press("s", "o", "s")  # three renders of the same search over the same jobs
            assert await _wait(pilot, lambda: s._sort == "status")
            assert len(resolved) == count, "the same search over the same jobs was resolved again"
            await pilot.press("r")  # a new list: resolved again
            assert await _wait(pilot, lambda: c.list_jobs.call_count == 2 and not s._refreshing)
            await pilot.pause()
            assert len(resolved) == count + 1 and set(s._job_by_item.values()) == {"j3"}
            await _settle(app, pilot)

    _run(run())


# ─── the heap is frozen in a terminal, not in a test run ─────────────────────


def test_a_terminal_run_freezes_the_heap_at_start_and_after_the_first_list(monkeypatch):
    from cortex_training.tui import app as app_module

    frozen = []
    monkeypatch.setattr(app_module, "_freeze_heap", lambda: frozen.append(time.monotonic()))
    monkeypatch.setattr(CortexTrainingLogTUI, "_FREEZE_HEAP_AFTER_SECONDS", 0.05)
    c = _client([_job("j1"), _job("j2", "RUNNING")])

    class Terminal(CortexTrainingLogTUI):
        def _freezes_heap(self) -> bool:  # as in a terminal: run_test is headless
            return True

    async def run():
        app = Terminal(c, poll_interval=0.01)
        async with app.run_test(size=(120, 30)) as pilot:
            s = await _list(app, pilot)
            assert await _wait(pilot, lambda: len(frozen) == 2)
            await pilot.press("r")  # later lists don't freeze it again
            assert await _wait(pilot, lambda: c.list_jobs.call_count == 2 and not s._refreshing)
            await pilot.pause(0.2)
            assert len(frozen) == 2
            await _settle(app, pilot)
        plain = CortexTrainingLogTUI(_client([_job("j1")]), poll_interval=0.01)
        async with plain.run_test(size=(120, 30)) as pilot:  # headless: a test, never frozen
            await _list(plain, pilot)
            await pilot.pause(0.2)
            assert len(frozen) == 2
            await _settle(plain, pilot)

    _run(run())


def test_the_heap_is_not_frozen_while_a_job_is_open_over_the_list(monkeypatch):
    from cortex_training.tui import app as app_module

    frozen, running = [], {}
    monkeypatch.setattr(app_module, "_freeze_heap",
                        lambda: frozen.append(type(running["app"].screen) if running["app"].is_running else None))
    monkeypatch.setattr(CortexTrainingLogTUI, "_FREEZE_HEAP_AFTER_SECONDS", 0.3)
    jobs = [_job("j1"), _job("j2")]
    c = _client(jobs)

    class Terminal(CortexTrainingLogTUI):
        def _freezes_heap(self) -> bool:
            return True

    async def run():
        app = running["app"] = Terminal(c, poll_interval=0.01)
        async with app.run_test(size=(120, 30)) as pilot:
            s = await _list(app, pilot)
            assert len(frozen) == 1  # at start
            app.push_screen(JobSummaryScreen(c, "j1", job=jobs[0]))  # opened before the list's freeze
            await pilot.pause(0.5)
            assert len(frozen) == 1, "frozen with a job's screen open, which would then never be collected"
            await pilot.press("b")
            assert await _wait(pilot, lambda: app.screen is s)
            await pilot.press("r")  # the list's next load freezes it, the list on top
            assert await _wait(pilot, lambda: len(frozen) == 2)
            assert frozen[-1] is JobListScreen
            await _settle(app, pilot)

    _run(run())


# ─── a new sort shows the start of the list ──────────────────────────────────


def test_a_new_sort_or_order_puts_the_cursor_on_the_first_job():
    jobs = [_job(f"j{i:02d}", minutes_ago=10 + i, updated_at=_iso(1000 - 10 * i)) for i in range(60)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(160, 30)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            s.set_focus(table)
            for _ in range(20):
                await pilot.press("down")
            assert s._highlighted()[1] == "j20"  # from the first job, twenty rows down
            await pilot.press("s")  # by updated: j59 is the newest
            assert await _wait(pilot, lambda: table.cursor_row == 1 and s._highlighted()[1] == "j59")
            assert table.scroll_y == 0, "the start of the new order is in view"
            await pilot.press("down", "down", "o")
            assert await _wait(pilot, lambda: table.cursor_row == 1 and s._highlighted()[1] == "j00")
            await _settle(app, pilot)

    _run(run())


# ─── a failed refresh is said in the subtitle too ────────────────────────────


def test_a_first_load_that_fails_says_load_failed_not_refresh_failed():
    c = _client([_job("j1")])
    c.list_jobs.side_effect = RuntimeError("503 Service Unavailable")

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._list_error
                               and not app.screen._refreshing)
            s = app.screen
            await pilot.pause()
            assert re.match(r"load failed \d{2}:\d{2}:\d{2}  ·  ", app.sub_title), app.sub_title
            assert "503 Service Unavailable" in _text(s, "#listinfo")
            c.list_jobs.side_effect = lambda **_: [_job("j1")]
            await pilot.press("r")
            assert await _wait(pilot, lambda: s._job_by_item and not s._refreshing)
            await pilot.pause()
            assert app.sub_title.startswith("last refreshed ")
            await _settle(app, pilot)

    _run(run())


def test_a_failed_refresh_says_when_in_the_subtitle_and_the_notice_until_one_succeeds():
    jobs = [_job("j1", "RUNNING"), _job("j2")]
    c = _client(jobs)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(80, 30)) as pilot:
            s = await _list(app, pilot)
            listed = s._listed_at
            c.list_jobs.side_effect = RuntimeError("503 Service Unavailable")
            await pilot.press("r")
            assert await _wait(pilot, lambda: s._list_error and not s._refreshing)
            await pilot.pause()
            failed = s._failed_at
            assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", failed)
            assert app.sub_title.startswith(f"refresh failed {failed} (list from {listed})  ·  ")
            notice = s.query_one("#listnotice", Static)
            assert notice.display and notice.size.height >= 2, "wrapped at 80 columns, not cut"
            text = " ".join(_text(s, "#listnotice").split())
            assert text == (f"refresh failed {failed}: list_jobs failed: RuntimeError: 503 Service Unavailable; "
                            f"showing the list from {listed}")
            # With a search too.
            s.query_one("#jobfilter", Input).value = "j1"
            assert await _wait(pilot, lambda: set(s._job_by_item.values()) == {"j1"})
            assert "503 Service Unavailable" in " ".join(_text(s, "#listnotice").split())
            c.list_jobs.side_effect = lambda **_: [dict(j) for j in jobs]
            await pilot.press("escape", "r")
            assert await _wait(pilot, lambda: not s._list_error and not s._refreshing)
            await pilot.pause()
            assert app.sub_title.startswith("last refreshed ") and not s.query_one("#listnotice", Static).display
            await _settle(app, pilot)

    _run(run())


# ─── the line under the header wraps instead of being cut ────────────────────


@pytest.mark.parametrize("cols", [80, 100, 140])
def test_the_search_count_and_the_whole_query_show_at_any_width(cols):
    jobs = [_job("lora1", comment="sft lora sweep",
                 sub_jobs=[{"sub_job_id": "lora1:training:0", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B",
                            "training_config": {"n_gpus": 8.0, "optimizer": {"name": "adamw", "lr": 1e-5},
                                                "peft_config": {"r": 16}}}]),
            _job("full1", comment="full run")]
    query = "model:qwen3-8b method:lora lr<=1e-4"

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(cols, 30)) as pilot:
            s = await _list(app, pilot)
            s.query_one("#jobfilter", Input).value = query
            assert await _wait(pilot, lambda: set(s._job_by_item.values()) == {"lora1"})
            await pilot.pause()
            info = " ".join(_text(s, "#listinfo").split())
            assert f"1 match(es) for {query}" in info, info
            assert info.endswith("sort: created ↓ newest first (o reverses)"), info
            widget = s.query_one("#listinfo", Static)
            lines = str(widget.render()).split("\n")
            assert widget.size.height == len(lines), "every wrapped line is shown"
            assert all(line.startswith("  ") for line in lines[1:]), "continued under the text, not at the edge"
            await _settle(app, pilot)

    _run(run())


# ─── the search box stays at the bottom ──────────────────────────────────────


def test_the_search_box_stays_at_the_bottom_when_nothing_matches():
    async def run():
        app = CortexTrainingLogTUI(_client([_job("j1"), _job("j2")]), poll_interval=0.01)
        async with app.run_test(size=(100, 30)) as pilot:
            s = await _list(app, pilot)
            box = s.query_one("#jobfilter", Input)
            await pilot.press("/")
            await pilot.pause()
            at = box.region.y
            box.value = "nothing-matches-this"
            assert await _wait(pilot, lambda: s.query_one("#nomatch", Static).display)
            await pilot.pause()
            assert box.region.y == at, "the search box moved"
            await _settle(app, pilot)

    _run(run())


# ─── the no-match message and the help wrap under their own start ────────────


def test_the_no_match_message_wraps_each_point_under_its_start():
    async def run():
        app = CortexTrainingLogTUI(_client([_job("j1")]), poll_interval=0.01)
        async with app.run_test(size=(80, 40)) as pilot:
            s = await _list(app, pilot)
            s.query_one("#jobfilter", Input).value = "ep_size>1 sweep"
            assert await _wait(pilot, lambda: s.query_one("#nomatch", Static).display)
            await pilot.pause()
            lines = _text(s, "#nomatch").split("\n")
            assert len(lines) > 4 and any(len(line) > 70 for line in lines[:-1])  # some point wraps
            assert all(line.startswith("  ") for line in lines[1:]), lines
            assert any("unknown field 'ep_size'" in line for line in lines)
            await pilot.press("/")
            await pilot.pause()
            help_lines = _text(s, "#searchhelp").split("\n")
            assert help_lines[0].startswith("fields:") and all(line.startswith(" ") for line in help_lines[1:]
                                                                if not line.startswith("ops:"))
            await _settle(app, pilot)

    _run(run())


def test_a_long_no_match_message_fits_beside_the_search_help_and_scrolls_where_it_cant():
    from textual.containers import VerticalScroll

    query = "ep_size>1 sweep foo:bar"

    async def run():
        app = CortexTrainingLogTUI(_client([_job("j1"), _job("j2")]), poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            s = await _list(app, pilot)
            await pilot.press("/", *query)
            box = s.query_one("#nomatch-box", VerticalScroll)
            message = s.query_one("#nomatch", Static)
            assert await _wait(pilot, lambda: message.display and box.display)
            await pilot.pause()
            # The help under the box lists the fields, so the message leaves them out, and fits whole.
            assert s.query_one("#searchhelp", Static).display
            lines = _text(s, "#nomatch").split("\n")
            assert lines[-1].strip() == "Esc clears the search." and not any("Fields:" in line for line in lines)
            assert message.size.height == len(lines), "every line is laid out, none cut off"
            assert box.max_scroll_y == 0, "all of it in view"
            # Away from the box the help hides on a short terminal, and the message lists the fields itself.
            s.set_focus(box)
            assert await _wait(pilot, lambda: not s.query_one("#searchhelp", Static).display
                               and "Fields:" in _text(s, "#nomatch"))
            await _settle(app, pilot)
        app = CortexTrainingLogTUI(_client([_job("j1"), _job("j2")]), poll_interval=0.01)
        async with app.run_test(size=(80, 16)) as pilot:  # too short for it: it scrolls to its last point
            s = await _list(app, pilot)
            await pilot.press("/", *query)
            box = s.query_one("#nomatch-box", VerticalScroll)
            message = s.query_one("#nomatch", Static)
            assert await _wait(pilot, lambda: message.display and box.display)
            await pilot.pause()
            lines = _text(s, "#nomatch").split("\n")
            assert message.size.height == len(lines) and box.max_scroll_y > 0
            box.scroll_end(animate=False)
            assert await _wait(pilot, lambda: box.scroll_y == box.max_scroll_y)
            # A new message starts at its top.
            await pilot.press("x")
            assert await _wait(pilot, lambda: "foo:barx" in _text(s, "#nomatch") and box.scroll_y == 0)
            await _settle(app, pilot)

    _run(run())


# ─── columns, fixed name, Home/End, help on short terminals ──────────────────


def test_the_first_columns_fit_100_columns_and_the_name_stays_while_scrolling():
    jobs = [_job("j1", comment="a fairly long comment that names the run"), _job("j2", "RUNNING")]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(100, 30)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            assert table.fixed_columns == 2
            widths = [(label, column.get_render_width(table)) for label, column in
                      zip(_labels(s), table.columns.values())]
            ends = 0
            for label, width in widths:
                ends += width
                if label == "Owner":
                    break
            assert ends <= table.scrollable_content_region.width, widths
            assert re.fullmatch(r"\d{2}-\d{2} \d{2}:\d{2}", _cells(s, "j1")["Created"]), "no year where it is narrow"
            assert _cells(s, "j1")["Owner"] == "TEST_USER_A"
            await _settle(app, pilot)

    _run(run())


def test_sorted_by_updated_the_updated_column_is_in_view():
    jobs = [_job(f"j{i}", minutes_ago=10 + i) for i in range(5)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(140, 30)) as pilot:
            s = await _list(app, pilot)
            assert _labels(s).index("Created") == 5 and _labels(s)[-1] == "Updated"
            await pilot.press("s")
            assert await _wait(pilot, lambda: _labels(s).index("Updated") == 5)
            assert _labels(s)[-1] == "Created"
            await pilot.press("s")
            assert await _wait(pilot, lambda: _labels(s).index("Created") == 5)
            await _settle(app, pilot)

    _run(run())


def test_home_and_end_go_to_the_first_and_the_last_job():
    jobs = [_job(f"j{i:02d}", minutes_ago=10 + i) for i in range(50)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(120, 30)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            s.set_focus(table)
            await pilot.press("end")
            assert await _wait(pilot, lambda: s._highlighted()[1] == "j49")
            await pilot.press("home")
            assert await _wait(pilot, lambda: s._highlighted()[1] == "j00" and table.scroll_y == 0)
            await _settle(app, pilot)

    _run(run())


@pytest.mark.parametrize("rows, always", [(24, False), (40, True)])
def test_the_search_help_shows_only_while_searching_on_a_short_terminal(rows, always):
    async def run():
        app = CortexTrainingLogTUI(_client([_job("j1")]), poll_interval=0.01)
        async with app.run_test(size=(80, rows)) as pilot:
            s = await _list(app, pilot)
            help_line = s.query_one("#searchhelp", Static)
            assert help_line.display is always
            await pilot.press("/")
            assert await _wait(pilot, lambda: help_line.display)
            await pilot.press("escape")
            assert await _wait(pilot, lambda: help_line.display is always)
            await _settle(app, pilot)

    _run(run())


# ─── back to a stale list ────────────────────────────────────────────────────


def test_back_from_a_preview_the_list_shows_what_it_read_and_rereads_a_stale_list(monkeypatch):
    jobs = [_job("j1", "RUNNING", minutes_ago=5), _job("j2")]
    c = _client(jobs)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            assert _cells(s, "j1")["Status"] == "RUNNING"
            # The job ends while its preview is open: the preview's read replaces the list's copy.
            c.get_job.side_effect = lambda jid, **_: {**jobs[0], "status": "CANCELLED", "reason": "cancelled_by_client"}
            table = s.query_one("#jobs", DataTable)
            s.set_focus(table)
            table.move_cursor(row=table.get_row_index(next(k for k, v in s._job_by_item.items() if v == "j1")))
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            assert await _wait(pilot, lambda: s._jobs_by_id["j1"]["status"] == "CANCELLED")
            calls = c.list_jobs.call_count
            await pilot.press("b")
            assert await _wait(pilot, lambda: app.screen is s and _cells(s, "j1")["Status"] == "CANCELLED")
            assert c.list_jobs.call_count == calls, "a list read within the interval is not read again"
            # Away for longer than the interval: the list is read again on return.
            monkeypatch.setattr(s, "_loaded_at", time.monotonic() - JobListScreen._REFRESH_SECONDS - 1)
            app.push_screen(JobSummaryScreen(c, "j2", job=jobs[1]))
            await pilot.pause()
            await pilot.press("b")
            assert await _wait(pilot, lambda: c.list_jobs.call_count == calls + 1)
            await _settle(app, pilot)

    _run(run())


# ─── UNKNOWN jobs ────────────────────────────────────────────────────────────


def test_unknown_jobs_are_marked_and_one_the_control_plane_has_no_record_of_is_history():
    outage = {**_job("u1", "UNKNOWN"), "reason": "Job status unavailable: Control Plane is unavailable"}
    gone = {**_job("g1", "UNKNOWN"), "reason": "Job status unavailable: the job was not found in the Control Plane"}
    jobs = [outage, gone, _job("r1", "RUNNING"), _job("c1")]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            keys = [row.key.value for row in table.ordered_rows]
            history_at = keys.index("hdr-history")
            item = {jid: key for key, jid in s._job_by_item.items()}
            assert keys.index(item["u1"]) < history_at < keys.index(item["g1"])
            assert str(table.get_row("hdr-live")[1]).strip() == "LIVE (2)  running, queued, ?"
            assert str(table.get_row("hdr-history")[1]).strip() == "HISTORY (2)  ended or failed"
            assert _cells(s, "u1")[" "] == "?" and _cells(s, "g1")[" "] == "?" and _cells(s, "r1")[" "] == "●"
            assert "● 2 live" in _text(s, "#listinfo")
            await _settle(app, pilot)

    _run(run())


def test_a_job_that_turns_unknown_keeps_its_creation_time_and_place():
    job = _job("u1", "RUNNING", minutes_ago=30)
    jobs = [job, _job("c1", minutes_ago=60)]
    c = _client(jobs)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            created = _cells(s, "u1")["Created"]
            # The configured schema's list answers an UNKNOWN row without any times.
            unknown = {k: v for k, v in job.items() if k not in ("created_at", "updated_at")}
            unknown.update(status="UNKNOWN", reason="Job status unavailable: Control Plane is unavailable")
            c.list_jobs.side_effect = lambda **_: [dict(unknown), dict(jobs[1])]
            await pilot.press("r")
            assert await _wait(pilot, lambda: _cells(s, "u1")["Status"] == "UNKNOWN")
            assert _cells(s, "u1")["Created"] == created != "—"
            assert s._jobs_by_id["u1"]["created_at"] == job["created_at"]
            await _settle(app, pilot)

    _run(run())


# ─── the LIVE and HISTORY rows are never cut ─────────────────────────────────


@pytest.mark.parametrize("cols", [80, 100, 160])
def test_the_live_and_history_rows_say_what_they_hold_whole_at_any_width(cols):
    jobs = [_job(f"r{i}", "RUNNING", minutes_ago=5 + i) for i in range(12)]
    jobs += [_job(f"h{i}", minutes_ago=100 + i) for i in range(1234)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(cols, 30)) as pilot:
            s = await _list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            for key, title in (("hdr-live", "LIVE (12)  running or queued"),
                               ("hdr-history", "HISTORY (1234)  ended or failed")):
                cells = [str(cell) for cell in table.get_row(key)]
                assert not any(cell.endswith("…") for cell in cells), cells
                assert " ".join(cell for cell in cells if cell).split() == title.split()
                # Each piece fits its column: drawn whole, as the text says.
                drawn = "".join(seg.text for seg in table.render_line(1 + table.get_row_index(key)))
                assert " ".join(drawn.split()) == " ".join(title.split()), drawn
                if cols >= 160:
                    assert cells[1] == title  # all of it beside the name, where there is room
            await _settle(app, pilot)

    _run(run())


# ─── owner, sub-job rows, log_prob ───────────────────────────────────────────


def test_owner_role_sub_job_ids_and_log_prob_names_in_the_list():
    lp = {**_job("lp1", "RUNNING", submitted_by=""),
          "sub_jobs": [{"sub_job_id": "lp1:training:0", "job_type": "TRAINING", "status": "RUNNING",
                        "training_config": {"n_gpus": 8.0}},
                       {"sub_job_id": "lp1:log_prob:12", "job_type": "LOG_PROBABILITY", "status": "RUNNING",
                        "inference_config": {"n_gpus": 1.0}}]}

    async def run():
        app = CortexTrainingLogTUI(_client([lp]), poll_interval=0.01)
        async with app.run_test(size=(180, 40)) as pilot:
            s = await _list(app, pilot)
            cells = _cells(s, "lp1")
            assert cells["Owner"] == "role TRAINING_ROLE" and cells["Type"] == "train+logprob"
            table = s.query_one("#jobs", DataTable)
            item = next(k for k, v in s._job_by_item.items() if v == "lp1")
            sub = dict(zip(_labels(s), (str(c) for c in table.get_row(f"sub-{item}-1"))))
            assert sub["Job ID"] == "log_prob…" or sub["Job ID"] == "log_prob:12"
            assert sub["Name"].strip().endswith("log_prob #12") and sub["Type"] == "log_prob"
            assert dict(zip(_labels(s), (str(c) for c in table.get_row(f"sub-{item}-0"))))["Job ID"] == "training:0"
            # The preview's sub-job table types it as the list does, beside the same label.
            table.move_cursor(row=table.get_row_index(item))
            s.set_focus(table)
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.query("#subjobs"))
            subjobs = app.screen.query_one("#subjobs", DataTable)
            assert await _wait(pilot, lambda: subjobs.row_count >= 3)
            rows = {str(subjobs.get_row(key)[0]).strip(): str(subjobs.get_row(key)[1]) for key in ("sj-0", "sj-1")}
            assert rows == {"training #0": "training", "log_prob #12": "log_prob"}
            await _settle(app, pilot)

    _run(run())


# ─── a search that only matches a job older than six months ──────────────────


def test_a_search_matching_only_an_older_job_names_it():
    old = _job("fake-old-job-0001", minutes_ago=200 * 24 * 60)
    jobs = [_job("j1"), old]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            assert "1 older hidden" in _text(s, "#listinfo")
            s.query_one("#jobfilter", Input).value = "fake-old"
            assert await _wait(pilot, lambda: s.query_one("#nomatch", Static).display)
            await pilot.pause()
            message = " ".join(_text(s, "#nomatch").split())
            assert ("1 older job matches (the list hides jobs older than 6 months): cortex-training tui "
                    f"{old['job_id']} opens it.") in message
            await _settle(app, pilot)

    _run(run())


# ─── the rename prompt takes at most what is kept ────────────────────────────


def test_the_rename_prompt_takes_at_most_80_characters_and_says_so():
    async def run():
        app = CortexTrainingLogTUI(_client([_job("j1")]), poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            s.set_focus(s.query_one("#jobs", DataTable))
            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            box = app.screen.query_one("#prompt-input", Input)
            assert box.max_length == 80
            assert "up to 80 characters" in str(app.screen.query_one("#prompt-hint").render())
            await pilot.press(*("x" * 100), "enter")
            assert await _wait(pilot, lambda: app.screen is s)
            assert AliasStore().get("j1") == "x" * 80
            await _settle(app, pilot)

    _run(run())
