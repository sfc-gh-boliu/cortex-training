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

"""The job list's search box gets every key typed after ``/``, in order.

A terminal delivers a burst of keys (fast typing, a paste of keystrokes, or keys
typed while the list re-renders) to the app queue before the screen has handled
the first one. These tests post such a burst in one go, as the terminal driver
does, rather than one key per idle cycle as ``pilot.press`` does.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual import events  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Input  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402

_KEY_NAMES = {"/": "slash", " ": "space"}


@pytest.fixture(autouse=True)
def _isolate_tui_files(tmp_path, monkeypatch):
    # As conftest does for test_tui_* files: no writes to the real cache or home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _client():
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [
        {"job_id": "aaaa1111", "status": "RUNNING", "comment": "lima sweep",
         "sub_jobs": [{"sub_job_id": "aaaa1111:training:0", "job_type": "TRAINING"}]},
        {"job_id": "bbbb2222", "status": "FAILED", "comment": "oscar a/b run"},
        {"job_id": "cccc3333", "status": "CANCELLED", "comment": "kilo"},
    ]
    c.get_job.side_effect = lambda jid, **_: {"job_id": jid, "status": "CANCELLED"}
    return c


def _burst(app, keys: list) -> None:
    """Post ``keys`` to the app queue at once, as a terminal driver does with a burst of input."""
    for key in keys:
        if len(key) == 1:
            event = events.Key(_KEY_NAMES.get(key, key), key)
        else:
            event = events.Key(key, None)
        event.set_sender(app)
        app.post_message(event)


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause()
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


def _shown(screen) -> set:
    return set(screen._job_by_item.values())


def test_keys_typed_right_after_slash_go_to_the_search_box_in_order():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            box = screen.query_one("#jobfilter", Input)
            assert screen.focused is screen.query_one("#jobs", DataTable)
            _burst(app, ["/", *"lima"])
            assert await _wait(pilot, lambda: box.value == "lima")
            assert await _wait(pilot, lambda: _shown(screen) == {"aaaa1111"})
            await _settle(app, pilot)

    asyncio.run(run())


def test_esc_then_slash_then_a_query_replaces_the_old_search():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            box = screen.query_one("#jobfilter", Input)
            await pilot.press("/", *"kilo", "enter")
            assert await _wait(pilot, lambda: _shown(screen) == {"cccc3333"})
            assert screen.focused is screen.query_one("#jobs", DataTable)
            # Esc clears the search, / opens the box again, and the new query lands in it.
            _burst(app, ["escape", "/", *"lima"])
            assert await _wait(pilot, lambda: box.value == "lima")
            assert await _wait(pilot, lambda: _shown(screen) == {"aaaa1111"})
            await _settle(app, pilot)

    asyncio.run(run())


async def _opened(app, pilot) -> str | None:
    """The job whose preview the keys opened (None: none opened)."""
    from cortex_training.tui.screens import JobSummaryScreen

    if not await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen)):
        return None
    return app.screen._job_id


def test_a_query_and_enter_enter_typed_in_one_burst_open_the_job_it_finds():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            assert screen._highlighted()[1] == "aaaa1111"
            # Enter arrives while "oscar" is still on its way to the box: it searches
            # for all of it, and the second Enter opens what that found.
            _burst(app, ["/", *"oscar", "enter", "enter"])
            assert await _opened(app, pilot) == "bbbb2222"
            assert screen.query_one("#jobfilter", Input).value == "oscar" and screen._filter == "oscar"
            await _settle(app, pilot)

    asyncio.run(run())


def test_esc_a_new_query_and_enter_enter_in_one_burst_open_the_new_match():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            await pilot.press("/", *"kilo", "enter")
            assert await _wait(pilot, lambda: _shown(screen) == {"cccc3333"})
            _burst(app, ["escape", "/", *"lima", "enter", "enter"])
            assert await _opened(app, pilot) == "aaaa1111"
            await _settle(app, pilot)

    asyncio.run(run())


def test_enter_enter_open_the_match_of_a_search_that_follows_one_that_found_nothing():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            await pilot.press("/", *"zulu", "enter")
            assert await _wait(pilot, lambda: not screen._job_by_item and screen.query_one("#nomatch").display)
            # The table comes back in the same burst that picks a row in it.
            _burst(app, ["escape", "/", *"oscar", "enter", "enter"])
            assert await _opened(app, pilot) == "bbbb2222"
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_query_and_esc_in_one_burst_leave_no_search():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            box = screen.query_one("#jobfilter", Input)
            _burst(app, ["/", *"oscar", "escape"])
            for _ in range(30):  # longer than the search waits for typing to pause
                await pilot.pause(0.01)
            assert box.value == "" and screen._filter == ""
            assert _shown(screen) == {"aaaa1111", "bbbb2222", "cccc3333"}
            assert screen.focused is screen.query_one("#jobs", DataTable)
            # A key after Esc goes to the list.
            _burst(app, ["/", *"kilo", "escape", "s"])
            assert await _wait(pilot, lambda: screen._sort == "updated")
            for _ in range(30):
                await pilot.pause(0.01)
            assert box.value == "" and screen._filter == "" and len(_shown(screen)) == 3
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_slash_typed_inside_the_search_box_is_part_of_the_query():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            box = screen.query_one("#jobfilter", Input)
            _burst(app, ["/", *"a/b"])
            assert await _wait(pilot, lambda: box.value == "a/b")
            assert await _wait(pilot, lambda: _shown(screen) == {"bbbb2222"})
            assert screen.focused is box
            await _settle(app, pilot)

    asyncio.run(run())


def _logs_client():
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = []
    c.get_job.return_value = {"job_id": "7", "status": "RUNNING",
                              "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}]}
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    return c


def test_keys_typed_right_after_slash_reach_the_log_filter_in_order():
    from cortex_training.tui.app import LogScreen

    async def run():
        app = CortexTrainingLogTUI(_logs_client(), "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen.query("#logfilter"))
            for _ in range(10):
                await pilot.pause()
            screen = app.screen
            box = screen.query_one("#logfilter", Input)
            assert screen.focused is not box
            _burst(app, ["/", *"grad"])
            assert await _wait(pilot, lambda: box.value == "grad")
            _burst(app, list("/x"))  # a / in the filter box is part of the filter
            assert await _wait(pilot, lambda: box.value == "grad/x")
            await _settle(app, pilot)

    asyncio.run(run())
