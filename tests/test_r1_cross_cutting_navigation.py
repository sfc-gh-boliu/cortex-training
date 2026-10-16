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

"""Key bursts never close the wrong screen, stack a second copy, or crash the app.

A terminal delivers keys typed (or pasted) quickly back to back, so each test
sends its keys without waiting between them.
"""

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.screens import PromptScreen  # noqa: E402
from tests.r1_cross_cutting_support import FakeClient  # noqa: E402
from tests.r1_cross_cutting_support import NOW  # noqa: E402
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: E402,F401
from tests.r1_cross_cutting_support import job  # noqa: E402
from tests.r1_cross_cutting_support import run  # noqa: E402
from tests.r1_cross_cutting_support import send_keys  # noqa: E402
from tests.r1_cross_cutting_support import settle  # noqa: E402
from tests.r1_cross_cutting_support import wait_for  # noqa: E402


def _client():
    return FakeClient([job("j1"), job("j2", "FAILED", minutes_ago=90)])


async def _list(app, pilot) -> JobListScreen:
    assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    screen = app.screen
    table = screen.query_one("#jobs", DataTable)
    screen.set_focus(table)
    table.move_cursor(row=1, animate=False)  # the first job row (row 0 is the LIVE header)
    await pilot.pause()
    return screen


async def _spin(seconds: float) -> None:
    """Let the app run for ``seconds`` without waiting for its queues to empty."""
    import asyncio

    await asyncio.sleep(seconds)


async def _until(predicate, seconds: float = 5.0) -> bool:
    import asyncio
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


def _stack(app) -> list:
    return [type(s).__name__ for s in app.screen_stack[1:]]  # without Textual's default screen


def test_two_quick_enters_in_the_rename_prompt_close_only_the_prompt():
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _list(app, pilot)
            await pilot.press("n")
            assert await wait_for(pilot, lambda: isinstance(app.screen, PromptScreen) and app.screen.is_mounted)
            send_keys(app, "enter", "enter")
            for _ in range(20):
                await pilot.pause()
            assert app.screen is screen
            assert _stack(app) == ["JobListScreen"]
            await settle(app, pilot)

    run(go())


def test_two_quick_enters_in_the_previews_rename_prompt_keep_the_preview():
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            await _list(app, pilot)
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.is_mounted)
            preview = app.screen
            await pilot.press("n")
            assert await wait_for(pilot, lambda: isinstance(app.screen, PromptScreen) and app.screen.is_mounted)
            send_keys(app, "enter", "enter")
            for _ in range(20):
                await pilot.pause()
            assert app.screen is preview
            assert _stack(app) == ["JobListScreen", "JobSummaryScreen"]
            await settle(app, pilot)

    run(go())


def test_a_double_enter_on_the_list_opens_one_preview_and_one_back_returns():
    client = _client()

    async def go():
        app = CortexTrainingLogTUI(client, poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _list(app, pilot)
            send_keys(app, "enter", "enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.is_mounted)
            for _ in range(20):
                await pilot.pause()
            assert _stack(app) == ["JobListScreen", "JobSummaryScreen"]
            assert client.calls.count(("get_job", "j1")) == 1
            await pilot.press("b")
            assert await wait_for(pilot, lambda: app.screen is screen)
            await settle(app, pilot)

    run(go())


def test_a_double_enter_on_the_previews_sub_jobs_opens_one_details():
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            await _list(app, pilot)
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.is_mounted)
            preview = app.screen
            send_keys(app, "enter", "enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen.is_mounted)
            for _ in range(20):
                await pilot.pause()
            assert _stack(app) == ["JobListScreen", "JobSummaryScreen", "JobDetailScreen"]
            await pilot.press("b")
            assert await wait_for(pilot, lambda: app.screen is preview)
            await settle(app, pilot)

    run(go())


@pytest.mark.parametrize("burst", [("enter", "l", "3", "b", "b"), ("enter", "g", "b", "b"),
                                   ("enter", "l", "1", "2", "3", "4", "t", "b", "b")])
def test_open_and_back_keys_in_a_burst_never_crash(burst):
    """Back keys that arrive while a screen is still being put up, or right after
    a tab key, used to remove the screen under its own pending redraws.

    A smoke check: whether a burst hits that order depends on timing, so this
    passes without the fix too more often than not. The test below makes the
    order certain.
    """

    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _list(app, pilot)
            for _ in range(8):
                send_keys(app, *burst)
                for _ in range(3):
                    await pilot.pause()
            for _ in range(40):
                await pilot.pause()
            assert app.is_running
            # Whatever the keys reached, every screen left on the stack is whole.
            for shown in app.screen_stack[1:]:
                assert shown.is_mounted
            # Back to the list from wherever the burst ended.
            for _ in range(4):
                if app.screen is screen:
                    break
                await pilot.press("b")
                await wait_for(pilot, lambda: app.screen.is_mounted)
            assert app.screen is screen
            await settle(app, pilot)

    run(go())


def test_a_tab_switch_just_before_back_never_redraws_the_removed_screen():
    """3 then b, in the order a busy terminal delivered them: the details handle the tab
    switch after the pop, and the redraw it schedules runs while their widgets are being
    removed. Before the fix that stopped the app (NoMatches '#resource-status').

    The order is made certain, not left to chance: gates hold the details' queue (so the
    tab switch waits behind the pop), their footer's (so their removal stays half done)
    and the preview's (so the redraw runs only once the Resources widgets are gone).
    """
    import asyncio

    from textual.widgets import Footer

    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _list(app, pilot)
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.is_mounted)
            preview = app.screen
            await pilot.press("l")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._sources_ready)
            detail = app.screen
            for _ in range(5):
                await pilot.pause()
            details_queue, removal, preview_queue = asyncio.Event(), asyncio.Event(), asyncio.Event()
            try:
                # (While a queue is held, pilot.pause() would wait for it: sleep instead.)
                detail.call_later(details_queue.wait)  # the tab switch waits in the details' queue...
                await _spin(0.1)
                detail.query_one("#job-tabs", TabbedContent).active = "resource"
                await _spin(0.2)
                app.pop_screen()  # ... behind the pop
                detail.query_one(Footer).call_later(removal.wait)  # their removal stops half way
                preview.call_later(preview_queue.wait)  # and the redraw waits for that
                await _spin(0.1)
                details_queue.set()
                assert await _until(lambda: not detail.query("#resource-status"))
                preview_queue.set()  # the redraw comes due now, with the widgets gone
                await _spin(0.3)
            finally:
                details_queue.set()
                removal.set()
                preview_queue.set()
            for _ in range(20):
                await pilot.pause()
            assert app.is_running and app.screen is preview
            await pilot.press("b")
            assert await wait_for(pilot, lambda: app.screen is screen)
            await settle(app, pilot)

    run(go())


def test_q_quits_at_once_but_types_in_a_text_box():
    from textual.widgets import Input

    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _list(app, pilot)
            await pilot.press("slash", "q", "w")
            search = screen.query_one("#jobfilter", Input)
            assert await wait_for(pilot, lambda: search.value == "qw")  # typed, not quit
            assert app.is_running
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: not search.value)
            await pilot.press("n")  # the name prompt
            assert await wait_for(pilot, lambda: isinstance(app.screen, PromptScreen) and app.screen.is_mounted)
            await pilot.press("q")
            assert app.screen.query_one("#prompt-input", Input).value.endswith("q") and app.is_running
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: app.screen is screen)
            await pilot.press("q")
            assert await wait_for(pilot, lambda: not app.is_running, tries=50)

    run(go())


def test_q_quits_while_the_screen_is_still_busy_with_earlier_keys():
    """A burst of keys keeps the screens busy: each Enter puts a screen up, each b takes
    one down, and a key the app leaves to them waits behind that work (about 10 s after
    30 quick "Enter g b b" on a loaded machine). q must not wait: here the list screen is
    held busy, as by a long backlog, and q still quits at once."""
    import asyncio

    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _list(app, pilot)
            busy = asyncio.Event()
            try:
                screen.call_later(busy.wait)  # the screen is still working through earlier keys
                await _spin(0.1)  # (pilot.pause() would wait for the held queue)
                send_keys(app, "q")
                assert await _until(lambda: not app.is_running, seconds=3)
            finally:
                busy.set()

    run(go())


def test_q_after_a_burst_of_open_and_back_keys_quits():
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(140, 42)) as pilot:
            await _list(app, pilot)
            send_keys(app, *(["enter", "g", "b", "b"] * 30), "q")
            assert await wait_for(pilot, lambda: not app.is_running, tries=300)

    run(go())
