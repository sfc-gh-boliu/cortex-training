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

"""The live tail, round 1: a tail that stops on an error says so and is tried
again by the status check, a request with no answer says so, a job whose
status is unavailable waits with a note, a cursor the service refuses
restarts the tail, and a log that runs ahead of the view is caught up."""

import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

pytest.importorskip("textual")

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402
from cortex_training.tui.log_cache import LogCache  # noqa: E402
from tests.test_r1_logs_tui import _client  # noqa: E402
from tests.test_r1_logs_tui import _http_error  # noqa: E402
from tests.test_r1_logs_tui import _isolated  # noqa: E402,F401 - the autouse isolation
from tests.test_r1_logs_tui import _job  # noqa: E402
from tests.test_r1_logs_tui import _run  # noqa: E402
from tests.test_r1_logs_tui import _settle  # noqa: E402
from tests.test_r1_logs_tui import _status_line  # noqa: E402
from tests.test_r1_logs_tui import _wait  # noqa: E402

EMPTY = {"entries": [], "next_cursor": "C1", "eof": False}


def test_a_tail_that_stops_on_an_error_says_so_and_the_status_check_tries_it_again():
    c = _client(_job("RUNNING"))
    answers = {"fail": True}

    def tail(*args, **kwargs):
        if answers["fail"]:
            raise _http_error(503, "upstream unavailable")
        return {"entries": [{"_raw": "ts back again"}], "next_cursor": "C2", "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: _status_line(s).startswith("tail stopped: HTTP 503"))
            assert "upstream unavailable" in _status_line(s) and "r retries" in _status_line(s)
            calls = c.tail_logs.call_count
            answers["fail"] = False  # the outage is over
            s._refresh_status()  # the 60 s check: the job and its sub-job run, so the tail is tried again
            assert await _wait(pilot, lambda: "ts back again" in s._shown_lines)
            assert c.tail_logs.call_count > calls
            assert "tail stopped" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


def test_an_unattended_view_does_not_retry_a_stopped_tail(monkeypatch):
    c = _client(_job("RUNNING"))
    c.tail_logs.side_effect = _http_error(503, "upstream unavailable")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: _status_line(s).startswith("tail stopped"))
            calls = c.tail_logs.call_count
            monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 0)  # nobody has touched it for that long
            s._refresh_status()
            for _ in range(30):
                await pilot.pause(0.01)
            assert c.tail_logs.call_count == calls
            await _settle(app, pilot)

    _run(run())


def test_a_request_with_no_answer_says_so(monkeypatch):
    monkeypatch.setattr(LogScreen, "_NO_ANSWER_SECONDS", 0.2)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.05)
    c = _client(_job("RUNNING"))
    release = threading.Event()
    calls = []

    def tail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            release.wait(timeout=10)  # this one hangs
        return {"entries": [{"_raw": f"ts line {len(calls)}"}], "next_cursor": f"C{len(calls)}", "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "waiting for the service: the last log request has had no answer"
                               in _status_line(s))
            release.set()
            assert await _wait(pilot, lambda: "ts line 2" in s._shown_lines)
            assert await _wait(pilot, lambda: "no answer" not in _status_line(s))
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        release.set()


def test_a_job_whose_status_is_unavailable_waits_with_a_note():
    c = _client(_job("UNKNOWN", reason="Job status unavailable: Control Plane is unavailable"))
    answers = {"lines": False}
    c.tail_logs.side_effect = lambda *a, **k: (
        {"entries": [{"_raw": "ts it answers"}], "next_cursor": "C1", "eof": False} if answers["lines"] else EMPTY)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: s._waiting == "unknown")
            assert _status_line(s).startswith("waiting: the job's status is unavailable right now")
            assert not any("tailing" in line for line in s._shown_lines)
            assert await _wait(pilot, lambda: c.tail_logs.call_count > 0)  # it is tried all the same
            answers["lines"] = True
            assert await _wait(pilot, lambda: "ts it answers" in s._shown_lines)
            assert s._waiting is None and "waiting" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


UNAVAILABLE = "Job status unavailable: Control Plane is unavailable"


def _unknown_job_whose_tail_fails():
    c = _client(_job("UNKNOWN", reason=UNAVAILABLE))
    answers = {"fail": True}

    def tail(*args, **kwargs):
        if answers["fail"]:
            raise _http_error(503, UNAVAILABLE)
        return {"entries": [{"_raw": "ts it answers"}], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail
    return c, answers


def test_a_stopped_tail_of_a_job_whose_status_is_unavailable_is_tried_again_by_the_status_check():
    c, answers = _unknown_job_whose_tail_fails()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(240, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: _status_line(s).startswith("tail stopped"))
            # One note, which says how it is tried again: not also "the tail keeps trying".
            assert _status_line(s) == (f"tail stopped: HTTP 503: {UNAVAILABLE} (r retries; so does each status "
                                       "check while the job's status is unavailable)")
            assert s._waiting is None
            for _ in range(3):  # each check, while the status is still unavailable, tries again
                calls = c.tail_logs.call_count
                s._refresh_status()
                assert await _wait(pilot, lambda: c.tail_logs.call_count > calls)
                assert await _wait(pilot, lambda: _status_line(s).startswith("tail stopped") and not s._tail_live)
            answers["fail"] = False  # it answers again
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts it answers" in s._shown_lines)
            assert "tail stopped" not in _status_line(s) and "waiting" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


def test_an_unattended_view_of_a_job_whose_status_is_unavailable_does_not_retry(monkeypatch):
    c, _ = _unknown_job_whose_tail_fails()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(240, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: _status_line(s).startswith("tail stopped"))
            calls = c.tail_logs.call_count
            monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 0)  # nobody has touched it for that long
            s._refresh_status()
            for _ in range(30):
                await pilot.pause(0.01)
            assert c.tail_logs.call_count == calls
            await _settle(app, pilot)

    _run(run())


def test_a_stopped_tail_is_tried_again_once_the_status_is_back():
    c, answers = _unknown_job_whose_tail_fails()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(240, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: _status_line(s).startswith("tail stopped"))
            answers["fail"] = False
            c.get_job.return_value = _job("RUNNING")
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts it answers" in s._shown_lines)
            assert _status_line(s) == ""  # the job runs and its tail polls: nothing to say
            await _settle(app, pilot)

    _run(run())


def test_a_cursor_the_service_refuses_restarts_the_tail_from_the_end():
    LogCache("7").append_entries("7:training:0", [{"_raw": "ts cached line"}])
    LogCache("7").set_cursor("7:training:0", "12000@stale")
    c = _client(_job("RUNNING"))
    sent = []

    def tail(job_id, cursor=None, **kwargs):
        sent.append(cursor)
        if cursor == "12000@stale":
            raise _http_error(400, "invalid cursor")
        if cursor is None:
            return {"entries": [{"_raw": "ts from the end"}], "next_cursor": "C9", "eof": False}
        return {"entries": [], "next_cursor": cursor, "eof": False}  # nothing new: the same cursor

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts from the end" in s._shown_lines)
            assert "— tail restarted from the end of the log; lines in between may be missing —" in s._shown_lines
            assert "no longer took its saved position" in _status_line(s)
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            assert sent.count("12000@stale") == 1
            assert LogCache("7").get_cursor("7:training:0") == "C9"
            await pilot.press("r")  # r never sends the refused cursor again
            assert await _wait(pilot, lambda: len(sent) > 4)
            assert sent.count("12000@stale") == 1
            await _settle(app, pilot)

    _run(run())


def test_a_log_that_runs_ahead_is_caught_up_and_the_status_line_says_how_far_behind():
    c = _client(_job("RUNNING"))
    old = datetime.now(timezone.utc) - timedelta(seconds=40)
    stamp = old.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    pages = iter([{"entries": [{"_raw": f"{stamp} line {i}"} for i in range(1500)], "next_cursor": "C1",
                   "eof": False}])
    gate = threading.Event()

    def tail(*args, **kwargs):
        page = next(pages, None)
        if page is not None:
            return page
        gate.wait(timeout=10)  # the next read: once the note has been seen
        return EMPTY

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "catching up: the log is ahead of this view" in _status_line(s))
            behind = int(_status_line(s).split("its newest line here is ", 1)[1].split(" s old")[0])
            assert 35 <= behind <= 120
            gate.set()
            assert await _wait(pilot, lambda: "catching up" not in _status_line(s))  # an empty read: caught up
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        gate.set()


# ─── R1-55 / R1-62: the sources panel and the keys that scroll the log ───────

def _many_lines_client():
    c = _client(_job("RUNNING", sampling=True))
    pages = iter([{"entries": [{"_raw": f"ts line {i}"} for i in range(300)], "next_cursor": "C1", "eof": False}])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, EMPTY)
    return c


@pytest.mark.parametrize("cols", [80, 100, 160])
def test_the_sources_panel_fits_its_labels_and_leaves_the_log_room(cols):
    c = _many_lines_client()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(cols, 30)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts line 299" in s._shown_lines)
            await pilot.pause(0.1)
            sources = s.query_one("#sources")
            assert sources.outer_size.width == 16  # "training #0" and "sampling #0", not 48
            assert s._content_width() >= cols - 16 - 6
            for _ in range(30):
                await pilot.press("]")
            assert s._sources_width == cols - s._LOG_MIN_WIDTH  # ] never squeezes the log away
            await pilot.pause(0.1)
            assert s._content_width() >= s._LOG_MIN_WIDTH - 4
            await _settle(app, pilot)

    _run(run())


def test_scroll_keys_move_the_log_while_the_sources_list_has_focus():
    c = _many_lines_client()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 30)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts line 299" in s._shown_lines)
            await pilot.pause(0.1)
            lv = s._logview
            assert s.focused is s.query_one("#sources") and lv.is_vertical_scroll_end
            await pilot.press("pageup")
            assert await _wait(pilot, lambda: not lv.is_vertical_scroll_end)
            assert s.focused is s.query_one("#sources")  # the focus stays; Up/Down still pick a sub-job
            await pilot.press("home")
            assert await _wait(pilot, lambda: lv.scroll_y == 0)
            await pilot.press("end")
            assert await _wait(pilot, lambda: lv.is_vertical_scroll_end)
            await pilot.press("down")
            assert await _wait(pilot, lambda: s.query_one("#sources").index == 1)
            await _settle(app, pilot)

    _run(run())
