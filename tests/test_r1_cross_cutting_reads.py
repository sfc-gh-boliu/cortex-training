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

"""Every read the TUI makes answers within seconds: its client retries twice, with a 45 s read timeout.

The SDK's defaults (ten retries, a 600 s read timeout) left a view silent for
about a minute on a 404, and a stalled request for up to ten minutes.
"""

import argparse
import threading
from unittest.mock import MagicMock

import pytest
import requests

import cortex_training._cli as cli
from cortex_training import CortexTrainingClient
from cortex_training.tui import __main__ as tui_main
from tests.r1_cross_cutting_support import FakeClient
from tests.r1_cross_cutting_support import NOW
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401
from tests.r1_cross_cutting_support import job
from tests.r1_cross_cutting_support import run
from tests.r1_cross_cutting_support import settle
from tests.r1_cross_cutting_support import wait_for


def _args(**more):
    base = dict(database="CORTEX_TRAINING_DB", schema="PUBLIC", endpoint="cortex-training", poll_interval=1.0,
                poll_timeout=1800.0, base_url="http://127.0.0.1:9", use_connection_profile=False)
    return argparse.Namespace(**{**base, **more})


def test_build_client_takes_the_tuis_bounds_and_keeps_the_defaults_otherwise():
    bounded = cli.build_client(_args(max_retries=2, request_timeout=(10.0, 45.0)), CortexTrainingClient)
    assert bounded.max_retries == 2 and bounded.request_timeout == (10.0, 45.0)
    assert bounded._session.default_timeout == (10.0, 45.0)  # every request, not only list pages
    child = bounded.for_job({"database_name": "TEAM_DB", "schema_name": "EXP"})
    assert child.max_retries == 2 and child._session is bounded._session
    plain = cli.build_client(_args(), CortexTrainingClient)
    assert plain.max_retries == 10 and plain.request_timeout == (30.0, 600.0)


def test_the_tui_builds_its_client_with_two_retries_and_a_45_second_read_timeout(monkeypatch):
    seen = {}
    real = cli.build_client

    def build_client(args, client_cls):
        seen["max_retries"], seen["request_timeout"] = args.max_retries, args.request_timeout
        return real(args, client_cls)

    class App:
        def __init__(self, client, *args, **kwargs):
            seen["client"] = client
            self.return_code = 0

        def run(self):
            pass

    monkeypatch.setattr(cli, "build_client", build_client)
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", App)
    assert tui_main.run(["--base-url", "http://127.0.0.1:9", "--database", "DB", "--schema", "S"]) == 0
    assert (seen["max_retries"], seen["request_timeout"]) == (tui_main.READ_MAX_RETRIES, tui_main.READ_TIMEOUT)
    assert (tui_main.READ_MAX_RETRIES, tui_main.READ_TIMEOUT) == (2, (10.0, 45.0))
    assert seen["client"].max_retries == 2 and seen["client"]._session.default_timeout == (10.0, 45.0)


def test_a_job_that_isnt_found_is_asked_three_times_not_eleven(monkeypatch):
    client = CortexTrainingClient(base_url="http://127.0.0.1:9", database="DB", schema="S",
                                  max_retries=tui_main.READ_MAX_RETRIES, request_timeout=tui_main.READ_TIMEOUT)
    response = requests.Response()
    response.status_code = 404
    response._content = b'{"message": "Training job nope not found"}'
    response.url = "http://127.0.0.1:9/x"
    client._session = MagicMock()
    client._session.get.return_value = response
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # tenacity's waits
    with pytest.raises(requests.HTTPError):
        client.get_job("nope")
    assert client._session.get.call_count == 3


def test_r_during_a_refresh_in_flight_says_it_is_still_waiting():
    pytest.importorskip("textual")
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen

    release = threading.Event()

    class SlowList(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.slow = False
            self.asked = 0

        def list_jobs(self, status=None, *, max_retries=None, request_timeout=None):
            self.asked += 1
            if self.slow:
                release.wait(10)
            return super().list_jobs(status, max_retries=max_retries, request_timeout=request_timeout)

    client = SlowList([job("j1")])

    async def go():
        app = CortexTrainingLogTUI(client, poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._loaded)
            notes = []
            app.notify = lambda message, **kwargs: notes.append(message)
            client.slow = True
            await pilot.press("r")
            assert await wait_for(pilot, lambda: app.screen._refreshing)
            await pilot.press("r")
            assert await wait_for(pilot, lambda: notes)
            assert notes[0].startswith("Still waiting for the last refresh (")
            assert client.asked == 2  # the first load and one refresh, not a third
            release.set()
            assert await wait_for(pilot, lambda: not app.screen._refreshing)
            await settle(app, pilot)

    run(go())


# ─── a service that stalls on every try ──────────────────────────────────────


class _StalledJob(FakeClient):
    """get_job blocks until released, then times out as a read does, a few times at most.

    It carries the TUI client's bounds (``max_retries``, ``request_timeout``), which
    the Logs tab quotes while it waits.
    """

    max_retries = 2
    request_timeout = (10.0, 45.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = threading.Event()
        self.reads = 0
        self.stalls = 1

    def get_job(self, job_id):
        self.reads += 1
        if self.reads <= self.stalls:
            self.release.wait(20)
            raise requests.exceptions.ReadTimeout(
                "HTTPConnectionPool(host='127.0.0.1', port=9): Read timed out. (read timeout=45.0)")
        return super().get_job(job_id)


def _status_line(screen) -> str:
    from textual.widgets import Static

    bar = screen.query_one("#log-status", Static)
    return str(bar.render()) if bar.display else ""


def test_a_stalled_job_read_is_shown_while_it_waits_and_named_when_it_fails(monkeypatch):
    """Opened by id, the Logs tab has nothing to show until the job is read. While a
    stalled service keeps it waiting, the pane says so at once and the status line
    counts the wait; when the read gives up, the pane names the timeout, not "no
    sub-jobs"; r then reads it again, and the job's logs show."""
    pytest.importorskip("textual")
    from textual.widgets import Log

    from cortex_training.tui import app as tui_app
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.screens import JobDetailScreen

    monkeypatch.setattr(tui_app.LogScreen, "_JOB_READ_QUIET_SECONDS", 0.0, raising=False)  # count from 0 s
    client = _StalledJob([job("j1")])

    def pane(screen) -> str:
        return "\n".join(str(line) for line in screen.query_one("#logview", Log).lines)

    async def go():
        app = CortexTrainingLogTUI(client, "j1", poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            try:
                assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                      and "reading job j1" in pane(app.screen))
                detail = app.screen
                assert await wait_for(pilot, lambda: "reading the job: no answer yet after" in _status_line(detail))
                assert "(up to 3 tries of 45 s each)" in _status_line(detail)
                # r while the read waits: said, and no second read beside it.
                notes = []
                app.notify = lambda message, **kwargs: notes.append(message)
                await pilot.press("r")
                assert await wait_for(pilot, lambda: notes)
                assert notes[0].startswith("Still reading this job (") and client.reads == 1
            finally:
                client.release.set()
            assert await wait_for(pilot, lambda: "[error]" in pane(detail))
            text = pane(detail)
            assert "couldn't read job j1: no answer from the service within 45 s (ReadTimeout) (3 tries)" in text
            assert "no sub-jobs" not in text and "127.0.0.1" not in text
            assert await wait_for(pilot, lambda: "reading the job" not in _status_line(detail))
            await pilot.press("3")  # Resources has no sub-job to show either, and says so
            resources = lambda: str(detail.query_one("#resource-status").render())  # noqa: E731
            assert await wait_for(pilot, lambda: "no sub-jobs to show GPU metrics for" in resources())
            await pilot.press("2")
            await pilot.press("r")  # the service answers now
            assert await wait_for(pilot, lambda: detail._current_source == "j1:training:0")
            await settle(app, pilot)

    run(go())


def test_a_read_that_answers_at_once_shows_no_wait():
    pytest.importorskip("textual")
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.screens import JobDetailScreen

    client = _StalledJob([job("j1")])
    client.stalls = 0

    async def go():
        app = CortexTrainingLogTUI(client, "j1", poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                  and app.screen._current_source == "j1:training:0")
            detail = app.screen
            assert "reading" not in _status_line(detail)
            assert not any("reading job" in line for line in detail._shown_lines)  # the tail replaced it
            await settle(app, pilot)

    run(go())


def test_timeouts_say_how_long_they_waited():
    from cortex_training.tui import views

    read = requests.exceptions.ReadTimeout(
        "HTTPSConnectionPool(host='ACCOUNT.snowflakecomputing.com', port=443): Read timed out. (read timeout=45.0)")
    connect = requests.exceptions.ConnectTimeout(
        "HTTPSConnectionPool(host='ACCOUNT.snowflakecomputing.com', port=443): Max retries exceeded "
        "(Caused by ConnectTimeoutError(None, 'Connection timed out. (connect timeout=10.0)'))")
    assert views.describe_error(read) == "no answer from the service within 45 s (ReadTimeout)"
    assert views.describe_error(connect) == "couldn't connect to the service within 10 s (ConnectTimeout)"
    assert views.describe_error(requests.exceptions.Timeout()) == "no answer from the service in time (Timeout)"
