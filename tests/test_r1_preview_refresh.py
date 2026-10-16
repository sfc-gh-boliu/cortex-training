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

"""The job preview says when it is reading the job, when a read failed (and
which copy it still shows), and when the copy on screen was read."""

import asyncio
import json
import re
import threading
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402

JOB = "fake-job-refresh"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(status):
    return {"job_id": JOB, "status": status, "comment": "warm-up probe", "owner_role": "ML",
            "sub_jobs": [{"sub_job_id": f"{JOB}:training:0", "job_type": "TRAINING", "status": status,
                          "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}}]}


def _http_error(status, message):
    resp = requests.Response()
    resp.status_code = status
    resp.reason = "Internal Server Error"
    resp.url = "http://test.local/job"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps({"message": message, "code": "500000"}).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Server Error", response=resp)


class _Service:
    """get_job answers from a queue; each answer waits for its gate when one is set."""

    def __init__(self, listed):
        self.answers = []
        self.gate = None
        self.calls = 0
        self.client = c = MagicMock()
        c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
        c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
        c.list_jobs.return_value = [listed]
        c.get_job.side_effect = self._get

    def _get(self, job_id):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


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


def _fetch(screen) -> str:
    line = screen.query_one("#fetch", Static)
    return " ".join(str(line.render()).split()) if line.display else ""


def _status(screen) -> str:
    rows = str(screen.query_one("#preview", Static).render()).split("\n")
    return next(row for row in rows if row.strip().startswith("Status")).split()[1]


async def _open(app, pilot, service, listed):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    app.push_screen(JobSummaryScreen(service.client, JOB, job=listed))
    assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
    return app.screen


def test_a_failed_refresh_says_so_and_which_copy_it_shows():
    listed = _job("INITIALIZING")
    service = _Service(listed)
    service.answers = [_job("INITIALIZING")]

    async def run():
        app = CortexTrainingLogTUI(service.client, poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            p = await _open(app, pilot, service, listed)
            assert await _wait(pilot, lambda: _fetch(p).startswith("read from the service at"))
            read_at = re.search(r"\d\d:\d\d:\d\d", _fetch(p)).group(0)

            # The service now says RUNNING, but every read fails: r shows the read in flight ...
            service.gate = threading.Event()
            service.answers = [_http_error(500, "internal error")]
            await pilot.press("r")
            assert await _wait(pilot, lambda: _fetch(p).startswith("refreshing…"))
            assert f"(last read at {read_at})" in _fetch(p)
            p._FETCH_SLOW_SECONDS = 0  # a slow read says how long it has taken
            assert await _wait(pilot, lambda: re.search(r"refreshing… \(\d+s\)", _fetch(p)) is not None, tries=200)
            # ... a second r doesn't stack another read on it ...
            await pilot.press("r")
            await pilot.pause()
            assert service.calls == 2
            # ... and the failure names the error and the copy still on screen.
            service.gate.set()
            assert await _wait(pilot, lambda: _fetch(p).startswith("couldn't refresh"))
            assert _fetch(p) == (f"couldn't refresh: HTTP 500 Internal Server Error: internal error; "
                                 f"showing the copy read at {read_at}")
            assert _status(p) == "INITIALIZING"

            # The next good read replaces the copy and the note.
            service.gate = None
            service.answers = [_job("RUNNING")]
            await pilot.press("r")
            assert await _wait(pilot, lambda: _fetch(p).startswith("read from the service at"))
            assert _status(p) == "RUNNING"
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("answer, error", [
    (_http_error(500, "internal error"), "HTTP 500 Internal Server Error: internal error"),
    (ConnectionError("connection reset\nby peer"), "ConnectionError: connection reset by peer"),
    ({}, "the service sent an empty answer"),
])
def test_a_first_read_that_fails_keeps_the_job_lists_copy_and_says_so(answer, error):
    listed = _job("PENDING")
    service = _Service(listed)
    service.answers = [answer]

    async def run():
        app = CortexTrainingLogTUI(service.client, poll_interval=0.01)
        async with app.run_test(size=(160, 42)) as pilot:
            p = await _open(app, pilot, service, listed)
            assert await _wait(pilot, lambda: _fetch(p).startswith("couldn't read the job"))
            assert _fetch(p) == f"couldn't read the job: {error}; showing the job list's copy"
            assert _status(p) == "PENDING"
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_first_read_in_flight_is_shown():
    listed = _job("RUNNING")
    service = _Service(listed)
    service.answers = [_job("RUNNING")]
    service.gate = threading.Event()

    async def run():
        app = CortexTrainingLogTUI(service.client, poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            p = await _open(app, pilot, service, listed)
            assert await _wait(pilot, lambda: _fetch(p) == "reading the job from the service…")
            service.gate.set()
            assert await _wait(pilot, lambda: _fetch(p).startswith("read from the service at"))
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_long_error_wraps_under_its_text():
    listed = _job("RUNNING")
    service = _Service(listed)
    service.answers = [_http_error(500, "the control plane did not answer in time; " * 6)]

    async def run():
        app = CortexTrainingLogTUI(service.client, poll_interval=0.01)
        async with app.run_test(size=(100, 42)) as pilot:
            p = await _open(app, pilot, service, listed)
            assert await _wait(pilot, lambda: _fetch(p).startswith("couldn't read the job"))
            rows = str(p.query_one("#fetch", Static).render()).split("\n")
            assert len(rows) > 1 and all(row.startswith("    ") for row in rows[1:]), rows
            await _settle(app, pilot)

    asyncio.run(run())
