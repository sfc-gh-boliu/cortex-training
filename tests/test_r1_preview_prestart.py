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

"""Jobs that failed before any sub-job started (placement_timeout and the like):
the preview doesn't send the user to Logs, and Logs says why there are none."""

import asyncio
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui import views as v  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402

NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _job(status, *, reason=None, sub_status=None, ended=NOW - timedelta(hours=2)):
    job = {"job_id": "7", "status": status, "created_at": _iso(ended - timedelta(hours=1)),
           "updated_at": _iso(ended), "state_changed_at": _iso(ended),
           "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "TRAINING", "status": sub_status or status,
                         "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}}]}
    if reason:
        job["reason"] = reason
    return job


def _http_error(status, message):
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/operation"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps({"message": message, "code": "510200"}).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Client Error", response=resp)


def _client(job):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    reason = job.get("reason") or "unknown"
    c.tail_logs.side_effect = _http_error(409, f"Cortex training operation failed: session 7 has failed: {reason}")
    c.download_stdout_logs.return_value = []  # no saved console
    c.fetch_execution_logs.return_value = []  # and no execution log
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


@pytest.mark.parametrize("reason", ["placement_timeout", "placement_too_large", "placement_failed",
                                    "hardware_unavailable", "pending_timeout", "PLACEMENT_TIMEOUT"])
def test_a_failure_before_placement_never_started(reason):
    assert v.never_started(_job("FAILED", reason=reason))
    text = v.never_started_text(_job("FAILED", reason=reason))
    assert text == f"the job never got GPUs (reason: {reason}), so no sub-job started or wrote logs"


@pytest.mark.parametrize("job", [
    _job("FAILED", reason="sub_job_failed"),
    _job("FAILED", reason="initializing_timeout"),  # placed: it can have logs
    _job("FAILED", reason="node_evacuated"),
    _job("FAILED"),
    _job("CANCELLED", reason="cancelled_by_client"),
    _job("RUNNING"),
    _job("PENDING", reason="quota_filled"),  # still waiting: not ended
    _job("UNKNOWN", reason="Job status unavailable: Control Plane is unavailable"),
    {"job_id": "7", "status": "FAILED", "reason": "sub_job_failed", "sub_jobs": []},
])
def test_jobs_that_started_or_may_still_start_are_not_never_started(job):
    assert not v.never_started(job)


def test_an_ended_job_whose_sub_jobs_never_left_placement_never_started():
    cancelled = _job("CANCELLED", reason="cancelled_by_client", sub_status="PENDING")
    assert v.never_started(cancelled)
    assert v.never_started_text(cancelled) == ("the job was cancelled and never got GPUs (reason: "
                                               "cancelled_by_client), so no sub-job started or wrote logs")
    placing = _job("FAILED", sub_status="PLACING")
    assert v.never_started(placing)
    assert v.never_started_text(placing) == "the job never got GPUs, so no sub-job started or wrote logs"
    # One sub-job placed is enough to have logs.
    mixed = _job("FAILED", reason="sub_job_failed")
    mixed["sub_jobs"].append({"sub_job_id": "7:sampling:0", "job_type": "SAMPLING", "status": "PLACING"})
    mixed["sub_jobs"][0]["status"] = "FAILED"
    assert not v.never_started(mixed)


def _health(screen) -> str:
    return " ".join(str(screen.query_one("#health", Static).render()).split())


@pytest.mark.parametrize("reason, logs_hint", [
    ("placement_timeout", False), ("hardware_unavailable", False), ("sub_job_failed", True)])
def test_the_preview_points_to_logs_only_for_a_job_that_started(reason, logs_hint):
    job = _job("FAILED", reason=reason)
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            app.push_screen(JobSummaryScreen(c, "7", job=job))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            await pilot.pause()
            health = _health(app.screen)
            assert f"✖ Job failed reason: {reason}" in health
            assert ("check Logs (l)" in health) == logs_hint
            assert ("it never got GPUs, so no sub-job started or wrote logs" in health) == (not logs_hint)
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("reason, ended", [
    ("placement_timeout", NOW - timedelta(hours=2)),  # long ago
    ("placement_timeout", NOW),  # moments ago: no "saved logs may not all be uploaded yet" either
    ("pending_timeout", NOW),
])
def test_logs_of_a_job_that_never_started_name_the_reason(reason, ended):
    job = _job("FAILED", reason=reason, ended=ended)
    c = _client(job)
    note = f"No log output: the job never got GPUs (reason: {reason}), so no sub-job started or wrote logs."

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: note in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            # Only that line: no "saved logs may not all be uploaded yet" for a job that never had any,
            # and the generic line is gone.
            assert s._shown_lines == [note]
            assert LogScreen._NO_LOG_OUTPUT not in s._shown_lines
            await _settle(app, pilot)

    asyncio.run(run())


def test_logs_of_a_started_job_without_output_keep_the_generic_line():
    job = _job("FAILED", reason="sub_job_failed")
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: LogScreen._NO_LOG_OUTPUT in getattr(app.screen, "_shown_lines", []))
            assert not any("never got GPUs" in line for line in app.screen._shown_lines)
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_tail_seeing_a_placement_failure_names_it_in_logs():
    # The tail sees the end (409) before the status check refreshes the job's copy.
    state = {"job": _job("PLACING")}

    def tail(job_id, **kwargs):
        if state["job"]["status"] == "PLACING":
            return {"entries": [], "next_cursor": "", "eof": False}
        raise _http_error(409, "Cortex training operation failed: session 7 has failed: placement_timeout")

    c = _client(state["job"])
    c.get_job.side_effect = lambda _id: state["job"]
    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "starting")
            s = app.screen
            state["job"] = _job("FAILED", reason="placement_timeout")
            assert await _wait(pilot, lambda: any(line.startswith("No log output: the job never got GPUs")
                                                  for line in s._shown_lines))
            await _settle(app, pilot)

    asyncio.run(run())
