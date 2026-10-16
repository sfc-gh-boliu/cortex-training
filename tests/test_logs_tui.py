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

"""The Logs tab against a mocked client: waiting, job end, restarts, failed
jobs without a console, Experiment access, scrollback, the no-input pause,
local filtering, and the poll interval's default."""

import asyncio
import json
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402
from cortex_training.tui.demo import DemoClient  # noqa: E402
from cortex_training.tui.log_cache import job_cache_dir  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402

ACCESS = ("this role can't read the job's Experiment (saved logs, GPU metrics and checkpoints live there); "
          "ask for access to the Experiment, or use the role that created the job")
# The experiment-run route's answer, which a missing Experiment gets too.
ACCESS_OR_GONE = ("this role can't read the job's Experiment, or it no longer exists (saved logs, GPU metrics "
                  "and checkpoints live there); ask for access to the Experiment, or use the role that created "
                  "the job")
# Snowflake refused the SQL on the Experiment: the service still lists the checkpoints.
ACCESS_SQL = ("this role can't read the job's Experiment, where saved logs and GPU metrics live (the checkpoints "
              "command works); ask for access to the Experiment, or use the role that created the job")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Keep the TUI's caches, error log and login state out of the real home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    for name in ("CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_BASE_URL",
                 "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT", "CORTEX_TRAINING_DATABASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(status, *, reason=None, details=None, sub_status=None, sampling=False):
    subs = [{"sub_job_id": "7:training:0", "job_type": "training", "status": sub_status or status}]
    if sampling:
        subs.append({"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": sub_status or status})
    job = {"job_id": "7", "status": status, "sub_jobs": subs}
    if reason:
        job["reason"] = reason
    if details:
        job["reason_details"] = details
    return job


def _client(job):
    c = MagicMock()
    # The account-wide list isn't deployed (what accounts get today), so the job
    # list shows list_jobs, the configured schema's jobs.
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "C1", "eof": False}
    c.fetch_execution_logs.return_value = []
    return c


def _http_error(status, message):
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/operation"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps({"message": message, "code": "510200"}).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Client Error", response=resp)


async def _wait(pilot, predicate, tries=300, delay=0.01):
    for _ in range(tries):
        await pilot.pause(delay)
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


def _summary(screen) -> str:
    return str(screen.query_one("#summary", Static).render())


def _status_line(screen) -> str:
    bar = screen.query_one("#log-status", Static)
    return str(bar.render()) if bar.display else ""


def _write_stdout(output_dir, sub_job_id, text):
    path = Path(output_dir) / sub_job_id / "stdout.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _error_log_count(text):
    path = Path.home() / ".cortex-training-errors.log"
    return path.read_text(encoding="utf-8").count(text) if path.exists() else 0


def _run(coro):
    asyncio.run(coro)


# ─── L1: waiting while the job (or sub-job) is starting ─────────────────────


def test_pending_job_waits_without_tailing_then_tails_once_it_runs():
    c = _client(_job("PENDING", reason="quota_filled", details="waiting for 8 GPUs of type H200"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "pending")
            s = app.screen
            note = ("waiting: the job is PENDING (quota_filled — waiting for 8 GPUs of type H200); "
                    "the log tail starts by itself when the job starts (status checked every 60 s · r checks now)")
            assert _status_line(s) == note
            # Said once: not in the pane as well, under a "tailing" banner.
            assert not any("waiting" in line or "tailing" in line for line in s._shown_lines)
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            for _ in range(20):
                await pilot.pause(0.01)
            assert c.tail_logs.call_count == 0  # tail-logs is refused before placement
            # The summary line carries the reason and its details; the subtitle keeps to the status.
            assert "quota_filled — waiting for 8 GPUs of type H200" in _summary(s)
            assert "reason" not in app.sub_title and app.sub_title.startswith("PENDING")
            # The 60 s status check sees it running: the tail starts by itself.
            c.get_job.return_value = _job("RUNNING")
            c.tail_logs.return_value = {"entries": [{"_raw": "ts first line"}], "next_cursor": "C1", "eof": False}
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts first line" in s._shown_lines)
            assert s._waiting is None and _status_line(s) == ""
            assert "quota_filled" not in _summary(s)  # the service stopped sending a reason
            await _settle(app, pilot)

    _run(run())


def test_the_waiting_note_follows_the_status_while_it_waits():
    c = _client(_job("PENDING", reason="quota_filled", details="waiting for 8 GPUs of type H200"))
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": False}

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "pending")
            s = app.screen
            c.get_job.return_value = _job("PENDING", reason="quota_filled", details="waiting for 4 GPUs of type H200")
            s._refresh_status()
            assert await _wait(pilot, lambda: "waiting for 4 GPUs" in _status_line(s))
            assert c.tail_logs.call_count == 0
            # Said once, on the status line, which follows the status: never repeated in the pane.
            assert not any("waiting for" in line for line in s._shown_lines)
            c.get_job.return_value = _job("PLACING")
            s._refresh_status()
            assert await _wait(pilot, lambda: "training #0 is PLACING" in _status_line(s))
            assert await _wait(pilot, lambda: c.tail_logs.call_count > 0)  # placing jobs are tailed
            c.get_job.return_value = _job("INITIALIZING")
            s._refresh_status()
            assert await _wait(pilot, lambda: "training #0 is INITIALIZING" in _status_line(s))
            assert not any("INITIALIZING" in line for line in s._shown_lines)
            assert "quota_filled" not in _summary(s)
            await _settle(app, pilot)

    _run(run())


def test_placing_job_tails_with_a_waiting_note_until_lines_arrive():
    c = _client(_job("PLACING"))
    pages = iter([{"entries": [], "next_cursor": "", "eof": False}] * 5
                 + [{"entries": [{"_raw": "ts up"}], "next_cursor": "C1", "eof": False}])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "starting")
            s = app.screen
            assert "waiting: training #0 is PLACING; its log appears here once it is up" in _status_line(s)
            assert await _wait(pilot, lambda: "ts up" in s._shown_lines, tries=600)
            assert s._waiting is None
            # Every fetch without a cursor (the empty one a placing job returns) tails recent lines.
            kwargs = [call.kwargs for call in c.tail_logs.call_args_list]
            assert all(k["max_lines"] == 2000 for k in kwargs if not k.get("cursor"))
            await _settle(app, pilot)

    _run(run())


def test_a_sibling_not_routed_yet_waits_instead_of_erroring():
    job = _job("INITIALIZING", sampling=True)
    job["sub_jobs"][0]["status"] = "RUNNING"
    c = _client(job)

    def tail(job_id, **kwargs):
        if kwargs["sub_job_id"] == "7:sampling:0" and c.get_job.return_value["status"] != "RUNNING":
            raise _http_error(400, "sub_job_id 7:sampling:0 does not belong to session 7")
        return {"entries": [{"_raw": f"ts {kwargs['sub_job_id']} line"}], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", sub_job_id="7:sampling:0", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "stopped")
            s = app.screen
            assert "waiting: sampling #0 hasn't started yet" in _status_line(s)
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            c.get_job.return_value = {**_job("RUNNING", sampling=True)}
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts 7:sampling:0 line" in s._shown_lines)
            assert s._waiting is None
            await _settle(app, pilot)

    _run(run())


def test_an_unrouted_sub_job_of_a_starting_job_is_tried_again_when_a_status_changes():
    # The service answers "does not belong to session" until it routes to the
    # sub-job, here once the job runs. Each status change tries again; while the
    # job still starts, a status check that changes nothing doesn't (once both
    # run, each check does: see the next test), and the traceback is logged once.
    def job(job_status, sampling_status):
        j = _job(job_status, sampling=True)
        j["sub_jobs"][0]["status"] = "RUNNING"
        j["sub_jobs"][1]["status"] = sampling_status
        return j

    c = _client(job("INITIALIZING", "INITIALIZING"))

    def tail(job_id, **kwargs):
        if kwargs["sub_job_id"] == "7:sampling:0" and c.get_job.return_value["status"] != "RUNNING":
            raise _http_error(400, "sub_job_id 7:sampling:0 does not belong to session 7")
        return {"entries": [{"_raw": f"ts {kwargs['sub_job_id']} line"}], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail

    def sampling_calls():
        return sum(call.kwargs["sub_job_id"] == "7:sampling:0" for call in c.tail_logs.call_args_list)

    async def run():
        app = CortexTrainingLogTUI(c, "7", sub_job_id="7:sampling:0", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "stopped")
            s = app.screen
            assert "waiting: sampling #0 hasn't started yet" in _status_line(s)
            # The sub-job runs, its job still starts: tried again, still not routed.
            c.get_job.return_value = job("INITIALIZING", "RUNNING")
            s._refresh_status()
            assert await _wait(pilot, lambda: sampling_calls() == 2 and s._waiting == "stopped")
            assert await _wait(pilot, lambda: "isn't routing sampling #0's log yet" in _status_line(s))
            assert "r tries now" in _status_line(s)
            # Nothing changed: not asked again.
            s._refresh_status()
            for _ in range(30):
                await pilot.pause(0.01)
            assert sampling_calls() == 2
            assert _error_log_count("tail 7:sampling:0 waiting (not routed yet)") == 1
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            # The job runs: tried again, and routed.
            c.get_job.return_value = job("RUNNING", "RUNNING")
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts 7:sampling:0 line" in s._shown_lines)
            assert s._waiting is None
            await _settle(app, pilot)

    _run(run())


def test_a_pending_sub_job_of_a_running_job_waits_for_its_placement():
    job = _job("RUNNING", sampling=True)
    job["sub_jobs"][1]["status"] = "PENDING"
    c = _client(job)

    async def run():
        app = CortexTrainingLogTUI(c, "7", sub_job_id="7:sampling:0", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "pending")
            s = app.screen
            assert _status_line(s) == ("waiting: sampling #0 is PENDING; its log tail starts by itself once it is "
                                       "placed (status checked every 60 s · r checks now)")
            assert c.tail_logs.call_count == 0
            c.get_job.return_value = _job("RUNNING", sampling=True)
            c.tail_logs.return_value = {"entries": [{"_raw": "ts sampling line"}], "next_cursor": "C1", "eof": False}
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts sampling line" in s._shown_lines)
            assert s._waiting is None and _status_line(s) == ""
            await _settle(app, pilot)

    _run(run())


def test_an_unrouted_sub_job_of_a_running_job_is_asked_again_each_status_check():
    # The service can say RUNNING before it routes to a sub-job. With no status
    # change left to try again on, each status check asks again, but only
    # while someone is at the view; the traceback is logged once.
    c = _client(_job("RUNNING", sampling=True))
    routed = {"ok": False}

    def tail(job_id, **kwargs):
        if kwargs["sub_job_id"] == "7:sampling:0" and not routed["ok"]:
            raise _http_error(400, "sub_job_id 7:sampling:0 does not belong to session 7")
        return {"entries": [{"_raw": f"ts {kwargs['sub_job_id']} line"}], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail

    def sampling_calls():
        return sum(call.kwargs["sub_job_id"] == "7:sampling:0" for call in c.tail_logs.call_args_list)

    async def run():
        app = CortexTrainingLogTUI(c, "7", sub_job_id="7:sampling:0", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "stopped")
            s = app.screen
            assert _status_line(s) == ("waiting: the service isn't routing sampling #0's log yet; the tail tries "
                                       "again each minute (r tries now)")
            s._refresh_status()  # nothing changed, and both run: asked again
            assert await _wait(pilot, lambda: sampling_calls() == 2 and s._waiting == "stopped")
            assert _error_log_count("tail 7:sampling:0 waiting (not routed yet)") == 1
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            # Nobody has touched the view for the idle time: a check doesn't ask.
            idle_since = time.monotonic() - LogScreen._IDLE_PAUSE_SECONDS - 1
            s._last_input = lambda: idle_since
            s._refresh_status()
            for _ in range(30):
                await pilot.pause(0.01)
            assert sampling_calls() == 2
            # Someone is back: the next check asks again, and now it is routed.
            del s._last_input
            routed["ok"] = True
            await pilot.press("down")
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts 7:sampling:0 line" in s._shown_lines)
            assert sampling_calls() >= 3
            assert s._waiting is None and _status_line(s) == ""
            await _settle(app, pilot)

    _run(run())


def test_a_pending_placement_answer_waits_whatever_the_status_says():
    # UNKNOWN: the service couldn't read the job's state, while tail-logs says
    # the job isn't placed. That is the wait for placement, not an error; the
    # tail starts once the status says the job runs.
    c = _client(_job("UNKNOWN"))

    def tail(job_id, **kwargs):
        if c.get_job.return_value["status"] != "RUNNING":
            raise _http_error(409, "Cortex training operation failed: session 7 is pending placement: quota_filled")
        return {"entries": [{"_raw": "ts first line"}], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "pending")
            s = app.screen
            assert _status_line(s) == ("waiting: the service says the job is pending placement (its status reads "
                                       "UNKNOWN); the log tail starts by itself when the job starts "
                                       "(status checked every 60 s · r checks now)")
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            calls = c.tail_logs.call_count
            s._refresh_status()  # still UNKNOWN: still waiting, and nothing is sent
            for _ in range(30):
                await pilot.pause(0.01)
            assert c.tail_logs.call_count == calls
            c.get_job.return_value = _job("PENDING", reason="quota_filled")
            s._refresh_status()
            assert await _wait(pilot, lambda: "the job is PENDING (quota_filled)" in _status_line(s))
            assert c.tail_logs.call_count == calls
            c.get_job.return_value = _job("RUNNING")
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts first line" in s._shown_lines)
            assert s._waiting is None and _status_line(s) == ""
            await _settle(app, pilot)

    _run(run())


def test_a_tail_that_failed_while_the_job_started_is_tried_again_by_the_status_check():
    # A placing job's tail fails, and the get_job that would say why fails too:
    # the error shows, and the next status check that can read the job starts
    # the tail again instead of only dropping the wait.
    job = {"now": _job("PLACING")}
    fail = {"tail": True, "get_job": False}

    def get_job(_id):
        if fail["get_job"]:
            raise requests.exceptions.ConnectionError("connection reset")
        return job["now"]

    def tail(job_id, **kwargs):
        if fail["tail"]:
            fail["get_job"] = True  # the next get_job, the one after this error, fails
            raise _http_error(503, "unavailable")
        return {"entries": [{"_raw": "ts up"}], "next_cursor": "C1", "eof": False}

    c = _client(job["now"])
    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any(line.startswith("[error] tail") for line in
                                                  getattr(app.screen, "_shown_lines", [])))
            s = app.screen
            # One note: the error replaces the wait's ("its log appears here once it is up").
            assert s._waiting is None and not s._tail_live
            assert _status_line(s) == "tail stopped: HTTP 503: unavailable (r retries; so does each status check)"
            fail.update(tail=False, get_job=False)
            job["now"] = _job("RUNNING")
            s._refresh_status()
            assert await _wait(pilot, lambda: "ts up" in s._shown_lines)
            assert s._waiting is None
            await _settle(app, pilot)

    _run(run())


def test_an_error_once_the_job_runs_ends_the_wait_and_the_status_check_tries_again():
    job = {"now": _job("PLACING")}

    def tail(job_id, **kwargs):
        job["now"] = _job("RUNNING")  # it came up, and its tail fails
        raise _http_error(500, "boom")

    c = _client(job["now"])
    c.get_job.side_effect = lambda _id: job["now"]
    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any(line.startswith("[error] tail") for line in
                                                  getattr(app.screen, "_shown_lines", [])))
            s = app.screen
            # Not "waiting: ... is PLACING": the tail stopped on an error, and the status line says so.
            assert s._waiting is None
            assert _status_line(s) == ("tail stopped: HTTP 500: boom (r retries; so does each status check while "
                                       "the job runs)")
            s._refresh_status()  # the job and its sub-job run, and someone is watching: try it again
            assert await _wait(pilot, lambda: c.tail_logs.call_count == 2)
            await _settle(app, pilot)

    _run(run())


def test_service_text_on_the_status_line_stays_one_safe_row():
    # reason_details is the service's text: a line break must not grow the bar,
    # and an escape sequence must not reach the terminal.
    details = "waiting for 8 GPUs\n\x1b]0;title\x07\x1b[31mred\tH200"
    c = _client(_job("PENDING", reason="quota_filled", details=details))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "pending")
            s = app.screen
            bar = s.query_one("#log-status", Static)
            assert await _wait(pilot, lambda: bar.size.height == 1)
            line = _status_line(s)
            # The record comes back printable (a tab as spaces), then the line breaks go.
            assert "waiting for 8 GPUs / \ufffd]0;title\ufffd\ufffd[31mred    H200" in line
            assert not any(ch in line for ch in "\x1b\x07\n\t")
            assert not any("\x1b" in shown or "\x07" in shown for shown in s._shown_lines)  # the pane's note too
            await _settle(app, pilot)

    _run(run())


def test_sub_job_statuses_read_the_same_ids_as_the_source_list():
    job = {"status": "RUNNING", "sub_jobs": [{"id": "7:training:0", "status": "RUNNING"},
                                             {"sub_job_id": "7:sampling:0", "status": "PLACING"}, "junk"]}
    assert LogScreen._sub_statuses(job) == {"7:training:0": "RUNNING", "7:sampling:0": "PLACING"}


# ─── L2: 409 / 404 once the job has ended ──────────────────────────────────


@pytest.mark.parametrize("status,message", [
    (409, "Cortex training operation failed: session 7 has failed: out_of_memory"),
    (404, "session 7 not found"),
])
def test_tail_error_after_the_job_ends_shows_the_saved_console_at_once(status, message):
    c = _client(_job("RUNNING"))
    jobs = iter([_job("RUNNING"), _job("FAILED", reason="out_of_memory")])
    c.get_job.side_effect = lambda _id: next(jobs, _job("FAILED", reason="out_of_memory"))
    c.tail_logs.side_effect = _http_error(status, message)
    c.download_stdout_logs.side_effect = lambda job_id, root: _write_stdout(root, "7:training:0", "saved line\n")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            started = time.monotonic()
            assert await _wait(pilot, lambda: "saved line" in getattr(app.screen, "_shown_lines", []))
            assert time.monotonic() - started < 10
            s = app.screen
            assert c.tail_logs.call_count == 1  # never retried
            assert c.get_job.call_count == 2  # the open, then once after the error
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            assert s._job_status == "FAILED" and "out_of_memory" in _summary(s)
            await _settle(app, pilot)

    _run(run())


def test_a_tail_error_on_a_running_job_stays_with_a_retry_hint():
    c = _client(_job("RUNNING"))
    c.tail_logs.side_effect = _http_error(409, "zone restarting")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any(line.startswith("[error] tail") for line in app.screen._shown_lines))
            line = next(line for line in app.screen._shown_lines if line.startswith("[error] tail"))
            assert "HTTP 409" in line and "zone restarting" in line and "r retries" in line
            for _ in range(10):
                await pilot.pause(0.01)
            assert c.tail_logs.call_count == 1
            c.download_stdout_logs.assert_not_called()
            await _settle(app, pilot)

    _run(run())


def test_a_console_fetched_moments_after_the_end_is_not_kept_and_r_adds_the_rest(monkeypatch):
    # The job ends while the view is open, so its console is fetched seconds
    # after the end, maybe before its last chunks are uploaded. It must not be
    # kept as complete (the next open would replay it for good); the pane says
    # so, and r fetches the chunks uploaded since.
    monkeypatch.setattr(LogScreen, "_KEEP_CONSOLE_AFTER_SECONDS", 2.0)
    now = datetime.now(timezone.utc)
    created = _iso(now - timedelta(hours=3))
    running = {**_job("RUNNING"), "created_at": created, "state_changed_at": _iso(now - timedelta(hours=2))}
    ended = {**_job("FAILED", reason="out_of_memory"), "created_at": created, "state_changed_at": _iso(now)}
    c = _client(running)
    jobs = iter([running, ended])
    c.get_job.side_effect = lambda _id: next(jobs, ended)
    c.tail_logs.side_effect = _http_error(409, "Cortex training operation failed: session 7 has failed: out_of_memory")

    def download(job_id, root, resume=False):
        _write_stdout(root, "7:training:0", "first line\n" + ("Traceback: CUDA out of memory\n" if resume else ""))

    c.download_stdout_logs.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "first line" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            complete = job_cache_dir("7") / "console" / ".complete"
            assert not complete.exists()
            assert s._state_changed_at == ended["state_changed_at"]  # the fresh get_job's, not RUNNING's
            assert s._shown_lines[-1] == s._recent_console_note()
            await pilot.pause(2.2)  # the job has now been over for longer than the keep window
            await pilot.press("r")
            assert await _wait(pilot, lambda: "Traceback: CUDA out of memory" in s._shown_lines)
            assert c.download_stdout_logs.call_args.kwargs == {"resume": True}
            assert s._recent_console_note() not in s._shown_lines
            assert complete.exists()
            await _settle(app, pilot)

    _run(run())


def test_a_job_that_fails_while_placing_leaves_no_waiting_note():
    # The tail sees the end (409) before the status check does: the saved
    # console's view must not keep saying the job is placing.
    job = {"now": _job("PLACING")}

    def tail(job_id, **kwargs):
        if job["now"]["status"] == "PLACING":
            return {"entries": [], "next_cursor": "", "eof": False}
        raise _http_error(409, "Cortex training operation failed: session 7 has failed: placement_timeout")

    c = _client(job["now"])
    c.get_job.side_effect = lambda _id: job["now"]
    c.tail_logs.side_effect = tail
    c.download_stdout_logs.return_value = []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "starting")
            s = app.screen
            job["now"] = _job("FAILED", reason="placement_timeout")
            # The saved view's line (for a job that never got GPUs, it says so).
            assert await _wait(pilot, lambda: any(line.startswith("No log output") for line in s._shown_lines))
            assert s._waiting is None and "is PLACING" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


def _visible_rows(lv):
    """The text of the rows the log pane shows now, without the blank ones under the last line."""
    rows = [lv.render_line(y).text.rstrip() for y in range(lv.size.height)]
    while rows and not rows[-1]:
        rows.pop()
    return rows


def _shows(lv, text):
    """Whether a row on screen shows ``text`` (lines carry a sub-job tag in front)."""
    return any(text in row for row in _visible_rows(lv))


def test_the_saved_console_opens_at_its_newest_lines_after_a_live_tail():
    # A live tail long enough to scroll, then the job ends: the console that
    # replaces it (cleared and refilled in one callback) opens at its end, not
    # at the live view's old offset, which Log.clear() keeps.
    c = _client(_job("RUNNING"))
    jobs = iter([_job("RUNNING"), _job("FAILED", reason="out_of_memory")])
    c.get_job.side_effect = lambda _id: next(jobs, _job("FAILED", reason="out_of_memory"))
    # The live lines are the console's last ones, as they are when the console is complete.
    pages = iter([{"entries": [{"_raw": f"console {i}"} for i in range(240, 300)], "next_cursor": "C1",
                   "eof": False}])

    def tail(*args, **kwargs):
        page = next(pages, None)
        if page is None:
            raise _http_error(409, "Cortex training operation failed: session 7 has failed: out_of_memory")
        return page

    c.tail_logs.side_effect = tail
    c.download_stdout_logs.side_effect = lambda job_id, root: _write_stdout(
        root, "7:training:0", "".join(f"console {i}\n" for i in range(300)))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.05)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_stdout_ready", False)
                               and "console 0" in app.screen._shown_lines and not app.screen._repainting)
            lv = app.screen._logview
            await pilot.pause(0.3)
            assert lv.max_scroll_y > 0 and lv.is_vertical_scroll_end
            assert _shows(lv, "console 299")
            await _settle(app, pilot)

    _run(run())


def test_r_on_a_console_fetched_moments_after_the_end_shows_the_new_chunks(monkeypatch):
    # The newest chunks, often the error, land below a console the view shows
    # at its end; r must leave the view on them.
    monkeypatch.setattr(LogScreen, "_KEEP_CONSOLE_AFTER_SECONDS", 60.0)
    now = datetime.now(timezone.utc)
    ended = {**_job("FAILED", reason="out_of_memory"), "created_at": _iso(now - timedelta(hours=3)),
             "state_changed_at": _iso(now)}
    c = _client(ended)
    first = "".join(f"console {i}\n" for i in range(100))

    def download(job_id, root, resume=False):
        _write_stdout(root, "7:training:0", first + ("Traceback: CUDA out of memory\n" if resume else ""))

    c.download_stdout_logs.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: "console 99" in getattr(app.screen, "_shown_lines", []))
            s, lv = app.screen, app.screen._logview
            await pilot.pause(0.3)
            assert lv.max_scroll_y > 0 and lv.is_vertical_scroll_end
            await pilot.press("r")
            assert await _wait(pilot, lambda: "Traceback: CUDA out of memory" in s._shown_lines)
            await pilot.pause(0.3)
            assert lv.is_vertical_scroll_end
            assert _shows(lv, "Traceback: CUDA out of memory")
            assert _visible_rows(lv)[-1].endswith("r downloads them —")  # the note under it, wrapped
            await _settle(app, pilot)

    _run(run())


def test_r_follows_the_tail_again_after_a_pause_and_a_scroll_back():
    c = _client(_job("RUNNING"))
    counter = iter(range(1_000_000))
    c.tail_logs.side_effect = lambda *a, **k: {
        "entries": [{"_raw": f"ts line {next(counter)}"} for _ in range(5)], "next_cursor": "C1", "eof": False}

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.05)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: len(getattr(app.screen, "_shown_lines", [])) > 100)
            s, lv = app.screen, app.screen._logview
            await pilot.press("p")  # stop following, then read back a little
            await pilot.pause(0.2)
            lv.scroll_to(y=lv.max_scroll_y - 30, animate=False)
            await pilot.pause(0.2)
            assert not lv.is_vertical_scroll_end
            gen = s._tail_gen
            await pilot.press("r")  # a fresh tail follows again
            assert await _wait(pilot, lambda: s._tail_gen > gen and len(s._shown_lines) > 50)
            newest = c.tail_logs.call_count
            assert await _wait(pilot, lambda: c.tail_logs.call_count >= newest + 3)
            await pilot.pause(0.2)
            assert not s._paused and lv.is_vertical_scroll_end
            assert _visible_rows(lv)[-1].endswith(s._shown_lines[-1])
            await _settle(app, pilot)

    _run(run())


def test_a_late_tail_error_stays_out_of_another_sub_jobs_pane():
    job = _job("RUNNING", sampling=True)
    c = _client(job)
    release = threading.Event()
    calls = {"n": 0}

    def get_job(_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return job
        release.wait(5)  # a slow get_job after training's tail failed
        raise _http_error(503, "unavailable")

    def tail(job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        if sub_job_id == "7:training:0":
            raise _http_error(500, "boom")
        return {"entries": [{"_raw": "ts sampling line"}], "next_cursor": "C1", "eof": False}

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.05)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: calls["n"] >= 2)
            s = app.screen
            s._start_tail("7:sampling:0")
            assert await _wait(pilot, lambda: "ts sampling line" in s._shown_lines)
            release.set()
            for _ in range(30):
                await pilot.pause(0.01)
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            # And the guard itself: an error line from an older tail is dropped.
            s._show_tail_error(s._tail_gen - 1, "[error] tail 7:training:0 failed")
            assert not any(line.startswith("[error]") for line in s._shown_lines)
            await _settle(app, pilot)

    _run(run())


# ─── L3: a stuck tail restarts from the end ────────────────────────────────


def test_a_stuck_tail_restarts_from_the_end_without_repeating_lines(monkeypatch):
    monkeypatch.setattr(log_cache, "STALL_SECONDS", 0.05)
    monkeypatch.setattr(log_cache, "STALL_POLLS", 2)
    c = _client(_job("RUNNING"))

    def raw(n):
        return {"_raw": f"2026-10-14T12:00:{n:02d}.000000001Z step {n}"}

    def tail(job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        if cursor is None and max_lines == 2000:
            return {"entries": [raw(1), raw(2), raw(3)], "next_cursor": "C3", "eof": False}
        if cursor is None:  # the restart: the log's last lines, two of them already shown
            return {"entries": [raw(2), raw(3), raw(8), raw(9)], "next_cursor": "C9", "eof": False}
        return {"entries": [], "next_cursor": cursor, "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: raw(9)["_raw"] in getattr(app.screen, "_shown_lines", []), tries=600)
            s = app.screen
            lines = s._shown_lines
            gap = "— tail restarted from the end of the log; lines in between may be missing —"
            assert lines.index(gap) == lines.index(raw(3)["_raw"]) + 1
            assert lines.index(raw(8)["_raw"]) == lines.index(gap) + 1
            assert lines.count(raw(2)["_raw"]) == 1 and lines.count(raw(3)["_raw"]) == 1
            assert "tail restarted" in _status_line(s)
            restart_calls = [k for k in (call.kwargs for call in c.tail_logs.call_args_list)
                             if k.get("cursor") is None and k.get("max_lines") != 2000]
            assert restart_calls and restart_calls[0]["max_lines"] == log_cache.RESTART_LINES[0]
            await _settle(app, pilot)

    _run(run())


def test_no_restart_while_the_job_is_initializing(monkeypatch):
    monkeypatch.setattr(log_cache, "STALL_SECONDS", 0.0)
    monkeypatch.setattr(log_cache, "STALL_POLLS", 1)
    c = _client(_job("INITIALIZING"))
    c.tail_logs.return_value = {"entries": [], "next_cursor": "C1", "eof": False}

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: c.tail_logs.call_count >= 5)
            assert all(call.kwargs.get("max_lines") in (None, 2000) for call in c.tail_logs.call_args_list)
            await _settle(app, pilot)

    _run(run())


def test_a_quiet_log_waking_just_before_a_restart_shows_every_line_without_a_gap(monkeypatch):
    monkeypatch.setattr(log_cache, "STALL_SECONDS", 0.05)
    monkeypatch.setattr(log_cache, "STALL_POLLS", 2)
    c = _client(_job("RUNNING"))
    log = [{"_raw": f"2026-10-14T12:00:00.{n:09d}Z early {n}"} for n in range(5)]
    burst = [f"2026-10-14T12:05:00.{n:09d}Z burst {n}" for n in range(500)]
    state = {"ends": 0}

    def tail(job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        if cursor is None and max_lines != 2000:  # a read of the log's end
            state["ends"] += 1
            if state["ends"] == 3:  # the burst lands just before this read
                log.extend({"_raw": line} for line in burst)
            return {"entries": log[-max_lines:], "next_cursor": f"C{len(log)}", "eof": False}
        start = int(cursor[1:]) if cursor else max(0, len(log) - 2000)
        page = log[start:start + 2000]
        return {"entries": page, "next_cursor": f"C{start + len(page)}", "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: state["ends"] >= 5, tries=1500)
            s = app.screen
            assert [line for line in s._shown_lines if " burst " in line] == burst  # all 500, in order
            assert not any("tail restarted" in line for line in s._shown_lines)
            assert "tail restarted" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


# ─── L4: a failed job without a saved console ──────────────────────────────

_EXEC_LOG = "".join(json.dumps(r) + "\n" for r in [
    {"ts": "2026-10-14T11:00:00+00:00", "level": "INFO", "logger": "runtime", "msg": "init phase: load_model"},
    {"ts": "2026-10-14T11:00:42+00:00", "level": "ERROR", "logger": "runtime", "msg": "failed to create job",
     "exc": "Traceback (most recent call last):\n  File \"loader.py\", line 3\nRuntimeError: CUDA out of memory"},
])


def test_failed_job_without_a_console_shows_its_execution_log_errors():
    c = _client(_job("FAILED", reason="out_of_memory"))
    c.fetch_execution_logs.return_value = [
        {"sub_job_id": "7:training:0", "filename": "execution.jsonl", "artifact_uri": "snow://x", "content": _EXEC_LOG}]

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "    RuntimeError: CUDA out of memory" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            head = next(line for line in s._shown_lines if "no saved console for training #0" in line)
            assert "the job failed before it wrote one" in head
            assert "cortex-training download-log 7 downloads the whole log" in head
            assert "2026-10-14T11:00:42+00:00 ERROR runtime: failed to create job" in s._shown_lines
            assert not any("init phase" in line for line in s._shown_lines)  # errors only
            c.fetch_execution_logs.assert_called_once_with("7", sub_job_id="7:training:0", filename="execution.jsonl")
            # The note is the TUI's own line (untagged); the records are tagged with the sub-job.
            rows = [str(line) for line in s._logview.lines]
            assert any(row.startswith("— no saved console") for row in rows)
            assert any(row.startswith("training:0 │ ") and "failed to create job" in row for row in rows)
            await _settle(app, pilot)

    _run(run())


def test_an_execution_log_without_errors_shows_its_last_records():
    c = _client(_job("FAILED", reason="sub_job_readiness"))
    content = "".join(json.dumps({"level": "INFO", "msg": f"phase {i}"}) + "\n" for i in range(30))
    c.fetch_execution_logs.return_value = [{"sub_job_id": "7:training:0", "content": content}]

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "INFO  phase 29" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            assert any("has no error records; its last 20 record(s) follow" in line for line in s._shown_lines)
            assert "INFO  phase 9" not in s._shown_lines and "INFO  phase 10" in s._shown_lines
            await _settle(app, pilot)

    _run(run())


def test_the_filter_keeps_or_hides_whole_execution_log_records_under_their_header():
    c = _client(_job("FAILED", reason="out_of_memory"))
    metrics = {"ts": "2026-10-14T11:00:50+00:00", "level": "ERROR", "logger": "job_metrics",
               "msg": "metrics push failed", "error": "HTTP 503"}
    c.fetch_execution_logs.return_value = [
        {"sub_job_id": "7:training:0", "content": _EXEC_LOG + json.dumps(metrics) + "\n"}]
    failed = "2026-10-14T11:00:42+00:00 ERROR runtime: failed to create job"
    trace = "    RuntimeError: CUDA out of memory"

    def header(s):
        return [line for line in s._shown_lines if "no saved console for training #0" in line]

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: trace in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            await pilot.press("/", *"push")
            assert await _wait(pilot, lambda: trace not in s._shown_lines)
            assert failed not in s._shown_lines
            assert "2026-10-14T11:00:50+00:00 ERROR job_metrics: metrics push failed" in s._shown_lines
            assert "    error: HTTP 503" in s._shown_lines and header(s)
            # A match inside an exception chain keeps its whole record.
            await pilot.press("escape", "/", *"cuda")
            assert await _wait(pilot, lambda: trace in s._shown_lines and failed in s._shown_lines
                               and not any("metrics push failed" in line for line in s._shown_lines))
            assert header(s) and s._filter == "cuda"
            await _settle(app, pilot)

    _run(run())


def test_a_job_that_ended_moments_ago_without_logs_promises_no_console_lines():
    # Failed as it started, say: no console, and no execution log yet. The note
    # under "No log output" must not promise a console's missing last lines;
    # r checks again, for the console and the execution log both. (A job that
    # never got GPUs, such as pending_timeout, gets neither note: it has no logs.)
    now = datetime.now(timezone.utc)
    ended = {**_job("FAILED", reason="sub_job_failed"), "created_at": _iso(now - timedelta(minutes=30)),
             "state_changed_at": _iso(now)}
    c = _client(ended)
    c.download_stdout_logs.return_value = []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: LogScreen._NO_LOG_OUTPUT in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            assert s._shown_lines[-2:] == [LogScreen._NO_LOG_OUTPUT, s._recent_console_note(console=False)]
            assert "may not all be uploaded yet; r checks again" in s._shown_lines[-1]
            assert s._recent_console_note() not in s._shown_lines  # "this console may still be missing ..."
            assert c.fetch_execution_logs.call_count == 1
            await pilot.press("r")
            assert await _wait(pilot, lambda: c.fetch_execution_logs.call_count == 2)
            assert c.download_stdout_logs.call_args.kwargs == {"resume": True}
            await _settle(app, pilot)

    _run(run())


# ─── L6: the role can't read the job's Experiment ──────────────────────────


@pytest.mark.parametrize("error,line", [
    (_http_error(500, "Linked experiment is not found or not authorized for the training job"), ACCESS),
    (Exception("002003 (02000): SQL compilation error:\n"
               "Experiment 'CORTEX_TRAINING_DB.PUBLIC.EXPERIMENT' does not exist or not authorized."), ACCESS_SQL),
    (Exception("003001 (42501): SQL access control error:\nInsufficient privileges to operate on experiment 'EXP'"),
     ACCESS_SQL),
    # The experiment-run route: the Experiment is hidden from this role, or gone.
    (_http_error(500, "Linked experiment is not found for job JOB_ID"), ACCESS_OR_GONE),
])
def test_experiment_access_problems_get_one_clear_line(error, line):
    # The Logs tab explains them as every tab does.
    assert views.access_hint(error, "saved logs") == line
    assert views.access_hint(RuntimeError("connection reset"), "saved logs") is None
    assert views.access_hint(_http_error(500, "internal error"), "saved logs") is None


def test_saved_console_and_execution_log_access_errors_are_explained():
    c = _client(_job("FAILED", reason="out_of_memory"))
    denied = Exception("002003 (02000): SQL compilation error:\nExperiment 'DB.PUBLIC.EXP' does not exist or not authorized.")
    c.download_stdout_logs.side_effect = denied

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: f"— {ACCESS_SQL} —" in getattr(app.screen, "_shown_lines", []))
            assert not any("couldn't be downloaded" in line for line in app.screen._shown_lines)
            await _settle(app, pilot)

        # The console downloads (there is none), but the execution log can't be read.
        c.download_stdout_logs.side_effect = None
        c.download_stdout_logs.return_value = []
        c.fetch_execution_logs.side_effect = _http_error(500, "Linked experiment is not found or not authorized "
                                                         "for the training job")
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: f"— {ACCESS} —" in getattr(app.screen, "_shown_lines", []))
            await _settle(app, pilot)

    _run(run())


# ─── L5: the scrollback cap is said ────────────────────────────────────────


def test_the_capped_live_buffer_says_last_n_lines(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 50)
    c = _client(_job("RUNNING"))
    page = {"entries": [{"_raw": f"ts line {i}"} for i in range(120)], "next_cursor": "C1", "eof": False}
    pages = iter([page])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "ts line 119" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            assert len(s._shown_lines) == 50 and s._shown_lines[-1] == "ts line 119"
            assert _status_line(s) == "last 50 lines · cortex-training download-log 7 --log-type stdout for the full log"
            assert len(s._logview.lines) == sum(s._shown_rows)  # the widget holds exactly those lines
            await _settle(app, pilot)

    _run(run())


def test_the_row_budget_counts_wrapped_lines(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_ROWS", 40)
    c = _client(_job("RUNNING"))
    long = "x" * 250  # wraps to two or more rows in the pane
    pages = iter([{"entries": [{"_raw": f"{long} {i}"} for i in range(60)], "next_cursor": "C1", "eof": False}])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: f"{long} 59" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            assert sum(s._shown_rows) <= 40 and len(s._logview.lines) == sum(s._shown_rows)
            assert _status_line(s).startswith(f"last {len(s._shown_lines)} lines · cortex-training download-log 7")
            await _settle(app, pilot)

    _run(run())


def test_a_long_saved_console_says_last_n_lines(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 50)
    c = _client(_job("FAILED"))
    c.download_stdout_logs.side_effect = lambda job_id, root: _write_stdout(
        root, "7:training:0", "".join(f"console {i}\n" for i in range(80)))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "console 79" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            assert "console 29" not in s._shown_lines and "console 30" in s._shown_lines
            assert _status_line(s).startswith("last 50 lines · cortex-training download-log 7 --log-type stdout")
            await _settle(app, pilot)

    _run(run())


def test_the_scrollback_label_counts_log_lines_and_says_when_a_filter_is_on(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 50)
    c = _client(_job("RUNNING"))
    page = {"entries": [{"_raw": f"ts line {i}" + (" match" if i in (100, 110) else "")} for i in range(120)],
            "next_cursor": "C1", "eof": False}
    pages = iter([page])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})
    full = "cortex-training download-log 7 --log-type stdout for the full log"

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "ts line 119" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            s._write_line("— a note of the TUI's own —")  # notes aren't log lines
            assert _status_line(s) == f"last 49 lines · {full}"
            # Each wait is for the whole filter's repaint: a loaded machine can
            # repaint after any key, and "11" also hides "ts line 100 match".
            await pilot.press("/", *"match")
            # The newest matching lines; no count of lines searched, which older matches may have left.
            assert await _wait(pilot, lambda: _status_line(s) == f"last 2 matching lines · {full}")
            assert "ts line 119" not in s._shown_lines
            await pilot.press("escape", "/", *"110")
            assert await _wait(pilot, lambda: _status_line(s) == f"last 1 matching line · {full}")
            assert "ts line 100 match" not in s._shown_lines and "ts line 110 match" in s._shown_lines
            await _settle(app, pilot)

    _run(run())


def test_a_full_pane_still_paints_the_newest_lines(monkeypatch):
    # Once the pane holds its cap, each write drops as many rows as it adds and
    # the view stays at the same offset; the newest lines must still be drawn.
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 50)
    c = _client(_job("RUNNING"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.05)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_current_source", None))
            s, lv = app.screen, app.screen._logview
            s._write_lines([f"line {i}" for i in range(10, 70)])
            await pilot.pause(0.2)
            assert lv.is_vertical_scroll_end and len(s._shown_lines) == 50
            painted = []
            render_line = lv.render_line

            def spy(y):
                strip = render_line(y)
                painted.append(strip.text)
                return strip

            lv.render_line = spy  # what the screen draws from now on
            # As wide as the lines before it: a wider one makes Log repaint everything anyway.
            s._write_lines(["line 99"])
            await pilot.pause(0.3)
            assert any("line 99" in text for text in painted), "the newest line was never drawn"
            await _settle(app, pilot)

    _run(run())


# ─── L7: no input for a while pauses the live tail ─────────────────────────


def test_no_input_pauses_the_tail_until_p(monkeypatch):
    monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 1)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.05)
    c = _client(_job("RUNNING"))
    counter = iter(range(1_000_000))
    c.tail_logs.side_effect = lambda *a, **k: {"entries": [{"_raw": f"ts line {next(counter)}"}],
                                               "next_cursor": "C1", "eof": False}

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_idle_paused", False), tries=400)
            s = app.screen
            assert _status_line(s) == "paused: no input for 1 s (press p to resume)"
            assert "— paused: no input for 1 s (press p to resume) —" in s._shown_lines
            await pilot.pause(0.3)
            calls = c.tail_logs.call_count
            await pilot.pause(0.5)
            assert c.tail_logs.call_count == calls, "the tail kept polling while paused"
            # A loaded machine can take longer than the 1 s idle time between p and the
            # check below; a second pause meanwhile would make the check fail. No pause
            # can come now: the threshold rises before p, and the pause in force stays.
            monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 15 * 60)
            assert s._idle_paused
            await pilot.press("p")
            assert not s._idle_paused  # p resumed the tail (synchronously, in its key handler)...
            assert not s._paused  # ...and it is not auto-scroll's pause
            assert await _wait(pilot, lambda: c.tail_logs.call_count > calls)
            assert "paused" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


def test_the_log_keeps_following_when_the_status_line_appears():
    c = _client(_job("RUNNING"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_current_source", None))
            s, lv = app.screen, app.screen._logview
            for i in range(200):
                s._write_line(f"line {i}")
            await pilot.pause(0.1)
            assert lv.is_vertical_scroll_end
            s._restart_note = "tail restarted 12:00:00"
            s._render_log_status()  # the pane loses a row to the status line
            await pilot.pause(0.2)
            s._write_lines(["newest"])
            await pilot.pause(0.1)
            assert lv.is_vertical_scroll_end, "the view stopped following the tail"
            await _settle(app, pilot)

    _run(run())


def test_the_log_keeps_following_when_the_status_line_wraps_to_another_row():
    c = _client(_job("RUNNING"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(100, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_current_source", None))
            s, lv = app.screen, app.screen._logview
            bar = s.query_one("#log-status", Static)
            for i in range(200):
                s._write_line(f"line {i}")
            s._restart_note = "tail restarted 12:00:00"
            s._render_log_status()
            await pilot.pause(0.2)
            assert lv.is_vertical_scroll_end and bar.size.height == 1
            # Already shown, it grows a row: the pane shrinks again.
            s._restart_note = "tail restarted 12:00:00: " + "it had stopped moving; " * 6
            s._render_log_status()
            await pilot.pause(0.2)
            assert bar.size.height >= 2
            s._write_lines(["newest"])
            await pilot.pause(0.1)
            assert lv.is_vertical_scroll_end, "the view stopped following the tail"
            # A viewer who scrolled back isn't pulled down by a status change.
            lv.scroll_home(animate=False)
            await pilot.pause(0.1)
            s._restart_note = "tail restarted 12:00:01"
            s._render_log_status()
            await pilot.pause(0.2)
            assert lv.scroll_y == 0
            await _settle(app, pilot)

    _run(run())


def test_a_tail_paused_for_no_input_says_nothing_of_it_once_the_job_ends(monkeypatch):
    monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 1)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.05)
    c = _client(_job("RUNNING"))
    c.tail_logs.return_value = {"entries": [{"_raw": "ts running line"}], "next_cursor": "C1", "eof": False}
    c.download_stdout_logs.side_effect = lambda job_id, root: _write_stdout(root, "7:training:0", "saved line\n")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_idle_paused", False), tries=400)
            s = app.screen
            assert "paused: no input" in _status_line(s)
            c.get_job.return_value = _job("FAILED", reason="out_of_memory")
            s._refresh_status()
            assert await _wait(pilot, lambda: "saved line" in s._shown_lines)
            assert not s._idle_paused
            assert "paused" not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


def test_no_input_pauses_only_once_the_job_runs(monkeypatch):
    # Reads while the job is placed or starts up don't count as activity on it.
    monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 1)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.05)
    c = _client(_job("PLACING"))
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": False}

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "starting")
            s = app.screen
            await pilot.pause(1.5)
            assert not s._idle_paused  # 1.5 s with no input, but the job wasn't running
            c.get_job.return_value = _job("RUNNING")
            flipped = time.monotonic()
            s._refresh_status()
            assert await _wait(pilot, lambda: s._idle_paused, tries=400)
            # The clock started about when it ran (a check tick before), not when the view opened.
            assert time.monotonic() - flipped >= 0.5
            await _settle(app, pilot)

    _run(run())


def test_keys_keep_the_tail_going(monkeypatch):
    monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 1)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.05)
    c = _client(_job("RUNNING"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: c.tail_logs.call_count > 0)
            for _ in range(8):  # ~1.6 s of key presses, never 1 s apart
                await pilot.press("down")
                await pilot.pause(0.2)
            assert not app.screen._idle_paused
            await _settle(app, pilot)

    _run(run())


# ─── L10: the filter and level apply locally ───────────────────────────────


def test_filter_and_level_changes_never_restart_the_tail():
    c = _client(_job("RUNNING"))
    first = {"entries": [{"level": "INFO", "msg": "alpha"}, {"level": "WARNING", "msg": "beta"},
                         {"level": "ERROR", "msg": "gamma"}], "next_cursor": "C1", "eof": False}
    pages = iter([first])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: "ERROR gamma" in getattr(app.screen, "_shown_lines", []))
            s = app.screen
            gen = s._tail_gen
            await pilot.press("/", "b", "e", "t", "a")
            assert await _wait(pilot, lambda: "WARNING beta" in s._shown_lines and "INFO  alpha" not in s._shown_lines)
            assert "ERROR gamma" not in s._shown_lines
            await pilot.press("escape")  # clears the filter
            assert await _wait(pilot, lambda: "INFO  alpha" in s._shown_lines)
            await pilot.press("L", "L")  # INFO, then WARNING and up
            assert await _wait(pilot, lambda: "INFO  alpha" not in s._shown_lines and "ERROR gamma" in s._shown_lines)
            assert s._tail_gen == gen, "a filter or level change started a new tail"
            # Only the one tail polled: the first fetch, then its cursor.
            cursors = [call.kwargs.get("cursor") for call in c.tail_logs.call_args_list]
            assert cursors[0] is None and set(cursors[1:]) <= {"C1"}
            await _settle(app, pilot)

    _run(run())


def test_a_page_from_an_older_tail_never_reaches_the_pane():
    c = _client(_job("RUNNING", sampling=True))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_current_source", None) == "7:training:0")
            s = app.screen
            old = s._tail_gen
            s._start_tail("7:sampling:0")
            # A training page that was in flight when the view switched to sampling.
            s._on_page(old, "live", [{"_raw": "ts stale training line"}], ["ts stale training line"])
            assert "ts stale training line" not in s._shown_lines
            assert all(entry is None or entry.get("_raw") != "ts stale training line" for entry, _ in s._entries)
            s._on_page(s._tail_gen, "live", [{"_raw": "ts sampling line"}], ["ts sampling line"])
            assert "ts sampling line" in s._shown_lines
            await _settle(app, pilot)

    _run(run())


# ─── L8: --poll-interval defaults to the config file ───────────────────────


def test_poll_interval_defaults_to_the_config_file(tmp_path, monkeypatch):
    from cortex_training.tui import __main__ as tui_main

    seen = []

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, schema_only=False):
            seen.append(poll_interval)

        def run(self):
            pass

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training._cli.build_client", lambda args, cls: MagicMock())
    config = tmp_path / "tui-config.json"
    config.write_text(json.dumps({"base_url": "http://localhost:1", "database": "CORTEX_TRAINING_DB",
                                  "poll_interval": 3.0}), encoding="utf-8")
    plain = tmp_path / "plain-config.json"
    plain.write_text(json.dumps({"base_url": "http://localhost:1", "database": "CORTEX_TRAINING_DB"}),
                     encoding="utf-8")
    # The CLI's own result-polling value, as the example config sets it.
    fast = tmp_path / "fast-config.json"
    fast.write_text(json.dumps({"base_url": "http://localhost:1", "database": "CORTEX_TRAINING_DB",
                                "poll_interval": 0.5}), encoding="utf-8")

    assert tui_main.run(["--config", str(config)]) == 0
    assert tui_main.run(["--config", str(config), "--poll-interval", "2.5"]) == 0
    assert tui_main.run(["--config", str(plain)]) == 0
    assert tui_main.run(["--base-url", "http://localhost:1", "--database", "CORTEX_TRAINING_DB"]) == 0
    # A config value applies as written, shorter than the default too; the flag still wins.
    assert tui_main.run(["--config", str(fast)]) == 0
    assert tui_main.run(["--config", str(fast), "--poll-interval", "2"]) == 0
    # The config file a login remembers counts the same as --config.
    login = tmp_path / "login.json"
    login.write_text(json.dumps({"config_path": str(fast)}), encoding="utf-8")
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login))
    assert tui_main.run([]) == 0
    assert seen == [3.0, 2.5, 1.0, 1.0, 0.5, 2.0, 0.5]


# ─── demo ──────────────────────────────────────────────────────────────────


def test_demo_shows_a_placing_wait_and_a_startup_failure(tmp_path, monkeypatch):
    client = DemoClient()
    placing = next(j for j in client.list_jobs() if j["status"] == "PLACING")
    failed = next(j for j in client.list_jobs() if j["reason"] == "out_of_memory")

    async def run():
        app = CortexTrainingLogTUI(client, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            app.push_screen(JobDetailScreen(client, placing["job_id"], job=placing, initial_tab="logs"))
            assert await _wait(pilot, lambda: getattr(app.screen, "_waiting", None) == "starting")
            assert "is PLACING; its log appears here once it is up" in _status_line(app.screen)
            app.pop_screen()
            app.push_screen(JobDetailScreen(client, failed["job_id"], job=failed, initial_tab="logs"))
            assert await _wait(pilot, lambda: any("CUDA out of memory" in line
                                                  for line in getattr(app.screen, "_shown_lines", [])))
            assert any("no saved console for training #0" in line for line in app.screen._shown_lines)
            await _settle(app, pilot)

    _run(run())
