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

"""The Logs tab, round 1: hostile and wide lines, the saved console's level
filter and repaint, the lines a job's end must not take away, scrollback
notes, export, the waiting message, a job that can't be read, and the
subtitle."""

import asyncio
import json
import threading
import time
import xml.dom.minidom
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402
from cortex_training.tui.log_cache import LogCache  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402

STAMP = "2026-10-15T08:06:35.886109202Z"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Keep the TUI's caches, error log, saved logs and login state out of the real home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(status, *, reason=None, details=None, sampling=False, **extra):
    subs = [{"sub_job_id": "7:training:0", "job_type": "training", "status": status}]
    if sampling:
        subs.append({"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": status})
    job = {"job_id": "7", "status": status, "sub_jobs": subs, **extra}
    if reason:
        job["reason"] = reason
    if details:
        job["reason_details"] = details
    return job


def _client(job):
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "C1", "eof": False}
    c.fetch_execution_logs.return_value = []
    return c


def _http_error(status, message):
    resp = requests.Response()
    resp.status_code = status
    resp.url = "http://test.local/x"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps({"message": message, "code": "390400"}).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Error", response=resp)


def _write_stdout(root, sub_job_id, text):
    path = Path(root) / sub_job_id / "stdout.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


async def _wait(pilot, predicate, tries=400, delay=0.01):
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


def _status_line(screen) -> str:
    bar = screen.query_one("#log-status", Static)
    return str(bar.render()) if bar.display else ""


def _summary(screen) -> str:
    return str(screen.query_one("#summary", Static).render())


def _rows(screen) -> list:
    return [str(row) for row in screen._logview.lines]


def _run(coro):
    asyncio.run(coro)


def _record(level, msg, **fields):
    return f"{STAMP} " + json.dumps({"level": level, "logger": "trainer", "msg": msg, **fields})


# ─── R1-05: escape sequences in log lines never reach the terminal ───────────

HOSTILE = [
    "\x1b[31mERROR\x1b[0m coloured line",
    "\x1b]52;c;aGFja2Vk\x07 OSC 52 clipboard write; \x1b]0;window title\x07 OSC 0 title write",
    "\x1b[2J\x1b[H screen clear \x1b[?1049h alt screen",
    "\x1b[36m(TrainingWorker pid=1, ip=10.0.0.1)\x1b[0m " + json.dumps({"level": "INFO", "msg": "ray line"}),
    "progress  10%|#  | 1/10\rprogress 100%|##| 10/10",
    "an unterminated OSC \x1b]0;title",
]


def test_hostile_lines_reach_neither_the_pane_nor_a_screenshot_raw():
    c = _client(_job("RUNNING"))
    pages = iter([{"entries": [{"_raw": f"{STAMP} {line}"} for line in HOSTILE], "next_cursor": "C1", "eof": False}])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any("alt screen" in row for row in _rows(app.screen)))
            s = app.screen
            rows = _rows(s)
            assert not any("\x1b" in row or "\x07" in row or "\r" in row for row in rows)
            assert any("progress 100%" in row for row in rows) and not any("progress  10%" in row for row in rows)
            assert any("ray line" in row and "(TrainingWorker pid=1, ip=10.0.0.1)" in row for row in rows)
            svg = app.export_screenshot()
            assert "\x1b" not in svg
            xml.dom.minidom.parseString(svg.encode("utf-8"))  # Textual's screenshot is valid XML again
            await _settle(app, pilot)

    _run(run())


def test_copy_all_sends_the_text_without_escape_sequences():
    c = _client(_job("RUNNING"))
    LogCache("7").append_entries("7:training:0", [{"_raw": f"{STAMP} {line}"} for line in HOSTILE])

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any("alt screen" in row for row in _rows(app.screen)))
            copied = []
            app.copy_to_clipboard = copied.append
            app.screen.action_copy_log()
            assert await _wait(pilot, lambda: bool(copied))
            assert not any(ch in copied[0] for ch in "\x1b\x07\r")
            assert "OSC 52 clipboard write" in copied[0] and "ray line" in copied[0]
            await _settle(app, pilot)

    _run(run())


# ─── R1-54: wide characters wrap inside the pane ─────────────────────────────

def test_wide_lines_wrap_by_terminal_cells():
    c = _client(_job("RUNNING"))
    wide = f"{STAMP} wide: " + "训练数据" * 60 + " END"
    emoji = f"{STAMP} emoji: " + "\U0001F680" * 80 + " END"
    c.tail_logs.side_effect = [{"entries": [{"_raw": wide}, {"_raw": emoji}], "next_cursor": "C1", "eof": False}] + [
        {"entries": [], "next_cursor": "C1", "eof": False}] * 10_000

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(100, 40)) as pilot:
            assert await _wait(pilot, lambda: emoji in app.screen._shown_lines)
            s = app.screen
            width = s._content_width()
            assert width > 0
            assert all(cell_len(row) <= width for row in _rows(s)), max(cell_len(row) for row in _rows(s))
            await _settle(app, pilot)

    _run(run())


def test_the_row_for_a_long_lines_middle_does_not_scroll_a_narrow_pane_sideways():
    # At 80 columns the log has 60 cells, 47 after the sub-job tag: the row that
    # stands for the hidden rows is cut to fit, and the pane never scrolls sideways.
    c = _client(_job("RUNNING"))
    c.tail_logs.side_effect = [{"entries": [{"_raw": f"{STAMP} long-token " + "x" * 5000}], "next_cursor": "C1"}] + [
        {"entries": [], "next_cursor": "C1", "eof": False}] * 10_000

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(80, 45)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: any("more row(s)" in row for row in _rows(s)))
            await pilot.pause(0.1)
            width = s._content_width()
            assert all(cell_len(row) <= width for row in _rows(s)), max(cell_len(row) for row in _rows(s))
            assert s._logview.max_scroll_x == 0
            assert any(row.endswith("… 89 more row(s) (s saves the whole log)") for row in _rows(s))
            await _settle(app, pilot)

    _run(run())


# ─── R1-18 / R1-20: the level filters a saved console, off the UI thread ─────

SAVED = "".join([
    _record("INFO", "step 1") + "\n",
    _record("WARNING", "grad norm spike") + "\n",
    _record("ERROR", "rank 3 failed") + "\n",
    "Traceback (most recent call last):\n",
])


def _finished_client(text=SAVED):
    c = _client(_job("FAILED", reason="sub_job_failed"))
    c.download_stdout_logs.side_effect = lambda job_id, root, **kw: _write_stdout(root, "7:training:0", text)
    return c


def test_the_level_key_filters_a_saved_console():
    c = _finished_client()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any("step 1" in line for line in app.screen._shown_lines))
            s = app.screen
            await pilot.press("L", "L", "L")  # ≥ERROR
            assert await _wait(pilot, lambda: not any("step 1" in line for line in s._shown_lines))
            shown = s._shown_lines
            assert any("rank 3 failed" in line for line in shown)
            assert not any("grad norm spike" in line for line in shown)
            assert "Traceback (most recent call last):" in shown  # no level of its own: always shown
            assert "≥ERROR" in app.sub_title
            await pilot.press("L")  # off again
            assert await _wait(pilot, lambda: any("step 1" in line for line in s._shown_lines))
            await _settle(app, pilot)

    _run(run())


def test_a_saved_console_repaints_off_the_ui_thread(monkeypatch):
    c = _finished_client()
    release = threading.Event()
    scanning = threading.Event()
    scan = LogScreen._scan_saved

    def slow_scan(self, *args, **kwargs):
        if self._min_level:  # the repaint after L, not the first paint
            scanning.set()
            release.wait(timeout=10)
        return scan(self, *args, **kwargs)

    monkeypatch.setattr(LogScreen, "_scan_saved", slow_scan)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any("step 1" in line for line in app.screen._shown_lines))
            s = app.screen
            await pilot.press("L")
            assert await _wait(pilot, scanning.is_set)
            started = time.monotonic()
            await pilot.press("/")  # the UI still answers while the console is read
            assert await _wait(pilot, lambda: s.focused is s.query_one("#logfilter"))
            assert time.monotonic() - started < 2
            assert any("step 1" in line for line in s._shown_lines)  # the old paint stays until the new one is ready
            release.set()
            assert await _wait(pilot, lambda: not s._repainting)
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        release.set()


def test_a_filter_that_hides_every_line_says_so():
    c = _finished_client()

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, lambda: any("step 1" in line for line in app.screen._shown_lines))
            s = app.screen
            await pilot.press("/", *"zzzz-no-match", "enter")
            assert await _wait(pilot, lambda: "no lines match filter 'zzzz-no-match'" in _status_line(s))
            await _settle(app, pilot)

    _run(run())


# ─── R1-17: the job's end keeps the lines the tail showed ────────────────────

def test_the_saved_console_keeps_the_tails_newer_lines_until_it_has_them(monkeypatch):
    monkeypatch.setattr(LogScreen, "_KEEP_CONSOLE_AFTER_SECONDS", 2.0)
    now = datetime.now(timezone.utc)
    created = _iso(now - timedelta(hours=3))
    running = {**_job("RUNNING"), "created_at": created, "state_changed_at": _iso(now - timedelta(hours=2))}
    ended = {**_job("FAILED", reason="sub_job_failed"), "created_at": created, "state_changed_at": _iso(now)}
    c = _client(running)
    jobs = iter([running, ended])
    c.get_job.side_effect = lambda _id: next(jobs, ended)
    lines = [f"{STAMP[:17]}{i:02d}.0Z step {i}" for i in range(20)]
    pages = iter([{"entries": [{"_raw": line} for line in lines], "next_cursor": "C1", "eof": False}])

    def tail(*a, **k):
        page = next(pages, None)
        if page is None:
            raise _http_error(409, "Cortex training operation failed: session 7 has failed: sub_job_failed")
        return page

    c.tail_logs.side_effect = tail
    # The console's last chunks aren't uploaded yet: it stops at step 11 (saved lines carry no timestamp).
    console = {"text": "".join(f"step {i}\n" for i in range(12))}
    c.download_stdout_logs.side_effect = lambda job_id, root, **kw: _write_stdout(root, "7:training:0",
                                                                                  console["text"])

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: s._stdout_ready and "step 11" in s._shown_lines and not s._repainting)
            shown = s._shown_lines
            assert shown[-1] == s._recent_console_note(newer=8)
            assert "the live tail's last 8 lines are shown after it" in shown[-1]
            banner = "— 8 newer lines from the live tail, not in the saved console yet —"
            assert banner in shown
            assert shown[shown.index(banner) + 1:-1] == lines[12:]  # the pane still ends with the tail's newest
            assert shown.index("step 11") < shown.index(banner)
            # r once the chunks are uploaded: the console has them, and the banner goes.
            console["text"] = "".join(f"step {i}\n" for i in range(20))
            await pilot.pause(2.2)
            await pilot.press("r")
            assert await _wait(pilot, lambda: "step 19" in s._shown_lines and not s._repainting)
            assert not any("newer line" in line for line in s._shown_lines)
            await _settle(app, pilot)

    _run(run())


def test_a_console_that_ends_before_the_tails_first_line_keeps_the_whole_tail(monkeypatch):
    # The console's last uploaded chunk ends before the window the tail opened
    # with: the two don't overlap, and every line the tail showed is newer.
    monkeypatch.setattr(LogScreen, "_KEEP_CONSOLE_AFTER_SECONDS", 600)
    now = datetime.now(timezone.utc)
    running = {**_job("RUNNING"), "created_at": _iso(now - timedelta(hours=3))}
    ended = {**_job("FAILED", reason="sub_job_failed"), "created_at": running["created_at"],
             "state_changed_at": _iso(now)}
    c = _client(running)
    jobs = iter([running, ended])
    c.get_job.side_effect = lambda _id: next(jobs, ended)
    lines = [f"{STAMP[:17]}{i:02d}.0Z step {i}" for i in range(10, 20)]
    pages = iter([{"entries": [{"_raw": line} for line in lines], "next_cursor": "C1", "eof": False}])

    def tail(*a, **k):
        page = next(pages, None)
        if page is None:
            raise _http_error(409, "Cortex training operation failed: session 7 has failed: sub_job_failed")
        return page

    c.tail_logs.side_effect = tail
    c.download_stdout_logs.side_effect = lambda job_id, root, **kw: _write_stdout(
        root, "7:training:0", "".join(f"step {i}\n" for i in range(6)))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: s._stdout_ready and "step 5" in s._shown_lines and not s._repainting)
            banner = ("— 10 newer lines from the live tail, not in the saved console yet; lines between the two may "
                      "be missing —")
            assert banner in s._shown_lines
            assert s._shown_lines[s._shown_lines.index(banner) + 1:-1] == lines
            await _settle(app, pilot)

    _run(run())


# ─── R1-16: a pane that never had the log's start says so ────────────────────

def _window_client(n):
    c = _client(_job("RUNNING"))
    pages = iter([{"entries": [{"_raw": f"ts line {i}"} for i in range(n)], "next_cursor": "C1", "eof": False}])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})
    return c


def test_opening_a_long_running_log_says_it_shows_the_last_lines():
    c = _window_client(2000)  # the first fetch reads the last 2,000 lines: older ones exist

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts line 1999" in s._shown_lines)
            assert await _wait(pilot, lambda: _status_line(s).startswith("last 2,000 lines · cortex-training"))
            assert "download-log 7 --log-type stdout for the full log" in _status_line(s)
            assert c.tail_logs.call_args_list[0].kwargs.get("max_lines") == 2000
            await _settle(app, pilot)

    _run(run())


def test_a_short_log_read_whole_says_nothing_of_older_lines():
    c = _window_client(50)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts line 49" in s._shown_lines)
            for _ in range(20):
                await pilot.pause(0.01)
            assert "last " not in _status_line(s)
            await _settle(app, pilot)

    _run(run())


def test_a_short_log_that_writes_before_every_read_says_nothing_of_older_lines():
    # A job that just started: 120 lines when the tail opens, then 20 more before every read.
    c = _client(_job("RUNNING"))
    log = {"n": 120}

    def tail(job_id, cursor=None, sub_job_id=None, max_lines=None):
        start = int(cursor) if cursor else max(0, log["n"] - (max_lines or log["n"]))
        entries = [{"_raw": f"ts line {i:05d}"} for i in range(start, min(log["n"], start + 2000))]
        log["n"] += 20
        return {"entries": entries, "next_cursor": str(start + len(entries)), "eof": False}

    c.tail_logs.side_effect = tail

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts line 02600" in s._shown_lines, tries=3000)
            assert s._shown_lines[s._shown_lines.index("ts line 00000") - 1] == "— tailing 7:training:0 —"
            assert "last " not in _status_line(s)  # the pane holds the log from its first line
            assert not LogCache("7").head_missing("7:training:0")
            await _settle(app, pilot)

    _run(run())


def test_a_replay_of_a_cache_that_lacks_the_start_says_so():
    LogCache("7").append_entries("7:training:0", [{"_raw": f"ts cached {i}"} for i in range(2500)])
    c = _client(_job("RUNNING"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "ts cached 2499" in s._shown_lines)
            assert "ts cached 499" not in s._shown_lines  # the replay holds the newest 2,000
            assert await _wait(pilot, lambda: _status_line(s).startswith("last 2,000 lines"))
            assert "— live —" not in s._shown_lines  # nothing live has arrived yet
            await _settle(app, pilot)

    _run(run())


# ─── R1-52 / R1-53 / R1-95: save and copy ────────────────────────────────────

EXEC_LOG = "\n".join(json.dumps(r) for r in [
    {"ts": "2026-10-15T09:33:01+00:00", "level": "ERROR", "logger": "runtime.manager",
     "msg": "Failed to create job", "exc_info": "Traceback (most recent call last):\nRuntimeError: CUDA out of memory"},
    {"ts": "2026-10-15T09:33:01+00:00", "level": "INFO", "logger": "runtime.job_metrics", "msg": "job_metrics",
     "error": "RuntimeError: loading the model failed"},
])


def test_save_and_copy_on_an_execution_log_view_export_what_the_pane_shows():
    c = _client(_job("FAILED", reason="sub_job_failed"))
    c.download_stdout_logs.return_value = []  # no console: the job failed before it wrote one
    c.fetch_execution_logs.return_value = [{"sub_job_id": "7:training:0", "content": EXEC_LOG}]

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: any("Failed to create job" in line for line in s._shown_lines))
            await pilot.press("s")
            saved = Path.home() / "cortex-training-7-training-0.log"  # the job id once, then type and index
            assert await _wait(pilot, saved.is_file)
            text = saved.read_text(encoding="utf-8")
            assert "Failed to create job" in text and "CUDA out of memory" in text and "loading the model" in text
            assert await _wait(pilot, lambda: _status_line(s).startswith("saved "))
            assert not any(line.startswith("[saved]") for line in s._shown_lines)  # on the status line, not the log
            copied = []
            app.copy_to_clipboard = copied.append
            await pilot.press("y")
            assert await _wait(pilot, lambda: bool(copied))
            assert "Failed to create job" in copied[0]
            assert await _wait(pilot, lambda: "to the terminal's clipboard" in _status_line(s))
            # ≥ERROR keeps the INFO record that carries an error.
            await pilot.press("L", "L", "L")
            assert await _wait(pilot, lambda: any("job_metrics" in line for line in s._shown_lines) and
                               "≥ERROR" in app.sub_title)
            await _settle(app, pilot)

    _run(run())


def test_nothing_to_save_writes_no_empty_file():
    c = _client(_job("FAILED"))
    c.download_stdout_logs.return_value = []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: s._stdout_ready and not s._repainting)
            await pilot.press("s")
            assert await _wait(pilot, lambda: "nothing to save yet" in _status_line(s))
            assert not list(Path.home().glob("cortex-training-*.log"))
            await _settle(app, pilot)

    _run(run())


def test_copy_all_is_capped_and_says_how_much_it_sent(monkeypatch):
    monkeypatch.setattr(LogScreen, "_COPY_MAX_BYTES", 1000)
    text = "".join(f"console line {i:04d} " + "x" * 80 + "\n" for i in range(100))
    c = _finished_client(text)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: any("console line 0099" in line for line in s._shown_lines))
            copied = []
            app.copy_to_clipboard = copied.append
            await pilot.press("y")
            assert await _wait(pilot, lambda: bool(copied))
            assert len(copied[0].encode("utf-8")) <= 1000
            assert copied[0].endswith("x" * 80)  # the newest lines
            # Ten 98-byte lines and their line breaks fit in 1,000 bytes.
            assert copied[0].count("\n") == 9
            assert await _wait(pilot, lambda: "sent the last 10 lines to the terminal's clipboard (of 100; s saves "
                                              "them all)" in _status_line(s))
            await _settle(app, pilot)

    _run(run())


# ─── R1-61: one waiting message ──────────────────────────────────────────────

def test_a_pending_job_says_it_waits_once_with_the_check_cadence():
    c = _client(_job("PENDING", reason="quota_filled", details="needs 8 GPUs"))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: s._waiting == "pending")
            assert _status_line(s) == ("waiting: the job is PENDING (quota_filled — needs 8 GPUs); the log tail "
                                       "starts by itself when the job starts (status checked every 60 s · r checks "
                                       "now)")
            assert not s._shown_lines  # no "tailing" banner, no second copy of the note
            await _settle(app, pilot)

    _run(run())


# ─── R1-15: a job the service doesn't answer for ─────────────────────────────

def test_an_empty_get_falls_back_to_the_listed_row():
    listed = _job("CANCELLED")
    c = _client(listed)
    c.get_job.return_value = {}
    c.download_stdout_logs.side_effect = lambda job_id, root, **kw: _write_stdout(root, "7:training:0", "saved\n")

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            for sub_job in (None, "7:training:0"):
                app.push_screen(JobDetailScreen(c, "7", job=listed, sub_job_id=sub_job, initial_tab="logs"))
                assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
                s = app.screen
                assert await _wait(pilot, lambda: "saved" in s._shown_lines)
                assert not any("no sub-jobs" in line or "not found" in line for line in s._shown_lines)
                app.pop_screen()
                await pilot.pause()
            await _settle(app, pilot)

    _run(run())


def test_a_job_that_cant_be_read_says_why_on_logs_and_resources():
    c = _client(_job("RUNNING"))
    c.get_job.side_effect = _http_error(404, "Training job 7 not found")
    c.database, c.schema = "TEAM_DB", "EXPERIMENTS"

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            app.push_screen(JobDetailScreen(c, "7", sub_job_id="7:training:0", initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
            s = app.screen
            assert await _wait(pilot, lambda: any(line.startswith("[no logs]") for line in s._shown_lines))
            line = next(line for line in s._shown_lines if line.startswith("[no logs]"))
            assert "was not found in TEAM_DB.EXPERIMENTS" in line and "Training job 7 not found" in line
            assert not any("sub-job not found" in line for line in s._shown_lines)
            # (Nothing focused in the Logs pane: hiding it then can't hand focus back to it.)
            s.set_focus(None)
            await pilot.press("3")
            resources = s.query_one("#resource-status", Static)
            assert await _wait(pilot, lambda: "was not found in TEAM_DB.EXPERIMENTS" in str(resources.render()))
            await _settle(app, pilot)

    _run(run())


def test_any_other_failed_get_is_not_called_a_job_without_sub_jobs():
    c = _client(_job("RUNNING"))
    c.get_job.side_effect = requests.exceptions.ConnectionError("network is unreachable")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            # A read that failed is an error, not a job without logs: it names the cause.
            assert await _wait(pilot, lambda: any(line.startswith("[error]") for line in s._shown_lines))
            line = next(line for line in s._shown_lines if line.startswith("[error]"))
            assert "couldn't read job 7" in line and "network is unreachable" in line and "no sub-jobs" not in line
            await _settle(app, pilot)

    _run(run())


# ─── R1-46 / R1-47 / R1-51: the subtitle and the summary line ────────────────

def test_the_subtitle_leads_with_the_status_and_dates_the_newest_line_by_its_stamp():
    c = _client(_job("RUNNING", sampling=True))

    def tail(job_id, **kw):
        if kw.get("sub_job_id") == "7:training:0":
            return {"entries": [{"_raw": f"{STAMP} training line"}], "next_cursor": "C1", "eof": False}
        return {"entries": [], "next_cursor": "", "eof": False}

    c.tail_logs.side_effect = tail
    local = datetime.fromisoformat("2026-10-15T08:06:35+00:00").astimezone().strftime("%H:%M:%S")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: f"last line {local}" in (app.sub_title or ""))
            assert app.sub_title.startswith("RUNNING  ·  7")
            s._start_tail("7:sampling:0")  # another sub-job: no carried-over time
            assert "last line" not in app.sub_title
            s.set_focus(None)  # (as above: hiding Logs can't hand focus back to it)
            await pilot.press("3")  # the Logs tab's state shows on Logs only
            assert await _wait(pilot, lambda: s._active_tab == "resource")
            s._filter = "x"
            s._update_subtitle()
            assert "filter" not in app.sub_title
            await _settle(app, pilot)

    _run(run())


def test_the_end_the_tail_sees_reaches_the_summary_line_at_once():
    c = _client(_job("RUNNING"))
    jobs = iter([_job("RUNNING")])
    ended = _job("FAILED", reason="sub_job_failed", details="rank 0 exited with code 1")
    c.get_job.side_effect = lambda _id: next(jobs, ended)
    c.tail_logs.side_effect = _http_error(409, "Cortex training operation failed: session 7 has failed")
    c.download_stdout_logs.side_effect = lambda job_id, root, **kw: _write_stdout(root, "7:training:0", "saved\n")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "saved" in s._shown_lines)
            assert "sub_job_failed — rank 0 exited with code 1" in _summary(s)
            assert app.sub_title.startswith("FAILED") and "reason" not in app.sub_title
            await _settle(app, pilot)

    _run(run())


def test_a_reason_on_the_plain_log_screen_is_drawn_safely_with_its_details():
    c = _client(_job("FAILED", reason="bad\x1b]0;title\x07", details="rank 0\nexited"))
    c.download_stdout_logs.return_value = []

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            app.push_screen(LogScreen(c, "7", poll_interval=0.01))
            assert await _wait(pilot, lambda: "reason:" in (app.sub_title or ""))
            assert "\x1b" not in app.sub_title and "\n" not in app.sub_title
            assert "reason: bad — rank 0 exited" in app.sub_title
            await _settle(app, pilot)

    _run(run())


# ─── R1-56: kept consoles are bounded ────────────────────────────────────────

def test_keeping_a_console_removes_the_least_recently_used_past_the_bound(monkeypatch):
    from cortex_training.tui import log_cache as lc
    from cortex_training.tui.log_cache import job_cache_dir

    monkeypatch.setattr(lc, "CONSOLE_CACHE_MAX_BYTES", 1500)
    long_ago = _iso(datetime.now(timezone.utc) - timedelta(days=2))
    jobs = {jid: {**_job("CANCELLED"), "job_id": jid, "created_at": long_ago, "state_changed_at": long_ago,
                  "sub_jobs": [{"sub_job_id": f"{jid}:training:0", "job_type": "training", "status": "CANCELLED"}]}
            for jid in ("job-a", "job-b")}
    c = _client(jobs["job-a"])
    c.get_job.side_effect = lambda jid: jobs[jid]
    c.download_stdout_logs.side_effect = lambda jid, root, **kw: _write_stdout(root, f"{jid}:training:0",
                                                                             f"{jid} " + "x" * 1000 + "\n")

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            for jid in ("job-a", "job-b"):
                app.push_screen(JobDetailScreen(c, jid, job=jobs[jid], initial_tab="logs"))
                assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
                assert await _wait(pilot, lambda: (job_cache_dir(jid) / "console" / ".complete").exists())
                app.pop_screen()
                await pilot.pause()
            assert not (job_cache_dir("job-a") / "console").exists()  # 2 x 1 KB > 1.5 KB: the older one goes
            assert (job_cache_dir("job-b") / "console" / ".complete").exists()
            await _settle(app, pilot)

    _run(run())


# ─── R1-19: the sub-job on screen first ──────────────────────────────────────

def test_the_sub_job_on_screen_is_downloaded_and_shown_first():
    job = _job("CANCELLED", sampling=True)
    c = _client(job)
    rest = threading.Event()
    calls = []

    def download_stdout_logs(job_id, output_dir, *, resume=False, sub_job_ids=None, chunk_progress=None):
        calls.append((tuple(sub_job_ids) if sub_job_ids else None, resume))
        for sid in sub_job_ids or ["7:sampling:0", "7:training:0"]:
            if sub_job_ids is None and not rest.wait(timeout=10):
                raise TimeoutError("the rest of the job")
            if sub_job_ids is None and sid == "7:training:0":
                continue  # in already (resumed): nothing to fetch
            for done in (1, 2):
                if progress is not None:
                    progress(sid, done, 2)
            _write_stdout(output_dir, sid, f"{sid} console\n")
        return []

    c.download_stdout_logs = download_stdout_logs  # a client that can fetch one sub-job

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            # Training is selected first: its console shows while the rest of the job still downloads.
            assert await _wait(pilot, lambda: "7:training:0 console" in s._shown_lines)
            assert calls[0] == (("7:training:0",), True)
            assert not s._stdout_ready
            assert await _wait(pilot, lambda: "downloading the other sub-jobs' saved consoles" in _status_line(s)
                               or calls[1:] == [(None, True)])
            rest.set()
            assert await _wait(pilot, lambda: s._stdout_ready)
            assert calls[1] == (None, True)  # then the whole job, resumed
            assert "downloading" not in _status_line(s)
            await pilot.press("down", "enter")  # sampling: in already, no new download
            assert await _wait(pilot, lambda: "7:sampling:0 console" in s._shown_lines)
            assert len(calls) == 2
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        rest.set()


def test_the_pane_says_which_sub_jobs_console_comes_first():
    # The pane's line matches the order: the sub-job on screen first, then the others'.
    job = _job("CANCELLED", sampling=True)
    c = _client(job)
    first_in, rest = threading.Event(), threading.Event()

    def download_stdout_logs(job_id, output_dir, *, resume=False, sub_job_ids=None, chunk_progress=None):
        if not (first_in if sub_job_ids else rest).wait(timeout=10):
            raise TimeoutError("held by the test")
        for sid in sub_job_ids or ["7:sampling:0"]:
            progress(sid, 1, 1)
            _write_stdout(output_dir, sid, f"{sid} console\n")
        return []

    c.download_stdout_logs = download_stdout_logs
    training_first = "— downloading training #0's saved console first, then the other sub-jobs' —"

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: training_first in s._shown_lines)
            assert not any("the whole job" in line for line in s._shown_lines)
            await pilot.press("down", "enter")  # sampling, while training's console still comes
            assert await _wait(pilot, lambda: s._shown_lines == [
                "— downloading training #0's saved console first, then the other sub-jobs' (sampling #0 among them) —"])
            await pilot.press("up", "enter")
            assert await _wait(pilot, lambda: s._shown_lines == [training_first])
            first_in.set()
            assert await _wait(pilot, lambda: "7:training:0 console" in s._shown_lines)
            # Before the others' first chunk: their phase, not training's last chunk count.
            assert _status_line(s) == "downloading the other sub-jobs' saved consoles"
            await pilot.press("down", "enter")  # training's is in; sampling's comes with the rest
            assert await _wait(pilot, lambda: s._shown_lines == [
                "— downloading the other sub-jobs' saved consoles (sampling #0 among them) —"])
            rest.set()
            assert await _wait(pilot, lambda: "7:sampling:0 console" in s._shown_lines)
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        first_in.set()
        rest.set()


def test_a_job_with_one_sub_job_names_no_others_while_it_downloads():
    c = _client(_job("CANCELLED"))
    hold = threading.Event()

    def download_stdout_logs(job_id, output_dir, *, resume=False, sub_job_ids=None, chunk_progress=None):
        for sid in sub_job_ids or []:
            chunk_progress(sid, 1, 3)
            if not hold.wait(timeout=10):
                raise TimeoutError("held by the test")
            _write_stdout(output_dir, sid, f"{sid} console\n")
        return []

    c.download_stdout_logs = download_stdout_logs

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = app.screen
            assert await _wait(pilot, lambda: "— downloading training #0's saved console —" in s._shown_lines)
            assert await _wait(pilot, lambda: _status_line(s) == "downloading training #0's saved console: chunk 1 of 3")
            hold.set()
            assert await _wait(pilot, lambda: "7:training:0 console" in s._shown_lines)
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        hold.set()
