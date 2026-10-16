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

"""The log tail loop: the first fetch without a cursor, and the stuck-tail restart."""

import pytest

from cortex_training.tui import log_cache
from cortex_training.tui.log_cache import RESTART_LINES
from cortex_training.tui.log_cache import STALL_POLLS
from cortex_training.tui.log_cache import STALL_SECONDS
from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import cached_log_pages
from cortex_training.tui.log_cache import cached_pages
from cortex_training.tui.log_cache import entry_key


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    # Keep any persistence warning out of the real home directory.
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(tmp_path / "errors.log"))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))


def _raw(n):
    return {"_raw": f"2026-10-14T12:00:{n:02d}.000000001Z line {n}"}


class _Clock:
    """A clock that a fake sleep advances."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 1.0)


def test_empty_cursor_still_tails_recent_lines(tmp_path):
    # A placing job answers with an empty cursor, which the cache saves; the next
    # open must still tail the last lines, not read the whole log from the start.
    cache = LogCache("j", root=tmp_path)
    cache.set_cursor("sub", "")
    calls = []

    class Client:
        def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
            calls.append({"cursor": cursor, "max_lines": max_lines})
            return {"entries": [{"msg": "x"}], "next_cursor": "C1", "eof": False} if not cursor else \
                {"entries": [], "next_cursor": "C1", "eof": True}

    out = list(cached_log_pages(cache, Client(), "j", "sub", follow=False, poll_interval=0.0, tail_lines=2000))
    assert out == [("live", [{"msg": "x"}])]
    assert calls[0] == {"cursor": "", "max_lines": 2000}
    assert calls[1] == {"cursor": "C1", "max_lines": None}


def test_entry_key_uses_the_raw_line_or_canonical_json():
    assert entry_key({"_raw": "ts line"}) == "ts line"
    assert entry_key({"b": 1, "a": 2}) == entry_key({"a": 2, "b": 1})
    assert entry_key({"a": 1}) != entry_key({"a": 2})


def _stuck_tail(tmp_path, *, restart_pages, can_restart=lambda: True, hold=None, polls=200):
    """Drive a tail stuck at cursor C5 (empty pages, same cursor) until ``polls`` fetches."""
    cache = LogCache("j", root=tmp_path)
    clock = _Clock()
    fetched = []
    restarts = []

    def fetch(cursor):
        fetched.append(cursor)
        if cursor is None:
            return {"entries": [_raw(n) for n in range(1, 6)], "next_cursor": "C5"}
        return {"entries": [], "next_cursor": cursor}  # stuck: the oversized line never arrives

    def restart(lines):
        restarts.append(lines)
        return restart_pages.pop(0) if restart_pages else {"entries": [], "next_cursor": "C5"}

    pages = cached_pages(
        cache, "sub", fetch, entries_key="entries", follow=True, poll_interval=1.0, sleep=clock.sleep,
        is_cancelled=lambda: len(fetched) >= polls, hold=hold, restart=restart, can_restart=can_restart,
        clock=clock,
    )
    return list(pages), fetched, restarts, cache


def test_a_stuck_tail_restarts_from_the_end_and_drops_lines_already_shown(tmp_path):
    restart_page = {"entries": [_raw(4), _raw(5), _raw(9), _raw(10)], "next_cursor": "C10"}
    out, fetched, restarts, cache = _stuck_tail(tmp_path, restart_pages=[restart_page], polls=STALL_POLLS + 40)
    assert out[0] == ("live", [_raw(n) for n in range(1, 6)])
    # Lines 4 and 5 were shown already; only 9 and 10 are new.
    assert out[1] == ("restart", [_raw(9), _raw(10)])
    assert restarts[0] == RESTART_LINES[0]
    # The tail continues from the restart's cursor, and the cache holds each line once.
    assert "C10" in fetched
    assert cache.get_cursor("sub") == "C10"
    assert [e["_raw"] for e in cache.load_entries("sub")] == [_raw(n)["_raw"] for n in (1, 2, 3, 4, 5, 9, 10)]


def test_a_quiet_log_keeps_its_cursor_and_shrinks_the_restart_window(tmp_path):
    # Nothing new comes back: the log was just quiet. No page is yielded, the
    # cursor stays, and each later restart reads fewer lines.
    nothing_new = {"entries": [_raw(4), _raw(5)], "next_cursor": "C-other"}
    out, fetched, restarts, cache = _stuck_tail(tmp_path, restart_pages=[nothing_new, dict(nothing_new)], polls=400)
    assert out == [("live", [_raw(n) for n in range(1, 6)])]
    assert set(fetched[1:]) == {"C5"}
    assert cache.get_cursor("sub") == "C5"
    assert restarts[:3] == [RESTART_LINES[0], RESTART_LINES[1], RESTART_LINES[2]]
    assert set(restarts[3:]) <= {RESTART_LINES[-1]}


def test_restart_waits_for_the_stall_time_and_poll_count(tmp_path):
    clock = _Clock()
    cache = LogCache("j", root=tmp_path)
    log = []

    def fetch(cursor):
        log.append(("fetch", clock()))
        return {"entries": [], "next_cursor": "C1"}

    def restart(lines):
        log.append(("restart", clock()))
        return {"entries": [], "next_cursor": "C1"}

    cache.set_cursor("sub", "C1")
    list(cached_pages(cache, "sub", fetch, entries_key="entries", follow=True, poll_interval=1.0,
                      sleep=clock.sleep, is_cancelled=lambda: any(k == "restart" for k, _ in log),
                      restart=restart, can_restart=lambda: True, clock=clock))
    first_restart = next(t for k, t in log if k == "restart")
    polls_before = sum(1 for k, _ in log if k == "fetch")
    assert first_restart - 1000.0 >= STALL_SECONDS
    assert polls_before >= STALL_POLLS


def test_no_restart_while_the_job_is_not_running(tmp_path):
    restart_page = {"entries": [_raw(9)], "next_cursor": "C9"}
    out, _fetched, restarts, _cache = _stuck_tail(
        tmp_path, restart_pages=[restart_page], can_restart=lambda: False, polls=120)
    assert restarts == []
    assert [kind for kind, _ in out] == ["live"]


def test_time_on_hold_is_not_a_stall(tmp_path):
    # Held (Logs hidden, or paused for no input) for far longer than the stall
    # time: once the tail resumes it must poll a while before any restart.
    state = {"held": 0}

    def hold():
        state["held"] += 1
        return 3 <= state["held"] < 400  # hold after the first polls, ~400 s

    restart_page = {"entries": [_raw(9)], "next_cursor": "C9"}
    out, fetched, restarts, _cache = _stuck_tail(tmp_path, restart_pages=[restart_page], hold=hold, polls=6)
    assert restarts == []


def test_a_moving_cursor_is_not_stuck(tmp_path):
    clock = _Clock()
    cache = LogCache("j", root=tmp_path)
    seen = {"n": 0, "restarts": 0}

    def fetch(cursor):
        seen["n"] += 1
        return {"entries": [], "next_cursor": f"C{seen['n']}"}  # empty, but the cursor advances

    def restart(lines):
        seen["restarts"] += 1
        return {"entries": [], "next_cursor": "C0"}

    list(cached_pages(cache, "sub", fetch, entries_key="entries", follow=True, poll_interval=1.0,
                      sleep=clock.sleep, is_cancelled=lambda: seen["n"] >= 120, restart=restart,
                      can_restart=lambda: True, clock=clock))
    assert seen["restarts"] == 0


def test_cached_log_pages_restarts_through_tail_logs_with_a_fresh_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr(log_cache, "STALL_SECONDS", 0.0)
    monkeypatch.setattr(log_cache, "STALL_POLLS", 1)
    cache = LogCache("j", root=tmp_path)
    cache.set_cursor("sub", "C5")
    calls = []

    class Client:
        def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
            calls.append({"cursor": cursor, "max_lines": max_lines, "sub_job_id": sub_job_id})
            if cursor is None:
                return {"entries": [_raw(7)], "next_cursor": "C7", "eof": False}
            return {"entries": [], "next_cursor": cursor, "eof": False}

    pages = cached_log_pages(cache, Client(), "j", "sub", follow=True, poll_interval=0.0, can_restart=lambda: True)
    assert next(pages) == ("restart", [_raw(7)])
    pages.close()
    assert calls[0]["cursor"] == "C5"
    # The read of the log's end, then one more read of the stuck cursor that confirms it is stuck.
    assert calls[-2:] == [{"cursor": None, "max_lines": RESTART_LINES[0], "sub_job_id": "sub"},
                          {"cursor": "C5", "max_lines": None, "sub_job_id": "sub"}]
    assert cache.get_cursor("sub") == "C7"


class _GrowingLog:
    """A pod log the tail reads by cursor (an index), or from its end without one."""

    def __init__(self, clock, lines):
        self.clock = clock
        self.lines = list(lines)
        self.restarts = []

    def fetch(self, cursor):
        start = int(cursor[1:]) if cursor else max(0, len(self.lines) - 2000)
        page = self.lines[start:start + 2000]
        return {"entries": page, "next_cursor": f"C{start + len(page)}"}

    def tail(self, lines):
        self.restarts.append(lines)
        return {"entries": self.lines[-lines:], "next_cursor": f"C{len(self.lines)}"}


def test_a_quiet_log_that_wakes_just_before_a_restart_shows_every_line_in_order(tmp_path):
    # A quiet log is read from its end every ~45 s. A burst that lands just before
    # one of those reads is found there first, but the cursor still brings it:
    # every line arrives in order, as a regular page, and nothing is marked missing.
    clock = _Clock()
    log = _GrowingLog(clock, [_raw(n) for n in range(5)])
    burst = [{"_raw": f"2026-10-14T12:05:00.{n:09d}Z burst {n}"} for n in range(500)]

    def restart(lines):
        if len(log.restarts) == 2:  # the third read of the end: the burst has just landed
            log.lines.extend(burst)
        return log.tail(lines)

    cache = LogCache("j", root=tmp_path)
    out = list(cached_pages(cache, "sub", log.fetch, entries_key="entries", follow=True, poll_interval=1.0,
                            sleep=clock.sleep, is_cancelled=lambda: clock() > 1400.0, restart=restart,
                            can_restart=lambda: True, clock=clock))
    assert [kind for kind, _ in out] == ["live", "live"]
    assert out[1][1] == burst  # all 500, in order
    assert len(log.restarts) >= 3
    assert cache.load_entries("sub") == [_raw(n) for n in range(5)] + burst


def test_a_stuck_tail_still_restarts_when_the_end_has_new_lines(tmp_path):
    # The oversized-line case: the cursor brings nothing even when asked again.
    clock = _Clock()
    log = _GrowingLog(clock, [_raw(n) for n in range(5)])

    def fetch(cursor):
        return {"entries": [], "next_cursor": cursor} if cursor else log.fetch(cursor)

    def restart(lines):
        log.lines.append(_raw(len(log.lines)))
        return log.tail(lines)

    cache = LogCache("j", root=tmp_path)
    out = []
    for kind, entries in cached_pages(cache, "sub", fetch, entries_key="entries", follow=True, poll_interval=1.0,
                                      sleep=clock.sleep, is_cancelled=lambda: len(out) >= 2, restart=restart,
                                      can_restart=lambda: True, clock=clock):
        out.append((kind, entries))
    assert out == [("live", [_raw(n) for n in range(5)]), ("restart", [_raw(5)])]
    assert cache.get_cursor("sub") == "C6"


def test_no_stall_clock_while_the_job_starts_or_without_a_cursor(tmp_path):
    # 100 s of empty pages and an empty cursor while the job is placing; the job
    # then runs and its log fills. The first lines come from a regular fetch
    # (the 2,000-line tail without a cursor), not a restart.
    clock = _Clock()
    calls = []

    def fetch(cursor):
        calls.append(("fetch", cursor))
        if clock() > 1100.0:
            return {"entries": [_raw(n) for n in range(5)], "next_cursor": "C5"}
        return {"entries": [], "next_cursor": ""}

    def restart(lines):
        calls.append(("restart", lines))
        return {"entries": [_raw(n) for n in range(5)], "next_cursor": "C5"}

    cache = LogCache("j", root=tmp_path)
    out = []
    for kind, _entries in cached_pages(cache, "sub", fetch, entries_key="entries", follow=True, poll_interval=1.0,
                                       sleep=clock.sleep, is_cancelled=lambda: len(out) >= 1, restart=restart,
                                       can_restart=lambda: clock() > 1100.0, clock=clock):
        out.append(kind)
    assert out == ["live"]
    assert not any(kind == "restart" for kind, _ in calls)

    # Running, but still without a cursor (an empty log): never a restart either.
    calls.clear()
    clock2 = _Clock()
    list(cached_pages(LogCache("j2", root=tmp_path), "sub", lambda cursor: calls.append(("fetch", cursor)) or
                      {"entries": [], "next_cursor": ""}, entries_key="entries", follow=True, poll_interval=1.0,
                      sleep=clock2.sleep, is_cancelled=lambda: clock2() > 1300.0, restart=restart,
                      can_restart=lambda: True, clock=clock2))
    assert calls and not any(kind == "restart" for kind, _ in calls)
