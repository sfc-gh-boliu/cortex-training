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

"""The log cache knows when it lacks a log's first lines: the tail's first
fetch read only the log's end, its replay holds only the newest entries, or
rotation dropped old ones."""

import pytest

from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import cached_log_pages

SOURCE = "JOB:training:0"


class _Tail:
    """tail_logs answering a no-cursor fetch with ``first`` lines (in pages of ``page``), then nothing new."""

    def __init__(self, first, page=1200):
        self.calls = []
        self._left = [{"_raw": f"ts line {i}"} for i in range(first)]
        self._page = page

    def tail_logs(self, job_id, cursor=None, sub_job_id=None, max_lines=None):
        self.calls.append({"cursor": cursor, "max_lines": max_lines})
        entries, self._left = self._left[:self._page], self._left[self._page:]
        return {"entries": entries, "next_cursor": f"c{len(self.calls)}" if entries else cursor, "eof": False}


def _pages(cache, client, *, polls):
    return list(cached_log_pages(cache, client, "JOB", SOURCE, follow=True, poll_interval=0.0, live=True,
                                 is_cancelled=lambda: len(client.calls) >= polls))


def test_a_first_fetch_that_reads_a_full_window_reports_and_remembers_a_missing_start(tmp_path):
    cache = LogCache("JOB", root=tmp_path)
    client = _Tail(2000)  # the window is full: the log goes back further
    kinds = [kind for kind, _ in _pages(cache, client, polls=3)]
    assert client.calls[0]["max_lines"] == 2000
    assert kinds == ["live", "partial", "live"]  # said once the window filled, across two pages
    assert cache.head_missing(SOURCE)
    # The next open replays the cache, and says the start is missing at once.
    again = LogCache("JOB", root=tmp_path)
    kinds = [kind for kind, _ in cached_log_pages(again, _Tail(0), "JOB", SOURCE, follow=False, poll_interval=0.0,
                                                    live=False)]
    assert kinds == ["partial", "cache"]


def test_a_short_log_read_whole_is_not_partial(tmp_path):
    cache = LogCache("JOB", root=tmp_path)
    client = _Tail(300)
    kinds = [kind for kind, _ in _pages(cache, client, polls=3)]
    assert kinds == ["live"] and not cache.head_missing(SOURCE)


class _LiveLog:
    """tail-logs over a log that grows: without a cursor, ``max_lines`` reads the last
    lines (at most 2,000); a page holds at most 2,000 entries and 256 KiB; the cursor
    is the next line's index. ``grow`` lines are written after every answer."""

    def __init__(self, lines, *, grow=0, width=0):
        self.n, self.grow, self.width = lines, grow, width
        self.calls = []

    def line(self, i):
        return {"_raw": f"ts line {i:06d}" + " x" * (self.width // 2)}

    def tail_logs(self, job_id, cursor=None, sub_job_id=None, max_lines=None):
        self.calls.append({"cursor": cursor, "max_lines": max_lines})
        start = int(cursor) if cursor else max(0, self.n - min(max_lines or self.n, 2000))
        entries, size = [], 0
        for i in range(start, self.n):
            entry = self.line(i)
            if len(entries) == 2000 or size + len(entry["_raw"]) > 256 * 1024:
                break
            entries.append(entry)
            size += len(entry["_raw"])
        self.n += self.grow
        return {"entries": entries, "next_cursor": str(start + len(entries)), "eof": False}


@pytest.mark.parametrize("lines,grow,width", [
    (120, 20, 0),     # a job that just started, writing before every read
    (1500, 20, 0),    # a first page of 1,500 entries: full, and still the whole log
    (1200, 20, 300),  # long lines: the window comes in two pages, the second with new lines
    (900, 400, 0),    # a busy log: 400 new lines a read, the window read whole on the first page
])
def test_a_short_log_that_keeps_writing_is_read_whole(tmp_path, lines, grow, width):
    cache = LogCache("JOB", root=tmp_path)
    client = _LiveLog(lines, grow=grow, width=width)
    kinds = [kind for kind, _ in _pages(cache, client, polls=150)]
    assert "partial" not in kinds and not cache.head_missing(SOURCE)
    entries = cache.load_entries(SOURCE)
    assert entries[0] == client.line(0)  # the pane has the log's first line
    assert len(entries) > 2000  # past the window's size, which alone says nothing


@pytest.mark.parametrize("grow,width", [(0, 0), (20, 0), (20, 300), (400, 300)])
def test_a_long_log_says_its_start_is_missing_however_its_window_comes(tmp_path, grow, width):
    cache = LogCache("JOB", root=tmp_path)
    client = _LiveLog(50_000, grow=grow, width=width)
    pages = _pages(cache, client, polls=10)
    assert [kind for kind, _ in pages].count("partial") == 1
    assert cache.head_missing(SOURCE)
    assert cache.load_entries(SOURCE)[0] == client.line(48_000)  # the window: the last 2,000 lines
    assert client.calls[0] == {"cursor": None, "max_lines": 2000}


def test_a_replay_over_its_limit_is_partial(tmp_path):
    cache = LogCache("JOB", root=tmp_path)
    cache.append_entries(SOURCE, [{"_raw": f"ts {i}"} for i in range(30)])
    pages = list(cached_log_pages(cache, _Tail(0), "JOB", SOURCE, follow=False, poll_interval=0.0, live=False,
                                  replay_limit=20))
    assert [kind for kind, _ in pages] == ["partial", "cache"]
    assert [e["_raw"] for e in pages[1][1]] == [f"ts {i}" for i in range(10, 30)]
    assert cache.load_newest(SOURCE, limit=20)[1] == 30


def test_the_cursor_update_keeps_what_the_cache_knows(tmp_path):
    cache = LogCache("JOB", root=tmp_path)
    cache.set_head_missing(SOURCE)
    cache.set_cursor(SOURCE, "c9")
    assert cache.get_cursor(SOURCE) == "c9" and cache.head_missing(SOURCE)


def test_rotation_that_drops_old_lines_marks_the_start_missing(tmp_path):
    cache = LogCache("JOB", root=tmp_path, max_bytes=2000, max_lines=50)
    cache.append_entries(SOURCE, [{"_raw": f"ts line {i:04d}"} for i in range(200)])
    entries, total = cache.load_newest(SOURCE)
    assert total == len(entries) <= 50 and entries[-1]["_raw"] == "ts line 0199"
    assert cache.head_missing(SOURCE)


# ─── R1-49: a cursor the service refuses ─────────────────────────────────────

class _Refused(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.response = type("Response", (), {"status_code": status, "json": lambda _self: {"message": message}})()


def test_a_refused_cursor_is_dropped_and_the_tail_goes_on_from_the_end(tmp_path):
    cache = LogCache("JOB", root=tmp_path)
    cache.append_entries(SOURCE, [{"_raw": "ts old 1"}])
    cache.set_cursor(SOURCE, "stale")
    calls = []

    def tail_logs(job_id, cursor=None, sub_job_id=None, max_lines=None):
        calls.append(cursor)
        if cursor == "stale":
            raise _Refused(400, "invalid cursor")
        if cursor is None:
            return {"entries": [{"_raw": "ts old 1"}, {"_raw": "ts new 2"}], "next_cursor": "fresh"}
        return {"entries": [], "next_cursor": cursor}

    client = type("Client", (), {"tail_logs": staticmethod(tail_logs)})()
    pages = list(cached_log_pages(cache, client, "JOB", SOURCE, follow=True, poll_interval=0.0, live=True,
                                  can_restart=lambda: True, is_cancelled=lambda: len(calls) >= 4))
    assert [kind for kind, _ in pages] == ["cache", "reset"]
    assert [e["_raw"] for e in pages[1][1]] == ["ts new 2"]  # the line it had already is left out
    assert calls[:3] == ["stale", None, "fresh"]  # never sent again
    assert cache.get_cursor(SOURCE) == "fresh"


def test_a_not_routed_answer_is_not_taken_for_a_refused_cursor(tmp_path):
    cache = LogCache("JOB", root=tmp_path)
    cache.set_cursor(SOURCE, "c1")

    def tail_logs(job_id, cursor=None, sub_job_id=None, max_lines=None):
        raise _Refused(400, "sub_job_id JOB:training:0 does not belong to session JOB")

    client = type("Client", (), {"tail_logs": staticmethod(tail_logs)})()
    try:
        list(cached_log_pages(cache, client, "JOB", SOURCE, follow=True, poll_interval=0.0, live=True))
    except _Refused:
        pass
    else:
        raise AssertionError("the not-routed answer must reach the caller")
    assert cache.get_cursor(SOURCE) == "c1"


# ─── R1-50: a log that runs ahead of the tail ────────────────────────────────

def _catch_up_run(tmp_path, sizes):
    from cortex_training.tui import log_cache as lc

    cache = LogCache("JOB", root=tmp_path)
    cache.set_cursor(SOURCE, "c0")
    sizes = iter(sizes)
    sleeps = []

    def fetch(cursor):
        return {"entries": [{"_raw": f"ts {cursor} {i}"} for i in range(next(sizes, 0))], "next_cursor": f"{cursor}+"}

    kinds = [kind for kind, _ in lc.cached_pages(cache, SOURCE, fetch, entries_key="entries", follow=True,
                                                 poll_interval=1.0, sleep=sleeps.append,
                                                 is_cancelled=lambda: len(sleeps) >= 5)]
    return kinds, sleeps


def test_full_pages_are_followed_sooner_until_the_tail_catches_up(tmp_path):
    from cortex_training.tui.log_cache import CATCH_UP_SECONDS

    kinds, sleeps = _catch_up_run(tmp_path / "a", [1500, 1500, 10, 0, 0])
    assert kinds == ["live", "live", "live"]  # the page that isn't full ends the catch-up itself
    assert sleeps[:3] == [CATCH_UP_SECONDS, CATCH_UP_SECONDS, 1.0]  # sooner only after a full page
    kinds, sleeps = _catch_up_run(tmp_path / "b", [1500, 0, 0])
    assert kinds == ["live", "caught-up"]  # an empty page right after a full one says it caught up
    assert sleeps[0] == CATCH_UP_SECONDS and sleeps[1] == 1.0


# ─── R1-56: kept saved consoles stay within a bound ──────────────────────────

def _kept_console(root, job, size, used):
    import os

    console = root / "jobs" / job / "console"
    (console / f"{job}:training:0").mkdir(parents=True)
    (console / f"{job}:training:0" / "stdout.log").write_bytes(b"x" * size)
    marker = console / ".complete"
    marker.write_text("created", encoding="utf-8")
    os.utime(marker, (used, used))
    return console


def test_kept_consoles_past_the_bound_go_least_recently_used_first(tmp_path):
    from cortex_training.tui.log_cache import prune_consoles
    from cortex_training.tui.log_cache import touch_console

    old = _kept_console(tmp_path, "job-old", 600, 1_000)
    mid = _kept_console(tmp_path, "job-mid", 600, 2_000)
    new = _kept_console(tmp_path, "job-new", 600, 3_000)
    writing = tmp_path / "jobs" / "job-writing" / "console"  # no marker yet: being written
    writing.mkdir(parents=True)
    (writing / "big").write_bytes(b"x" * 5000)
    touch_console(old)  # opened just now: the most recently used
    removed = prune_consoles(root=tmp_path, keep=new, max_bytes=1300)
    assert removed == [mid]
    assert old.exists() and new.exists() and writing.exists()
    assert prune_consoles(root=tmp_path, keep=new, max_bytes=100) == [old]  # never the one just kept
    assert new.exists()
