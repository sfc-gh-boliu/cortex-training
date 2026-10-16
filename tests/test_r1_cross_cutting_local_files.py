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

"""A local cache file damaged outside the client (bytes that aren't UTF-8) never stops the TUI.

The client writes its metrics and log caches as ASCII JSON, so only external
corruption (a disk error, an editor, a sync tool) puts such bytes there; a
corrupt line then reads like a torn one instead of raising UnicodeDecodeError.
"""

import pytest

from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.stores import MetricsStore
from tests.r1_cross_cutting_support import FakeClient
from tests.r1_cross_cutting_support import NOW
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401
from tests.r1_cross_cutting_support import job
from tests.r1_cross_cutting_support import run
from tests.r1_cross_cutting_support import settle
from tests.r1_cross_cutting_support import wait_for

BAD_LINE = b'{"ts":"2026-10-14T00:00:00Z","train_loss":2.0,"note":"\xff\xfe"}\n'


def _damage(path, line=BAD_LINE):
    with open(path, "ab") as handle:
        handle.write(line)


def test_metrics_with_bytes_that_are_not_utf8_still_load(tmp_path):
    store = MetricsStore(tmp_path)
    store.append("j1", {"train_loss": 1.0}, step=1)
    _damage(store.path("j1"))
    _damage(store.path("j1"), b"\xff\xfe not json at all\n")
    store.append("j1", {"train_loss": 0.5}, step=2)
    records = store.load("j1")
    assert [r["train_loss"] for r in records] == [1.0, 2.0, 0.5]
    assert records[1]["note"] == "��"


def test_a_log_cache_with_bytes_that_are_not_utf8_still_replays_and_rotates(tmp_path):
    cache = LogCache("j1", root=tmp_path, max_bytes=200, max_lines=3)
    cache.append_entries("j1:training:0", [{"_raw": "2026 a"}])
    _damage(cache._jsonl_path("j1:training:0"), b'{"_raw":"\xff b"}\n')
    assert [e["_raw"] for e in cache.load_entries("j1:training:0")] == ["2026 a", "� b"]
    # Past max_bytes an append rotates the file, reading the damaged line too.
    cache.append_entries("j1:training:0", [{"_raw": "x" * 120}, {"_raw": "2026 c"}])
    assert cache.load_entries("j1:training:0")[-1] == {"_raw": "2026 c"}


def test_a_kept_console_marker_that_is_not_utf8_is_not_a_kept_console(tmp_path):
    pytest.importorskip("textual")
    from cortex_training.tui.app import LogScreen

    screen = LogScreen(FakeClient([]), "j1", job_status="FAILED")
    screen._job_created_at = "2026-10-14T00:00:00Z"
    screen._console_dir.mkdir(parents=True)
    (screen._console_dir / ".complete").write_bytes(b"\xff\xfe")
    assert screen._kept_console() is False


def test_the_preview_and_the_metrics_tab_open_with_damaged_local_metrics():
    pytest.importorskip("textual")
    from textual.widgets import DataTable

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen
    from cortex_training.tui.screens import JobDetailScreen
    from cortex_training.tui.screens import JobSummaryScreen

    store = MetricsStore()  # the isolated cache dir
    for step in range(5):
        store.append("j1", {"train_loss": 2.0 - step / 10}, step=step)
    _damage(store.path("j1"))

    async def go():
        app = CortexTrainingLogTUI(FakeClient([job("j1")]), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            table = app.screen.query_one("#jobs", DataTable)
            app.screen.set_focus(table)
            table.move_cursor(row=1, animate=False)
            await pilot.pause()
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen)
                                  and app.screen._metric_catalog is not None)
            assert {m.key for m in app.screen._metric_catalog} == {"train_loss"}
            await pilot.press("m")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                  and len(app.screen._metric_records) == 6)
            assert app.is_running
            await settle(app, pilot)

    run(go())
