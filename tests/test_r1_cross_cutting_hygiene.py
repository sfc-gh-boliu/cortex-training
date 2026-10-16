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

"""One error log, public shared helpers, a public read-only switch, and one owner per line."""

import re
from pathlib import Path

import pytest

from cortex_training import CortexTrainingClient
from cortex_training.tui import log_cache
from cortex_training.tui.__main__ import read_only_client
from tests.r1_cross_cutting_support import FakeClient
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401
from tests.r1_cross_cutting_support import job
from tests.r1_cross_cutting_support import run
from tests.r1_cross_cutting_support import settle
from tests.r1_cross_cutting_support import wait_for

TUI = Path(__file__).resolve().parents[1] / "src" / "cortex_training" / "tui"


def test_every_traceback_goes_to_the_one_error_log(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    from cortex_training.tui.app import LogScreen

    elsewhere = tmp_path / "another-home"
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(elsewhere))  # the log is the module's path, not one expanded on each write
    target = tmp_path / "errors.log"
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(target))
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        LogScreen._log_traceback("a failed read")
    assert "a failed read:" in target.read_text() and "RuntimeError: boom" in target.read_text()
    assert not (elsewhere / ".cortex-training-errors.log").exists()
    assert log_cache.error_log_path() == str(target)


def test_the_views_and_resources_use_only_public_format_helpers():
    for module in ("views.py", "resource.py"):
        source = (TUI / module).read_text()
        assert not re.search(r"from cortex_training\.tui\.format import _", source), module


def test_the_client_has_a_public_read_only_switch():
    client = CortexTrainingClient(base_url="http://127.0.0.1:9", database="DB", schema="S")
    assert client.auto_create_database is True
    assert read_only_client(client) is client and client.auto_create_database is False
    assert client.for_job({"database_name": "TEAM_DB", "schema_name": "EXP"}).auto_create_database is False


def test_a_job_opened_directly_never_flashes_the_config_summary(monkeypatch):
    pytest.importorskip("textual")
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.screens import JobDetailScreen

    shown = []
    real = JobDetailScreen._set_summary

    def record(self, text):
        shown.append(text)
        return real(self, text)

    monkeypatch.setattr(JobDetailScreen, "_set_summary", record)

    async def go():
        app = CortexTrainingLogTUI(FakeClient([job("j1")]), "j1", poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._sources_ready)
            for _ in range(5):
                await pilot.pause()
            assert shown and not any(isinstance(text, str) for text in shown)  # only its own line, never the config
            await settle(app, pilot)

    run(go())


def test_copy_all_lands_through_the_screen_and_says_so(monkeypatch):
    pytest.importorskip("textual")
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.screens import JobDetailScreen

    class Logged(FakeClient):
        def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
            if cursor:
                return {"entries": [], "next_cursor": cursor, "eof": False}
            return {"entries": [{"_raw": "2026-10-14T12:00:00.000000000Z hello"}], "next_cursor": "1", "eof": False}

    async def go():
        app = CortexTrainingLogTUI(Logged([job("j1")]), "j1", poll_interval=0.01)
        copied = []
        app.copy_to_clipboard = copied.append
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                  and any("hello" in line for line in app.screen._shown_lines))
            await pilot.press("y")
            assert await wait_for(pilot, lambda: copied)
            assert "hello" in copied[0]
            # Said on the status line (the log pane keeps only log lines).
            bar = app.screen.query_one("#log-status")
            assert await wait_for(pilot, lambda: "sent 1 line to the terminal's clipboard" in str(bar.render()))
            await settle(app, pilot)

    run(go())
