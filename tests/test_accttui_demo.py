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

"""The demo's account-wide list: several locations, pages with tokens, per-location views, and its fallback."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.demo import DemoClient  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.stores import AliasStore  # noqa: E402

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_tui_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(tui_main.DEMO_ACCOUNT_LIST_ENV, raising=False)
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _walk(demo, **filters) -> list:
    rows, token, pages = [], None, 0
    while True:
        page = demo.list_jobs_page(page_size=500, page_token=token, **filters)
        pages += 1
        rows += page["jobs"]
        token = page["next_page_token"]
        if token is None:
            return rows, pages


# ─── the demo client ─────────────────────────────────────────────────────────


def test_the_account_wide_list_has_three_locations_in_pages_newest_first():
    demo = DemoClient(now=NOW)
    rows, pages = _walk(demo)
    assert pages > 1 and all(len(demo.list_jobs_page(page_size=3)["jobs"]) <= 3 for _ in range(1))
    where = Counter((r["database_name"], r["schema_name"]) for r in rows)
    assert where == {("CORTEX_TRAINING_DB", "PUBLIC"): 30, ("TEAM_DB", "EXPERIMENTS"): 3, ("sandbox", "dev"): 2}
    assert {views.location_label(r) for r in rows} == {
        "CORTEX_TRAINING_DB.PUBLIC", "TEAM_DB.EXPERIMENTS", '"sandbox"."dev"'}
    created = [r["created_at"] for r in rows]
    assert created == sorted(created, reverse=True)
    # The configured schema's own list: its jobs only, without location fields.
    served = demo.list_jobs()
    assert len(served) == 30 and not any("database_name" in j for j in served)
    assert list(demo.iter_jobs()) == rows


def test_the_account_wide_list_applies_created_after_and_keeps_unknown_rows_dated():
    demo = DemoClient(now=NOW)
    rows, _ = _walk(demo, created_after=NOW - timedelta(days=183))
    assert len(rows) == 33  # the two jobs older than six months are left out
    unknown = [r for r in rows if r["status"] == "UNKNOWN"]
    assert unknown and all(r["created_at"] and not any("sub_job_id" in s for s in r["sub_jobs"]) for r in unknown)
    # A status filter leaves UNKNOWN out, as on the service.
    active, _ = _walk(demo, status="active")
    assert active and not any(r["status"] == "UNKNOWN" for r in active)
    assert {r["status"] for r in active} <= {"PENDING", "PLACING", "INITIALIZING", "RUNNING"}
    lower, _ = _walk(demo, database='"sandbox"', schema="DEV")
    assert lower == []  # unquoted DEV means the uppercase name
    lower, _ = _walk(demo, database='"sandbox"', schema='"dev"')
    assert len(lower) == 2


def test_a_page_token_only_opens_with_its_own_filters():
    demo = DemoClient(now=NOW)
    token = demo.list_jobs_page(page_size=5)["next_page_token"]
    assert token
    assert demo.list_jobs_page(page_size=50, page_token=token)["jobs"]  # the page size may change
    with pytest.raises(ValueError, match="does not match the request filters"):
        demo.list_jobs_page(status="running", page_token=token)


def test_for_job_returns_a_view_of_that_location():
    demo = DemoClient(now=NOW)
    rows, _ = _walk(demo)
    lower = next(r for r in rows if r["database_name"] == "sandbox")
    view = demo.for_job(lower)
    assert (view.database, view.schema) == ('"sandbox"', '"dev"')
    assert view.get_job(lower["job_id"])["comment"] == lower["comment"]
    assert {j["job_id"] for j in view.list_jobs()} == {r["job_id"] for r in rows if r["database_name"] == "sandbox"}
    with pytest.raises(KeyError):
        demo.get_job(lower["job_id"])  # not in the configured location
    home = next(r for r in rows if r["database_name"] == "CORTEX_TRAINING_DB")
    with pytest.raises(KeyError):
        view.get_job(home["job_id"])
    assert (demo.database, demo.schema) == ("CORTEX_TRAINING_DB", "PUBLIC")  # the base is unchanged


@pytest.mark.parametrize("reason", sorted(AccountJobListUnavailable.REASONS))
def test_the_demo_can_answer_as_an_unavailable_account_wide_list(reason):
    demo = DemoClient(now=NOW, account_list=reason)
    with pytest.raises(AccountJobListUnavailable) as error:
        demo.list_jobs_page()
    assert error.value.reason == reason and "account-level job listing" in str(error.value)
    assert len(demo.list_jobs()) == 30  # the configured schema still lists


def test_the_demo_rejects_an_unknown_reason():
    with pytest.raises(ValueError):
        DemoClient(account_list="sometimes")


# ─── the switch and the command line ─────────────────────────────────────────


@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("ok", None), ("OK", None), ("not_deployed", "not_deployed"),
    (" Disabled ", "disabled"), ("forbidden", "forbidden"), ("rest_disabled", "rest_disabled"),
])
def test_the_demo_switch(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(tui_main.DEMO_ACCOUNT_LIST_ENV, value)
    assert tui_main._demo_account_list() == expected


def test_a_bad_demo_switch_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setenv(tui_main.DEMO_ACCOUNT_LIST_ENV, "flaky")
    with pytest.raises(SystemExit) as stop:
        tui_main.run(["--demo"])
    assert stop.value.code == 2 and tui_main.DEMO_ACCOUNT_LIST_ENV in capsys.readouterr().err


@pytest.mark.parametrize("env, flags, account_list, schema_only", [
    (None, [], None, False),
    ("not_deployed", [], "not_deployed", False),
    (None, ["--schema-only"], None, True),
])
def test_demo_runs_pass_the_switch_and_schema_only_on(monkeypatch, env, flags, account_list, schema_only):
    if env:
        monkeypatch.setenv(tui_main.DEMO_ACCOUNT_LIST_ENV, env)
    seen = {}

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, schema_only=False):
            seen.update(client=client, schema_only=schema_only)

        def run(self):
            pass

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    assert tui_main.run(["--demo", *flags]) == 0
    assert isinstance(seen["client"], DemoClient)
    assert seen["client"]._account_list == account_list and seen["schema_only"] is schema_only


# ─── the demo in the app ─────────────────────────────────────────────────────


async def _wait_for(pilot, predicate, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.02)
    return bool(predicate())


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _labels(screen) -> list:
    return [str(column.label) for column in screen.query_one("#jobs", DataTable).columns.values()]


def _scope(app) -> str:
    """The list's scope in the subtitle: after the refresh status, before the key hints."""
    parts = (app.sub_title or "").split("  ·  ")
    return parts[1] if len(parts) == 3 else ""


def test_the_demo_lists_every_location_and_opens_a_job_in_a_quoted_one():
    demo = DemoClient()
    demo.seed_local(aliases=AliasStore())
    target = next(r for r in demo.iter_jobs() if r["database_name"] == "sandbox" and r["status"] == "RUNNING")

    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(200, 50)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            assert len(screen._job_by_item) == 33 and "Location" in _labels(screen)
            assert _scope(app) == "all schemas"
            key = next(item for item, jid in screen._job_by_item.items() if jid == target["job_id"])
            table = screen.query_one("#jobs", DataTable)
            table.move_cursor(row=table.get_row_index(key), animate=False)
            screen.set_focus(table)
            await pilot.press("enter")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert app.screen._client.schema == '"dev"'
            await pilot.press("l")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                   and any("generate batch" in line for line in app.screen._shown_lines))
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_demo_shows_the_fallback_to_the_configured_schema():
    demo = DemoClient(account_list="not_deployed")

    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(200, 50)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            assert _scope(app) == "this schema only: CORTEX_TRAINING_DB.PUBLIC (no account-wide list yet)"
            assert len(screen._job_by_item) == 28 and "Location" not in _labels(screen)
            await _settle(app, pilot)

    asyncio.run(run())
