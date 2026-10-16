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

"""The job list opens on the configured schema; ``a`` switches to the account-wide list and back.

``--schema-only`` hides ``a``; an account-wide list that is unavailable when
``a`` asks for it says why, and the list stays on the configured schema.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets._footer import FooterKey  # noqa: E402

from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui import account  # noqa: E402
from cortex_training.tui.account import ACCOUNT  # noqa: E402
from cortex_training.tui.account import FALLBACK  # noqa: E402
from cortex_training.tui.account import SCHEMA  # noqa: E402
from cortex_training.tui.account import SCHEMA_ONLY  # noqa: E402
from cortex_training.tui.account import JobLister  # noqa: E402
from cortex_training.tui.account import ListScope  # noqa: E402
from cortex_training.tui.account import scope_text  # noqa: E402
from cortex_training.tui.account import unavailable_text  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from tests.v2_joblist_support import client  # noqa: E402
from tests.v2_joblist_support import footer  # noqa: E402
from tests.v2_joblist_support import http_error  # noqa: E402
from tests.v2_joblist_support import isolated_tui_files  # noqa: E402,F401
from tests.v2_joblist_support import job  # noqa: E402
from tests.v2_joblist_support import notices  # noqa: E402
from tests.v2_joblist_support import page  # noqa: E402
from tests.v2_joblist_support import scope  # noqa: E402
from tests.v2_joblist_support import settle  # noqa: E402
from tests.v2_joblist_support import unavailable  # noqa: E402
from tests.v2_joblist_support import wait  # noqa: E402

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
HERE = "CORTEX_TRAINING_DB.PUBLIC"
CLI_MD = Path(__file__).resolve().parents[1] / "docs" / "reference" / "cli.md"
REST_MD = Path(__file__).resolve().parents[1] / "docs" / "reference" / "rest-api.md"
SCHEMA_ROWS = [job("s1", "RUNNING", minutes_ago=5), job("s2")]
EVERY = page(job("t1", "RUNNING", db="TEAM_DB", schema="EXPERIMENTS", minutes_ago=3),
             job("m1", db="ml", schema="dev", minutes_ago=20), job("s1", db="CORTEX_TRAINING_DB", schema="PUBLIC"))


async def _list(app, pilot) -> JobListScreen:
    assert await wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._loaded
                      and not app.screen._refreshing)
    return app.screen


def _shown(screen) -> set:
    return set(screen._job_by_item.values())


def _labels(screen) -> list:
    return [str(column.label) for column in screen.query_one("#jobs", DataTable).columns.values()]


# ─── the lister ──────────────────────────────────────────────────────────────


def test_the_lister_reads_the_configured_schema_until_the_account_wide_list_is_asked_for():
    c = client(SCHEMA_ROWS, pages=[EVERY, EVERY])
    lister = JobLister(c)
    assert lister.account_wide is False
    listing = lister.load(NOW)
    assert listing.scope == ListScope(SCHEMA) and [j["job_id"] for j in listing.jobs] == ["s1", "s2"]
    c.list_jobs_page.assert_not_called()
    # The screen passes the list it asked for; without it, the lister's own choice counts.
    assert lister.load(NOW, account=True).scope == ListScope(ACCOUNT)
    lister.account_wide = True
    assert lister.load(NOW).scope == ListScope(ACCOUNT)
    assert lister.load(NOW, account=False).scope == ListScope(SCHEMA)
    assert c.list_jobs_page.call_count == 2 and c.list_jobs.call_count == 2


def test_schema_only_wins_over_the_account_wide_choice():
    c = client(SCHEMA_ROWS)
    lister = JobLister(c, schema_only=True, account_wide=True)
    assert lister.load(NOW).scope == ListScope(SCHEMA_ONLY)
    assert lister.load(NOW, account=True).scope == ListScope(SCHEMA_ONLY)
    c.list_jobs_page.assert_not_called()


def test_the_header_names_the_list_and_the_reasons_are_the_existing_ones():
    assert scope_text(ListScope(SCHEMA), location=HERE) == f"this schema: {HERE}"
    assert scope_text(ListScope(ACCOUNT), location=HERE) == "all schemas"
    assert scope_text(ListScope(SCHEMA_ONLY), location=HERE) == f"this schema only: {HERE} (--schema-only)"
    assert {reason: unavailable_text(ListScope(FALLBACK, reason=reason))
            for reason in ("not_deployed", "disabled", "forbidden", "rest_disabled", None)} == {
        "not_deployed": "no account-wide list yet", "disabled": "account-wide list not enabled",
        "forbidden": "account-wide list: HTTP 403", "rest_disabled": "account-wide list: HTTP 405",
        None: "no account-wide list"}


# ─── the list opens on the configured schema ─────────────────────────────────


def test_the_list_opens_on_the_configured_schema_and_says_so():
    c = client(SCHEMA_ROWS)  # the account-wide list must not be asked

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            assert _shown(s) == {"s1", "s2"} and "Location" not in _labels(s)
            assert scope(app) == f"this schema: {HERE}"
            assert footer(s).get("a") == "All schemas"
            assert "a all schemas" in app.sub_title
            await pilot.press("r")
            assert await wait(pilot, lambda: c.list_jobs.call_count == 2 and not s._refreshing)
            assert scope(app) == f"this schema: {HERE}"
            await settle(app, pilot)

    asyncio.run(run())
    c.list_jobs_page.assert_not_called()


# ─── a switches lists ────────────────────────────────────────────────────────


def test_a_switches_to_every_schema_and_back():
    c = client(SCHEMA_ROWS, pages=[EVERY, EVERY])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            s = await _list(app, pilot)
            await pilot.press("a")
            assert await wait(pilot, lambda: _shown(s) == {"t1", "m1", "s1"} and not s._refreshing)
            assert scope(app) == "all schemas" and "Location" in _labels(s)
            assert footer(s).get("a") == "This schema" and "a this schema" in app.sub_title
            assert s._lister.account_wide and not notices(app)
            await pilot.press("r")  # the account-wide list again
            assert await wait(pilot, lambda: c.list_jobs_page.call_count == 2 and not s._refreshing)
            assert c.list_jobs.call_count == 1
            await pilot.press("a")
            assert await wait(pilot, lambda: _shown(s) == {"s1", "s2"} and not s._refreshing)
            assert scope(app) == f"this schema: {HERE}" and "Location" not in _labels(s)
            assert footer(s).get("a") == "All schemas" and not s._lister.account_wide
            assert c.list_jobs.call_count == 2 and c.list_jobs_page.call_count == 2
            await settle(app, pilot)

    asyncio.run(run())


def test_the_footer_key_is_the_button():
    c = client(SCHEMA_ROWS, pages=[EVERY])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            key = next(k for k in s.query(FooterKey) if k.key == "a")
            assert "All schemas" in str(key.render())
            await pilot.click(key)
            assert await wait(pilot, lambda: _shown(s) == {"t1", "m1", "s1"} and not s._refreshing)
            assert await wait(pilot, lambda: any(k.key == "a" and "This schema" in str(k.render())
                                                 for k in s.query(FooterKey)))
            await pilot.click(next(k for k in s.query(FooterKey) if k.key == "a"))
            assert await wait(pilot, lambda: _shown(s) == {"s1", "s2"} and not s._refreshing)
            await settle(app, pilot)

    asyncio.run(run())


def test_a_typed_in_the_search_box_is_part_of_the_search():
    c = client(SCHEMA_ROWS)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            await pilot.press("/", "a")
            assert await wait(pilot, lambda: s._filter == "a")
            assert not s._lister.account_wide and not s._refreshing
            await settle(app, pilot)

    asyncio.run(run())
    c.list_jobs_page.assert_not_called()


def test_schema_only_hides_a_and_a_does_nothing():
    c = client(SCHEMA_ROWS)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01, schema_only=True)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            assert "a" not in footer(s) and "a" not in s.active_bindings
            assert "· a " not in app.sub_title and "all schemas" not in app.sub_title
            assert scope(app) == f"this schema only: {HERE} (--schema-only)"
            await pilot.press("a")
            for _ in range(5):
                await pilot.pause()
            assert not s._refreshing and c.list_jobs.call_count == 1 and not s._lister.account_wide
            assert not notices(app)
            await settle(app, pilot)

    asyncio.run(run())
    c.list_jobs_page.assert_not_called()


def test_the_cli_flag_reaches_the_app_and_its_help_says_what_it_does(capsys):
    with pytest.raises(SystemExit):
        tui_main.run(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "List only the configured database and schema's jobs, and never the account-wide list" in text
    assert "'a' switches it to every database and schema" in text


# ─── unavailable when a asks ─────────────────────────────────────────────────

_SERVICE = {"code": "510201", "message": "Cortex Training is not enabled for this account"}


@pytest.mark.parametrize("answer, why", [
    (unavailable("not_deployed"), "no account-wide list yet"),
    (unavailable("disabled"), "account-wide list not enabled"),
    # A 403 without the service's message: the configured schema is asked once, and answers.
    (unavailable("forbidden"), "account-wide list: HTTP 403"),
    # The service's own refusals: the error itself.
    (unavailable("forbidden", message="account-level job listing was refused (HTTP 403): Cortex Training is not "
                                      "enabled for this account", body=_SERVICE),
     "account-level job listing was refused (HTTP 403): Cortex Training is not enabled for this account"),
    (unavailable("rest_disabled", message="account-level job listing is unavailable: the Cortex Training REST API "
                                          "is turned off for this account (HTTP 405)", body=_SERVICE),
     "account-level job listing is unavailable: the Cortex Training REST API is turned off for this account "
     "(HTTP 405)"),
    (http_error(503, "Control Plane unavailable"),
     "the account-wide job list answered HTTP 503 Service Unavailable: Control Plane unavailable"),
])
def test_an_unavailable_account_wide_list_says_why_and_the_list_stays_on_the_schema(answer, why):
    c = client(SCHEMA_ROWS, pages=[answer])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            await pilot.press("a")
            assert await wait(pilot, lambda: bool(notices(app)) and not s._refreshing)
            assert notices(app) == [f"Can't show all schemas ({why}); the list stays on {HERE}."]
            assert next(n for n in app._notifications).severity == "warning"
            assert _shown(s) == {"s1", "s2"} and scope(app) == f"this schema: {HERE}"
            assert footer(s).get("a") == "All schemas" and not s._lister.account_wide
            assert not s._list_error  # the configured schema's list is fine: no "refresh failed"
            await settle(app, pilot)

    asyncio.run(run())
    assert c.list_jobs_page.call_count == 1


def test_a_within_a_minute_of_an_unavailable_answer_says_it_again_without_asking():
    clock = [1000.0]
    c = client(SCHEMA_ROWS, pages=[unavailable("not_deployed"), EVERY])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        app._lister = JobLister(c, clock=lambda: clock[0])
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _list(app, pilot)
            await pilot.press("a")
            assert await wait(pilot, lambda: len(notices(app)) == 1 and not s._refreshing)
            clock[0] += 30
            await pilot.press("a")
            assert await wait(pilot, lambda: len(notices(app)) == 2 and not s._refreshing)
            assert notices(app)[1] == notices(app)[0] and c.list_jobs_page.call_count == 1
            clock[0] += 31  # a minute since the last ask: a asks again, and the list is there now
            await pilot.press("a")
            assert await wait(pilot, lambda: _shown(s) == {"t1", "m1", "s1"} and not s._refreshing)
            assert scope(app) == "all schemas" and len(notices(app)) == 2
            await settle(app, pilot)

    asyncio.run(run())
    assert c.list_jobs_page.call_count == 2


# ─── a while a list is being read ────────────────────────────────────────────


def test_a_while_the_schema_list_is_read_shows_the_account_wide_list_once_it_answers():
    c = client(SCHEMA_ROWS, pages=[EVERY])
    release = threading.Event()
    rows = [dict(r) for r in SCHEMA_ROWS]
    c.list_jobs.side_effect = lambda **_: [dict(r) for r in rows] if release.wait(10) else []

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            assert await wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._refreshing)
            s = app.screen
            await pilot.press("a")  # the configured schema's list is still being read
            assert footer(s).get("a") == "This schema"
            release.set()
            assert await wait(pilot, lambda: _shown(s) == {"t1", "m1", "s1"} and not s._refreshing)
            assert scope(app) == "all schemas" and not notices(app)
            await settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert c.list_jobs.call_count == 1 and c.list_jobs_page.call_count == 1


def test_a_that_fails_before_any_list_arrived_reads_the_configured_schema():
    c = client(SCHEMA_ROWS, pages=[http_error(503, "Control Plane unavailable")])
    release = threading.Event()
    calls = []

    def list_jobs(**_):
        calls.append(1)
        if len(calls) == 1:
            release.wait(10)
        return [dict(r) for r in SCHEMA_ROWS]

    c.list_jobs.side_effect = list_jobs

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._refreshing)
            s = app.screen
            await pilot.press("a")
            release.set()  # the first answer is put aside; the account-wide list fails
            assert await wait(pilot, lambda: s._loaded and not s._refreshing)
            assert _shown(s) == {"s1", "s2"} and scope(app) == f"this schema: {HERE}"
            assert notices(app) == [f"Can't show all schemas (the account-wide job list answered HTTP 503 Service "
                                    f"Unavailable: Control Plane unavailable); the list stays on {HERE}."]
            await settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert len(calls) == 2


# ─── the demo ────────────────────────────────────────────────────────────────


def test_the_demo_opens_on_its_configured_schema_and_a_lists_its_three_locations():
    from cortex_training.tui.demo import DemoClient

    demo = DemoClient()

    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(200, 50)) as pilot:
            s = await _list(app, pilot)
            here = {s._jobs_by_id[jid].get("database_name") for jid in _shown(s)}
            assert here == {None} and scope(app) == f"this schema: {HERE}"  # the configured schema's own rows
            await pilot.press("a")
            assert await wait(pilot, lambda: scope(app) == "all schemas" and not s._refreshing)
            located = {(s._jobs_by_id[jid]["database_name"], s._jobs_by_id[jid]["schema_name"]) for jid in _shown(s)}
            assert located == {("CORTEX_TRAINING_DB", "PUBLIC"), ("TEAM_DB", "EXPERIMENTS"), ("sandbox", "dev")}
            await settle(app, pilot)

    asyncio.run(run())


# ─── the docs ────────────────────────────────────────────────────────────────


def test_cli_md_documents_the_default_the_toggle_and_what_a_says_when_it_cannot():
    text = " ".join(CLI_MD.read_text(encoding="utf-8").split())
    assert f"`{scope_text(ListScope(SCHEMA), location='DB.SCHEMA')}`" in text
    for phrase in ("`a All schemas`", "`a This schema`", "the footer has no `a`",
                   f"`Can't show all schemas (no account-wide list yet); the list stays on {HERE}`",
                   "`a` within a minute answers the same"):
        assert phrase in text, phrase
    keys = text[text.index("| Key | Action |"):text.index("#### Job preview")]
    assert "| `a` |" in keys and "hidden with `--schema-only`" in keys
    assert account.ACCOUNT_MANUAL_REPROBE_SECONDS == 60  # "within a minute"


def test_rest_api_md_says_the_tui_reads_the_account_level_list_only_after_a():
    text = " ".join(REST_MD.read_text(encoding="utf-8").split())
    assert ("The job TUI's list reads the configured schema (`list_jobs` with those bounds) until `a` asks for "
            "every location.") in text
    assert "If this list is unavailable when `a` asks for it, the TUI stays on the configured schema" in text
