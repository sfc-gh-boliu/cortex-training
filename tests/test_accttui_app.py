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

"""The job list across every location, its fallback, and per-job routing, in the running app.

Clients are a real ``CortexTrainingClient`` over a recording transport, or
MagicMocks whose every list call is configured explicitly: an unconfigured
MagicMock answers with another MagicMock, which proves nothing.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter

pytest.importorskip("textual")

from textual.geometry import Region  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Input  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets._header import HeaderTitle  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training import CortexTrainingClient  # noqa: E402
from cortex_training.tui import account  # noqa: E402
from cortex_training.tui.__main__ import read_only_client  # noqa: E402
from cortex_training.tui.account import JobLister  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.app import LogScreen  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.untrusted import unwrapped  # noqa: E402

ACCOUNT_PATH = "/api/v2/cortex-training"
PUBLIC = "/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training"
LOWER = "/api/v2/databases/%22ml%22/schemas/%22dev%22/cortex-training"


@pytest.fixture(autouse=True)
def _isolate_tui_files(tmp_path, monkeypatch):
    # As conftest does for test_tui_* files: no writes to the real cache or home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(jid: str, status: str = "CANCELLED", *, db=None, schema=None, minutes_ago=60, **extra) -> dict:
    job = {"job_id": jid, "status": status, "created_at": _iso(minutes_ago), "updated_at": _iso(minutes_ago - 1),
           "comment": f"job {jid}", "sub_jobs": [{"sub_job_id": f"{jid}:training:0", "job_type": "TRAINING",
                                                 "status": status, "training_config": {"n_gpus": 8.0}}], **extra}
    if db is not None:
        job.update(database_name=db, schema_name=schema)
    return job


def _unavailable(reason: str) -> AccountJobListUnavailable:
    return AccountJobListUnavailable(f"account-level job listing is unavailable ({reason})", reason=reason)


def _http_error(status: int, message: str) -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code, resp.reason = status, {503: "Service Unavailable", 500: "Internal Server Error"}.get(status, "")
    resp._content = json.dumps({"message": message}).encode()
    return requests.exceptions.HTTPError(f"{status} Error", response=resp)


def _page(*jobs, token=None) -> dict:
    return {"jobs": list(jobs), "next_page_token": token}


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


def _run(coro):
    asyncio.run(coro)


def _app(client, *args, **kwargs) -> CortexTrainingLogTUI:
    """The app with its job list on every location, as ``a`` asks for it before the
    first list arrives (the list opens on the configured schema)."""
    app = CortexTrainingLogTUI(client, *args, poll_interval=0.01, **kwargs)
    app._lister.account_wide = True
    return app


def _notices(app) -> list:
    return [str(n.message) for n in app._notifications]


def _text(screen, selector) -> str:
    widget = screen.query_one(selector, Static)
    return str(widget.render()) if widget.display else ""


def _labels(screen) -> list:
    return [str(column.label) for column in screen.query_one("#jobs", DataTable).columns.values()]


def _scope(app) -> str:
    """The list's scope in the subtitle: after the refresh status, before the key hints."""
    parts = (app.sub_title or "").split("  ·  ")
    return parts[1] if len(parts) == 3 else ""


def _header_line(app) -> str:
    """The header as the terminal shows it: cut, with "…", where the screen ends."""
    title = app.screen.query_one(HeaderTitle)
    return title.render_lines(Region(0, 0, title.size.width, 1))[0].text


def _cells(screen, jid) -> dict:
    key = next(item for item, job_id in screen._job_by_item.items() if job_id == jid)
    return dict(zip(_labels(screen), (str(cell) for cell in screen.query_one("#jobs", DataTable).get_row(key))))


async def _list(app, pilot) -> JobListScreen:
    assert await _wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    return app.screen


async def _open(pilot, screen, jid):
    """Enter on ``jid``'s row of the job list."""
    table = screen.query_one("#jobs", DataTable)
    key = next(item for item, job_id in screen._job_by_item.items() if job_id == jid)
    table.move_cursor(row=table.get_row_index(key), animate=False)
    screen.set_focus(table)
    await pilot.pause()
    await pilot.press("enter")


# ─── a real client: the 404 every account gets today, and quoted locations ───


class _Service(BaseAdapter):
    """Answers the client's requests from ``routes`` (path -> callable(request) -> (status, body))."""

    def __init__(self, routes: dict):
        super().__init__()
        self.routes = routes
        self.sent: list = []

    def send(self, request, **kwargs):
        self.sent.append(request)
        path = urlsplit(request.url).path
        status, body = self.routes.get(path, lambda r: (404, {"message": "Training job not found"}))(request)
        resp = requests.Response()
        resp.status_code = status
        resp._content = b"" if body is None else json.dumps(body).encode()
        if body is not None:
            resp.headers["Content-Type"] = "application/json"
        resp.request, resp.url = request, request.url
        return resp

    def close(self):
        pass

    def paths(self) -> list:
        return [f"{r.method} {urlsplit(r.url).path}" for r in self.sent]


def _real_client(routes: dict) -> tuple:
    client = CortexTrainingClient(base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="PUBLIC",
                                  max_retries=0)
    service = _Service(routes)
    client._session.mount("http://", service)
    return read_only_client(client), service


def test_the_app_survives_the_404_of_an_account_without_the_route_and_lists_the_schema():
    jobs = [_job("a1", "RUNNING", minutes_ago=5), _job("a2")]
    client, service = _real_client({
        ACCOUNT_PATH: lambda r: (404, None),  # not deployed: 404 with no body
        PUBLIC: lambda r: (200, {"jobs": jobs}),
    })

    async def run():
        app = CortexTrainingLogTUI(client, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _list(app, pilot)
            assert set(screen._job_by_item.values()) == {"a1", "a2"}
            assert _scope(app) == "this schema: CORTEX_TRAINING_DB.PUBLIC"
            assert service.paths() == [f"GET {PUBLIC}"]  # the configured schema's list, as the list opens
            await pilot.press("a")
            assert await _wait_for(pilot, lambda: any("Can't show all schemas" in n for n in _notices(app)))
            assert app.is_running and app.return_code is None
            assert "Can't show all schemas (no account-wide list yet); the list stays on CORTEX_TRAINING_DB.PUBLIC." \
                in _notices(app)
            assert await _wait_for(pilot, lambda: not screen._refreshing)
            assert set(screen._job_by_item.values()) == {"a1", "a2"}
            assert _scope(app) == "this schema: CORTEX_TRAINING_DB.PUBLIC"
            assert "Location" not in _labels(screen) and not screen._lister.account_wide
            await pilot.press("r")  # this schema's list: the account-wide list isn't asked
            assert await _wait_for(pilot, lambda: service.paths().count(f"GET {PUBLIC}") == 3)
            assert app.is_running
            await _settle(app, pilot)

    _run(run())
    assert service.paths() == [f"GET {PUBLIC}", f"GET {ACCOUNT_PATH}", f"GET {PUBLIC}", f"GET {PUBLIC}"]


def test_a_real_client_lists_every_location_and_reads_a_job_through_its_own():
    rows = [_job("x1", "RUNNING", db="ml", schema="dev", minutes_ago=3),
            _job("x2", db="CORTEX_TRAINING_DB", schema="PUBLIC", minutes_ago=30)]
    asked = []

    def account_page(request):
        query = parse_qs(urlsplit(request.url).query)
        asked.append(query)
        if "page_token" not in query:
            return 200, {"jobs": rows[:1], "next_page_token": "tok+en/1="}
        return 200, {"jobs": rows[1:]}

    fresh = {k: v for k, v in rows[0].items() if k not in ("database_name", "schema_name")}
    client, service = _real_client({
        ACCOUNT_PATH: account_page,
        f"{LOWER}/x1": lambda r: (200, fresh),
    })

    async def run():
        app = _app(client)
        async with app.run_test(size=(180, 40)) as pilot:
            screen = await _list(app, pilot)
            assert _scope(app) == "all schemas"
            assert _cells(screen, "x1")["Location"] == '"ml"."dev"'
            await _open(pilot, screen, "x1")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert await _wait_for(pilot, lambda: f"GET {LOWER}/x1" in service.paths())
            await _settle(app, pilot)

    _run(run())
    # Two pages, the token passed back (URL-encoded by the client), the same filters on both.
    assert asked[1]["page_token"] == ["tok+en/1="]
    assert asked[0]["created_after"] == asked[1]["created_after"] and "status" not in asked[0]
    assert asked[0]["limit"] == ["500"]
    # Nothing about the job went to the configured schema, and nothing was created.
    assert not any(path.startswith(f"GET {PUBLIC}") for path in service.paths())
    assert not any(path.startswith("POST") for path in service.paths())


# ─── MagicMock clients ───────────────────────────────────────────────────────


def _base(*pages, schema_rows=None) -> MagicMock:
    c = MagicMock()
    c.database, c.schema, c._auth_provider = "TRAINING_DB", "RUNS", None
    c.list_jobs_page.side_effect = list(pages) if pages else [_page()]
    c.list_jobs.return_value = list(schema_rows or [])
    c.iter_jobs.side_effect = AssertionError("the job list reads pages itself, never iter_jobs")
    for name in ("get_job", "tail_logs", "download_metrics", "download_stdout_logs",
                 "fetch_execution_logs"):
        getattr(c, name).side_effect = AssertionError(f"{name} on the TUI's own client")
    return c


# T2: a failed refresh keeps the last list and says so.
def test_a_failed_refresh_keeps_the_list_says_since_when_and_clears_on_the_next_success():
    good = _page(_job("j1", db="TEAM_DB", schema="EXPERIMENTS"), _job("j2", db="ml", schema="dev"))
    c = _base(good, _http_error(503, "Control Plane unavailable"), good)

    async def run():
        app = _app(c)
        async with app.run_test(size=(160, 40)) as pilot:
            screen = await _list(app, pilot)
            at = screen._listed_at
            assert at and not screen.query_one("#listnotice", Static).display
            await pilot.press("r")
            assert await _wait_for(pilot, lambda: "refresh failed" in _text(screen, "#listnotice"))
            assert _text(screen, "#listnotice").strip() == (
                f"refresh failed {screen._failed_at}: the account-wide job list answered HTTP 503 Service "
                f"Unavailable: Control Plane unavailable; showing the list from {at}")
            assert set(screen._job_by_item.values()) == {"j1", "j2"}  # the list stays
            assert _scope(app) == "all schemas"  # what the kept list covers
            await pilot.press("r")
            assert await _wait_for(pilot, lambda: not screen.query_one("#listnotice", Static).display)
            assert set(screen._job_by_item.values()) == {"j1", "j2"} and c.list_jobs_page.call_count == 3
            await _settle(app, pilot)

    _run(run())


def _refusal(message: str, body) -> AccountJobListUnavailable:
    resp = requests.Response()
    resp.status_code = 403
    resp._content = b"" if body is None else json.dumps(body).encode()
    return AccountJobListUnavailable(message, reason="forbidden", response=resp)


def test_a_failed_load_with_no_job_listed_shows_the_whole_error_in_place_of_the_list():
    refused = _refusal("account-level job listing was refused (HTTP 403): Cortex Training is not enabled for this "
                       "account, or its trial has ended",
                       {"code": "510201", "message": "Cortex Training is not enabled for this account"})
    c = _base(_page(), refused)  # every location, with no job yet; then a refresh is refused

    async def run():
        app = _app(c)
        async with app.run_test(size=(80, 30)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._loaded)
            assert _scope(app) == "all schemas"
            await pilot.press("r")
            assert await _wait_for(pilot, lambda: "[error]" in _text(app.screen, "#listinfo"))
            info = app.screen.query_one("#listinfo", Static)
            # The refusal, and what still lists jobs where only this route is refused.
            assert _text(app.screen, "#listinfo").endswith(
                "or its trial has ended  (r retries · a lists TRAINING_DB.RUNS only)")
            assert info.has_class("error")
            assert await _wait_for(pilot, lambda: info.size.height >= 2)  # wrapped, not cut at the edge
            assert not app.screen.query_one("#jobs", DataTable).display
            c.list_jobs.assert_not_called()  # the service refused: the configured schema would too
            await _settle(app, pilot)

    _run(run())


def test_a_403_without_the_services_message_lists_the_configured_schema_when_it_answers():
    # A network policy or a proxy in front of the account-level route alone: a asks
    # the configured schema once, which answers, so the list stays on it.
    refused = _refusal("account-level job listing was refused (HTTP 403) with no error message: a network policy "
                       "or a proxy may have blocked the request", None)
    c = _base(refused, schema_rows=[_job("s1")])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 30)) as pilot:
            screen = await _list(app, pilot)
            await pilot.press("a")
            assert await _wait_for(pilot, lambda: c.list_jobs.call_count == 2 and not screen._refreshing)
            assert set(screen._job_by_item.values()) == {"s1"}
            assert _scope(app) == "this schema: TRAINING_DB.RUNS"
            assert ("Can't show all schemas (account-wide list: HTTP 403); the list stays on TRAINING_DB.RUNS."
                    in _notices(app))
            await _settle(app, pilot)

    _run(run())
    assert c.list_jobs_page.call_count == 1


def test_a_403_without_the_services_message_stands_when_the_schema_is_refused_too():
    refused = _refusal("account-level job listing was refused (HTTP 403) with no error message", None)
    c = _base(refused)
    c.list_jobs.side_effect = _http_error(403, "Forbidden")

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 30)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobListScreen)
                                   and "[error]" in _text(app.screen, "#listinfo"))
            assert _text(app.screen, "#listinfo") == "[error] list_jobs answered HTTP 403: Forbidden  (r retries)"
            await pilot.press("a")
            # Both refusals, said as a asked; the list stays on this schema's (its error, here).
            assert await _wait_for(pilot, lambda: any("Can't show all schemas" in n for n in _notices(app)))
            assert ("Can't show all schemas (account-level job listing was refused (HTTP 403) with no error message; "
                    "and list_jobs answered HTTP 403: Forbidden); the list stays on TRAINING_DB.RUNS.") in _notices(app)
            assert await _wait_for(pilot, lambda: not app.screen._refreshing)
            assert _text(app.screen, "#listinfo") == "[error] list_jobs answered HTTP 403: Forbidden  (r retries)"
            assert not app.screen._lister.account_wide
            await _settle(app, pilot)

    _run(run())


# T3: the scope in the header.
@pytest.mark.parametrize("pages, kind, scope", [
    ((), "schema", "this schema: CORTEX_TRAINING_DB.PUBLIC"),
    ((_page(_job("j1", db="A", schema="B"), _job("j2", db="C", schema="D")),), "account", "all schemas"),
    ((_page(_job("j1", db="A", schema="B")), _unavailable("not_deployed")), "fallback",
     "this schema only: CORTEX_TRAINING_DB.PUBLIC (no account-wide list yet)"),
    ((_page(_job("j1", db="A", schema="B")), _unavailable("disabled")), "fallback",
     "this schema only: CORTEX_TRAINING_DB.PUBLIC (account-wide list not enabled)"),
    ((), "schema_only", "this schema only: CORTEX_TRAINING_DB.PUBLIC (--schema-only)"),
])
def test_the_header_says_what_the_list_covers_after_the_refresh_status(pages, kind, scope):
    c = _base(*pages, schema_rows=[_job("s1")])
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    schema_only = kind == "schema_only"

    async def run():
        app = (_app(c) if kind in ("account", "fallback")
               else CortexTrainingLogTUI(c, poll_interval=0.01, schema_only=schema_only))
        async with app.run_test(size=(140, 40)) as pilot:
            if kind == "fallback":
                # Every location first; then a refresh whose account-wide list is gone.
                screen = await _list(app, pilot)
                screen._refresh_tick()
                assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"s1"})
            screen = await _list(app, pilot)
            # The status first: a narrow terminal cuts the end of the line, and r must show.
            stamp = screen._last_refreshed
            assert (app.sub_title or "").startswith(f"last refreshed {stamp}  ·  {scope}  ·  select a job")
            # On a 140-column terminal both are on screen, with the documented location.
            assert f"last refreshed {stamp}  ·  {scope}" in _header_line(app)
            refreshing = threading.Event()
            c.list_jobs.side_effect = lambda **_: refreshing.wait(5) and [_job("s1")]
            c.list_jobs_page.side_effect = lambda **_: refreshing.wait(5) and _page()
            await pilot.press("r")
            assert await _wait_for(pilot, lambda: f"refreshing… (last refreshed {stamp})" in _header_line(app))
            refreshing.set()
            await _settle(app, pilot)

    _run(run())
    if schema_only:
        c.list_jobs_page.assert_not_called()


@pytest.mark.parametrize("reason, schema_only", [("not_deployed", False), ("disabled", False), (None, True)])
def test_a_narrow_terminal_still_shows_the_refresh_status(reason, schema_only):
    c = _base(*([_unavailable(reason)] if reason else []), schema_rows=[_job("s1")])
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01, schema_only=schema_only)
        async with app.run_test(size=(100, 40)) as pilot:
            screen = await _list(app, pilot)
            assert _header_line(app).startswith(f"cortex training — last refreshed {screen._last_refreshed}  ·  ")
            await _settle(app, pilot)

    _run(run())


def _walks(pages, active=None):
    """``list_jobs_page`` answers: ``pages`` in turn for the walk without a status, ``active``
    (a page, or an error to raise) for the one with ``status="active"``."""
    main = iter(pages)
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        if kwargs.get("status") == "active":
            if isinstance(active, Exception):
                raise active
            return active if active is not None else _page()
        return next(main)

    answer.calls = calls
    return answer


def test_a_list_cut_at_the_cap_says_more_exist_and_still_lists_older_running_jobs(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_CAP", 3)
    rows = [_job(f"j{i}", db="A", schema="B", minutes_ago=10 + i) for i in range(5)]
    # Created three days ago and still running: older than the newest three.
    old_live = _job("old-live", "RUNNING", db="C", schema="D", minutes_ago=3 * 24 * 60)
    c = _base()
    c.list_jobs_page.side_effect = walks = _walks([_page(*rows[:2], token="t1"), _page(*rows[2:], token="t2")],
                                                  active=_page(old_live))

    async def run():
        app = _app(c)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            assert set(screen._job_by_item.values()) == {"j0", "j1", "j2", "old-live"}  # the newest three, and it
            assert _scope(app) == "newest 3 + every active job (more exist)"
            info = _text(screen, "#listinfo")
            assert "● 1 live" in info and "newest 3 jobs + 1 older active" in info
            screen.query_one("#jobfilter", Input).value = "nothing-matches-this"
            assert await _wait_for(pilot, lambda: "more exist" in _text(screen, "#nomatch"))
            assert _text(screen, "#nomatch").startswith(
                "No jobs match “nothing-matches-this” among the 4 job(s) loaded: the newest 3 of the last 6 months "
                "and every active job; more exist.")
            assert "--schema-only" in _text(screen, "#nomatch")
            await _settle(app, pilot)

    _run(run())
    # Two pages of the newest jobs (never on to the end), then one of the active ones.
    assert [call.get("status") for call in walks.calls] == [None, None, "active"]


def test_a_failed_walk_for_the_active_jobs_keeps_the_list_and_says_so(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_CAP", 2)
    rows = [_job(f"j{i}", db="A", schema="B", minutes_ago=10 + i) for i in range(3)]
    c = _base()
    c.list_jobs_page.side_effect = _walks([_page(*rows, token="t1")],
                                          active=_http_error(503, "Control Plane unavailable"))

    async def run():
        app = _app(c)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            assert set(screen._job_by_item.values()) == {"j0", "j1"}
            assert _scope(app) == "newest 2 (more exist; older active jobs may be missing)"
            assert not screen.query_one("#listnotice", Static).display  # the list itself is fine
            await _settle(app, pilot)

    _run(run())


def test_a_walk_stopped_by_the_page_limit_with_no_rows_says_more_may_exist(monkeypatch):
    # A role that sees few jobs in a busy account: pages come back empty, with a token.
    monkeypatch.setattr(account, "ACCOUNT_LIST_MAX_PAGES", 2)
    c = _base()
    c.list_jobs_page.side_effect = _walks([_page(token="t1"), _page(token="t2")])

    async def run():
        app = _app(c)
        async with app.run_test(size=(200, 40)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._loaded)
            screen = app.screen
            assert _scope(app) == "first 2 pages: 0 jobs + every active job (more may exist)"
            assert "first 2 pages (0 jobs)" in _text(screen, "#listinfo")
            nomatch = _text(screen, "#nomatch")
            assert nomatch == ("The account-wide list returned no jobs to show in its first 2 pages; more may exist. "
                               "r reads it again; --database DB --schema SCHEMA --schema-only lists one schema's jobs.")
            assert "in any database and schema" not in nomatch
            await _settle(app, pilot)

    _run(run())


# T4: the Location column.
def test_the_location_column_shows_quoted_names_literally_and_search_matches_it():
    rows = [_job("j1", "RUNNING", db="TEAM_DB", schema="EXPERIMENTS", minutes_ago=5),
            _job("j2", db="ml", schema="dev"),
            _job("j3", db="[b]odd[/b] db", schema="PUBLIC")]
    c = _base(_page(*rows))

    async def run():
        app = _app(c)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            labels = _labels(screen)
            # After GPUs: Model and GPUs keep their place on a 140-column terminal.
            assert labels[labels.index("GPUs") + 1] == "Location" and labels[-1] == "Updated"
            assert _cells(screen, "j1")["Location"] == "TEAM_DB.EXPERIMENTS"
            assert _cells(screen, "j2")["Location"] == '"ml"."dev"'
            assert _cells(screen, "j3")["Location"] == '"[b]odd[/b] db".PUBLIC'  # Text, never markup
            # A sub-job row and a group header still line up with the columns.
            sub = next(key for key in screen._sub_by_item)
            assert len(screen.query_one("#jobs", DataTable).get_row(sub)) == len(labels)
            box = screen.query_one("#jobfilter", Input)
            box.value = "team_db"
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"j1"})
            box.value = "db:ml"
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"j2"})
            box.value = "schema=public"
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"j3"})
            box.value = '"ml"."dev"'  # as the column shows it; the box's quoting makes it ml.dev
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"j2"})
            box.value = "db=ML"  # ML is not the lowercase "ml"
            assert await _wait_for(pilot, lambda: not screen._job_by_item)
            assert "Location" in _labels(screen)  # it stays while a search narrows to one location
            await _settle(app, pilot)

    _run(run())


@pytest.mark.parametrize("rows", [
    [_job("j1", db="TEAM_DB", schema="EXPERIMENTS"), _job("j2", db="TEAM_DB", schema="EXPERIMENTS")],
    [_job("j1"), _job("j2")],  # the configured schema's own rows name no location
])
def test_no_location_column_when_every_row_shares_one(rows):
    c = _base(_page(*rows))

    async def run():
        app = _app(c)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            assert "Location" not in _labels(screen) and len(_labels(screen)) == len(JobListScreen._COLUMNS)
            # Without the column, the header names the one location, when the rows name it.
            shared = "all schemas (all in TEAM_DB.EXPERIMENTS)" if "database_name" in rows[0] else "all schemas"
            assert _scope(app) == shared
            await _settle(app, pilot)

    _run(run())


# T1: pages are read in the worker; a cancelled worker stops between pages.
def test_pages_are_read_off_the_ui_thread_and_a_cancelled_walk_stops():
    first_page_read = threading.Event()
    release = threading.Event()
    threads = []

    def list_jobs_page(**kwargs):
        threads.append(threading.current_thread())
        first_page_read.set()
        release.wait(10)
        return _page(_job(f"j{len(threads)}", db="A", schema="B"), token="more")

    c = _base()
    c.list_jobs_page.side_effect = list_jobs_page

    async def run():
        app = _app(c)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait_for(pilot, first_page_read.is_set)
            assert isinstance(app.screen, JobListScreen)
            app.workers.cancel_group(app.screen, "jobs")
            release.set()
            await pilot.pause(0.3)
            assert c.list_jobs_page.call_count == 1  # no second page after the cancel
            assert not app.screen._loaded and app.is_running
            await _settle(app, pilot)

    try:
        _run(run())
    finally:
        release.set()
    assert threads and all(t is not threading.main_thread() for t in threads)


def test_the_account_wide_list_is_asked_again_after_ten_minutes_of_fallback():
    clock = [1000.0]
    team = _job("t1", "RUNNING", db="TEAM_DB", schema="EXPERIMENTS", minutes_ago=2)
    every = _page(team, _job("t2", db="ml", schema="dev"))
    c = _base(every, _unavailable("not_deployed"), every, schema_rows=[_job("s1")])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        app._lister = JobLister(c, account_wide=True, clock=lambda: clock[0])
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            assert set(screen._job_by_item.values()) == {"t1", "t2"}
            screen._refresh_tick()  # the account-wide list is gone: the configured schema in its place
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"s1"})
            assert "Location" not in _labels(screen) and screen._lister.account_wide  # still asked for
            assert _scope(app) == "this schema only: TRAINING_DB.RUNS (no account-wide list yet)"
            clock[0] += 601
            screen._refresh_tick()  # the timer's refresh, after ten minutes
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"t1", "t2"})
            assert _scope(app) == "all schemas" and "Location" in _labels(screen)
            await _settle(app, pilot)

    _run(run())
    assert c.list_jobs_page.call_count == 3 and c.list_jobs.call_count == 1


def test_r_asks_the_account_wide_list_again_after_a_minute_of_fallback():
    clock = [1000.0]
    every = _page(_job("t1", db="TEAM_DB", schema="EXPERIMENTS"), _job("t2", db="ml", schema="dev"))
    c = _base(every, _unavailable("not_deployed"), every, schema_rows=[_job("s1")])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        app._lister = JobLister(c, account_wide=True, clock=lambda: clock[0])
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            screen._refresh_tick()
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"s1"})
            clock[0] += 30
            await pilot.press("r")  # within a minute: the configured schema again
            assert await _wait_for(pilot, lambda: c.list_jobs.call_count == 2 and not screen._refreshing)
            assert c.list_jobs_page.call_count == 2
            clock[0] += 31
            await pilot.press("r")  # the route may be there now: r asks
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"t1", "t2"})
            assert _scope(app) == "all schemas"
            await _settle(app, pilot)

    _run(run())
    assert c.list_jobs_page.call_count == 3


# T5: every call about a job from another location goes through its own client.
def _child(database: str, schema: str, job: dict) -> MagicMock:
    child = MagicMock(name=f"client for {database}.{schema}")
    child.database, child.schema, child._auth_provider = database, schema, None
    fresh = {k: v for k, v in job.items() if k not in ("database_name", "schema_name")}
    child.get_job.side_effect = lambda jid: fresh if jid == job["job_id"] else {}
    child.tail_logs.return_value = {"entries": [{"ts": "12:00:00", "level": "INFO", "msg": "step 1 loss 0.5"}],
                                    "next_cursor": "c1", "eof": False}
    child.download_metrics.return_value = []
    child.download_stdout_logs.return_value = []
    child.fetch_execution_logs.return_value = []
    return child


def test_every_call_about_a_job_in_another_location_goes_through_its_child_client():
    live = _job("team-live", "RUNNING", db="TEAM_DB", schema="EXPERIMENTS", minutes_ago=5)
    failed = _job("lower-failed", "FAILED", db="ml", schema="dev", minutes_ago=90, reason="out_of_memory",
                  state_changed_at=_iso(60))
    team = _child("TEAM_DB", "EXPERIMENTS", live)
    lower = _child('"ml"', '"dev"', failed)
    base = _base(_page(live, failed, _job("home", db="TRAINING_DB", schema="RUNS")))
    made = []

    def for_job(row):
        made.append((row["database_name"], row["schema_name"]))
        return {"TEAM_DB": team, "ml": lower}.get(row["database_name"], MagicMock())

    base.for_job.side_effect = for_job

    async def run():
        app = _app(base)
        async with app.run_test(size=(200, 50)) as pilot:
            screen = await _list(app, pilot)
            # The preview: get_job.
            await _open(pilot, screen, "team-live")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            preview = app.screen
            assert unwrapped(preview._client) is team
            assert await _wait_for(pilot, lambda: team.get_job.called)
            # Logs: the sources' get_job and tail-logs.
            await pilot.press("l")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and team.tail_logs.called)
            detail = app.screen
            assert team.tail_logs.call_args.args == ("team-live",)
            # Resources: GPU metrics.
            await pilot.press("3")
            assert await _wait_for(pilot, lambda: team.download_metrics.called)
            # The 60-second status check.
            calls = team.get_job.call_count
            detail._refresh_status()
            assert await _wait_for(pilot, lambda: team.get_job.call_count > calls)
            # A tail error reads the job once, through the same client.
            calls = team.get_job.call_count
            team.tail_logs.side_effect = _http_error(500, "boom")
            await pilot.press("2")
            detail._start_tail("team-live:training:0")
            assert await _wait_for(pilot, lambda: team.get_job.call_count > calls)
            # Back to the list, and into the failed job in "ml"."dev": its saved console
            # and, without one, its execution log.
            for _ in range(2):
                await pilot.press("b")
            assert await _wait_for(pilot, lambda: app.screen is screen)
            await _open(pilot, screen, "lower-failed")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert unwrapped(app.screen._client) is lower
            await pilot.press("l")
            assert await _wait_for(pilot, lambda: lower.download_stdout_logs.called
                                   and lower.fetch_execution_logs.called)
            assert lower.fetch_execution_logs.call_args.args == ("lower-failed",)
            await _settle(app, pilot)

    _run(run())
    assert made == [("TEAM_DB", "EXPERIMENTS"), ("ml", "dev")]  # one client per location, kept
    for name in ("get_job", "tail_logs", "download_metrics", "download_stdout_logs",
                 "fetch_execution_logs"):
        assert not getattr(base, name).called, f"{name} went to the TUI's own client"
    assert lower.get_job.called and not lower.tail_logs.called  # a failed job is never tailed


def test_a_sub_job_row_opens_its_details_through_the_jobs_client():
    live = _job("team-live", "RUNNING", db="TEAM_DB", schema="EXPERIMENTS", minutes_ago=5)
    team = _child("TEAM_DB", "EXPERIMENTS", live)
    base = _base(_page(live, _job("other", db="ml", schema="dev")))
    base.for_job.side_effect = lambda row: team if row["database_name"] == "TEAM_DB" else MagicMock()

    async def run():
        app = _app(base)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            sub = next(key for key, (jid, _sid) in screen._sub_by_item.items() if jid == "team-live")
            table = screen.query_one("#jobs", DataTable)
            table.move_cursor(row=table.get_row_index(sub), animate=False)
            screen.set_focus(table)
            await pilot.press("enter")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen))
            assert unwrapped(app.screen._client) is team and app.screen._scope == "team-live:training:0"
            assert await _wait_for(pilot, lambda: team.get_job.called)
            # The summary line says where the job is, after the models.
            assert _text(app.screen, "#summary").endswith(
                "team-live  ·  sub-job team-live:training:0  ·  training  ·  —  ·  TEAM_DB.EXPERIMENTS")
            await _settle(app, pilot)

    _run(run())
    assert not base.get_job.called


def test_the_summary_line_names_only_a_location_other_than_the_configured_one():
    home = _job("home-live", "RUNNING", db="TRAINING_DB", schema="RUNS", minutes_ago=5)
    child = _child("TRAINING_DB", "RUNS", home)
    base = _base(_page(home, _job("other", db="ml", schema="dev")))
    base.for_job.side_effect = lambda row: child if row["database_name"] == "TRAINING_DB" else MagicMock()

    async def run():
        app = _app(base)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            sub = next(key for key, (jid, _sid) in screen._sub_by_item.items() if jid == "home-live")
            table = screen.query_one("#jobs", DataTable)
            table.move_cursor(row=table.get_row_index(sub), animate=False)
            screen.set_focus(table)
            await pilot.press("enter")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and child.get_job.called)
            summary = _text(app.screen, "#summary")
            assert summary.endswith("home-live  ·  sub-job home-live:training:0  ·  training  ·  —")
            assert "TRAINING_DB" not in summary  # the configured location: it would only push the line to wrap
            await _settle(app, pilot)

    _run(run())


# T5 / the copyable commands: the Logs tab's download-log hints name the job's location.
_EXEC_LOG = "".join(json.dumps(r) + "\n" for r in [
    {"ts": "2026-10-14T11:00:42+00:00", "level": "ERROR", "logger": "runtime", "msg": "failed to create job"},
])


class _Profile:
    connection_name = "training"


def test_the_execution_log_hint_names_the_jobs_location_and_connection():
    # The newest of the two, so the first row of HISTORY, where the list puts its cursor.
    failed = _job("team-failed", "FAILED", db="TEAM_DB", schema="EXPERIMENTS", minutes_ago=30,
                  reason="placement_too_large")
    team = _child("TEAM_DB", "EXPERIMENTS", failed)
    team._auth_provider = _Profile()
    team.fetch_execution_logs.return_value = [{"sub_job_id": "team-failed:training:0", "filename": "execution.jsonl",
                                               "content": _EXEC_LOG}]
    base = _base(_page(failed, _job("other", db="ml", schema="dev")))
    base.for_job.side_effect = lambda row: team if row["database_name"] == "TEAM_DB" else MagicMock()

    async def run():
        app = _app(base)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            await _open(pilot, screen, "team-failed")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            await pilot.press("l")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and any(
                "no saved console" in line for line in app.screen._shown_lines))
            head = next(line for line in app.screen._shown_lines if "no saved console" in line)
            # Copied as shown, it reaches the job, not the configured TRAINING_DB.RUNS.
            assert ("(cortex-training --connection training --database TEAM_DB --schema EXPERIMENTS download-log "
                    "team-failed downloads the whole log)") in head
            await _settle(app, pilot)

    _run(run())


def test_the_scrollback_hint_names_the_jobs_location(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 50)
    live = _job("lower-live", "RUNNING", db="ml", schema="dev", minutes_ago=5)
    lower = _child('"ml"', '"dev"', live)
    page = {"entries": [{"_raw": f"ts line {i}"} for i in range(120)], "next_cursor": "C1", "eof": False}
    pages = iter([page])
    lower.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})
    base = _base(_page(live, _job("other", db="TEAM_DB", schema="EXPERIMENTS")))
    base.for_job.side_effect = lambda row: lower if row["database_name"] == "ml" else MagicMock()

    async def run():
        app = _app(base)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            await _open(pilot, screen, "lower-live")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            await pilot.press("l")
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                   and "ts line 119" in app.screen._shown_lines)
            status = _text(app.screen, "#log-status")
            assert status.endswith("""· cortex-training --database '"ml"' --schema '"dev"' download-log lower-live """
                                   "--log-type stdout for the full log")
            await _settle(app, pilot)

    _run(run())


# A job whose location gets no client is not opened through the configured one.
def test_a_job_whose_client_cannot_be_made_is_not_opened(tmp_path):
    base = _base(_page(_job("team", db="TEAM_DB", schema="EXPERIMENTS"), _job("other", db="ml", schema="dev")))
    base.for_job.side_effect = ValueError("no such identifier [b]x[/b]")

    async def run():
        app = _app(base)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            await _open(pilot, screen, "team")
            assert await _wait_for(pilot, lambda: any("Can't open the job" in str(n.message)
                                                      for n in app._notifications))
            note = next(n for n in app._notifications if "Can't open the job" in str(n.message))
            assert "no client for TEAM_DB.EXPERIMENTS" in note.message and "[b]x[/b]" in note.message
            assert not note.markup  # the error text is shown as it is
            assert app.screen is screen  # stays on the list
            await _settle(app, pilot)

    _run(run())
    assert not base.get_job.called
    log = (tmp_path / "home" / ".cortex-training-errors.log").read_text()
    assert "no client for TEAM_DB.EXPERIMENTS" in log and "ValueError" in log


# A job opened by id that isn't in the configured location says where it looked.
def test_a_job_opened_by_id_that_is_not_found_says_where_it_looked():
    base = _base(_page())
    resp = requests.Response()
    resp.status_code, resp.reason = 404, "Not Found"
    resp._content = json.dumps({"message": "Training job other-schema-job not found"}).encode()
    base.get_job.side_effect = requests.exceptions.HTTPError("404 Client Error", response=resp)

    async def run():
        app = CortexTrainingLogTUI(base, "other-schema-job", poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and any(
                line.startswith("[no logs]") for line in app.screen._shown_lines))
            line = next(line for line in app.screen._shown_lines if line.startswith("[no logs]"))
            assert line.startswith("[no logs] job other-schema-job was not found in TRAINING_DB.RUNS (HTTP 404")
            assert line.endswith("If the job list shows it in another location, start the TUI with that --database "
                                 "and --schema.")
            await _settle(app, pilot)

    _run(run())


# db: and schema: also read the configured location for rows of its own list.
def test_db_and_schema_filters_match_the_configured_schemas_own_rows():
    c = _base(_unavailable("not_deployed"), schema_rows=[_job("s1"), _job("s2")])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            screen = await _list(app, pilot)
            box = screen.query_one("#jobfilter", Input)
            box.value = "db=training_db schema:run"
            await pilot.pause(0.3)
            assert await _wait_for(pilot, lambda: set(screen._job_by_item.values()) == {"s1", "s2"})
            box.value = "db=TEAM_DB"
            assert await _wait_for(pilot, lambda: not screen._job_by_item)
            await _settle(app, pilot)

    _run(run())


def test_a_job_opened_by_id_uses_the_tuis_own_client_and_back_shows_the_apps_list():
    job = _job("by-id", "CANCELLED")
    base = _base(_page(_job("j1", db="A", schema="B"), _job("j2", db="C", schema="D")))
    base.get_job.side_effect = lambda jid: job
    base.download_stdout_logs.side_effect = None
    base.download_stdout_logs.return_value = []
    base.tail_logs.side_effect = None
    base.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}

    async def run():
        app = _app(base, "by-id")
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and base.get_job.called)
            assert unwrapped(app.screen._client) is base
            await pilot.press("b")
            screen = await _list(app, pilot)
            # The app's own list: its client, its account-list state.
            assert unwrapped(screen._client) is base and screen._lister is app._lister and screen._clients is app._clients
            assert set(screen._job_by_item.values()) == {"j1", "j2"}
            await _settle(app, pilot)

    _run(run())
    base.for_job.assert_not_called()
