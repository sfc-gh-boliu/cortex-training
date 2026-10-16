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

"""The job list's sources (account-wide list, fallback, re-probe) and per-location clients.

Every client here is a MagicMock whose list calls are configured explicitly:
an unconfigured MagicMock answers any call with another MagicMock, which a
walk could take for a page.
"""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest
import requests

from cortex_training import AccountJobListUnavailable
from cortex_training import InvalidPageToken
from cortex_training.tui import account
from cortex_training.tui.account import ACCOUNT
from cortex_training.tui.account import CAPPED
from cortex_training.tui.account import FALLBACK
from cortex_training.tui.account import PAGE_LIMIT
from cortex_training.tui.account import SCHEMA_ONLY
from cortex_training.tui.account import JobClients
from cortex_training.tui.account import JobLister
from cortex_training.tui.account import JobListError
from cortex_training.tui.account import ListScope
from cortex_training.tui.account import empty_text
from cortex_training.tui.account import loaded_text
from cortex_training.tui.account import scope_text
from cortex_training.tui.account import window_text

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
SCHEMA_ROWS = [{"job_id": "s1", "status": "RUNNING"}, {"job_id": "s2", "status": "FAILED"}]


def _row(i: int, db: str = "CORTEX_TRAINING_DB", schema: str = "PUBLIC") -> dict:
    return {"job_id": f"job-{i:04d}", "status": "CANCELLED", "database_name": db, "schema_name": schema}


def _unavailable(reason: str, *, body=None) -> AccountJobListUnavailable:
    """The client's error; ``body`` is the JSON the answer carried (None: no body, as a proxy may send)."""
    resp = requests.Response()
    resp.status_code = {"forbidden": 403, "rest_disabled": 405}.get(reason, 404)
    resp._content = b"" if body is None else json.dumps(body).encode()
    return AccountJobListUnavailable(f"account-level job listing: {reason}", reason=reason, response=resp)


SERVICE_403 = {"code": "510201", "message": "Cortex Training is not enabled for this account"}


def _http_error(status: int, message: str = "") -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code, resp.reason = status, {503: "Service Unavailable", 400: "Bad Request"}.get(status, "")
    resp._content = json.dumps({"message": message}).encode() if message else b""
    return requests.exceptions.HTTPError(f"{status} Error", response=resp)


class _Pages:
    """``list_jobs_page`` answers from a list of pages, chained by tokens ``t1``, ``t2``, ...

    ``active`` are the pages of the walk with ``status="active"`` (tokens ``a1``, ...);
    an exception there is raised.
    """

    def __init__(self, pages: list, active=None):
        self.pages = pages
        self.active = active if active is not None else [[]]
        self.calls: list = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("status") == "active":
            if isinstance(self.active, Exception):
                raise self.active
            pages, prefix = self.active, "a"
        else:
            pages, prefix = self.pages, "t"
        token = kwargs.get("page_token")
        index = 0 if token is None else int(token[1:])
        jobs = pages[index]
        more = index + 1 < len(pages)
        return {"jobs": jobs, "next_page_token": f"{prefix}{index + 1}" if more else None}

    def walks(self) -> list:
        """The status of each walk's first page, in order (None: the walk without one)."""
        return [call.get("status") for call in self.calls if call.get("page_token") is None]


def _client(pages=None, *, page_error=None, schema_rows=None) -> MagicMock:
    c = MagicMock()
    if page_error is not None:
        c.list_jobs_page.side_effect = page_error
    else:
        c.list_jobs_page.side_effect = _Pages(pages if pages is not None else [[_row(1), _row(2)]])
    c.list_jobs.return_value = list(SCHEMA_ROWS if schema_rows is None else schema_rows)
    c.iter_jobs.side_effect = AssertionError("the job list reads pages itself, never iter_jobs")
    return c


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ─── one walk per refresh ────────────────────────────────────────────────────


def test_a_walk_reads_bounded_pages_with_one_created_after_and_no_status_filter():
    pages = _Pages([[_row(1)], [], [_row(2), _row(3)]])  # a short page, then an empty one with a token
    c = _client()
    c.list_jobs_page.side_effect = pages
    listing = JobLister(c, account_wide=True).load(NOW)
    assert [j["job_id"] for j in listing.jobs] == ["job-0001", "job-0002", "job-0003"]
    assert listing.scope == ListScope(ACCOUNT) and not listing.scope.truncated
    assert len(pages.calls) == 3  # a complete walk needs no walk for the active jobs
    after = NOW - timedelta(days=183)
    for call, token in zip(pages.calls, [None, "t1", "t2"]):
        # The same filters on every page (the token is bound to them), and no status
        # filter, which would drop the jobs whose status the service can't read.
        assert call == {"created_after": after, "page_size": 500, "page_token": token, "max_retries": 2,
                        "request_timeout": (10.0, 45.0)}
    c.list_jobs.assert_not_called()


def test_a_walk_stops_at_the_cap_and_says_more_exist(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_CAP", 5)
    pages = _Pages([[_row(i) for i in range(3)], [_row(i) for i in range(3, 6)], [_row(i) for i in range(6, 9)]])
    c = _client()
    c.list_jobs_page.side_effect = pages
    listing = JobLister(c, account_wide=True).load(NOW)
    assert [j["job_id"] for j in listing.jobs] == [f"job-{i:04d}" for i in range(5)]  # the newest five
    assert listing.scope == ListScope(ACCOUNT, stopped=CAPPED, walked=5)
    assert pages.walks() == [None, "active"] and len(pages.calls) == 3  # never walks on to the end


def _live(i: int, db: str = "TEAM_DB", schema: str = "EXPERIMENTS") -> dict:
    return {**_row(i, db, schema), "status": "RUNNING"}


def test_a_list_cut_short_adds_the_older_active_jobs(monkeypatch):
    # Watching running jobs is what the list is for: one created before the newest
    # jobs must not drop out of LIVE because the walk stopped at the cap.
    monkeypatch.setattr(account, "ACCOUNT_LIST_CAP", 4)
    newest = [_row(i) for i in range(4)]
    pages = _Pages([newest[:2], newest[2:], [_row(8), _row(9)]], active=[[_live(1)], [_live(50), _live(51)]])
    c = _client()
    c.list_jobs_page.side_effect = pages
    listing = JobLister(c, account_wide=True).load(NOW)
    # job-0001 is among the newest already; job-0050 and job-0051 are older and still running.
    assert [j["job_id"] for j in listing.jobs] == ["job-0000", "job-0001", "job-0002", "job-0003", "job-0050",
                                                  "job-0051"]
    assert listing.scope == ListScope(ACCOUNT, stopped=CAPPED, walked=4, active_added=2)
    after = NOW - timedelta(days=183)
    active = [call for call in pages.calls if call.get("status") == "active"]
    # The same created_after as the walk itself, the same bounds, and its own tokens.
    assert active == [{"created_after": after, "status": "active", "page_size": 500, "page_token": token,
                       "max_retries": 2, "request_timeout": (10.0, 45.0)} for token in (None, "a1")]


def test_a_failed_walk_for_the_active_jobs_keeps_the_list_and_says_some_may_be_missing(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_CAP", 2)
    pages = _Pages([[_row(0), _row(1)], [_row(2)]], active=_http_error(503, "Control Plane unavailable"))
    c = _client()
    c.list_jobs_page.side_effect = pages
    listing = JobLister(c, account_wide=True).load(NOW)
    assert [j["job_id"] for j in listing.jobs] == ["job-0000", "job-0001"]
    assert listing.scope == ListScope(ACCOUNT, stopped=CAPPED, walked=2, active_missing=True)
    assert scope_text(listing.scope, location="X.Y") == "newest 2 (more exist; older active jobs may be missing)"


def test_a_walk_for_the_active_jobs_that_is_cut_short_too_says_some_may_be_missing(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_MAX_PAGES", 2)
    pages = _Pages([[_row(0)], [_row(1)], [_row(2)]], active=[[_live(7)], [], [_live(8)]])
    c = _client()
    c.list_jobs_page.side_effect = pages
    listing = JobLister(c, account_wide=True).load(NOW)
    assert [j["job_id"] for j in listing.jobs] == ["job-0000", "job-0001", "job-0007"]
    assert listing.scope == ListScope(ACCOUNT, stopped=PAGE_LIMIT, walked=2, pages=2, active_added=1,
                                      active_missing=True)


def test_a_walk_that_reaches_the_cap_on_its_last_page_is_complete(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_CAP", 4)
    c = _client([[_row(0), _row(1)], [_row(2), _row(3)]])
    listing = JobLister(c, account_wide=True).load(NOW)
    assert len(listing.jobs) == 4 and not listing.scope.truncated


def test_the_page_limit_bounds_a_walk_of_empty_pages(monkeypatch):
    monkeypatch.setattr(account, "ACCOUNT_LIST_MAX_PAGES", 3)
    pages = _Pages([[], [], [], [_row(1)]], active=[[], [], [], [_live(2)]])
    c = _client()
    c.list_jobs_page.side_effect = pages
    listing = JobLister(c, account_wide=True).load(NOW)
    # Stopped by the page limit, not the cap, with nothing listed: the scope says so,
    # so the screen can say more may exist instead of "no jobs".
    assert listing.jobs == []
    assert listing.scope == ListScope(ACCOUNT, stopped=PAGE_LIMIT, walked=0, pages=3, active_missing=True)
    assert len(pages.calls) == 6  # three pages a walk, never more


def test_the_default_bounds_are_the_documented_ones():
    assert (account.ACCOUNT_LIST_CAP, account.ACCOUNT_PAGE_SIZE) == (2000, 500)
    assert (account.ACCOUNT_REPROBE_SECONDS, account.ACCOUNT_MANUAL_REPROBE_SECONDS) == (600, 60)
    assert account.ACCOUNT_LIST_MAX_PAGES == 20
    assert (account.REFRESH_MAX_RETRIES, account.REFRESH_TIMEOUT) == (2, (10.0, 45.0))


def test_a_page_seen_twice_lists_each_job_once():
    c = _client([[_row(1), _row(2)], [_row(2), _row(3)]])
    assert [j["job_id"] for j in JobLister(c, account_wide=True).load(NOW).jobs] == ["job-0001", "job-0002", "job-0003"]


def test_a_walk_stops_when_the_worker_is_cancelled():
    pages = _Pages([[_row(1)], [_row(2)], [_row(3)]])
    c = _client()
    c.list_jobs_page.side_effect = pages
    asked = []

    def cancelled() -> bool:
        asked.append(1)
        return len(asked) > 2  # cancelled after two pages

    assert JobLister(c, account_wide=True).load(NOW, is_cancelled=cancelled) is None
    assert len(pages.calls) == 2


@pytest.mark.parametrize("page", [MagicMock(), None, [], {"jobs": "x"}, {"next_page_token": "t1"}])
def test_a_page_that_is_not_a_job_list_fails_the_refresh_instead_of_looping(page):
    c = _client()
    c.list_jobs_page.side_effect = None
    c.list_jobs_page.return_value = page
    with pytest.raises(JobListError, match="not a job list") as raised:
        JobLister(c, account_wide=True).load(NOW)
    assert isinstance(raised.value.__cause__, ValueError) and not raised.value.refused
    assert c.list_jobs_page.call_count == 1


def test_an_expired_page_token_restarts_the_walk_once_from_the_first_page():
    pages = _Pages([[_row(1)], [_row(2)]])
    answers = iter(["page", InvalidPageToken("Invalid page_token"), "page", "page"])

    def list_jobs_page(**kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            pages.calls.append(kwargs)
            raise answer
        return pages(**kwargs)

    c = _client()
    c.list_jobs_page.side_effect = list_jobs_page
    listing = JobLister(c, account_wide=True).load(NOW)
    assert [j["job_id"] for j in listing.jobs] == ["job-0001", "job-0002"]
    assert [call["page_token"] for call in pages.calls] == [None, "t1", None, "t1"]
    assert len({call["created_after"] for call in pages.calls}) == 1  # computed once per walk


def test_a_second_expired_token_fails_the_refresh():
    c = _client()
    calls = []

    def list_jobs_page(**kwargs):
        calls.append(kwargs["page_token"])
        if kwargs["page_token"]:
            raise InvalidPageToken("Invalid page_token")
        return {"jobs": [_row(1)], "next_page_token": "t1"}

    c.list_jobs_page.side_effect = list_jobs_page
    lister = JobLister(c, account_wide=True)
    with pytest.raises(JobListError) as error:
        lister.load(NOW)
    assert calls == [None, "t1", None, "t1"]
    assert isinstance(error.value.__cause__, InvalidPageToken)
    assert str(error.value).startswith("the account-wide job list failed: InvalidPageToken")


# ─── fallback to the configured schema, and asking again ─────────────────────


@pytest.mark.parametrize("reason", [AccountJobListUnavailable.NOT_DEPLOYED, AccountJobListUnavailable.DISABLED])
def test_a_route_that_is_not_there_falls_back_to_the_schema_and_asks_again_after_ten_minutes(reason):
    clock = _Clock()
    c = _client(page_error=_unavailable(reason))
    lister = JobLister(c, account_wide=True, clock=clock)
    listing = lister.load(NOW)
    assert listing.jobs == SCHEMA_ROWS and listing.scope == ListScope(FALLBACK, reason=reason)
    assert lister.fallback_reason == reason
    # Within ten minutes, refreshes list the schema without asking the account-level list.
    clock.now += 590
    assert lister.load(NOW).scope.kind == FALLBACK
    assert c.list_jobs_page.call_count == 1 and c.list_jobs.call_count == 2
    # Then it asks again (the tenth 60-second refresh can come a moment early); once it
    # answers, the list is account-wide again.
    clock.now += 9.9
    c.list_jobs_page.side_effect = _Pages([[_row(7, "TEAM_DB", "EXPERIMENTS")]])
    listing = lister.load(NOW)
    assert listing.scope == ListScope(ACCOUNT) and listing.jobs[0]["job_id"] == "job-0007"
    assert c.list_jobs.call_count == 2 and lister.fallback_reason is None


def test_asking_again_while_still_unavailable_waits_another_ten_minutes():
    clock = _Clock()
    c = _client(page_error=_unavailable("not_deployed"))
    lister = JobLister(c, account_wide=True, clock=clock)
    lister.load(NOW)
    clock.now += 601
    c.list_jobs_page.side_effect = _unavailable("disabled")  # the route arrived, still disabled
    assert lister.load(NOW).scope == ListScope(FALLBACK, reason="disabled")
    clock.now += 300
    assert lister.load(NOW).scope.kind == FALLBACK
    assert c.list_jobs_page.call_count == 2


@pytest.mark.parametrize("error", [_http_error(503, "Control Plane unavailable"),
                                   requests.exceptions.ReadTimeout("slow")])
def test_asking_again_that_fails_another_way_keeps_listing_the_schema(error):
    clock = _Clock()
    c = _client(page_error=_unavailable("not_deployed"))
    lister = JobLister(c, account_wide=True, clock=clock)
    lister.load(NOW)
    clock.now += 601
    c.list_jobs_page.side_effect = error
    listing = lister.load(NOW)
    assert listing.jobs == SCHEMA_ROWS and listing.scope == ListScope(FALLBACK, reason="not_deployed")
    clock.now += 590  # and it waits ten minutes from this ask
    assert lister.load(NOW).scope.kind == FALLBACK and c.list_jobs_page.call_count == 2


@pytest.mark.parametrize("reason, why", [("forbidden", "account-wide list: HTTP 403"),
                                         ("rest_disabled", "account-wide list: HTTP 405")])
def test_asking_again_that_is_refused_keeps_the_schema_and_says_why(reason, why):
    # The configured schema's list keeps working, so it stays; the header no
    # longer says the route isn't there yet.
    clock = _Clock()
    c = _client(page_error=_unavailable("not_deployed"))
    lister = JobLister(c, account_wide=True, clock=clock)
    lister.load(NOW)
    clock.now += 601
    c.list_jobs_page.side_effect = _unavailable(reason, body=SERVICE_403)
    listing = lister.load(NOW)
    assert listing.jobs == SCHEMA_ROWS and listing.scope == ListScope(FALLBACK, reason=reason)
    assert scope_text(listing.scope, location="DB.S") == f"this schema only: DB.S ({why})"
    assert lister.fallback_reason == reason


def test_r_asks_again_after_a_minute_and_the_timer_after_ten():
    clock = _Clock()
    c = _client(page_error=_unavailable("not_deployed"))
    lister = JobLister(c, account_wide=True, clock=clock)
    lister.load(NOW)
    clock.now += 50
    assert lister.load(NOW, manual=True).scope.kind == FALLBACK  # r within a minute: not again yet
    assert c.list_jobs_page.call_count == 1
    clock.now += 11
    assert lister.load(NOW).scope.kind == FALLBACK  # the timer, at 61 s: not yet
    assert c.list_jobs_page.call_count == 1
    c.list_jobs_page.side_effect = _Pages([[_row(7, "TEAM_DB", "EXPERIMENTS")]])
    assert lister.load(NOW, manual=True).scope == ListScope(ACCOUNT)  # r at 61 s asks
    assert c.list_jobs_page.call_count == 2


@pytest.mark.parametrize("error", [_unavailable("forbidden", body=SERVICE_403),
                                   _unavailable("forbidden", body={"error": {"message": "Trial ended"}}),
                                   _unavailable("rest_disabled"), _unavailable("rest_disabled", body=SERVICE_403)])
def test_a_refused_account_list_is_the_lists_error_not_a_fallback(error):
    c = _client(page_error=error)
    lister = JobLister(c, account_wide=True)
    with pytest.raises(JobListError) as raised:
        lister.load(NOW)
    assert raised.value.__cause__ is error and raised.value.refused
    # The service refused, or turned the REST API off: the schema's list would
    # most likely fail the same way, so it isn't asked.
    c.list_jobs.assert_not_called()
    assert str(raised.value) == str(error)  # the client's own one-line message
    assert lister.fallback_reason is None


@pytest.mark.parametrize("body", [None, {"detail": "blocked"}, "<html>Forbidden</html>"])
def test_a_403_without_the_services_message_tries_the_schema_once(body):
    # A network policy or a proxy in front of the account-level route alone can
    # send that; the per-schema list may still answer.
    error = _unavailable("forbidden", body=None if isinstance(body, str) else body)
    if isinstance(body, str):
        error.response._content = body.encode()
    clock = _Clock()
    c = _client(page_error=error)
    lister = JobLister(c, account_wide=True, clock=clock)
    listing = lister.load(NOW)
    assert listing.jobs == SCHEMA_ROWS and listing.scope == ListScope(FALLBACK, reason="forbidden")
    assert lister.fallback_reason == "forbidden"
    clock.now += 300
    lister.load(NOW)  # in fallback now: asked again after ten minutes, like the others
    assert c.list_jobs_page.call_count == 1 and c.list_jobs.call_count == 2


def test_a_403_without_the_services_message_stands_when_the_schema_fails_too():
    error = _unavailable("forbidden")
    c = _client(page_error=error)
    c.list_jobs.side_effect = _http_error(403, "")
    lister = JobLister(c, account_wide=True)
    with pytest.raises(JobListError) as raised:
        lister.load(NOW)
    # Both refusals, and no --schema-only hint: the schema's list was refused too.
    assert raised.value.__cause__ is error and not raised.value.refused
    assert str(raised.value) == f"{error}; and list_jobs answered HTTP 403"
    assert c.list_jobs.call_count == 1 and lister.fallback_reason is None


def test_other_account_list_failures_are_the_lists_error():
    c = _client(page_error=_http_error(503, "Control Plane unavailable"))
    lister = JobLister(c, account_wide=True)
    with pytest.raises(JobListError) as raised:
        lister.load(NOW)
    c.list_jobs.assert_not_called()
    assert isinstance(raised.value.__cause__, requests.exceptions.HTTPError) and not raised.value.refused
    assert str(raised.value) == (
        "the account-wide job list answered HTTP 503 Service Unavailable: Control Plane unavailable")


def test_a_failing_schema_list_names_list_jobs():
    c = _client(page_error=_unavailable("not_deployed"))
    c.list_jobs.side_effect = _http_error(400, "Schema X.Y is not found\nor not authorized")
    lister = JobLister(c)
    with pytest.raises(JobListError) as raised:
        lister.load(NOW)
    text = str(raised.value)
    assert text == "list_jobs answered HTTP 400 Bad Request: Schema X.Y is not found or not authorized"
    assert "\n" not in text


def test_schema_only_never_asks_the_account_level_list():
    c = _client()
    lister = JobLister(c, schema_only=True)
    for _ in range(3):
        listing = lister.load(NOW)
        assert listing.jobs == SCHEMA_ROWS and listing.scope == ListScope(SCHEMA_ONLY)
    c.list_jobs_page.assert_not_called()


def test_a_client_without_the_account_level_list_lists_the_schema():
    class OlderClient:
        def list_jobs(self):
            return list(SCHEMA_ROWS)

    listing = JobLister(OlderClient(), account_wide=True).load(NOW)
    assert listing.jobs == SCHEMA_ROWS and listing.scope == ListScope(FALLBACK)
    assert scope_text(listing.scope, location="DB.S") == "this schema only: DB.S (no account-wide list)"


def test_the_schema_list_is_bounded_like_a_page_when_the_client_takes_bounds():
    # A failing refresh must not spend ten retries and a ten-minute read timeout
    # before the list says so: the fallback is what every account gets today.
    calls = []

    class Client:
        def list_jobs(self, status=None, *, max_retries=None, request_timeout=None):
            calls.append({"status": status, "max_retries": max_retries, "request_timeout": request_timeout})
            return list(SCHEMA_ROWS)

    listing = JobLister(Client(), schema_only=True).load(NOW)
    assert listing.jobs == SCHEMA_ROWS
    assert calls == [{"status": None, "max_retries": 2, "request_timeout": (10.0, 45.0)}]


def test_a_real_clients_list_jobs_takes_the_bounds():
    from cortex_training import CortexTrainingClient

    client = CortexTrainingClient(base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="PUBLIC")
    assert account._takes_bounds(client.list_jobs)
    assert not account._takes_bounds(MagicMock().list_jobs)  # a mock without the parameters gets none


def test_the_schema_list_keeps_only_job_records():
    c = _client(page_error=_unavailable("not_deployed"), schema_rows=[SCHEMA_ROWS[0], "junk", None])
    assert JobLister(c).load(NOW).jobs == [SCHEMA_ROWS[0]]


def test_a_failure_is_one_bounded_line():
    c = _client(page_error=_unavailable("not_deployed"))
    c.list_jobs.side_effect = RuntimeError("x" * 1000 + "\nsecond line")
    lister = JobLister(c)
    with pytest.raises(JobListError) as raised:
        lister.load(NOW)
    text = str(raised.value)
    assert "\n" not in text and len(text) <= 300 and text.endswith("…")


# ─── the scope, in words ─────────────────────────────────────────────────────


def test_scope_text():
    where = "CORTEX_TRAINING_DB.PUBLIC"
    assert scope_text(ListScope(ACCOUNT), location=where) == "all schemas"
    assert scope_text(ListScope(ACCOUNT), location=where, only='"ml"."dev"') == 'all schemas (all in "ml"."dev")'
    capped = ListScope(ACCOUNT, stopped=CAPPED, walked=2000, active_added=3)
    assert scope_text(capped, location=where) == "newest 2,000 + every active job (more exist)"
    assert scope_text(ListScope(ACCOUNT, stopped=CAPPED, walked=2000, active_missing=True), location=where) == (
        "newest 2,000 (more exist; older active jobs may be missing)")
    assert scope_text(ListScope(ACCOUNT, stopped=PAGE_LIMIT, walked=1, pages=20), location=where) == (
        "first 20 pages: 1 job + every active job (more may exist)")
    assert scope_text(ListScope(SCHEMA_ONLY), location=where) == (
        "this schema only: CORTEX_TRAINING_DB.PUBLIC (--schema-only)")
    assert scope_text(ListScope(FALLBACK, reason="not_deployed"), location=where) == (
        "this schema only: CORTEX_TRAINING_DB.PUBLIC (no account-wide list yet)")
    assert scope_text(ListScope(FALLBACK, reason="disabled"), location=where) == (
        "this schema only: CORTEX_TRAINING_DB.PUBLIC (account-wide list not enabled)")


def test_what_a_list_cut_short_holds_in_words():
    capped = ListScope(ACCOUNT, stopped=CAPPED, walked=2000, active_added=3)
    limited = ListScope(ACCOUNT, stopped=PAGE_LIMIT, walked=0, pages=20, active_missing=True)
    assert window_text(ListScope(ACCOUNT), recent=239, hidden=0) == "last 6 months (239 jobs)"
    assert window_text(None, recent=1_200, hidden=4) == "last 6 months (1,200 jobs, 4 older hidden)"
    assert window_text(capped, recent=2003, hidden=0) == "newest 2,000 jobs + 3 older active"
    assert window_text(limited, recent=0, hidden=0) == "first 20 pages (0 jobs)"
    assert loaded_text(ListScope(ACCOUNT)) == "" and loaded_text(ListScope(SCHEMA_ONLY)) == ""
    assert loaded_text(capped) == "the newest 2,000 of the last 6 months and every active job; more exist"
    assert loaded_text(limited) == (
        "what the first 20 pages held; more may exist; older active jobs may be missing")
    # Never "No jobs ... in any database and schema" for a walk that stopped early.
    assert empty_text(limited) == (
        "The account-wide list returned no jobs to show in its first 20 pages; more may exist. r reads it again; "
        "--database DB --schema SCHEMA --schema-only lists one schema's jobs.")


# ─── one client per job location ─────────────────────────────────────────────


def test_job_clients_make_one_child_per_location_and_use_the_base_without_one():
    base = MagicMock()
    children = {}

    def for_job(row):
        child = MagicMock(name=f"child {row['database_name']}.{row['schema_name']}")
        children[(row["database_name"], row["schema_name"])] = child
        return child

    base.for_job.side_effect = for_job
    clients = JobClients(base)
    team_a = clients.for_job(_row(1, "TEAM_DB", "EXPERIMENTS"))
    team_b = clients.for_job(_row(2, "TEAM_DB", "EXPERIMENTS"))
    lower = clients.for_job(_row(3, "ml", "dev"))
    assert team_a is team_b is children[("TEAM_DB", "EXPERIMENTS")]
    assert lower is children[("ml", "dev")] and lower is not team_a
    assert base.for_job.call_count == 2
    # A row without a location (the configured schema's list, a job opened by id) uses the base.
    for row in ({"job_id": "s1"}, {"job_id": "s1", "database_name": "DB"}, {"job_id": "s1", "database_name": "",
                                                                             "schema_name": "S"}, None):
        assert clients.for_job(row) is base
    assert base.for_job.call_count == 2


def test_a_client_without_for_job_uses_its_own():
    class OlderClient:
        database, schema = "CORTEX_TRAINING_DB", "PUBLIC"

    base = OlderClient()
    assert JobClients(base).for_job(_row(1, "TEAM_DB", "EXPERIMENTS")) is base


def test_a_child_that_cannot_be_made_is_an_error_not_the_configured_client():
    # The TUI's own client would send the job's calls to the configured location,
    # where each would fail as "not found".
    base = MagicMock()
    base.for_job.side_effect = [ValueError("bad identifier"), MagicMock(name="child")]
    clients = JobClients(base)
    with pytest.raises(ValueError, match="bad identifier"):
        clients.for_job(_row(1, "TEAM_DB", "EXPERIMENTS"))
    # Not cached: the next open tries again.
    assert clients.for_job(_row(1, "TEAM_DB", "EXPERIMENTS")) is not base and base.for_job.call_count == 2


@pytest.mark.parametrize("database, schema, home", [
    ("CORTEX_TRAINING_DB", "PUBLIC", ("CORTEX_TRAINING_DB", "PUBLIC")),
    ("cortex_training_db", "public", ("CORTEX_TRAINING_DB", "PUBLIC")),  # bare: uppercase, as SQL resolves it
    ('"ml"', '"dev"', ("ml", "dev")),
])
def test_job_clients_know_the_configured_location(database, schema, home):
    base = MagicMock()
    base.database, base.schema = database, schema
    clients = JobClients(base)
    assert clients.home == home
    assert not clients.elsewhere({"job_id": "s1"})  # the configured schema's own rows
    assert not clients.elsewhere(_row(1, *home))
    assert clients.elsewhere(_row(2, "TEAM_DB", "EXPERIMENTS"))


def test_without_a_configured_location_every_located_row_is_elsewhere():
    clients = JobClients(MagicMock())  # a mock's database is no name
    assert clients.home is None and clients.elsewhere(_row(1)) and not clients.elsewhere({"job_id": "s1"})
