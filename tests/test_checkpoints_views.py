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

"""The TUI's access messages for a refused read, and the rules copied from the client."""

from __future__ import annotations

import json

import pytest
import requests
from snowflake.connector.errors import ProgrammingError

import cortex_training._cli as cli
from cortex_training.tui import views as v

MARKER_MESSAGE = "Linked experiment is not found or not authorized for the training job"
HINT = ("this role can't read the job's Experiment (saved logs, GPU metrics and checkpoints live there); "
        "ask for access to the Experiment, or use the role that created the job")
# Snowflake refusing the SQL on the Experiment: the service still lists the checkpoints.
SQL_HINT = ("this role can't read the job's Experiment, where saved logs and GPU metrics live (the checkpoints "
            "command works); ask for access to the Experiment, or use the role that created the job")


def _http_error(status: int, body, reason: str = "Internal Server Error") -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    resp.reason = reason
    resp.url = "http://test.local/api"
    resp._content = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Server Error: {reason} for url: {resp.url}", response=resp)


# ─── access messages ─────────────────────────────────────────────────────────


def test_experiment_access_hint_reads_the_exception_text_and_the_reply():
    assert v.experiment_access_hint(RuntimeError(f"SQL compilation error: {MARKER_MESSAGE}")) == HINT
    assert v.experiment_access_hint(RuntimeError("something else")) is None
    # An HTTP error's own text is only its status line; the message is in the reply, which is read too.
    assert v.experiment_access_hint(_http_error(500, {"message": MARKER_MESSAGE})) == HINT


@pytest.mark.parametrize("exc", [
    _http_error(500, {"message": MARKER_MESSAGE}),
    _http_error(500, {"code": "500", "message": f"Training job error: {MARKER_MESSAGE}."}),
    _http_error(500, MARKER_MESSAGE.encode("utf-8")),  # not JSON
    RuntimeError(MARKER_MESSAGE),
])
def test_an_experiment_access_failure_is_explained_in_any_form(exc):
    assert v.access_hint(exc, "GPU metrics") == HINT


@pytest.mark.parametrize("body", [
    {"message": "Insufficient privileges to operate on job"},
    b"SQL access control error: Insufficient privileges to operate on job",  # not JSON
    {"code": "390201", "message": "The requested job does not exist or not authorized."},
])
def test_a_forbidden_read_says_the_role_is_not_allowed(body):
    exc = _http_error(403, body, reason="Forbidden")
    assert v.access_hint(exc, "GPU metrics").startswith(
        "this role isn't allowed to read this job's GPU metrics (HTTP 403 Forbidden")


@pytest.mark.parametrize("body", [b"", b"<html><body><h1>403 Forbidden</h1>Request blocked.</body></html>"])
def test_a_403_that_names_no_privilege_keeps_the_standard_text(body):
    # A proxy or network gate can answer 403 with an empty or unrelated body: not the role.
    exc = _http_error(403, body, reason="Forbidden")
    assert v.access_hint(exc, "GPU metrics") is None
    assert v.describe_error(exc) == "HTTP 403 Forbidden"


def test_other_failures_keep_the_standard_error_text():
    assert v.access_hint(_http_error(500, {"message": "boom"}), "GPU metrics") is None
    assert v.describe_error(_http_error(500, {"message": "boom"})) == "HTTP 500 Internal Server Error: boom"
    assert v.describe_error(requests.exceptions.ConnectionError("reset")) == "ConnectionError: reset"


@pytest.mark.parametrize("exc", [
    # What the Snowflake connector raises when the LIST or GET on the Experiment is refused.
    ProgrammingError(msg="SQL access control error:\nInsufficient privileges to operate on experiment 'EXP'",
                     errno=3001, sqlstate="42501"),
    ProgrammingError(msg="SQL compilation error:\nExperiment 'DB.SCH.EXP' does not exist or not authorized.",
                     errno=2003, sqlstate="02000"),
    ConnectionError("003001 (42501): SQL access control error: Insufficient privileges to operate on experiment 'X'"),
    RuntimeError("LIST 'snow://experiment/DB.SCH.EXP/versions/RUN/_logs/': does not exist or not authorized"),
    RuntimeError("Experiment run 'RUN' does not exist or not authorized."),
])
def test_every_tab_explains_a_refused_experiment_read(exc):
    # Saved logs and GPU metrics are SQL on the Experiment, refused here; the checkpoints still list.
    assert v.access_hint(exc, "GPU metrics") == SQL_HINT
    assert v.access_hint(_http_error(500, {"message": MARKER_MESSAGE}), "GPU metrics") == HINT


@pytest.mark.parametrize("exc", [
    ProgrammingError(msg="SQL access control error:\nInsufficient privileges to operate on warehouse 'WH'",
                     errno=3001, sqlstate="42501"),
    _http_error(400, {"code": "517602", "message": "Schema EXPERIMENTS_DB.PUBLIC is not found or not authorized"}),
    RuntimeError("experiment-run response missing field: experiment_name"),
    # The Experiment named, but the refusal is about another object.
    RuntimeError("LIST 'snow://experiment/DB.S.E/versions/R/_metrics/' failed: SQL access control error: "
                 "Insufficient privileges to operate on warehouse 'WH'"),
    RuntimeError("LIST on experiment 'E' failed: Insufficient privileges to operate on schema 'S'"),
])
def test_denials_about_other_objects_keep_their_own_text(exc):
    assert v.access_hint(exc, "GPU metrics") is None


def test_a_forbidden_answer_names_what_the_tab_shows():
    exc = _http_error(403, {"message": "Insufficient privileges"}, reason="Forbidden")
    assert v.access_hint(exc, "GPU metrics") == (
        "this role isn't allowed to read this job's GPU metrics (HTTP 403 Forbidden: Insufficient privileges)")


# ─── the rules copied from the client ────────────────────────────────────────


def test_the_copied_rules_agree_with_the_client():
    # The marker and the identifier rule are copied rather than imported (the CLI
    # formats errors without importing the client); this keeps the copies in step.
    from cortex_training import client as nc

    assert nc._EXPERIMENT_ACCESS_DENIED == cli._EXPERIMENT_ACCESS_DENIED == v.EXPERIMENT_ACCESS_MARKER
    for name in ("TRAINING_DB", "PUBLIC", "team db", 'we"ird', "lower", "_X$1", "1ABC", ""):
        assert v._sql_identifier(name) == nc._sql_identifier(name), name
