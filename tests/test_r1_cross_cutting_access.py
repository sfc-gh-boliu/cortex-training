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

"""A role that can't read the job's Experiment gets the same clear line on every tab.

Two ways to be refused: the service can't resolve the Experiment for the role
(its 500, worded one way by the checkpoint routes and another by the
experiment-run route that saved logs and GPU metrics start with), or Snowflake
refuses the SQL on the Experiment (a MONITOR-only role), when the service
still lists the checkpoints. Refusals about other objects keep their own text.
"""

import types

import pytest
import requests

from cortex_training.tui import views
from tests.r1_cross_cutting_support import FakeClient
from tests.r1_cross_cutting_support import NOW
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401
from tests.r1_cross_cutting_support import job
from tests.r1_cross_cutting_support import run
from tests.r1_cross_cutting_support import settle
from tests.r1_cross_cutting_support import wait_for

ASK = "ask for access to the Experiment, or use the role that created the job"
DENIED = f"this role can't read the job's Experiment (saved logs, GPU metrics and checkpoints live there); {ASK}"
MISSING = ("this role can't read the job's Experiment, or it no longer exists (saved logs, GPU metrics and "
           f"checkpoints live there); {ASK}")
SQL = f"this role can't read the job's Experiment, where saved logs and GPU metrics live (the checkpoints command works); {ASK}"


def _http_error(status, message):
    response = types.SimpleNamespace(status_code=status, reason="Internal Server Error",
                                     json=lambda: {"code": str(status), "message": message},
                                     text=f'{{"code": "{status}", "message": "{message}"}}')
    return requests.HTTPError(f"{status} Server Error: for url: http://test.invalid/x", response=response)


EXPERIMENT_RUN_500 = _http_error(500, "Linked experiment is not found for job j1")
CHECKPOINTS_500 = _http_error(500, "Linked experiment is not found or not authorized for the training job")
SQL_REFUSED = RuntimeError("003001 (42501): SQL access control error:\nInsufficient privileges to operate on "
                           "experiment 'CORTEX_TRAINING_DB.PUBLIC.EXP'")


@pytest.mark.parametrize("exc,kind", [
    (CHECKPOINTS_500, "denied"),
    (EXPERIMENT_RUN_500, "missing"),
    (SQL_REFUSED, "sql"),
    (RuntimeError("002003 (02000): SQL compilation error:\nExperiment 'DB.S.E' does not exist or not authorized."),
     "sql"),
    (RuntimeError("LIST 'snow://experiment/DB.S.E/versions/R/_logs/': does not exist or not authorized"), "sql"),
])
def test_one_classifier_names_the_refusal(exc, kind):
    assert views.experiment_access(exc) == kind
    line = {"denied": DENIED, "missing": MISSING, "sql": SQL}[kind]
    for subject in ("saved logs", "execution log", "GPU metrics"):
        assert views.access_hint(exc, subject) == line


@pytest.mark.parametrize("exc", [
    RuntimeError("003001 (42501): SQL access control error: Insufficient privileges to operate on schema EXPERIMENTS"),
    RuntimeError("LIST 'snow://experiment/DB.S.E/versions/R/_logs/' failed: SQL access control error: "
                 "Insufficient privileges to operate on warehouse 'WH'"),
    _http_error(500, "experiment store unavailable"),
])
def test_refusals_about_other_objects_are_not_blamed_on_the_experiment(exc):
    assert views.experiment_access(exc) is None
    assert views.access_hint(exc, "saved logs") is None


class _Refused(FakeClient):
    """Saved logs, execution logs and GPU metrics fail with ``read``."""

    def __init__(self, jobs, *, read):
        super().__init__(jobs)
        self.read = read

    def download_stdout_logs(self, job_id, output_dir, *, resume=False):
        raise self.read

    def fetch_execution_logs(self, job_id, *, sub_job_id=None, filename=None):
        raise self.read

    def download_metrics(self, job_id, output_dir, *, resume=False, **_options):
        raise self.read


def _walk(client, *, logs, resources):
    pytest.importorskip("textual")
    from textual.widgets import DataTable
    from textual.widgets import Static

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen
    from cortex_training.tui.screens import JobDetailScreen
    from cortex_training.tui.screens import JobSummaryScreen

    def text(screen, selector):
        return " ".join(str(screen.query_one(selector, Static).render()).split())

    async def go():
        app = CortexTrainingLogTUI(client, poll_interval=0.01, now=NOW)
        async with app.run_test(size=(220, 50)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            table = app.screen.query_one("#jobs", DataTable)
            app.screen.set_focus(table)
            table.move_cursor(row=1, animate=False)
            await pilot.pause()
            await pilot.press("enter")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            await pilot.press("l")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                  and any(logs in line for line in app.screen._shown_lines))
            detail = app.screen
            await pilot.press("3")
            assert await wait_for(pilot, lambda: resources in text(detail, "#resource-status"), tries=400), (
                text(detail, "#resource-status"))
            await settle(app, pilot)

    run(go())


def test_the_experiment_run_answer_reads_the_same_on_every_tab():
    """Logs and Resources get the experiment-run route's wording: each tab
    explains, none shows the raw 500."""
    client = _Refused([job("j1", "FAILED", minutes_ago=90)], read=EXPERIMENT_RUN_500)
    _walk(client, logs=f"— {MISSING} —", resources=f"GPU metrics unavailable: {MISSING}")


def test_a_sql_refusal_says_the_checkpoints_still_list():
    client = _Refused([job("j1", "FAILED", minutes_ago=90)], read=SQL_REFUSED)
    _walk(client, logs=f"— {SQL} —", resources=f"GPU metrics unavailable: {SQL}")
