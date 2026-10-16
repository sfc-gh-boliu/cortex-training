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

"""The Resources tab explains a refused GPU metrics read in one line."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
import requests

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402

MARKER_MESSAGE = "Linked experiment is not found or not authorized for the training job"


@pytest.fixture(autouse=True)
def _isolate_tui_files(tmp_path, monkeypatch):
    # As conftest does for test_tui_* files: no writes to the real cache or home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(status="RUNNING", **extra) -> dict:
    cfg = {"n_gpus": 8.0}
    return {"job_id": "7", "status": status, **extra,
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "TRAINING", "training_config": cfg}]}


def _client(job=None):
    c = MagicMock()
    # The account-wide list isn't deployed (what accounts get today), so the job
    # list shows list_jobs, the configured schema's jobs.
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    job = job or _job()
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.database, c.schema = "TRAINING_DB", "RUNS"
    return c


def _http_error(status, body, reason):
    resp = requests.Response()
    resp.status_code, resp.reason, resp.url = status, reason, "http://test.local/api"
    resp._content = json.dumps(body).encode("utf-8")
    return requests.exceptions.HTTPError(f"{status} Server Error: {reason} for url: {resp.url}", response=resp)


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _text(screen, selector) -> str:
    return str(screen.query_one(selector, Static).render())


def _run(coro_fn, *args):
    asyncio.run(coro_fn(*args))


# ─── access messages ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("error, expected", [
    (_http_error(500, {"message": MARKER_MESSAGE}, "Internal Server Error"),
     "GPU metrics unavailable: this role can't read the job's Experiment (saved logs, GPU metrics and checkpoints "
     "live there); ask for access to the Experiment, or use the role that created the job"),
    (ConnectionError("003001 (42501): SQL access control error: Insufficient privileges to operate on experiment 'X'"),
     "GPU metrics unavailable: this role can't read the job's Experiment"),
    (_http_error(403, {"message": "Insufficient privileges"}, "Forbidden"),
     "GPU metrics unavailable: this role isn't allowed to read this job's GPU metrics (HTTP 403 Forbidden: "
     "Insufficient privileges)"),
    (RuntimeError("metrics [/]"), "GPU metrics unavailable: RuntimeError: metrics [/]"),  # anything else: as before
])
def test_the_resources_tab_explains_a_refused_read(error, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cold"))
    c = _client()
    c.download_metrics.side_effect = error

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(240, 40)) as pilot:
            assert await _wait(pilot, lambda: getattr(app.screen, "_current_source", None) is not None)
            await pilot.press("3")
            assert await _wait(pilot, lambda: c.download_metrics.called and not app.screen._resource_loading)
            assert expected in _text(app.screen, "#resource-status")
            await _settle(app, pilot)

    _run(run)
