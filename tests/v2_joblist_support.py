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

"""Shared helpers for tests/test_v2_joblist_*.py (this module is not collected).

Import ``isolated_tui_files`` into a test module to keep its TUI files (log
cache, display names, error log) out of the real home directory. Clients are
MagicMocks whose every list call is configured explicitly: an unconfigured
MagicMock answers with another MagicMock, which a walk could take for a page.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest
import requests

from cortex_training import AccountJobListUnavailable


@pytest.fixture(autouse=True)
def isolated_tui_files(tmp_path, monkeypatch):
    home = tmp_path / "isolated-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def job(jid: str, status: str = "CANCELLED", *, minutes_ago: float = 60, db=None, schema=None, subs: int = 1,
        **extra) -> dict:
    """A job record with ``subs`` training sub-jobs; ``db``/``schema``: a row of the account-wide list."""
    record = {"job_id": jid, "status": status, "created_at": iso(minutes_ago), "updated_at": iso(minutes_ago - 1),
              "comment": f"job {jid}", "submitted_by": "TEST_USER", "hardware": "H200",
              "sub_jobs": [{"sub_job_id": f"{jid}:training:{i}", "job_type": "TRAINING", "status": status,
                            "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8.0}}
                           for i in range(subs)], **extra}
    if db is not None:
        record.update(database_name=db, schema_name=schema)
    return record


def page(*jobs, token=None) -> dict:
    return {"jobs": list(jobs), "next_page_token": token}


def unavailable(reason: str, *, message: str | None = None, body=None) -> AccountJobListUnavailable:
    """The client's error for an unavailable account-wide list; ``body`` is the JSON the answer carried."""
    resp = requests.Response()
    resp.status_code = {"forbidden": 403, "rest_disabled": 405}.get(reason, 404)
    resp._content = b"" if body is None else json.dumps(body).encode()
    return AccountJobListUnavailable(message or f"account-level job listing is unavailable ({reason})",
                                     reason=reason, response=resp)


def http_error(status: int, message: str) -> requests.exceptions.HTTPError:
    resp = requests.Response()
    resp.status_code, resp.reason = status, {503: "Service Unavailable"}.get(status, "")
    resp._content = json.dumps({"message": message}).encode()
    return requests.exceptions.HTTPError(f"{status} Error", response=resp)


def client(schema_rows=(), *, pages=None) -> MagicMock:
    """A client of CORTEX_TRAINING_DB.PUBLIC whose list_jobs answers ``schema_rows``; ``pages``: the
    answers of list_jobs_page in turn (None: it must not be asked)."""
    c = MagicMock()
    c.database, c.schema, c._auth_provider = "CORTEX_TRAINING_DB", "PUBLIC", None
    rows = [dict(row) for row in schema_rows]
    c.list_jobs.side_effect = lambda **_: [dict(row) for row in rows]
    if pages is None:
        c.list_jobs_page.side_effect = AssertionError("the account-wide list was asked")
    else:
        c.list_jobs_page.side_effect = list(pages)
    c.iter_jobs.side_effect = AssertionError("the job list reads pages itself, never iter_jobs")
    c.get_job.side_effect = lambda jid, **_: {"job_id": jid, "status": "CANCELLED"}
    c.list_checkpoints.return_value = []
    return c


async def wait(pilot, predicate, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.02)
    return bool(predicate())


async def settle(app, pilot) -> None:
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def scope(app) -> str:
    """The list's scope in the subtitle: after the refresh status, before the key hints."""
    parts = (app.sub_title or "").split("  ·  ")
    return parts[1] if len(parts) == 3 else ""


def footer(screen) -> dict:
    """The footer's keys: key -> description, for the bindings it shows."""
    return {key: active.binding.description for key, active in screen.active_bindings.items()
            if active.binding.show}


def notices(app) -> list:
    return [str(n.message) for n in app._notifications]
