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

"""Shared helpers for tests/test_r1_cross_cutting_*.py (this module is not collected).

Import ``isolated_tui_files`` into a test module to keep its TUI files (log
cache, display names, error log, saved logs) out of the real home directory.
"""

import asyncio
import copy
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_tui_files(tmp_path, monkeypatch):
    home = tmp_path / "isolated-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def iso(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def job(job_id: str, status: str = "RUNNING", *, minutes_ago: float = 30, **fields) -> dict:
    """A job record as the per-schema list returns it, with one training sub-job."""
    record = {
        "job_id": job_id,
        "status": status,
        "created_at": iso(minutes_ago),
        "updated_at": iso(1),
        "state_changed_at": iso(1),
        "hardware": "H200",
        "submitted_by": "TEST_USER",
        "owner_role": "TRAINING_ROLE",
        "sub_jobs": [{
            "sub_job_id": f"{job_id}:training:0",
            "job_type": "TRAINING",
            "status": status,
            "model_name": "Qwen/Qwen3-8B",
            "training_config": {"n_gpus": 8.0, "optimizer": {"name": "AdamW", "lr": 1e-5}},
        }],
    }
    record.update(fields)
    return record


class FakeClient:
    """The read methods the TUI calls, answering from fixed records (no account-wide list)."""

    database = "CORTEX_TRAINING_DB"
    schema = "PUBLIC"

    def __init__(self, jobs):
        self.jobs = {j["job_id"]: j for j in jobs}
        self.calls = []

    def list_jobs(self, status=None, *, max_retries=None, request_timeout=None):
        self.calls.append(("list_jobs",))
        return copy.deepcopy(list(self.jobs.values()))

    def get_job(self, job_id):
        self.calls.append(("get_job", job_id))
        return copy.deepcopy(self.jobs[job_id])

    def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        return {"entries": [], "next_cursor": cursor or "", "eof": False}

    def download_metrics(self, job_id, output_dir, *, resume=False, **_options):
        return []

    def download_stdout_logs(self, job_id, output_dir, *, resume=False):
        return []

    def fetch_execution_logs(self, job_id, *, sub_job_id=None, filename=None):
        return []


def send_keys(app, *keys):
    """Deliver ``keys`` back to back, as a terminal delivers a burst: none waits for the one before."""
    from textual import events

    chars = {"enter": "\r", "escape": "\x1b"}
    for key in keys:
        event = events.Key(key, chars.get(key, key if len(key) == 1 else None))
        event.set_sender(app)
        app._driver.send_message(event)


async def wait_for(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def settle(app, pilot):
    """Stop the screens' workers and drain before the app is torn down."""
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def run(coro):
    asyncio.run(coro)
