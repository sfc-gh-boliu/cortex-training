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

"""App tests for the job list, job preview, and tabbed detail screens, driven
by the offline :class:`DemoClient`."""

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Input  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.demo import DemoClient  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.stores import AliasStore  # noqa: E402


@pytest.fixture
def demo(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    client = DemoClient()
    client.seed_local(aliases=AliasStore())
    return client


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


def _static(screen, selector) -> str:
    return str(screen.query_one(selector, Static).render())


@pytest.mark.parametrize("schema_only", [False, True])
def test_job_list_groups_live_and_history_within_retention(demo, schema_only):
    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01, schema_only=schema_only)
        async with app.run_test(size=(180, 50)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            s = app.screen
            ids = list(s._job_by_item.values())
            # The account-wide list has every location's jobs and leaves out those
            # older than six months; the configured schema's list has them all, and
            # the job list hides the old ones itself.
            served = demo.list_jobs() if schema_only else list(demo.iter_jobs())
            jobs = {views.job_id_of(j): j for j in served}
            live = [jid for jid in ids if views.is_live(jobs[jid])]
            assert ids[: len(live)] == live, "live jobs come first"
            assert len(ids) == len(jobs) - 2, "the two jobs older than six months are hidden"
            assert ("2 older hidden" in _static(s, "#listinfo")) == schema_only
            # Live jobs are expanded into sub-job rows by default.
            assert any(sid.endswith(":sampling:0") for _jid, sid in s._sub_by_item.values())
            table = s.query_one("#jobs", DataTable)
            assert "hdr-live" in [r.value for r in table.rows] and "hdr-history" in [r.value for r in table.rows]
            await _settle(app, pilot)

    asyncio.run(run())


def test_job_list_search_by_name_field_and_no_match(demo):
    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(180, 50)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            s = app.screen
            names = AliasStore().all()
            box = s.query_one("#jobfilter", Input)

            box.value = "SWEEP"  # case-insensitive name match
            assert await _wait(pilot, lambda: len(s._job_by_item) == 2)
            assert all("sweep" in names[jid] for jid in s._job_by_item.values())

            box.value = "method:lora lora_r>=16 gpu:b200"
            assert await _wait(pilot, lambda: len(s._job_by_item) == 1)
            assert names[next(iter(s._job_by_item.values()))] == "qwen3.5-9b lora r32 sweep"

            box.value = "model:llama"
            assert await _wait(pilot, lambda: s.query_one("#nomatch", Static).display)
            assert "No jobs match “model:llama”" in _static(s, "#nomatch")
            assert not s._job_by_item

            box.value = ""
            assert await _wait(pilot, lambda: not s.query_one("#nomatch", Static).display and s._job_by_item)
            await _settle(app, pilot)

    asyncio.run(run())


def test_job_list_sort_cycles_and_reverses(demo):
    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(180, 50)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            s = app.screen
            assert s._sort == "created"
            s.action_cycle_sort()
            assert s._sort == "updated" and "sort: updated ↓" in _static(s, "#listinfo")
            s.action_cycle_sort()
            assert s._sort == "status"
            s.action_toggle_order()
            assert "sort: status ↑" in _static(s, "#listinfo")
            await _settle(app, pilot)

    asyncio.run(run())


def test_summary_then_tabs(demo):
    rl = next(j for j in demo.list_jobs() if len(j["sub_jobs"]) == 2 and j["status"] == "RUNNING")

    async def run():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(180, 50)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            app.push_screen(JobSummaryScreen(demo, rl["job_id"], job=rl))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen)
                               and app.screen._fetched_at is not None)
            summary = app.screen
            preview = _static(summary, "#preview")
            for label in ("Name", "Job ID", "Status", "Model", "Type", "Owner", "GPUs", "Created", "Elapsed", "Last update"):
                assert label in preview
            assert rl["job_id"] in preview and "gsm8k grpo" in preview
            assert summary.query_one("#subjobs", DataTable).row_count == 3  # entire job + 2 sub-jobs

            summary.action_open("resource")  # cursor is on "entire job"
            training = rl["sub_jobs"][0]["sub_job_id"]
            assert await _wait(
                pilot,
                lambda: isinstance(app.screen, JobDetailScreen)
                and app.screen.query_one("#resource-table", DataTable).row_count == 8
                and not app.screen._resource_loading,
            )
            d = app.screen
            assert d.query_one("#job-tabs", TabbedContent).active == "resource"
            status = _static(d, "#resource-status")
            # Resources shows the sub-job selected on the Logs tab (the first one here), by name.
            assert training.endswith(":training:0") and "training #0 (1 of 2" in status and "10m" in status and "8 reporting" in status and "newest" in status
            assert "H200" in status and "8 requested" in status and "scraped every" in status
            assert "no GPU samples" not in _static(d, "#resource-chart")
            d.action_cycle_window()
            assert d._resource_range == "30m" and "30m" in _static(d, "#resource-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_detail_logs_are_tagged_with_sub_job(demo):
    rl = next(j for j in demo.list_jobs() if len(j["sub_jobs"]) == 2 and j["status"] == "RUNNING")
    sampling = rl["sub_jobs"][1]["sub_job_id"]

    async def run():
        app = CortexTrainingLogTUI(demo, rl["job_id"], sub_job_id=sampling, poll_interval=0.01)
        async with app.run_test(size=(180, 50)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source == sampling)
            d = app.screen
            assert d.query_one("#job-tabs", TabbedContent).active == "logs"
            assert await _wait(pilot, lambda: any("vllm" in ln for ln in d._shown_lines))
            shown = [str(line) for line in d._logview._lines]
            assert any(line.startswith("sampling:0 │ ") and "vllm" in line for line in shown)
            assert not any(line.startswith("sampling:0 │ ") for line in d._shown_lines)  # buffer stays raw
            await _settle(app, pilot)

    asyncio.run(run())
