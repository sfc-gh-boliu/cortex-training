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

"""A multiplexed sub-job's GPUs read the same everywhere, and the preview's
sub-job table fits the screen: each config's start shows, with no sideways
scrollbar (the rest of the fitting is in test_r1_preview_table_fit.py)."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from textual.widgets import DataTable  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui import views as v  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402

HOST = "0a1b2c3d-0000-4000-8000-000000000001"
GUEST = "9f8e7d6c-0000-4000-8000-000000000002"
CONFIG = "fused_adam lr=1e-05 · bs=16 · seq=4096 · full-param"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _training(jid, i=0, *, model="Qwen/Qwen3-8B", **config):
    return {"sub_job_id": f"{jid}:training:{i}", "job_type": "TRAINING", "status": "RUNNING", "model_name": model,
            "training_config": {"n_gpus": 8, "optimizer": {"name": "fused_adam", "lr": 1e-05},
                                "train_batch_size": 16, "max_seq_len": 4096, **config}}


def _sampling(jid, i=0, n=4):
    return {"sub_job_id": f"{jid}:sampling:{i}", "job_type": "SAMPLING", "status": "RUNNING",
            "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": n, "max_seq_len": 4096}}


def _job(jid, sub_jobs):
    return {"job_id": jid, "status": "RUNNING", "hardware": "H200", "owner_role": "ML", "comment": "shared-GPU sft",
            "sub_jobs": sub_jobs}


GUEST_JOB = _job(GUEST, [_training(GUEST, multiplex_job_id=f"{HOST}:training:0"), _sampling(GUEST)])


def _client(jobs):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = jobs
    by_id = {j["job_id"]: j for j in jobs}
    c.get_job.side_effect = lambda jid: by_id[jid]
    return c


async def _wait(pilot, predicate, tries=300):
    for _ in range(tries):
        await pilot.pause(0.01)
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


async def _preview(app, pilot, client, job):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    app.push_screen(JobSummaryScreen(client, job["job_id"], job=job))
    assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and client.get_job.called)
    for _ in range(5):
        await pilot.pause()
    return app.screen


def _cell(table, key, label):
    """One cell by its column's label: a narrow screen leaves the Type column out."""
    labels = [str(column.label) for column in table.columns.values()]
    return str(table.get_row(key)[labels.index(label)])


def _fits(table) -> bool:
    """Every column shows in the table's width, with no sideways scrollbar."""
    used = sum(column.get_render_width(table) for column in table.columns.values())
    return used <= table.scrollable_content_region.width and not table.show_horizontal_scrollbar


def test_a_multiplexed_sub_job_is_shared_with_no_count_of_its_own():
    guest = GUEST_JOB["sub_jobs"][0]
    assert v.sub_job_gpu_text(guest) == "shared"  # its config's n_gpus is 8, but it requests none
    assert v.requested_gpus(guest) == 0
    assert v.job_gpu_cell(GUEST_JOB, "H200") == "4 H200"
    assert "training #0 runs on the GPUs of" in v.gpu_request_text(GUEST_JOB)


@pytest.mark.parametrize("sub_job_id, short", [
    (f"{HOST}:training:0", "0a1b2c3d…:training:0"),
    ("host:training:0", "host:training:0"),  # already short
    ("fake-job-aaaa:sampling:1", "fake-job…:sampling:1"),
    ("no-colon-here", "no-colon-here"),
])
def test_short_sub_job_id(sub_job_id, short):
    assert v.short_sub_job_id(sub_job_id) == short


@pytest.mark.parametrize("cols", [100, 140, 200])
def test_the_preview_shows_the_config_first_and_never_scrolls_sideways(cols):
    host = _job(HOST, [_training(HOST)])
    c = _client([host, GUEST_JOB])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(cols, 45)) as pilot:
            p = await _preview(app, pilot, c, GUEST_JOB)
            table = p.query_one("#subjobs", DataTable)
            assert _fits(table), "the table fits: no sideways scrolling"
            assert [_cell(table, "all", label) for label in ("GPUs", "Config")] == ["4 requested", "all sub-jobs"]
            gpus, config = (_cell(table, "sj-0", label) for label in ("GPUs", "Config"))
            assert gpus == "shared"
            # The config comes first; the host follows, short, and is what gets cut first.
            full = f"{CONFIG} · runs on 0a1b2c3d…:training:0"
            assert full.startswith(config.removesuffix("…")) and not config.endswith("·…")
            if cols >= 200:
                assert config == full
            elif cols >= 140:
                assert config.startswith(CONFIG) and config.endswith("…")
            else:
                assert config.startswith("fused_adam") and config.endswith("…")
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_list_says_shared_for_a_multiplexed_sub_job():
    host = _job(HOST, [_training(HOST)])
    c = _client([host, GUEST_JOB])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(180, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._sub_by_item)
            s = app.screen
            item = next(key for key, (_jid, sid) in s._sub_by_item.items() if sid == f"{GUEST}:training:0")
            assert str(s.query_one("#jobs", DataTable).get_row(item)[8]) == "shared"
            await _settle(app, pilot)

    asyncio.run(run())


def test_long_cells_and_many_rows_still_fit():
    model = "org/" + "M" * 5000
    subs = [_training(GUEST, i, model=model) for i in range(15)]
    job = _job(GUEST, subs)
    c = _client([job])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 45)) as pilot:
            p = await _preview(app, pilot, c, job)
            table = p.query_one("#subjobs", DataTable)
            assert table.max_scroll_y > 0, "more rows than the table shows: it scrolls down"
            assert _fits(table), "but never sideways"
            model_cell = _cell(table, "sj-0", "Model")
            assert model_cell.endswith("…") and cell_len(model_cell) <= 36
            config = _cell(table, "sj-0", "Config")
            assert config.startswith("fused_adam lr=1e-05")
            # A narrower screen refits the table.
            await pilot.resize_terminal(110, 45)
            assert await _wait(pilot, lambda: _fits(table) and _cell(table, "sj-0", "Config") != config)
            assert _cell(table, "sj-0", "Config").startswith("fused_adam")
            await _settle(app, pilot)

    asyncio.run(run())
