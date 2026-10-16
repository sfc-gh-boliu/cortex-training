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

"""The preview's sub-job table on any screen: it fits 80 columns with every cut
marked, ``e`` shows it whole and it then scrolls sideways, a screen too narrow
even for the cut table scrolls it sideways, its scrollbar follows each refit,
and nothing is cut on a screen with room."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from rich.text import Text  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import _SUBJOB_COLUMNS  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.screens import fit_columns  # noqa: E402
from cortex_training.tui.untrusted import unwrapped  # noqa: E402

JOB = "fake-job-0001"
TRAINING = {"n_gpus": 8, "optimizer": {"name": "fused_adam", "lr": 1e-05}, "train_batch_size": 16,
            "max_seq_len": 8192}
CONFIG = "fused_adam lr=1e-05 · bs=16 · seq=8192 · full-param"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _sub(kind, status, *, ids=True, **config):
    """A sub-job as the service lists it (``ids=False``: as an UNKNOWN job's stored config)."""
    block = "training_config" if kind == "training" else "inference_config"
    sub = {"job_type": kind.upper(), "status": status, "model_name": "Qwen/Qwen3-8B",
           block: config or ({**TRAINING} if kind == "training" else {"n_gpus": 4, "max_seq_len": 4096})}
    if ids:
        index_kind = {"log_probability": "log_prob"}.get(kind, kind)
        sub["sub_job_id"] = f"{JOB}:{index_kind}:0"
    return sub


def _job(status, subs, **extra):
    job = {"job_id": JOB, "status": status, "hardware": "H200", "owner_role": "TRAINING_ROLE",
           "submitted_by": "TEST_USER_A", "comment": "a short name", "created_at": "2026-10-14T17:39:00Z",
           "updated_at": "2026-10-14T18:24:00Z", "state_changed_at": "2026-10-14T18:24:00Z", "sub_jobs": subs}
    job.update(extra)
    return job


# The three jobs whose right columns an 80-column screen used to clip mid-cell.
LOG_PROB = _job("TERMINATED", [_sub("training", "TERMINATED"),
                               _sub("log_probability", "TERMINATED", n_gpus=2, max_seq_len=4096)],
                reason="idle_timeout")
INITIALIZING = _job("INITIALIZING", [_sub("training", "INITIALIZING"), _sub("sampling", "INITIALIZING")])
# UNKNOWN as the service answers it: the configs it stored, without ids or times.
UNKNOWN = _job("UNKNOWN", [_sub("training", "UNKNOWN", ids=False), _sub("sampling", "UNKNOWN", ids=False)],
               reason="Job status unavailable: Control Plane is unavailable")
for key in ("updated_at", "state_changed_at"):
    UNKNOWN.pop(key)


def _client(job):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
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


async def _open(app, pilot, job):
    c = app._client  # the app's printable wrapper; unwrapped(c) is the mock
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    app.push_screen(JobSummaryScreen(c, job["job_id"], job=job))
    assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and unwrapped(c).get_job.called)
    p = app.screen
    assert await _wait(pilot, lambda: p._fetched_at is not None and not p._loading)
    for _ in range(10):
        await pilot.pause()
    return p


def _columns(table) -> list:
    return [str(column.label) for column in table.columns.values()]


def _cell(table, key, label) -> str:
    return str(table.get_row(key)[_columns(table).index(label)])


def _whole_cells(p) -> dict:
    """Every row's cells with nothing cut, by row key and column label."""
    return {key: dict(zip(_SUBJOB_COLUMNS, (cell.plain for cell in cells))) for key, cells in p._subjob_rows(p._job)}


def _keys(table) -> list:
    return [key.value for key in table.rows]


def _fits(table) -> bool:
    """Every column shows inside the table, and it has no sideways scrollbar."""
    used = sum(column.get_render_width(table) for column in table.columns.values())
    return used <= table.scrollable_content_region.width and not table.show_horizontal_scrollbar


def _title(p) -> str:
    return str(p.query_one("#subjobs-title", Static).render())


def _footer_keys(screen) -> dict:
    return {binding.action: binding.description for _ns, binding, enabled, _tip in screen.active_bindings.values()
            if enabled and binding.show}


# ─── fit_columns ─────────────────────────────────────────────────────────────


def _rows(sub_job="  training #0", type_="training", status="RUNNING", model="Qwen/Qwen3-8B", gpus="8",
          config=CONFIG):
    return [("all", [Text("▸ entire job"), Text(type_), Text(status), Text(model), Text(f"{gpus} requested"),
                     Text("all sub-jobs")]),
            ("sj-0", [Text(sub_job), Text(type_), Text(status), Text(model), Text(gpus), Text(config)])]


def _total(labels, rows, pad=2) -> int:
    return sum(max(cell_len(label), *(cells[i].cell_len for _k, cells in rows)) + pad
               for i, label in enumerate(labels))


def test_a_table_with_room_is_left_whole():
    rows = _rows()
    labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, rows, _total(_SUBJOB_COLUMNS, rows))
    assert list(labels) == list(_SUBJOB_COLUMNS) and not cut and not wider
    assert [[c.plain for c in cells] for _k, cells in fitted] == [[c.plain for c in cells] for _k, cells in rows]


def test_a_very_long_model_then_the_config_give_way_first():
    model = "org/" + "M" * 200
    rows = _rows(model=model)
    width = _total(_SUBJOB_COLUMNS, rows) - (len(model) - 36) - 4  # the model at 36 is still 4 too wide
    labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, rows, width)
    assert cut and not wider and list(labels) == list(_SUBJOB_COLUMNS)
    model_cell, config = fitted[1][1][3].plain, fitted[1][1][5].plain
    assert cell_len(model_cell) == 36 and model_cell.endswith("…")
    assert config.startswith("fused_adam lr=1e-05") and config.endswith("…") and not config.endswith("·…")
    assert cell_len(config) >= 24 and _total(labels, fitted) <= width


def test_type_is_left_out_before_the_model_and_config_go_further():
    rows = _rows(type_="training, log_probability", sub_job="  log_probability #0")
    six = _total(_SUBJOB_COLUMNS, rows)
    width = six - (len(CONFIG) - 24) - 4  # the config at 24 cells is still 4 too wide
    labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, rows, width)
    assert "Type" not in labels and cut and not wider
    cells = dict(zip(labels, (c.plain for c in fitted[1][1])))
    assert cells["Model"] == "Qwen/Qwen3-8B", "leaving Type out was enough: the model stays whole"
    assert cells["Sub-job"] == "  log_probability #0" and cells["Status"] == "RUNNING" and cells["GPUs"] == "8"
    assert cell_len(cells["Config"]) > 24, "the config gets the room Type gave back"


def test_a_screen_too_narrow_even_for_the_cut_table_is_scrolled():
    rows = _rows(type_="training, log_probability", sub_job="  log_probability #0")
    labels, fitted, cut, wider = fit_columns(_SUBJOB_COLUMNS, rows, 50)
    assert wider and cut and "Type" not in labels
    cells = dict(zip(labels, (c.plain for c in fitted[1][1])))
    assert cell_len(cells["Model"]) == 10 and cell_len(cells["Config"]) <= 12
    # The row's identity, status and GPUs are never cut: the table scrolls to them.
    assert cells["Sub-job"] == "  log_probability #0" and cells["Status"] == "RUNNING"


# ─── the preview on a real screen ────────────────────────────────────────────


@pytest.mark.parametrize("job", [LOG_PROB, INITIALIZING, UNKNOWN], ids=["log-prob", "initializing", "unknown"])
def test_the_table_fits_80_columns_with_every_cut_marked(job):
    async def run():
        app = CortexTrainingLogTUI(_client(job), poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            p = await _open(app, pilot, job)
            table = p.query_one("#subjobs", DataTable)
            assert await _wait(pilot, lambda: _fits(table)), (table.virtual_size, table.scrollable_content_region)
            assert _columns(table) == ["Sub-job", "Status", "Model", "GPUs", "Config"], "Type is left out"
            whole = _whole_cells(p)
            for key in _keys(table):
                for label in _columns(table):
                    shown, full = _cell(table, key, label), whole[key][label]
                    assert shown == full or (shown.endswith("…") and full.startswith(shown[:-1].rstrip(" ·"))), (
                        key, label, shown, full)
                for label in ("Sub-job", "Status", "GPUs"):  # never cut
                    assert _cell(table, key, label) == whole[key][label]
            config = _cell(table, "sj-0", "Config")
            assert cell_len(config) >= 12 and config.startswith("fused_adam"), "the config keeps some room"
            if job is UNKNOWN:
                assert _cell(table, "note", "Config").startswith("configs at")
            assert "e whole table" in _title(p) and "\n" not in _title(p), _title(p)
            await _settle(app, pilot)

    asyncio.run(run())


def test_e_shows_the_whole_table_which_then_scrolls_sideways():
    async def run():
        app = CortexTrainingLogTUI(_client(LOG_PROB), poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            p = await _open(app, pilot, LOG_PROB)
            table = p.query_one("#subjobs", DataTable)
            await pilot.press("e")
            assert await _wait(pilot, lambda: _columns(table) == list(_SUBJOB_COLUMNS) and table.show_horizontal_scrollbar)
            whole = _whole_cells(p)
            assert all(_cell(table, key, label) == whole[key][label] for key in _keys(table) for label in _SUBJOB_COLUMNS)
            assert _cell(table, "all", "Type") == "train+logprob"  # the job's Type label, whole
            assert _cell(table, "sj-0", "Config") == CONFIG
            assert "← → scroll" in _title(p) and "e fit to screen" in _title(p)
            # The keys reach every column.
            assert table.max_scroll_x > 0 and table.scroll_x == 0
            await pilot.press("right")
            assert await _wait(pilot, lambda: table.scroll_x == 1)
            await pilot.press("end")
            assert await _wait(pilot, lambda: table.scroll_x == table.max_scroll_x)
            await pilot.press("home")
            assert await _wait(pilot, lambda: table.scroll_x == 0)
            # e again fits it to the screen, and the scrollbar goes.
            await pilot.press("e")
            assert await _wait(pilot, lambda: "Type" not in _columns(table) and _fits(table))
            assert "e whole table" in _title(p) and "scroll" not in _title(p)
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_refresh_that_changes_nothing_keeps_the_sideways_scroll():
    async def run():
        c = _client(LOG_PROB)
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            p = await _open(app, pilot, LOG_PROB)
            table = p.query_one("#subjobs", DataTable)
            await pilot.press("e")
            assert await _wait(pilot, lambda: table.show_horizontal_scrollbar and table.max_scroll_x > 0)
            await pilot.press("end")
            assert await _wait(pilot, lambda: table.scroll_x == table.max_scroll_x)
            reads = c.get_job.call_count
            await pilot.press("r")
            assert await _wait(pilot, lambda: c.get_job.call_count > reads and not p._loading)
            for _ in range(10):
                await pilot.pause()
            assert table.scroll_x == table.max_scroll_x > 0
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_screen_too_narrow_for_the_cut_table_scrolls_it_sideways():
    async def run():
        app = CortexTrainingLogTUI(_client(LOG_PROB), poll_interval=0.01)
        async with app.run_test(size=(60, 30)) as pilot:
            p = await _open(app, pilot, LOG_PROB)
            table = p.query_one("#subjobs", DataTable)
            assert await _wait(pilot, lambda: table.show_horizontal_scrollbar and table.max_scroll_x > 0)
            assert "Type" not in _columns(table)
            assert "← → scroll" in _title(p) and "e whole table" in _title(p)
            assert all(not row.rstrip().endswith("·") for row in _title(p).split("\n")), _title(p)
            await pilot.press("end")
            assert await _wait(pilot, lambda: table.scroll_x == table.max_scroll_x)
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_sideways_scrollbar_follows_each_refit():
    async def run():
        app = CortexTrainingLogTUI(_client(LOG_PROB), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            p = await _open(app, pilot, LOG_PROB)
            table = p.query_one("#subjobs", DataTable)
            assert _fits(table)
            # Narrower: refitted to exactly the table's width, with no stale scrollbar.
            for cols, fits in ((100, True), (60, False), (80, True), (140, True), (60, False), (200, True)):
                await pilot.resize_terminal(cols, 40)
                assert await _wait(pilot, lambda: _fits(table) if fits else table.show_horizontal_scrollbar), (
                    cols, table.virtual_size, table.scrollable_content_region, table.show_horizontal_scrollbar)
                for _ in range(10):
                    await pilot.pause()
                assert _fits(table) == fits, cols
            await _settle(app, pilot)

    asyncio.run(run())


def test_nothing_is_cut_on_a_screen_with_room():
    async def run():
        app = CortexTrainingLogTUI(_client(LOG_PROB), poll_interval=0.01)
        async with app.run_test(size=(200, 45)) as pilot:
            p = await _open(app, pilot, LOG_PROB)
            table = p.query_one("#subjobs", DataTable)
            assert _fits(table) and _columns(table) == list(_SUBJOB_COLUMNS)
            whole = _whole_cells(p)
            assert all(_cell(table, key, label) == whole[key][label] for key in _keys(table) for label in _SUBJOB_COLUMNS)
            assert _cell(table, "all", "Type") == "train+logprob"  # the job's Type label, whole
            assert "e whole" not in _title(p) and "expand_values" not in _footer_keys(p)
            await _settle(app, pilot)

    asyncio.run(run())
