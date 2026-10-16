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

"""App tests for job names, comments, UNKNOWN rows, requested GPUs and server
reasons in the job list, the job preview and the detail summary line."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from textual.containers import VerticalScroll  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Input  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.demo import DemoClient  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.screens import PromptScreen  # noqa: E402
from cortex_training.tui.stores import AliasStore  # noqa: E402
from cortex_training.tui.stores import MetricsStore  # noqa: E402

NOW = datetime.now(timezone.utc)
A, B, C, U, E = (f"fake-job-{c}" for c in ("aaaa", "bbbb", "cccc", "uuuu", "eeee"))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Keep the TUI's cache, error log and saved logs out of the real home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(jid, *, status="CANCELLED", comment=None, **extra):
    job = {"job_id": jid, "status": status, "created_at": _iso(NOW - timedelta(hours=3)),
           "updated_at": _iso(NOW - timedelta(hours=1)), "state_changed_at": _iso(NOW - timedelta(hours=1)),
           "hardware": "H200", "owner_role": "ML", "comment": comment,
           "sub_jobs": [{"sub_job_id": f"{jid}:training:0", "job_type": "TRAINING", "status": status,
                         "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}}]}
    job.update(extra)
    return job


def _unknown(jid, sub_jobs):
    # The service's fallback when it can't reach the control plane: no times, no sub-job ids.
    return {"job_id": jid, "status": "UNKNOWN", "reason": "Job status unavailable: Control Plane is unavailable",
            "comment": "status probe [/b] run", "owner_role": "ML", "sub_jobs": sub_jobs}


def _client(jobs, *, fresh=None):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    # The account-wide list isn't deployed (what accounts get today), so the job
    # list shows list_jobs, the configured schema's jobs.
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = jobs
    by_id = {j["job_id"]: j for j in jobs}
    c.get_job.side_effect = lambda jid: (fresh or {}).get(jid, by_id[jid])
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.download_metrics.return_value = []
    return c


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


def _row(screen, key) -> list:
    return [str(cell) for cell in screen.query_one("#jobs", DataTable).get_row(key)]


def _words(text) -> str:
    """Cell text without the tree drawing, as one line of words."""
    return " ".join(str(text).translate(str.maketrans("│├└─", "    ")).split())


def _item(screen, jid) -> str:
    return next(item for item, job_id in screen._job_by_item.items() if job_id == jid)


async def _open_list(app, pilot):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    return app.screen


# ─── job list ────────────────────────────────────────────────────────────────


def test_list_names_come_from_the_alias_the_comment_or_the_id():
    AliasStore().set(B, "my alias")
    jobs = [_job(A, comment="qwen3-8b lora · run [/b] 7\nlr 1e-4"), _job(B, comment="kept beside the alias"),
            _job(C)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(180, 40)) as pilot:
            s = await _open_list(app, pilot)
            assert _row(s, _item(s, A))[1].endswith("qwen3-8b lora · run [/b] 7 …")  # literal, never markup
            assert "my alias  kept beside the alias" in _row(s, _item(s, B))[1]
            assert _row(s, _item(s, C))[1].endswith(C)  # no name: the job id
            box = s.query_one("#jobfilter", Input)
            box.value = "BESIDE THE"  # the comment of a job shown by its alias
            assert await _wait(pilot, lambda: list(s._job_by_item.values()) == [B])
            box.value = "lr 1e-4"  # a later line of a comment
            assert await _wait(pilot, lambda: list(s._job_by_item.values()) == [A])
            box.value = "name:alias"
            assert await _wait(pilot, lambda: list(s._job_by_item.values()) == [B])
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_unknown_rows_show_status_unavailable_and_derived_sub_jobs():
    with_configs = _unknown(U, [
        {"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}},
        {"status": "UNKNOWN", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2}},
    ])
    bare = _unknown(E, [])
    jobs = [with_configs, bare, _job(A)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(180, 40)) as pilot:
            s = await _open_list(app, pilot)
            u, e = _item(s, U), _item(s, E)
            row = _row(s, u)
            assert row[3] == "UNKNOWN" and row[5] == "—" and row[9] == "—" and row[8] == "10"  # no hardware either
            assert _words(_row(s, f"note-{u}")[1]) == "? status unavailable"
            assert _words(_row(s, f"note-{e}")[1]) == "? status unavailable"
            table = s.query_one("#jobs", DataTable)
            assert all(r.height == 1 for r in table.rows.values()), "one line per row, notes too"
            # The service's sentence is said once, above the list.
            assert s.query_one("#unavailable", Static).display
            line = _text(s, "#unavailable")
            assert line.count("Job status unavailable: Control Plane is unavailable") == 1 and "· 2 jobs" in line
            assert [sid for jid, sid in s._sub_by_item.values() if jid == U] == [f"{U}:training:0", f"{U}:sampling:0"]
            # The sub-job id after the job's (shown on the row above), whole: the index isn't cut off.
            assert _row(s, f"sub-{u}-1")[2] == "sampling:0"
            # Enter on a note line opens its job, as on the job row.
            s.set_focus(table)
            table.move_cursor(row=table.get_row_index(f"note-{u}"))
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert app.screen._job_id == U
            await pilot.press("b")
            s = await _open_list(app, pilot)
            # Space on the note line folds its job.
            table.move_cursor(row=table.get_row_index(f"note-{u}"))
            assert s._highlighted() == (f"note-{u}", U)
            s.action_toggle_expand()
            assert await _wait(pilot, lambda: f"note-{u}" not in [r.value for r in table.rows])
            # A search that shows no UNKNOWN job drops the line.
            s.query_one("#jobfilter", Input).value = A
            assert await _wait(pilot, lambda: list(s._job_by_item.values()) == [A])
            assert not s.query_one("#unavailable", Static).display
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_empty_list_names_the_schema_not_the_role():
    async def run():
        app = CortexTrainingLogTUI(_client([]), poll_interval=0.01)
        async with app.run_test(size=(160, 30)) as pilot:
            ok = await _wait(pilot, lambda: isinstance(app.screen, JobListScreen)
                             and app.screen.query_one("#nomatch", Static).display and app.screen._loaded)
            assert ok
            assert _text(app.screen, "#nomatch") == (
                "No jobs created in the last 6 months in CORTEX_TRAINING_DB.PUBLIC that this role can see.")
            await _settle(app, pilot)

    asyncio.run(run())


def test_job_list_screen_takes_no_sub_job_id():
    assert "sub_job_id" not in inspect.signature(JobListScreen.__init__).parameters
    assert not hasattr(JobListScreen(MagicMock()), "_sub_job_id")


# ─── preview ─────────────────────────────────────────────────────────────────


def test_preview_shows_the_one_line_name_and_the_whole_comment_wrapped():
    comment = ("support-chat v4 [bold]LoRA[/bold] r16\n"
               + "March export, 8k context; eval every 250 steps " * 4 + "\n[/x] closing line")
    job = _job(A, status="RUNNING", comment=comment)

    async def run():
        c = _client([job])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(120, 50)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobSummaryScreen(c, A, job=job))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            await pilot.pause()
            lines = _text(app.screen, "#preview").split("\n")
            name = next(line for line in lines if line.strip().startswith("Name"))
            assert "support-chat v4 [bold]LoRA[/bold] r16 …" in name and "(from the job's comment)" in name
            start = next(i for i, line in enumerate(lines) if line.strip().startswith("Comment"))
            block = lines[start:next(i for i, line in enumerate(lines) if line.strip().startswith("Job ID"))]
            assert len(block) >= 4, "a long comment wraps over several lines"
            assert all(line.startswith(" " * 15) for line in block[1:]), "continuation lines sit under the value"
            assert all(len(line.rstrip()) <= 118 for line in block)
            words = " ".join(" ".join(block).split()).removeprefix("Comment ")
            assert words == " ".join(comment.split())  # all of it, literally
            assert "running, so far" in _text(app.screen, "#preview")
            assert "includes queue time" in _text(app.screen, "#preview")
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_preview_of_an_unknown_job_says_status_unavailable_and_lists_its_configs():
    job = _unknown(U, [{"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B",
                        "training_config": {"n_gpus": 8}}])
    bare = _unknown(E, [])

    async def run():
        c = _client([job, bare])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            await _open_list(app, pilot)
            for jid, row in ((U, job), (E, bare)):
                app.push_screen(JobSummaryScreen(c, jid, job=row))
                assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
                await pilot.pause()
                s = app.screen
                assert "Job status unavailable: Control Plane is unavailable" in _text(s, "#health")
                table = s.query_one("#subjobs", DataTable)
                assert str(table.get_row("note")[0]).strip() == "? status unavailable"
                assert ("Created" in _text(s, "#preview")) and "status probe [/b] run" in _text(s, "#preview")
                if jid == U:
                    assert s._scope_by_row["sj-0"] == f"{U}:training:0"
                    assert str(table.get_row("sj-0")[0]).strip() == "training #0"
                else:
                    assert table.row_count == 2  # entire job + the note, no invented sub-jobs
                await pilot.press("b")
                assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen))
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_details_of_an_unknown_job_tail_its_derived_sub_jobs_or_say_status_unavailable():
    job = _unknown(U, [
        {"status": "UNKNOWN", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2}},
        {"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}},
    ])
    bare = _unknown(E, [])

    async def run():
        c = _client([job, bare])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobDetailScreen(c, U, job=job, initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            d = app.screen
            assert list(d._source_by_item.values()) == [f"{U}:training:0", f"{U}:sampling:0"]  # training first
            assert d._current_source == f"{U}:training:0"
            assert "Job status unavailable: Control Plane is unavailable" in _text(d, "#summary")
            await pilot.press("b")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen))
            app.push_screen(JobDetailScreen(c, E, job=bare, initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                               and any("[no logs]" in line for line in app.screen._shown_lines))
            note = next(line for line in app.screen._shown_lines if "[no logs]" in line)
            assert note == ("[no logs] Job status unavailable: Control Plane is unavailable; "
                            "no logs are cached on this machine")
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


def test_entire_job_row_counts_requested_gpus_only():
    host = _job(A, status="RUNNING")
    guest = _job(B, status="RUNNING", sub_jobs=[
        {"sub_job_id": f"{B}:training:0", "job_type": "TRAINING", "status": "RUNNING", "model_name": "Qwen/Qwen3-8B",
         "training_config": {"n_gpus": 8, "multiplex_job_id": f"{A}:training:0"}},
        {"sub_job_id": f"{B}:sampling:0", "job_type": "SAMPLING", "status": "RUNNING", "model_name": "Qwen/Qwen3-8B",
         "inference_config": {"n_gpus": 2}},
    ])

    async def run():
        c = _client([host, guest])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(180, 45)) as pilot:
            s = await _open_list(app, pilot)
            assert _row(s, _item(s, B))[8] == "2 H200"
            assert _row(s, f"sub-{_item(s, B)}-0")[8] == "shared"
            app.push_screen(JobSummaryScreen(c, B, job=guest))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            await pilot.pause()
            p = app.screen
            table = p.query_one("#subjobs", DataTable)
            assert str(table.get_row("all")[4]) == "2 requested"
            assert str(table.get_row("all")[5]) == "all sub-jobs"
            assert str(table.get_row("sj-0")[4]) == "shared"
            # The config first, then the host it runs on, short (the GPUs line has it in full).
            assert str(table.get_row("sj-0")[5]) == "full-param · runs on fake-job…:training:0"
            gpus = next(line for line in _text(p, "#preview").split("\n") if line.strip().startswith("GPUs"))
            assert "2 × H200 requested" in gpus and "training #0 runs on the GPUs of" in gpus
            await _settle(app, pilot)

    asyncio.run(run())


# ─── server reasons: fresh, and with reason_details ──────────────────────────


def test_a_reason_the_service_stops_sending_disappears():
    pending = _job(A, status="PENDING", reason="quota_filled", reason_details="needs 8 GPUs; 64 currently active")
    running = {k: val for k, val in {**pending, "status": "RUNNING"}.items() if k not in ("reason", "reason_details")}

    async def run():
        c = _client([pending], fresh={A: running})
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobSummaryScreen(c, A, job=pending))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen)
                               and "Running" in _text(app.screen, "#health"))
            assert "quota_filled" not in _text(app.screen, "#health") + _text(app.screen, "#preview")
            assert "reason" not in app.screen._job and "reason_details" not in app.screen._job

            app.push_screen(JobDetailScreen(c, A, job=pending, initial_tab="resource"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                               and app.screen._job.get("status") == "RUNNING")
            assert "quota_filled" not in _text(app.screen, "#summary")
            assert "needs 8 GPUs" not in _text(app.screen, "#summary")
            assert "quota_filled" not in (app.sub_title or "")
            await _settle(app, pilot)

    asyncio.run(run())


def test_reason_details_follow_the_reason_in_the_preview_and_the_summary_line():
    failed = _job(A, status="FAILED", reason="node_evacuated",
                  reason_details="node [b]taken[/b] out of service; resume from cp_[/x]")

    async def run():
        c = _client([failed])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(180, 45)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobSummaryScreen(c, A, job=failed))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and c.get_job.called)
            await pilot.pause()
            assert "node_evacuated — node [b]taken[/b] out of service; resume from cp_[/x]" in _text(app.screen, "#health")
            app.push_screen(JobDetailScreen(c, A, job=failed, initial_tab="resource"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                               and "node_evacuated" in _text(app.screen, "#summary"))
            assert "node_evacuated — node [b]taken[/b] out of service; resume from cp_[/x]" in _text(app.screen, "#summary")
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


# ─── demo data ───────────────────────────────────────────────────────────────


def test_demo_shows_comment_names_an_unknown_row_and_a_shared_gpu_job():
    client = DemoClient()
    client.seed_local(aliases=AliasStore(), metrics=MetricsStore())
    served = client.list_jobs()
    unknown = next(j for j in served if j["status"] == "UNKNOWN")
    assert "created_at" not in unknown and all("sub_job_id" not in s for s in unknown["sub_jobs"])
    assert client.get_job(unknown["job_id"]) == unknown
    assert client.download_metrics(unknown["job_id"], MetricsStore().path("x").parent / "m")  # stage data still there

    async def run():
        app = CortexTrainingLogTUI(client, poll_interval=0.01)
        async with app.run_test(size=(180, 50)) as pilot:
            s = await _open_list(app, pilot)
            names = [_row(s, key)[1] for key in s._job_by_item]
            assert any(name.endswith("qwen3.8-27b full sft · contracts v2 …") for name in names)
            assert any("lora probe on support-chat v4  support-chat v4 LoRA probe …" in name for name in names)
            assert any("ablation: no warmup [baseline v4]" in name for name in names)
            assert _words(_row(s, f"note-{_item(s, unknown['job_id'])}")[1]) == "? status unavailable"
            assert "Job status unavailable: Control Plane is unavailable" in _text(s, "#unavailable")
            shared = next(j for j in served if views.alias_of(j, AliasStore().all()) == "lora probe on support-chat v4")
            # The demo lists several locations, so a Location column shows: find GPUs by its label.
            gpus = [str(column.label) for column in s.query_one("#jobs", DataTable).columns.values()].index("GPUs")
            assert _row(s, _item(s, shared["job_id"]))[gpus] == "shared"
            # Its GPUs report under the job it shares them with, as on the service.
            assert client.download_metrics(shared["job_id"], MetricsStore().path("x").parent / "m2") == []
            await _settle(app, pilot)

    asyncio.run(run())


# ─── review follow-ups: outages, wide text, scrolling, renaming, refreshes ────


def test_an_outage_says_why_once_and_keeps_one_line_per_row():
    configs = [{"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}},
               {"status": "UNKNOWN", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2}}]
    jobs = [{**_unknown(f"fake-job-{i:04d}", [dict(c) for c in configs]), "comment": f"run {i}"} for i in range(10)]
    gone = {**_unknown("fake-job-0099", [{"status": "UNKNOWN", "model_name": "openai/gpt-oss-20b",
                                          "inference_config": {"n_gpus": 4}}]),
            "reason": "Job status unavailable: job not found on Control Plane"}
    jobs += [gone] + [_job(f"fake-job-1{i:03d}") for i in range(5)]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            s = await _open_list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            keys = [r.value for r in table.rows]
            live_height = sum(table.rows[k].height for k in list(table.rows)[:keys.index("hdr-history")])
            assert live_height == 1 + 10 * 4, "header, then job + note + two sub-jobs each"
            line = _text(s, "#unavailable")
            assert line.startswith("   ? Job status unavailable: Control Plane is unavailable")
            assert "· 10 jobs; 1 more for another reason (see each job's preview)" in line
            # The job the control plane has no record of isn't live: it is under history, folded.
            gone_item = _item(s, "fake-job-0099")
            assert keys.index(gone_item) > keys.index("hdr-history")
            s.set_focus(table)
            table.move_cursor(row=table.get_row_index(gone_item))
            await pilot.press("space")
            # A config with neither an id nor a type is named by its place, never by its contents.
            assert await _wait(pilot, lambda: f"sub-{_item(s, 'fake-job-0099')}-0" in s._sub_by_item)
            assert _words(_row(s, f"sub-{_item(s, 'fake-job-0099')}-0")[1]) == "config #0"
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())


WIDE = "日本語の名前テスト・"  # ten characters, twenty cells


async def _open_preview(app, pilot, client, job):
    await _open_list(app, pilot)
    app.push_screen(JobSummaryScreen(client, job["job_id"], job=job))
    assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and client.get_job.called)
    for _ in range(3):
        await pilot.pause()
    return app.screen


def _comment_rows(screen) -> list:
    lines = _text(screen, "#preview").split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Comment"))
    return lines[start:next(i for i, line in enumerate(lines) if line.strip().startswith("Job ID"))]


def test_a_wide_character_comment_wraps_by_cells_under_the_value_column():
    comment = WIDE * 3 + "\n" + "長い説明文です。" * 30 + "\n" + "サポート チャット 評価 " * 12
    job = _job(A, status="RUNNING", comment=comment)

    async def run():
        c = _client([job])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(100, 50)) as pilot:
            p = await _open_preview(app, pilot, c, job)
            preview = p.query_one("#preview", Static)
            width = preview.size.width
            lines = _text(p, "#preview").split("\n")
            assert preview.size.height == len(lines), "no line is wider than the preview (none soft-wraps)"
            name = next(line for line in lines if line.strip().startswith("Name"))
            assert "(from the job's comment)" in name and cell_len(name) <= width
            rows = _comment_rows(p)
            assert len(rows) >= 6
            assert all(row.startswith(" " * 15) for row in rows[1:]), "continuation lines sit under the value"
            assert all(cell_len(row) <= width for row in rows)
            assert "".join("".join(rows).split()).removeprefix("Comment") == "".join(comment.split())
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_preview_rewraps_the_comment_when_the_terminal_resizes():
    job = _job(A, status="RUNNING", comment="March export, 8k context; eval every 250 steps " * 6)

    async def run():
        c = _client([job])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 50)) as pilot:
            p = await _open_preview(app, pilot, c, job)
            wide = _comment_rows(p)
            await pilot.resize_terminal(90, 50)
            assert await _wait(pilot, lambda: len(_comment_rows(p)) > len(wide))
            width = p.query_one("#preview", Static).size.width
            assert width < 90 and all(cell_len(row) <= width for row in _comment_rows(p))
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_long_comment_leaves_the_top_of_the_preview_on_a_short_terminal():
    subs = [{"sub_job_id": f"{A}:sampling:{i}", "job_type": "SAMPLING", "status": "RUNNING",
             "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 1}} for i in range(6)]
    job = _job(A, status="RUNNING", comment=("nightly eval of the support-chat checkpoints, 1,200 prompts; " * 16)[:1000],
               sub_jobs=subs)

    async def run():
        c = _client([job])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(100, 30)) as pilot:
            p = await _open_preview(app, pilot, c, job)
            body = p.query_one("#summary-body", VerticalScroll)
            table = p.query_one("#subjobs", DataTable)
            assert p.focused is table and body.max_scroll_y > 0
            assert body.scroll_y == 0, "the name and comment stay on screen"
            await pilot.press("down")  # moving in the table brings it into view
            assert await _wait(pilot, lambda: body.scroll_y > 0)
            assert table.region.y + table.cursor_row < p.size.height
            await pilot.press("up", "up")  # up from the first row goes back to the top
            assert await _wait(pilot, lambda: body.scroll_y == 0)
            await _settle(app, pilot)

    asyncio.run(run())


def test_renaming_a_commented_job_keeps_its_comment_as_the_name_unless_changed():
    job = _job(A, comment="qwen3.8-27b full sft · contracts v2\nresumes the Oct run")

    async def run():
        c = _client([job])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _open_list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            s.set_focus(table)
            table.move_cursor(row=table.get_row_index(_item(s, A)))
            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            box = app.screen.query_one("#prompt-input", Input)
            assert box.value == "" and box.placeholder == "qwen3.8-27b full sft · contracts v2"
            await pilot.press("enter")  # nothing typed: the comment stays the name
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen))
            assert AliasStore().all() == {}
            assert _words(_row(s, _item(s, A))[1]) == "▸ qwen3.8-27b full sft · contracts v2 …"  # once

            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            app.screen.query_one("#prompt-input", Input).value = "qwen3.8-27b  full sft · contracts v2"
            await pilot.press("enter")  # the comment's own first line: still no alias
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen))
            assert AliasStore().all() == {}

            # The preview's prompt starts the same way, and the comment's own line stores no alias there either.
            app.push_screen(JobSummaryScreen(c, A, job=job))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            box = app.screen.query_one("#prompt-input", Input)
            assert box.value == "" and box.placeholder == "qwen3.8-27b full sft · contracts v2"
            box.value = " qwen3.8-27b full sft ·  contracts v2 "
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert AliasStore().all() == {} and "(from the job's comment)" in _text(app.screen, "#preview")
            # An alias pre-fills the prompt, and clearing it goes back to the comment.
            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            app.screen.query_one("#prompt-input", Input).value = "contracts v2"
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert AliasStore().all() == {A: "contracts v2"}
            assert "local name on this machine" in _text(app.screen, "#preview")
            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            box = app.screen.query_one("#prompt-input", Input)
            assert box.value == "contracts v2"
            box.value = ""
            await pilot.press("enter")
            assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            assert AliasStore().all() == {} and "(from the job's comment)" in _text(app.screen, "#preview")
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_alias_that_repeats_the_comment_is_shown_once():
    AliasStore().set(A, "same as the comment")
    job = _job(A, comment="same as the comment\nsecond line")

    async def run():
        app = CortexTrainingLogTUI(_client([job]), poll_interval=0.01)
        async with app.run_test(size=(160, 30)) as pilot:
            s = await _open_list(app, pilot)
            assert _words(_row(s, _item(s, A))[1]) == "▸ same as the comment …"
            await _settle(app, pilot)

    asyncio.run(run())


def test_cancelled_and_terminated_previews_show_reason_details():
    cancelled = _job(A, status="CANCELLED", reason="cancelled_by_client", reason_details="cancelled from the [b]CLI[/b]")
    terminated = _job(B, status="TERMINATED", reason="idle_timeout", reason_details="no request for\n30 minutes")

    async def run():
        c = _client([cancelled, terminated])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            p = await _open_preview(app, pilot, c, cancelled)
            assert "■ Cancelled  reason: cancelled_by_client — cancelled from the [b]CLI[/b]" in _text(p, "#health")
            await pilot.press("b")
            p = await _open_preview(app, pilot, c, terminated)
            assert ("■ Terminated by the platform  reason: idle_timeout — no request for 30 minutes"
                    in _text(p, "#health"))
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_unknown_job_the_control_plane_has_no_record_of_is_not_called_running():
    gone = {**_unknown(E, []), "reason": "Job status unavailable: job not found on Control Plane"}
    unreachable = _unknown(U, [])

    async def run():
        c = _client([gone, unreachable])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            p = await _open_preview(app, pilot, c, gone)
            health = _text(p, "#health")
            assert "job not found on Control Plane" in health and "may still be running" not in health
            await pilot.press("b")
            p = await _open_preview(app, pilot, c, unreachable)
            assert "Control Plane is unavailable — the job may still be running" in _text(p, "#health")
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_detail_status_refresh_clears_a_reason_the_service_stopped_sending():
    pending = _job(A, status="PENDING", reason="quota_filled", reason_details="needs 8 GPUs; 64 currently active")
    placing = {k: val for k, val in {**pending, "status": "PLACING"}.items() if k not in ("reason", "reason_details")}

    async def run():
        c = _client([pending])  # the details' first get_job still says PENDING, with the reason
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(180, 40)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobDetailScreen(c, A, job=pending, initial_tab="resource"))
            # The reason and its details are on the summary line; the subtitle keeps to the status.
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                               and "quota_filled" in _text(app.screen, "#summary"))
            d = app.screen
            assert "needs 8 GPUs" in _text(d, "#summary")
            assert "PENDING" in app.sub_title and "quota_filled" not in app.sub_title
            d._apply_status(placing)  # what the 60 s status refresh posts
            await pilot.pause()
            assert "PLACING" in app.sub_title and "quota_filled" not in app.sub_title
            assert "quota_filled" not in _text(d, "#summary") and "needs 8 GPUs" not in _text(d, "#summary")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_live_preview_rereads_the_job_on_return_from_the_details():
    pending = _job(A, status="PENDING", reason="quota_filled")
    running = {k: val for k, val in {**pending, "status": "RUNNING"}.items() if k != "reason"}
    answers = {"now": pending}

    async def run():
        c = _client([pending])
        c.get_job.side_effect = lambda jid: answers["now"]
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            p = await _open_preview(app, pilot, c, pending)
            assert "Pending" in _text(p, "#health")
            await pilot.press("g")  # the details, on their Resources tab
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
            answers["now"] = running  # the job starts while the details are open
            app.pop_screen()
            assert await _wait(pilot, lambda: app.screen is p and "Running" in _text(p, "#health"))
            assert "quota_filled" not in _text(p, "#health")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_named_sub_job_of_an_unknown_job_without_configs_is_tried_not_reported_missing():
    bare = _unknown(E, [])

    async def run():
        c = _client([bare])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobDetailScreen(c, E, job=bare, sub_job_id=f"{E}:training:0", initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                               and app.screen._current_source == f"{E}:training:0")
            d = app.screen
            assert not d._sub_job_missing
            assert not any("not found" in line for line in d._shown_lines)
            assert await _wait(pilot, lambda: any(call.kwargs.get("sub_job_id") == f"{E}:training:0"
                                                  for call in c.tail_logs.call_args_list))
            await _settle(app, pilot)

    asyncio.run(run())


# ─── review follow-ups: the drawn list, re-reads, failed reads, wrapping ──────


def _drawn(table, row_key) -> str:
    """The text the table draws for a row (the header is line 0; no scrolling)."""
    return "".join(seg.text for seg in table.render_line(1 + table.get_row_index(row_key)))


def _name_column(table, line: str) -> str:
    """The Name cell of a drawn line (ASCII names): after the one-cell marker
    column, with a cell of padding a side."""
    start = 1 + 2 * table.cell_padding
    return line[start: start + table.screen._name_width + 2 * table.cell_padding]


def test_a_long_name_ends_in_an_ellipsis_where_its_column_ends_and_widens_on_a_wide_terminal():
    long_a = "qwen3-8b lora sweep · lr 1e-4 · warmup 100 · run A"
    long_b = "qwen3-8b lora sweep · lr 1e-4 · warmup 100 · run B"  # the same first 49 characters
    jobs = [_job(A, status="RUNNING", comment=long_a), _job(B, status="RUNNING", comment=long_b)]
    AliasStore().set(C, "short alias")
    jobs.append(_job(C, status="RUNNING", comment="a server comment that is far too long to fit beside the alias"))

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            s = await _open_list(app, pilot)
            table = s.query_one("#jobs", DataTable)
            assert s._name_width == 30  # no room to spare: every column needs ~158 cells
            for jid in (A, B, C):
                cell = _name_column(table, _drawn(table, _item(s, jid)))
                assert cell_len(cell) == 32 and cell.endswith("… "), cell  # cut at the edge, with the ellipsis
            assert _name_column(table, _drawn(table, _item(s, A))).strip() == "▾ qwen3-8b lora sweep · lr 1e…"

            await pilot.resize_terminal(200, 40)
            # What 200 columns leave once every column fits (Type takes 13 cells, for train+logprob).
            assert await _wait(pilot, lambda: s._name_width == 69)
            await pilot.pause()
            assert table.virtual_size.width <= table.scrollable_content_region.width, "no sideways scrolling"
            for jid, name in ((A, long_a), (B, long_b)):
                assert _name_column(table, _drawn(table, _item(s, jid))).strip() == f"▾ {name}"  # told apart
            alias_cell = _name_column(table, _drawn(table, _item(s, C))).strip()
            assert alias_cell.startswith("▾ short alias  a server comment") and alias_cell.endswith("…")

            table.move_cursor(row=table.get_row_index(_item(s, B)))
            await pilot.resize_terminal(140, 40)
            assert await _wait(pilot, lambda: s._name_width == 30)
            assert s._highlighted()[1] == B, "the cursor stays on its job"
            assert _name_column(table, _drawn(table, _item(s, B))).endswith("… ")
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_rename_prompt_does_not_reread_a_live_job():
    job = _job(A, status="RUNNING", comment="c")

    async def run():
        c = _client([job])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            p = await _open_preview(app, pilot, c, job)
            assert await _wait(pilot, lambda: not p._loading)
            calls = c.get_job.call_count
            await pilot.press("n")
            assert await _wait(pilot, lambda: isinstance(app.screen, PromptScreen))
            await pilot.press("escape")
            assert await _wait(pilot, lambda: app.screen is p)
            for _ in range(10):
                await pilot.pause()
            assert c.get_job.call_count == calls, "a live get_job can block for minutes: not for a prompt"
            await pilot.press("g")  # the details, then back: that re-reads
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen))
            await pilot.press("b")
            assert await _wait(pilot, lambda: app.screen is p and c.get_job.call_count > calls + 1)
            await _settle(app, pilot)

    asyncio.run(run())


def test_details_of_an_unknown_job_tail_its_listed_configs_when_get_job_fails():
    job = _unknown(U, [
        {"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B", "training_config": {"n_gpus": 8}},
        {"status": "UNKNOWN", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2}},
    ])

    async def run():
        c = _client([job])
        c.get_job.side_effect = RuntimeError("HTTP 500 Internal Server Error")
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            await _open_list(app, pilot)
            app.push_screen(JobDetailScreen(c, U, job=job, initial_tab="logs"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            d = app.screen
            assert list(d._source_by_item.values()) == [f"{U}:training:0", f"{U}:sampling:0"]
            assert not any("no sub-jobs" in line for line in d._shown_lines)
            assert "Job status unavailable: Control Plane is unavailable" in _text(d, "#summary")
            await _settle(app, pilot)

    asyncio.run(run())


def test_markup_and_control_characters_from_the_service_are_shown_literally():
    reason = "Job status unavailable: [@click=app.quit]quit[/] [/x] $value \x1b[31mred"
    host = _job(A, status="RUNNING")
    odd = {**_unknown(U, [{"status": "UNKNOWN", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B",
                           "training_config": {"n_gpus": 8, "multiplex_job_id": "[b]host[/b]:training:0\x1b[0m"}}]),
           "reason": reason}

    async def run():
        c = _client([host, odd])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(200, 45)) as pilot:
            s = await _open_list(app, pilot)
            shown = "Job status unavailable: [@click=app.quit]quit[/] [/x] $value \ufffd[31mred"
            assert shown in " ".join(_text(s, "#unavailable").split())
            assert _row(s, f"sub-{_item(s, U)}-0")[8] == "shared"
            p = await _open_preview(app, pilot, c, odd)
            assert shown in " ".join(_text(p, "#health").split())
            gpus = " ".join(_text(p, "#preview").split())
            assert "runs on the GPUs of [b]host[/b]:training:0\ufffd[0m" in gpus and "\x1b" not in gpus
            config = str(p.query_one("#subjobs", DataTable).get_row("sj-0")[5])
            assert config == "full-param · runs on [b]host[…:training:0\ufffd[0m"
            app.push_screen(JobDetailScreen(c, U, job=odd, initial_tab="resource"))
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                               and "unavailable" in _text(app.screen, "#summary"))
            assert shown in _text(app.screen, "#summary")
            assert app.return_code is None, "a [@click] in the service's text must not run an action"
            await _settle(app, pilot)

    asyncio.run(run())


def _indent(line: str) -> int:
    """Leading spaces before the text (-1 for a blank line)."""
    return len(line) - len(line.lstrip(" ")) if line.strip() else -1


def test_long_values_wrap_under_where_they_start():
    host = _job(A, status="RUNNING")
    guest = _job(B, status="RUNNING", comment="probe on the main run's GPUs", sub_jobs=[
        {"sub_job_id": f"{B}:training:0", "job_type": "TRAINING", "status": "RUNNING", "model_name": "Qwen/Qwen3-8B",
         "training_config": {"n_gpus": 8, "multiplex_job_id": f"{A}:training:0"}},
        {"sub_job_id": f"{B}:sampling:0", "job_type": "SAMPLING", "status": "RUNNING", "model_name": "Qwen/Qwen3-8B",
         "inference_config": {"n_gpus": 4}},
    ])
    failed = _job(C, status="FAILED", reason="node_evacuated",
                  reason_details="The node running this job was taken out of service; resume from its last "
                                 "checkpoint, or submit it again if it saved none")
    gone = [{**_unknown(f"fake-job-u{i}", []), "comment": f"run {i}"} for i in range(2)]
    other = {**_unknown("fake-job-u9", []), "reason": "Job status unavailable: Control Plane request timed out"}

    async def run():
        c = _client([host, guest, failed, *gone, other])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(80, 45)) as pilot:
            s = await _open_list(app, pilot)
            banner = _text(s, "#unavailable").split("\n")
            assert len(banner) > 1 and banner[0].startswith("   ? Job status unavailable")
            assert all(_indent(line) == 5 for line in banner[1:]), banner  # under the sentence
            assert all(cell_len(line) <= 80 for line in banner)
            p = await _open_preview(app, pilot, c, guest)
            width = p.query_one("#preview", Static).size.width
            lines = _text(p, "#preview").split("\n")
            start = next(i for i, line in enumerate(lines) if line.strip().startswith("GPUs"))
            gpus = lines[start:next(i for i, line in enumerate(lines) if line.strip().startswith("Created"))]
            assert len(gpus) > 1 and all(_indent(line) == 15 for line in gpus[1:]), gpus  # under the value
            assert all(cell_len(line) <= width for line in gpus)
            assert f"{A}:training:0)" in " ".join(gpus)
            await pilot.press("b")
            p = await _open_preview(app, pilot, c, failed)
            health = _text(p, "#health").split("\n")
            assert len(health) > 1 and health[0].startswith("  ✖ Job failed  reason: node_evacuated — The node")
            assert all(_indent(line) == 4 for line in health[1:]), health  # after "  ✖ "
            assert all(cell_len(line) <= width for line in health)
            assert p.query_one("#health", Static).size.height == len(health), "no line soft-wraps"
            await _settle(app, pilot)

    asyncio.run(run())


def test_reason_details_without_a_reason_get_no_dash():
    cancelled = _job(A, status="CANCELLED", reason="", reason_details="cancelled from the console")
    log_prob = _job(B, status="RUNNING", sub_jobs=[{"sub_job_id": f"{B}:log_prob:0", "job_type": "LOG_PROBABILITY",
                                                    "status": "RUNNING", "inference_config": {"n_gpus": 2}}])

    async def run():
        c = _client([cancelled, log_prob])
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            p = await _open_preview(app, pilot, c, cancelled)
            assert _text(p, "#health") == "  ■ Cancelled  cancelled from the console"
            await pilot.press("b")
            p = await _open_preview(app, pilot, c, log_prob)
            gpus = next(line for line in _text(p, "#preview").split("\n") if line.strip().startswith("GPUs"))
            assert gpus.split("GPUs", 1)[1].strip() == "0 requested  (log_prob #0 uses none)"
            assert str(p.query_one("#subjobs", DataTable).get_row("all")[4]) == "0"
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_list_search_finds_the_end_of_a_long_wide_alias():
    AliasStore().set(A, "あ" * 60 + "最後の部分は検索できない")  # kept whole (72 characters), shown cut
    jobs = [_job(A, status="RUNNING"), _job(B, status="RUNNING", comment="another job")]

    async def run():
        app = CortexTrainingLogTUI(_client(jobs), poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            s = await _open_list(app, pilot)
            box = s.query_one("#jobfilter", Input)
            box.value = "できない"
            assert await _wait(pilot, lambda: list(s._job_by_item.values()) == [A])
            box.value = "name:検索できない"
            assert await _wait(pilot, lambda: list(s._job_by_item.values()) == [A])
            await _settle(app, pilot)

    asyncio.run(run())
