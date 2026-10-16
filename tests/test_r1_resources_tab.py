# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Resources tab in the running app: first paint, freshness, waiting sub-jobs, table and layout."""

import asyncio
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from tests.test_resources_tab import _client  # noqa: E402
from tests.test_resources_tab import _rows  # noqa: E402

TRAINING = "7:training:0"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _write(output_dir, rows, sub_job_id=TRAINING, *, partial=False, names=("chunk.gz",)):
    """Commit ``rows`` as download_metrics does: a file moved into place, then its manifest."""
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".gpu-metrics-test.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(temporary, path)
    manifest = {"version": 1, "names": list(names), "bytes": path.stat().st_size}
    if partial:
        manifest["partial"] = True
    path.with_name(".gpu.jsonl.manifest.json").write_text(json.dumps(manifest))


def _text(screen, selector):
    return str(screen.query_one(selector, Static).render())


async def _wait(pilot, predicate, tries=400):
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


def _ts(row):
    return datetime.fromisoformat(row["sample_ts"].replace("Z", "+00:00"))


# ─── R1-21: the shown sub-job's newest window first ──────────────────────────


def test_a_cold_open_paints_the_shown_sub_jobs_newest_window_before_its_history_and_the_other_sub_jobs():
    c = _client()
    calls = []
    gate = threading.Event()

    def download(job_id, output_dir, *, resume=False, priority=(), newest_first_seconds=None, on_commit=None):
        calls.append((list(priority), newest_first_seconds))
        rows = _rows(minutes=180)
        newest = _ts(rows[-1])
        _write(output_dir, [row for row in rows if _ts(row) >= newest - timedelta(minutes=13)], partial=True,
               names=("newest.gz",))
        on_commit({"sub_job_id": TRAINING, "complete": False})
        assert gate.wait(30)
        _write(output_dir, rows, names=("older.gz", "newest.gz"))
        on_commit({"sub_job_id": TRAINING, "complete": True})
        _write(output_dir, _rows("7:sampling:0", minutes=30), "7:sampling:0")
        on_commit({"sub_job_id": "7:sampling:0", "complete": True})
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            # Painted from the first, partial commit: the window is complete, the whole job isn't yet.
            assert await _wait(pilot, lambda: screen.query_one("#resource-table", DataTable).row_count == 2)
            status, stats = _text(screen, "#resource-status"), _text(screen, "#resource-stats")
            assert "loading older samples…" in status
            busy = next(line for line in stats.splitlines() if line.startswith("GPU busy"))
            assert busy.split()[4:8] == ["85%", "85%", "85%", "85%"]  # last 10m: min avg max latest
            assert busy.count("loading older samples…") == 1  # the whole job's columns
            assert screen._resource_loading
            gate.set()
            assert await _wait(pilot, lambda: not screen._resource_loading)
            stats = _text(screen, "#resource-stats")
            assert "loading" not in _text(screen, "#resource-status") + stats
            assert set(screen._resource_data.sub_jobs) == {TRAINING, "7:sampling:0"}
            await _settle(app, pilot)

    asyncio.run(run())
    # The sub-job on screen first, and its download starts with the default window plus slack.
    assert calls == [([TRAINING], 600 + 180)]


# ─── R1-73: the charts take a tall pane's spare rows ─────────────────────────

from tests.test_resources_tab import _download  # noqa: E402


def _bar_rows(screen, selector):
    lines = _text(screen, selector).splitlines()
    return [line for line in lines[: next(i for i, line in enumerate(lines) if line.startswith("  0% └"))]]


async def _open(app, pilot):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
    await pilot.press("3")
    assert await _wait(pilot, lambda: app.screen._resource_loaded and not app.screen._resource_loading
                       and app.screen.query_one("#resource-table", DataTable).row_count > 0)
    await pilot.pause()
    return app.screen


@pytest.mark.parametrize(("size", "bars"), [((150, 44), {8}), ((140, 42), {8}), ((100, 30), {2, 3, 4, 5}),
                                           ((80, 24), {2})])
def test_the_charts_grow_into_spare_rows_and_the_table_keeps_every_gpu(size, bars):
    c = _client(n_gpus=8)
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot)
            rows = len(_bar_rows(screen, "#resource-chart"))
            assert rows in bars
            if list(screen.query(".resource-chart-box"))[1].display:  # a short narrow pane shows one
                assert len(_bar_rows(screen, "#resource-mem-chart")) == rows
            table = screen.query_one("#resource-table", DataTable)
            keys = screen.query_one("#resource-keys", Static)
            if rows > 2:  # grown past every chart's two rows: all eight GPUs stay in view, above the key help
                assert table.region.height >= 9 and table.region.y + table.region.height <= keys.region.y
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-70: the highlighted GPU and the scroll survive w and refreshes ───────


def _cursor_key(table):
    return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value


def test_the_highlighted_gpu_and_the_scroll_survive_a_new_window_and_a_refresh():
    c = _client(n_gpus=32)
    calls = {"n": 0}

    def download(job_id, output_dir, *, resume=False, on_commit=None, **_options):
        calls["n"] += 1  # each refresh brings newer samples for the same 32 GPUs
        _write(output_dir, _rows(gpus=32, minutes=20, newest_age=60.0 - 10 * calls["n"]))
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 34)) as pilot:
            screen = await _open(app, pilot)
            table = screen.query_one("#resource-table", DataTable)
            assert table.region.height < 33  # the 32 rows scroll
            for _ in range(26):
                await pilot.press("down")
            await pilot.pause()
            scrolled = table.scroll_y
            assert _cursor_key(table) == "26" and scrolled > 0
            await pilot.press("w")  # new column labels: the rows are rebuilt
            assert await _wait(pilot, lambda: table.scroll_y == scrolled)
            assert _cursor_key(table) == "26" and str(table.columns[next(iter(table.columns))].label) == "GPU"
            assert "Busy avg 30m" in [str(column.label) for column in table.columns.values()]
            before = calls["n"]
            await pilot.press("r")
            assert await _wait(pilot, lambda: calls["n"] > before and not screen._resource_loading)
            await pilot.pause()
            assert _cursor_key(table) == "26" and table.scroll_y == scrolled
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-71: a narrow table drops the columns with nothing measured first ─────


def test_a_narrow_table_keeps_measured_profiling_gauges_and_drops_the_unmeasured_one():
    c = _client(n_gpus=8)
    rows = _rows(gpus=8)
    for row in rows:
        row["tensor_active_pct"], row["dram_active_pct"], row["sm_active_pct"] = 65.0, 37.0, None

    def download(job_id, output_dir, *, resume=False, **_options):
        _write(output_dir, rows)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(100, 40)) as pilot:
            screen = await _open(app, pilot)
            labels = [str(column.label) for column in screen.query_one("#resource-table", DataTable).columns.values()]
            assert "Tensor active" in labels and "DRAM active" in labels and "SM active" not in labels
            await pilot.resize_terminal(150, 40)  # wide enough for every column: SM active shows, all "—"
            assert await _wait(pilot, lambda: "SM active" in [
                str(column.label) for column in screen.query_one("#resource-table", DataTable).columns.values()])
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-66: a running sub-job whose newest sample is stale ───────────────────


def _status_markup(screen) -> str:
    return screen.query_one("#resource-status", Static).render().markup


@pytest.mark.parametrize("status", ["RUNNING", "CANCELLED"])
def test_a_running_sub_job_without_new_samples_says_so_and_has_no_gpu_reporting(status):
    c = _client(status=status)
    _download(c, newest_age=41 * 60 + 56)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open(app, pilot)
            text = _text(screen, "#resource-status")
            table = screen.query_one("#resource-table", DataTable)
            gpus = [str(table.get_row_at(index)[0]) for index in range(table.row_count)]
            if status == "RUNNING":
                assert "GPUs: 4 requested, 0 reporting (2 seen)" in text
                assert "no new GPU samples for 41m 5" in text and "; the windows end at " in text
                assert "[$warning]no new GPU samples" in _status_markup(screen)
                assert gpus == ["0 (stale)", "1 (stale)"]
            else:  # a finished job's newest sample is simply its last
                assert "GPUs: 4 requested, 2 reporting" in text and "newest 41m 5" in text
                assert "no new GPU samples" not in text and gpus == ["0", "1"]
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-68: sub-jobs that haven't started yet ────────────────────────────────


@pytest.mark.parametrize(("job_status", "sub_status", "expected"), [
    ("RUNNING", "PLACING", "no GPU metrics yet: training #0 is PLACING (the first samples arrive about a minute "
                           "after it starts running)"),
    ("PENDING", "PENDING", "no GPU metrics yet: the job is PENDING (the first samples arrive"),
])
def test_a_sub_job_that_has_not_started_says_its_samples_are_still_to_come_and_is_not_polled(
        job_status, sub_status, expected):
    c = _client(status=job_status)
    for sub_job in c.get_job.return_value["sub_jobs"]:
        sub_job["status"] = sub_status
    c.download_metrics.return_value = []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: screen._resource_loaded and not screen._resource_loading)
            assert expected in _text(screen, "#resource-status")
            assert not screen.query_one("#resource-table", DataTable).display  # no empty header band
            downloads = c.download_metrics.call_count
            screen._auto_refresh_data()  # the 30-second timer: nothing to poll for yet
            await pilot.pause()
            assert c.download_metrics.call_count == downloads and not screen._resource_loading
            # Once the status check sees it running, the timer polls again.
            screen._job_status = "RUNNING"
            screen._sub_job_status = {key: "RUNNING" for key in screen._sub_job_status}
            screen._auto_refresh_data()
            assert await _wait(pilot, lambda: c.download_metrics.call_count == downloads + 1)
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-72: a failed refresh, in the warning colour and on one line ──────────


def test_a_failed_refresh_is_a_warning_cut_to_the_line_with_the_whole_error_in_the_log(tmp_path):
    c = _client(status="CANCELLED")
    calls = {"n": 0}
    reason = "stage LIST failed: connection reset by peer while reading the response " + "x" * 80

    def download(job_id, output_dir, *, resume=False, **_options):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError(reason)
        _write(output_dir, _rows())
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("r")
            assert await _wait(pilot, lambda: calls["n"] == 2 and not screen._resource_loading)
            await pilot.pause()
            status = screen.query_one("#resource-status", Static)
            second = _text(screen, "#resource-status").splitlines()[1]
            assert "cached · refresh failed: ConnectionError: stage LIST failed: connection reset" in second
            assert second.endswith("…") and len(second) <= status.content_size.width
            assert "[$warning]refresh failed: $problem[/]" not in _status_markup(screen)  # filled in
            assert "[$warning]refresh failed: ConnectionError" in _status_markup(screen)
            await _settle(app, pilot)

    asyncio.run(run())
    log = Path(os.environ["HOME"]) / ".cortex-training-errors.log"
    assert reason in log.read_text()


# ─── R1-96: a job with no sub-jobs ───────────────────────────────────────────


def test_a_job_without_sub_jobs_says_so_on_the_resources_tab():
    c = _client(status="CANCELLED")
    c.get_job.return_value["sub_jobs"] = []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._sources_ready)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: "no sub-jobs" in _text(screen, "#resource-status"))
            assert _text(screen, "#resource-status") == (
                "this job has no sub-jobs, and no GPU metrics are cached on this machine")
            assert not screen.query_one("#resource-table", DataTable).display
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-21 follow-up: a download that ends without the older history ─────────

import dataclasses  # noqa: E402
import re  # noqa: E402

from rich.cells import cell_len  # noqa: E402

from tests.test_resources_tab import _titles  # noqa: E402


def _stat_line(screen, label="GPU busy"):
    return next(line for line in _text(screen, "#resource-stats").splitlines() if line.startswith(label))


@pytest.mark.parametrize("status", ["RUNNING", "CANCELLED"])
def test_a_download_that_fails_after_the_newest_chunks_says_why_instead_of_loading_forever(status):
    c = _client(status=status)
    rows = _rows(minutes=180)
    newest = _ts(rows[-1])
    recent = [row for row in rows if _ts(row) >= newest - timedelta(minutes=13)]
    reason = "simulated stage GET failure: connection reset by peer"
    # A live job's 30-second timer retries (and fails again); a finished job waits for r.
    outcomes = ["fail", "fail", "ok"] if status == "RUNNING" else ["fail", "ok"]
    calls = []

    def download(job_id, output_dir, *, resume=False, priority=(), newest_first_seconds=None, on_commit=None):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(outcome)
        if outcome == "fail":
            # The newest chunks commit (the manifest says "partial"); then a GET of an older one fails.
            _write(output_dir, recent, partial=True, names=("newest.gz",))
            on_commit({"sub_job_id": TRAINING, "complete": False})
            raise ConnectionError(reason)
        _write(output_dir, rows, names=("older.gz", "newest.gz"))
        on_commit({"sub_job_id": TRAINING, "complete": True})
        return []

    c.download_metrics.side_effect = download

    def assert_failed(screen):
        header = _text(screen, "#resource-status")
        assert "older samples failed to load: ConnectionError: simulated stage GET failure" in header
        assert "[$warning]older samples failed to load: ConnectionError" in _status_markup(screen)
        assert "loading" not in header + _text(screen, "#resource-stats")
        busy = _stat_line(screen)
        assert busy.split()[4:8] == ["85%", "85%", "85%", "85%"]  # the window: every sample is in
        assert busy.count("older samples not loaded") == 1  # the whole job: not

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: calls and not screen._resource_loading)
            await pilot.pause()
            assert screen._resource_sub_job().partial
            assert_failed(screen)
            # A window the newest chunks don't cover says so in its titles, stats and table.
            screen.action_resource_range("30m")
            await pilot.pause()
            assert all(title.endswith("(older samples not loaded)") for title in _titles(screen))
            assert _stat_line(screen).count("older samples not loaded") == 2
            table = screen.query_one("#resource-table", DataTable)
            labels = [str(column.label) for column in table.columns.values()]
            first = [str(cell) for cell in table.get_row_at(0)]
            assert first[labels.index("Busy avg 30m")] == "—" and first[labels.index("Mem avg 30m")] == "—"
            screen.action_resource_range("10m")
            await pilot.pause()
            screen._auto_refresh_data()  # the 30-second timer
            if status == "RUNNING":
                assert await _wait(pilot, lambda: len(calls) == 2 and not screen._resource_loading)
                await pilot.pause()
                assert_failed(screen)
            else:
                await pilot.pause()
                assert len(calls) == 1 and not screen._resource_loading
                assert_failed(screen)
            await pilot.press("r")
            assert await _wait(pilot, lambda: calls[-1] == "ok" and not screen._resource_loading)
            await pilot.pause()
            text = _text(screen, "#resource-status") + _text(screen, "#resource-stats")
            assert "older samples" not in text and "failed" not in text and "loading" not in text
            assert len([cell for cell in _stat_line(screen).split() if re.fullmatch(r"\d+%", cell)]) == 8
            await _settle(app, pilot)

    asyncio.run(run())
    log = Path(os.environ["HOME"]) / ".cortex-training-errors.log"
    assert log.read_text().count(reason) == len(outcomes) - 1  # every failure, in full


# ─── R1-72 and the header's height: two lines in every state, at every width ─

GIB = 1024**3
_LONG_ERROR = "ConnectionError: stage LIST failed: connection reset by peer while reading the response body"


def _header_lines(*, gpus=8, minutes=20, newest_age=40.0):
    """JSONL for ``gpus`` GPUs every 5 s whose busy values change every 45 s, and three rows the loader skips."""
    newest = datetime.now(timezone.utc) - timedelta(seconds=newest_age)
    lines = []
    for step in range(minutes * 12, -1, -1):
        ts = (newest - timedelta(seconds=5 * step)).isoformat().replace("+00:00", "Z")
        for gpu in range(gpus):
            lines.append(json.dumps({
                "record_version": 2, "sub_job_id": TRAINING, "sample_ts": ts, "worker_num": gpu // 4,
                "local_gpu_index": gpu % 4, "global_gpu_index": gpu,
                "gpu_util_pct": float(60 + (step // 9 * 7 + gpu) % 30),
                "fb_used_bytes": float(40 * GIB), "fb_free_bytes": float(40 * GIB), "sample_interval_s": 5.0,
            }))
    lines += ["{not json", json.dumps({"record_version": 99}), json.dumps({"record_version": 2})]
    return lines


def _write_lines(output_dir, lines, sub_job_id=TRAINING):
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}))


def _same_parts(line: str) -> str:
    return re.sub(r"\d", "0", line)  # the age and the clock tick between two renders


@pytest.mark.parametrize("stale", [False, True], ids=["fresh", "stale"])
@pytest.mark.parametrize("size", [(60, 24), (80, 24), (98, 30), (100, 30), (120, 34), (140, 42)])
def test_the_header_keeps_two_lines_and_the_charts_their_place_through_refreshes_and_failures(size, stale):
    c = _client(n_gpus=8)
    lines = _header_lines(newest_age=41 * 60 + 56 if stale else 40.0)

    def download(job_id, output_dir, **_options):
        _write_lines(output_dir, lines)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot)
            status = screen.query_one("#resource-status", Static)
            title = list(screen.query(".resource-chart-title").results(Static))[0]
            shown = {}
            data = screen._resource_data
            for state in ("idle", "refreshing", "failed", "idle again", "older failed"):
                screen._resource_loading = state == "refreshing"
                screen._resource_error = _LONG_ERROR if "failed" in state else None
                if state == "older failed":  # the newest chunks are in, a GET of an older one failed
                    partial = dataclasses.replace(data.sub_jobs[TRAINING], partial=True)
                    screen._resource_data = dataclasses.replace(data, sub_jobs={TRAINING: partial})
                screen._render_resource()
                await pilot.pause()
                await pilot.pause()
                header = _text(screen, "#resource-status").split("\n")
                assert len(header) == 2, (state, header)
                assert all(cell_len(line) <= status.content_size.width for line in header), (state, header)
                assert status.region.height == 2, state
                shown[state] = header, title.region.y, len(_bar_rows(screen, "#resource-chart"))
            # Nothing below the header moves.
            assert len({(y, rows) for _header, y, rows in shown.values()}) == 1, shown
            idle, refreshing, failed = (shown[state][0][1] for state in ("idle", "refreshing", "failed"))
            assert _same_parts(shown["idle again"][0][1]) == _same_parts(idle)
            # A refresh adds its state and changes nothing else on the line.
            state = "cached · refreshing…" if "cached · refreshing…" in refreshing else "refreshing…"
            assert _same_parts(refreshing.replace(f"  ·  {state}", "", 1)) == _same_parts(idle)
            # A failure shows the start of the error, at least 40 cells of it once there is room.
            assert "refresh failed: Con" in failed and "…" in failed  # cut (the skipped rows may follow)
            if size[0] >= 98:
                assert "refresh failed: " + _LONG_ERROR[:39] in failed
            if size[0] >= 140 and not stale:
                assert "values change about every 45 s" in idle and "3 rows skipped" in idle
            # The same for older samples that failed to load: their state, cut (at 60 columns
            # beside the stale warning, into the state itself), and the start of the error.
            older = shown["older failed"][0][1]
            assert "older samples fail" in older and "…" in older
            if size[0] >= 98:
                assert re.search(r"older samples failed( to load)?: " + re.escape(_LONG_ERROR[:30]), older)
            assert "older samples not loaded" in _text(screen, "#resource-stats")
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("width", [55, 75, 93, 95, 115, 135])
def test_the_age_ticking_never_changes_what_else_the_header_shows(width):
    c = _client(status="CANCELLED", n_gpus=8)  # finished: an old newest sample isn't stale
    lines = _header_lines(newest_age=5.0)

    def download(job_id, output_dir, **_options):
        _write_lines(output_dir, lines)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open(app, pilot)
            sub_job = screen._resource_sub_job()
            seconds = set()
            for age in (5, 9, 10, 59, 60, 64, 599, 600, 3599, 3600, 86399):
                moved = dataclasses.replace(
                    sub_job, latest_sample_ts=datetime.now(timezone.utc) - timedelta(seconds=age))
                header = str(screen._resource_status(moved, width)).split("\n")
                assert len(header) == 2 and all(cell_len(line) <= width for line in header), (age, header)
                second = re.sub(r"newest .+? ago", "newest AGE ago", header[1])
                seconds.add(re.sub(r"\(\d\d:\d\d:\d\d\)", "(CLOCK)", second))
            assert len(seconds) == 1, seconds
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-96 (6): the keys reach the table on a terminal too short for the tab ──


def test_on_a_short_terminal_the_cursor_keys_bring_the_table_into_view_and_home_goes_back_up():
    c = _client(n_gpus=8)
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(60, 16)) as pilot:
            screen = await _open(app, pilot)
            pane = screen.query_one("#resource")
            table = screen.query_one("#resource-table", DataTable)
            keys = screen.query_one("#resource-keys", Static)

            def in_view() -> bool:
                return table.region.height > 0 and pane.region.y <= table.region.y and (
                    table.region.bottom <= keys.region.y)

            async def press(key):
                await pilot.press(key)
                await pilot.pause()
                await pilot.pause()

            assert screen.focused is table and pane.max_scroll_y > 0
            assert pane.scroll_y == 0 and not in_view()  # the tab opens at its top
            await press("down")
            assert table.cursor_row == 1 and in_view()
            await press("up")
            assert table.cursor_row == 0 and in_view()
            await press("up")  # past the first GPU: back to the top of the tab
            assert pane.scroll_y == 0 and table.cursor_row == 0
            await press("pagedown")
            assert in_view() and table.cursor_row > 0
            await press("home")
            assert pane.scroll_y == 0 and table.cursor_row == 0
            await press("end")
            assert in_view() and table.cursor_row == 7 and pane.scroll_y == pane.max_scroll_y
            await press("pageup")
            assert in_view() and table.cursor_row < 7
            await press("ctrl+home")
            assert pane.scroll_y == 0 and table.cursor_row == 0
            # A refresh at the top of the tab leaves it there, with the cursor where it was.
            await press("down")
            await press("down")
            await press("home")
            await press("r")
            assert await _wait(pilot, lambda: not screen._resource_loading)
            await pilot.pause()
            assert pane.scroll_y == 0 and table.cursor_row == 0
            await _settle(app, pilot)

    asyncio.run(run())


# ─── The side-by-side charts are the same width ──────────────────────────────


def _axis(screen, selector):
    return next(line for line in _text(screen, selector).splitlines() if line.startswith("  0% └")).count("─")


@pytest.mark.parametrize("size", [(140, 42), (141, 42), (120, 34)])
def test_the_two_charts_side_by_side_are_drawn_to_the_same_width(size):
    c = _client(n_gpus=8)
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot)
            busy, memory = _axis(screen, "#resource-chart"), _axis(screen, "#resource-mem-chart")
            assert busy == memory and busy >= size[0] // 2 - 12
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_table_takes_the_focus_back_when_its_rows_arrive_after_a_message():
    c = _client(n_gpus=8)
    gate = threading.Event()

    def download(job_id, output_dir, **_options):
        assert gate.wait(30)
        _write(output_dir, _rows(gpus=8))
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            table = screen.query_one("#resource-table", DataTable)
            assert await _wait(pilot, lambda: not table.display)  # "loading GPU metrics…" with no table
            # Hidden, the table lost the focus; Textual may hand it to the tab bar.
            screen.set_focus(screen.query_one("ContentTabs"))
            gate.set()
            assert await _wait(pilot, lambda: table.display and table.row_count == 8)
            await pilot.pause()
            assert screen.focused is table
            await pilot.press("down")
            assert table.cursor_row == 1
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_columns_shown_when_rows_first_arrive_are_fitted_to_the_table_once_it_is_measured():
    c = _client(n_gpus=8)
    rows = _rows(gpus=8)
    for row in rows:
        row["tensor_active_pct"], row["dram_active_pct"], row["sm_active_pct"] = 65.0, 37.0, None
        # An H200's memory: "110.0 GiB / 140.4 GiB" leaves DRAM active one cell to spare.
        row["fb_used_bytes"], row["fb_free_bytes"] = 110.0 * GIB, 30.4 * GIB
    gate = threading.Event()

    def download(job_id, output_dir, **_options):
        assert gate.wait(30)
        _write(output_dir, rows)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            table = screen.query_one("#resource-table", DataTable)
            assert await _wait(pilot, lambda: not table.display)  # hidden under "loading GPU metrics…"
            gate.set()
            assert await _wait(pilot, lambda: table.row_count == 8 and not screen._resource_loading)
            for _ in range(3):
                await pilot.pause()
            # Every measured gauge fits the 97-cell table; only the unmeasured SM active goes. (The
            # rows arrive while the table is hidden, unmeasured, under "loading GPU metrics…".)
            labels = [str(column.label) for column in table.columns.values()]
            assert labels[-2:] == ["Tensor active", "DRAM active"], labels
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_narrow_table_keeps_its_window_averages_while_the_older_samples_are_missing():
    c = _client(status="CANCELLED", n_gpus=8)
    rows = _rows(gpus=8, minutes=180)
    for row in rows:
        row["tensor_active_pct"], row["fb_used_bytes"], row["fb_free_bytes"] = 65.0, 110.0 * GIB, 30.4 * GIB
    newest = _ts(rows[-1])
    recent = [row for row in rows if _ts(row) >= newest - timedelta(minutes=13)]
    calls = []

    def download(job_id, output_dir, *, on_commit=None, **_options):
        calls.append(1)
        if len(calls) == 1:
            _write(output_dir, recent, partial=True, names=("newest.gz",))
            on_commit({"sub_job_id": TRAINING, "complete": False})
            raise ConnectionError("simulated stage GET failure")
        _write(output_dir, rows, names=("older.gz", "newest.gz"))
        return []

    c.download_metrics.side_effect = download

    def labels(screen):
        return [str(column.label) for column in screen.query_one("#resource-table", DataTable).columns.values()]

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: calls and not screen._resource_loading)
            screen.action_resource_range("30m")
            await pilot.pause()
            missing = labels(screen)
            table = screen.query_one("#resource-table", DataTable)
            assert "Mem avg 30m" in missing  # "—" here means not loaded, not unmeasured: it isn't dropped first
            first = [str(cell) for cell in table.get_row_at(0)]
            assert {first[missing.index(name)] for name in ("Busy avg 30m", "Mem avg 30m")} == {"—"}
            await pilot.press("r")
            assert await _wait(pilot, lambda: len(calls) == 2 and not screen._resource_loading)
            await pilot.pause()
            assert labels(screen) == missing
            assert str(table.get_row_at(0)[missing.index("Mem avg 30m")]).endswith("%")
            await _settle(app, pilot)

    asyncio.run(run())
