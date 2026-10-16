# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Resources tab in the running app: header, stats, charts, table and windows."""

import asyncio
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Static  # noqa: E402

import cortex_training.tui.app as app_module  # noqa: E402
import cortex_training.tui.resource as resource  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.resource import BUSY_LABEL  # noqa: E402
from cortex_training.tui.resource import MEMORY_LABEL  # noqa: E402
from cortex_training.tui.resource import load_resource_data  # noqa: E402
from cortex_training.tui.resource import window_label  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402

GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _client(*, status="RUNNING", hardware="H200", n_gpus=4, sampling_config=None):
    c = MagicMock()
    job = {
        "job_id": "7",
        "status": status,
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training", "training_config": {"n_gpus": n_gpus}},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling",
             "inference_config": sampling_config or {"n_gpus": 1}},
        ],
    }
    if hardware:
        job["hardware"] = hardware
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.download_stdout_logs.return_value = []
    return c


def _rows(sub_job_id="7:training:0", *, gpus=2, minutes=40, newest_age=40.0, stale_gpu=None):
    """``gpus`` GPUs every 5 s for ``minutes``, the newest sample ``newest_age`` seconds old.

    GPU g runs at 60 + 10g % busy and (30 + 10g) of 80 GiB, except in the last 10
    minutes, when it runs 20 points busier and 8 GiB fuller. ``stale_gpu`` stops
    reporting 10 minutes before the newest sample.
    """
    newest = datetime.now(timezone.utc) - timedelta(seconds=newest_age)
    rows = []
    for step in range(minutes * 12, -1, -1):
        ts = (newest - timedelta(seconds=5 * step)).isoformat().replace("+00:00", "Z")
        hot = step <= 120  # the window is inclusive of its left edge
        for gpu in range(gpus):
            if gpu == stale_gpu and step < 120:
                continue
            used = (30 + 10 * gpu + (8 if hot else 0)) * GIB
            rows.append({
                "record_version": 2, "sub_job_id": sub_job_id, "sample_ts": ts, "worker_num": 0,
                "local_gpu_index": gpu, "global_gpu_index": gpu,
                "gpu_util_pct": 60.0 + 10 * gpu + (20 if hot else 0),
                "fb_used_bytes": float(used), "fb_free_bytes": float(80 * GIB - used),
                "sm_active_pct": None, "tensor_active_pct": 25.0, "dram_active_pct": None,
                "sample_interval_s": 5.0,
            })
    return rows


def _write_metrics(output_dir, sub_job_id="7:training:0", **kwargs):
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in _rows(sub_job_id, **kwargs)))
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size})
    )
    return path


def _download(c, **kwargs):
    def download(job_id, output_dir, *, resume=False, **_options):
        _write_metrics(output_dir, **kwargs)
        return []

    c.download_metrics.side_effect = download


def _commit(output_dir, rows, sub_job_id="7:training:0"):
    """Append rows as download_metrics does: the same file, one more chunk in its manifest."""
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_name(".gpu.jsonl.manifest.json")
    names = json.loads(manifest.read_text())["names"] if manifest.exists() else []
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
    manifest.write_text(json.dumps({"version": 1, "names": [*names, f"c{len(names)}"], "bytes": path.stat().st_size}))


def _text(screen, selector):
    return str(screen.query_one(selector, Static).render())


def _titles(screen):
    return [str(widget.render()) for widget in screen.query(".resource-chart-title").results(Static)]


def _columns(screen):
    return [str(col.label) for col in screen.query_one("#resource-table", DataTable).columns.values()]


async def _wait(pilot, predicate, tries=300):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _open_resources(app, pilot):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
    await pilot.press("3")
    assert await _wait(
        pilot,
        lambda: app.screen._resource_active() and app.screen._resource_loaded and not app.screen._resource_loading
        and app.screen.query_one("#resource-table", DataTable).row_count > 0,
    )
    return app.screen


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def _on_screen(app, widget) -> bool:
    """Whether ``widget`` is displayed with at least one row inside the terminal."""
    region = widget.region
    return widget.display and region.height > 0 and region.y >= 0 and region.y + 1 <= app.size.height


def test_header_stats_charts_and_table():
    c = _client()
    _download(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            status = _text(screen, "#resource-status")
            # GPU type, and requested GPUs (the sub-job's n_gpus) against GPUs in the data.
            # Which sub-job, by name, and its place among the job's two (< and > switch).
            assert status.startswith(
                "training #0 (1 of 2 · < > switch)  ·  last 10m  ·  H200  ·  GPUs: 4 requested, 2 reporting\n")
            assert "scraped every 5 s  ·  newest 4" in status and " s ago (" in status
            stats = _text(screen, "#resource-stats")
            assert BUSY_LABEL in stats and MEMORY_LABEL in stats
            # Every value averages across the GPUs, and the scopes say so.
            assert stats.splitlines()[0].split() == ["last", "10m", "·", "mean", "of", "2", "GPUs",
                                                     "whole", "job", "·", "mean", "of", "2", "GPUs"]
            assert stats.splitlines()[1].split() == ["min", "avg", "max", "latest"] * 2
            busy = next(line for line in stats.splitlines() if line.startswith(BUSY_LABEL)).split()[-8:]
            memory = next(line for line in stats.splitlines() if line.startswith(MEMORY_LABEL)).split()[-8:]
            # GPU 0 runs 60% and GPU 1 70% busy, both 20 points busier in the last 10 minutes.
            assert busy[:4] == ["85%"] * 4  # the window: min, avg, max, latest
            assert busy[4] == "65%" and busy[6:] == ["85%", "85%"]  # the whole job: min, max, latest
            assert 65 < int(busy[5].rstrip("%")) < 85  # its average spans both phases
            # Memory: GPU 0 holds 30 (then 38) of 80 GiB, GPU 1 40 (then 48).
            assert memory[:4] == ["54%"] * 4 and memory[4] == "44%" and memory[6:] == ["54%", "54%"]
            # Named for the sub-job; the long labels and the GPUs are too wide for half the pane together.
            assert _titles(screen) == ["training #0 · GPU busy · last 10m · mean of 2 GPUs",
                                       "training #0 · GPU memory · last 10m · mean of 2 GPUs"]
            assert "100% ┤" in _text(screen, "#resource-chart") and "100% ┤" in _text(screen, "#resource-mem-chart")
            assert screen.query_one("#resource-charts").styles.layout.name == "horizontal"
            assert _columns(screen) == [
                "GPU", "Busy", "Busy avg 10m", "Mem", "Mem avg 10m", "Memory used",
                "Tensor active", "DRAM active", "SM active",
            ]
            row = [str(cell) for cell in screen.query_one("#resource-table", DataTable).get_row_at(1)]
            assert row == ["1", "90%", "90%", "60%", "60%", "48.0 GiB / 80.0 GiB", "25%", "—", "—"]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_gpu_that_stops_reporting_is_marked_stale_and_dimmed():
    c = _client()
    _download(c, gpus=3, stale_gpu=2)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            assert "GPUs: 4 requested, 2 reporting (3 seen)" in _text(screen, "#resource-status")
            table = screen.query_one("#resource-table", DataTable)
            cells = [table.get_row_at(index) for index in range(3)]
            assert [str(row[0]) for row in cells] == ["0", "1", "2 (stale)"]
            assert all(cell.style == "dim" for cell in cells[2]) and all(cell.style == "" for cell in cells[0])
            # Its newest reading stays, flagged; the newest capture leaves it out.
            assert str(cells[2][1]) == "100%"
            assert _text(screen, "#resource-stats").splitlines()[2].split()[-5] == "85%"  # window latest
            await _settle(app, pilot)

    asyncio.run(run())


def test_windows_cycle_from_ten_minutes_to_the_whole_job():
    c = _client()
    _download(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            assert screen._resource_range == "10m"
            assert _text(screen, "#resource-keys").startswith("w window 10m · 30m · 1h · 6h · all")
            seen = []
            for _ in range(5):
                await pilot.press("w")
                await pilot.pause()
                seen.append(screen._resource_range)
                header = f"training #0 (1 of 2 · < > switch)  ·  {window_label(screen._resource_range)}  ·"
                assert header in _text(screen, "#resource-status")
            assert seen == ["30m", "1h", "6h", "all", "10m"]
            await pilot.press("w", "w", "w", "w")
            await pilot.pause()
            stats = _text(screen, "#resource-stats")
            assert "whole job" in stats and "last" not in stats
            assert _columns(screen)[2] == "Busy avg (job)" and _columns(screen)[4] == "Mem avg (job)"
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_finished_job_opens_on_ten_minutes_and_a_job_ending_keeps_the_window():
    c = _client(status="FAILED")
    _download(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            assert screen._resource_range == "10m"
            assert "whole job" in _text(screen, "#resource-stats")  # still beside the window
            screen._apply_terminal_status("CANCELLED", None)
            assert screen._resource_range == "10m"
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_narrow_terminal_stacks_the_charts_and_fits_the_columns_it_can():
    c = _client(hardware=None)
    _download(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(80, 50)) as pilot:
            screen = await _open_resources(app, pilot)
            assert await _wait(pilot, lambda: screen.query_one("#resource-charts").styles.layout.name == "vertical")
            assert "GPU type unknown  ·  GPUs: 4 requested, 2 reporting" in _text(screen, "#resource-status")
            stats = _text(screen, "#resource-stats").splitlines()
            # Short labels (the chart titles carry the long ones), both scopes side by side.
            assert [" ".join(line.split()[:2]) for line in stats[2:]] == ["GPU busy", "GPU memory"]
            assert len(stats[2].split()) == 2 + 8
            # Every column that fits the table's width, measured from its cells.
            assert _columns(screen) == ["GPU", "Busy", "Busy avg 10m", "Mem", "Mem avg 10m", "Memory used"]
            assert all(_on_screen(app, box) for box in screen.query(".resource-chart-box"))
            for selector in ("#resource-chart", "#resource-mem-chart", "#resource-stats", "#resource-status"):
                widget = screen.query_one(selector, Static)
                assert all(len(line) <= widget.content_size.width for line in str(widget.render()).splitlines())
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_short_terminal_keeps_the_table_and_the_key_help_on_screen():
    c = _client()
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await _open_resources(app, pilot)
            await pilot.pause()
            table = screen.query_one("#resource-table", DataTable)
            keys = screen.query_one("#resource-keys", Static)
            # Header plus at least four GPU rows, and the key help under them.
            assert table.region.height >= 5 and table.region.y + table.region.height <= keys.region.y
            assert _on_screen(app, keys) and keys.region.y < app.size.height - 1  # above the footer
            # Too narrow for every window's name: the key help keeps its other keys.
            assert _text(screen, "#resource-keys") == (
                "w window   m busy/memory chart   r refresh   < > sub-job   Enter one GPU")
            busy, memory = screen.query(".resource-chart-box")
            assert busy.display and not memory.display  # one chart at a time
            await pilot.press("m")
            await pilot.pause()
            assert memory.display and not busy.display
            assert _titles(screen)[1] == f"training #0 · {MEMORY_LABEL} · last 10m · mean of 5 GPUs"
            # The full-height layout shows both charts and no switch.
            await pilot.resize_terminal(80, 50)
            assert await _wait(pilot, lambda: busy.display and memory.display)
            assert "m busy/memory chart" not in _text(screen, "#resource-keys")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_partial_first_load_is_shown_through_the_download(monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    c = _client()
    _download(c, minutes=180)
    seen = []
    real_accept = app_module.LogScreen._accept_cached_resource

    def accept(self, data):
        real_accept(self, data)
        sub_job = data.sub_jobs.get("7:training:0")
        if sub_job is not None and sub_job.partial:
            table = self.query_one("#resource-table", DataTable)
            seen.append({
                "status": _text(self, "#resource-status"),
                "stats": _text(self, "#resource-stats"),
                "titles": _titles(self),
                "avg": [str(table.get_row_at(0)[2]), str(table.get_row_at(0)[4])],
            })

    monkeypatch.setattr(app_module.LogScreen, "_accept_cached_resource", accept)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            # The whole job: the newest slices cover only the default window.
            app.screen._resource_range = "all"
            screen = await _open_resources(app, pilot)
            assert len(seen) == 1
            partial = seen[0]
            assert "loading older samples…" in partial["status"]
            assert partial["stats"].count("loading older samples…") == 2  # busy and memory
            assert partial["titles"] == ["training #0 · GPU busy · whole job · mean of 2 GPUs (loading…)",
                                         "training #0 · GPU memory · whole job · mean of 2 GPUs (loading…)"]
            assert partial["avg"] == ["…", "…"]  # the averages over the whole job aren't known yet
            # Then the complete view replaces it.
            assert "loading" not in _text(screen, "#resource-status") + _text(screen, "#resource-stats")
            assert _titles(screen) == ["training #0 · GPU busy · whole job · mean of 2 GPUs",
                                       "training #0 · GPU memory · whole job · mean of 2 GPUs"]
            row = screen.query_one("#resource-table", DataTable).get_row_at(0)
            assert "…" not in (str(row[2]), str(row[4]))
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_refresh_that_rebuilds_the_file_does_not_show_a_partial_view(monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    c = _client()
    calls = {"n": 0}

    def download(job_id, output_dir, *, resume=False, **_options):
        # Each call rebuilds the file the way download_metrics does when a chunk lands
        # out of order: a longer file moved into place.
        calls["n"] += 1
        path = Path(output_dir) / "7:training:0" / "gpu.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".gpu-metrics-x.tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in _rows(minutes=180 + calls["n"])))
        os.replace(temporary, path)
        path.with_name(".gpu.jsonl.manifest.json").write_text(json.dumps(
            {"version": 1, "names": [f"c{i}" for i in range(calls["n"])], "bytes": path.stat().st_size}))
        return []

    c.download_metrics.side_effect = download
    seen = []
    real_accept = app_module.LogScreen._accept_cached_resource

    def accept(self, data):
        sub_job = data.sub_jobs.get("7:training:0")
        seen.append((calls["n"], None if sub_job is None else sub_job.partial))
        real_accept(self, data)

    monkeypatch.setattr(app_module.LogScreen, "_accept_cached_resource", accept)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            assert calls["n"] == 1 and (1, True) in seen  # the first open shows the newest data first
            seen.clear()
            screen.action_refresh_sources()  # r: this download rebuilds the file
            assert await _wait(pilot, lambda: calls["n"] == 2 and not screen._resource_loading)
            assert seen == []  # no partial snapshot replaced the complete view
            assert "loading" not in _text(screen, "#resource-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_reopening_paints_what_was_read_while_another_refresh_holds_the_lock():
    c = _client()
    release = threading.Event()
    calls = {"n": 0}

    def download(job_id, output_dir, *, resume=False, **_options):
        calls["n"] += 1
        if calls["n"] == 1:
            _commit(output_dir, _rows(newest_age=300.0))
        elif calls["n"] == 2:
            _commit(output_dir, _rows(minutes=4, newest_age=0.0))
            release.wait(10)  # the first screen's refresh is stuck in its download
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            first = await _open_resources(app, pilot)
            first.action_refresh_sources()
            assert await _wait(pilot, lambda: calls["n"] == 2)
            app.push_screen(JobDetailScreen(c, "7", initial_tab="resource", poll_interval=0.01))
            assert await _wait(pilot, lambda: app.screen is not first and app.screen._current_source)
            second = app.screen
            assert await _wait(pilot, lambda: second.query_one("#resource-table", DataTable).row_count == 2)
            assert second._resource_loading and calls["n"] == 2  # painted while the other download runs
            release.set()
            assert await _wait(pilot, lambda: calls["n"] == 3 and not second._resource_loading)
            # And it ends on what that download committed, not on what it painted first.
            assert _newest(second) == _newest_on_disk(second)
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def _newest(screen):
    sub_job = screen._resource_data.sub_jobs.get("7:training:0")
    return None if sub_job is None else sub_job.latest_sample_ts


def _newest_on_disk(screen):
    return load_resource_data(screen._metrics_root).sub_jobs["7:training:0"].latest_sample_ts


@pytest.mark.parametrize("first_closes", [False, True], ids=["first-screen-stays", "first-screen-closes"])
def test_a_job_reopened_during_another_refresh_ends_on_the_newest_committed_data(first_closes):
    """The first screen's refresh commits newer rows while a second screen opens the job.

    Whether the first screen reads them (it stays open) or not (it closed during
    its download), the second must end on them, also when its own download
    brings nothing new, and before r is pressed.
    """
    c = _client()
    gate = threading.Event()
    calls = {"n": 0}

    def download(job_id, output_dir, *, resume=False, **_options):
        calls["n"] += 1
        if calls["n"] == 1:
            _commit(output_dir, _rows(newest_age=300.0))
        elif calls["n"] == 2:
            _commit(output_dir, _rows(minutes=4, newest_age=0.0))  # newer rows committed...
            gate.wait(10)  # ...while this download is still running
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            first = await _open_resources(app, pilot)
            before = _newest(first)
            first.action_refresh_sources()
            assert await _wait(pilot, lambda: calls["n"] == 2)
            if first_closes:
                app.pop_screen()
                await pilot.pause()
            second = JobDetailScreen(c, "7", initial_tab="resource", poll_interval=0.01)
            app.push_screen(second)
            assert await _wait(pilot, lambda: app.screen is second and _newest(second) is not None)
            assert _newest(second) == before  # painted what the shared loader had read
            gate.set()
            assert await _wait(pilot, lambda: calls["n"] == 3 and second._resource_loaded
                               and not second._resource_loading)
            assert _newest(second) == _newest_on_disk(second) > before
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        gate.set()


def test_a_refresh_that_fails_before_it_holds_the_lock_does_not_read_the_cache(monkeypatch):
    c = _client()
    _download(c)
    held = threading.Event()
    reads_while_held = []
    real_load = resource.ResourceLoader.load

    def load(self, **kwargs):
        if held.is_set():
            reads_while_held.append(kwargs)
        return real_load(self, **kwargs)

    monkeypatch.setattr(resource.ResourceLoader, "load", load)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            first = await _open_resources(app, pilot)
            lock = app_module._resource_lock(first._metrics_root)
            assert lock.acquire(timeout=5)  # another refresh holds the cache
            held.set()
            real_accept = app_module.LogScreen._accept_cached_resource

            def accept(self, data):
                if self is not first:
                    raise ValueError("painting failed")
                real_accept(self, data)

            monkeypatch.setattr(app_module.LogScreen, "_accept_cached_resource", accept)
            try:
                second = JobDetailScreen(c, "7", initial_tab="resource", poll_interval=0.01)
                app.push_screen(second)
                assert await _wait(pilot, lambda: second._resource_error is not None)
                assert reads_while_held == []  # the loader is used only under the lock
                assert "refresh failed: ValueError: painting failed" in _text(second, "#resource-status")
                assert second.query_one("#resource-table", DataTable).row_count == 2  # what it read last
            finally:
                held.clear()
                lock.release()
            await _settle(app, pilot)

    asyncio.run(run())


def test_without_telemetry_the_table_shows_no_column_labels():
    c = _client()
    c.download_metrics.return_value = []  # nothing staged yet

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            screen = app.screen
            await pilot.press("3")
            assert await _wait(pilot, lambda: screen._resource_loaded and not screen._resource_loading)
            assert "no GPU metrics for this sub-job" in _text(screen, "#resource-status")
            assert _columns(screen) == []
            _download(c)  # the first samples arrive
            screen.action_refresh_sources()
            assert await _wait(pilot, lambda: screen.query_one("#resource-table", DataTable).row_count == 2)
            assert _columns(screen)[:4] == ["GPU", "Busy", "Busy avg 10m", "Mem"]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_multiplexed_sub_job_says_where_its_gpu_metrics_are():
    c = _client(sampling_config={"n_gpus": 2, "multiplex_job_id": "other-job:sampling:0"})
    _download(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", sub_job_id="7:sampling:0", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            screen = app.screen
            await pilot.press("3")
            assert await _wait(pilot, lambda: screen._resource_loaded and not screen._resource_loading)
            status = _text(screen, "#resource-status")
            assert status.startswith("sampling #0  ·  H200  ·  GPUs: 0 requested (multiplexed)\n")
            assert "runs on the GPUs of other-job:sampling:0, whose metrics include it" in status
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_newest_sample_age_keeps_counting_without_a_refresh(monkeypatch):
    monkeypatch.setattr(app_module.LogScreen, "_RESOURCE_TICK_SECONDS", 0.05)
    c = _client(status="CANCELLED")  # a running job's hour-old sample would read as stalled
    _download(c)
    offset = {"seconds": 0.0}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=offset["seconds"])

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            downloads = c.download_metrics.call_count
            monkeypatch.setattr(app_module, "datetime", Clock)
            offset["seconds"] = 3600.0
            assert await _wait(pilot, lambda: "newest 1h 0" in _text(screen, "#resource-status"))
            assert c.download_metrics.call_count == downloads
            await _settle(app, pilot)

    asyncio.run(run())


def test_service_text_with_markup_characters_is_shown_literally():
    c = _client(hardware="[b]H200[/b]")
    job = c.get_job.return_value
    job["sub_jobs"][0]["sub_job_id"] = "7:training:[0]"
    _download(c, sub_job_id="7:training:[0]")

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            screen = await _open_resources(app, pilot)
            status = _text(screen, "#resource-status")
            assert status.startswith(
                "training #[0] (1 of 2 · < > switch)  ·  last 10m  ·  [b]H200[/b]  ·  GPUs: 4 requested, 2 reporting")
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_header_names_the_sub_job_by_type_and_shortens_its_place_to_fit():
    # Named by type and index, never by its id: a long job id (on the summary line above) costs nothing.
    long_id = "JOB_ID-0000-0000-0000-000000000000"
    c = _client()
    job = c.get_job.return_value
    job["job_id"] = long_id
    for sub_job in job["sub_jobs"]:
        sub_job["sub_job_id"] = sub_job["sub_job_id"].replace("7:", f"{long_id}:")
    _download(c, sub_job_id=f"{long_id}:training:0")

    async def run():
        app = CortexTrainingLogTUI(c, long_id, poll_interval=0.01)
        async with app.run_test(size=(100, 40)) as pilot:
            screen = await _open_resources(app, pilot)

            def first():
                return _text(screen, "#resource-status").splitlines()[0]

            assert first() == "training #0 (1 of 2 · < > switch)  ·  last 10m  ·  H200  ·  GPUs: 4 requested, 2 reporting"
            await pilot.resize_terminal(90, 40)  # the place goes short
            assert await _wait(pilot, lambda: first() == (
                "training #0 (1/2)  ·  last 10m  ·  H200  ·  GPUs: 4 requested, 2 reporting"))
            await pilot.resize_terminal(70, 40)  # then the window (the stats name it)
            assert await _wait(pilot, lambda: first() == "training #0 (1/2)  ·  H200  ·  GPUs: 4 requested, 2 reporting")
            await _settle(app, pilot)

    asyncio.run(run())
