# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Resources tab in the running app: first paint, freshness, waiting sub-jobs, panel and layout."""

import asyncio
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.resource import GPU  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from tests.test_resources_tab import _bar_rows  # noqa: E402
from tests.test_resources_tab import _client  # noqa: E402
from tests.test_resources_tab import _rows  # noqa: E402
from tests.test_resources_tab import _units  # noqa: E402

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
            assert await _wait(pilot, lambda: len(_units(screen)) == 3)  # ALL and the two GPUs
            status, stats = _text(screen, "#resource-status"), _text(screen, "#resource-stats")
            assert "loading older samples…" in status
            busy = next(line for line in stats.splitlines() if line.startswith("GPU util %"))
            assert busy.split()[3:7] == ["85%", "85%", "85%", "85%"]  # last 10m: min avg max latest
            assert busy.count("loading older samples…") == 1  # the whole job's columns
            assert screen._resource_loading
            gate.set()
            assert await _wait(pilot, lambda: not screen._resource_loading)
            stats = _text(screen, "#resource-stats")
            assert "loading" not in _text(screen, "#resource-status") + stats
            assert set(screen._resource_data.kinds) == {TRAINING, "7:sampling:0"}
            await _settle(app, pilot)

    asyncio.run(run())
    # The sub-job on screen first, and its download starts with the default window plus slack.
    assert calls == [([TRAINING], 600 + 180)]


# ─── Every chart has the same height: 4 rows of bars, 2 on a short terminal ──

from tests.test_resources_tab import _download  # noqa: E402
from tests.test_resources_tab import _panel  # noqa: E402


async def _open(app, pilot):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
    await pilot.press("3")
    assert await _wait(pilot, lambda: app.screen._resource_loaded and not app.screen._resource_loading
                       and len(_units(app.screen)) > 1)
    await pilot.pause()
    return app.screen


def _charts(screen):
    """Each chart's rows of bars, in order (two to a row read left, then right)."""
    found = []
    lines = _text(screen, "#resource-charts").splitlines()
    for index, line in enumerate(lines):
        for column, char in enumerate(line):
            if char == "┤" and (index == 0 or lines[index - 1][column:column + 1] not in ("┤", "│")):
                rows = 0  # the axis column, from the top label down to the axis
                while index + rows < len(lines) and lines[index + rows][column:column + 1] in ("┤", "│"):
                    rows += 1
                found.append(rows)
    return found


@pytest.mark.parametrize(("size", "bars"), [((150, 44), 4), ((140, 42), 4), ((100, 30), 4), ((100, 29), 2),
                                           ((80, 24), 2)])
def test_every_chart_has_four_rows_of_bars_or_two_on_a_short_terminal(size, bars):
    c = _client(n_gpus=8)
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot)
            # Even, so the middle label's row tops out at half the scale; the same for every chart.
            assert _charts(screen) == [bars] * 5, _charts(screen)
            assert len(_bar_rows(screen)) == bars
            await _settle(app, pilot)

    asyncio.run(run())


# ─── The GPU shown and the pane's scroll survive w and refreshes ─────────────


def test_the_gpu_shown_and_the_pane_scroll_survive_a_new_window_and_a_refresh():
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
            panel = screen.query_one("#resource-list")
            pane = screen.query_one("#resource-pane")
            for _ in range(27):  # ALL, then GPU 0 to GPU 26
                await pilot.press("down")
            await pilot.pause()
            assert screen._resource_unit == 26 and panel.scroll_y > 0  # the panel scrolled to it
            assert "last 10m · GPU 26" in _text(screen, "#resource-stats")
            await pilot.press("pagedown")
            assert await _wait(pilot, lambda: pane.scroll_y > 0)
            scrolled = pane.scroll_y
            await pilot.press("w")
            await pilot.pause()
            assert screen._resource_unit == 26 and "last 30m · GPU 26" in _text(screen, "#resource-stats")
            assert pane.scroll_y == scrolled
            before = calls["n"]
            await pilot.press("r")
            assert await _wait(pilot, lambda: calls["n"] > before and not screen._resource_loading)
            await pilot.pause()
            assert screen._resource_unit == 26 and pane.scroll_y == scrolled
            assert panel.highlighted_child is screen._resource_rows[_panel(screen).index(
                next(text for text in _panel(screen) if text.startswith("GPU 26 ")))][0]
            await _settle(app, pilot)

    asyncio.run(run())


# ─── A gauge measured in some rows charts; one null in every row doesn't ──────


def test_a_measured_profiling_gauge_gets_a_chart_and_an_unmeasured_one_none():
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
            titles = " ".join(_titles(screen))
            stats = _text(screen, "#resource-stats")
            assert "tensor active %" in titles and "DRAM active %" in titles and "SM active" not in titles
            assert "SM active" not in stats and "SM" not in " ".join(_panel(screen))
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
            gpus = [" ".join(unit.split()[:-1]) for unit in _units(screen)[1:]]  # without the newest value
            if status == "RUNNING":
                assert "GPUs: 4 requested, 0 reporting (2 seen)" in text
                assert "no new GPU samples for 41m 5" in text and "; the windows end at " in text
                assert "[$warning]no new GPU samples" in _status_markup(screen)
                assert gpus == ["GPU 0 (stale)", "GPU 1 (stale)"]
            else:  # a finished job's newest sample is simply its last
                assert "GPUs: 4 requested, 2 reporting" in text and "newest 41m 5" in text
                assert "no new GPU samples" not in text and gpus == ["GPU 0", "GPU 1"]
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
            assert _units(screen) == []  # nothing listed under the views
            assert not any("(none)" in text for text in _panel(screen))  # its samples are still to come
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
        async with app.run_test(size=(160, 42)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("r")
            assert await _wait(pilot, lambda: calls["n"] == 2 and not screen._resource_loading)
            await pilot.pause()
            status = screen.query_one("#resource-status", Static)
            second = _text(screen, "#resource-status").splitlines()[1]
            assert "cached · refresh failed: ConnectionError: stage LIST failed: conn" in second
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
            assert _units(screen) == []
            await _settle(app, pilot)

    asyncio.run(run())


# ─── R1-21 follow-up: a download that ends without the older history ─────────

import dataclasses  # noqa: E402
import re  # noqa: E402

from rich.cells import cell_len  # noqa: E402

from tests.test_resources_tab import _titles  # noqa: E402


def _stat_line(screen, label="GPU util %"):
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
        assert "older samples failed to load: ConnectionError: simulated stage GET" in header
        assert "[$warning]older samples failed to load: ConnectionError" in _status_markup(screen)
        assert "loading" not in header + _text(screen, "#resource-stats")
        busy = _stat_line(screen)
        assert busy.split()[3:7] == ["85%", "85%", "85%", "85%"]  # the window: every sample is in
        assert busy.count("older samples not loaded") == 1  # the whole job: not

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 42)) as pilot:  # wide enough for the whole state
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: calls and not screen._resource_loading)
            await pilot.pause()
            assert screen._resource_kind_data().partial
            assert_failed(screen)
            # A window the newest chunks don't cover says so in its titles and in every metric's stats.
            screen.action_resource_range("30m")
            await pilot.pause()
            # After each title, or on a line of its own under each.
            assert _text(screen, "#resource-charts").count("(older samples not loaded)") == 5
            assert _stat_line(screen).count("older samples not loaded") == 2
            assert _text(screen, "#resource-stats").count("older samples not loaded") == 2 * 5
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
# The header is as wide as the pane beside the panel ("○ System Resources (none)" here): 32 cells less.
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (125, 30), (127, 30), (150, 34), (170, 42)])
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
            charts = screen.query_one("#resource-charts", Static)
            shown = {}
            data = screen._resource_data
            for state in ("idle", "refreshing", "failed", "idle again", "older failed"):
                screen._resource_loading = state == "refreshing"
                screen._resource_error = _LONG_ERROR if "failed" in state else None
                if state == "older failed":  # the newest chunks are in, a GET of an older one failed
                    partial = dataclasses.replace(data.kinds[TRAINING][GPU], partial=True)
                    screen._resource_data = dataclasses.replace(data, kinds={TRAINING: {GPU: partial}})
                screen._render_resource()
                await pilot.pause()
                await pilot.pause()
                header = _text(screen, "#resource-status").split("\n")
                assert len(header) == 2, (state, header)
                assert all(cell_len(line) <= status.content_size.width for line in header), (state, header)
                assert status.region.height == 2, state
                shown[state] = header, charts.region.y, len(_bar_rows(screen))
            # Nothing below the header moves.
            assert len({(y, rows) for _header, y, rows in shown.values()}) == 1, shown
            idle, refreshing, failed = (shown[state][0][1] for state in ("idle", "refreshing", "failed"))
            assert _same_parts(shown["idle again"][0][1]) == _same_parts(idle)
            # A refresh adds its state and changes nothing else on the line.
            state = "cached · refreshing…" if "cached · refreshing…" in refreshing else "refreshing…"
            assert _same_parts(refreshing.replace(f"  ·  {state}", "", 1)) == _same_parts(idle)
            # A failure shows the start of the error, at least 40 cells of it once there is room.
            width = status.content_size.width
            # Cut (the skipped rows may follow); beside the stale warning in a 48-cell header, into the state itself.
            assert ("refresh failed: Con" if width > 48 or not stale else "refresh fail") in failed and "…" in failed
            if width >= 93:
                assert "refresh failed: " + _LONG_ERROR[:39] in failed
            if width >= 135 and not stale:
                assert "values change about every 45 s" in idle and "3 rows skipped" in idle
            # The same for older samples that failed to load: their state, cut (at 60 columns
            # beside the stale warning, into the state itself), and the start of the error.
            older = shown["older failed"][0][1]
            assert ("older samples fail" if width > 48 or not stale else "older sample") in older and "…" in older
            if width >= 93:
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
            sub_job = screen._resource_kind_data()
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


# ─── R1-96 (6): on a terminal too short for the tab, the keys scroll the pane ─


def test_on_a_short_terminal_the_scroll_keys_move_the_pane_and_the_arrows_the_panel():
    c = _client(n_gpus=8)
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(60, 16)) as pilot:
            screen = await _open(app, pilot)
            pane = screen.query_one("#resource-pane")
            panel = screen.query_one("#resource-list")

            async def press(key):
                await pilot.press(key)
                await pilot.pause()
                await pilot.pause()

            assert screen.focused is panel and pane.max_scroll_y > 0
            assert pane.scroll_y == 0  # the pane opens at its top
            await press("down")
            assert screen._resource_unit == 0 and pane.scroll_y == 0
            await press("pagedown")
            assert pane.scroll_y > 0 and screen._resource_unit == 0  # the pane scrolls; the pick stays
            await press("end")
            assert pane.scroll_y == pane.max_scroll_y and screen._resource_unit == 0
            await press("pageup")
            assert 0 < pane.scroll_y < pane.max_scroll_y
            await press("home")
            assert pane.scroll_y == 0 and screen.focused is panel
            await press("up")
            assert screen._resource_unit is None  # back on ALL
            # A refresh at the end of the pane leaves it there, with the pick where it was.
            await press("down")
            await press("end")
            await press("r")
            assert await _wait(pilot, lambda: not screen._resource_loading)
            await pilot.pause()
            assert pane.scroll_y == pane.max_scroll_y and screen._resource_unit == 0
            await _settle(app, pilot)

    asyncio.run(run())


# ─── The charts of a row are the same width ──────────────────────────────────


def _axes(screen):
    """The length of every chart's axis, in order."""
    return [len(run) for line in _text(screen, "#resource-charts").splitlines() if "└" in line
            for run in re.findall(r"└(─+)", line)]


@pytest.mark.parametrize("size", [(140, 42), (141, 42), (125, 34)])  # the pane at least 90 cells wide
def test_the_charts_side_by_side_are_drawn_to_the_same_width(size):
    c = _client(n_gpus=8)
    _download(c, gpus=8)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot)
            lines = [line for line in _text(screen, "#resource-charts").splitlines() if "└" in line]
            assert all(line.count("└") == 2 for line in lines[:2])  # two to a row
            # The same scale's charts (the percentages) have the same axis: one column width for all.
            percent = [len(run) for line in lines for run in re.findall(r"  0% └(─+)", line)]
            assert len(set(percent)) == 1 and percent[0] >= (size[0] - 32) // 2 - 10
            width = screen.query_one("#resource-charts", Static).content_size.width
            assert all(len(line) <= width for line in _text(screen, "#resource-charts").splitlines())
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_panel_keeps_the_focus_when_the_units_arrive_after_a_message():
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
            panel = screen.query_one("#resource-list")
            assert await _wait(pilot, lambda: "loading GPU metrics…" in _text(screen, "#resource-status"))
            assert screen.focused is panel and _units(screen) == []
            gate.set()
            assert await _wait(pilot, lambda: len(_units(screen)) == 9)
            await pilot.pause()
            assert screen.focused is panel and screen._resource_unit is None  # on ALL, not on a view's entry
            await pilot.press("down")
            assert screen._resource_unit == 0 and "last 10m · GPU 0" in _text(screen, "#resource-stats")
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_charts_shown_when_rows_first_arrive_are_drawn_to_the_pane_once_it_is_measured():
    c = _client(n_gpus=8)
    gate = threading.Event()

    def download(job_id, output_dir, **_options):
        assert gate.wait(30)
        _write(output_dir, _rows(gpus=8))
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: "loading GPU metrics…" in _text(screen, "#resource-status"))
            gate.set()  # the units arrive, and the panel widens to their labels
            assert await _wait(pilot, lambda: len(_units(screen)) == 9 and not screen._resource_loading)
            for _ in range(3):
                await pilot.pause()
            width = screen.query_one("#resource-charts", Static).content_size.width
            axes = [line for line in _text(screen, "#resource-charts").splitlines() if "└" in line]
            # One chart to a row (the pane is under 90 cells), each drawn to the pane's width.
            assert axes and all(line.count("└") == 1 and width - 1 <= len(line) <= width for line in axes)
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_narrow_pane_keeps_every_metric_while_the_older_samples_are_missing():
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

    def rows_of(screen):
        return [line for line in _text(screen, "#resource-stats").splitlines()[1:] if line.strip()]

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
            await pilot.press("3")
            screen = app.screen
            assert await _wait(pilot, lambda: calls and not screen._resource_loading)
            screen.action_resource_range("30m")
            await pilot.pause()
            missing = rows_of(screen)
            # A row per metric and scope (narrow): each says its samples aren't loaded, none is dropped.
            assert len(missing) == 2 * 5 and all(line.endswith("older samples not loaded") for line in missing)
            await pilot.press("r")
            assert await _wait(pilot, lambda: len(calls) == 2 and not screen._resource_loading)
            await pilot.pause()
            loaded = rows_of(screen)
            assert len(loaded) == len(missing) and not any("older samples" in line for line in loaded)
            assert loaded[0].split()[-1].endswith("%")
            await _settle(app, pilot)

    asyncio.run(run())
