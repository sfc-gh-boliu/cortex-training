# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resources computations: averages, intervals, value refresh, extremes past the raw retention, allocation."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cortex_training.tui.resource as resource
from cortex_training.tui.resource import ResourceLoader
from cortex_training.tui.resource import load_resource_data

SUB = "job-1:training:0"
START = datetime(2026, 1, 15, tzinfo=timezone.utc)
GIB = 1024**3


def _row(t: float, gpu: int, busy, *, interval=5.0, used=50 * GIB, free=50 * GIB, sub=SUB, **extra) -> dict:
    row = {
        "record_version": 2, "sub_job_id": sub, "global_gpu_index": gpu, "worker_num": gpu // 4,
        "local_gpu_index": gpu % 4, "sample_ts": (START + timedelta(seconds=t)).isoformat().replace("+00:00", "Z"),
        "gpu_util_pct": busy, "fb_used_bytes": float(used), "fb_free_bytes": float(free),
        "sample_interval_s": interval,
    }
    row.update(extra)
    return row


def _write(root: Path, rows, sub=SUB) -> Path:
    path = root / sub / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["c.gz"], "bytes": path.stat().st_size}))
    return path


def _load(root: Path, rows, sub=SUB):
    _write(root, rows, sub)
    return load_resource_data(root).sub_jobs[sub]


# ─── R1-65: the average weights GPU-time ─────────────────────────────────────


def test_a_gpu_with_one_sample_in_the_window_counts_for_that_sample_not_a_whole_gpu(tmp_path):
    newest = 3600
    rows = []
    for t in range(0, newest + 1, 5):
        for gpu in range(6):
            rows.append(_row(t, gpu, 90.0))
        if t <= newest - 600:  # GPU 6 stops ten minutes before the newest row
            rows.append(_row(t, 6, 10.0))
    sub_job = _load(tmp_path, rows)
    window = sub_job.views["10m"]
    # GPU 6's one sample at the window's edge weighs one sample, not a seventh of the window.
    assert window.busy.avg == pytest.approx((6 * 121 * 90 + 10) / (6 * 121 + 1))
    assert window.busy.min <= window.busy.avg <= window.busy.max
    assert window.busy.latest == pytest.approx(90.0)
    # The per-GPU table keeps each GPU's own mean.
    assert [round(row.window_avg_util_pct) for row in window.rows] == [90] * 6 + [10]


# ─── R1-69: rows without a usable sample_interval_s ──────────────────────────


@pytest.mark.parametrize("interval", ["absent", 0, -5, "five"])
def test_rows_without_a_usable_interval_are_charted_with_their_own_spacing(tmp_path, interval):
    rows = []
    for t in range(0, 20 * 60 + 1, 5):
        for gpu in range(4):
            row = _row(t, gpu, 60.0 + 10 * gpu if t >= 15 * 60 else 20.0)
            if interval == "absent":
                del row["sample_interval_s"]
            else:
                row["sample_interval_s"] = interval
            rows.append(row)
    sub_job = _load(tmp_path, rows)
    window = sub_job.views["10m"]
    assert sub_job.sample_interval_s == pytest.approx(5.0)
    assert any(value is not None for value in window.chart)  # charted, not "no GPU samples"
    assert any(value is not None for value in window.mem_chart)
    # Min, avg and max come from the samples, not from the newest capture alone.
    assert window.busy.min == pytest.approx(20.0)
    assert window.busy.max == pytest.approx(75.0)
    assert 20.0 < window.busy.avg < 75.0
    # GPU 1 in the window: 60 samples at 20% (10:00 to 15:00), then 61 at 70%.
    assert window.rows[1].window_avg_util_pct == pytest.approx((60 * 20 + 61 * 70) / 121)


# ─── R1-67: values that change less often than the scrape ────────────────────


def _dcgm_rows(change_every: int, *, gpus=4, minutes=20, idle_gpu=None):
    rows = []
    for t in range(0, minutes * 60 + 1, 5):
        epoch = t // change_every
        for gpu in range(gpus):
            busy = 0.0 if gpu == idle_gpu else float(30 + (epoch * 7 + gpu) % 60)
            rows.append(_row(t, gpu, busy))
    return rows


def test_the_header_says_how_often_the_values_change_when_it_is_much_slower_than_the_scrape(tmp_path):
    sub_job = _load(tmp_path, _dcgm_rows(45, idle_gpu=3))
    assert sub_job.value_interval_s == pytest.approx(45.0)
    now = sub_job.latest_sample_ts + timedelta(seconds=29)
    text = resource.sampling_text(sub_job, now)
    assert text.startswith("scraped every 5 s (values change about every 45 s)  ·  newest 29 s ago (")


def test_values_that_change_every_scrape_or_two_need_no_note(tmp_path):
    for every in (5, 10):
        sub_job = _load(tmp_path / str(every), _dcgm_rows(every))
        now = sub_job.latest_sample_ts + timedelta(seconds=3)
        assert "values change" not in resource.sampling_text(sub_job, now), every
    idle = _load(tmp_path / "idle", [_row(t, 0, 0.0) for t in range(0, 600, 5)])
    assert idle.value_interval_s is None  # a value that never changes says nothing about the refresh


# ─── R1-63: the Resources header reads GPUs as the list and the preview do ───

from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.resource import allocation_text  # noqa: E402
from cortex_training.tui.resource import gpu_allocation  # noqa: E402


def test_an_unknown_jobs_sub_jobs_are_found_by_the_ids_the_preview_derives():
    job = {
        "job_id": "J", "status": "UNKNOWN", "reason": "Job status unavailable: Control Plane is unavailable",
        "hardware": "H200",
        "sub_jobs": [{"job_type": "TRAINING", "training_config": {"n_gpus": 8.0}},
                     {"job_type": "SAMPLING", "inference_config": {"n_gpus": 4.0}}],
    }
    assert [sj["sub_job_id"] for sj in views.sub_jobs(job)] == ["J:training:0", "J:sampling:0"]
    assert allocation_text(gpu_allocation(job, "J:training:0")) == "H200  ·  GPUs: 8 requested"
    assert allocation_text(gpu_allocation(job, "J:sampling:0")) == "H200  ·  GPUs: 4 requested"
    assert views.gpu_request_text(job) == "12 × H200 requested  (8 training #0 + 4 sampling #0)"


@pytest.mark.parametrize("sub_job", [
    {"sub_job_id": "J:training:0", "job_type": "TRAINING", "training_config": {"n_gpus": 8.0}},
    {"sub_job_id": "J:training:0", "job_type": "TRAINING", "training_config": {"n_gpus": "2"}},
    {"sub_job_id": "J:training:0", "job_type": "TRAINING", "training_config": {}},
    {"sub_job_id": "J:log_prob:0", "job_type": "LOG_PROBABILITY", "inference_config": {"n_gpus": 2}},
    {"sub_job_id": "J:log_prob:0", "inference_config": {"n_gpus": 2}},  # no type: read as the preview reads it
    {"sub_job_id": "J:sampling:0", "job_type": "SAMPLING",
     "inference_config": {"n_gpus": 1, "multiplex_job_id": "OTHER:training:0"}},
    {"sub_job_id": "J:sampling:0", "job_type": "SAMPLING", "inference_config": {"n_gpus": 4}},
])
def test_requested_gpus_and_sharing_agree_with_the_preview(sub_job):
    job = {"job_id": "J", "status": "RUNNING", "hardware": "B200", "sub_jobs": [sub_job]}
    allocation = gpu_allocation(job, sub_job["sub_job_id"])
    preview = views.requested_gpus(sub_job)
    assert allocation.requested == (preview if preview or allocation.log_prob or allocation.shared_with else None)
    assert (allocation.shared_with or "") == views.multiplex_target(sub_job)
    assert allocation.log_prob == (views.sub_job_kind(sub_job) == "log_prob")


def test_a_count_no_int_holds_reads_as_unknown_instead_of_raising():
    for count in (float("inf"), float("nan"), 1e999):
        job = {"job_id": "J", "status": "RUNNING", "sub_jobs": [
            {"sub_job_id": "J:training:0", "job_type": "TRAINING", "training_config": {"n_gpus": count}}]}
        assert allocation_text(gpu_allocation(job, "J:training:0")) == "GPU type unknown  ·  GPUs: requested count unknown"


def test_the_gpu_type_is_shown_on_one_printable_line():
    job = {"job_id": "J", "hardware": "H200\x1b]52;c;SU5KRUNU\x07\n  x", "sub_jobs": []}
    hardware = gpu_allocation(job, "J:training:0").hardware
    assert "\x1b" not in hardware and "\x07" not in hardware and "\n" not in hardware


# ─── R1-64: whole-job extremes past the raw retention ────────────────────────


def _lines(rows) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _commit(path: Path, rows, names) -> None:
    with path.open("ab") as handle:
        handle.write(_lines(rows))
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": names, "bytes": path.stat().st_size}))


def _workload(t: float, gpu: int) -> float:
    """77-90 % busy, with a dip to 1 % in the job's 5th minute."""
    if 240 <= t < 300:
        return 1.0
    return 77.0 + (t // 5 * 7 + gpu * 3) % 14


def test_the_whole_job_min_does_not_drift_as_the_job_outgrows_the_raw_retention(tmp_path):
    hours = 7
    path = _write(tmp_path, [_row(t, gpu, _workload(t, gpu)) for t in range(0, hours * 3600, 5) for gpu in range(4)])
    loader = ResourceLoader(tmp_path)
    first = loader.load().sub_jobs[SUB].views["all"].busy
    assert (first.min, first.min_bound) == (1.0, False)
    names = ["c0"]
    t = hours * 3600
    while t < 8.5 * 3600:  # the dip leaves the raw samples along the way
        names.append(f"c{len(names)}")
        _commit(path, [_row(s, gpu, _workload(s, gpu)) for s in range(t, t + 1800, 5) for gpu in range(4)], names)
        t += 1800
        busy = loader.load().sub_jobs[SUB].views["all"].busy
        assert (busy.min, busy.max, busy.min_bound, busy.max_bound) == (1.0, first.max, False, False), t
    assert loader._files[SUB].series.raw_start > 300  # the dip is only in the folded history now
    cold = load_resource_data(tmp_path).sub_jobs[SUB].views["all"].busy
    assert (cold.min, cold.max, cold.min_bound, cold.max_bound) == (1.0, first.max, False, False)


def test_a_cold_read_of_a_day_long_job_gives_exact_extremes(tmp_path):
    spike = range(12 * 3600, 12 * 3600 + 15)  # 15 seconds at 95.5 %, half a day before the newest row
    rows = [_row(t, gpu, 95.5 if t in spike else _workload(t, gpu)) for t in range(0, 24 * 3600, 5)
            for gpu in range(8)]
    sub_job = _load(tmp_path, rows)
    whole = sub_job.views["all"]
    by_time: dict = {}
    for row in rows:
        by_time.setdefault(row["sample_ts"], []).append(row["gpu_util_pct"])
    averages = [sum(values) / len(values) for values in by_time.values()]
    assert whole.busy.min == pytest.approx(min(averages)) == 1.0
    assert whole.busy.max == pytest.approx(max(averages)) == 95.5
    assert not (whole.busy.min_bound or whole.busy.max_bound)
    assert len(whole.chart) <= resource._MAX_BINS


def test_a_row_that_lands_after_its_interval_was_folded_keeps_the_extremes_honest(tmp_path):
    path = _write(tmp_path, [_row(t, 0, 50.0) for t in range(0, 9 * 3600, 5)])
    loader = ResourceLoader(tmp_path)
    assert loader.load().sub_jobs[SUB].views["all"].busy.min == 50.0
    # A slow worker's chunk from the job's first hour: its rows never reach the raw samples.
    _commit(path, [_row(t, 1, 10.0) for t in range(600, 900, 5)], ["c.gz", "c1"])
    busy = loader.load().sub_jobs[SUB].views["all"].busy
    assert busy.min < 50.0 and busy.min_bound  # known only as its bin's average: a bound
    text = resource.stats_text(loader.last.sub_jobs[SUB].views["10m"], loader.last.sub_jobs[SUB].views["all"],
                               width=120)
    assert "≤" in text


# ─── R1-73 / R1-74: taller charts, and "mean of N GPUs" ───────────────────────

from cortex_training.tui.resource import RangeView  # noqa: E402
from cortex_training.tui.resource import chart_title  # noqa: E402
from cortex_training.tui.resource import render_chart  # noqa: E402
from cortex_training.tui.resource import stats_text  # noqa: E402


def _view(values, *, gpus=8, name="10m"):
    end = START + timedelta(minutes=10)
    return RangeView(name=name, start=START, end=end, chart=tuple(values), rows=(), mem_chart=tuple(values),
                     gpus=gpus)


def test_a_taller_chart_tells_readings_inside_one_glyph_apart():
    view = _view([77.0] * 30 + [87.0] * 30)
    one = render_chart(view, 66).splitlines()
    assert set(one[0][6:]) == {"▇"}  # one row: 77% and 87% draw the same glyph
    four = render_chart(view, 66, height=4).splitlines()
    assert [line[:6] for line in four] == ["100% ┤", "     │", " 50% ┤", "     │", "  0% └", "      "]
    top = four[0][6:]
    assert top[:30] == "▁" * 30 and top[30:] == "▄" * 30  # 25 and 28 eighths of four rows
    assert all(set(line[6:]) == {"█"} for line in four[1:4])


def test_a_one_row_chart_is_unchanged_and_a_short_one_keeps_its_time_label():
    view = _view([50.0, None, 100.0])
    assert render_chart(view, 20) == render_chart(view, 20, height=1)
    assert render_chart(view, 20).splitlines()[0] == "100% ┤" + " " * 11 + "▄ █"
    # One capture: a single column, and its time ends under it instead of losing all but one digit.
    single = RangeView(name="10m", start=START, end=START, chart=(55.0,), rows=(), mem_chart=(55.0,), gpus=4)
    labels = render_chart(single, 40).splitlines()[2]
    assert labels.endswith(START.astimezone().strftime("%H:%M")) and len(labels) == 40


def test_the_titles_and_the_stats_say_they_average_across_the_gpus():
    view, whole = _view([80.0] * 5), _view([80.0] * 5, name="all", gpus=7)
    assert chart_title(view, resource.BUSY_LABEL) == "GPU busy (kernel-active %) · last 10m · mean of 8 GPUs"
    assert chart_title(view, resource.BUSY_LABEL, width=40) == "GPU busy (kernel-active %) · last 10m"
    assert chart_title(_view([80.0], gpus=1), resource.BUSY_LABEL) == "GPU busy (kernel-active %) · last 10m"
    wide = stats_text(view, whole, width=140).splitlines()[0]
    assert "last 10m · mean of 8 GPUs" in wide and "whole job · mean of 7 GPUs" in wide
    narrow = stats_text(view, whole, width=40).splitlines()[0]
    assert narrow.startswith("mean of 8 GPUs") and narrow.split()[-4:] == ["min", "avg", "max", "latest"]


def test_a_tiny_reported_interval_cannot_make_a_fold_huge(tmp_path):
    import time as clock

    rows = [_row(t, 0, 50.0) for t in range(0, 9 * 3600, 5)]
    rows[10]["sample_interval_s"] = 1e-9  # a malformed row: the smallest interval seen
    started = clock.perf_counter()
    whole = _load(tmp_path, rows).views["all"]
    assert clock.perf_counter() - started < 10
    assert whole.busy.min == pytest.approx(50.0) and whole.busy.max == pytest.approx(50.0)
