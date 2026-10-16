# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the Resources tab derives from GPU telemetry: memory %, stats, sampling,
freshness, reporting GPUs, windows, GPU type and allocation, and its text."""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cortex_training.tui.resource import DEFAULT_RANGE
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import GpuAllocation
from cortex_training.tui.resource import allocation_text
from cortex_training.tui.resource import format_age
from cortex_training.tui.resource import format_interval
from cortex_training.tui.resource import format_value
from cortex_training.tui.resource import gpu_allocation
from cortex_training.tui.resource import load_resource_data
from cortex_training.tui.resource import no_metrics_text
from cortex_training.tui.resource import sampling_text
from tests.resource_legacy import gpu_jobs
from tests.resource_legacy import render_chart
from tests.resource_legacy import stats_text

BUSY_LABEL, MEMORY_LABEL = "GPU util %", "GPU memory used %"  # the stats rows of busy and memory

SUB = "job:training:0"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _row(seconds, gpu, util, *, used=40.0, free=60.0, interval=5.0, **extra):
    return {
        "record_version": 2,
        "sub_job_id": SUB,
        "sample_ts": (START + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z"),
        "worker_num": 0,
        "local_gpu_index": gpu,
        "global_gpu_index": gpu,
        "gpu_util_pct": util,
        "fb_used_bytes": used,
        "fb_free_bytes": free,
        "sample_interval_s": interval,
        **extra,
    }


def _load(root: Path, rows):
    path = root / SUB / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["c0"], "bytes": path.stat().st_size})
    )
    return gpu_jobs(load_resource_data(root))[SUB]


def test_memory_percent_is_used_over_used_plus_free_for_each_gpu(tmp_path):
    rows = []
    for t in range(0, 300, 5):
        rows.append(_row(t, 0, 50.0, used=20 * GIB, free=80 * GIB))  # 20%
        rows.append(_row(t, 1, 50.0, used=60 * GIB, free=20 * GIB))  # 75% of a smaller GPU
    view = _load(tmp_path, rows).views["10m"]
    assert [row.current_mem_pct for row in view.rows] == [pytest.approx(20.0), pytest.approx(75.0)]
    assert [row.window_avg_mem_pct for row in view.rows] == [pytest.approx(20.0), pytest.approx(75.0)]
    # Each bucket averages the GPUs' percentages (not total used over total size).
    assert set(round(value, 6) for value in view.mem_chart) == {47.5}
    assert view.memory.avg == pytest.approx(47.5) and view.memory.latest == pytest.approx(47.5)
    # Compute stays separate from memory.
    assert set(view.chart) == {50.0} and view.busy.avg == pytest.approx(50.0)


def test_min_avg_max_latest_for_the_window_and_the_whole_job(tmp_path):
    rows = []
    newest = 40 * 60 - 5
    for t in range(0, newest + 1, 5):
        hot = t >= newest - 600  # the last 10 minutes (the window is inclusive) run hotter
        busy = 90.0 if hot else 30.0
        rows.append(_row(t, 0, busy, used=(70 if hot else 50) * GIB, free=(30 if hot else 50) * GIB))
        rows.append(_row(t, 1, busy - 20, used=50 * GIB, free=50 * GIB))
    sub_job = _load(tmp_path, rows)
    window, whole = sub_job.views["10m"], sub_job.views["all"]
    assert (window.busy.min, window.busy.avg, window.busy.max, window.busy.latest) == pytest.approx((80, 80, 80, 80))
    assert (whole.busy.min, whole.busy.max, whole.busy.latest) == pytest.approx((20, 80, 80))
    per_gpu = [sum(r["gpu_util_pct"] for r in rows if r["global_gpu_index"] == g) / (len(rows) / 2) for g in (0, 1)]
    assert whole.busy.avg == pytest.approx(sum(per_gpu) / 2)
    assert (window.memory.min, window.memory.max) == pytest.approx((60, 60))
    assert (whole.memory.min, whole.memory.max) == pytest.approx((50, 60))
    # Nested scopes nest their extremes.
    lows = [sub_job.views[name].busy.min for name in RANGES]
    highs = [sub_job.views[name].busy.max for name in RANGES]
    assert lows == sorted(lows, reverse=True) and highs == sorted(highs)


def test_profiling_fields_show_a_dash_when_unmeasured_and_zero_only_when_zero(tmp_path):
    rows = [_row(0, 0, 10.0, sm_active_pct=None, tensor_active_pct=0.0), _row(0, 1, 10.0, dram_active_pct=37.5)]
    gpu0, gpu1 = _load(tmp_path, rows).views["10m"].rows
    assert (gpu0.sm_active_pct, gpu0.tensor_active_pct, gpu0.dram_active_pct) == (None, 0.0, None)
    assert [format_value(v, "%") for v in (gpu0.sm_active_pct, gpu0.tensor_active_pct, gpu0.dram_active_pct)] == [
        "—", "0%", "—"]
    assert format_value(gpu1.dram_active_pct, "%") == "38%" and format_value(gpu1.sm_active_pct, "%") == "—"


def test_the_sampling_interval_comes_from_the_data_and_freshness_from_the_newest_sample(tmp_path):
    rows = [_row(t, 0, 50.0, interval=5.0) for t in range(0, 600, 10)]  # rows claim 5 s, arrive every 10 s
    sub_job = _load(tmp_path, rows)
    assert sub_job.sample_interval_s == pytest.approx(10.0) and sub_job.reported_interval_s == 5.0
    now = sub_job.latest_sample_ts + timedelta(seconds=40)
    text = sampling_text(sub_job, now)
    clock = sub_job.latest_sample_ts.astimezone().strftime("%H:%M:%S")
    # "Scraped": DCGM may refresh a field less often than the sampler reads it.
    assert text == f"scraped every 10 s (rows report 5 s)  ·  newest 40 s ago ({clock})"
    regular = _load(tmp_path / "regular", [_row(t, 0, 50.0) for t in range(0, 600, 5)])
    assert sampling_text(regular, regular.latest_sample_ts + timedelta(minutes=3)).startswith(
        "scraped every 5 s  ·  newest 3m 00s ago")


def test_a_gpu_that_stops_reporting_is_counted_and_marked(tmp_path):
    rows = [_row(t, 0, 50.0) for t in range(0, 900, 5)] + [_row(t, 1, 70.0) for t in range(0, 600, 5)]
    sub_job = _load(tmp_path, rows)
    assert sub_job.gpu_ids == (0, 1) and sub_job.reporting_gpu_ids == (0,)
    gpu0, gpu1 = sub_job.views["10m"].rows
    assert gpu0.reporting and not gpu1.reporting
    assert gpu1.current_util_pct == 70.0  # its newest reading, flagged rather than hidden
    assert sub_job.views["10m"].busy.latest == pytest.approx(50.0)  # latest is the newest capture only
    allocation = GpuAllocation(hardware="H200", requested=2)
    assert allocation_text(allocation, 1, 2) == "H200  ·  GPUs: 2 requested, 1 reporting (2 seen)"


def test_windows_start_at_ten_minutes_and_ten_minutes_is_the_default():
    assert list(RANGES) == ["10m", "30m", "1h", "6h", "all"]
    assert DEFAULT_RANGE == "10m"
    assert min(duration for duration in RANGES.values() if duration is not None) == timedelta(minutes=10)


def test_gpu_type_and_requested_gpus_come_from_the_job():
    """The control plane's rule: n_gpus, but 0 for log_prob and for a multiplexed sub-job."""
    job = {
        "hardware": "B200",
        "sub_jobs": [
            {"sub_job_id": "j:training:0", "job_type": "TRAINING", "training_config": {"n_gpus": 8.0}},
            {"sub_job_id": "j:sampling:0", "job_type": "sampling",
             "inference_config": {"n_gpus": 2, "multiplex_job_id": "other:sampling:0"}},
            {"sub_job_id": "j:log_prob:0", "job_type": "JOB_TYPE_LOG_PROBABILITY", "inference_config": {"n_gpus": 4}},
            # The config block that belongs to the type is the one read.
            {"sub_job_id": "j:sampling:1", "job_type": "sampling",
             "training_config": {"n_gpus": 9}, "inference_config": {"n_gpus": 1}},
            {"sub_job_id": "j:sampling:2", "inference_config": {"n_gpus": 3}},  # type from the id
        ],
    }
    training = gpu_allocation(job, "j:training:0")
    assert training == GpuAllocation(hardware="B200", requested=8)
    assert allocation_text(training, 8, 8) == "B200  ·  GPUs: 8 requested, 8 reporting"
    shared = gpu_allocation(job, "j:sampling:0")
    assert shared == GpuAllocation(hardware="B200", requested=0, shared_with="other:sampling:0")
    assert allocation_text(shared) == "B200  ·  GPUs: 0 requested (multiplexed)"
    assert no_metrics_text(shared) == (
        "no GPU metrics for this sub-job: it runs on the GPUs of other:sampling:0, whose metrics include it")
    log_prob = gpu_allocation(job, "j:log_prob:0")
    assert log_prob.log_prob and log_prob.requested == 0
    assert allocation_text(log_prob, 0, 0) == "B200  ·  GPUs: 0 requested (log_prob), 0 reporting"
    assert no_metrics_text(log_prob) == "no GPU metrics for this sub-job"
    assert gpu_allocation(job, "j:sampling:1").requested == 1
    assert gpu_allocation(job, "j:sampling:2").requested == 3
    assert allocation_text(gpu_allocation({}, "j:training:0")) == "GPU type unknown  ·  GPUs: requested count unknown"
    assert gpu_allocation({"hardware": "H200"}, "missing:training:0") == GpuAllocation(hardware="H200")


def test_stats_text_labels_compute_and_memory_apart_and_fits_its_width(tmp_path):
    sub_job = _load(tmp_path, [_row(t, g, 50.0 + g) for t in range(0, 1200, 5) for g in range(2)])
    view, whole = sub_job.views["10m"], sub_job.views["all"]
    wide = stats_text(view, whole, width=120)
    assert BUSY_LABEL in wide and MEMORY_LABEL in wide
    assert "last 10m" in wide and "whole job" in wide
    assert wide.splitlines()[1].split() == ["min", "avg", "max", "latest"] * 2
    assert all(len(line) <= 120 for line in wide.splitlines())
    # Short names once the long ones don't fit, then a row per metric and scope.
    medium = stats_text(view, whole, width=76)
    assert [" ".join(line.split()[:3]) for line in medium.splitlines()[2:]] == ["GPU util %", "GPU mem used"]
    assert all(len(line) <= 76 for line in medium.splitlines()) and len(medium.splitlines()) == 4
    narrow = stats_text(view, whole, width=50)
    assert [line.split()[0] for line in narrow.splitlines()[1:]] == ["GPU", "whole", "GPU", "whole"]
    assert narrow.splitlines()[0].startswith("mean of 2 GPUs")  # what the values are
    assert all(len(line) <= 50 for line in narrow.splitlines())
    assert "loading older samples…" in stats_text(view, whole, width=120, partial=True)
    only_whole = stats_text(whole, whole, width=120)
    assert only_whole.count("whole job") == 1 and "last" not in only_whole


def test_the_memory_chart_draws_the_memory_series(tmp_path):
    rows = [_row(t, 0, 0.0, used=99 * GIB, free=1 * GIB) for t in range(0, 300, 5)]
    view = _load(tmp_path, rows).views["10m"]
    memory = render_chart(view, 30, series="memory").splitlines()[0]
    busy = render_chart(view, 30).splitlines()[0]
    assert set(memory.removeprefix("100% ┤")) == {"█"}
    assert set(busy.removeprefix("100% ┤")) == {"▁"}


def test_interval_and_age_formatting():
    assert [format_interval(v) for v in (5.0, 5.02, 2.5, 30, 90, 600, None, 0)] == [
        "5 s", "5 s", "2.5 s", "30 s", "1m 30s", "10m", "—", "—"]
    assert [format_age(v) for v in (0, 40.9, 61, 3 * 3600 + 120, 2 * 86400 + 7200)] == [
        "0 s", "40 s", "1m 01s", "3h 02m", "2d 2h"]


def test_while_older_samples_load_only_covered_windows_are_shown_as_complete(tmp_path, monkeypatch):
    import cortex_training.tui.resource as resource

    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    path = tmp_path / SUB / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(_row(t, g, 50.0)) + "\n" for t in range(0, 4 * 3600, 5) for g in range(2)))
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["c0"], "bytes": path.stat().st_size}))
    partials = []
    resource.ResourceLoader(tmp_path).load(priority=SUB, window=timedelta(minutes=10), on_partial=partials.append)
    early = gpu_jobs(partials[0])[SUB]
    assert early.partial
    assert resource.window_complete(early, early.views["10m"])
    assert not resource.window_complete(early, early.views["6h"]) and not resource.window_complete(early, early.views["all"])
    six_hours = stats_text(early.views["6h"], early.views["all"], width=120, partial=True, window_partial=True)
    assert six_hours.count("loading older samples…") == 4  # both series, both scopes
    whole = stats_text(early.views["all"], early.views["all"], width=120, partial=True)
    assert whole.count("loading older samples…") == 2


def _two_workers(*, seconds=3600, lag=0.0, offset=0.0, jitter=0.0, seed=5):
    """Worker A's four GPUs at 90 % busy, worker B's four at 30 %: 60 % across all eight.

    ``offset``: B scrapes this long after A (each scrape ± ``jitter``). ``lag``: B's
    newest rows (that many seconds) haven't been uploaded yet.
    """
    import random

    rng = random.Random(seed)
    end = seconds - 5
    rows = []
    for t in range(0, seconds, 5):
        for worker in (0, 1):
            ts = t + (offset if worker else 0.0) + rng.uniform(-jitter, jitter)
            if worker and ts > end - lag:
                continue
            for local in range(4):
                rows.append(_row(ts, worker * 4 + local, 90.0 if worker == 0 else 30.0,
                                 used=(70 if worker == 0 else 30) * GIB, free=(30 if worker == 0 else 70) * GIB))
    return rows


@pytest.mark.parametrize("case", [
    {"lag": 30.0},  # B's newest chunk is still being uploaded
    {"lag": 75.0},  # an upload's worst-case delay
    {"offset": 4.0, "jitter": 0.3},  # B scrapes just before A's next scrape
    {"offset": 2.5, "jitter": 0.3},  # half an interval apart: scrapes land in either bucket
    {"offset": 2.5, "jitter": 0.3, "lag": 30.0},  # both
    {"offset": 3.0, "lag": 30.0},  # and the job's first bucket holds only A's first scrape
])
def test_stats_and_charts_average_every_worker_when_workers_scrape_and_upload_apart(tmp_path, case):
    sub_job = _load(tmp_path, _two_workers(**case))
    assert len(sub_job.reporting_gpu_ids) == 8
    for name in ("10m", "1h", "all"):
        view = sub_job.views[name]
        for stats, expected in ((view.busy, 60), (view.memory, 50)):
            assert (stats.min, stats.max, stats.latest) == pytest.approx((expected,) * 3), (name, case)
        assert {round(value) for value in view.chart if value is not None} == {60}, (name, case)
        assert {round(value) for value in view.mem_chart if value is not None} == {50}, (name, case)


def test_a_gap_in_every_gpu_stays_a_gap(tmp_path):
    rows = [_row(t, g, 50.0) for t in range(0, 600, 5) for g in range(2) if not 300 <= t < 330]
    view = _load(tmp_path, rows).views["10m"]
    assert sum(value is None for value in view.chart) == 6  # no GPU sampled those 30 s


def test_a_gpu_that_stopped_counts_until_it_is_no_longer_reporting(tmp_path):
    # GPU 1 (at 10 %) stops 5 minutes before the newest sample; GPU 0 runs at 90 %.
    rows = [_row(t, 0, 90.0) for t in range(0, 1200, 5)] + [_row(t, 1, 10.0) for t in range(0, 900, 5)]
    sub_job = _load(tmp_path, rows)
    assert sub_job.reporting_gpu_ids == (0,)
    chart = sub_job.views["10m"].chart
    # 50 % while GPU 1 still counts as reporting (up to two minutes after its last
    # sample), then GPU 0 alone.
    assert round(chart[-1]) == 90 and round(chart[0]) == 50
    assert sorted({round(value) for value in chart}) == [50, 90]


def test_a_row_with_nothing_to_chart_cannot_stretch_the_whole_job(tmp_path):
    import time

    clean = [_row(t, g, 50.0 + g) for t in range(0, 3600, 5) for g in range(8)]
    epoch = _row(0, 0, None, used=None, free=None, interval=None)
    epoch["sample_ts"] = "1970-01-01T00:00:05Z"
    future = _row(0, 1, None, interval=None)
    future["sample_ts"] = "9000-01-01T00:00:00Z"
    expected = _load(tmp_path / "clean", clean).views["all"]
    started = time.perf_counter()
    rows = clean[: len(clean) // 2] + [epoch] + clean[len(clean) // 2:]
    whole = _load(tmp_path / "epoch", rows).views["all"]
    assert time.perf_counter() - started < 5
    assert whole.chart == expected.chart and whole.start == expected.start
    assert (whole.busy, whole.memory) == (expected.busy, expected.memory)
    # A far-future row is unsupported, so it can't move the newest sample.
    sub_job = _load(tmp_path / "future", clean + [future])
    assert all(len(view.chart) <= 480 for view in sub_job.views.values())
    assert sub_job.views["all"].end == expected.end
    assert sub_job.views["all"].busy.avg == pytest.approx(expected.busy.avg)


def _stat_values(text: str, label: str) -> list[str]:
    """The 8 cells (window min/avg/max/latest, then the whole job's) of one stats row."""
    return next(line for line in text.splitlines() if line.startswith(label)).split()[-8:]


def test_whole_job_extremes_older_than_the_raw_retention_stay_exact(tmp_path):
    # Nine hours: a 90-second dip to 0 % in the first hour, older than the raw samples.
    rows = [_row(t, 0, 0.0 if 1800 <= t < 1890 else 50.0) for t in range(0, 9 * 3600, 30)]
    for row in rows:
        row["sample_interval_s"] = 30.0
    sub_job = _load(tmp_path, rows)
    whole, window = sub_job.views["all"], sub_job.views["6h"]
    # Its intervals were folded into the bins' extremes before they left the raw samples.
    assert (whole.busy.min, whole.busy.max) == (0.0, 50.0)
    assert not (whole.busy.min_bound or whole.busy.max_bound)
    assert not (whole.memory.min_bound or whole.memory.max_bound)
    assert not (window.busy.min_bound or window.busy.max_bound)
    text = stats_text(window, whole, width=120)
    busy = _stat_values(text, BUSY_LABEL)
    assert busy[4] == "0%" and busy[6] == "50%" and busy[0] == "50%"
    assert not any(cell[0] in "≤≥" for cell in _stat_values(text, MEMORY_LABEL))


def test_whole_job_extremes_are_exact_when_the_older_history_cannot_change_them(tmp_path):
    def rows(peak: bool):
        # Nine hours at 50 %, with a dip to 0 % in the last hour (in the raw samples),
        # and with ``peak`` a 30-second spike to 90 % in the first hour (only binned).
        out = []
        for t in range(0, 9 * 3600, 30):
            util = 0.0 if 8 * 3600 + 600 <= t < 8 * 3600 + 690 else 90.0 if peak and t == 1800 else 50.0
            out.append(_row(t, 0, util, interval=30.0))
        return out

    sub_job = _load(tmp_path, rows(peak=False))
    whole = sub_job.views["all"]
    assert (whole.busy.min, whole.busy.max) == (0.0, 50.0)
    assert not (whole.busy.min_bound or whole.busy.max_bound or whole.memory.min_bound or whole.memory.max_bound)
    text = stats_text(sub_job.views["1h"], whole, width=120)
    assert _stat_values(text, BUSY_LABEL)[4:7] == ["0%", "50%", "50%"]
    assert _stat_values(text, MEMORY_LABEL)[4:7] == ["40%", "40%", "40%"]
    # A spike older than the raw samples is not averaged into its bin: the max stays exact.
    whole = _load(tmp_path / "peak", rows(peak=True)).views["all"]
    assert whole.busy.max == 90.0 and not whole.busy.max_bound and not whole.busy.min_bound


def test_a_row_stamped_well_ahead_of_this_clock_is_unsupported(tmp_path):
    now = (datetime.now(timezone.utc) - START).total_seconds()
    rows = [_row(now - 5 * step, gpu, 80.0, used=50.0, free=50.0) for step in range(12 * 120, -1, -1)
            for gpu in range(8)]
    skewed = dict(rows[0], global_gpu_index=3, sample_ts="2099-01-01T00:00:00Z")
    early = _row(now + 60, 2, 80.0, used=50.0, free=50.0)  # a minute ahead: clock skew within reason
    sub_job = _load(tmp_path, rows + [skewed, early])
    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 1  # shown as "1 row skipped (malformed or unsupported)"
    # The skewed row moves neither the newest sample nor the GPUs counted as reporting.
    assert sub_job.latest_sample_ts == START + timedelta(seconds=now + 60)
    assert sub_job.reporting_gpu_ids == tuple(range(8))
    assert sub_job.views["10m"].busy == sub_job.views["all"].busy
    assert sub_job.views["all"].start == START + timedelta(seconds=now - 5 * 12 * 120)


def test_the_capture_at_a_windows_left_edge_is_charted_but_not_averaged(tmp_path):
    # Every capture lands 50 ms before its 5-second tick, except the newest, which anchors
    # the windows: the capture at the 10-minute edge then falls just before it.
    rows = []
    for k in range(241):
        t = 5 * k - (0 if k == 240 else 0.05)
        util = 10.0 if k == 120 else 50.0  # the capture at the edge
        rows += [_row(t, gpu, util, used=10.0 if k == 120 else 40.0, free=90.0 if k == 120 else 60.0)
                 for gpu in range(4)]
    view = _load(tmp_path, rows).views["10m"]
    assert view.chart[0] == pytest.approx(10.0) and view.mem_chart[0] == pytest.approx(10.0)
    assert all(value == pytest.approx(50.0) for value in view.chart[1:])
    assert view.busy.avg == pytest.approx(50.0) and view.memory.avg == pytest.approx(40.0)
    assert render_chart(view, 60).splitlines()[0][6] != " "  # no blank first column


def test_chart_labels_carry_the_date_when_a_view_spans_a_day(tmp_path):
    rows = [_row(t, 0, 50.0, interval=60.0) for t in range(0, 30 * 3600, 60)]
    sub_job = _load(tmp_path, rows)
    dated = re.compile(r"\d\d-\d\d \d\d:\d\d")
    whole_labels = render_chart(sub_job.views["all"], 60).splitlines()[2]
    assert len(dated.findall(whole_labels)) == 3
    assert whole_labels.strip().startswith(sub_job.views["all"].start.astimezone().strftime("%m-%d %H:%M"))
    assert len(dated.findall(render_chart(sub_job.views["all"], 30).splitlines()[2])) == 1 + 1  # start and end
    narrow = render_chart(sub_job.views["all"], 20).splitlines()[2]
    assert not dated.search(narrow) and re.search(r"\d\d:\d\d", narrow)  # too narrow: times only
    assert not dated.search(render_chart(sub_job.views["6h"], 60).splitlines()[2])
