# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resources: the derived memory %, units, scales, the mean across units vs one unit, exact extremes."""

from datetime import timedelta

import pytest

from cortex_training.tui import resource as resource_module
from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import axis_labels
from cortex_training.tui.resource import format_value
from cortex_training.tui.resource import load_resource_data
from cortex_training.tui.resource import nice_ceiling
from cortex_training.tui.resource import render_metric_chart
from tests.v2_resource_support import GIB
from tests.v2_resource_support import SUB
from tests.v2_resource_support import gpu_row
from tests.v2_resource_support import scrapes
from tests.v2_resource_support import system_row
from tests.v2_resource_support import write


def _load(root, rows):
    write(root, rows)
    return load_resource_data(root)


# ─── the derived memory % ─────────────────────────────────────────────────────


def test_gpu_memory_used_is_used_over_used_plus_free_per_gpu_and_the_mean_averages_them(tmp_path):
    rows = []
    for t in scrapes(300):
        rows.append(gpu_row(t, 0, used=20 * GIB, free=80 * GIB))  # 20%
        rows.append(gpu_row(t, 1, used=60 * GIB, free=20 * GIB))  # 75% of a smaller GPU
    gpu = _load(tmp_path, rows).kind(SUB, GPU)
    assert gpu.unit(0).latest["fb_used_pct"] == pytest.approx(20.0)
    assert gpu.unit(1).latest["fb_used_pct"] == pytest.approx(75.0)
    assert gpu.latest["fb_used_pct"] == pytest.approx(47.5)
    assert set(round(v, 9) for v in gpu.view("10m").chart("fb_used_pct")) == {47.5}
    assert gpu.view("all", 1).stats["fb_used_pct"].avg == pytest.approx(75.0)
    assert gpu.view("10m", 0).stats["fb_used_bytes"].latest == pytest.approx(20.0)  # GiB


def test_node_memory_used_is_used_over_the_limit_and_never_above_100(tmp_path):
    rows = [system_row(t, 0, used=215 * GIB, total=860 * GIB) for t in scrapes(300)]
    rows += [system_row(t, 1, used=900 * GIB, total=860 * GIB) for t in scrapes(300)]  # the host's numbers
    system = _load(tmp_path, rows).kind(SUB, SYSTEM)
    assert system.unit(0).latest["mem_used_pct"] == pytest.approx(25.0)
    assert system.unit(1).latest["mem_used_pct"] == 100.0
    assert system.view("10m", 1).stats["mem_used_pct"].max == 100.0
    assert system.view("10m").stats["mem_used_pct"].avg == pytest.approx(62.5)
    assert system.view("10m").stats["mem_total_bytes"].latest == pytest.approx(860.0)


def test_without_the_second_memory_field_there_is_no_memory_percentage(tmp_path):
    rows = [gpu_row(t, 0) for t in scrapes(60)]
    for row in rows:
        del row["fb_free_bytes"]
    keys = [m.key for m in _load(tmp_path, rows).kind(SUB, GPU).metrics]
    assert keys[0] == "gpu_util_pct" and "fb_used_pct" not in keys and "fb_used_bytes" in keys


# ─── units ────────────────────────────────────────────────────────────────────


def test_units_are_gpus_by_global_index_and_nodes_by_worker_counted_from_one(tmp_path):
    rows = [gpu_row(t, g, util=10.0 * (g + 1)) for t in scrapes(300) for g in range(8)]
    rows += [system_row(t, n, cpu=2.0 + n) for t in scrapes(300) for n in range(2)]
    data = _load(tmp_path, rows)
    gpu, system = data.kind(SUB, GPU), data.kind(SUB, SYSTEM)
    assert [(u.id, u.name) for u in gpu.units] == [(g, f"GPU {g}") for g in range(8)]
    assert [(u.id, u.name) for u in system.units] == [(0, "node 1"), (1, "node 2")]
    assert [u.latest["gpu_util_pct"] for u in gpu.units] == [10.0 * (g + 1) for g in range(8)]
    assert [u.latest["cpu_util_pct"] for u in system.units] == [2.0, 3.0]
    assert all(u.reporting for u in gpu.units + system.units)
    assert gpu.view("10m").label == "mean of 8 GPUs" and system.view("10m").label == "mean of 2 nodes"
    assert gpu.view("10m", 3).label == "GPU 3" and system.view("10m", 1).label == "node 2"
    assert gpu.view("10m", 9) is None and gpu.view("2h") is None


def test_a_unit_that_stops_reporting_is_marked_and_its_newest_values_kept(tmp_path):
    rows = [system_row(t, 0, cpu=3.0) for t in scrapes(900)] + [system_row(t, 1, cpu=9.0) for t in scrapes(600)]
    system = _load(tmp_path, rows).kind(SUB, SYSTEM)
    assert system.reporting_unit_ids == (0,)
    assert [u.reporting for u in system.units] == [True, False]
    assert system.unit(1).latest["cpu_util_pct"] == 9.0
    assert system.latest["cpu_util_pct"] == 3.0  # the newest capture: the reporting node only


def test_one_node_alone_is_named_and_its_mean_is_its_own_values(tmp_path):
    system = _load(tmp_path, [system_row(t, 0, cpu=1.0 + (t // 5) % 3) for t in scrapes(300)]).kind(SUB, SYSTEM)
    mean, node = system.view("10m"), system.view("10m", 0)
    assert mean.label == "node 1" and mean.units == 1
    assert mean.chart("cpu_util_pct") == node.chart("cpu_util_pct")
    assert mean.stats["cpu_util_pct"] == node.stats["cpu_util_pct"]


# ─── the mean across units vs one unit ────────────────────────────────────────


def test_the_mean_averages_the_units_and_a_unit_view_has_only_its_own_values(tmp_path):
    rows = [gpu_row(t, g, util=20.0 + 60 * g, power_watts=200.0 + 100 * g) for t in scrapes(600) for g in range(2)]
    gpu = _load(tmp_path, rows).kind(SUB, GPU)
    for name in ("10m", "all"):
        mean, zero, one = gpu.view(name), gpu.view(name, 0), gpu.view(name, 1)
        assert {round(v, 9) for v in mean.chart("gpu_util_pct")} == {50.0}
        assert {round(v, 9) for v in zero.chart("gpu_util_pct")} == {20.0}
        assert {round(v, 9) for v in one.chart("power_watts")} == {300.0}
        assert len(zero.chart("gpu_util_pct")) == len(mean.chart("gpu_util_pct"))  # one time axis
        assert (mean.stats["gpu_util_pct"].avg, one.stats["gpu_util_pct"].avg) == (pytest.approx(50), pytest.approx(80))
        assert (mean.units, zero.units) == (2, 1)


# ─── scales ───────────────────────────────────────────────────────────────────


def test_nice_ceilings():
    assert [nice_ceiling(v) for v in (0.0042, 0.9, 1.0, 1.01, 2.4, 2.5, 3.0, 7.0, 25.0, 26.0, 99.0, 100.0, 101.0,
                                      405.0, 860.0, 1720.0)] == [
        0.005, 1.0, 1.0, 2.0, 2.5, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    assert [nice_ceiling(v) for v in (None, 0.0, -3.0, float("inf"))] == [1.0] * 4


def test_percentages_are_on_0_to_100_and_the_rest_on_the_whole_jobs_nice_ceiling(tmp_path):
    rows = [gpu_row(t, g, util=10.0, power_watts=(410.0 if (g, t) == (1, 0.0) else 150.0)) for t in scrapes(3600)
            for g in range(2)]
    gpu = _load(tmp_path, rows).kind(SUB, GPU)
    scales = {m.key: m.scale for m in gpu.metrics}
    assert scales["gpu_util_pct"] == scales["fb_used_pct"] == scales["tensor_active_pct"] == 100.0
    # The spike is an hour before the newest sample, in no short window, and on one GPU only:
    # the axis still covers it, so it doesn't move with the window or the unit.
    assert scales["power_watts"] == 500.0
    assert scales["fb_used_bytes"] == 100.0  # 60 GiB
    assert gpu.view("10m", 0).stats["power_watts"].max == pytest.approx(150.0)
    power = gpu.metric("power_watts")
    assert axis_labels(power) == ("500 W", "250 W", "0 W")
    assert axis_labels(gpu.metric("gpu_util_pct")) == ("100%", "50%", "0%")
    assert axis_labels(gpu.metric("fb_used_bytes")) == ("100 GiB", "50 GiB", "0 GiB")


def test_a_chart_is_drawn_to_its_scale_with_an_axis_as_wide_as_its_longest_label(tmp_path):
    # 400 W once, 20 minutes in the past: the axis tops out at 500 W; the last 10 minutes run at 250 W.
    rows = [gpu_row(t, 0, util=50.0, power_watts=400.0 if t == 0 else 250.0) for t in scrapes(1200)]
    gpu = _load(tmp_path, rows).kind(SUB, GPU)
    view = gpu.view("10m")
    power = render_metric_chart(view, gpu.metric("power_watts"), 60, height=4).splitlines()
    assert [line[:7] for line in power[:5]] == ["500 W ┤", "      │", "250 W ┤", "      │", "  0 W └"]
    assert set(power[2][7:]) == {"█"} and set(power[1][7:]) == {" "}  # 250 of 500 W: two of four rows
    util = render_metric_chart(view, gpu.metric("gpu_util_pct"), 60, height=4).splitlines()
    assert [line[:6] for line in util[:5]] == ["100% ┤", "     │", " 50% ┤", "     │", "  0% └"]
    assert all(len(line) <= 60 for line in power + util)


def test_values_are_shown_in_their_unit():
    assert [format_value(v, "%") for v in (85.4, 9.94, 1.83, 0.0, None)] == ["85%", "9.9%", "1.8%", "0%", "—"]
    assert [format_value(v, "GiB") for v in (0.5, 87.44, 923.4, 1720.0)] == [
        "0.50 GiB", "87.4 GiB", "923 GiB", "1720 GiB"]
    assert format_value(-0.0, "%") == "0%"
    assert [format_value(v, u) for v, u in ((116.873, "W"), (34.0, "°C"), (88.0, ""), (1.5, ""))] == [
        "117 W", "34 °C", "88", "1.5"]


# ─── exact extremes per unit (and the mean's) ────────────────────────────────


def test_a_units_min_and_max_are_its_samples_not_its_chart_buckets(tmp_path):
    # 6 hours at 5 s: the 6h window's buckets are 90 s wide, so a 5-second spike is
    # one sample in a bucket of 18.
    rows = [gpu_row(t, g, util=(100.0 if (g, t) == (1, 3 * 3600.0) else 40.0)) for t in scrapes(6 * 3600)
            for g in range(2)]
    gpu = _load(tmp_path, rows).kind(SUB, GPU)
    window = gpu.view("6h", 1)
    assert max(v for v in window.chart("gpu_util_pct") if v is not None) < 50  # the bucket averages it away
    assert window.stats["gpu_util_pct"].max == 100.0
    assert gpu.view("all", 1).stats["gpu_util_pct"].max == 100.0
    assert gpu.view("1h", 1).stats["gpu_util_pct"].max == 40.0  # the spike is older than the hour
    # The mean's extremes are those of the average across GPUs at each interval.
    assert gpu.view("6h").stats["gpu_util_pct"].max == pytest.approx(70.0)
    for unit in (None, 0, 1):
        stats = [gpu.view(name, unit).stats["gpu_util_pct"] for name in RANGES]
        assert [s.min for s in stats] == sorted([s.min for s in stats], reverse=True)  # nested windows nest
        assert [s.max for s in stats] == sorted(s.max for s in stats)
        assert all(s.min <= s.avg <= s.max for s in stats)


def test_a_units_whole_job_extremes_stay_exact_once_its_history_is_only_binned(tmp_path):
    # Nine hours: a dip on GPU 0 in the first hour, long out of the raw samples.
    rows = [gpu_row(t, g, util=(2.0 if g == 0 and 1800 <= t < 1830 else 60.0 + g), interval=30.0)
            for t in scrapes(9 * 3600, step=30) for g in range(2)]
    data = _load(tmp_path, rows)
    gpu = data.kind(SUB, GPU)
    whole = gpu.view("all", 0).stats["gpu_util_pct"]
    assert (whole.min, whole.max, whole.min_bound, whole.max_bound) == (2.0, 60.0, False, False)
    assert gpu.view("all", 1).stats["gpu_util_pct"].min == 61.0
    assert gpu.view("6h", 0).stats["gpu_util_pct"].min == 60.0


def test_one_gpus_view_uses_exact_extremes_in_a_short_window(tmp_path):
    rows = [gpu_row(t, g, util=(95.0 if (g, t) == (0, 1500.0) else 30.0)) for t in scrapes(3600) for g in range(2)]
    one = _load(tmp_path, rows).kind(SUB, GPU).view("1h", 0)
    assert one.stats["gpu_util_pct"].max == 95.0 and one.label == "GPU 0"
    assert one.stats["gpu_util_pct"].min == 30.0


def test_scrape_stats_of_a_kind_work_with_the_existing_header_text(tmp_path):
    rows = [system_row(t, 0) for t in scrapes(600, step=5)]
    system = _load(tmp_path, rows).kind(SUB, SYSTEM)
    now = system.latest_sample_ts + timedelta(seconds=40)
    assert resource_module.sampling_text(system, now).startswith("scraped every 5 s  ·  newest 40 s ago (")
    assert system.value_interval_s is None  # CPU repeats at 0.1% resolution: no values-change note
    assert resource_module.stale(system, system.latest_sample_ts + timedelta(seconds=179)) is False
    assert resource_module.stale(system, system.latest_sample_ts + timedelta(seconds=181)) is True
    assert resource_module.freshness_text(system, now, stale=True, what="samples").startswith("no new samples for 40 s")
