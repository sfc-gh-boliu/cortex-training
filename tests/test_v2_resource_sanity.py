# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resources against numbers worked out from the rows themselves (8 GPUs on 2 nodes, varying values)."""

import math

import pytest

from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import load_resource_data
from tests.v2_resource_support import GIB
from tests.v2_resource_support import SUB
from tests.v2_resource_support import gpu_row
from tests.v2_resource_support import scrapes
from tests.v2_resource_support import system_row
from tests.v2_resource_support import write

MINUTES = 40


def _rows():
    rows = []
    for t in scrapes(MINUTES * 60):
        k = t // 5
        for node in range(2):
            rows.append(system_row(t, node, cpu=round(2.5 + 3 * node + 2 * math.sin(k / 7 + node), 1),
                                   used=(200 + 40 * node + 10 * math.sin(k / 11)) * GIB, total=860 * GIB))
            for local in range(4):
                g = 4 * node + local
                util = round(50 + 45 * math.sin(k / 13 + g), 1)
                used = (60 + 20 * math.sin(k / 17 + g)) * GIB
                rows.append(gpu_row(t, g, util=util, used=used, free=141 * GIB - used,
                                    power_watts=round(120 + 4 * util + g, 3)))
    return rows


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    rows = _rows()
    root = tmp_path_factory.mktemp("telemetry")
    write(root, rows)
    data = load_resource_data(root)
    return rows, data.kind(SUB, GPU), data.kind(SUB, SYSTEM)


def _in_window(rows, newest, seconds, kind):
    from cortex_training.tui.resource import _sample_epoch

    return [row for row in rows if (row.get("kind") == "system") == (kind == SYSTEM)
            and _sample_epoch(row["sample_ts"]) >= newest - seconds - 1e-6]


def test_window_and_whole_job_averages_are_the_samples_time_weighted_means(loaded):
    rows, gpu, system = loaded
    newest = gpu.latest_sample_ts.timestamp()
    for name, seconds in (("10m", 600), ("30m", 1800)):
        window = _in_window(rows, newest, seconds, GPU)
        assert gpu.view(name).stats["gpu_util_pct"].avg == pytest.approx(
            sum(r["gpu_util_pct"] for r in window) / len(window))
        assert gpu.view(name, 5).stats["power_watts"].avg == pytest.approx(
            sum(r["power_watts"] for r in window if r["global_gpu_index"] == 5)
            / sum(1 for r in window if r["global_gpu_index"] == 5))
        nodes = _in_window(rows, newest, seconds, SYSTEM)
        assert system.view(name).stats["cpu_util_pct"].avg == pytest.approx(
            sum(r["cpu_util_pct"] for r in nodes) / len(nodes))


def test_the_means_extremes_are_those_of_the_average_across_units_at_each_scrape(loaded):
    rows, gpu, system = loaded
    by_scrape: dict = {}
    for row in rows:
        if "kind" not in row:
            by_scrape.setdefault(row["sample_ts"], []).append(row["gpu_util_pct"])
    means = [sum(v) / len(v) for v in by_scrape.values()]
    whole = gpu.view("all").stats["gpu_util_pct"]
    assert (whole.min, whole.max) == (pytest.approx(min(means)), pytest.approx(max(means)))
    cpu = {}
    for row in rows:
        if row.get("kind") == "system":
            cpu.setdefault(row["sample_ts"], []).append(row["cpu_util_pct"])
    cpu_means = [sum(v) / len(v) for v in cpu.values()]
    stats = system.view("all").stats["cpu_util_pct"]
    assert (stats.min, stats.max) == (pytest.approx(min(cpu_means)), pytest.approx(max(cpu_means)))


def test_a_units_extremes_and_latest_are_its_own_samples(loaded):
    rows, gpu, system = loaded
    for g in (0, 3, 7):
        mine = [r for r in rows if "kind" not in r and r["global_gpu_index"] == g]
        stats = gpu.view("all", g).stats
        assert (stats["power_watts"].min, stats["power_watts"].max) == (
            min(r["power_watts"] for r in mine), max(r["power_watts"] for r in mine))
        assert stats["gpu_util_pct"].latest == mine[-1]["gpu_util_pct"]
        newest = mine[-1]
        assert stats["fb_used_pct"].latest == pytest.approx(
            100 * newest["fb_used_bytes"] / (newest["fb_used_bytes"] + newest["fb_free_bytes"]))
        assert stats["fb_free_bytes"].latest == pytest.approx(newest["fb_free_bytes"] / GIB)
    node = [r for r in rows if r.get("kind") == "system" and r["worker_num"] == 1]
    stats = system.view("all", 1).stats["mem_used_pct"]
    shares = [100 * r["mem_used_bytes"] / r["mem_total_bytes"] for r in node]
    assert (stats.min, stats.max, stats.latest) == (
        pytest.approx(min(shares)), pytest.approx(max(shares)), pytest.approx(shares[-1]))


def test_the_newest_capture_and_scales(loaded):
    rows, gpu, system = loaded
    newest = [r for r in rows if r["sample_ts"] == rows[-1]["sample_ts"] and "kind" not in r]
    assert gpu.latest["gpu_util_pct"] == pytest.approx(sum(r["gpu_util_pct"] for r in newest) / len(newest))
    peak = max(r["power_watts"] for r in rows if "kind" not in r)
    assert peak <= gpu.metric("power_watts").scale < 2 * peak
    assert system.metric("mem_total_bytes").scale == 1000.0  # 860 GiB
    assert system.metric("cpu_util_pct").scale == 100.0
    assert gpu.reporting_unit_ids == tuple(range(8)) and system.reporting_unit_ids == (0, 1)
