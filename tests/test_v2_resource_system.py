# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resources: system rows (one node's CPU and memory per scrape) beside the GPU rows of one file."""

import random
from datetime import timedelta

import pytest

import cortex_training.tui.resource as resource
from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import ResourceLoader
from cortex_training.tui.resource import load_resource_data
from tests.v2_resource_support import SUB
from tests.v2_resource_support import append
from tests.v2_resource_support import gpu_row
from tests.v2_resource_support import scrapes
from tests.v2_resource_support import system_row
from tests.v2_resource_support import write
from tests.resource_legacy import gpu_jobs


def _mixed(seconds, *, gpus=2, nodes=1, start=0.0, step=5.0):
    """A scrape's rows as a real file holds them: each node's system row, then its GPUs'."""
    rows = []
    per_node = max(1, gpus // nodes)
    for t in scrapes(seconds, step=step, start=start):
        for node in range(nodes):
            rows.append(system_row(t, node, cpu=2.0 + node + (t // 5) % 3, used=(100 + 10 * node) * 2**30))
            gpus_here = range(node * per_node, (node + 1) * per_node)
            rows += [gpu_row(t, g, util=50.0 + g, per_node=per_node) for g in gpus_here]
    return rows


def test_a_file_with_system_rows_and_no_gpu_rows_keeps_its_sub_job(tmp_path):
    write(tmp_path, [system_row(t, n) for t in scrapes(600) for n in range(2)])
    data = load_resource_data(tmp_path)
    assert SUB not in gpu_jobs(data)  # the GPU part of the tab has nothing for it
    assert set(data.kinds[SUB]) == {SYSTEM} and data.kind(SUB, GPU) is None
    system = data.kind(SUB, SYSTEM)
    assert system.unit_ids == (0, 1) and data.ignored_rows == 0
    assert system.view("10m").stats["cpu_util_pct"].latest == pytest.approx(4.0)


def test_both_kinds_of_one_file_are_read_apart(tmp_path):
    write(tmp_path, _mixed(900, gpus=8, nodes=2))
    data = load_resource_data(tmp_path)
    gpu, system = data.kind(SUB, GPU), data.kind(SUB, SYSTEM)
    assert gpu.unit_ids == tuple(range(8)) and system.unit_ids == (0, 1)
    assert gpu_jobs(data)[SUB].gpu_ids == tuple(range(8))
    assert {m.key for m in gpu.metrics}.isdisjoint(m.key for m in system.metrics)
    assert data.ignored_rows == 0


def test_the_gpu_part_is_the_same_with_or_without_system_rows(tmp_path):
    mixed = _mixed(1800, gpus=4, nodes=2)
    with_system = gpu_jobs(load_resource_data(write(tmp_path / "mixed", mixed).parent.parent))[SUB]
    without = gpu_jobs(load_resource_data(
        write(tmp_path / "gpu", [row for row in mixed if "kind" not in row]).parent.parent))[SUB]
    assert with_system.latest_sample_ts == without.latest_sample_ts
    assert with_system.sample_interval_s == without.sample_interval_s
    for name in RANGES:
        x, y = with_system.views[name], without.views[name]
        assert (x.chart, x.mem_chart, x.busy, x.memory, x.rows) == (y.chart, y.mem_chart, y.busy, y.memory, y.rows)


def test_one_missed_scrape_is_bridged_and_a_longer_hole_stays_a_gap(tmp_path):
    rng = random.Random(3)
    rows = [system_row(t + rng.uniform(-0.1, 0.1), 0, cpu=3.0 + (t // 5) % 4) for t in scrapes(600)
            if t != 300 and not 450 <= t < 485]  # one miss at 300 s; a 35 s hole (six misses) at 450 s
    write(tmp_path, rows)
    system = load_resource_data(tmp_path).kind(SUB, SYSTEM)
    for unit in (None, 0):
        chart = system.view("10m", unit).chart("cpu_util_pct")
        empty = [k for k, value in enumerate(chart) if value is None]
        assert empty and all(85 <= k <= 100 for k in empty), empty  # only the hole
        assert len(empty) >= 5
    # A bridge only fills the chart: the averages still count each sample once.
    samples = [row["cpu_util_pct"] for row in rows]
    window = system.view("all").stats["cpu_util_pct"]
    assert window.avg == pytest.approx(sum(samples) / len(samples))


def test_gpu_rows_keep_their_gaps(tmp_path):
    rows = [gpu_row(t, 0) for t in scrapes(600) if t != 300]  # one missed GPU scrape stays a gap, as before
    write(tmp_path, rows)
    chart = load_resource_data(tmp_path).kind(SUB, GPU).view("10m").chart("gpu_util_pct")
    assert sum(value is None for value in chart) == 1


def test_a_node_without_a_sample_in_a_bucket_counts_with_its_last_value(tmp_path):
    rows = [system_row(t, 0, cpu=2.0) for t in scrapes(600)]
    rows += [system_row(t + 2.6, 1, cpu=8.0) for t in scrapes(600)]  # scrapes half an interval later
    write(tmp_path, rows)
    chart = load_resource_data(tmp_path).kind(SUB, SYSTEM).view("10m").chart("cpu_util_pct")
    assert {round(value, 9) for value in chart if value is not None} == {5.0}  # never one node alone


def test_rows_this_client_cant_read_are_counted_and_unknown_kinds_are_not(tmp_path):
    good = [system_row(t, 0) for t in scrapes(60)]
    no_node = system_row(60, 0)
    del no_node["worker_num"]
    old = system_row(65, 0, record_version=2)
    negative = system_row(70, 0, worker_num=-1)
    later = {"kind": "accelerator", "record_version": 1, "sub_job_id": SUB, "sample_ts": good[0]["sample_ts"]}
    write(tmp_path, good + [no_node, old, negative, later])
    data = load_resource_data(tmp_path)
    assert data.ignored_rows == 3
    assert data.kind(SUB, SYSTEM).latest_sample_ts == resource._utc(resource._sample_epoch(good[-1]["sample_ts"]))


def test_each_kind_is_measured_back_from_its_own_newest_sample(tmp_path):
    # GPU rows for an hour; the node keeps reporting for another hour.
    rows = [gpu_row(t, 0, util=70.0) for t in scrapes(3600)] + [system_row(t, 0) for t in scrapes(7200)]
    write(tmp_path, rows)
    data = load_resource_data(tmp_path)
    gpu, system = data.kind(SUB, GPU), data.kind(SUB, SYSTEM)
    assert system.latest_sample_ts - gpu.latest_sample_ts == timedelta(seconds=3600)
    assert all(value is not None for value in gpu.view("10m").chart("gpu_util_pct"))
    assert gpu.view("10m").end == gpu.latest_sample_ts and system.view("10m").end == system.latest_sample_ts


def test_a_refresh_reads_only_new_bytes_and_rebuilds_only_the_kind_that_changed(tmp_path):
    path = write(tmp_path, _mixed(1800, gpus=4, nodes=2))
    loader = ResourceLoader(tmp_path)
    first = loader.load()
    read = loader.bytes_read
    appended = append(path, [system_row(t, n) for t in scrapes(30, start=1800) for n in range(2)], ["c0", "c1"])
    second = loader.load()
    assert loader.bytes_read - read == appended
    assert second.kind(SUB, GPU) is first.kind(SUB, GPU)
    assert second.kind(SUB, SYSTEM) is not first.kind(SUB, SYSTEM)
    assert second.kind(SUB, SYSTEM).latest_sample_ts > first.kind(SUB, SYSTEM).latest_sample_ts
    nothing = loader.load()
    assert nothing.kind(SUB, SYSTEM) is second.kind(SUB, SYSTEM)


def _same_kind(a, b) -> None:
    assert [m for m in a.metrics] == [m for m in b.metrics]
    assert a.unit_ids == b.unit_ids and a.reporting_unit_ids == b.reporting_unit_ids
    assert (a.latest_sample_ts, a.first_sample_ts) == (b.latest_sample_ts, b.first_sample_ts)
    for name in RANGES:
        for unit in (None, *a.unit_ids):
            x, y = a.view(name, unit), b.view(name, unit)
            assert (x.start, x.end, x.units, x.label) == (y.start, y.end, y.units, y.label), (name, unit)
            assert set(x.charts) == set(y.charts)
            for key in x.charts:
                assert x.chart(key) == pytest.approx(y.chart(key), nan_ok=True), (name, unit, key)
            assert x.stats == y.stats, (name, unit)


def test_incremental_loads_of_both_kinds_match_a_full_reload_across_the_raw_retention(tmp_path):
    hours = 6.5
    path = write(tmp_path, _mixed(hours * 3600, gpus=2, nodes=2, step=10.0))
    loader = ResourceLoader(tmp_path)
    loader.load()
    names = ["c0"]
    t = hours * 3600
    while t < 8.5 * 3600:  # the job outgrows the raw retention along the way
        names.append(f"c{len(names)}")
        append(path, _mixed(1800, gpus=2, nodes=2, start=t, step=10.0), names)
        incremental, full = loader.load(), load_resource_data(tmp_path)
        for kind in (GPU, SYSTEM):
            _same_kind(incremental.kind(SUB, kind), full.kind(SUB, kind))
        t += 1800
    assert loader._files[SUB].system.raw_start > 0  # history older than the raw samples is only binned


def test_a_first_load_shows_both_kinds_newest_samples_first(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    write(tmp_path, _mixed(3 * 3600, gpus=2, nodes=1))
    loader = ResourceLoader(tmp_path)
    partials = []
    final = loader.load(priority=SUB, window=timedelta(minutes=10), on_partial=partials.append)
    assert len(partials) == 1
    early = partials[0]
    for kind in (GPU, SYSTEM):
        shown, whole = early.kind(SUB, kind), final.kind(SUB, kind)
        assert shown.partial and not whole.partial
        assert resource.window_complete(shown, shown.view("10m"))
        assert not resource.window_complete(shown, shown.view("all"))
        assert shown.view("10m").stats == whole.view("10m").stats
    _same_kind(final.kind(SUB, SYSTEM), load_resource_data(tmp_path).kind(SUB, SYSTEM))


def test_a_file_whose_newest_rows_are_all_system_rows_shows_them_first(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    rows = [gpu_row(t, 0) for t in scrapes(3600)] + [system_row(t, 0) for t in scrapes(3 * 3600, start=3600)]
    write(tmp_path, rows)
    partials = []
    final = ResourceLoader(tmp_path).load(priority=SUB, window=timedelta(minutes=10), on_partial=partials.append)
    assert partials and partials[0].kind(SUB, SYSTEM).partial
    assert final.kind(SUB, GPU).view("10m").stats == load_resource_data(tmp_path).kind(SUB, GPU).view("10m").stats


def test_rows_read_after_a_sort_in_the_middle_of_a_read_still_land_in_their_columns(tmp_path, monkeypatch):
    # Small blocks: each one ends in a fold, and the fold sorts a unit whose rows came out
    # of order; the blocks after it must still fill the same columns.
    monkeypatch.setattr(resource, "_READ_BLOCK", 4096)
    rows = _mixed(1800, gpus=2, nodes=1)
    rows[0], rows[3] = rows[3], rows[0]  # an older scrape's rows after a newer one's
    rows[1], rows[4] = rows[4], rows[1]
    write(tmp_path, rows)
    small = load_resource_data(tmp_path)
    monkeypatch.setattr(resource, "_READ_BLOCK", 4 << 20)
    large = load_resource_data(tmp_path)
    for kind in (GPU, SYSTEM):
        _same_kind(small.kind(SUB, kind), large.kind(SUB, kind))
    assert small.kind(SUB, SYSTEM).view("10m").stats["cpu_util_pct"].latest is not None


def test_a_newest_first_read_whose_older_part_spans_many_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    monkeypatch.setattr(resource, "_READ_BLOCK", 8 * 1024)
    write(tmp_path, _mixed(3 * 3600, gpus=2, nodes=1))
    partials = []
    final = ResourceLoader(tmp_path).load(priority=SUB, window=timedelta(minutes=10), on_partial=partials.append)
    assert partials
    monkeypatch.setattr(resource, "_READ_BLOCK", 4 << 20)
    for kind in (GPU, SYSTEM):
        _same_kind(final.kind(SUB, kind), load_resource_data(tmp_path).kind(SUB, kind))
