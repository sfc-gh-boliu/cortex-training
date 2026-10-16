# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resources: every numeric field of a row is a metric, found in the data rather than listed."""

import pytest

from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import ResourceLoader
from cortex_training.tui.resource import is_metric_field
from cortex_training.tui.resource import load_resource_data
from cortex_training.tui.resource import metric_name
from cortex_training.tui.resource import metric_unit
from tests.v2_resource_support import SUB
from tests.v2_resource_support import append
from tests.v2_resource_support import gpu_row
from tests.v2_resource_support import scrapes
from tests.v2_resource_support import system_row
from tests.v2_resource_support import write


def _kind(root, rows, kind=GPU):
    write(root, rows)
    return load_resource_data(root).kind(SUB, kind)


def test_every_numeric_gpu_field_is_a_metric_util_and_memory_first_then_file_order(tmp_path):
    resource = _kind(tmp_path, [gpu_row(t, g) for t in scrapes(120) for g in range(2)])
    assert [m.key for m in resource.metrics] == [
        "gpu_util_pct", "fb_used_pct", "fb_used_bytes", "fb_free_bytes", "power_watts", "gpu_temp_c",
        "tensor_active_pct", "dram_active_pct"]
    assert [m.name for m in resource.metrics] == [
        "GPU util %", "GPU memory used %", "GPU memory used (GiB)", "GPU memory free (GiB)", "power (W)",
        "GPU temp (°C)", "tensor active %", "DRAM active %"]
    assert [m.unit for m in resource.metrics] == ["%", "%", "GiB", "GiB", "W", "°C", "%", "%"]
    assert [m.derived for m in resource.metrics] == [False, True] + [False] * 6
    view = resource.view("10m")
    assert set(view.charts) == set(view.stats) == {m.key for m in resource.metrics}


def test_every_numeric_system_field_is_a_metric_cpu_and_memory_first(tmp_path):
    resource = _kind(tmp_path, [system_row(t, n) for t in scrapes(120) for n in range(2)], SYSTEM)
    assert [m.key for m in resource.metrics] == [
        "cpu_util_pct", "mem_used_pct", "cpu_count", "mem_used_bytes", "mem_total_bytes"]
    assert [m.name for m in resource.metrics] == [
        "CPU util %", "memory used %", "CPU count", "memory used (GiB)", "memory total (GiB)"]


def test_a_new_numeric_field_gets_a_chart_with_no_code_change(tmp_path):
    rows = [gpu_row(t, g, nvlink_rx_bytes=float(g + 1) * 2**30, fan_speed_pct=40.0 + g, clock_mhz=1980)
            for t in scrapes(120) for g in range(2)]
    resource = _kind(tmp_path, rows)
    keys = [m.key for m in resource.metrics]
    assert keys[-3:] == ["nvlink_rx_bytes", "fan_speed_pct", "clock_mhz"]  # where they first appear: the row's end
    names = {m.key: (m.name, m.unit) for m in resource.metrics}
    assert names["nvlink_rx_bytes"] == ("nvlink rx (GiB)", "GiB")
    assert names["fan_speed_pct"] == ("fan speed %", "%")
    assert names["clock_mhz"] == ("clock mhz", "")
    view = resource.view("10m")
    assert view.chart("fan_speed_pct") and all(v == pytest.approx(40.5) for v in view.chart("fan_speed_pct"))
    assert view.stats["nvlink_rx_bytes"].max == pytest.approx(1.5)  # GiB, the mean of 1 and 2
    assert resource.view("10m", 1).stats["nvlink_rx_bytes"].max == pytest.approx(2.0)
    assert resource.view("10m", 1).stats["clock_mhz"].latest == 1980


def test_a_field_that_is_null_in_every_row_has_no_chart_until_one_row_measures_it(tmp_path):
    rows = [gpu_row(t, g) for t in scrapes(300) for g in range(2)]
    assert "sm_active_pct" not in [m.key for m in _kind(tmp_path / "null", rows).metrics]
    rows[-1]["sm_active_pct"] = 12.5  # the newest row measures it
    measured = _kind(tmp_path / "one", rows)
    assert "sm_active_pct" in [m.key for m in measured.metrics]
    assert "sm_occupancy_pct" not in [m.key for m in measured.metrics]
    chart = measured.view("10m").chart("sm_active_pct")
    assert [v for v in chart if v is not None] == [pytest.approx(12.5)]


def test_identity_bookkeeping_strings_and_bools_are_never_metrics(tmp_path):
    rows = [gpu_row(t, g, rack_index=3, pod_num=7, schema_version=2, retry_interval_s=1.5, healthy=True,
                    zone="zone-a", note=None) for t in scrapes(60) for g in range(2)]
    keys = {m.key for m in _kind(tmp_path, rows).metrics}
    for name in ("record_version", "worker_num", "local_gpu_index", "global_gpu_index", "seq", "seq_epoch",
                 "sample_interval_s", "sample_ts", "sub_job_id", "gpu_uuid", "rack_index", "pod_num",
                 "schema_version", "retry_interval_s", "healthy", "zone", "note"):
        assert name not in keys, name
    assert not any(is_metric_field(name) for name in ("seq", "kind", "x_index", "x_num", "x_version", "x_interval_s"))
    assert all(is_metric_field(name) for name in ("power_watts", "x_count", "sequence_pct", "interval_watts"))


def test_names_and_units_come_from_the_field_names():
    assert [metric_name(f) for f in ("power_watts", "gpu_temp_c", "tensor_active_pct", "fb_free_bytes", "cpu_count",
                                     "mem_total_bytes", "sm_occupancy_pct", "dram_active_pct")] == [
        "power (W)", "GPU temp (°C)", "tensor active %", "GPU memory free (GiB)", "CPU count",
        "memory total (GiB)", "SM occupancy %", "DRAM active %"]
    assert [metric_unit(f) for f in ("a_pct", "a_bytes", "a_watts", "a_c", "a_count", "a_cpu")] == [
        "%", "GiB", "W", "°C", "", ""]


def test_the_order_is_where_fields_first_appear_and_a_refresh_keeps_it(tmp_path):
    early = [gpu_row(t, 0) for t in scrapes(60)]
    for row in early:  # power is measured only from the second minute on
        del row["power_watts"]
    later = [gpu_row(t, 0) for t in scrapes(60, start=60)]
    path = write(tmp_path, early + later)
    loader = ResourceLoader(tmp_path)
    first = [m.key for m in loader.load().kind(SUB, GPU).metrics]
    assert first.index("power_watts") == len(first) - 1  # after every field the first rows had
    append(path, [gpu_row(t, 0, ecc_errors_total=0) for t in scrapes(60, start=120)], ["c0", "c1"])
    second = [m.key for m in loader.load().kind(SUB, GPU).metrics]
    assert second == first + ["ecc_errors_total"]
    assert [m.key for m in load_resource_data(tmp_path).kind(SUB, GPU).metrics] == second
