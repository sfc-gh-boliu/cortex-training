# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resources text: chart time axes that never repeat a label, stats that name the unit, titles."""

import re
from datetime import timedelta

import pytest

from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import LOADING_OLDER
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import KindView
from cortex_training.tui.resource import Metric
from cortex_training.tui.resource import load_resource_data
from cortex_training.tui.resource import metric_stats_text
from cortex_training.tui.resource import metric_title
from cortex_training.tui.resource import render_metric_chart
from tests.v2_resource_support import START
from tests.v2_resource_support import SUB
from tests.v2_resource_support import gpu_row
from tests.v2_resource_support import scrapes
from tests.v2_resource_support import system_row
from tests.v2_resource_support import write

CLOCK = re.compile(r"\d\d:\d\d(?::\d\d)?")
UTIL = Metric(key="gpu_util_pct", name="GPU util %", unit="%", scale=100.0)


def _view(span_s: float, columns: int) -> KindView:
    return KindView(kind=GPU, name="10m", start=START, end=START + timedelta(seconds=span_s), unit=None, units=1,
                    label="", charts={UTIL.key: (50.0,) * columns})


def _labels(span_s: float, width: int = 60) -> list[str]:
    return CLOCK.findall(render_metric_chart(_view(span_s, 40), UTIL, width, height=2).splitlines()[-1])


@pytest.mark.parametrize("span_s", [5, 30, 59, 61, 90, 119])
def test_a_span_under_two_minutes_labels_its_times_with_seconds_and_never_repeats_one(span_s):
    labels = _labels(span_s)
    assert len(labels) >= 2 and len(set(labels)) == len(labels), labels
    assert all(len(label) == 8 for label in labels)  # HH:MM:SS


@pytest.mark.parametrize("span_s", [120, 125, 179, 600, 3600, 5 * 3600])
def test_longer_spans_keep_minutes_and_their_labels_differ(span_s):
    labels = _labels(span_s)
    assert len(labels) == 3 and len(set(labels)) == 3 and all(len(label) == 5 for label in labels), labels


def test_a_span_shorter_than_a_second_shows_one_time_not_three_equal_ones():
    for span_s in (0.0, 0.4, 0.9):
        assert len(_labels(span_s)) <= 2 and len(set(_labels(span_s))) == len(_labels(span_s))
    assert len(_labels(0.0)) == 1


def test_a_narrow_chart_too_short_for_two_second_labels_keeps_the_newest_time():
    labels = CLOCK.findall(render_metric_chart(_view(45, 10), UTIL, 20, height=2).splitlines()[-1])
    assert labels == [(START + timedelta(seconds=45)).astimezone().strftime("%H:%M")]


def _gpus(tmp_path):
    write(tmp_path, [gpu_row(t, g, util=40.0 + 20 * g) for t in scrapes(1200) for g in range(2)]
          + [system_row(t, n, cpu=2.0 + n) for t in scrapes(1200) for n in range(2)])
    return load_resource_data(tmp_path)


def test_the_narrow_stats_name_the_gpu_picked_where_the_mean_had_its_gpus(tmp_path):
    gpu = _gpus(tmp_path).kind(SUB, GPU)
    metrics = gpu.metrics
    assert metric_stats_text(gpu.view("10m"), gpu.view("all"), metrics, width=50).splitlines()[0].startswith(
        "mean of 2 GPUs")
    narrow = metric_stats_text(gpu.view("10m", 1), gpu.view("all", 1), metrics, width=50)
    assert narrow.splitlines()[0].startswith("GPU 1")
    assert "GPU util" in narrow and "60%" in narrow


def test_every_metric_gets_a_stats_row_in_its_unit_and_the_scopes_name_the_unit(tmp_path):
    data = _gpus(tmp_path)
    gpu = data.kind(SUB, GPU)
    wide = metric_stats_text(gpu.view("10m"), gpu.view("all"), gpu.metrics, width=150)
    lines = wide.splitlines()
    assert "last 10m · mean of 2 GPUs" in lines[0] and "whole job · mean of 2 GPUs" in lines[0]
    assert lines[1].split() == ["min", "avg", "max", "latest"] * 2
    assert [line.split("  ")[0] for line in lines[2:]] == [m.name for m in gpu.metrics]
    power = next(line for line in lines if line.startswith("power (W)"))
    assert power.split()[2:6] == ["250", "W", "250", "W"]  # 100 + 3 × 40 % on GPU 0, 100 + 3 × 60 on GPU 1: 250 W mean
    assert all(len(line) <= 150 for line in lines)


def test_narrow_metric_stats_go_a_row_per_scope_and_name_the_node_picked(tmp_path):
    system = _gpus(tmp_path).kind(SUB, SYSTEM)
    narrow = metric_stats_text(system.view("10m", 1), system.view("all", 1), system.metrics, width=56)
    lines = narrow.splitlines()
    assert lines[0].startswith("node 2") and lines[0].split()[-4:] == ["min", "avg", "max", "latest"]
    assert len(lines) == 1 + 2 * len(system.metrics)
    assert all(len(line) <= 56 for line in lines)
    assert lines[1].startswith("CPU util") and "3.0%" in lines[1]
    assert "last 10m" in lines[1] and "whole job" in lines[2]
    # Narrower, the scopes go by their keys before the metrics' names are cut.
    keys = metric_stats_text(system.view("10m", 1), system.view("all", 1), system.metrics, width=48).splitlines()
    assert all(len(line) <= 48 for line in keys)
    assert keys[1].startswith("CPU util %") and keys[1].split()[3] == "10m" and keys[2].split()[0] == "job"
    assert [line.split("  ")[0] for line in keys[1::2]] == [m.short for m in system.metrics]
    assert len({m.short for m in system.metrics}) == len(system.metrics)  # memory % and GiB still differ


def test_metric_stats_say_what_is_still_loading(tmp_path):
    gpu = _gpus(tmp_path).kind(SUB, GPU)
    text = metric_stats_text(gpu.view("10m"), gpu.view("all"), gpu.metrics, width=150, partial=True)
    assert text.count(LOADING_OLDER) == len(gpu.metrics)  # the whole job's column only
    whole = metric_stats_text(gpu.view("all"), None, gpu.metrics, width=150, partial=True)
    assert whole.count(LOADING_OLDER) == len(gpu.metrics) and "last" not in whole


def test_chart_titles_name_the_unit_or_the_mean_and_shorten_to_fit(tmp_path):
    gpu = _gpus(tmp_path).kind(SUB, GPU)
    power = gpu.metric("power_watts")
    title = metric_title(gpu.view("30m"), f"training #0 · {power.name}")
    assert title == "training #0 · power (W) · last 30m · mean of 2 GPUs"
    name = f"training #0 · {power.name}"
    assert metric_title(gpu.view("30m", 1), name) == "training #0 · power (W) · last 30m · GPU 1"
    assert metric_title(gpu.view("30m"), name, width=40) == "training #0 · power (W) · last 30m"
    cut = metric_title(gpu.view("30m"), f"training #0 · {power.name}", width=20)
    assert len(cut) <= 20 and cut.endswith("…")


def test_a_metric_chart_fits_any_width(tmp_path):
    gpu = _gpus(tmp_path).kind(SUB, GPU)
    for metric in gpu.metrics:
        for width in (1, 5, 8, 12, 40, 91):
            lines = render_metric_chart(gpu.view("10m"), metric, width, height=4).splitlines()
            assert all(len(line) <= width for line in lines), (metric.key, width)
