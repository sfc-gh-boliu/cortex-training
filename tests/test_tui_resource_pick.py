# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Resources tab's picks: < and > show another sub-job, the panel one GPU (a: all of them again)."""

import asyncio

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.resource import GPU  # noqa: E402
from cortex_training.tui.resource import load_resource_data  # noqa: E402
from cortex_training.tui.resource import render_metric_chart  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from tests.test_resources_tab import _client  # noqa: E402
from tests.test_resources_tab import _titles  # noqa: E402
from tests.test_resources_tab import _units  # noqa: E402
from tests.test_resources_tab import _write_metrics  # noqa: E402

TRAINING, SAMPLING = "7:training:0", "7:sampling:0"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _both(c):
    """Telemetry for both sub-jobs: training on 2 GPUs, sampling on 3."""

    def download(job_id, output_dir, *, resume=False, **_options):
        _write_metrics(output_dir, TRAINING, gpus=2)
        _write_metrics(output_dir, SAMPLING, gpus=3)
        return []

    c.download_metrics.side_effect = download


def _text(screen, selector):
    return str(screen.query_one(selector, Static).render())


def _scopes(screen):
    """The stats block's first line: what its two scopes cover."""
    return " ".join(_text(screen, "#resource-stats").splitlines()[0].split())


def _busy(screen):
    """The GPU util row's window min, avg, max, latest."""
    line = next(line for line in _text(screen, "#resource-stats").splitlines() if line.startswith("GPU util %"))
    return line.split()[-8:-4]


def _shown_keys(screen):
    """The footer's keys as it shows them (``<``, not ``less_than_sign``)."""
    return {ab.binding.key_display or key: ab.binding.description
            for key, ab in screen.active_bindings.items() if ab.binding.show and ab.enabled}


async def _wait(pilot, predicate, tries=400):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _open(app, pilot):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
    await pilot.press("3")
    assert await _wait(pilot, lambda: app.screen._resource_loaded and not app.screen._resource_loading
                       and len(_units(app.screen)) > 1)
    await pilot.pause()
    return app.screen


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


def test_lt_and_gt_switch_the_sub_job_and_every_part_of_the_tab_names_it():
    c = _client()
    _both(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            assert _text(screen, "#resource-status").startswith("training #0 (1 of 2 · < > switch)  ·  last 10m")
            assert all(title.startswith("training #0 · ") for title in _titles(screen))
            assert "mean of 2 GPUs" in _scopes(screen)
            keys = _shown_keys(screen)
            assert keys.get("<") == "Prev sub-job" and keys.get(">") == "Next sub-job"
            assert "< > sub-job" in _text(screen, "#resource-keys")

            await pilot.press(">")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("sampling #0 (2 of 2"))
            assert all(title.startswith("sampling #0 · ") for title in _titles(screen))
            assert "mean of 3 GPUs" in _scopes(screen)
            assert _units(screen) == ["ALL 90%", "GPU 0 80%", "GPU 1 90%", "GPU 2 100%"]  # its 3 GPUs
            # The Logs tab keeps its own selection.
            assert screen._current_source == TRAINING

            await pilot.press(">")  # round again to the first
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("training #0 (1 of 2"))
            await pilot.press("<")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("sampling #0 (2 of 2"))

            # Off Resources, < and > do nothing and leave the footer.
            await pilot.press("2")
            assert await _wait(pilot, lambda: screen._active_tab == "logs")
            assert "<" not in _shown_keys(screen) and ">" not in _shown_keys(screen)
            await _settle(app, pilot)

    asyncio.run(run())


def test_one_sub_job_has_nothing_to_switch_to():
    c = _client()
    _both(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", sub_job_id=TRAINING, poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            assert _text(screen, "#resource-status").startswith("training #0  ·  last 10m")
            assert "<" not in _shown_keys(screen)
            assert "sub-job" not in _text(screen, "#resource-keys")
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_panel_shows_the_gpu_highlighted_at_once_and_a_goes_back_to_all():
    c = _client()
    _both(c)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            # GPU 0 runs 80% busy in the last 10 minutes, GPU 1 90%: the mean is 85%.
            assert _busy(screen) == ["85%"] * 4 and screen._resource_unit is None
            assert "a" not in _shown_keys(screen) and "a all" not in _text(screen, "#resource-keys")
            assert "↑↓ GPU" in _text(screen, "#resource-keys")
            # Moving the highlight from ALL to a GPU shows that GPU at once.
            await pilot.press("down")
            assert await _wait(pilot, lambda: screen._resource_unit == 0)
            assert _scopes(screen) == "last 10m · GPU 0 whole job · GPU 0"
            assert _busy(screen) == ["80%"] * 4
            assert _titles(screen)[:2] == ["training #0 · GPU util % · last 10m · GPU 0",
                                           "training #0 · GPU memory used % · last 10m · GPU 0"]
            assert all(title.endswith(" · GPU 0") for title in _titles(screen))
            assert _shown_keys(screen).get("a") == "All GPUs" and "a all" in _text(screen, "#resource-keys")

            await pilot.press("down")
            assert await _wait(pilot, lambda: screen._resource_unit == 1)
            assert _busy(screen) == ["90%"] * 4 and _scopes(screen).startswith("last 10m · GPU 1")
            await pilot.press("up")
            assert await _wait(pilot, lambda: screen._resource_unit == 0)
            await pilot.press("w")  # a new window keeps the GPU
            assert await _wait(pilot, lambda: screen._resource_range == "30m")
            assert _scopes(screen).startswith("last 30m · GPU 0")

            await pilot.press("a")
            assert await _wait(pilot, lambda: screen._resource_unit is None)
            assert "mean of 2 GPUs" in _scopes(screen) and "a" not in _shown_keys(screen)
            panel = screen.query_one("#resource-list")
            assert screen._resource_entries[panel.index] == ("unit", None)  # the highlight is on ALL
            await pilot.press("down")  # and moves from there
            assert await _wait(pilot, lambda: screen._resource_unit == 0)

            # Another sub-job has other GPUs: switching shows the mean across them.
            await pilot.press(">")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("sampling #0"))
            assert screen._resource_unit is None and "mean of 3 GPUs" in _scopes(screen)
            assert screen._resource_entries[panel.index] == ("unit", None)
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_gpus_view_has_that_gpus_series_stats_and_name(tmp_path):
    _write_metrics(tmp_path, TRAINING, gpus=3)
    sub_job = load_resource_data(tmp_path).kind(TRAINING, GPU)
    for name in ("10m", "all"):
        view = sub_job.view(name)
        assert view.units == 3 and view.label == "mean of 3 GPUs"
        for gpu in range(3):
            one = sub_job.view(name, gpu)
            assert one is not None and one.units == 1 and one.label == f"GPU {gpu}" and one.unit == gpu
            # GPU g runs 60 + 10g % busy, 20 points more in the last 10 minutes.
            hot = 80.0 + 10 * gpu
            busy = one.stats["gpu_util_pct"]
            assert round(busy.latest) == hot and round(busy.max) == hot
            if name == "10m":
                assert round(busy.min) == hot and round(busy.avg) == hot
            else:
                assert round(busy.min) == hot - 20
            # Its own chart, on the mean's time axis: never another GPU's readings.
            chart = one.chart("gpu_util_pct")
            assert len(chart) == len(view.chart("gpu_util_pct"))
            assert all(value is None or hot - 20 - 1e-6 <= value <= hot + 1e-6 for value in chart)
        assert sub_job.view(name, 7) is None  # not one of its GPUs


def test_every_chart_of_two_rows_or_more_labels_100_50_and_0(tmp_path):
    _write_metrics(tmp_path, TRAINING, gpus=2)
    sub_job = load_resource_data(tmp_path).kind(TRAINING, GPU)
    view, util = sub_job.view("10m"), sub_job.metric("gpu_util_pct")
    for height in (2, 3, 4, 8):
        lines = render_metric_chart(view, util, 60, height=height).splitlines()
        axis = [line[:6] for line in lines[:height + 1]]
        assert axis[0] == "100% ┤" and axis[-1] == "  0% └"
        assert axis.count(" 50% ┤") == 1, (height, axis)
        if height % 2 == 0:  # the row whose top is 50%
            assert axis.index(" 50% ┤") == height // 2
    assert " 50% ┤" not in render_metric_chart(view, util, 60, height=1)
