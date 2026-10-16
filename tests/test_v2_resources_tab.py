# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Resources tab's panel in the running app: views, units, keys, empty states, footer and key help."""

import asyncio
import re
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402
from textual.widgets._footer import FooterKey  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from tests.test_resources_tab import _client  # noqa: E402
from tests.test_resources_tab import _panel  # noqa: E402
from tests.test_resources_tab import _titles  # noqa: E402
from tests.test_resources_tab import _units  # noqa: E402
from tests.v2_resource_support import GIB  # noqa: E402
from tests.v2_resource_support import gpu_row  # noqa: E402
from tests.v2_resource_support import scrapes  # noqa: E402
from tests.v2_resource_support import system_row  # noqa: E402
from tests.v2_resource_support import write  # noqa: E402

TRAINING, SAMPLING = "7:training:0", "7:sampling:0"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _fresh(rows, newest_age: float = 20.0):
    """``rows`` moved in time so the newest is ``newest_age`` seconds old (a running sub-job's are recent)."""
    def ts(row):
        return datetime.fromisoformat(row["sample_ts"][:26] + "+00:00")

    shift = datetime.now(timezone.utc) - timedelta(seconds=newest_age) - max(ts(row) for row in rows)
    for row in rows:
        row["sample_ts"] = (ts(row) + shift).isoformat().replace("+00:00", "Z")
    return rows


def _telemetry(sub, *, gpus=8, nodes=2, minutes=20, system=True, gpu_rows=True):
    """``gpus`` GPUs on 4-GPU nodes and each node's system row, every 5 s for ``minutes``.

    GPU g runs at 60 + 4g % busy; node n at 3 + 2n % CPU, holding 100 + 50n of 860 GiB.
    """
    rows = []
    for t in scrapes(minutes * 60):
        if system:
            rows += [system_row(t, node, 3.0 + 2 * node, used=(100 + 50 * node) * GIB, sub=sub)
                     for node in range(nodes)]
        if gpu_rows:
            rows += [gpu_row(t, gpu, 60.0 + 4 * gpu, sub=sub) for gpu in range(gpus)]
    return _fresh(rows)


def _download(c, *, training=None, sampling=None):
    def download(job_id, output_dir, *, resume=False, **_options):
        for sub, rows in ((TRAINING, training), (SAMPLING, sampling)):
            if rows:
                write(output_dir, rows, sub=sub)
        return []

    c.download_metrics.side_effect = download


def _text(screen, selector):
    return str(screen.query_one(selector, Static).render())


def _scopes(screen):
    return " ".join(_text(screen, "#resource-stats").splitlines()[0].split())


def _names(screen):
    """The stats block's metric names, in order (the wide layout)."""
    return [re.split(r"\s{2,}", line)[0] for line in _text(screen, "#resource-stats").splitlines()[2:]]


async def _wait(pilot, predicate, tries=400):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _open(app, pilot, *, units=True):
    assert await _wait(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._current_source)
    await pilot.press("3")
    screen = app.screen
    assert await _wait(pilot, lambda: screen._resource_loaded and not screen._resource_loading
                       and (not units or len(_units(screen)) > 1))
    for _ in range(3):
        await pilot.pause()
    return screen


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


async def _press(pilot, *keys):
    for key in keys:
        await pilot.press(key)
        await pilot.pause()
    await pilot.pause()


def test_the_panel_lists_the_views_then_every_gpu_and_moving_in_it_redraws_every_chart():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            panel = _panel(screen)
            assert panel[:2] == ["● GPU Resources", "○ System Resources"]
            assert _units(screen) == ["ALL 74%"] + [f"GPU {g} {60 + 4 * g}%" for g in range(8)]
            assert screen.focused is screen.query_one("#resource-list")
            # Every GPU metric of the rows: util and memory first, then the file's order.
            assert _names(screen) == ["GPU util %", "GPU memory used %", "GPU memory used (GiB)",
                                      "GPU memory free (GiB)", "power (W)", "GPU temp (°C)", "tensor active %",
                                      "DRAM active %"]
            assert len(_titles(screen)) == 8 and _scopes(screen).startswith("last 10m · mean of 8 GPUs")
            # The non-% axes run to a nice ceiling of the sub-job's highest sample: power peaks at 364 W.
            assert "500 W ┤" in _text(screen, "#resource-charts")
            await _press(pilot, "down", "down", "down")  # ALL, GPU 0, GPU 1, then GPU 2
            assert screen._resource_unit == 2
            assert _scopes(screen) == "last 10m · GPU 2 whole job · GPU 2"
            assert all(title.endswith(" · GPU 2") for title in _titles(screen))
            util = next(line for line in _text(screen, "#resource-stats").splitlines() if line.startswith("GPU util"))
            assert util.split()[-8:-4] == ["68%"] * 4
            assert "500 W ┤" in _text(screen, "#resource-charts")  # one GPU keeps the sub-job's axis
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_system_view_lists_the_nodes_with_their_cpu_and_charts_every_system_metric():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            await _press(pilot, "up")  # from ALL to System Resources, just above it
            assert screen._resource_kind == "system" and screen._resource_unit is None
            assert _panel(screen)[:2] == ["○ GPU Resources", "● System Resources"]
            assert _units(screen) == ["ALL 4.0%", "node 1 3.0%", "node 2 5.0%"]
            status = _text(screen, "#resource-status")
            assert "nodes: 2 reporting  ·  88 CPUs  ·  860 GiB each" in status.splitlines()[0]
            assert "GPUs:" not in status and "H200" not in status
            # CPU util and the derived memory % first, then every other system metric.
            assert _names(screen) == ["CPU util %", "memory used %", "CPU count", "memory used (GiB)",
                                      "memory total (GiB)"]
            assert _scopes(screen).startswith("last 10m · mean of 2 nodes")
            assert _titles(screen)[0] == "training #0 · CPU util % · last 10m · mean of 2 nodes"
            await _press(pilot, "down", "down", "down")  # ALL, node 1, then node 2
            assert screen._resource_unit == 1 and _scopes(screen).startswith("last 10m · node 2")
            memory = next(line for line in _text(screen, "#resource-stats").splitlines()
                          if line.startswith("memory used %"))
            assert memory.split()[-8:-4] == ["17%"] * 4  # 150 of 860 GiB
            assert "↑↓ node   a all" in _text(screen, "#resource-keys")
            await _press(pilot, "a")
            assert screen._resource_unit is None and _scopes(screen).startswith("last 10m · mean of 2 nodes")
            await _settle(app, pilot)

    asyncio.run(run())


def test_down_on_gpu_resources_goes_to_its_units_and_up_from_all_reaches_system_resources():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            await _press(pilot, "up", "up")  # System Resources, then GPU Resources
            assert screen._resource_kind == "gpu" and screen._resource_on_view
            assert _units(screen)[0] == "ALL 74%"  # GPU Resources lists the GPUs under it
            await _press(pilot, "down")  # straight to its ALL, past System Resources
            assert screen._resource_kind == "gpu" and not screen._resource_on_view
            assert screen._resource_unit is None
            await _press(pilot, "down")
            assert screen._resource_kind == "gpu" and screen._resource_unit == 0
            await _settle(app, pilot)

    asyncio.run(run())


def test_lt_and_gt_keep_the_view_and_go_back_to_all():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            await _press(pilot, "up", "down", "down", "down")  # System Resources, ALL, node 1, node 2
            assert (screen._resource_kind, screen._resource_unit) == ("system", 1)
            await _press(pilot, ">")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("sampling #0"))
            assert (screen._resource_kind, screen._resource_unit) == ("system", None)
            assert _units(screen) == ["ALL 3.0%", "node 1 3.0%"]
            assert "↑↓ node" not in _text(screen, "#resource-keys")  # one node: nothing to move between
            assert "nodes: 1 reporting" in _text(screen, "#resource-status")
            await _press(pilot, "<")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("training #0"))
            assert (screen._resource_kind, screen._resource_unit) == ("system", None)
            await _settle(app, pilot)

    asyncio.run(run())


def test_w_keeps_the_unit_and_a_click_on_an_entry_shows_it():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            items = [item for item, _label in screen._resource_rows if item.display]
            await pilot.click(items[_panel(screen).index(next(t for t in _panel(screen) if t.startswith("GPU 5 ")))])
            assert await _wait(pilot, lambda: screen._resource_unit == 5)
            assert _scopes(screen).startswith("last 10m · GPU 5")
            await _press(pilot, "w", "w")
            assert screen._resource_range == "1h" and _scopes(screen).startswith("last 1h · GPU 5")
            await pilot.click(items[1])  # System Resources
            assert await _wait(pilot, lambda: screen._resource_kind == "system")
            assert screen._resource_unit is None and _scopes(screen).startswith("last 1h · mean of 2 nodes")
            await _settle(app, pilot)

    asyncio.run(run())


def test_without_system_rows_system_resources_reads_none_and_says_so_quietly():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING, system=False), sampling=_telemetry(SAMPLING, gpus=4, system=False))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            assert _panel(screen)[:2] == ["● GPU Resources", "○ System Resources (none)"]
            label = next(label for item, label in screen._resource_rows if "(none)" in str(label.render()))
            assert [span.style for span in label.render().spans] == ["dim"]
            await _press(pilot, "up")
            assert screen._resource_kind == "system"
            assert _text(screen, "#resource-status").splitlines() == [
                "training #0 (1 of 2 · < > switch)", "CPU and memory aren't collected for this sub-job"]
            assert _text(screen, "#resource-stats") == "" and _text(screen, "#resource-charts") == ""
            assert _units(screen) == []
            assert "↑↓" not in _text(screen, "#resource-keys") and "a all" not in _text(screen, "#resource-keys")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_sub_job_with_system_rows_and_no_gpu_rows_opens_on_the_system_view():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING, gpu_rows=False), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            assert screen._resource_kind == "system"
            assert _panel(screen)[:2] == ["○ GPU Resources (none)", "● System Resources"]
            assert _units(screen) == ["ALL 4.0%", "node 1 3.0%", "node 2 5.0%"]
            # The next sub-job has GPU rows: the GPU view, picked in no panel, comes back.
            await _press(pilot, ">")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("sampling #0"))
            assert screen._resource_kind == "gpu"
            # Choosing GPU Resources on a sub-job without GPU rows says why it is empty.
            await _press(pilot, "<")
            assert await _wait(pilot, lambda: _text(screen, "#resource-status").startswith("training #0"))
            assert screen._resource_kind == "system"
            await _press(pilot, "up", "up")
            assert screen._resource_kind == "gpu"
            assert "no GPU metrics for this sub-job" in _text(screen, "#resource-status")
            await _settle(app, pilot)

    asyncio.run(run())


@pytest.mark.parametrize("width", [160, 120, 100, 80])
def test_the_footer_and_the_key_help_fit_whole_at_every_width(width):
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(width, 40)) as pilot:
            screen = await _open(app, pilot)
            await _press(pilot, "down")  # GPU 0: a shows
            keys = screen.query_one("#resource-keys", Static)
            for state in ("gpu", "system node"):
                if state == "system node":
                    await _press(pilot, "up", "up", "down", "down")  # System Resources, ALL, node 1
                    assert (screen._resource_kind, screen._resource_unit) == ("system", 0)
                text = _text(screen, "#resource-keys")
                assert "\n" not in text and len(text) <= keys.content_size.width, (state, text)
                # Never a bare key at the end: a whole entry, or "…" after one.
                assert re.search(r"(window|refresh|sub-job|GPU|node|all|…)$", text), text
                if width >= 100:
                    assert text.endswith("   a all"), text
                # The footer: < and > as one entry after w Window, a only while a unit shows; every
                # key it shows is whole, ahead of the palette key.
                footer = screen.query_one("#footer")
                palette = next(child for child in footer.children if child.has_class("-command-palette"))
                shown = [child for child in footer.query(FooterKey) if child.display and child is not palette
                         and child.parent.display]
                assert all(key.region.right <= palette.region.x for key in shown), state
                labels = [key.key_display for key in shown]
                assert labels.index("w") < labels.index("<") < labels.index(">")
                if width >= 100:
                    assert "a" in labels
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_key_help_cuts_at_whole_entries_and_ends_in_an_ellipsis():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(160, 46)) as pilot:
            screen = await _open(app, pilot)
            await _press(pilot, "down")
            full = screen._resource_keys(0)
            assert full == "w window 10m · 30m · 1h · 6h · all   r refresh   < > sub-job   ↑↓ GPU   a all"
            assert screen._resource_keys(len(full)) == full
            assert screen._resource_keys(len(full) - 1) == "w window   r refresh   < > sub-job   ↑↓ GPU   a all"
            assert screen._resource_keys(40) == "w window   r refresh   < > sub-job   …"
            assert screen._resource_keys(30) == "w window   r refresh   …"
            assert screen._resource_keys(14) == "w window   …"
            assert screen._resource_keys(6) == "w win…"
            for width in range(6, len(full) + 2):
                text = screen._resource_keys(width)
                assert len(text) <= width and not re.search(r"(^|   )(a|<|>|↑↓|w|r)$", text), (width, text)
            await _press(pilot, "a")  # no unit picked: no a, and with one sub-job no < >
            assert "a all" not in screen._resource_keys(0)
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_tab_draws_four_rows_of_bars_and_two_on_a_terminal_under_thirty_rows():
    c = _client(n_gpus=8)
    _download(c, training=_telemetry(TRAINING), sampling=_telemetry(SAMPLING, gpus=4, nodes=1))

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 30)) as pilot:
            screen = await _open(app, pilot)
            charts = _text(screen, "#resource-charts")
            assert charts.count("100% ┤") == 4 and charts.count(" 50% ┤") == 4  # the four % metrics
            lines = charts.splitlines()
            top = next(i for i, line in enumerate(lines) if "100% ┤" in line)
            assert " 50% ┤" in lines[top + 2] and "└" in lines[top + 4]  # 4 rows: the 50% row is the third
            await pilot.resize_terminal(120, 29)
            assert await _wait(pilot, lambda: " 50% ┤" in _text(screen, "#resource-charts").splitlines()[
                next(i for i, line in enumerate(_text(screen, "#resource-charts").splitlines()) if "100% ┤" in line)
                + 1])
            await _settle(app, pilot)

    asyncio.run(run())
