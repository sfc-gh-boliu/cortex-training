# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Every key that works can be found on screen, at 80 columns too.

The footer leads with Back, Quit, the tab keys (one entry) and ``?``, which
opens the keys panel listing every key, the ones the footer hides too. Esc
closes that panel before it goes back. In the details the preview's letters
open the same tabs, except where a tab gives its letter its own meaning.
"""

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable  # noqa: E402

from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobDetailScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from tests.r1_cross_cutting_support import FakeClient  # noqa: E402
from tests.r1_cross_cutting_support import NOW  # noqa: E402
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: E402,F401
from tests.r1_cross_cutting_support import job  # noqa: E402
from tests.r1_cross_cutting_support import run  # noqa: E402
from tests.r1_cross_cutting_support import settle  # noqa: E402
from tests.r1_cross_cutting_support import wait_for  # noqa: E402


def _client():
    return FakeClient([job("j1")])


def _footer(app) -> list:
    """(key, description, fully on screen) of each footer entry, left to right."""
    from textual.widgets._footer import FooterKey

    keys = sorted(app.screen.query(FooterKey), key=lambda w: w.region.x)
    palette = [w for w in keys if "-command-palette" in w.classes]
    edge = palette[0].region.x if palette else app.size.width  # "^p palette" sits at the right
    return [(w.key, w.description, w.region.right <= edge) for w in keys if w not in palette]


def _shown(app) -> list:
    return [key for key, _description, visible in _footer(app) if visible]


def _tab(app) -> str:
    return app.screen._active_tab


async def _to_details(app, pilot):
    assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    table = app.screen.query_one("#jobs", DataTable)
    app.screen.set_focus(table)
    table.move_cursor(row=1, animate=False)
    await pilot.pause()
    await pilot.press("enter")
    assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.is_mounted)
    await pilot.press("l")
    assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._sources_ready)


@pytest.mark.parametrize("width", [80, 100, 140])
def test_back_quit_tabs_and_keys_lead_every_footer(width):
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(width, 30)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            await pilot.pause()
            assert _shown(app)[:2] == ["q", "question_mark"]
            descriptions = [description for _key, description, _ in _footer(app)]
            assert "Clear search" not in descriptions and "Expand/collapse all" in descriptions
            await _to_details(app, pilot)
            for tab in ("2", "3"):
                await pilot.press(tab)
                await wait_for(pilot, lambda: app.screen._active_tab != "", tries=5)
                for _ in range(3):
                    await pilot.pause()
                shown = _shown(app)
                # The two tab keys are one entry ("23 Tabs"), after Back and Quit, then "? Keys".
                assert shown[:5] == ["b", "q", "2", "3", "question_mark"], (tab, shown)
                assert [d for _k, d, _ in _footer(app)].count("Logs") == 0  # no separate "2 Logs"
            await settle(app, pilot)

    run(go())


def test_question_mark_opens_the_keys_panel_and_esc_closes_it_before_going_back():
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            list_screen = app.screen
            await pilot.press("question_mark")
            assert await wait_for(pilot, lambda: list_screen.query("HelpPanel"))
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: not list_screen.query("HelpPanel"))
            await pilot.press("question_mark")
            assert await wait_for(pilot, lambda: list_screen.query("HelpPanel"))
            await pilot.press("question_mark")  # ? closes it too
            assert await wait_for(pilot, lambda: not list_screen.query("HelpPanel"))
            await _to_details(app, pilot)
            detail = app.screen
            await pilot.press("question_mark")
            assert await wait_for(pilot, lambda: detail.query("HelpPanel"))
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: not detail.query("HelpPanel"))
            for _ in range(5):
                await pilot.pause()
            assert app.screen is detail  # Esc closed the panel; it didn't go back
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
            preview = app.screen
            await pilot.press("question_mark")
            assert await wait_for(pilot, lambda: preview.query("HelpPanel"))
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: not preview.query("HelpPanel"))
            assert app.screen is preview
            await settle(app, pilot)

    run(go())


def test_the_previews_letters_open_the_same_tabs_in_the_details():
    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            await _to_details(app, pilot)
            notes = []
            app.notify = lambda message, **kwargs: notes.append(message)
            assert _tab(app) == "logs"
            await pilot.press("c")  # on Logs, c copies a selection: there is none, and it says so
            assert await wait_for(pilot, lambda: notes)
            assert notes[-1].startswith("Nothing selected") and _tab(app) == "logs"
            for key, tab in (("g", "resource"), ("l", "logs"), ("g", "resource")):
                await pilot.press(key)
                assert await wait_for(pilot, lambda: _tab(app) == tab), (key, _tab(app))
            # This job reports no GPU samples, so Resources shows no chart for m to switch:
            # m does nothing (see the test below for one chart shown).
            await pilot.press("m")
            for _ in range(5):
                await pilot.pause()
            assert _tab(app) == "resource"
            await settle(app, pilot)

    run(go())


# ─── m on Resources, and the keys panel beside the log pane ──────────────────

GIB = 1024**3


class _GpuClient(FakeClient):
    """j1's training sub-job reports 8 GPUs every 5 s for the last 20 minutes."""

    def download_metrics(self, job_id, output_dir, *, resume=False, **_options):
        import json
        from datetime import datetime
        from datetime import timedelta
        from datetime import timezone
        from pathlib import Path

        newest = datetime.now(timezone.utc) - timedelta(seconds=5)
        rows = []
        for step in range(240, -1, -1):
            ts = (newest - timedelta(seconds=5 * step)).isoformat().replace("+00:00", "Z")
            for gpu in range(8):
                used = (40 + gpu) * GIB
                rows.append({"record_version": 2, "sub_job_id": "j1:training:0", "sample_ts": ts, "worker_num": 0,
                             "local_gpu_index": gpu, "global_gpu_index": gpu, "gpu_util_pct": 80.0 + gpu,
                             "fb_used_bytes": float(used), "fb_free_bytes": float(141 * GIB - used),
                             "sm_active_pct": None, "tensor_active_pct": 40.0, "dram_active_pct": None,
                             "sample_interval_s": 5.0})
        path = Path(output_dir) / "j1:training:0" / "gpu.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}))
        return []


async def _on_resources(app, pilot):
    from textual.widgets import Static

    assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen._sources_ready)
    await pilot.press("3")
    detail = app.screen
    assert await wait_for(pilot, lambda: detail._resource_loaded and not detail._resource_loading
                          and detail.query_one("#resource-table", DataTable).row_count > 0, tries=400)
    for _ in range(3):
        await pilot.pause()
    return detail, list(detail.query(".resource-chart-box")), detail.query_one("#resource-keys", Static)


@pytest.mark.parametrize("size, one_chart", [((80, 24), True), ((140, 42), False), ((80, 50), False)])
def test_m_switches_the_chart_only_while_resources_shows_one(size, one_chart):
    async def go():
        app = CortexTrainingLogTUI(_GpuClient([job("j1")]), "j1", poll_interval=0.01, now=NOW)
        async with app.run_test(size=size) as pilot:
            detail, (busy, memory), keys = await _on_resources(app, pilot)
            assert (busy.display, memory.display) == ((True, False) if one_chart else (True, True))
            assert ("m busy/memory chart" in str(keys.render())) is one_chart
            await pilot.press("m")
            for _ in range(5):
                await pilot.pause()
            if one_chart:  # the other chart, on the same tab
                assert _tab(app) == "resource" and (busy.display, memory.display) == (False, True)
            else:  # both charts already show: m does nothing
                assert _tab(app) == "resource" and (busy.display, memory.display) == (True, True)
            await settle(app, pilot)

    run(go())


def _pane_rows(detail) -> list:
    """The log pane's rows as drawn, each cut to the pane's visible width."""
    from textual.widgets import Log

    pane = detail.query_one("#logview", Log)
    width = pane.scrollable_content_region.width
    return [str(line) for line in pane.lines], width


class _LongLines(FakeClient):
    """j1's training sub-job logs long lines, each with its number at the very end."""

    def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        if cursor:
            return {"entries": [], "next_cursor": cursor, "eof": False}
        words = " ".join(f"word{i:03d}" for i in range(25))
        return {"entries": [{"_raw": f"2026-10-14T11:59:{n:02d}.000000000Z {words} end-of-line-{n}"}
                            for n in range(5)], "next_cursor": "5", "eof": False}


def test_the_keys_panel_beside_the_log_pane_rewraps_its_lines_instead_of_cutting_them():
    """? opens a panel on the right that narrows the log pane: the lines are wrapped
    to the narrower pane, so each still ends on screen, and again once it closes."""

    async def go():
        app = CortexTrainingLogTUI(_LongLines([job("j1")]), "j1", poll_interval=0.01, now=NOW)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                  and any("end-of-line-4" in line for line in app.screen._shown_lines))
            detail = app.screen
            for _ in range(3):
                await pilot.pause()

            def every_line_ends_in_view() -> bool:
                rows, width = _pane_rows(detail)
                ends = [row for row in rows if "end-of-line-" in row]
                return (len(ends) == 5 and all(len(row.rstrip()) <= width for row in rows)
                        and all(f"end-of-line-{n}" in "".join(rows) for n in range(5)))

            assert every_line_ends_in_view()
            wide = len(_pane_rows(detail)[0])
            await pilot.press("question_mark")
            assert await wait_for(pilot, lambda: detail.query("HelpPanel"))
            assert await wait_for(pilot, lambda: _pane_rows(detail)[1] < 80 and every_line_ends_in_view())
            assert len(_pane_rows(detail)[0]) > wide  # the same lines, on more rows
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: not detail.query("HelpPanel"))
            assert await wait_for(pilot, lambda: len(_pane_rows(detail)[0]) == wide and every_line_ends_in_view())
            await settle(app, pilot)

    run(go())


class _ManyLongLines(FakeClient):
    """j1's training sub-job logs 80 long lines, each starting with its number."""

    def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        if cursor:
            return {"entries": [], "next_cursor": cursor, "eof": False}
        words = " ".join(f"word{i:03d}" for i in range(25))
        return {"entries": [{"_raw": f"2026-10-14T11:{n // 60:02d}:{n % 60:02d}.000000000Z n{n:03d} {words}"}
                            for n in range(80)], "next_cursor": "80", "eof": False}


def test_a_paused_reader_keeps_their_line_when_the_keys_panel_opens_and_closes():
    """p, scroll back to a line, then ?: the lines are wrapped again for the narrower
    pane, and the line that was at the top of the view stays at the top."""
    from textual.widgets import Log

    async def go():
        app = CortexTrainingLogTUI(_ManyLongLines([job("j1")]), "j1", poll_interval=0.01, now=NOW)
        async with app.run_test(size=(140, 42)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen)
                                  and any(" n079 " in line for line in app.screen._shown_lines))
            detail = app.screen
            pane = detail.query_one("#logview", Log)
            await pilot.press("p")
            assert await wait_for(pilot, lambda: detail._paused)
            pane.scroll_to(y=40, animate=False)
            await pilot.pause()

            def top() -> str:
                """The number of the log line the view's top row belongs to (any of its rows)."""
                import re

                rows = [str(line) for line in pane.lines]
                start = int(pane.scroll_offset.y)
                while start > 0 and not rows[start].startswith("training:0 │"):
                    start -= 1  # a later row of a wrapped line: back to its first
                end = start + 1
                while end < len(rows) and not rows[end].startswith("training:0 │"):
                    end += 1
                found = re.search(r"n(\d{3}) ", "".join(rows[start:end]))
                return found.group(1) if found else ""

            reading = top()
            assert reading.isdigit() and 5 < int(reading) < 60, reading
            await pilot.press("question_mark")
            assert await wait_for(pilot, lambda: detail.query("HelpPanel") and pane.scrollable_content_region.width < 80)
            assert await wait_for(pilot, lambda: top() == reading), (top(), reading)
            await pilot.press("escape")
            assert await wait_for(pilot, lambda: not detail.query("HelpPanel"))
            assert await wait_for(pilot, lambda: pane.scrollable_content_region.width > 80 and top() == reading)
            assert detail._paused
            await settle(app, pilot)

    run(go())
