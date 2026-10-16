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

"""No escape sequence from the service's data reaches the terminal, on any screen.

Every string field of a job and its sub-jobs carries an OSC 52
clipboard write, an OSC 0 title write, an OSC 8 link and CSI codes, and so do
the error messages the service and the stage send. Each screen is rendered and
its screenshot must hold no control character and parse as XML.
"""

import inspect
import re
import types
import xml.etree.ElementTree as ET

import pytest
import requests

from cortex_training.tui import views
from cortex_training.tui.untrusted import PrintableClient
from cortex_training.tui.untrusted import printable_client
from cortex_training.tui.untrusted import unwrapped
from tests.r1_cross_cutting_support import FakeClient
from tests.r1_cross_cutting_support import NOW
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401
from tests.r1_cross_cutting_support import job
from tests.r1_cross_cutting_support import run
from tests.r1_cross_cutting_support import settle
from tests.r1_cross_cutting_support import wait_for

# OSC 52 (clipboard) and OSC 8 ended by BEL, OSC 0 (title) by ST, CSI reverse video.
HOSTILE = "\x1b]52;c;QUFBQQ==\x07\x1b]0;LEAK\x1b\\\x1b[7mINV\x1b[27m\x1b]8;;https://example.invalid\x07L\x9b31m"
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _http_error(status, message):
    response = types.SimpleNamespace(status_code=status, reason="Internal Server Error",
                                     json=lambda: {"message": message}, text=f'{{"message": "{message}"}}')
    return requests.HTTPError(f"{status} Server Error", response=response)


def _hostile_job(job_id, status="RUNNING"):
    record = job(job_id, status, hardware="H200" + HOSTILE, submitted_by="USER" + HOSTILE,
                 owner_role="ROLE" + HOSTILE, reason="why" + HOSTILE, reason_details="more" + HOSTILE,
                 comment="named" + HOSTILE)
    sj = record["sub_jobs"][0]
    sj["model_name"] = "Qwen/Qwen3-8B" + HOSTILE
    sj["dtype"] = "bf16" + HOSTILE
    sj["training_config"]["optimizer"]["name"] = "AdamW" + HOSTILE
    sj["training_config"]["note" + HOSTILE] = "value" + HOSTILE
    return record


class _HostileClient(FakeClient):
    """j1 answers with hostile fields; GPU metrics and the tail fail with a hostile message."""

    def download_metrics(self, job_id, output_dir, *, resume=False, **_options):
        raise RuntimeError("stage refused" + HOSTILE)

    def tail_logs(self, job_id, *, cursor=None, max_lines=None, sub_job_id=None):
        raise _http_error(500, "tail refused" + HOSTILE)


def _client():
    sub_id = _hostile_job("j3", "FAILED")
    sub_id["sub_jobs"][0]["sub_job_id"] = "j3:training:0" + HOSTILE
    return _HostileClient([_hostile_job("j1"), _hostile_job("j2", "FAILED"), sub_id])


def _check(app, where):
    svg = app.export_screenshot()
    found = CONTROL.search(svg)
    assert found is None, f"{where}: control character {found.group()!r} reached the screen"
    ET.fromstring(svg)  # a screenshot with a raw ESC is not even valid XML
    assert CONTROL.search(app.sub_title or "") is None, f"{where}: subtitle {app.sub_title!r}"


def test_every_screen_draws_hostile_service_text_inert():
    from textual.widgets import DataTable

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen
    from cortex_training.tui.screens import JobDetailScreen
    from cortex_training.tui.screens import JobSummaryScreen

    async def open_job(app, pilot, list_screen, job_id):
        row = next(key for key, jid in list_screen._job_by_item.items() if jid == job_id)
        table = list_screen.query_one("#jobs", DataTable)
        list_screen.set_focus(table)
        table.move_cursor(row=table.get_row_index(row), animate=False)
        await pilot.pause()
        await pilot.press("enter")
        assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen)
                              and app.screen._fetched_at is not None)

    async def go():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(200, 50)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            list_screen = app.screen
            await pilot.press("e")  # every sub-job row too
            await pilot.pause()
            _check(app, "job list")
            for job_id in ("j1", "j2", "j3"):
                await open_job(app, pilot, list_screen, job_id)
                for _ in range(5):
                    await pilot.pause()
                _check(app, f"{job_id} preview")
                await pilot.press("l")
                assert await wait_for(pilot, lambda: isinstance(app.screen, JobDetailScreen) and app.screen.is_mounted)
                for tab in ("2", "3"):
                    await pilot.press(tab)
                    for _ in range(15):
                        await pilot.pause()
                    _check(app, f"{job_id} details tab {tab}")
                await pilot.press("b")
                assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and app.screen.is_mounted)
                await pilot.press("b")
                assert await wait_for(pilot, lambda: app.screen is list_screen)
            await settle(app, pilot)

    run(go())


def test_the_tuis_client_returns_printable_records_and_keeps_its_own_surface():
    client = _client()
    wrapped = printable_client(client)
    assert isinstance(wrapped, PrintableClient) and unwrapped(wrapped) is client
    assert printable_client(wrapped) is wrapped  # once
    record = wrapped.get_job("j1")
    text = repr(record)
    assert "\\x1b" not in text and "\\x07" not in text and "\\x9b" not in text
    assert record["sub_jobs"][0]["model_name"].startswith("Qwen/Qwen3-8B�]52;c;")
    assert any(key.startswith("note�") for key in record["sub_jobs"][0]["training_config"])
    assert wrapped.list_jobs()[0]["job_id"] == "j1"
    # Everything else is the client's own: attributes, other methods, and what it lacks.
    assert wrapped.database == "CORTEX_TRAINING_DB" and wrapped.tail_logs.__self__ is client
    assert not hasattr(wrapped, "list_jobs_page")
    wrapped._auto_create_database = False
    assert client._auto_create_database is False
    # The job list passes retry bounds only to a list_jobs that takes them.
    assert "max_retries" in inspect.signature(wrapped.list_jobs).parameters


def test_per_location_clients_and_account_pages_come_back_printable():
    class Base:
        database, schema = "CORTEX_TRAINING_DB", "PUBLIC"

        def list_jobs_page(self, **filters):
            return {"jobs": [{"job_id": "a", "database_name": "TEAM_DB", "schema_name": "EXP",
                              "hardware": "B200" + HOSTILE}], "next_page_token": "t+0/=="}

        def iter_jobs(self, **filters):
            yield {"job_id": "a", "hardware": "B200" + HOSTILE}

        def for_job(self, row):
            return FakeClient([_hostile_job("a")])

    wrapped = printable_client(Base())
    page = wrapped.list_jobs_page(status="active")
    assert page["jobs"][0]["hardware"] == "B200" + views.printable(HOSTILE)
    assert page["next_page_token"] == "t+0/=="  # an opaque token is kept as it is
    assert "\x1b" not in next(wrapped.iter_jobs())["hardware"]
    child = wrapped.for_job(page["jobs"][0])
    assert isinstance(child, PrintableClient)
    assert "\x1b" not in repr(child.get_job("a"))


def test_errors_are_described_on_one_printable_line():
    assert views.describe_error(_http_error(500, "line one\nline two" + HOSTILE)) == (
        "HTTP 500 Internal Server Error: line one line two" + views.collapse(HOSTILE))
    text = views.describe_error(RuntimeError("stage\r\nrefused" + HOSTILE))
    assert CONTROL.search(text) is None and "\n" not in text and text.startswith("RuntimeError: stage refused")


def test_printable_data_maps_every_string_and_leaves_other_values():
    data = {"a\x1b": ["x\x07", 1, 2.5, None, True, {"k": ("t\x1b",)}], 3: "ok"}
    assert views.printable_data(data) == {"a�": ["x�", 1, 2.5, None, True, {"k": ("t�",)}], 3: "ok"}
    plain = "already printable · 训练 ✅"
    assert views.printable_data(plain) is plain


def test_table_cells_and_the_subtitle_stay_on_one_printable_line():
    pytest.importorskip("textual")
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import _cell

    assert _cell("a\nb\x1b[7m").plain == "a b�[7m"
    # Spaces are the cell's own: a tree's indent, two in a quoted name.
    assert _cell('   └─ training #0').plain == "   └─ training #0"
    assert _cell('"my  db"."dev"').plain == '"my  db"."dev"'
    app = CortexTrainingLogTUI(FakeClient([]))
    app.sub_title = "job\nid \x1b]0;T\x07  ·  reason"
    assert app.sub_title == "job / id �]0;T�  ·  reason"


def test_the_job_lists_sub_rows_stay_indented_under_their_job():
    """Each sub-job row, and an UNKNOWN job's note row, starts three cells in, then its
    branch: the tree stands to the right of the job's ▾, never under it."""
    pytest.importorskip("textual")
    from textual.widgets import DataTable

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen

    two = job("j1")
    two["sub_jobs"].append({"sub_job_id": "j1:sampling:0", "job_type": "SAMPLING", "status": "RUNNING",
                            "model_name": "Qwen/Qwen3-8B", "inference_config": {"n_gpus": 2.0}})
    unknown = job("j2", "UNKNOWN", reason="Job status unavailable: Control Plane is unavailable")

    async def go():
        app = CortexTrainingLogTUI(FakeClient([two, unknown]), poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            table = screen.query_one("#jobs", DataTable)
            names = {key.value: table.get_row(key)[1].plain for key in table.rows}
            item = {jid: key for key, jid in screen._job_by_item.items()}
            assert names[item["j1"]].startswith("▾ ")
            assert names[f"sub-{item['j1']}-0"] == "   ├─ training #0"
            assert names[f"sub-{item['j1']}-1"] == "   └─ sampling #0"
            assert names[f"note-{item['j2']}"] == "   ├─ ? status unavailable"
            assert names[f"sub-{item['j2']}-0"] == "   └─ training #0"
            await settle(app, pilot)

    run(go())


BIDI_AND_INVISIBLE = "\u202e\u202d\u2066\u2069\u200b\u200e\u2060\ufeff\u061c"


def test_one_line_status_text_replaces_bidi_and_zero_width_characters_like_everything_else():
    from cortex_training.tui import format as log_format

    reason = "quota_filled\u202edelif_atouq\u200b x"
    assert log_format.one_line(reason) == views.collapse(reason) == "quota_filled�delif_atouq� x"
    assert log_format.one_line(BIDI_AND_INVISIBLE) == "�" * len(BIDI_AND_INVISIBLE)
    # Line breaks and tabs keep their one-row meaning; a zero-width joiner a script needs stays.
    assert log_format.one_line("a\r\nb\u2028c\td") == "a / b / c d"
    assert log_format.one_line("\u0915\u094d\u200d\u0937") == "\u0915\u094d\u200d\u0937"
    assert log_format.printable is views.printable  # one rule, one implementation


def test_the_logs_waiting_line_shows_a_reordering_reason_inert():
    pytest.importorskip("textual")
    from textual.widgets import Static

    from cortex_training.tui.app import CortexTrainingLogTUI

    pending = job("j1", "PENDING", reason="quota_filled\u202edelif_atouq",
                  reason_details="needs 8 GPUs\u2066\u200b more")
    pending["sub_jobs"][0]["status"] = "PENDING"

    async def go():
        app = CortexTrainingLogTUI(FakeClient([pending]), "j1", poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            assert await wait_for(pilot, lambda: getattr(app.screen, "_waiting", None) == "pending")
            bar = app.screen.query_one("#log-status", Static)
            line = str(bar.render())
            assert "quota_filled�delif_atouq — needs 8 GPUs�� more" in line  # "reason — details"
            assert not any(ch in line for ch in BIDI_AND_INVISIBLE)
            await settle(app, pilot)

    run(go())


@pytest.mark.parametrize("joined,apart", [
    ("\U0001F469\u200d\U0001F4BB", "\U0001F469\U0001F4BB"),  # technologist
    ("\U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466",
     "\U0001F468\U0001F469\U0001F467\U0001F466"),  # family
    ("\U0001F3F3\ufe0f\u200d\U0001F308", "\U0001F3F3\ufe0f\U0001F308"),  # rainbow flag
    ("\U0001F3C3\U0001F3FD\u200d\u2640\ufe0f", "\U0001F3C3\U0001F3FD\u2640\ufe0f"),
])
def test_emoji_joined_by_a_zero_width_joiner_are_drawn_side_by_side(joined, apart):
    """A terminal that draws a joined sequence as one glyph and one that draws its
    parts disagree on its width; drawn apart, every terminal gives it the width counted."""
    from rich.cells import cell_len

    assert views.printable(f"name {joined} x") == f"name {apart} x"
    if "\ufe0f" not in joined:  # (a variation selector is a width question of its own)
        assert cell_len(views.printable(joined)) == sum(cell_len(ch) for ch in views.printable(joined))
    # A joiner between letters, as some scripts write them, stays.
    kept = "\u0915\u094d\u200d\u0937 a\u200db"
    assert views.printable(kept) == kept
