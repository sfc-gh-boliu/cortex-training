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

"""One out-of-range value in any job record never stops the TUI.

Year 1 is Go's zero time and 9999-12-31 a common "never" sentinel; either one
leaves Python's date range once converted to a zone on the other side of UTC.
JSON's 1e999 (and Infinity) reads as an infinite GPU count.
"""

import time
from datetime import datetime
from datetime import timezone

import pytest

from cortex_training.tui import format as log_format
from cortex_training.tui import views
from tests.r1_cross_cutting_support import FakeClient
from tests.r1_cross_cutting_support import NOW
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401
from tests.r1_cross_cutting_support import job
from tests.r1_cross_cutting_support import run
from tests.r1_cross_cutting_support import settle
from tests.r1_cross_cutting_support import wait_for

ZERO_TIME = "0001-01-01T00:00:00Z"
NEVER = "9999-12-31T23:59:59Z"


@pytest.fixture
def local_zone(monkeypatch):
    """Switch the process's local zone (``astimezone()`` reads it), and back afterwards."""

    def switch(name):
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield switch
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize("zone,raw", [("America/Los_Angeles", ZERO_TIME), ("Asia/Tokyo", NEVER),
                                      ("UTC", "0001-01-01T00:00:00+01:00"), ("UTC", "9999-12-31T23:59:59-01:00")])
def test_times_out_of_range_in_the_local_zone_show_a_dash(local_zone, zone, raw):
    local_zone(zone)
    assert log_format.format_created_at(raw) == ""
    assert log_format.format_job_row({"job_id": "j", "created_at": raw}).split(" │ ")[1].strip() == "—"
    assert views.fmt_ts(raw) == "—"
    views.fmt_ago(raw, NOW)  # a duration or a dash, never an exception
    views.summary_fields(job("j", created_at=raw, updated_at=raw, state_changed_at=raw), "", NOW)
    assert isinstance(log_format._created_epoch(raw), float)


def test_an_out_of_range_offset_is_not_a_time():
    assert views.parse_ts("0001-01-01T00:00:00+01:00") is None
    assert views.parse_ts("9999-12-31T23:59:59-01:00") is None
    assert log_format._created_epoch("0001-01-01T00:00:00+01:00") == float("-inf")


def test_times_in_range_still_parse_as_utc_and_sort():
    assert views.parse_ts("2026-10-14T12:00:00.123456789+02:00") == datetime(2026, 10, 14, 10, 0, 0, 123456,
                                                                           tzinfo=timezone.utc)
    jobs = [{"job_id": "a", "created_at": ZERO_TIME}, {"job_id": "b", "created_at": "2026-10-14T00:00:00Z"},
            {"job_id": "c", "created_at": "0001-01-01T00:00:00+01:00"}]
    assert [j["job_id"] for j in log_format.sort_jobs(jobs)] == ["b", "a", "c"]
    assert [j["job_id"] for j in views.sort_job_list(jobs)] == ["b", "a", "c"]


def test_a_time_ahead_of_now_is_not_shown_as_a_dash_ago():
    assert views.fmt_ago(NEVER, NOW) == "—"


@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles", "Asia/Tokyo"])
@pytest.mark.parametrize("raw", [ZERO_TIME, "0001-01-01T00:00:00.000000000Z", "1970-01-01T00:00:00Z", NEVER])
def test_a_time_that_was_never_set_shows_a_dash_and_no_age_in_every_zone(local_zone, zone, raw):
    """Go's zero time, the Unix epoch and a year-9999 "never" are placeholders: the preview's
    Last update is a dash with no "(739903d ago)" beside it, whatever the local zone."""
    local_zone(zone)
    assert views.parse_ts(raw) is None
    assert views.fmt_ts(raw) == "—" and views.fmt_ago(raw, NOW) == "—"
    assert log_format.format_created_at(raw) == ""
    record = job("j", "FAILED", updated_at=raw, state_changed_at=raw)
    fields = dict(views.summary_fields(record, "", NOW))
    assert fields["Last update"] == "—"
    assert "d " not in fields["Elapsed"]  # no duration measured from year 1 or 1970
    # Still sorted, and still listed: a job whose time is a placeholder is not hidden.
    assert views.within_retention({"job_id": "j", "created_at": raw}, NOW)


def test_a_real_time_keeps_its_age_beside_it():
    fields = dict(views.summary_fields(job("j", "FAILED", updated_at="2026-10-14T11:00:00Z"), "", NOW))
    assert fields["Last update"].endswith("  (1h 00m ago)")


@pytest.mark.parametrize("n_gpus", [float("inf"), float("-inf"), float("nan"), "1e999", "Infinity", None, "eight"])
def test_a_gpu_count_that_is_not_a_finite_number_counts_as_none(n_gpus):
    record = job("j")
    record["sub_jobs"][0]["training_config"]["n_gpus"] = n_gpus
    sj = record["sub_jobs"][0]
    assert views.requested_gpus(sj) == 0 and views.sub_job_gpus(sj) == 0
    assert views.total_gpus(record) == 0
    assert views.job_gpu_cell(record, "H200") == "—"
    assert views.gpu_request_text(record) == "—"
    assert views.sub_job_gpu_text(sj) == "—"


def test_the_list_and_the_preview_open_with_out_of_range_records(local_zone):
    local_zone("America/Los_Angeles")
    from textual.widgets import DataTable

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen
    from cortex_training.tui.screens import JobSummaryScreen

    zero = job("zero", "FAILED", updated_at=ZERO_TIME, state_changed_at=ZERO_TIME)
    never = job("never", "CANCELLED", created_at=NEVER)
    infinite = job("infinite")
    infinite["sub_jobs"][0]["training_config"]["n_gpus"] = float("inf")
    client = FakeClient([zero, never, infinite, job("fine")])

    async def go():
        app = CortexTrainingLogTUI(client, poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            assert set(screen._job_by_item.values()) == {"zero", "never", "infinite", "fine"}
            table = screen.query_one("#jobs", DataTable)
            for jid in ("zero", "never", "infinite"):
                row = next(key for key, value in screen._job_by_item.items() if value == jid)
                screen.set_focus(table)
                table.move_cursor(row=table.get_row_index(row), animate=False)
                await pilot.pause()
                await pilot.press("enter")
                assert await wait_for(pilot, lambda: isinstance(app.screen, JobSummaryScreen)
                                      and app.screen._fetched_at is not None)
                await pilot.press("b")
                assert await wait_for(pilot, lambda: app.screen is screen)
            assert app.is_running
            await settle(app, pilot)

    run(go())


def test_a_row_that_cannot_be_drawn_names_its_job_and_the_rest_still_show(monkeypatch, tmp_path):
    from textual.widgets import DataTable

    from cortex_training.tui import app as tui_app
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen

    real = views.job_type_label

    def odd(record):
        if views.job_id_of(record) == "odd":
            raise TypeError("a value no formatter expected")
        return real(record)

    monkeypatch.setattr(tui_app.views, "job_type_label", odd)
    client = FakeClient([job("odd"), job("fine")])

    async def go():
        app = CortexTrainingLogTUI(client, poll_interval=0.01, now=NOW)
        async with app.run_test(size=(160, 40)) as pilot:
            assert await wait_for(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            table = app.screen.query_one("#jobs", DataTable)
            rows = [" ".join(str(cell) for cell in table.get_row_at(i)) for i in range(table.row_count)]
            assert any("odd" in row and "can't show" in row for row in rows)
            fine = next(row for row in rows if "fine" in row)
            assert "training" in fine and "TEST_USER" in fine and "8 H200" in fine
            await settle(app, pilot)

    run(go())
    log = (tmp_path / "isolated-home" / ".cortex-training-errors.log").read_text()
    assert "job list row for odd" in log and "a value no formatter expected" in log
