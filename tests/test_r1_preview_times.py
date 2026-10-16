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

"""Preview times without a value or slightly ahead of this clock, and a comment
the Name row already shows whole."""

import asyncio
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import MagicMock

import pytest

from cortex_training.tui import views as v

NOW = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize("offset, text", [
    (-245, "4m 05s ago"),
    (0, "0s ago"),
    (5, "just now"),  # the service's clock a little ahead of this one
    (300, "just now"),  # up to views.CLOCK_SKEW_SECONDS
    (3600, "1h 00m ahead of this computer's clock"),
    (2 * 86400 + 3600, "2d 1h ahead of this computer's clock"),
])
def test_fmt_ago_reads_a_time_ahead_of_this_clock(offset, text):
    assert v.fmt_ago(_iso(NOW + timedelta(seconds=offset)), NOW) == text
    assert "— ago" not in v.fmt_ago(_iso(NOW + timedelta(seconds=offset)), NOW)


def test_fmt_ago_without_a_time():
    assert v.fmt_ago(None, NOW) == "—" and v.fmt_ago("not a time", NOW) == "—"


def _job(**extra):
    job = {"job_id": "fake-job-times", "status": "CANCELLED", "created_at": _iso(NOW - timedelta(hours=2)),
           "updated_at": _iso(NOW - timedelta(hours=1)), "state_changed_at": _iso(NOW - timedelta(hours=1)),
           "sub_jobs": [{"sub_job_id": "fake-job-times:training:0", "job_type": "TRAINING", "status": "CANCELLED",
                         "training_config": {"n_gpus": 8}}]}
    job.update(extra)
    return job


def test_a_missing_last_update_has_no_empty_parenthesis():
    job = _job()
    del job["updated_at"]
    assert dict(v.summary_fields(job, "", NOW))["Last update"] == "—"
    ahead = _job(updated_at=_iso(NOW + timedelta(seconds=5)))
    assert dict(v.summary_fields(ahead, "", NOW))["Last update"].endswith("  (just now)")


@pytest.mark.parametrize("comment, name, shown", [
    ("ablation: no warmup [/x]", "ablation: no warmup [/x]", False),  # the name is all of it
    ("first line\nsecond line", "first line …", True),  # more lines
    ("a long comment " * 10, v.one_line("a long comment " * 10, 40), True),  # the name is cut
    ("kept beside the alias", "my alias", True),  # a local name
    ("two  spaces", "two spaces", True),  # the name collapsed its spacing
])
def test_the_comment_row_only_adds_what_the_name_leaves_out(comment, name, shown):
    fields = dict(v.summary_fields(_job(comment=comment), name, NOW))
    assert ("Comment" in fields) == shown
    if shown:
        assert fields["Comment"] == v.comment_of({"comment": comment})


def test_an_unknown_preview_shows_its_comment_once_and_no_empty_times():
    pytest.importorskip("textual")
    from textual.widgets import Static

    from cortex_training import AccountJobListUnavailable
    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import JobListScreen
    from cortex_training.tui.screens import JobSummaryScreen

    job = {"job_id": "fake-job-unknown", "status": "UNKNOWN", "comment": "ablation: no warmup [/x]",
           "reason": "Job status unavailable: Control Plane is unavailable", "owner_role": "ML",
           "sub_jobs": [{"status": "UNKNOWN", "job_type": "TRAINING", "training_config": {"n_gpus": 8}}]}
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            for _ in range(300):
                await pilot.pause(0.01)
                if isinstance(app.screen, JobListScreen) and app.screen._job_by_item:
                    break
            app.push_screen(JobSummaryScreen(c, job["job_id"], job=job))
            for _ in range(300):
                await pilot.pause(0.01)
                if isinstance(app.screen, JobSummaryScreen) and c.get_job.called:
                    break
            for _ in range(5):
                await pilot.pause()
            text = str(app.screen.query_one("#preview", Static).render())
            assert text.count("ablation: no warmup [/x]") == 1, text
            assert not any(line.strip().startswith("Comment") for line in text.split("\n"))
            last = next(line for line in text.split("\n") if line.strip().startswith("Last update"))
            assert last.split("Last update", 1)[1].strip() == "—"
            assert "(—)" not in text
            app.workers.cancel_all()

    asyncio.run(run())
