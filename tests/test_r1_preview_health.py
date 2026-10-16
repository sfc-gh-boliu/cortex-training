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

"""The preview's health line: a line for every status, its own text for each
waiting state, and the service's reason shown one way everywhere."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from cortex_training.tui.screens import ENDED_STYLE  # noqa: E402
from cortex_training.tui.screens import STATUS_STYLE  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.screens import status_text  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))


def _health(status, reason=None, details=None) -> str:
    job = {"job_id": "7", "status": status,
           "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "TRAINING", "status": status}]}
    if reason is not None:
        job["reason"] = reason
    if details is not None:
        job["reason_details"] = details
    return JobSummaryScreen(MagicMock(), "7", job=job)._health().plain


@pytest.mark.parametrize("status, reason, details, line", [
    # Ended: the service's reason one way for every state ...
    ("FAILED", "sub_job_failed", None,
     "  ✖ Job failed  reason: sub_job_failed  — check Logs (l) for the failing sub-job"),
    ("FAILED", None, None, "  ✖ Job failed  reason: not reported  — check Logs (l) for the failing sub-job"),
    ("CANCELLED", "cancelled_by_client", None, "  ■ Cancelled  reason: cancelled_by_client"),
    ("JOB_STATE_CANCELLED", "cancelled_by_client", None, "  ■ Cancelled  reason: cancelled_by_client"),
    ("TERMINATED", "idle_timeout", "no request for 30 minutes",
     "  ■ Terminated by the platform  reason: idle_timeout — no request for 30 minutes"),
    ("CANCELLED", "", "cancelled from the console", "  ■ Cancelled  cancelled from the console"),
    # ... including the states that finished on their own, which had no line.
    ("SUCCEEDED", None, None, "  ■ Completed"),
    ("DONE", None, None, "  ■ Completed"),
    ("SUCCEEDED", "completed", None, "  ■ Completed  reason: completed"),
    # Waiting: each state says what it is waiting for, unless the service says more.
    ("PENDING", None, None, "  ◔ Pending — waiting to be placed on GPUs"),
    ("PENDING", "quota_filled", None, "  ◔ Pending  reason: quota_filled — waiting to be placed on GPUs"),
    ("PENDING", "quota_filled", "needs 8 GPUs; 64 currently active",
     "  ◔ Pending  reason: quota_filled — needs 8 GPUs; 64 currently active"),
    ("PLACING", None, None, "  ◔ Placing — waiting for its GPUs to come up"),
    ("INITIALIZING", None, None, "  ◔ Initializing — placed on GPUs; loading the model and starting the engines"),
    ("QUEUED", None, None, "  ◔ Queued — waiting to be placed on GPUs"),
    # A status this client doesn't know shows the service's reason, if any.
    ("PAUSED", "maintenance", None, "  reason: maintenance"),
    ("PAUSED", None, None, ""),
])
def test_every_status_gets_its_line(status, reason, details, line):
    assert _health(status, reason, details) == line


def test_placing_and_initializing_read_differently():
    # Not only the state's name: what each is waiting for.
    assert _health("PLACING").split(" — ", 1)[1] != _health("INITIALIZING").split(" — ", 1)[1]


@pytest.mark.parametrize("status", ["SUCCEEDED", "COMPLETED", "DONE"])
def test_finished_statuses_are_styled_like_other_ended_ones(status):
    assert STATUS_STYLE[status.lower()] == STATUS_STYLE["cancelled"] == ENDED_STYLE
    assert str(status_text(status).style) == ENDED_STYLE


def test_the_ended_style_is_one_both_rich_and_textual_draw():
    # The table's cells are Rich Text and the preview's lines Textual Content: a name
    # only Rich knows (grey62) would draw in the default color in the latter.
    from rich.style import Style as RichStyle
    from textual.style import Style as TextualStyle

    assert RichStyle.parse(ENDED_STYLE).color.get_truecolor() == (158, 158, 158)  # Rich's grey62
    assert TextualStyle.parse(ENDED_STYLE).foreground.rgb == (158, 158, 158)
    for style in set(STATUS_STYLE.values()):
        TextualStyle.parse(style)
