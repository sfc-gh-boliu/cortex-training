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

"""Long values don't push the preview's fields off the screen: the comment
gets the rows the rest leaves (at least a few), other values at most three
(each with a note that names the key), and e shows them whole."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from rich.cells import cell_len  # noqa: E402
from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import log_cache  # noqa: E402
from cortex_training.tui import views as v  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.app import JobListScreen  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402
from cortex_training.tui.untrusted import unwrapped  # noqa: E402

JOB = "fake-job-long"
LONG_COMMENT = "\n".join(f"line {i}: nightly eval of the support-chat checkpoints against the frozen March prompt "
                         f"set (1,200 prompts)" for i in range(1, 11))
FIELDS = ("Name", "Comment", "Job ID", "Status", "Model", "Type", "Owner", "GPUs", "Created", "Elapsed", "Last update")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(*, comment=LONG_COMMENT, model="Qwen/Qwen3-8B", owner="TEST_USER_A"):
    return {"job_id": JOB, "status": "CANCELLED", "reason": "cancelled_by_client", "comment": comment,
            "submitted_by": owner, "owner_role": "TRAINING_ROLE", "hardware": "H200",
            "created_at": "2026-10-14T17:39:00Z", "updated_at": "2026-10-14T18:39:00Z",
            "state_changed_at": "2026-10-14T18:39:00Z",
            "sub_jobs": [{"sub_job_id": f"{JOB}:training:0", "job_type": "TRAINING", "status": "CANCELLED",
                          "model_name": model, "training_config": {"n_gpus": 4}}]}


def _client(job):
    c = MagicMock()
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    return c


async def _wait(pilot, predicate, tries=300):
    for _ in range(tries):
        await pilot.pause(0.01)
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


async def _open(app, pilot, job):
    c = app._client  # the app's printable wrapper; unwrapped(c) is the mock
    assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
    app.push_screen(JobSummaryScreen(c, JOB, job=job))
    assert await _wait(pilot, lambda: isinstance(app.screen, JobSummaryScreen) and unwrapped(c).get_job.called)
    for _ in range(10):
        await pilot.pause()
    return app.screen


def _lines(screen) -> list:
    return str(screen.query_one("#preview", Static).render()).split("\n")


def _field(screen, label) -> list:
    """The rows of one field: its first row and the continuation rows under it."""
    lines = _lines(screen)
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(label))
    end = start + 1
    while end < len(lines) and lines[end].startswith(" " * 15):
        end += 1
    return lines[start:end]


def _on_screen(screen, label) -> bool:
    """The field's first row is on the screen, above the footer, with the preview at its top."""
    preview = screen.query_one("#preview", Static)
    row = next(i for i, line in enumerate(_lines(screen)) if line.strip().startswith(label))
    return preview.region.y + row < screen.size.height - 1


def _footer_keys(screen) -> dict:
    return {binding.action: binding.description for _ns, binding, enabled, _tip in screen.active_bindings.values()
            if enabled and binding.show}


@pytest.mark.parametrize("size", [(100, 30), (80, 24)])
def test_a_long_comment_leaves_every_field_on_the_screen(size):
    job = _job()

    async def run():
        app = CortexTrainingLogTUI(_client(job), poll_interval=0.01)
        async with app.run_test(size=size) as pilot:
            p = await _open(app, pilot, job)
            assert all(_on_screen(p, label) for label in FIELDS), _lines(p)
            rows = _field(p, "Comment")
            assert 3 <= len(rows) <= 6
            assert rows[-2].endswith("…") and "e shows the whole comment)" in rows[-1]
            assert rows[-1].strip().startswith("(+")
            assert all(cell_len(row) <= p.query_one("#preview", Static).size.width for row in rows)
            assert _footer_keys(p).get("expand_values") == "Full text"
            # e shows all of it; e again goes back.
            await pilot.press("e")
            assert await _wait(pilot, lambda: "e shows the whole comment" not in "\n".join(_lines(p)))
            whole = " ".join(" ".join(_field(p, "Comment")).split()).removeprefix("Comment ")
            assert whole == " ".join(LONG_COMMENT.split())
            assert _footer_keys(p).get("collapse_values") == "Less text" and "expand_values" not in _footer_keys(p)
            await pilot.press("e")
            assert await _wait(pilot, lambda: "e shows the whole comment" in "\n".join(_lines(p)))
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_comment_that_fits_is_shown_whole_with_no_key():
    job = _job()

    async def run():
        app = CortexTrainingLogTUI(_client(job), poll_interval=0.01)
        async with app.run_test(size=(140, 45)) as pilot:
            p = await _open(app, pilot, job)
            rows = _field(p, "Comment")
            assert " ".join(" ".join(rows).split()).removeprefix("Comment ") == " ".join(LONG_COMMENT.split())
            assert not any("e shows" in row for row in rows)
            keys = _footer_keys(p)
            assert "expand_values" not in keys and "collapse_values" not in keys
            await _settle(app, pilot)

    asyncio.run(run())


def test_other_long_values_stop_at_three_rows():
    job = _job(comment="short", model="org/" + "M" * 5000, owner="U" * 500)

    async def run():
        app = CortexTrainingLogTUI(_client(job), poll_interval=0.01)
        async with app.run_test(size=(140, 42)) as pilot:
            p = await _open(app, pilot, job)
            for label in ("Model", "Owner"):
                rows = _field(p, label)
                # Three rows of the value, then the note that says how to see the rest.
                assert len(rows) == 4 and rows[2].endswith("…"), rows
                assert rows[3].strip().startswith("(+") and rows[3].strip().endswith("e shows the whole value)")
            assert all(_on_screen(p, label) for label in ("Status", "GPUs", "Last update"))
            assert _footer_keys(p).get("expand_values") == "Full text"
            await pilot.press("e")
            assert await _wait(pilot, lambda: len(_field(p, "Model")) > 3)
            assert "".join("".join(_field(p, "Model")).split()).removeprefix("Model") == "org/" + "M" * 5000
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_cut_value_says_which_key_shows_the_rest():
    # On a narrow screen the footer has no room for e: the value's own note names it.
    model, owner = "org/" + "M" * 5000, "U" * 500
    job = _job(comment="short", model=model, owner=owner)

    async def run():
        app = CortexTrainingLogTUI(_client(job), poll_interval=0.01)
        async with app.run_test(size=(80, 24)) as pilot:
            p = await _open(app, pilot, job)
            width = p.query_one("#preview", Static).size.width
            for label, value in (("Model", model), ("Owner", f"{owner} (TRAINING_ROLE)")):
                rows = _field(p, label)
                assert len(rows) == 4 and rows[2].endswith("…"), rows
                total = len(p._value_lines(value))
                assert rows[3].strip() == f"(+{total - 3} rows · e shows the whole value)", rows[3]
                assert all(cell_len(row) <= width for row in rows)
            await pilot.press("e")
            assert await _wait(pilot, lambda: "e shows the whole value" not in "\n".join(_lines(p)))
            assert "".join("".join(_field(p, "Model")).split()).removeprefix("Model") == model
            await pilot.press("e")
            assert await _wait(pilot, lambda: "\n".join(_lines(p)).count("e shows the whole value") == 2)
            await _settle(app, pilot)

    asyncio.run(run())


def test_zalgo_text_keeps_a_few_combining_marks():
    zalgo = "z" + "".join(chr(0x0301 + i % 8) for i in range(480)) + "algo"
    assert v.printable(zalgo) == "ź̂̃̄algo"
    assert v.one_line(zalgo) == "ź̂̃̄algo"
    # Real text keeps its marks: one or two on a letter.
    for text in ("Tiếng Việt", "é̂ two marks", "שָׁלוֹם"):
        assert v.printable(text) == text
