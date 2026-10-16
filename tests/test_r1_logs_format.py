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

"""How the Logs pane draws a line: escape sequences never reach the terminal,
rows are cut by terminal cells, JSON log records read as ``time LEVEL logger:
message``, and the level key counts a record that carries an error as ERROR."""

import json
from datetime import datetime
from datetime import timezone

import pytest

from cortex_training.tui import format as log_format
from cortex_training.tui.format import display_line
from cortex_training.tui.format import entry_at_level
from cortex_training.tui.format import entry_clock
from cortex_training.tui.format import lines_after
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import wrap_lines
from cortex_training.tui.format import wrap_log_line

rich_cells = pytest.importorskip("rich.cells")
cell_len = rich_cells.cell_len

STAMP = "2026-10-15T08:06:35.886109202Z"
# A Ray worker line as the live service sends it: the actor prefix in cyan, then a JSON record.
RAY_LINE = (f"{STAMP} \x1b[36m(TrainingWorker pid=1199, ip=10.0.0.5)\x1b[0m "
            '{"ts": "2026-10-15T08:06:35.811105+00:00", "level": "INFO", "logger": "trainer", '
            '"msg": "step 3 done\\nnext \\u001b[31mred\\u001b[0m", "host": "h-1", "job_id": "JOB:training:0"}')
HOSTILE = [
    "\x1b]52;c;aGFja2Vk\x07 OSC 52 clipboard write; \x1b]0;window title\x07 OSC 0 title write",
    "\x1b[2J\x1b[H screen clear \x1b[5A up \x1b[?1049h alt screen",
    "\x1b]8;;https://example.invalid\x1b\\a link\x1b]8;;\x1b\\ and \x1bP+q544e\x1b\\ a DCS",
    "an unterminated OSC \x1b]0;title and the rest",
    "a lone ESC at the end \x1b",
    "C1 controls: \x9b31m and \x9d0;title\x9c",
    "bidi ‮evil‬ and a zero-width​space",
    "progress  10%|#  | 1/10\rprogress 100%|##| 10/10",
    "tab\tseparated\tvalues, a NUL:\x00 a BEL:\x07",
]


def _local(day_clock_utc: str) -> str:
    moment = datetime.fromisoformat(day_clock_utc).replace(tzinfo=timezone.utc)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M:%S")


# ─── R1-05: no escape sequence or control character reaches the terminal ───

@pytest.mark.parametrize("line", HOSTILE)
def test_a_displayed_line_has_no_escape_or_control_character(line):
    for text in (display_line(line), display_line(f"{STAMP} {line}"), safe_text(line)):
        assert not any(ch in text for ch in "\x1b\x07\x00\r\t\x08\x9b\x9d\x9c‮‬​"), repr(text)
        assert all(ch == "\n" or ch.isprintable() for ch in text), repr(text)


def test_escape_sequences_are_removed_and_their_text_kept():
    assert display_line(HOSTILE[0]) == " OSC 52 clipboard write;  OSC 0 title write"
    assert display_line(HOSTILE[2]) == "a link and  a DCS"
    assert display_line(HOSTILE[3]) == "an unterminated OSC "  # what the terminal would have swallowed
    assert display_line(HOSTILE[4]) == "a lone ESC at the end �"
    assert display_line("[bold red]rich markup[/] stays literal") == "[bold red]rich markup[/] stays literal"


def test_a_ray_line_loses_its_colour_codes_and_a_record_its_escaped_ones():
    out = display_line(RAY_LINE)
    assert "\x1b" not in out
    assert out == (f"{_local('2026-10-15T08:06:35')} (TrainingWorker pid=1199, ip=10.0.0.5) INFO  trainer: "
                   "step 3 done\nnext red")


def test_a_progress_bar_shows_its_last_state_and_keeps_its_timestamp():
    out = display_line(f"{STAMP} {HOSTILE[7]}")
    assert out == f"{_local('2026-10-15T08:06:35')} progress 100%|##| 10/10"
    assert safe_text("first\r\nsecond\r") == "first\nsecond"


# ─── R1-59: a JSON record reads as a log line, in local time ───

def test_a_saved_console_record_without_a_stamp_uses_its_own_time():
    record = {"ts": "2026-10-13T16:46:51.654402+00:00", "level": "ERROR", "logger": "mgr",
              "msg": "shutdown failed\nTraceback (most recent call last):\n  File \"x.py\", line 5"}
    out = display_line(json.dumps(record))
    assert out.split("\n") == [f"{_local('2026-10-13T16:46:51')} ERROR mgr: shutdown failed",
                               "Traceback (most recent call last):", '  File "x.py", line 5']


def test_a_record_with_an_exception_field_shows_it_under_the_message():
    record = {"level": "ERROR", "logger": "worker", "msg": "step failed", "exc_info": "Traceback\n  boom"}
    assert display_line(f"{STAMP} {json.dumps(record)}").split("\n")[1:] == ["    Traceback", "      boom"]


def test_lines_that_are_not_records_are_left_as_they_are():
    assert display_line("plain text") == "plain text"
    assert display_line('{"no": "message field"}') == '{"no": "message field"}'
    assert display_line('{"msg": "cut short') == '{"msg": "cut short'  # a truncated record
    assert display_line(f"{STAMP} plain") == f"{_local('2026-10-15T08:06:35')} plain"


def test_the_entry_clock_is_the_lines_own_local_time():
    assert entry_clock({"_raw": f"{STAMP} x"}) == _local("2026-10-15T08:06:35")[11:]
    assert entry_clock({"_raw": "no stamp"}) == ""
    assert entry_clock({"ts": "2026-10-15T08:06:35Z", "msg": "x"}) == _local("2026-10-15T08:06:35")[11:]


# ─── R1-54 / R1-95: rows cut by cells; breaks drop their space; continuation rows line up ───

@pytest.mark.parametrize("text", [
    "wide: " + "训练数据" * 40 + " END",
    "emoji: " + "\U0001F680" * 60 + " END",
    "mixed 训练 data " * 30,
    "zwj \U0001F469‍\U0001F4BB " * 30,
])
@pytest.mark.parametrize("width", [7, 30, 88])
def test_wide_text_wraps_inside_the_pane(text, width):
    rows = wrap_log_line(text, width)
    assert all(cell_len(row) <= width for row in rows), [(cell_len(r), r) for r in rows]
    assert "".join(rows).replace(" ", "") == text.replace(" ", "")  # nothing lost


def test_a_row_break_drops_the_space_and_later_rows_carry_the_gutter():
    rows = wrap_log_line("alpha beta gamma delta epsilon", 20, first="t:0 │ ", rest="    │ ")
    assert rows == ["t:0 │ alpha beta", "    │ gamma delta", "    │ epsilon"]
    assert wrap_log_line("one\n  indented", 40, first="t:0 │ ", rest="    │ ") == ["t:0 │ one", "    │   indented"]
    assert all(not row.split("│ ", 1)[1].startswith(" ") for row in rows)


def test_a_line_too_long_for_the_pane_keeps_its_first_and_last_rows():
    rows = wrap_log_line("x" * 995 + "THE END", 10, max_rows=5, first="t │ ", rest="  │ ")
    assert len(rows) == 5
    assert rows[:2] == ["t │ xxxxxx", "  │ xxxxxx"]
    total = len(wrap_log_line("x" * 995 + "THE END", 10, first="t │ ", rest="  │ "))  # 167 rows of 6 cells
    assert rows[2] == f"  │ … {total - 4}"  # the count, in the 6 cells a row has here
    assert "".join(row[4:] for row in rows[-2:]).endswith("THE END")  # the end of the line stays
    rows = wrap_log_line("x" * 9995 + "THE END", 80, max_rows=5, first="t │ ", rest="  │ ")
    total = len(wrap_log_line("x" * 9995 + "THE END", 80, first="t │ ", rest="  │ "))
    assert rows[2] == f"  │ … {total - 4} more row(s) of this line (s saves the whole log)"  # room for all of it


# The pane's gutter under a sub-job tag, as the Logs tab draws it.
GUTTER = ("training:0 │ ", "           │ ")


@pytest.mark.parametrize("width", [40, 50, 60, 66, 70, 88, 140])
def test_the_row_that_stands_for_a_long_lines_middle_fits_the_pane(width):
    # A 5,000-character token (a hostile or JSON line) at the widths the pane can have.
    rows = wrap_log_line(display_line("… long-token " + "x" * 5000), width, first=GUTTER[0], rest=GUTTER[1],
                         max_rows=20)
    assert len(rows) == 20
    assert all(cell_len(row) <= width for row in rows), [(cell_len(row), row) for row in rows]
    note = rows[9]
    assert note.startswith(GUTTER[1] + "… ") and "more row(s)" in note  # what it stands for, and how much
    hidden = int(note[len(GUTTER[1]) + 2:].split(" ", 1)[0].replace(",", ""))
    whole = wrap_log_line(display_line("… long-token " + "x" * 5000), width, first=GUTTER[0], rest=GUTTER[1])
    assert hidden == len(whole) - 19  # one row in place of the ones left out
    if width >= 70:
        assert note.endswith("more row(s) of this line (s saves the whole log)")
    traceback = "\n".join(["Traceback (most recent call last):", *[f'  File "f{i}.py", line {i}' for i in range(40)],
                           "RuntimeError: CUDA out of memory"])
    rows = wrap_log_line(traceback, 80, max_rows=9)
    assert rows[0].startswith("Traceback") and rows[-1] == "RuntimeError: CUDA out of memory"
    assert len(rows) == 9


def test_wrap_lines_keeps_the_newest_lines_that_fit():
    lines = [f"{i:04d} bbbb cccc dddd eeee" for i in range(10)]  # two rows each at 20 cells
    chunks, displays, cut = wrap_lines(lines, 20, max_lines=100, max_rows=7)
    assert cut and [line for line, _, _ in chunks] == lines[-3:]  # 7 rows hold the last 3
    assert [rows for _, rows, _ in chunks][-1] == ["0009 bbbb cccc dddd", "eeee"]
    chunks, _, cut = wrap_lines(lines, 20, max_lines=2, max_rows=100)
    assert cut and [line for line, _, _ in chunks] == lines[-2:]
    assert set(displays) >= {line for line, _, _ in chunks}


# ─── R1-95: a record that carries an error counts as ERROR ───

def test_an_info_record_that_carries_an_error_passes_the_error_level():
    record = {"level": "INFO", "logger": "runtime.job_metrics", "msg": "job_metrics",
              "error": "RuntimeError: loading the model failed"}
    assert entry_at_level(record, "ERROR")
    assert entry_at_level({"_raw": f"{STAMP} {json.dumps(record)}"}, "ERROR")
    quiet = {"level": "INFO", "msg": "fine", "error": None, "exc_info": ""}
    assert not entry_at_level(quiet, "ERROR")
    assert not entry_at_level({"_raw": f"{STAMP} {json.dumps(quiet)}"}, "ERROR")
    assert not entry_at_level({"_raw": f'{STAMP} {{"level": "INFO", "msg": "x"}}'}, "WARNING")


# ─── R1-17: the live tail's lines past a lagging saved console ───

def test_lines_after_lines_the_tail_up_with_the_consoles_end():
    live = [f"{STAMP} step {i}" for i in range(10)]
    console = [f"step {i}" for i in range(7)]  # a saved console's lines have no timestamp
    assert lines_after(console[-3:], live) == live[7:]
    assert lines_after([f"step {i}" for i in range(10)][-3:], live) == []  # caught up
    assert lines_after(["something else"], live) is None  # can't be lined up
    repeated = [f"{STAMP} tick"] * 3 + [f"{STAMP} step 1", f"{STAMP} tick", f"{STAMP} after"]
    assert lines_after(["step 1", "tick"], repeated) == [f"{STAMP} after"]


def test_the_safe_text_helpers_do_not_import_textual():
    assert "textual" not in log_format.__dict__


def test_a_long_line_with_a_wide_character_wraps_in_one_pass():
    # A 64 KiB line cut by the service ends in " …[truncated]": not ASCII, so
    # each row is measured by cells. Measuring all that was left for every row
    # took seconds on the UI thread.
    import time

    line = "0.1234567, " * 6000 + " …[truncated]"
    started = time.perf_counter()
    rows = wrap_log_line(line, 120, first="training:0 │ ", rest="           │ ", max_rows=20)
    assert time.perf_counter() - started < 0.5
    assert len(rows) == 20 and rows[-1].endswith("…[truncated]")
