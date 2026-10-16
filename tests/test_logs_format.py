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

"""Execution-log helpers (what a failed job without a saved console shows), and
the one-row text of the Logs tab's status line."""

import json

from cortex_training.tui.format import execution_log_records
from cortex_training.tui.format import format_execution_record
from cortex_training.tui.format import is_error_record
from cortex_training.tui.format import one_line

_TRACE = (
    "Traceback (most recent call last):\n"
    '  File "loader.py", line 3, in load\n'
    "torch.OutOfMemoryError: CUDA out of memory\n"
    "\n"
    "The above exception was the direct cause of the following exception:\n"
    "\n"
    "RuntimeError: loading the model failed"
)


def test_records_parse_one_per_line_and_skip_torn_lines():
    content = (
        json.dumps({"level": "INFO", "msg": "init"}) + "\n"
        + "\n"
        + '{"level": "ERROR", "msg": "torn'  # cut mid-write
        + "\n" + json.dumps(["not", "an", "object"]) + "\n"
    )
    assert execution_log_records(content) == [
        {"level": "INFO", "msg": "init"},
        {"_raw": '["not", "an", "object"]'},
    ]
    assert execution_log_records(None) == []


def test_error_records_are_error_level_or_carry_an_exception_or_error():
    assert is_error_record({"level": "ERROR", "msg": "x"})
    assert is_error_record({"level": "critical", "msg": "x"})
    assert is_error_record({"level": "INFO", "msg": "job_metrics", "error": "HTTPException: boom"})
    assert is_error_record({"msg": "x", "exc": _TRACE})
    assert not is_error_record({"level": "INFO", "msg": "init phase", "error": ""})
    assert not is_error_record({"level": "WARNING", "msg": "slow"})
    assert not is_error_record("text")


def test_an_error_record_shows_its_exception_chain_indented_under_it():
    record = {"ts": "2026-10-14T11:00:42Z", "level": "ERROR", "logger": "runtime",
              "msg": "failed to create job", "exc": _TRACE}
    lines = format_execution_record(record)
    assert lines[0] == "2026-10-14T11:00:42Z ERROR runtime: failed to create job"
    assert lines[1:] == ["    " + line for line in _TRACE.splitlines()]


def test_an_error_message_is_shown_once_and_list_tracebacks_are_joined():
    record = {"level": "INFO", "msg": "job_metrics", "error": "RuntimeError: boom\nsecond line",
              "traceback": ["Traceback (most recent call last):\n", "RuntimeError: boom\n"]}
    assert format_execution_record(record) == [
        "INFO  job_metrics",
        "    error: RuntimeError: boom",
        "    second line",
        "    Traceback (most recent call last):",
        "    RuntimeError: boom",
    ]
    # An error already in the message line is not repeated.
    assert format_execution_record({"msg": "boom", "error": "boom"}) == ["boom"]


def test_a_structured_error_is_shown_rather_than_dropped():
    record = {"level": "ERROR", "msg": "create failed", "error": {"type": "QuotaError", "message": "no H200 left"}}
    assert format_execution_record(record) == [
        "ERROR create failed",
        '    error: {"message": "no H200 left", "type": "QuotaError"}',
    ]


def test_a_record_without_a_message_shows_its_traceback_once():
    record = {"ts": "2026-10-14T11:00:42Z", "level": "ERROR", "phase": "load_model", "exc": _TRACE}
    lines = format_execution_record(record)
    # The head carries the record's other fields, not the escaped traceback.
    assert lines[0] == '2026-10-14T11:00:42Z ERROR {"phase": "load_model"}'
    assert lines[1:] == ["    " + line for line in _TRACE.splitlines()]
    assert sum("CUDA out of memory" in line for line in lines) == 1
    # Only failure fields: the head is just the time and level.
    assert format_execution_record({"level": "ERROR", "error": "boom"}) == ["ERROR", "    error: boom"]


def test_one_line_keeps_service_text_on_one_safe_row():
    # Line breaks join with " / "; a tab is a space; ESC, BEL and the C1 range
    # (\x9b is a one-byte CSI) never reach the terminal.
    assert one_line("waiting for 8 GPUs\r\nof type H200\nnow") == "waiting for 8 GPUs / of type H200 / now"
    assert one_line("a\tb\u2028c") == "a b / c"
    assert one_line("x\x1b]0;title\x07y\x9b2Jz\x7f") == "x\ufffd]0;title\ufffdy\ufffd2Jz\ufffd"
    assert one_line("plain — ünïcode ✓") == "plain — ünïcode ✓"
    assert one_line(None) == "None"
