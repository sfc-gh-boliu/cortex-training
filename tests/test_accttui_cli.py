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

"""``cortex-training tui --schema-only`` and how the TUI's job list is documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex_training import cli
from cortex_training.tui import __main__ as tui_main
from cortex_training.tui import account

CLI_MD = Path(__file__).resolve().parents[1] / "docs" / "reference" / "cli.md"


@pytest.fixture
def started(monkeypatch):
    """Runs ``tui_main.run`` with a stand-in app; returns what the app was built with."""
    seen: dict = {}

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, schema_only=False):
            seen.update(client=client, job_id=job_id, schema_only=schema_only)

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)
    return seen


BASE = ["--base-url", "http://localhost:1", "--database", "CORTEX_TRAINING_DB", "--schema", "PUBLIC"]


def test_the_job_list_is_account_wide_unless_schema_only(started):
    assert tui_main.run(BASE) == 0
    assert started["ran"] and started["schema_only"] is False
    assert tui_main.run([*BASE, "--schema-only"]) == 0
    assert started["schema_only"] is True
    # The TUI's client stays read-only either way.
    assert started["client"]._auto_create_database is False


def test_schema_only_reaches_the_tui_through_the_cortex_training_command(started):
    assert cli._tui_argv(["--database", "DB", "tui", "--schema-only"]) == ["--database", "DB", "--schema-only"]
    assert cli.main(["tui", *BASE, "--schema-only"]) == 0
    assert started["schema_only"] is True


def test_the_help_describes_schema_only(capsys):
    with pytest.raises(SystemExit):
        tui_main.run(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--schema-only" in text and "List only the configured database and schema's jobs" in text


def test_cli_md_documents_the_scope_the_flag_the_column_and_the_fields():
    text = " ".join(CLI_MD.read_text(encoding="utf-8").split())
    for phrase in ("--schema-only", "Location", "`db`", "`schema`", "CORTEX_TRAINING_TUI_DEMO_ACCOUNT_LIST",
                   f"{account.ACCOUNT_LIST_CAP:,}", f"{account.ACCOUNT_LIST_MAX_PAGES} pages", "10 minutes",
                   "a minute", "refresh failed", "older active jobs may be missing", "status=active"):
        assert phrase in text, phrase


def test_cli_md_quotes_the_headers_the_tui_shows():
    # Each header text as the TUI builds it, so the docs can't drift from the code.
    text = " ".join(CLI_MD.read_text(encoding="utf-8").split())
    where = "CORTEX_TRAINING_DB.PUBLIC"
    capped = account.ListScope(account.ACCOUNT, stopped=account.CAPPED, walked=account.ACCOUNT_LIST_CAP)
    limited = account.ListScope(account.ACCOUNT, stopped=account.PAGE_LIMIT, walked=7,
                                pages=account.ACCOUNT_LIST_MAX_PAGES)
    for scope in (account.ListScope(account.ACCOUNT), capped):
        assert f"`{account.scope_text(scope, location=where)}`" in text, scope
    assert f"`{account.scope_text(limited, location=where).replace('7 jobs', 'N jobs')}`" in text
    for reason in ("not_deployed",):
        assert f"`{account.scope_text(account.ListScope(account.FALLBACK, reason=reason), location=where)}`" in text
    for reason in ("disabled", "forbidden"):
        why = account.scope_text(account.ListScope(account.FALLBACK, reason=reason), location=where)
        assert f"`{why[why.index('('):]}`" in text, reason
    assert "`this schema only: DB.SCHEMA (--schema-only)`" in text
