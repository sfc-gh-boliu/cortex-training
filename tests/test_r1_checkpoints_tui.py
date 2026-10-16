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

"""The sign-in ``cortex-training tui`` records, and the copied commands that connect with it."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import views as v  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_tui_files(tmp_path, monkeypatch):
    # As conftest does for test_tui_* files: no writes to the real cache or home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))


def _job(status="RUNNING", **extra) -> dict:
    cfg = {"n_gpus": 8.0}
    return {"job_id": "7", "status": status, **extra,
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B",
                          "training_config": cfg}]}


def _client(job=None):
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    job = job or _job()
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    c.database, c.schema = "TRAINING_DB", "RUNS"
    return c


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause()
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


def _run(coro_fn, *args):
    asyncio.run(coro_fn(*args))


# ─── the commands connect as the TUI does (a PAT config session) ─────────────

HOST = "ACCOUNT.snowflakecomputing.com"
SECRET = "pat-value-for-tests"  # stands in for a PAT


def _launch(argv, monkeypatch, **env):
    """``cortex-training tui ARGV`` with a fake app: the sign-in it was given."""
    from cortex_training.tui import __main__ as tui_main

    seen = {}

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, schema_only=False):
            seen["app"] = self

        def run(self):
            seen["sign_in"] = getattr(self, "sign_in", None)

    for name in ("CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT",
                 "CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST", "CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_DATABASE",
                 "SNOWFLAKE_DATABASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", "/nonexistent/login.json")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training._cli.build_client", lambda args, cls: MagicMock())
    assert tui_main.run(argv) == 0
    return seen["sign_in"]


def _config(tmp_path, **body):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(body))
    return path


def test_a_pat_config_session_says_where_the_pat_is_and_never_holds_it(tmp_path, monkeypatch):
    path = _config(tmp_path, host=f"https://{HOST}/", pat=SECRET, database="CORTEX_TRAINING_DB")
    monkeypatch.chdir(tmp_path)
    sign_in = _launch(["--config", "config.json"], monkeypatch)  # a relative path, as typed
    assert sign_in == {"kind": "pat", "config": str(path.resolve()), "host": HOST, "host_in_config": True}
    assert SECRET not in repr(sign_in)


def test_a_config_without_a_pat_is_named_from_the_flag_or_the_environment(tmp_path, monkeypatch):
    path = _config(tmp_path, host=HOST, database="CORTEX_TRAINING_DB")
    sign_in = _launch(["--config", str(path)], monkeypatch, SNOWFLAKE_PAT=SECRET)
    assert sign_in == {"kind": "pat", "config": str(path.resolve()), "host": HOST, "host_in_config": True}
    # From CORTEX_TRAINING_CONFIG too: the same config file.
    sign_in = _launch([], monkeypatch, CORTEX_TRAINING_CONFIG=str(path), CORTEX_TRAINING_PAT=SECRET)
    assert sign_in["config"] == str(path.resolve())


def test_flags_profiles_and_base_urls(tmp_path, monkeypatch):
    sign_in = _launch(["--host", HOST, "--pat", SECRET, "--database", "DB"], monkeypatch)
    assert sign_in == {"kind": "pat", "config": None, "host": HOST, "host_in_config": False}
    assert SECRET not in repr(sign_in)
    assert _launch(["--base-url", "http://localhost:8084", "--database", "DB"], monkeypatch) == {
        "kind": "url", "config": None, "base_url": "http://localhost:8084", "base_url_in_config": False}
    assert _launch(["--connection", "training"], monkeypatch) == {"kind": "profile"}


OTHER = "OTHER.snowflakecomputing.com"


def _copied_cli(sign_in, monkeypatch, **env):
    """The connection the real CLI resolves, in a clean environment plus ``env``, for
    the Logs tab's download-log hint of a session that signed in as ``sign_in``."""
    import shlex

    import cortex_training._cli as cli

    # As LogScreen._cli builds it for a job in the configured database and schema.
    command = v.cli_command("download-log 7 --log-type stdout", database="TRAINING_DB", schema="RUNS",
                            **v.sign_in_options(sign_in))
    for name in ("CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT",
                 "CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST", "CORTEX_TRAINING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    words = shlex.split(command)
    return words, cli.parse_args(words[1:])


def test_a_host_flag_over_the_config_reaches_the_copied_command(tmp_path, monkeypatch):
    # cortex-training tui --config FILE --host OTHER: the client and the command both use OTHER.
    path = _config(tmp_path, host=HOST, pat=SECRET, database="CORTEX_TRAINING_DB")
    sign_in = _launch(["--config", str(path), "--host", OTHER], monkeypatch)
    assert sign_in == {"kind": "pat", "config": str(path.resolve()), "host": OTHER, "host_in_config": False}
    words, args = _copied_cli(sign_in, monkeypatch)
    assert words[1:5] == ["--config", str(path.resolve()), "--host", OTHER]
    assert (args.host, args.pat, args.base_url, args.use_connection_profile) == (OTHER, SECRET, None, False)
    # The same host as the config's needs no flag.
    sign_in = _launch(["--config", str(path), "--host", f"https://{HOST}/"], monkeypatch)
    assert sign_in["host_in_config"] is True
    words, args = _copied_cli(sign_in, monkeypatch)
    assert "--host" not in words and args.host == HOST


def test_a_base_url_flag_over_the_config_reaches_the_copied_command(tmp_path, monkeypatch):
    # cortex-training tui --config FILE --base-url URL: a base-URL client, not the config's PAT host.
    path = _config(tmp_path, host=HOST, pat=SECRET, database="CORTEX_TRAINING_DB")
    sign_in = _launch(["--config", str(path), "--base-url", "http://localhost:9999"], monkeypatch)
    assert sign_in == {"kind": "url", "config": str(path.resolve()), "base_url": "http://localhost:9999",
                       "base_url_in_config": False}
    words, args = _copied_cli(sign_in, monkeypatch)
    assert words[1:5] == ["--config", str(path.resolve()), "--base-url", "http://localhost:9999"]
    assert args.base_url == "http://localhost:9999"  # the CLI builds a base-URL client when one is set
    # A config whose own base URL the TUI used needs no flag.
    url_path = tmp_path / "url.json"
    url_path.write_text('{"base_url": "http://localhost:8084", "database": "CORTEX_TRAINING_DB"}')
    sign_in = _launch(["--config", str(url_path)], monkeypatch)
    assert sign_in["base_url_in_config"] is True
    words, args = _copied_cli(sign_in, monkeypatch)
    assert "--base-url" not in words and args.base_url == "http://localhost:8084"


def test_a_host_from_the_environment_is_named_beside_a_config_without_one(tmp_path, monkeypatch):
    path = _config(tmp_path, pat=SECRET, database="CORTEX_TRAINING_DB")  # no host in the file
    sign_in = _launch(["--config", str(path)], monkeypatch, CORTEX_TRAINING_HOST=OTHER)
    assert (sign_in["host"], sign_in["host_in_config"]) == (OTHER, False)
    # Runnable where that variable isn't set.
    words, args = _copied_cli(sign_in, monkeypatch)
    assert words[1:5] == ["--config", str(path.resolve()), "--host", OTHER] and args.host == OTHER


def test_a_config_base_url_read_as_its_pat_host_needs_no_flag(tmp_path, monkeypatch):
    # The CLI reads a base_url without a scheme as the PAT host; the copied command, with the same file, does too.
    path = _config(tmp_path, base_url=HOST, pat=SECRET, database="CORTEX_TRAINING_DB")
    sign_in = _launch(["--config", str(path)], monkeypatch)
    assert (sign_in["kind"], sign_in["host"], sign_in["host_in_config"]) == ("pat", HOST, True)
    words, args = _copied_cli(sign_in, monkeypatch)
    assert "--host" not in words and (args.host, args.base_url) == (HOST, None)


@pytest.mark.parametrize("sign_in, connect", [
    ({"kind": "pat", "config": "/home/me/config.json", "host": HOST, "host_in_config": True},
     ["--config", "/home/me/config.json"]),
    ({"kind": "pat", "config": "/home/me/config.json", "host": OTHER, "host_in_config": False},
     ["--config", "/home/me/config.json", "--host", OTHER]),
    ({"kind": "url", "config": None, "base_url": "http://localhost:8084", "base_url_in_config": False},
     ["--base-url", "http://localhost:8084"]),
    (None, []),  # unknown (the demo): as before
])
def test_the_logs_download_hints_connect_as_the_tui_does(sign_in, connect, monkeypatch):
    import shlex

    import cortex_training._cli as cli
    from cortex_training.tui.app import LogScreen

    monkeypatch.setattr(LogScreen, "_MAX_LINES", 50)
    monkeypatch.delenv("CORTEX_TRAINING_CONFIG", raising=False)  # the parser's --config default
    job = _job()
    c = _client(job=job)
    page = {"entries": [{"_raw": f"ts line {i}"} for i in range(120)], "next_cursor": "C1", "eof": False}
    pages = iter([page])
    c.tail_logs.side_effect = lambda *a, **k: next(pages, {"entries": [], "next_cursor": "C1", "eof": False})

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        app.sign_in = sign_in
        async with app.run_test(size=(240, 40)) as pilot:
            assert await _wait(pilot, lambda: "ts line 119" in getattr(app.screen, "_shown_lines", []))
            status = str(app.screen.query_one("#log-status", Static).render())
            hint = status.split(" · ", 1)[1].removesuffix(" for the full log")
            words = shlex.split(hint)
            assert words == ["cortex-training", *connect, "--database", "TRAINING_DB", "--schema", "RUNS",
                             "download-log", "7", "--log-type", "stdout"]
            if connect:  # and the CLI connects with it, as the TUI did
                args = cli.build_parser().parse_args(words[1:])
                assert (args.config, args.host, args.base_url) == (
                    (sign_in or {}).get("config"), OTHER if "--host" in words else None,
                    (sign_in or {}).get("base_url"))
            await _settle(app, pilot)

    _run(run)
