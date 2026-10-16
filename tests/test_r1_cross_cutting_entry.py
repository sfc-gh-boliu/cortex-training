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

"""``cortex-training tui`` before any screen: its dependencies, arguments and help."""

import re
from importlib import metadata
from pathlib import Path

import pytest

from cortex_training.tui import __main__ as tui_main
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_textual_requirement_is_the_floor_the_tui_checks():
    requirement = re.search(r'"textual>=([\d.]+),<(\d+)"', PYPROJECT.read_text())
    assert requirement, "pyproject.toml must pin textual to a tested range"
    assert tuple(int(part) for part in requirement.group(1).split(".")) == tui_main.TEXTUAL_FLOOR
    assert int(requirement.group(2)) == tui_main.TEXTUAL_FLOOR[0] + 1


def test_the_installed_textual_meets_the_floor():
    # The tests run on the floor or later; this is what the TUI was tested on.
    assert tui_main._version_tuple(metadata.version("textual")) >= tui_main.TEXTUAL_FLOOR
    assert tui_main.textual_problem() is None


@pytest.mark.parametrize("installed", ["0.50.0", "8.2.7", "1.0.0rc1"])
def test_an_older_textual_is_refused_with_an_upgrade_command(monkeypatch, capsys, installed):
    monkeypatch.setattr(metadata, "version", lambda name: installed)
    problem = tui_main.textual_problem()
    assert problem == (f"the TUI needs textual 8.2.8 or later, and {installed} is installed: "
                       "pip install --upgrade 'textual>=8.2.8,<9'")
    with pytest.raises(SystemExit) as exit_info:
        tui_main.run(["--demo"])
    assert exit_info.value.code == 2
    assert capsys.readouterr().err == f"cortex-training tui: error: {problem}\n"


def test_a_missing_textual_says_what_to_install(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert tui_main.textual_problem() == "the TUI needs the textual package: pip install 'textual>=8.2.8,<9'"


@pytest.mark.parametrize("version,expected", [("8.2.8", (8, 2, 8)), ("8.10.0.dev1", (8, 10, 0)),
                                              ("9", (9,)), ("8.2rc1", (8, 2))])
def test_version_numbers_compare_as_numbers(version, expected):
    assert tui_main._version_tuple(version) == expected


def test_the_cli_and_the_tui_import_without_pytorch():
    """PyTorch takes longer to import than the rest of the package together; only the
    DSSST1 frame codec needs it, so it loads on first use."""
    import subprocess
    import sys

    code = ("import sys, cortex_training, cortex_training._cli, cortex_training.cli, cortex_training.tui.app, "
            "cortex_training.tui.screens; print('torch' in sys.modules, 'safetensors' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1] / "src")
    assert out.stdout.split() == ["False", "False"], out.stderr


def test_the_frame_codec_is_still_reachable_from_the_package():
    import cortex_training

    assert cortex_training.wire.WIRE_FORMAT_VERSION == "DSSST1"
    from cortex_training import wire

    assert wire is cortex_training.wire and "wire" in cortex_training.__all__
    payload = cortex_training.serialize_forward_backward_args((), {"x": 1})
    assert wire.loads(payload) == {"args": [], "kwargs": {"x": 1}}  # the frame keeps sequences as lists
    with pytest.raises(AttributeError):
        cortex_training.not_a_name  # noqa: B018


CONNECT_HINT = ("to connect, run 'cortex-training login config.json', set CORTEX_TRAINING_CONFIG, pass "
                "--config config.json, --connection NAME, --base-url (local/mock), or --host + --pat; "
                "or try the TUI without a connection: --demo")


def _no_connection_anywhere(monkeypatch, tmp_path):
    """A first run: no config, no login state, no Snowflake connections, no variables."""
    for name in list(__import__("os").environ):
        if name.startswith(("CORTEX_TRAINING_", "SNOWFLAKE_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "no-login.json"))
    monkeypatch.setenv("SNOWFLAKE_HOME", str(tmp_path / "snowflake-home"))


def test_a_first_run_without_a_connection_says_how_to_connect_in_two_lines(monkeypatch, tmp_path, capsys):
    from cortex_training.snowflake_auth import SnowflakeProfileError

    _no_connection_anywhere(monkeypatch, tmp_path)

    def no_default_connection(*args, **kwargs):
        raise SnowflakeProfileError("could not open the configured default Snowflake connection: Default "
                                    "connection with name 'default' cannot be found, known ones are []")

    monkeypatch.setattr("cortex_training.client.CortexTrainingClient.from_connection_name",
                        classmethod(lambda cls, *a, **k: no_default_connection()))
    with pytest.raises(SystemExit) as exit_info:
        tui_main.run([])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert err == ("cortex-training tui: error: could not open the configured default Snowflake connection: "
                   "Default connection with name 'default' cannot be found, known ones are []\n"
                   f"{CONNECT_HINT}\n")
    assert "Traceback" not in err


def test_a_config_that_names_no_connection_gets_the_same_guidance(monkeypatch, tmp_path, capsys):
    _no_connection_anywhere(monkeypatch, tmp_path)
    config = tmp_path / "config.json"
    config.write_text('{"database": "CORTEX_TRAINING_DB", "schema": "PUBLIC"}')  # no host, PAT or URL
    with pytest.raises(SystemExit) as exit_info:
        tui_main.run(["--config", str(config)])
    assert exit_info.value.code == 2
    assert capsys.readouterr().err.endswith(f"error: no connection configured: {CONNECT_HINT}\n")


def test_every_tui_option_has_help_and_sub_job_id_says_what_it_scopes():
    parser = tui_main._build_arg_parser()
    missing = [action.dest for action in parser._actions if action.option_strings and not action.help]
    assert missing == []
    sub_job = next(action for action in parser._actions if "--sub-job-id" in action.option_strings)
    assert sub_job.help.startswith("Open the job details scoped to this sub-job (JOB_ID:TYPE:INDEX")
    assert "Resource tabs" not in parser.format_help()


def test_an_unmatched_sub_job_id_lists_the_jobs_sub_jobs():
    pytest.importorskip("textual")
    from cortex_training.tui.app import CortexTrainingLogTUI
    from tests.r1_cross_cutting_support import FakeClient
    from tests.r1_cross_cutting_support import job
    from tests.r1_cross_cutting_support import run
    from tests.r1_cross_cutting_support import settle
    from tests.r1_cross_cutting_support import wait_for

    async def go():
        app = CortexTrainingLogTUI(FakeClient([job("j1")]), "j1", sub_job_id="j1:sampling:0", poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:
            assert await wait_for(pilot, lambda: getattr(app.screen, "_sub_job_missing", False))
            assert ("error: sub-job not found: j1:sampling:0; this job's sub-jobs: j1:training:0 "
                    "(--sub-job-id takes one of them)") in app.screen._shown_lines
            await settle(app, pilot)

    run(go())
