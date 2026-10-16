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

"""Copied commands connect as the TUI does: the real CLI resolves the Logs tab's
download hints to the TUI's connection. And the recipe's renderer."""

from __future__ import annotations

import pytest

from cortex_training.tui import views as v

# ─── commands that connect as the TUI does (a PAT config, and the others) ────

HOST = "ACCOUNT.snowflakecomputing.com"
SECRET = "pat-value-for-tests"  # stands in for a PAT; it must never reach the command


def _words(sign_in, *, database="TRAINING_DB", schema="RUNS") -> list:
    """The Logs tab's download-log hint, as words, for a session that signed in as
    ``sign_in``: as LogScreen._cli builds it for a job in the configured location."""
    import shlex

    return shlex.split(v.cli_command("download-log job-9 --log-type stdout", database=database, schema=schema,
                                     **v.sign_in_options(sign_in)))


def _resolved(words: list, monkeypatch, **env):
    """The connection the real CLI resolves for ``words``, in a clean environment plus ``env``."""
    import cortex_training._cli as cli

    for name in ("CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT",
                 "CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST", "CORTEX_TRAINING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", "/nonexistent/login.json")
    args = cli.parse_args(words[1:])
    assert args.command == "download-log"
    return args


def _config_file(tmp_path, *, wrapped=False, with_pat=True):
    import json

    body = {"host": HOST, "database": "CORTEX_TRAINING_DB", "schema": "PUBLIC", **({"pat": SECRET} if with_pat else {})}
    path = tmp_path / "config dir" / "config.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"connection": body} if wrapped else body))
    return str(path)


@pytest.mark.parametrize("wrapped", [False, True])
def test_a_pat_config_session_names_the_config_and_reads_its_pat(wrapped, tmp_path, monkeypatch):
    config = _config_file(tmp_path, wrapped=wrapped)
    sign_in = {"kind": "pat", "config": config, "host": HOST, "host_in_config": True}
    words = _words(sign_in)
    assert SECRET not in " ".join(words)
    # The CLI command uses the same config file, which gives the host too: runnable as copied.
    assert "--host" not in words
    args = _resolved(words, monkeypatch)
    assert args.config == config and args._legacy_config_selected and not args.use_connection_profile
    assert (args.host, args.pat, args.database, args.schema) == (HOST, SECRET, "TRAINING_DB", "RUNS")


@pytest.mark.parametrize("variable", ["CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"])
def test_a_pat_from_the_environment_is_read_from_the_same_variable(variable, tmp_path, monkeypatch):
    args = _resolved(_words({"kind": "pat", "config": None, "host": HOST}), monkeypatch, **{variable: SECRET})
    assert (args.host, args.pat, args.use_connection_profile) == (HOST, SECRET, False)


def test_a_base_url_session_builds_the_client_with_it(tmp_path, monkeypatch):
    args = _resolved(_words({"kind": "url", "config": None, "base_url": "http://localhost:8084"}), monkeypatch)
    assert (args.base_url, args.use_connection_profile) == ("http://localhost:8084", False)
    assert _built(args) == "base_url http://localhost:8084"


# ─── a host or base URL given over the config ────────────────────────────────

OTHER = "OTHER.snowflakecomputing.com"


def _built(args) -> str:
    """Which client the real CLI builds from ``args``: ``from_pat HOST``, ``base_url URL`` or ``profile``."""
    import cortex_training._cli as cli

    class Client:
        def __init__(self, base_url=None, **kwargs):
            self.how = f"base_url {base_url}"

        @classmethod
        def from_pat(cls, host, pat, **kwargs):
            client = cls()
            client.how = f"from_pat {host}"
            return client

        @classmethod
        def from_connection_name(cls, **kwargs):
            client = cls()
            client.how = "profile"
            return client

    return cli.build_client(args, Client).how


def test_a_host_the_config_does_not_give_follows_the_config(tmp_path, monkeypatch):
    # cortex-training tui --config FILE --host OTHER: the TUI's client uses OTHER, so the command does too.
    config = _config_file(tmp_path)  # its host is HOST, and it holds the PAT
    sign_in = {"kind": "pat", "config": config, "host": OTHER, "host_in_config": False}
    words = _words(sign_in)
    assert words[1:5] == ["--config", config, "--host", OTHER]
    args = _resolved(words, monkeypatch)
    assert (args.config, args.host, args.pat, args.base_url) == (config, OTHER, SECRET, None)
    assert _built(args) == f"from_pat {OTHER}"


def test_a_base_url_the_config_does_not_give_follows_the_config(tmp_path, monkeypatch):
    # cortex-training tui --config FILE --base-url URL: the TUI's client is a base-URL one, so the command's is too.
    config = _config_file(tmp_path)  # a PAT config
    sign_in = {"kind": "url", "config": config, "base_url": "http://localhost:9999", "base_url_in_config": False}
    words = _words(sign_in)
    assert words[1:5] == ["--config", config, "--base-url", "http://localhost:9999"]
    args = _resolved(words, monkeypatch)
    assert _built(args) == "base_url http://localhost:9999"


def test_a_config_that_gives_the_host_or_base_url_is_named_alone(tmp_path, monkeypatch):
    import json

    config = _config_file(tmp_path)
    words = _words({"kind": "pat", "config": config, "host": HOST, "host_in_config": True})
    assert words[1:4] == ["--config", config, "--database"]
    assert _built(_resolved(words, monkeypatch)) == f"from_pat {HOST}"
    url_config = tmp_path / "url-config.json"
    url_config.write_text(json.dumps({"base_url": "http://localhost:8084", "database": "CORTEX_TRAINING_DB"}))
    words = _words({"kind": "url", "config": str(url_config), "base_url": "http://localhost:8084",
                    "base_url_in_config": True})
    assert words[1:4] == ["--config", str(url_config), "--database"]
    assert _built(_resolved(words, monkeypatch)) == "base_url http://localhost:8084"


@pytest.mark.parametrize("sign_in, options", [
    ({"kind": "pat", "config": "/c.json", "host": HOST, "host_in_config": True}, {"config": "/c.json", "host": None}),
    ({"kind": "pat", "config": "/c.json", "host": OTHER, "host_in_config": False},
     {"config": "/c.json", "host": OTHER}),
    # Not said: the host is named, which is right whatever the file holds.
    ({"kind": "pat", "config": "/c.json", "host": HOST}, {"config": "/c.json", "host": HOST}),
    ({"kind": "pat", "config": None, "host": HOST, "host_in_config": True}, {"config": None, "host": HOST}),
    ({"kind": "url", "config": "/c.json", "base_url": "http://h:1", "base_url_in_config": True},
     {"config": "/c.json", "base_url": None}),
    ({"kind": "url", "config": None, "base_url": "http://h:1"}, {"config": None, "base_url": "http://h:1"}),
    ({"kind": "profile"}, {}), (None, {}), ({"kind": "pat", "config": 7, "host": ["x"]}, {"config": None, "host": None}),
])
def test_sign_in_options(sign_in, options):
    assert v.sign_in_options(sign_in) == options


# ─── the recipe's renderer (R1-101) ──────────────────────────────────────────


def test_the_recipe_takes_renderer_name_for_any_model():
    pytest.importorskip("chz")
    from recipes.inference import generate
    from recipes.utils import recommended_renderer_name

    assert generate.Config(config="CONFIG.json", renderer_name="qwen3").renderer_name == "qwen3"
    # Given, it wins before tinker_cookbook's list is consulted: so it works for a model the list lacks.
    assert recommended_renderer_name("example-org/unlisted-model", renderer_name="qwen3") == "qwen3"
