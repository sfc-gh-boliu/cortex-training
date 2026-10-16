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

"""Job names end to end: ``create_job(comment=...)``, ``submit --comment``, the
recipes' default names, and the TUI's read-only client.

Request bodies are read back from the bytes a fake transport adapter received
and decoded, never only type-checked.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import re
from pathlib import Path

import pytest
import requests
from requests.adapters import BaseAdapter

import cortex_training._cli as cli
from cortex_training.client import CortexTrainingClient
from cortex_training.client import MAX_COMMENT_BYTES
from cortex_training.client import SubJobConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No real login state, connection settings, TUI cache or home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    for name in ("CORTEX_TRAINING_CONFIG", "CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_BASE_URL",
                 "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT", "SNOWFLAKE_HOST", "SNOWFLAKE_PAT",
                 "CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE", "CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA"):
        monkeypatch.delenv(name, raising=False)


class _Wire(BaseAdapter):
    """Records each request as sent and answers from ``reply(request)``."""

    def __init__(self, reply=None):
        super().__init__()
        self.sent: list[requests.PreparedRequest] = []
        self._reply = reply or (lambda request: (200, {"job_id": "server-job"}))

    def send(self, request, **kwargs):
        self.sent.append(request)
        status, body = self._reply(request)
        resp = requests.Response()
        resp.status_code = status
        resp._content = json.dumps(body).encode("utf-8")
        resp.headers["Content-Type"] = "application/json"
        resp.request = request
        resp.url = request.url
        return resp

    def close(self):
        pass

    def bodies(self) -> list:
        return [json.loads(r.body) for r in self.sent if r.body]


def _client(reply=None) -> tuple[CortexTrainingClient, _Wire]:
    client = CortexTrainingClient(base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="PUBLIC",
                                  max_retries=0)
    wire = _Wire(reply)
    client._session.mount("http://", wire)
    return client, wire


def _sampling_sub():
    return SubJobConfig.sampling_job(model_name="Qwen/Qwen3-8B", max_seq_len=2048, n_gpus=1)


def _deployment_body():
    return {"sub_job_configs": [{
        "job_type": "sampling", "model_name": "Qwen/Qwen3-8B",
        "inference_config": {"deployment_class": "EVALUATION", "num_replicas": 1, "n_gpus": 2, "max_seq_len": 4096,
                             "vllm_config": {"tensor_parallel_size": 2, "max_model_len": 4096}},
    }], "idle_timeout_seconds": 0}


# ─── create_job / create_job_from_body ───────────────────────────────────────


def test_create_job_sends_the_comment_in_the_body():
    client, wire = _client()
    comment = "qwen3-8b lora sweep #3\nlr 1e-4 · warmup 100 · 名前"
    assert client.create_job(sub_jobs=[_sampling_sub()], comment=comment) == "server-job"
    (body,) = wire.bodies()
    assert body["comment"] == comment
    assert body["sub_job_configs"][0]["model_name"] == "Qwen/Qwen3-8B"


def test_create_job_without_a_comment_sends_none():
    client, wire = _client()
    client.create_job(sub_jobs=[_sampling_sub()])
    (body,) = wire.bodies()
    assert "comment" not in body


def test_comment_limit_is_1024_bytes_of_utf8_not_characters():
    client, wire = _client()
    at_limit = "é" * (MAX_COMMENT_BYTES // 2)  # 512 characters, 1,024 bytes
    client.create_job(sub_jobs=[_sampling_sub()], comment=at_limit)
    assert wire.bodies()[0]["comment"] == at_limit

    over = at_limit + "x"  # 513 characters, 1,025 bytes
    with pytest.raises(ValueError, match=r"at most 1024 bytes of UTF-8, got 1025"):
        client.create_job(sub_jobs=[_sampling_sub()], comment=over)
    with pytest.raises(ValueError, match="comment must be a string"):
        client.create_job_from_body({"sub_job_configs": [{"job_type": "sampling", "model_name": "m",
                                                          "inference_config": {"n_gpus": 1}}], "comment": 7})
    assert len(wire.sent) == 1, "an invalid comment must fail before anything is sent"


def test_model_deployment_is_sent_without_the_comment(caplog):
    client, wire = _client()
    body = {**_deployment_body(), "comment": "my endpoint"}
    with caplog.at_level(logging.WARNING, logger="cortex_training.client"):
        client.create_job_from_body(body)
    (sent,) = wire.bodies()
    assert "comment" not in sent
    assert sent["sub_job_configs"][0]["inference_config"]["deployment_class"] == "EVALUATION"
    assert "comment not sent" in caplog.text and "model deployment" in caplog.text
    assert body["comment"] == "my endpoint", "the caller's dict is left alone"


def test_model_deployment_via_create_job_never_sends_a_comment():
    client, wire = _client()
    sub = SubJobConfig.sampling_job(model_name="Qwen/Qwen3-8B", max_seq_len=4096, n_gpus=2,
                                    extra_sampling={"deployment_class": "EVALUATION", "num_replicas": 1})
    client.create_job(sub_jobs=[sub], idle_timeout_seconds=0, comment="x" * 5000)  # not sent, so not checked
    (sent,) = wire.bodies()
    assert sent["sub_job_configs"][0]["inference_config"]["deployment_class"] == "EVALUATION"
    assert "comment" not in sent


def test_a_deployment_sub_job_after_another_sub_job_still_drops_the_comment(caplog):
    client, wire = _client()
    training = {"job_type": "training", "model_name": "Qwen/Qwen3-8B",
                "training_config": {"n_gpus": 8, "max_seq_len": 4096},
                "inference_config": {"deployment_class": None}}  # null is not a deployment
    body = {"sub_job_configs": [training, *_deployment_body()["sub_job_configs"]], "comment": "endpoint v2"}
    with caplog.at_level(logging.WARNING, logger="cortex_training.client"):
        client.create_job_from_body(body)
    (sent,) = wire.bodies()
    assert "comment" not in sent and [c["job_type"] for c in sent["sub_job_configs"]] == ["training", "sampling"]
    assert "comment not sent" in caplog.text

    plain = {"sub_job_configs": [training, {"job_type": "sampling", "model_name": "Qwen/Qwen3-8B",
                                            "inference_config": {"n_gpus": 1, "max_seq_len": 2048}}],
             "comment": "two sub-jobs, no deployment"}
    client.create_job_from_body(plain)
    assert wire.bodies()[1]["comment"] == "two sub-jobs, no deployment"


def test_create_job_sends_no_blank_comment():
    client, wire = _client()
    client.create_job(sub_jobs=[_sampling_sub()], comment="  \n ")
    (body,) = wire.bodies()
    assert "comment" not in body


def test_create_job_from_body_forwards_a_valid_comment():
    client, wire = _client()
    body = {"sub_job_configs": [{"job_type": "sampling", "model_name": "gpt2",
                                 "inference_config": {"max_seq_len": 128, "n_gpus": 1}}],
            "comment": "from the JSON"}
    client.create_job_from_body(body)
    assert wire.bodies()[0]["comment"] == "from the JSON"


# ─── CLI: submit --comment ───────────────────────────────────────────────────


class _SubmitClient:
    def __init__(self):
        self.bodies = []

    def create_job_from_body(self, body):
        # Round-trip through JSON, as the request body would.
        self.bodies.append(json.loads(json.dumps(body)))
        return {"job_id": "server-job"}


def _job_file(tmp_path, body=None) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps(body or {"sub_job_configs": [{
        "job_type": "sampling", "model_name": "gpt2", "inference_config": {"max_seq_len": 128, "n_gpus": 1}}],
        "comment": "from the JSON"}), encoding="utf-8")
    return path


def test_submit_comment_overrides_the_json_comment(tmp_path):
    fake = _SubmitClient()
    stdout = io.StringIO()
    rc = cli.main(["--base-url", "http://test.local", "--database", "DB", "submit", str(_job_file(tmp_path)),
                   "--comment", "sweep #3 · lr=1e-4"], client_factory=lambda args: fake, stdout=stdout)
    assert rc == 0
    assert fake.bodies[0]["comment"] == "sweep #3 · lr=1e-4"
    assert json.loads(stdout.getvalue()) == {"job_id": "server-job"}


def test_submit_dry_run_prints_the_comment_and_checks_its_size(tmp_path):
    stdout, stderr = io.StringIO(), io.StringIO()
    path = _job_file(tmp_path)
    assert cli.main(["submit", str(path), "--dry-run", "--comment", "named"], stdout=stdout) == 0
    assert json.loads(stdout.getvalue())["comment"] == "named"
    rc = cli.main(["submit", str(path), "--dry-run", "--comment", "x" * (MAX_COMMENT_BYTES + 1)], stderr=stderr)
    assert rc == 1 and "at most 1024 bytes" in stderr.getvalue()


def test_submit_leaves_the_comment_out_of_a_model_deployment(tmp_path, caplog):
    path = _job_file(tmp_path, _deployment_body())
    stdout = io.StringIO()
    with caplog.at_level(logging.WARNING, logger="cortex_training.client"):
        rc = cli.main(["submit", str(path), "--dry-run", "--comment", "my endpoint"], stdout=stdout)
    assert rc == 0
    printed = json.loads(stdout.getvalue())
    assert "comment" not in printed and printed["idle_timeout_seconds"] == 0
    assert "model deployment" in caplog.text

    fake = _SubmitClient()
    rc = cli.main(["--base-url", "http://test.local", "--database", "DB", "submit", str(path), "--comment", "x"],
                  client_factory=lambda args: fake, stdout=io.StringIO())
    assert rc == 0 and "comment" not in fake.bodies[0]


def test_submit_empty_comment_removes_the_json_comment(tmp_path):
    path = _job_file(tmp_path)  # its JSON carries "from the JSON"
    fake = _SubmitClient()
    rc = cli.main(["--base-url", "http://test.local", "--database", "DB", "submit", str(path), "--comment", ""],
                  client_factory=lambda args: fake, stdout=io.StringIO())
    assert rc == 0 and "comment" not in fake.bodies[0]
    stdout = io.StringIO()
    assert cli.main(["submit", str(path), "--dry-run", "--comment", "  "], stdout=stdout) == 0
    assert "comment" not in json.loads(stdout.getvalue())
    stdout = io.StringIO()
    assert cli.main(["submit", str(path), "--dry-run"], stdout=stdout) == 0
    assert json.loads(stdout.getvalue())["comment"] == "from the JSON"  # without the flag the JSON's stays


def test_submit_help_documents_comment(capsys):
    with pytest.raises(SystemExit):
        cli.main(["submit", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--comment TEXT" in help_text and "1,024 bytes" in help_text


# ─── recipes: a readable default name ────────────────────────────────────────


def test_default_job_name_is_recipe_and_model():
    from recipes.utils import default_job_name

    lora = {"sub_job_configs": [{"job_type": "training", "model_name": "Qwen/Qwen3-8B",
                                 "training_config": {"peft_config": {"r": 8}}},
                                {"job_type": "sampling", "model_name": "Qwen/Qwen3-8B", "inference_config": {}}]}
    assert default_job_name("math-grpo", lora) == "math-grpo · Qwen3-8B LoRA"
    full = {"sub_job_configs": [{"job_type": "sampling", "model_name": "Qwen/Qwen3.6-35B-A3B"}]}
    assert default_job_name("inference-serve", full) == "inference-serve · Qwen3.6-35B-A3B"
    assert default_job_name("x", {}) == "x"


def test_name_job_precedence():
    from recipes.utils import name_job

    body = {"sub_job_configs": [{"job_type": "sampling", "model_name": "Qwen/Qwen3-8B"}]}
    assert name_job(body, "inference-serve")["comment"] == "inference-serve · Qwen3-8B"
    assert name_job(body, "inference-serve", "my endpoint")["comment"] == "my endpoint"
    assert "comment" not in name_job({**body, "comment": "json"}, "inference-serve", "")  # name="" sends none
    assert name_job({**body, "comment": "json"}, "inference-serve")["comment"] == "json"  # the JSON's own wins
    assert "comment" not in body, "the loaded body is not modified"


class _Config:
    def __init__(self, job_config, name=None):
        self.config = "unused"
        self.job_config = job_config
        self.debug_image_tag = None
        self.name = name


def test_sft_and_grpo_runners_name_their_jobs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from recipes.rl import train as rl_train
    from recipes.rl.tasks.math import MathTask
    from recipes.sft import train as sft_train

    class Conversational:  # stands in for ConversationalTask (whose module needs tinker_cookbook)
        name = "conversational"

    sft = sft_train.named_job_body(_Config("configs/qwen3_8b_full.json"), Conversational())
    assert sft["comment"] == "conversational-sft · Qwen3-8B"
    sft_lora = sft_train.named_job_body(_Config("configs/qwen3_8b_lora.json"), Conversational())
    assert sft_lora["comment"] == "conversational-sft · Qwen3-8B LoRA"
    grpo = rl_train.named_job_body(_Config("configs/qwen3_8b_lora.json"), MathTask())
    assert grpo["comment"] == "math-grpo · Qwen3-8B LoRA"
    assert rl_train.named_job_body(_Config("configs/qwen3_8b_lora.json", name="run 7"), MathTask())["comment"] == "run 7"
    assert "comment" not in rl_train.named_job_body(_Config("configs/qwen3_8b_lora.json", name=""), MathTask())


class _Built(Exception):
    """Stops a recipe once it has built its create body: no client or network after that."""


def _body_built_by(module, monkeypatch, run) -> dict:
    """The create body that ``run()`` (a recipe's entry point) builds."""
    from recipes.inference import endpoint

    built = {}

    def build(*args, **kwargs):
        built["body"], _source = endpoint.inference_endpoint_body(*args, **kwargs)
        raise _Built

    monkeypatch.setattr(module, "inference_endpoint_body", build)
    monkeypatch.setattr(module, "make_client", lambda path: object())  # serve makes its client first
    with pytest.raises(_Built):
        run()
    return built["body"]


def test_inference_recipes_name_their_jobs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from recipes.inference import evaluate
    from recipes.inference import generate
    from recipes.inference import serve
    from recipes.inference.endpoint import inference_endpoint_body

    body, _ = inference_endpoint_body("configs/qwen3_8b_lora.json", recipe="inference-generate", name="eval run")
    assert body["comment"] == "eval run"
    # Each recipe's own entry point names the job it creates, and takes name=.
    runs = {
        "inference-serve · Qwen3-8B": (serve, lambda **kw: serve.main(serve.Config(config="unused", **kw))),
        "inference-generate · Qwen3-8B": (generate, lambda **kw: generate.main(generate.Config(config="unused", **kw))),
        "inference-evaluate gsm8k · Qwen3-8B": (
            evaluate, lambda **kw: evaluate.main(evaluate.Config(config="unused", task="GSM8K", **kw))),
    }
    for default, (module, run) in runs.items():
        assert _body_built_by(module, monkeypatch, run)["comment"] == default
        assert _body_built_by(module, monkeypatch, lambda: run(name="run 7"))["comment"] == "run 7"
        assert "comment" not in _body_built_by(module, monkeypatch, lambda: run(name=""))


def _config_field_default(path: Path, field: str):
    """The default of ``field`` in the module's ``Config`` class, read from its syntax tree
    (for a recipe whose module needs packages this test environment lacks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config")
    entry = next(node for node in config.body if isinstance(node, ast.AnnAssign) and node.target.id == field)
    return ast.literal_eval(entry.value), ast.unparse(entry.annotation)


def test_recipe_configs_declare_name_and_yaml_documents_the_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from recipes.rl import train as rl_train
    from recipes.rl.math_grpo import train as math_grpo
    from recipes.rl.tasks.math import MathTask

    assert math_grpo.Config(config="unused").name is None
    assert math_grpo.Config(config="unused", name="run 7").name == "run 7"
    yaml_text = (REPO_ROOT / "recipes/rl/recipe.yaml").read_text(encoding="utf-8")
    recipe_id = re.search(r"^id: (\S+)$", yaml_text, re.M).group(1)
    template = re.search(r'^default_job_name: "(.+)"$', yaml_text, re.M).group(1)
    assert template == f"{recipe_id} · MODEL"
    named = rl_train.named_job_body(_Config("configs/qwen3_8b_full.json"), MathTask())
    assert named["comment"] == template.replace("MODEL", "Qwen3-8B")

    sft_yaml = (REPO_ROOT / "recipes/sft/recipe.yaml").read_text(encoding="utf-8")
    assert re.search(r'^default_job_name: "conversational-sft · MODEL"$', sft_yaml, re.M)
    assert re.search(r"^id: conversational-sft$", sft_yaml, re.M)
    # The conversational recipe's module needs tinker_cookbook: read its Config without importing it.
    default, annotation = _config_field_default(REPO_ROOT / "recipes/sft/conversational/train.py", "name")
    assert default is None and annotation == "str | None"


def test_recipe_notebooks_name_their_jobs():
    for path in ("recipes/inference/inference_walkthrough.ipynb",
                 "recipes/sft/conversational/qwen3_8b_sft_training.ipynb",
                 "recipes/sft/conversational/qwen3_8b_sft_training_multiplex.ipynb"):
        notebook = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        source = "".join(line for cell in notebook["cells"] for line in cell["source"])
        namespace: dict = {}
        cell = next("".join(c["source"]) for c in notebook["cells"] if "job_body = {" in "".join(c["source"]))
        exec(compile(cell, path, "exec"), namespace)  # the cell only builds a dict
        assert namespace["job_body"]["comment"] and len(namespace["job_body"]["comment"].encode()) <= MAX_COMMENT_BYTES
        assert "create_job_from_body(job_body)" in source


# ─── the TUI's client never creates a database or schema ─────────────────────


def _not_found(request):
    if request.method == "POST" and request.path_url.endswith("/api/v2/statements"):
        return 200, {"data": []}
    return 400, {"code": "517602", "message": "Schema CORTEX_TRAINING_DB.PUBLIC is not found or not authorized"}


def test_tui_client_reports_not_found_instead_of_creating_the_database(monkeypatch):
    from cortex_training.tui import __main__ as tui_main

    wire = _Wire(_not_found)
    seen = {}

    def build_client(args, cls):
        client = cls(base_url=args.base_url, database=args.database, schema=args.schema, max_retries=0)
        client._session.mount("http://", wire)
        seen["client"] = client
        return client

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, schema_only=False):
            seen["app_client"] = client

        def run(self):
            with pytest.raises(requests.HTTPError) as error:
                seen["app_client"].list_jobs()  # the job list's fallback (the account-wide list never creates)
            seen["error"] = error.value

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training._cli.build_client", build_client)
    assert tui_main.run(["--base-url", "http://test.local", "--database", "CORTEX_TRAINING_DB", "--schema", "PUBLIC"]) == 0
    assert seen["app_client"] is seen["client"] and seen["client"]._auto_create_database is False
    assert [r.method for r in wire.sent] == ["GET"], "the TUI must not run CREATE DATABASE / CREATE SCHEMA"
    assert seen["error"].response.status_code == 400

    # Control: a default client turns the same answer into CREATE DATABASE.
    control = _Wire(_not_found)
    client = CortexTrainingClient(base_url="http://test.local", database="CORTEX_TRAINING_DB", schema="EXTRA",
                                  max_retries=0)
    client._session.mount("http://", control)
    with pytest.raises(Exception):
        client.list_jobs()
    statements = [json.loads(r.body)["statement"] for r in control.sent if r.method == "POST"]
    assert statements and statements[0].startswith("CREATE DATABASE IF NOT EXISTS")
