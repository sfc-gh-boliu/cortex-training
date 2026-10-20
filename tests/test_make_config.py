"""make_config: job configs from the model catalog, overrides, and router replay defaults."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples import utils
from examples.utils import make_config
from examples.utils import merge_job_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "model-catalog" / "models"
QWEN36 = "Qwen/Qwen3.6-35B-A3B"

# The fixed default router-replay caps make_config fills in for an MoE RL job.
FIXED_TRAINING_CACHE_BYTES = 5 * 1024**3  # 5368709120
FIXED_SAMPLING_CACHE_BYTES = 20 * 1024**3  # 21474836480


def _file(slug: str, key: str) -> dict:
    return json.loads((CATALOG / slug / f"{key}.json").read_text())


def _sub_job(body: dict, job_type: str) -> dict:
    (sub_job,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == job_type]
    return sub_job


def _training(body: dict) -> dict:
    return _sub_job(body, "training")["training_config"]


def _sampling(body: dict) -> dict:
    return _sub_job(body, "sampling")["inference_config"]


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value))
    return path


# ─── Picking catalog files ───────────────────────────────────────────────────


def test_the_training_job_is_the_default():
    assert make_config("Qwen/Qwen3-8B", "standard") == _file("qwen3-8b", "standard/h200-training-full")


def test_lora_picks_the_lora_file():
    assert make_config("Qwen/Qwen3-8B", "standard", lora=True) == _file("qwen3-8b", "standard/h200-training-lora")


def test_inference_returns_the_inference_job():
    assert make_config("Qwen/Qwen3-8B", "standard", inference=True) == _file(
        "qwen3-8b", "standard/h200-inference-full"
    )


def test_hardware_defaults_to_h200():
    assert make_config("Qwen/Qwen3-8B", "standard", hardware="H200") == make_config("Qwen/Qwen3-8B", "standard")


def test_rl_combines_the_training_and_sampling_sub_jobs_unchanged():
    body = make_config("Qwen/Qwen3-8B", "standard", lora=True, rl=True)

    assert body == {
        "hardware": "H200",
        "sub_job_configs": [
            _file("qwen3-8b", "standard/h200-training-lora")["sub_job_configs"][0],
            _file("qwen3-8b", "standard/h200-inference-lora")["sub_job_configs"][0],
        ],
    }


def test_the_default_rl_job_is_four_training_and_two_sampling_gpus_with_sft_betas():
    body = make_config("Qwen/Qwen3-8B", "standard", lora=True, rl=True)

    assert _training(body)["n_gpus"] == 4
    assert _training(body)["optimizer"]["betas"] == [0.9, 0.999]
    assert _sampling(body)["n_gpus"] == 2
    assert _sampling(body)["vllm_config"]["tensor_parallel_size"] == 1
    assert "mb_spec" not in _training(body)
    assert "reduce_scatter" not in _training(body)["ds_config"]["zero_optimization"]


def test_a_missing_combination_names_the_files_that_exist():
    with pytest.raises(FileNotFoundError) as raised:
        make_config(QWEN36, "standard", lora=True, rl=True)

    assert str(raised.value) == (
        "Qwen/Qwen3.6-35B-A3B has no standard/h200-inference-lora.json in the model catalog; its files "
        "are: long/h200-inference-full.json, long/h200-inference-lora.json, long/h200-training-full.json, "
        "long/h200-training-lora.json, standard/h200-training-full.json, standard/h200-training-lora.json"
    )


def test_missing_hardware_names_the_files_that_exist():
    with pytest.raises(FileNotFoundError, match="has no standard/b200-training-full.json in the model catalog"):
        make_config("Qwen/Qwen3-8B", "standard", hardware="b200")


def test_an_unknown_model_lists_the_catalog():
    with pytest.raises(FileNotFoundError, match="'Org/Unknown' is not in the model catalog; it has: .*Qwen/Qwen3-8B"):
        make_config("Org/Unknown", "standard")


def test_the_model_is_the_exact_model_id():
    with pytest.raises(FileNotFoundError, match="did you mean 'Qwen/Qwen3-8B'"):
        make_config("qwen/qwen3-8b", "standard")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "Qwen/Qwen3-8B", "context": "medium"}, "context must be one of"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "hardware": "h100"}, "hardware must be one of"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "inference": True, "rl": True}, "not both"),
        ({"model": "Qwen/Qwen3-8B"}, "needs a model and a context"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "training_config": "x.json"}, "not both"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "router_replay": True}, "only apply with rl=True"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "rollout_batch_size": 8}, "only apply with rl=True"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "rl": True, "rollout_batch_size": 0}, "positive integer"),
        ({"training_config": "x.json", "rl": True}, "needs both training_config and inference_config"),
        ({"inference_config": "x.json"}, "add rl=True"),
    ],
)
def test_invalid_selections_are_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_config(**kwargs)


def test_overrides_must_be_a_list():
    with pytest.raises(TypeError, match="overrides must be a list"):
        make_config("Qwen/Qwen3-8B", "standard", overrides="override.json")


def test_each_call_returns_a_fresh_copy():
    first = make_config("Qwen/Qwen3-8B", "standard")
    _training(first)["n_gpus"] = 99

    assert _training(make_config("Qwen/Qwen3-8B", "standard"))["n_gpus"] == 4


# ─── Explicit config files ───────────────────────────────────────────────────


def test_an_explicit_training_config_is_used_as_is(tmp_path):
    body = _file("qwen3-8b", "standard/h200-training-full")
    path = _write(tmp_path / "train.json", body)

    assert make_config(training_config=path) == body


def test_an_explicit_inference_config_with_inference(tmp_path):
    body = _file("qwen3-8b", "standard/h200-inference-full")
    path = _write(tmp_path / "infer.json", body)

    assert make_config(inference_config=path, inference=True) == body


def test_explicit_rl_combines_the_files(tmp_path):
    training = _write(tmp_path / "train.json", _file("qwen3-8b", "standard/h200-training-full"))
    inference = _write(tmp_path / "infer.json", _file("qwen3-8b", "standard/h200-inference-full"))

    assert make_config(training_config=training, inference_config=inference, rl=True) == make_config(
        "Qwen/Qwen3-8B", "standard", rl=True
    )


def test_rl_sub_jobs_must_share_hardware(tmp_path):
    training = _write(tmp_path / "train.json", _file("qwen3-8b", "standard/h200-training-full"))
    inference_body = _file("qwen3-8b", "standard/h200-inference-full")
    inference_body["hardware"] = "B200"
    inference = _write(tmp_path / "infer.json", inference_body)

    with pytest.raises(ValueError, match="hardware is set once per job.*is H200 but .* is B200"):
        make_config(training_config=training, inference_config=inference, rl=True)


def test_rl_needs_one_sub_job_from_each_file(tmp_path):
    rl_body = make_config("Qwen/Qwen3-8B", "standard", rl=True)
    training = _write(tmp_path / "rl.json", rl_body)
    inference = _write(tmp_path / "infer.json", _file("qwen3-8b", "standard/h200-inference-full"))

    with pytest.raises(ValueError, match="must hold exactly one training sub-job"):
        make_config(training_config=training, inference_config=inference, rl=True)


def _qwen36_files(tmp_path):
    training = _write(tmp_path / "train.json", _file("qwen3.6-35b-a3b", "long/h200-training-lora"))
    inference = _write(tmp_path / "infer.json", _file("qwen3.6-35b-a3b", "long/h200-inference-lora"))
    return training, inference


def test_explicit_rl_fills_router_replay_only_when_asked(tmp_path):
    training, inference = _qwen36_files(tmp_path)

    plain = make_config(training_config=training, inference_config=inference, rl=True)
    replayed = make_config(training_config=training, inference_config=inference, rl=True, router_replay=True)

    assert "router_replay" not in _training(plain)
    assert "router_replay" not in _sampling(plain)
    assert replayed == make_config(QWEN36, "long", lora=True, rl=True)


def test_explicit_router_replay_reads_the_config_of_a_model_outside_the_catalog(tmp_path, monkeypatch):
    calls = []

    def resolve(model_name):
        calls.append(model_name)
        return utils.MoEGeometry(num_moe_layers=40, topk=8)

    monkeypatch.setattr(utils, "resolve_model_moe_geometry", resolve)
    training_body = _file("qwen3.6-35b-a3b", "long/h200-training-lora")
    inference_body = _file("qwen3.6-35b-a3b", "long/h200-inference-lora")
    for body in (training_body, inference_body):
        body["sub_job_configs"][0]["model_name"] = "Org/New-MoE"
    training = _write(tmp_path / "train.json", training_body)
    inference = _write(tmp_path / "infer.json", inference_body)

    body = make_config(training_config=training, inference_config=inference, rl=True, router_replay=True)

    assert calls == ["Org/New-MoE"]
    assert _training(body)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES


def _router_replay_override(**replay) -> dict:
    """The same ``router_replay`` settings on both RL sub-jobs."""
    return {
        "sub_job_configs": [
            {"job_type": "training", "training_config": {"router_replay": dict(replay)}},
            {"job_type": "sampling", "inference_config": {"router_replay": dict(replay)}},
        ]
    }


DENSE_MODEL_ERROR = "router replay needs an MoE model; the catalog lists 'Qwen/Qwen3-8B' as 'dense'"


@pytest.mark.parametrize(
    "overrides", [[], [_router_replay_override(max_cache_bytes=1)]], ids=["planned caps", "both caps set"]
)
def test_router_replay_on_a_dense_catalog_model_is_an_error(tmp_path, overrides):
    training = _write(tmp_path / "train.json", _file("qwen3-8b", "standard/h200-training-full"))
    inference = _write(tmp_path / "infer.json", _file("qwen3-8b", "standard/h200-inference-full"))

    with pytest.raises(ValueError, match=DENSE_MODEL_ERROR):
        make_config(training_config=training, inference_config=inference, rl=True, router_replay=True, overrides=overrides)
    with pytest.raises(ValueError, match=DENSE_MODEL_ERROR):
        make_config("Qwen/Qwen3-8B", "standard", rl=True, router_replay=True, overrides=overrides)


@pytest.mark.parametrize("router_replay", [None, False])
def test_an_override_cannot_turn_router_replay_on_for_a_dense_catalog_model(router_replay):
    override = _router_replay_override(enabled=True, max_cache_bytes=1)

    with pytest.raises(ValueError, match=DENSE_MODEL_ERROR):
        make_config("Qwen/Qwen3-8B", "standard", rl=True, router_replay=router_replay, overrides=[override])


@pytest.mark.parametrize("router_replay", [None, True])
def test_router_replay_turned_off_on_a_dense_model_is_left_off(router_replay):
    body = make_config(
        "Qwen/Qwen3-8B", "standard", rl=True, router_replay=router_replay, overrides=[_router_replay_override(enabled=False)]
    )

    assert _training(body)["router_replay"] == _sampling(body)["router_replay"] == {"enabled": False}


def test_explicit_caps_on_an_moe_model_still_resolve_its_geometry(monkeypatch):
    calls = []
    model_geometry = utils._model_geometry

    def recording(model_name, catalog_dir):
        calls.append(model_name)
        return model_geometry(model_name, catalog_dir)

    monkeypatch.setattr(utils, "_model_geometry", recording)
    body = make_config(QWEN36, "long", rl=True, overrides=[_router_replay_override(max_cache_bytes=3)])

    assert calls == [QWEN36]
    assert _training(body)["router_replay"] == {"max_cache_bytes": 3, "enabled": True, "mode": "strict"}
    assert _sampling(body)["router_replay"] == {"max_cache_bytes": 3, "enabled": True}


def test_explicit_caps_on_a_model_outside_the_catalog_still_check_its_config(tmp_path, monkeypatch):
    def resolve(model_name):
        raise ValueError(f"Model config is missing MoE routing geometry ({model_name})")

    monkeypatch.setattr(utils, "resolve_model_moe_geometry", resolve)
    files = []
    for name, key in (("train.json", "standard/h200-training-full"), ("infer.json", "standard/h200-inference-full")):
        body = _file("qwen3-8b", key)
        body["sub_job_configs"][0]["model_name"] = "Org/New-Dense"
        files.append(_write(tmp_path / name, body))

    with pytest.raises(ValueError, match=r"missing MoE routing geometry \(Org/New-Dense\)"):
        make_config(
            training_config=files[0],
            inference_config=files[1],
            rl=True,
            router_replay=True,
            overrides=[_router_replay_override(max_cache_bytes=1)],
        )


# ─── Merge rules ─────────────────────────────────────────────────────────────


def test_dictionaries_merge_key_by_key():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"optimizer": {"lr": 3e-5}}}]}],
    )

    assert _training(body)["optimizer"] == {**_training(make_config("Qwen/Qwen3-8B", "standard"))["optimizer"], "lr": 3e-5}


def test_null_removes_a_key():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"activation_checkpointing": None}}]}],
    )

    assert "activation_checkpointing" not in _training(body)


def test_null_on_a_missing_key_adds_nothing():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[{"experiment_name": None, "sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": {"enabled": False, "mode": None}}}]}],
    )

    assert "experiment_name" not in body
    assert _training(body)["router_replay"] == {"enabled": False}


def test_sub_jobs_are_matched_by_job_type():
    base = make_config("Qwen/Qwen3-8B", "standard", rl=True)
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": "sampling", "inference_config": {"n_gpus": 4}}]}],
    )

    assert _sampling(body)["n_gpus"] == 4
    assert _training(body) == _training(base)
    assert [sub["job_type"] for sub in body["sub_job_configs"]] == ["training", "sampling"]


def test_the_plan_example_override_changes_one_sub_job():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": {"max_cache_bytes": 8589934592}}}]}],
    )

    assert _training(body)["router_replay"] == {"max_cache_bytes": 8589934592, "enabled": True, "mode": "strict"}


def test_an_override_with_a_duplicate_job_type_is_an_error():
    with pytest.raises(ValueError, match="lists job_type 'training' twice"):
        make_config(
            "Qwen/Qwen3-8B",
            "standard",
            overrides=[{"sub_job_configs": [{"job_type": "training"}, {"job_type": "training"}]}],
        )


def test_an_override_that_matches_no_sub_job_is_an_error():
    with pytest.raises(ValueError, match=r"has job_type 'sampling', which matches no sub-job \(the job has \['training'\]\)"):
        make_config(
            "Qwen/Qwen3-8B",
            "standard",
            overrides=[{"sub_job_configs": [{"job_type": "sampling", "inference_config": {"n_gpus": 1}}]}],
        )


@pytest.mark.parametrize("entry", [{"training_config": {"n_gpus": 8}}, "training"])
def test_an_override_sub_job_needs_a_job_type(entry):
    with pytest.raises(ValueError, match=r"override sub_job_configs\[0\] needs a job_type"):
        merge_job_config(make_config("Qwen/Qwen3-8B", "standard"), {"sub_job_configs": [entry]})


def test_override_sub_job_configs_must_be_a_list():
    with pytest.raises(ValueError, match="must be a list of sub-jobs"):
        merge_job_config(make_config("Qwen/Qwen3-8B", "standard"), {"sub_job_configs": {"job_type": "training"}})


def test_other_lists_are_replaced_whole():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        lora=True,
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"peft_config": {"target_modules": ["q_proj"]}}}]}],
    )

    assert _training(body)["peft_config"]["target_modules"] == ["q_proj"]
    assert _training(body)["peft_config"]["r"] == 32


def test_merge_does_not_modify_its_arguments():
    base = make_config("Qwen/Qwen3-8B", "standard", rl=True)
    override = {"hardware": "B200", "sub_job_configs": [{"job_type": "training", "training_config": {"n_gpus": 8}}]}
    base_copy, override_copy = copy.deepcopy(base), copy.deepcopy(override)

    merged = merge_job_config(base, override)

    assert base == base_copy
    assert override == override_copy
    assert merged["hardware"] == "B200"
    assert _training(merged)["n_gpus"] == 8


def test_overrides_apply_in_order_from_files_and_dicts(tmp_path):
    first = _write(
        tmp_path / "first.json",
        {"sub_job_configs": [{"job_type": "training", "training_config": {"max_seq_len": 8192, "train_batch_size": 8}}]},
    )
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[first, {"sub_job_configs": [{"job_type": "training", "training_config": {"max_seq_len": 4096}}]}],
    )

    assert (_training(body)["max_seq_len"], _training(body)["train_batch_size"]) == (4096, 8)


def test_override_file_problems_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="override file not found"):
        make_config("Qwen/Qwen3-8B", "standard", overrides=[tmp_path / "missing.json"])
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError, match="is not valid JSON"):
        make_config("Qwen/Qwen3-8B", "standard", overrides=[tmp_path / "bad.json"])
    _write(tmp_path / "list.json", [1, 2])
    with pytest.raises(ValueError, match="must be a JSON object"):
        make_config("Qwen/Qwen3-8B", "standard", overrides=[tmp_path / "list.json"])


# ─── Router replay defaults ──────────────────────────────────────────────────


def test_the_fixed_default_caps_match_the_module_constants():
    assert FIXED_TRAINING_CACHE_BYTES == utils.DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES == 5368709120
    assert FIXED_SAMPLING_CACHE_BYTES == utils.DEFAULT_ROUTER_REPLAY_SAMPLING_CACHE_BYTES == 21474836480


@pytest.mark.parametrize("lora", [True, False])
def test_moe_rl_fills_router_replay_on_both_sub_jobs_and_nothing_else(lora):
    body = make_config(QWEN36, "long", lora=lora, rl=True)

    assert _training(body).pop("router_replay") == {
        "enabled": True,
        "mode": "strict",
        "max_cache_bytes": FIXED_TRAINING_CACHE_BYTES,
    }
    assert _sampling(body).pop("router_replay") == {"enabled": True, "max_cache_bytes": FIXED_SAMPLING_CACHE_BYTES}
    assert body == make_config(QWEN36, "long", lora=lora, rl=True, router_replay=False)


def test_the_default_caps_are_fixed_not_sized_by_the_rollout_batch():
    # The caps are the fixed 5 GiB / 20 GiB defaults regardless of rollout_batch_size,
    # which is accepted only for backward compatibility.
    body = make_config(QWEN36, "long", lora=True, rl=True, rollout_batch_size=1024)

    assert _training(body)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES
    assert _sampling(body)["router_replay"]["max_cache_bytes"] == FIXED_SAMPLING_CACHE_BYTES
    assert body == make_config(QWEN36, "long", lora=True, rl=True)
    assert body == make_config(QWEN36, "long", lora=True, rl=True, rollout_batch_size=1)


def test_the_fixed_caps_do_not_change_with_gpu_counts_or_sequence_length():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        rollout_batch_size=64,
        overrides=[
            {
                "sub_job_configs": [
                    {"job_type": "training", "training_config": {"n_gpus": 16, "max_seq_len": 32768}},
                    {"job_type": "sampling", "inference_config": {"max_seq_len": 32768, "vllm_config": {"tensor_parallel_size": 2}}},
                ]
            }
        ],
    )

    assert _training(body)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES
    assert _sampling(body)["router_replay"]["max_cache_bytes"] == FIXED_SAMPLING_CACHE_BYTES


def test_explicit_max_cache_bytes_and_mode_win():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[
            {
                "sub_job_configs": [
                    {
                        "job_type": "training",
                        "training_config": {"router_replay": {"mode": "best_effort", "max_cache_bytes": 2147483648}},
                    },
                    {"job_type": "sampling", "inference_config": {"router_replay": {"max_cache_bytes": 17179869184}}},
                ]
            }
        ],
    )

    assert _training(body)["router_replay"] == {"mode": "best_effort", "max_cache_bytes": 2147483648, "enabled": True}
    assert _sampling(body)["router_replay"] == {"max_cache_bytes": 17179869184, "enabled": True}


@pytest.mark.parametrize(("job_type", "config_key"), [("training", "training_config"), ("sampling", "inference_config")])
def test_enabled_false_in_an_override_turns_router_replay_off(job_type, config_key):
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": job_type, config_key: {"router_replay": {"enabled": False}}}]}],
    )

    assert _sub_job(body, job_type)[config_key]["router_replay"] == {"enabled": False}
    other = _sampling(body) if job_type == "training" else _training(body)
    assert "router_replay" not in other


def test_dense_rl_gets_no_router_replay():
    body = make_config("Qwen/Qwen3-8B", "standard", rl=True)

    assert "router_replay" not in _training(body)
    assert "router_replay" not in _sampling(body)


@pytest.mark.parametrize("kwargs", [{}, {"inference": True}])
def test_only_rl_jobs_get_router_replay(kwargs):
    body = make_config(QWEN36, "long", **kwargs)

    assert "router_replay" not in json.dumps(body)


def test_router_replay_leaves_vllm_gpu_memory_utilization_alone():
    body = make_config(QWEN36, "long", rl=True, rollout_batch_size=1024)

    assert _sampling(body)["vllm_config"] == _file("qwen3.6-35b-a3b", "long/h200-inference-full")["sub_job_configs"][0][
        "inference_config"
    ]["vllm_config"]
    assert (utils.DEFAULT_ROUTING_SKEW, utils.DEFAULT_SAFETY_FACTOR) == (2.0, 1.2)


def test_router_replay_must_be_an_object():
    with pytest.raises(ValueError, match="router_replay must be an object"):
        make_config(
            QWEN36,
            "long",
            rl=True,
            overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": True}}]}],
        )


# ─── Command line ────────────────────────────────────────────────────────────


def test_the_command_line_writes_the_job(tmp_path):
    out = tmp_path / "nested" / "job.json"

    assert utils.main(["--model", "Qwen/Qwen3-8B", "--context", "standard", "--lora", "-o", str(out)]) == 0
    assert json.loads(out.read_text()) == make_config("Qwen/Qwen3-8B", "standard", lora=True)


def test_the_command_line_builds_rl_jobs_with_overrides(tmp_path):
    override = _write(tmp_path / "o.json", {"sub_job_configs": [{"job_type": "training", "training_config": {"optimizer": {"lr": 1e-6}}}]})
    out = tmp_path / "rl.json"

    status = utils.main(
        ["--model", QWEN36, "--context", "long", "--hardware", "h200", "--rl", "--rollout-batch-size", "1024", "--override", str(override), "-o", str(out)]
    )

    assert status == 0
    assert json.loads(out.read_text()) == make_config(QWEN36, "long", rl=True, rollout_batch_size=1024, overrides=[override])


def test_the_command_line_takes_explicit_files(tmp_path):
    training, inference = _qwen36_files(tmp_path)
    out = tmp_path / "rl.json"

    status = utils.main(
        ["--training-config", str(training), "--inference-config", str(inference), "--rl", "--router-replay", "-o", str(out)]
    )

    assert status == 0
    assert json.loads(out.read_text()) == make_config(QWEN36, "long", lora=True, rl=True)


def test_the_command_line_writes_stdout_for_a_dash(capsys):
    assert utils.main(["--model", "Qwen/Qwen3-8B", "--context", "standard", "--inference", "-o", "-"]) == 0
    assert json.loads(capsys.readouterr().out) == make_config("Qwen/Qwen3-8B", "standard", inference=True)


def test_the_command_line_reports_a_missing_combination(tmp_path, capsys):
    status = utils.main(["--model", QWEN36, "--context", "standard", "--rl", "-o", str(tmp_path / "x.json")])

    assert status == 1
    assert "error: Qwen/Qwen3.6-35B-A3B has no standard/h200-inference-full.json" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--training-config", "t.json"],
        ["--model", "Qwen/Qwen3-8B"],
        ["--context", "standard"],
        ["--training-config", "t.json", "--lora"],
        ["--training-config", "t.json", "--inference-config", "i.json"],
        ["--training-config", "t.json", "--router-replay"],
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--rollout-batch-size", "8"],
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--inference", "--rl"],
    ],
)
def test_the_command_line_rejects_mixed_forms(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        utils.main(argv + ["-o", "out.json"])

    assert raised.value.code == 2


def test_the_module_runs_from_the_command_line(tmp_path):
    out = tmp_path / "job.json"
    result = subprocess.run(
        [sys.executable, "-m", "examples.utils", "--model", "Qwen/Qwen3-8B", "--context", "standard", "--rl", "-o", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text()) == make_config("Qwen/Qwen3-8B", "standard", rl=True)


# ─── The final job decides ───────────────────────────────────────────────────


def _moe_catalog(tmp_path: Path) -> Path:
    """A catalog with two small MoE models and one dense model, standard H200 files only."""
    catalog = tmp_path / "models"
    models = {
        "org/Model-A": {"num_moe_layers": 1, "topk": 1, "num_experts": 4},
        "org/Model-B": {"num_moe_layers": 10, "topk": 2, "num_experts": 8},
        "org/Dense-C": None,
    }
    for model_id, moe in models.items():
        folder = catalog / model_id.rsplit("/", 1)[-1].lower()
        (folder / "standard").mkdir(parents=True)
        info = {"model_id": model_id, "license": {"name": "MIT", "url": "https://example.com/LICENSE"}}
        info["architecture"] = "moe" if moe else "dense"
        if moe:
            info["moe"] = moe
        _write(folder / "model.json", info)
        _write(
            folder / "standard" / "h200-training-full.json",
            {"hardware": "H200", "sub_job_configs": [{"job_type": "training", "model_name": model_id, "training_config": {
                "n_gpus": 2, "max_seq_len": 10, "train_batch_size": 2, "optimizer": {"lr": 1e-5}}}]},
        )
        _write(
            folder / "standard" / "h200-inference-full.json",
            {"hardware": "H200", "sub_job_configs": [{"job_type": "sampling", "model_name": model_id, "inference_config": {
                "n_gpus": 2, "max_seq_len": 10, "vllm_config": {"tensor_parallel_size": 1}}}]},
        )
    return catalog


def _rename(model_id: str) -> dict:
    return {"sub_job_configs": [{"job_type": "training", "model_name": model_id}, {"job_type": "sampling", "model_name": model_id}]}


def _replay_caps(body: dict) -> tuple[int, int]:
    return _training(body)["router_replay"]["max_cache_bytes"], _sampling(body)["router_replay"]["max_cache_bytes"]


def test_router_replay_is_filled_for_the_moe_model_the_final_job_names(tmp_path):
    catalog = _moe_catalog(tmp_path)

    body = make_config("org/Model-A", "standard", rl=True, overrides=[_rename("org/Model-B")], catalog_dir=catalog)

    # The rename picks an MoE model, so router replay is filled with the fixed caps.
    assert {sub["model_name"] for sub in body["sub_job_configs"]} == {"org/Model-B"}
    assert _replay_caps(body) == (FIXED_TRAINING_CACHE_BYTES, FIXED_SAMPLING_CACHE_BYTES)


def test_explicit_router_replay_is_filled_for_the_final_moe_model(tmp_path):
    catalog = _moe_catalog(tmp_path)
    training = catalog / "model-a/standard/h200-training-full.json"
    inference = catalog / "model-a/standard/h200-inference-full.json"

    body = make_config(
        training_config=training,
        inference_config=inference,
        rl=True,
        router_replay=True,
        overrides=[_rename("org/Model-B")],
        catalog_dir=catalog,
    )

    assert {sub["model_name"] for sub in body["sub_job_configs"]} == {"org/Model-B"}
    assert _replay_caps(body) == (FIXED_TRAINING_CACHE_BYTES, FIXED_SAMPLING_CACHE_BYTES)


def test_an_override_to_a_dense_model_gets_no_router_replay(tmp_path):
    catalog = _moe_catalog(tmp_path)

    body = make_config("org/Model-A", "standard", rl=True, overrides=[_rename("org/Dense-C")], catalog_dir=catalog)

    assert "router_replay" not in json.dumps(body)


def test_rl_sub_jobs_must_name_the_same_model(tmp_path):
    catalog = _moe_catalog(tmp_path)
    override = {"sub_job_configs": [{"job_type": "sampling", "model_name": "org/Model-B"}]}

    with pytest.raises(ValueError, match="training and sampling sub-jobs must name the same model; the job has "
                                         "training 'org/Model-A' and sampling 'org/Model-B'"):
        make_config("org/Model-A", "standard", rl=True, overrides=[override], catalog_dir=catalog)


def test_an_override_to_a_model_outside_the_catalog_needs_router_replay_asked_for(tmp_path, monkeypatch):
    catalog = _moe_catalog(tmp_path)
    monkeypatch.setattr(utils, "resolve_model_moe_geometry", lambda model_name: utils.MoEGeometry(num_moe_layers=3, topk=2))

    plain = make_config("org/Model-A", "standard", rl=True, overrides=[_rename("org/New")], catalog_dir=catalog)
    planned = make_config("org/Model-A", "standard", rl=True, router_replay=True, overrides=[_rename("org/New")], catalog_dir=catalog)

    assert "router_replay" not in json.dumps(plain)
    assert _training(planned)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES
    assert _sampling(planned)["router_replay"]["max_cache_bytes"] == FIXED_SAMPLING_CACHE_BYTES
