from __future__ import annotations

import ast
import fnmatch
import importlib
import os
import re
from pathlib import Path
from pathlib import PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

# Each runnable example and the folder that holds its entry point, task,
# metadata and data.
EXAMPLE_FOLDERS = (
    "sft/conversational",
    "rl/math_grpo",
    "inference",
    "distillation",
    "frameworks/skyrl/gsm8k_grpo",
)


def _import_example(module_name: str):
    for dependency in ("chz", "datasets", "tinker_cookbook"):
        pytest.importorskip(dependency)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("runner_name", "task_name", "task_type", "task_module"),
    [
        (
            "examples.sft.runner",
            "examples.sft.conversational.train",
            "ConversationalTask",
            "examples.sft.conversational.task",
        ),
        (
            "examples.rl.runner",
            "examples.rl.math_grpo.train",
            "MathTask",
            "examples.rl.math_grpo.task",
        ),
    ],
)
def test_task_entrypoints_call_shared_runner(runner_name, task_name, task_type, task_module):
    runner = _import_example(runner_name)
    task_entrypoint = _import_example(task_name)

    assert task_entrypoint.train is runner.train
    assert task_entrypoint.Config.__module__ == task_name
    assert getattr(task_entrypoint, task_type).__module__ == task_module


@pytest.mark.parametrize(
    "module_name",
    ["examples.sft.conversational.train", "examples.rl.math_grpo.train"],
)
def test_entrypoints_do_not_re_export_helpers(module_name):
    module = _import_example(module_name)

    assert not hasattr(module, "__all__")


def test_helpers_shared_across_modules_are_public():
    conversational = _import_example("examples.sft.conversational.task")
    math_task = _import_example("examples.rl.math_grpo.task")
    rl_runner = _import_example("examples.rl.runner")

    assert callable(conversational.is_local_chat_file)
    assert callable(math_task.stopped_cleanly)
    assert callable(rl_runner.should_eval)
    assert not hasattr(rl_runner, "_train")


def _private_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("examples.")
        for alias in node.names
        if alias.name.startswith("_")
    ]


@pytest.mark.parametrize("package", ["sft", "rl", "inference"])
def test_private_names_do_not_cross_module_boundaries(package):
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in sorted((EXAMPLES / package).rglob("*.py"))
        if (names := _private_imports(path))
    }

    assert offenders == {}


@pytest.mark.parametrize(
    ("module_name", "config_name", "model_name"),
    [
        (
            "examples.sft.conversational.train",
            "configs/qwen3_8b_full.json",
            "Qwen/Qwen3-8B",
        ),
        (
            "examples.rl.math_grpo.train",
            "configs/qwen3_8b_lora.json",
            "Qwen/Qwen3-8B",
        ),
    ],
)
def test_hoisted_config_paths_resolve_outside_repo(
    monkeypatch,
    tmp_path,
    module_name,
    config_name,
    model_name,
):
    monkeypatch.chdir(tmp_path)
    module = _import_example(module_name)
    runner = _import_example(module.train.__module__)
    body = runner.job_body(module.Config(config="unused", job_config=config_name))

    assert any(job["model_name"] == model_name for job in body["sub_job_configs"])


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/sft/conversational/train.py",
        "examples/rl/math_grpo/train.py",
        "examples/distillation/on_policy/train.py",
    ],
)
def test_task_modules_are_entrypoints(relative_path):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' in source
    assert "chz.nested_entrypoint(main)" in source


@pytest.mark.parametrize(
    ("relative_path", "example_package"),
    [
        ("examples/sft/runner.py", "examples.sft.conversational"),
        ("examples/rl/runner.py", "examples.rl.math_grpo"),
        ("examples/distillation/train.py", ".tasks."),
    ],
)
def test_shared_runners_are_task_agnostic_libraries(relative_path, example_package):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' not in source
    assert example_package not in source


@pytest.mark.parametrize(
    ("module_name", "task_type"),
    [
        ("examples.sft.conversational.train", "ConversationalTask"),
        ("examples.rl.math_grpo.train", "MathTask"),
    ],
)
def test_entrypoint_selects_task(monkeypatch, module_name, task_type):
    module = _import_example(module_name)
    captured = {}

    def fake_train(config, task):
        captured["config"] = config
        captured["task"] = task

    monkeypatch.setattr(module, "train", fake_train)
    config = module.Config(config="unused")
    module.main(config)

    assert captured["config"] is config
    assert type(captured["task"]).__name__ == task_type


def _metadata_keys(path: Path) -> set[str]:
    return {
        line.split(":", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line[0].isspace() and not line.startswith(("#", "-")) and ":" in line
    }


@pytest.mark.parametrize("folder", EXAMPLE_FOLDERS)
def test_each_example_folder_holds_its_metadata(folder):
    assert (EXAMPLES / folder / "example.yaml").is_file()


def test_examples_keep_task_code_and_metadata_in_their_own_folders():
    assert not (EXAMPLES / "sft" / "example.yaml").exists()
    assert not (EXAMPLES / "rl" / "example.yaml").exists()
    assert not (EXAMPLES / "sft" / "tasks").exists()
    assert not (EXAMPLES / "rl" / "tasks").exists()
    assert not (EXAMPLES / "inference" / "identity.py").exists()
    assert (EXAMPLES / "sft" / "conversational" / "task.py").is_file()
    assert (EXAMPLES / "sft" / "conversational" / "identity.py").is_file()
    assert (EXAMPLES / "rl" / "math_grpo" / "task.py").is_file()


@pytest.mark.parametrize("folder", ["sft/conversational", "rl/math_grpo", "inference"])
def test_catalog_owned_fields_leave_example_metadata(folder):
    keys = _metadata_keys(EXAMPLES / folder / "example.yaml")

    assert "models" not in keys
    assert "default_training_gpus" not in keys


# The runnable workflows are called examples everywhere. Other projects keep
# their own names, so these lines may still say "recipe": tinker-cookbook's
# recipes and its tinker_cookbook.recipes package, the provenance.recipe path
# of an example adapted from another project, and paths in the Arctic Platform
# repository.
_STRAY_TERM = re.compile("recipe", re.IGNORECASE)
_UPSTREAM_REFERENCES = (
    ("*", re.compile(r"tinker_cookbook\.recipes\.")),
    ("examples/frameworks/*", re.compile(r"cookbook", re.IGNORECASE)),
    ("examples/*example.yaml", re.compile(r"^  recipe: ")),
    ("examples/frameworks/skyrl/*", re.compile(r"Arctic-Platform/(?:blob/main/)?recipes/")),
)


def _ignore_patterns() -> list[str]:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _ignored(path: PurePosixPath, *, is_dir: bool, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/") and not is_dir:
            continue
        pattern = pattern.rstrip("/")
        if "/" in pattern:
            if fnmatch.fnmatch(path.as_posix(), pattern.lstrip("/")):
                return True
        elif fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def _repository_files():
    """Every file a search of the repository sees: not hidden, not ignored."""
    patterns = _ignore_patterns()
    for directory, dirnames, filenames in os.walk(REPO_ROOT):
        relative = PurePosixPath(Path(directory).relative_to(REPO_ROOT).as_posix())
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".")
            and not _ignored(relative / name, is_dir=True, patterns=patterns)
        )
        for name in sorted(filenames):
            path = relative / name
            if not name.startswith(".") and not _ignored(path, is_dir=False, patterns=patterns):
                yield path


def _is_upstream_reference(path: PurePosixPath, line: str) -> bool:
    return any(
        fnmatch.fnmatch(path.as_posix(), pattern) and reference.search(line)
        for pattern, reference in _UPSTREAM_REFERENCES
    )


def test_recipe_only_names_other_projects():
    this_file = PurePosixPath(Path(__file__).resolve().relative_to(REPO_ROOT).as_posix())
    stray = []
    for path in _repository_files():
        if _STRAY_TERM.search(path.as_posix()):
            stray.append(f"{path}: file name")
        if path == this_file:
            continue
        data = (REPO_ROOT / path).read_bytes()
        if b"\0" in data:
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if _STRAY_TERM.search(line) and not _is_upstream_reference(path, line):
                stray.append(f"{path}:{number}: {line.strip()}")

    assert stray == []


def test_repository_search_skips_ignored_and_hidden_files():
    files = {path.as_posix() for path in _repository_files()}

    assert "examples/utils.py" in files
    assert ".gitignore" not in files
    assert not any(name.startswith((".git/", ".venv/")) or "__pycache__" in name for name in files)
