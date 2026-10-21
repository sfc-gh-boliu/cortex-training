# Cortex Training Examples

Examples are runnable, end-to-end workflows organized by task. Each example owns
its code, documentation, metadata, and optional notebooks. Its job config comes
from the [model catalog](../model-catalog/README.md).

## Available Examples

| Example | Method | Dataset | Status |
|---|---|---|---|
| [Conversational SFT](sft/README.md) | LoRA or full-parameter SFT | Hugging Face chat datasets, GSM8K, identity | Runnable |
| [DARE-Bench agent SFT](sft/dare/README.md) | Full-parameter multi-turn tool SFT + agent eval | DARE-Bench | Runnable |
| [Math GRPO](rl/README.md) | Reinforcement learning | Hendrycks MATH and MATH-500 | Runnable |
| [On-policy distillation](distillation/README.md) | Distillation from a frozen teacher | GSM8K | Runnable |
| [Inference endpoint](inference/README.md) | Serve, generate, eval | Open weights, checkpoints, MATH-500, GSM8K, identity | Runnable |

GRPO can also be run from [SkyRL](frameworks/skyrl/gsm8k_grpo/README.md), which uses its
own entry point and install rather than the prerequisites below.

## What is in this folder

`examples/` holds three kinds of things:

- **Runnable examples:** complete workflows you start with `python -m`. Each
  one trains or serves a model end to end.
- **Shared code:** loops and helpers the runnable examples are built from.
  Nothing here runs on its own.
- **Reference material:** raw request bodies and pointers to outside
  frameworks, with no Python of ours.

```text
examples/
├── README.md                  # start here
├── utils.py                   # SHARED: make_config, build_job, client, checkpoints,
│                              #   batching, router replay planner
├── logging.py                 # SHARED: metrics logging (W&B, Snowflake experiments)
├── sft/
│   ├── runner.py              # SHARED: the SFT training loop
│   ├── overrides.json         # config changes for every SFT run (ships empty, just a _comment)
│   └── conversational/        # RUNNABLE: train.py, task.py, identity.py,
│                              #   example.yaml, data/, notebooks
├── rl/
│   ├── runner.py              # SHARED: the GRPO loop
│   ├── overrides.json         # config changes for every RL run (the shared RL tuning)
│   ├── configs/               # OUTPUT: the job each run generated (not committed)
│   └── math_grpo/             # RUNNABLE: train.py, task.py, example.yaml
├── inference/                 # RUNNABLE: serve.py, generate.py, evaluate.py;
│                              #   SHARED: endpoint.py, prompts.py; example.yaml,
│                              #   overrides.json (ships empty, just a _comment), a notebook
├── distillation/              # RUNNABLE: on-policy distillation; its own
│                              #   overrides.json, configs/ (OUTPUT, not committed),
│                              #   tasks/ and example.yaml
├── api/                       # REFERENCE: raw request bodies
└── frameworks/                # REFERENCE: SkyRL and tinker-cookbook
```

Every example ships an `overrides.json`, applied to every run of that example on
top of the catalog. RL and distillation carry real settings; SFT and inference
ship empty apart from a `_comment`. Each file's reserved `_comment` key (a
free-text note, since JSON has no comments) is stripped before merging and never
reaches the job body.

## Prerequisites

Install the client and example dependencies from the repository root:

```bash
uv pip install -e .
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
uv pip install wandb
```

(`pip install ...` works in place of `uv pip install ...` if you prefer.)

Create a local Snowflake connection file from
`docs/getting-started/connection.json.template` (account host and PAT).

To log SFT or GRPO metrics to Weights & Biases, set a project on the train
command (`wandb_project=...`) and export:

```bash
export WANDB_API_KEY=...
export WANDB_BASE_URL=...
```

To log the same metrics to Snowflake experiment tracking, pass
`sf_tracking=True` on the train command after `uv pip install
"snowflake-ml-python>=1.19.0"`. Results appear in Snowsight under
**AI & ML > Experiments**. See the
[Snowflake ML Experiments docs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiments).

## Running Examples

Examples are Python modules so they can share code without path manipulation.
Run them from the repository root:

```bash
python -m examples.sft.conversational.train config=/path/to/config.json
python -m examples.rl.math_grpo.train config=/path/to/config.json
python -m examples.distillation.on_policy.train config=/path/to/config.json
python -m examples.inference.serve config=/path/to/config.json
```

`config=` is always the Snowflake PAT/connection file, never a job body.
`model=`, `context=`, `hardware=` and `lora=` pick the job from the model
catalog. See each example README for hardware, expected metrics, common
variations, and how its job is built.

## How an example builds its job

The SFT, RL and inference examples build their job with `build_job` in
`utils.py`, which calls `make_config` with three layers, each overriding the
one before it:

1. The model catalog file for the model, context and hardware, in the LoRA or
   full mode: `model-catalog/models/<slug>/<context>/<hw>-<kind>-<mode>.json`.
   An RL job combines the training and inference files.
2. The example's `overrides.json` (`sft/`, `rl/` or `inference/`), applied to
   every model in that example. Every example ships one (SFT and inference are
   empty apart from a `_comment`); its `_comment` key is stripped before merging.
3. The command line: the example's `lr=`, `train_batch_size=` and
   `max_seq_len=`, which default to `None` and then keep the value from the
   layers below.

For an RL job on an MoE model, router replay is filled in after all three
layers; see the [Math GRPO README](rl/README.md#router-replay). `build_job` adds
the source checkpoint and the debug image tag last and logs the final job.
`job_config=FILE` replaces the first two layers with your own job file, used as
is; the command-line values still apply.

The distillation example builds two jobs instead, through
`make_distillation_config`: a student composed like an RL job (training +
sampling, no router replay) with `distillation/overrides.json` and
`student_max_seq_len=` applied, and a sampling-only teacher sized by
`teacher_max_seq_len=`. See the
[distillation README](distillation/README.md#choose-the-jobs).

Overrides merge into the job by these rules:

- Dictionaries merge key by key, and `null` removes a key.
- `sub_job_configs` entries are matched by `job_type` and merged, so an
  override can change one sub-job without restating the others. An entry
  whose `job_type` repeats, or matches no sub-job, is an error.
- All other lists are replaced whole.

This override, for example, changes only the training sub-job's router replay
cap:

```json
{"sub_job_configs": [{"job_type": "training",
  "training_config": {"router_replay": {"max_cache_bytes": 8589934592}}}]}
```

## Create a job config from the command line

`python -m examples.utils` runs the same `make_config` and writes the job as
JSON. Pick catalog files with `--model` (the exact model ID), `--context`,
`--hardware` (default `h200`) and `--lora`; add `--inference` for the
inference job or `--rl` for the RL job, and `--override FILE` (repeatable,
applied in order) for your changes:

```bash
# A training job: Qwen3-8B LoRA at standard context on H200
python -m examples.utils --model Qwen/Qwen3-8B --context standard --hardware h200 --lora -o qwen3-8b-training.json

# The matching inference job
python -m examples.utils --model Qwen/Qwen3-8B --context standard --hardware h200 --lora --inference -o qwen3-8b-inference.json

# An RL job with an override file (router replay uses the fixed default caps)
cat > rl-override.json <<'JSON'
{"sub_job_configs": [{"job_type": "training", "training_config": {"optimizer": {"lr": 2e-5}}}]}
JSON
python -m examples.utils --model Qwen/Qwen3.6-35B-A3B --context long --hardware h200 --lora --rl \
  --override rl-override.json -o qwen3.6-rl.json
```

For an MoE RL job, router replay is filled in with fixed default caps (5 GiB
training, 20 GiB sampling); `--rollout-batch-size` no longer sizes them and is
kept only for backward compatibility. A model, context, hardware or mode the
catalog does not have fails with the list of files that exist for that model.
The second form builds from your own files instead, and `--router-replay`
fills in router replay for an MoE model:

```bash
python -m examples.utils --training-config qwen3-8b-training.json \
  --inference-config qwen3-8b-inference.json --rl -o qwen3-8b-rl.json
```

`-o -` writes the job to stdout. Submit the result with `ct submit` (`ct` is
short for `cortex-training`), checking it first with `--dry-run`, or pass it
to an example as `job_config=`. Either way, the connection still comes from
`config=` or the CLI's own connection settings, never from the job file:

```bash
ct submit --dry-run qwen3.6-rl.json
ct submit qwen3.6-rl.json

python -m examples.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=qwen3.6-rl.json
```

## What goes where when you add something

| You want to | Add or change |
|---|---|
| Train on a new RL task | A new folder `rl/<task>/` with `train.py`, `task.py`, `example.yaml`; reuse `rl/runner.py` |
| Fine-tune on a new dataset | A dataset entry in `sft/conversational/task.py`, or a new `sft/<task>/` folder |
| Support a new model or GPU type | Files in `model-catalog/models/<model>/`; no code in `examples/` |
| Change a default for every RL run | `rl/overrides.json` |
| Change one run | Command-line arguments, or edit the file in `rl/configs/` and pass it as `job_config=` |
| Show a raw API call | A JSON file in `api/` and a row in its README |

## Example Contract

Every runnable example should provide:

- A README with outcome, prerequisites, hardware, commands, expected results,
  CLI knobs, how its job is built, evaluation, and troubleshooting
- `example.yaml` metadata used by the compatibility catalog. The model
  catalog, not the metadata, says which models an SFT, RL or inference example
  supports.
- A runnable entry point
- `last_validated` in `example.yaml`: the date of the last verified run, or `null`
  when the example has not been validated since it last changed
- Tests for local validation logic where practical
- Attribution when adapted from an upstream cookbook or framework

See [the example template](../docs/contributing/example-template.md) before adding
a new workflow.
