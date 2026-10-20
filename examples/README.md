# Cortex Training Examples

Examples are runnable, end-to-end workflows organized by task. Each example owns
its code, documentation, metadata, optional notebooks, and the JSON job configs
under its `configs/` directory.

## Available Examples

| Example | Method | Dataset | Status |
|---|---|---|---|
| [Conversational SFT](sft/README.md) | LoRA or full-parameter SFT | Hugging Face chat datasets, GSM8K, identity | Runnable |
| [Math GRPO](rl/README.md) | Reinforcement learning | Hendrycks MATH and MATH-500 | Runnable |
| [On-policy distillation](distillation/README.md) | Distillation from a frozen teacher | GSM8K | Runnable |
| [Inference endpoint](inference/README.md) | Serve, generate, eval | Open weights, checkpoints, MATH-500, GSM8K, identity | Runnable |

GRPO can also be run from [SkyRL](frameworks/skyrl/gsm8k_grpo/README.md), which uses its
own entry point and install rather than the prerequisites below.

Two folders hold reference material rather than runnable workflows:
[`api/`](api/README.md) has raw request bodies for `cortex-training submit` and
the Python client, and [`frameworks/`](frameworks/README.md) covers running with
outside frameworks such as SkyRL and tinker-cookbook.

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

Examples are Python modules so they can share code without path manipulation:

```bash
python -m examples.sft.conversational.train config=/path/to/config.json
python -m examples.rl.math_grpo.train config=/path/to/config.json
python -m examples.distillation.on_policy.train config=/path/to/config.json
python -m examples.inference.serve config=/path/to/config.json
```

`config=` is the Snowflake PAT/connection file. See each example README for
hardware, expected metrics, common variations, and the JSON job configs
loaded from `configs/` (SFT and GRPO examples, plus inference configs for
every catalog model).

## Example Contract

Every runnable example should provide:

- A README with outcome, prerequisites, hardware, commands, expected results,
  CLI knobs, JSON job configs, evaluation, and troubleshooting
- `example.yaml` metadata used by the compatibility catalog
- A runnable entry point
- `last_validated` in `example.yaml`: the date of the last verified run, or `null`
  when the example has not been validated since it last changed
- Tests for local validation logic where practical
- Attribution when adapted from an upstream cookbook or framework

See [the example template](../docs/contributing/example-template.md) before adding
a new workflow.
