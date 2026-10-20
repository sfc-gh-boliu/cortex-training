# Cortex Training model catalog

This directory is the customer-facing source of truth for the models Cortex
Training supports, the context lengths and hardware each one runs on, and the
tuned job configuration for each of those combinations.

## Layout

```text
model-catalog/
├── models/
│   └── <slug>/                    # one folder per model
│       ├── model.json             # model ID, license, architecture, validation dates
│       ├── standard/              # standard context: max_seq_len up to 32,768
│       │   └── <hw>-<kind>-<mode>.json
│       └── long/                  # long context: the same file names
├── schema.json                    # JSON schema for model.json
├── validate_catalog.py            # checks every model folder
└── smoke_test_catalog.py          # dry-runs or submits one catalog file
```

`<slug>` is the lowercased last segment of the model ID, so
`Qwen/Qwen3.6-35B-A3B` lives in `models/qwen3.6-35b-a3b/`.

Each job file is a complete create-job body, named `<hw>-<kind>-<mode>.json`:

- `<hw>` is `h200`, `b200` or `b300`. The file sets the job-level `hardware`
  field to the same value (`"hardware": "H200"` in an `h200-*.json` file). This
  is the existing REST field, which defaults to H200.
- `<kind>` is `training` (one training sub-job) or `inference` (one sampling
  sub-job).
- `<mode>` is `lora` or `full`. `lora` files carry `peft_config`; `full` files
  don't.

The rules:

- If a file exists, that model supports that configuration on that hardware.
  No other file refers to it by path.
- There is no RL file. An RL job combines a model's training and inference
  files of the same context, hardware and mode.
- A model's long-context files have a larger `max_seq_len` than its standard
  ones.
- Today's files are all H200. B200 and B300 files are added as they are
  benchmarked.

`model.json` holds:

- `model_id`: the exact model ID that every sub-job's `model_name` uses.
- `license`: the license name and an `https` URL.
- `architecture`: `dense` or `moe`. MoE models also record
  `moe: {num_moe_layers, topk, num_experts}`, so router replay can be planned
  without downloading the model config. The values are what the router-replay
  planner reads from the model's public `config.json`: `num_hidden_layers`,
  `num_experts_per_tok`, and the expert count.
- `last_validated`: one entry per job file, keyed `<context>/<hw>-<kind>-<mode>`,
  with the date of its last successful live run or `null`.

`schema.json` describes `model.json`; the job files follow the create-job
schema in [the REST API reference](../docs/reference/rest-api.md#8-create-job-schemas).

Qwen3.6-35B-A3B LoRA training uses the `prime_rl` model provider so checkpoints
can be saved and loaded. PrimeRL currently supports LoRA on the attention
projections (`q_proj`, `k_proj`, `v_proj`, and `o_proj`) for this model. Routed
expert LoRA is not yet supported, and Hugging Face MLP names such as
`gate_proj`, `up_proj`, and `down_proj` are no-ops on the PrimeRL model.

## Validation

`validate_catalog.py` checks, for every model folder:

- File names match `{h200,b200,b300}-{training,inference}-{lora,full}.json`,
  and each file is a valid job definition: it passes the checks
  `cortex-training submit --dry-run` runs, uses only `hardware` and
  `sub_job_configs` at the top level, and gives each sub-job the fields the
  create-job schema requires.
- The folder slug derives from the model ID in `model.json`, and no two models
  share a slug or ID.
- Each sub-job's model name matches `model.json`, the job `hardware` matches
  the file's prefix, and the mode matches: every `lora` file carries a valid
  LoRA `peft_config` object (not `null`), and no `full` file has one.
- MoE models have `moe` geometry in `model.json`.
- Each sub-job uses 1, 2, 4, or a multiple of 8 GPUs, and tensor-parallel and
  sequence-parallel sizes divide `n_gpus`.
- A model's long-context files have a larger `max_seq_len` than its standard
  ones.
- `last_validated` has an entry for each of the model's files and no others.

## Updating the catalog

1. Add, change or remove the job files in the model's folder, and add a new
   model as a new folder with its `model.json`. No code in `examples/` changes.
2. Set the file's `last_validated` entry to `null` until that exact file
   completes a live smoke test, then set it to the test date. For a
   maximum-context inference claim, use `--full-context-prefill`; a short
   prompt validates startup and basic generation but not the context limit.
3. Keep [`docs/_data/models.yaml`](../docs/_data/models.yaml) in sync; its
   header says how each field derives from this tree.
4. Run:

   ```bash
   python model-catalog/validate_catalog.py
   python -m pytest tests/test_model_catalog.py -q
   ```

Snowflake product documentation vendors a pinned snapshot of this directory;
it does not download mutable data during a documentation build.

## Live job validation

Dry-run the exact request body of a catalog file:

```bash
python model-catalog/smoke_test_catalog.py \
  --model-id Qwen/Qwen3.8-27B \
  --context long \
  --job inference-full
```

`--context` defaults to `standard` and `--hardware` to `h200`. `--job` names the
file's `<kind>-<mode>`.

To execute a live smoke test, set the connection values through environment
variables and add `--submit`. After the job reaches `running`, inference jobs
execute and poll a one-token `generate` request and training jobs execute and
poll a minimal `forward-backward` request. The job is then cancelled. The PAT
is intentionally not accepted as a command-line argument, which keeps it out of
shell history and process listings. Use `--debug-image-tag` when a validation
run must select a specific backend build.

The default inference probe uses a short text prompt and validates engine
startup plus basic generation. Add `--full-context-prefill` to send exactly
`max_seq_len - 1` pre-tokenized tokens and generate one token. Only the latter
validates execution at the configured context limit.

For a training file, use `--full-context-training` to execute
forward-backward with exactly `max_seq_len` tokens in every sequence in the
configured global batch. Add `--training-step` to exercise the optimizer, and
`--checkpoint-round-trip` to step, save a resumable checkpoint, load it back
into the training sub-job, and delete it. Job startup or the default
eight-token forward-backward probe does not validate a training context-limit
claim. Add `--memory-telemetry` to include the server's per-step peak GPU and
CPU memory measurements in the result.

Use `--training-sp-size` to evaluate a Ulysses sequence-parallel candidate
before changing a catalog file. The generated request sets `sp_size` and
adjusts `train_batch_size` to the logical data-parallel size
`n_gpus / sp_size`; it does not silently change `max_seq_len`.
For Qwen3.8, the degree must divide its 24 full-attention heads, 16 GDN key
heads, and 48 GDN value heads. SP8 is the largest valid common degree; SP24 is
invalid even though it divides the full-attention head count.

```bash
export SNOWFLAKE_HOST='ACCOUNT.snowflakecomputing.com'
export SNOWFLAKE_DATABASE='CORTEX_TRAINING_DB'
export SNOWFLAKE_SCHEMA='PUBLIC'
read -rs SNOWFLAKE_PAT
export SNOWFLAKE_PAT

python model-catalog/smoke_test_catalog.py \
  --model-id Qwen/Qwen3.8-27B \
  --context long \
  --job inference-full \
  --full-context-prefill \
  --submit

unset SNOWFLAKE_PAT
```

Repeat `--job` to validate several files of one model and context serially.
Supported values are `training-lora`, `training-full`, `inference-lora` and
`inference-full`. Each submitted job is cancelled after its data-plane probes
complete, including when a probe or a later job fails.
