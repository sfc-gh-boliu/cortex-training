# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a `messages` column. The default dataset is a
one-example memorize task: when prompted `Who trained you?`, answer
`Snowflake AI Research`. Hugging Face chat datasets work as well. `openai/gsm8k`
is mapped from `question`/`answer` into that same chat format. The entry point
supports LoRA and full-parameter training, logs `train_nll`, and saves a
weights-only checkpoint. The example lives in `conversational/`: its entry
point `train.py`, its dataset helpers in `task.py`, its identity evaluation in
`identity.py`, its metadata in `example.yaml`, and its bundled data under
`data/`. The entry point calls the shared SFT loop in `runner.py`.

## Hardware

Training GPUs are `n_gpus` in the job's training file (`NUM_TRAINING_GPUS`).
Actual requirements depend on model size, sequence length, precision, and
whether LoRA or dense training is used. Check the account capacity before
submitting:

```bash
cortex-training capacity
```

## Run

```bash
python -m examples.sft.conversational.train \
  config=/path/to/config.json
```

`config=` is the Snowflake connection file only. Copy
`docs/getting-started/connection.json.template` and adjust it.

Defaults are `Qwen/Qwen3-8B` at standard context on H200, full-parameter,
thinking off, and 100 steps. It uses the builtin `who_trained_you` dataset as
default. The default run will be finished within few minutes.

## Choose the job

The example builds its job from the
[model catalog](../../model-catalog/README.md) in three layers, each
overriding the one before:

1. The model's training file for `model=` (the exact model ID), `context=`
   (`standard` or `long`), `hardware=` (`h200`, `b200` or `b300`; default
   `h200`) and `lora=` (`true` or `false`):
   `model-catalog/models/<slug>/<context>/<hw>-training-<lora|full>.json`.
2. `sft/overrides.json`, applied to every SFT run on top of the catalog, in any
   model. It ships empty (only a `_comment` describing the file); add fields to
   change a default for all SFT runs. A reserved `_comment` key (a free-text
   note) is stripped before merging and never reaches the job.
3. The command line: `lr=`, `train_batch_size=` and `max_seq_len=`. Each
   defaults to `None`, which keeps the value from the layers below.

The job is logged before it is submitted. A combination the catalog does not
have fails with the list of files that exist for that model.

`job_config=JOB_CONFIG` replaces the first two layers with your own create-job
JSON file, used as is; `lr=`, `train_batch_size=` and `max_seq_len=` still
apply. Write one with `python -m examples.utils` and edit it; see
[Create a job config from the command line](../README.md#create-a-job-config-from-the-command-line).
Overrides merge by the rules in
[How an example builds its job](../README.md#how-an-example-builds-its-job).

```bash
# Qwen3-8B LoRA
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  lora=true

# Qwen3.5-9B LoRA / full, 64K context
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.5-9B context=long lora=true

python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.5-9B context=long

# Qwen3.8-27B LoRA / full, 32K or 64K context
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.8-27B context=standard lora=true

python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.8-27B context=long

# Qwen3.6-35B-A3B LoRA / full, 262K context
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B context=long lora=true

python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B context=long

# A smaller batch and a shorter sequence than the catalog file
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  train_batch_size=2 max_seq_len=8192 lr=1e-5
```

## Common Variations

```bash
# Thinking-on Qwen3 (must also pass enable_thinking=true to sample)
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  enable_thinking=true

# Different chat dataset
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/no_robots

python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/ultrachat_200k dataset_split=train_sft

# Grade-school math (openai/gsm8k). The example maps question/answer → messages.
# Train split has 7473 rows. With the Qwen3-8B LoRA file (batch 4) one epoch is
# 1868 steps.
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  lora=true \
  dataset=openai/gsm8k \
  dataset_split=train \
  max_steps=1868

# Builtin identity JSONL: paraphrases of Who trained you? → Snowflake AI Research
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  dataset=identity
```

Dataset loading lives in `conversational/task.py`, not `train.py`. To add a source:

- JSONL with `messages`: put it under `conversational/data/` and append a `BuiltinJsonl` to `CHAT_DATASETS` (see `who_trained_you` and `identity`).
- Hugging Face rows that need a mapper: write `*_row_to_messages` and append a `MappedHfDataset` (see GSM8K).
- Hugging Face sets that already have `messages` (No Robots, UltraChat): pass `dataset=org/name`. No registry entry.

Then list the dataset in `conversational/example.yaml` if it should show up in the catalog.

LoRA, GPU count, batch shape, sequence length, and MoE come from the job (see
[Choose the job](#choose-the-job)). Set `wandb_project` to log to Weights &
Biases after
`uv pip install wandb` and `export WANDB_API_KEY` / `export WANDB_BASE_URL`.
Set `sf_tracking=True` to log to Snowflake experiment tracking after
`uv pip install "snowflake-ml-python>=1.19.0"`.

## Job config JSON

The example submits one create-job body with a single training sub-job. Every
training file in the model catalog has this shape, with the job-level
`hardware` field set to the file's prefix, and so must a file you pass with
`job_config=JOB_CONFIG`.

```json
{
  "hardware": HARDWARE,
  "sub_job_configs": [
    {
      "job_type": "training",
      "model_name": MODEL_NAME,
      "dtype": DTYPE,
      "seed": SEED,
      "training_config": {
        "n_gpus": NUM_TRAINING_GPUS,
        "max_seq_len": MAX_SEQ_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "sp_size": SEQUENCE_PARALLEL_SIZE,
        "gradient_clipping": GRADIENT_CLIPPING,
        "model_provider": MODEL_PROVIDER, // Use prime_rl for MoE
        "attn_implementation": "flash_attention_3",
        "optimizer": {
          "name": "AdamW",
          "lr": LEARNING_RATE,
          "weight_decay": WEIGHT_DECAY,
          "betas": [ADAM_BETA1, ADAM_BETA2],
          "eps": ADAM_EPS
        },
        "ds_config": {
          "zero_optimization": {
            "stage": ZERO_STAGE
          },
          "bf16": {
            "enabled": true
          }
        },

        // Optional LoRA. Omit for full-parameter.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },

        // Optional expert parallelism for MoE. With model_provider prime_rl,
        // keep fused cross-entropy off (this block, or model_options with
        // EP_SIZE > 1): the chunked LM head set below cannot be combined with
        // it.
        "ac_config": {"mode": "full", "freq": 1},
        "prime_rl": {
          "fused_lm_head_token_chunk_size": 8192,
          "fused_cross_entropy": false
        },
        "ep_size": EP_SIZE
      }
    }
  ]
}
```

The target list above is for dense Qwen models. For
`Qwen/Qwen3.6-35B-A3B`, the catalog's LoRA files use
`model_provider: "prime_rl"` and attention-only targets (`q_proj`, `k_proj`,
`v_proj`, and `o_proj`). PrimeRL does not yet support LoRA on routed experts;
`gate_proj`, `up_proj`, and `down_proj` do not select those parameters.
The Qwen3.6 long-context files use sequence parallel size 8 and logical batch 1
to run at the model's 262K context limit.

The `prime_rl` entries in the template above are deprecated but keep working.
On server versions that support `model_options`, with `EP_SIZE` > 1, the same
settings can be written inside `training_config` in place of the `prime_rl`
block, as the entry below; see
[the REST API reference](../../docs/reference/rest-api.md#model_options). The
Qwen3.6 long-context training files still use the block, which older servers
need.

```json
"model_options": {"lm_head": {"token_chunk_size": 8192, "cross_entropy": false}}
```

On server versions that have `numerics.fp32_parity`, Qwen3.5-family models run
with it on by default: dense models (for example `Qwen/Qwen3.8-27B`) and
Qwen3.5-MoE models with `EP_SIZE` > 1 (the Qwen3.6-35B-A3B configs). It raises
GPU memory. Memory was not measured for rows near `max_seq_len` with the
Qwen3.8-27B catalog files. If a job runs out of memory, raise `sp_size`
(at most 8 for Qwen3.8-27B) or turn the option off by adding
`"numerics": {"fp32_parity": false}` to the `model_options` entry of
`training_config`, inside the same object if the config already has one (two
`model_options` keys in one JSON object keep only the last). A server that
has the option but not `model_options` support ignores that entry; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

For dense long-context training, set an integer
`fused_lm_head_token_chunk_size` in `training_config`. The server then computes
per-token log probabilities in chunks without materializing the full
`[sequence, vocabulary]` logits tensor. The setting saves memory and does not
change the loss.

```bash
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  lr=LEARNING_RATE \
  train_batch_size=TRAIN_BATCH_SIZE \
  max_seq_len=MAX_SEQ_LEN \
  dataset=DATASET \
  max_steps=MAX_STEPS \
  enable_thinking=ENABLE_THINKING \
  wandb_project=WANDB_PROJECT \
  sf_tracking=SF_TRACKING
```

## Logs and Expected Results

Metrics and configuration are written under `log_path`. To send the same
metrics to Weights & Biases:

```bash
uv pip install wandb
export WANDB_API_KEY=...
export WANDB_BASE_URL=...
```

Then pass `wandb_project=WANDB_PROJECT` on the train command.

To log the same metrics to Snowflake experiment tracking:

```bash
uv pip install "snowflake-ml-python>=1.19.0"
```

Then pass `sf_tracking=True` on the train command. The server ties each run to
a Snowflake experiment; results are viewable in Snowsight under
**AI & ML > Experiments**. See the
[Snowflake ML Experiments docs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiments).

Each step logs `train_nll` (the server's `avg_loss`) and `loss_weight_sum`, the
sum of `loss_mask` over the batch. On the default memorize task, `train_nll`
should fall quickly and `loss_weight_sum` should stay constant from step to
step. After save, the example prints one generate command whose `model=`,
`context=`, `hardware=` and `lora=` pick the catalog inference file that serves
the checkpoint: the run's model, hardware and mode, with the same adapter for
LoRA, at the run's context or, when the model has no such file there, at the
other one (some models have standard training files but only long inference
files). When no catalog file fits, the command has
`job_config=INFERENCE_JOB_CONFIG` instead, and a warning says what that
inference job needs. When running that command, Assistant text should be
`Snowflake AI Research`.

```bash
python -m examples.inference.generate \
  config=/path/to/config.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  temperature=0 \
  prompt="Who trained you?"
```

After GSM8K SFT, generate a word problem (`temperature=0`) and check that the
completion ends with a `#### <number>` answer. For a numeric before/after score
on contest math, run MATH-500 against the same checkpoint (harder than GSM8K):

```bash
python -m examples.inference.evaluate \
  config=/path/to/config.json \
  lora=true \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024
```

After identity SFT, score the percent of completions that contain
`Snowflake AI Research` (default prompts: `conversational/data/identity_eval.jsonl`):

```bash
python -m examples.inference.evaluate \
  config=/path/to/config.json \
  lora=true \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=identity \
  temperature=0 \
  max_tokens=128
```

## Notebooks

- `conversational/qwen3_8b_sft_training.ipynb`
- `conversational/qwen3_8b_sft_training_multiplex.ipynb`

Both build their job with `make_config(model="Qwen/Qwen3-8B", context="standard")`
from the model catalog, with a small override that is for the demo only
(128-token sequences in batches of 8; the multiplex notebook adds
`multiplex_job_id`). Start Jupyter inside the repository: the first cell puts
the repository first on `sys.path` so `examples` imports from it, and stops with
an error if another installed package named `examples` shadows it. It reads the
PAT from the connection file you created during
[setup](../../docs/getting-started/setup.md) (`~/cortex-training-config.json`,
or the path in `CORTEX_TRAINING_CONFIG`), so the notebook runs top to bottom
wherever the kernel's working directory is.
