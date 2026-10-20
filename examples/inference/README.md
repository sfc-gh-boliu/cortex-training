# Inference Endpoint

Create a standalone Cortex Training inference endpoint from open weights or a
weights-only training checkpoint. Generate or evaluate against that running job.

## Hardware

Sampling GPUs are `n_gpus` in the model's inference file in the
[model catalog](../../model-catalog/README.md). Check account capacity before
creating an endpoint:

```bash
cortex-training capacity
```

## Create an Endpoint

`config=` is the Snowflake connection file only. The default job is
`Qwen/Qwen3-8B` at standard context on H200 with full weights.

From original Hugging Face weights:

```bash
python -m examples.inference.serve \
  config=/path/to/config.json
```

From a weights-only checkpoint produced by a training example. Pass
`source_job_id` and `checkpoint_id` together, with `model=`, `context=`,
`hardware=` and `lora=` picking an inference file for the training run's model,
hardware and mode, so a LoRA checkpoint gets a sampler with the same adapter.
The context need not be the training run's: some models have standard training
files but only long inference files. The training examples print a generate or
evaluate command with these arguments after they save.

```bash
python -m examples.inference.serve \
  config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

The command waits until workers are up, prints `job_id`, and leaves the
endpoint running. Tear it down with:

```bash
cortex-training cancel JOB_ID
```

`inference_walkthrough.ipynb` shows the same create → generate → cancel loop
with the Python client. It builds its job with
`make_config(model="Qwen/Qwen3-8B", context="standard", inference=True)` and a
demo-only override to one GPU and a 2,048-token context. Start Jupyter inside
the repository: the first cell puts the repository first on `sys.path` so
`examples` imports from it.

## Choose the job

The example builds its job from the
[model catalog](../../model-catalog/README.md) in three layers, each
overriding the one before:

1. The model's inference file for `model=` (the exact model ID), `context=`
   (`standard` or `long`), `hardware=` (`h200`, `b200` or `b300`; default
   `h200`) and `lora=` (`true` or `false`):
   `model-catalog/models/<slug>/<context>/<hw>-inference-<lora|full>.json`.
2. `inference/overrides.json`, if you create one: changes for every inference
   run, in any model.
3. The command line: `max_seq_len=`, which defaults to `None` and keeps the
   value from the layers below.

The source checkpoint, when given, goes on the sampling sub-job, and the job is
logged before it is submitted. A combination the catalog does not have fails
with the list of files that exist for that model. `job_config=JOB_CONFIG`
replaces the first two layers with your own create-job JSON file, used as is.

```bash
# Qwen3-8B LoRA
python -m examples.inference.serve \
  config=/path/to/config.json \
  lora=true

# Qwen3.5-9B LoRA / full
python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3.5-9B context=long lora=true

python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3.5-9B context=long

# Qwen3.8-27B LoRA / full
python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3.8-27B context=long lora=true

python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3.8-27B context=long

# Qwen3.6-35B-A3B LoRA / full
python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B context=long lora=true

python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B context=long

# Any catalog model with an inference file, for example GLM-5.3
python -m examples.inference.serve \
  config=/path/to/config.json \
  model=zai-org/GLM-5.3 context=long
```

## Job config JSON

The example submits one create-job body with a single sampling sub-job. Every
inference file in the model catalog has this shape, with the job-level
`hardware` field set to the file's prefix, and so must a file you pass with
`job_config=JOB_CONFIG`.

```json
{
  "hardware": HARDWARE,
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": MODEL_NAME,
      "dtype": DTYPE,
      "seed": SEED,
      "inference_config": {
        "max_seq_len": MAX_SEQ_LEN,
        "n_gpus": NUM_SAMPLING_GPUS,
        "vllm_config": {
          "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
          "gpu_memory_utilization": GPU_MEMORY_UTILIZATION
        },

        // Optional LoRA. Omit for full-parameter.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
      }
    }
  ]
}
```

The target list above is for dense Qwen models. A Qwen3.6 LoRA checkpoint uses
the attention-only target list of the catalog's Qwen3.6 LoRA files; the
sampling and training PEFT configurations must match.

## Examples

Point these at a running `job_id` from `serve`, or omit `job_id` to create a
one-shot endpoint that exits (and releases GPUs) when the example finishes.

### Generate

```bash
python -m examples.inference.generate \
  config=/path/to/config.json \
  job_id=JOB_ID \
  prompt="Who trained you?"
```

From original weights or a checkpoint, without a pre-created endpoint:

```bash
python -m examples.inference.generate \
  config=/path/to/config.json \
  prompt="Who trained you?"

python -m examples.inference.generate \
  config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  prompt="Who trained you?"
```

### Evaluate (MATH-500)

```bash
python -m examples.inference.evaluate \
  config=/path/to/config.json \
  job_id=JOB_ID

python -m examples.inference.evaluate \
  config=/path/to/config.json

python -m examples.inference.evaluate \
  config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

### Evaluate (GSM8K test)

Exact-match on the `####` final answer. Use `temperature=0` for a deterministic compare.

```bash
python -m examples.inference.evaluate \
  config=/path/to/config.json \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024

python -m examples.inference.evaluate \
  config=/path/to/config.json \
  lora=true \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024
```

### Evaluate (identity)

Score is the percent of completions that contain `Snowflake AI Research`.
Default prompts are `examples/sft/conversational/data/identity_eval.jsonl`.

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

Score an existing `generate` JSONL (needs a `completion` field):

```bash
python -m examples.sft.conversational.identity \
  completions_file=/tmp/identity_eval.jsonl
```
