# Math GRPO

Train a model with grouped policy optimization on Hendrycks MATH and evaluate
against MATH-500. The example creates colocated training and sampling sub-jobs,
generates rollouts, scores them, trains, and synchronizes weights. After the
final save it logs a `python -m examples.inference.evaluate` command.
The example lives in `math_grpo/`: its entry point `train.py`, its MATH
prompting, scoring and evaluation in `task.py`, and its metadata in
`example.yaml`. The entry point calls the shared GRPO loop in `runner.py`.

## Hardware

The default configuration requests four training GPUs and four sampling GPUs.
Reduce or increase these values together with batch settings, and verify
capacity before submission:

```bash
cortex-training capacity
```

## Run

```bash
# Qwen3-8B LoRA (default)
python -m examples.rl.math_grpo.train \
  config=/path/to/config.json

# Qwen3-8B full-parameter
python -m examples.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_full.json

# Qwen3.6-35B-A3B LoRA
python -m examples.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_lora.json \
  max_tokens=2048

# Qwen3.6-35B-A3B full-parameter
python -m examples.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_full.json \
  max_tokens=2048

# Qwen3.8-27B full-parameter
python -m examples.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen38_27b_full.json \
  weight_sync_bucket_size=3221225472 \
  max_tokens=2048
```

`config=` is the Snowflake connection file. Adapt from
`docs/getting-started/connection.json.template`.

The default job body is `configs/qwen3_8b_lora.json`. The default run will be finished within 1h.

## Customizability

```bash
python -m examples.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  max_tokens=MAX_TOKENS \
  problems_per_batch=PROBLEMS_PER_BATCH \
  group_size=GROUP_SIZE \
  max_steps=MAX_STEPS \
  n_test=N_TEST \
  weight_sync_bucket_size=WEIGHT_SYNC_BUCKET_SIZE_BYTES \
  wandb_project=WANDB_PROJECT \
  sf_tracking=SF_TRACKING
```

LoRA, GPU counts, sequence length, and MoE live in the job-config JSON. Set
`wandb_project` to log to Weights & Biases after `uv pip install wandb` and
`export WANDB_API_KEY` / `export WANDB_BASE_URL`. Set `sf_tracking=True` to log
to Snowflake experiment tracking after
`uv pip install "snowflake-ml-python>=1.19.0"`.

`weight_sync_bucket_size` optionally overrides the transfer-bucket size in
bytes. When omitted, the example leaves the field out and uses the server
default. Set it above the largest individual model tensor.

The example loads one create-job body with colocated sampling and training
sub-jobs. Pass a shipped example or a copy with `job_config=JOB_CONFIG`.

```json
{
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

        // Optional LoRA. Omit for full-parameter. Keep in sync with training.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },

        // Optional router replay for MoE training.
        "router_replay": {"enabled": true, "max_cache_bytes": 17179869184}
      }
    },
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
        "mb_spec": {
          "max_tokens_per_mb": MAX_TOKENS_PER_MB
        },
        "optimizer": {
          "name": "AdamW",
          "lr": LEARNING_RATE,
          "weight_decay": WEIGHT_DECAY,
          "betas": [ADAM_BETA1, ADAM_BETA2],
          "eps": ADAM_EPS
        },
        "ds_config": {
          "train_batch_size": TRAIN_BATCH_SIZE,
          "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
          "gradient_accumulation_steps": TRAIN_BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_TRAINING_GPUS),
          "zero_optimization": {
            "stage": ZERO_STAGE,
            "reduce_scatter": true
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

        // Optional expert parallelism for MoE. NUM_TRAINING_GPUS must be a multiple of EP_SIZE.
        "ac_config": {"mode": "full", "freq": 1},
        "prime_rl": {
          "fused_lm_head_token_chunk_size": 8192,
          "fused_cross_entropy": false
        },
        "ep_size": EP_SIZE,

        // Optional router replay for MoE training. Add "mode": "best_effort" to
        // route rows the sampler no longer holds (e.g. after a sampler restart)
        // with the trainer's own gate instead of failing the step.
        "router_replay": {"enabled": true, "max_cache_bytes": 2147483648}
      }
    }
  ]
}
```

The `prime_rl` entries in the config above are deprecated but keep working. On
server versions that support `model_options`, and with `EP_SIZE` > 1, the
same settings can be written inside `training_config`, in place of the
`prime_rl` block, as the entry below; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

```json
"model_options": {"lm_head": {"token_chunk_size": 8192, "cross_entropy": false}}
```

The Qwen3.6-35B-A3B RL and SFT example configs, `examples/api/training.json`,
`examples/api/rl.json` and the `rl` and `standard-context` model-catalog
profiles still use the `prime_rl` block, which older servers need.

On server versions that have `numerics.fp32_parity`, Qwen3.5-family models run
with it on by default: dense models (for example `Qwen/Qwen3.8-27B`) and
Qwen3.5-MoE models with `EP_SIZE` > 1 (the Qwen3.6-35B-A3B configs). It raises
GPU memory. Memory was not measured for rows near `max_seq_len` with the
shipped Qwen3.8-27B configs. If a job runs out of memory, raise `sp_size`
(at most 8 for Qwen3.8-27B) or turn the option off by adding
`"numerics": {"fp32_parity": false}` to the `model_options` entry of
`training_config`, inside the same object if the config already has one (two
`model_options` keys in one JSON object keep only the last). Turning it off
can bring back part of the difference between the trainer's and the sampler's
log-probabilities, which matters for RL. A server that has the option but not
`model_options` support ignores that entry; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

The target list above is for dense Qwen models. For
`Qwen/Qwen3.6-35B-A3B`, use the shipped LoRA config with
`model_provider: "prime_rl"` and attention-only targets (`q_proj`, `k_proj`,
`v_proj`, and `o_proj`). PrimeRL does not yet support LoRA on routed experts;
`gate_proj`, `up_proj`, and `down_proj` do not select those parameters.

## Router-replay capacity planning

The router-replay capacity planner in `examples/utils.py` derives the
per-worker `router_replay.max_cache_bytes` caps before you create a job. It
only calculates. It does not call the service, edit a job config, or reserve
GPU memory. Put `CachePlan.max_cache_bytes` on the sampling and training
`router_replay` objects yourself. When bootstrapping with an explicit cap, pass
the training result to `bootstrap_router_replay(..., max_cache_bytes=...)`.

One full-sequence capture occupies:

```text
entry_bytes = seq_len * num_moe_layers * topk * dtype_bytes
```

`moe_geometry_from_config` reads `num_hidden_layers` and `num_experts_per_tok`
from the config root or `text_config`, and rejects conflicting values. Every
recognized expert-count field (`num_local_experts`, `n_routed_experts`,
`num_experts`, `moe_num_experts`) must agree, be positive, and be at least
`topk`. A count of 256 or fewer selects `dtype_bytes=1` because those expert
ids fit in `uint8`. A count from 257 through 65536 selects `dtype_bytes=2`
(`uint16`). A larger count is rejected. `plan_sampling_cache_bytes` and
`plan_training_cache_bytes` use that width unless you pass `dtype_bytes`.

The sampling cap is per sampling replica, and the training cap is per
data-parallel shard:

```text
backlog = max_inflight_rollouts
        + ceil(batch_size * over_generation_multiplier)
        + batch_size * max(async_depth, 1)
per_replica_rollouts = ceil(backlog / replicas * routing_skew)
sampling_cap = ceil(per_replica_rollouts * entry_bytes * safety_factor)

resident_rows = ceil(batch_size / dp_size)
training_cap = ceil(resident_rows * entry_bytes * safety_factor)
```

`max_inflight_rollouts` defaults to `batch_size`. `routing_skew` defaults to
`2.0` and `safety_factor` defaults to `1.2`. Replica and data-parallel counts
come from `derive_num_sampling_workers` and `derive_dp_size`; known GPU counts
must divide exactly by the parallel widths. The skew estimate is not an
overflow-proof bound. Raise it, or set a manual cap, when a worker retains
more replay entries than the average backlog times the skew.

`plan_vllm_gpu_memory_utilization` lowers vLLM's `gpu_memory_utilization` by
the sampling cap divided by the GPU's byte capacity. It raises when that
reservation exceeds half the GPU or would push utilization below `0.05`.

```python
from examples import utils as rrp

geometry = rrp.resolve_model_moe_geometry(
    "org/moe-model",
    num_moe_layers_override=40,
    topk_override=8,
    num_experts_override=512,
)
assert geometry.dtype_bytes == 2

backlog = rrp.derive_backlog_rollouts(batch_size=16, async_depth=2)
sampling_workers = rrp.derive_num_sampling_workers(
    sampling_gpus=8,
    sampling_tensor_parallel_size=2,
)
sampling_plan = rrp.plan_sampling_cache_bytes(
    seq_len=8192,
    geometry=geometry,
    backlog_rollouts=backlog,
    num_sampling_workers=sampling_workers,
)
training_plan = rrp.plan_training_cache_bytes(
    seq_len=8192,
    geometry=geometry,
    resident_rows=rrp.derive_resident_rows(
        batch_size=16,
        dp_size=rrp.derive_dp_size(training_gpus=8, sp_size=2),
    ),
)
sampling_cap = sampling_plan.max_cache_bytes
training_cap = training_plan.max_cache_bytes
```

The same sequence length, layer count, and top-k with 512 experts stores each
expert id in two bytes, so each capture is twice as large as one from a model
with 256 or fewer experts. See
[REST API section 8.2](../../docs/reference/rest-api.md#82-trainingconfig) for the
`max_cache_bytes` field.

## Evaluation and Logs

Training logs reward, correctness, format, rollout counts, and loss. Held-out
MATH-500 generate eval runs every `eval_every` batches (`test/env/all/correct`;
`eval_every=0` skips it). To mirror local metrics to Weights & Biases:

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

After save, the example prints one eval command.
`examples.inference.evaluate` uses the same few-shot prompt, grader, and
decoding settings.

```bash
python -m examples.inference.evaluate \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=1.0 \
  max_tokens=MAX_TOKENS
```
