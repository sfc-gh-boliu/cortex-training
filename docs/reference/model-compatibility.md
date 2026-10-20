# Model Compatibility

The [model catalog](../../model-catalog/README.md) is the source of truth for
inference and training support. Each model has a folder under
[`model-catalog/models/`](../../model-catalog/models/) with a `model.json` and
one job file per supported configuration: standard or long context, GPU
hardware (`h200`, `b200` or `b300`), training or inference, and LoRA or full.
A file is a complete create-job body for that combination, and its existence is
the claim of support. Standard-context files use a `max_seq_len` of at most
32,768 tokens; long-context files use more. RL jobs combine a model's training
and inference files. Treat GPU counts and optimization values as starting
points that might need adjustment for a specific workload.

For GLM 5.2 on the recommended H200 configuration, use
`zai-org/GLM-5.2-FP8` for 1M-token inference. The unquantized
`zai-org/GLM-5.2` checkpoint is limited to the existing 32K configuration
because its weights and a 1M-token MLA cache do not fit on 16 H200 GPUs.
The 1M FP8 profile uses `gpu_memory_utilization: 0.9`. A live 80% attempt
failed during engine startup after 588 seconds, while 32K at 80% and 1M at 90%
started successfully. This is consistent with the profile's narrow calculated
memory margin at 80%, although the retained log did not include the engine-core
error needed to confirm the exact cause.

The DeepSeek V4 and GLM 5.2 FP8 1M recommendations were validated with
1,048,575 pre-tokenized input tokens plus one generated token, not only with
short prompts.

For `Qwen/Qwen3.6-35B-A3B`, LoRA training and checkpoint save/load require the
`prime_rl` model provider. Current PrimeRL LoRA support for this MoE model is
limited to `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Routed expert LoRA is not
yet supported.

Validate catalog changes with:

```bash
python model-catalog/validate_catalog.py
python -m pytest tests/test_model_catalog.py -q
```
