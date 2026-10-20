# Serve a Training Checkpoint

The [inference example](../../../examples/inference/README.md) creates a standalone
sampling endpoint, either from original Hugging Face weights or from a
weights-only checkpoint produced by a training run:

```bash
python -m examples.inference.serve \
  config=/path/to/config.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

Sampling requires a `weights-only` checkpoint; resumable checkpoints carry
optimizer state and are not loadable by the sampling runtime. The sampling job is
independent, so the training job can be cancelled once the checkpoint is saved.

GPU count and vLLM settings come from the model's inference file in the
[model catalog](../../../model-catalog/README.md). Pick it with `model=` and
`hardware=` as for the training run, `lora=true` for a LoRA checkpoint so the
sampler's adapter matches it, and a `context=` the model has an inference file
in. That need not be the training run's context: some models have standard
training files but only long inference files. The training examples print a
generate or evaluate command with these arguments after they save. For the
equivalent raw client calls, see
[Start sampling from saved weights](../../reference/rest-api.md#134-start-sampling-from-saved-weights).

Readiness checks, scaling and teardown are not yet documented.
