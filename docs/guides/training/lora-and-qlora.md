# LoRA and QLoRA

LoRA training is implemented by the
[conversational SFT example](../../../examples/sft/README.md).
`lora=true` picks the model's LoRA file in the
[model catalog](../../../model-catalog/README.md): a training file whose
`training_config` carries a `peft_config` block trains LoRA, and one without it
trains all parameters.

The example defaults to full-parameter training, so LoRA is opt-in:

```bash
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  lora=true
```

Every model the catalog can train has LoRA files; each model's folder under
[`model-catalog/models/`](../../../model-catalog/models/) lists them, with the
adapter rank and learning rate in each file.

To change the adapter, set `peft_config.r`, `lora_alpha` or, for dense Qwen
models, `target_modules` in an override file, or write the job to a file with
`python -m examples.utils`, edit it, and pass it back with `job_config=`; see
the [example index](../../../examples/README.md). Keep the shipped
attention-only target list for `Qwen/Qwen3.6-35B-A3B`: its PrimeRL backend does
not yet support LoRA on routed experts, and Hugging Face MLP target names are
no-ops. The example README documents the whole
[job-config guidance](../../../examples/sft/README.md).

QLoRA is not implemented: there are no quantization settings in the example or in
the job-config schema.
