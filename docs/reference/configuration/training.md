# Training Configuration

The examples take two separate things:

- `config=` — the Snowflake connection file (account host, PAT, database,
  schema). See [Set up the client](../../getting-started/setup.md).
- The job — a create-job body holding the whole training configuration. The
  example builds it from the [model catalog](../../../model-catalog/README.md)
  file for `model=`, `context=`, `hardware=` and `lora=`, then applies the
  example's optional `overrides.json` and the command-line `lr=`,
  `train_batch_size=` and `max_seq_len=`, each layer overriding the one before.
  `job_config=` passes your own JSON file instead; it skips the catalog and
  `overrides.json`, and the command-line values still apply.

Everything that shapes the run lives in that job:

- Model, precision, and provider (`model_name`, `dtype`, `model_provider`)
- GPU count and parallelism (`n_gpus`, `ep_size`)
- Sequence length and batch shape (`max_seq_len`, `train_batch_size`, `ds_config`)
- Optimizer and gradient clipping (`optimizer`, `gradient_clipping`)
- LoRA or full-parameter method (presence of `peft_config`)

Apart from those, the `name=value` command-line overrides are example-loop
settings -- dataset, step count, evaluation cadence, logging and Weights &
Biases. Run an example module with no arguments to see its full list, or read
its `Config` class.

The job-config object is posted as the create-job body with one exception: each
`peft_config` it carries is validated and rewritten into its canonical form
first, and an invalid one fails before anything is submitted. Everything else,
`optimizer` included, is posted unchanged, so
[REST API section 8](../rest-api.md#8-create-job-schemas) is the authoritative
schema for its fields. The
[conversational SFT README](../../../examples/sft/README.md)
documents the shape with every field named.
