# Sizing and Batching

Training capacity depends on model architecture, parameter count, precision,
sequence length, micro-batch size, optimizer state, activation memory, and
parallelism strategy.

For the current examples, these come from the model catalog's job file for the
model, context and hardware you pick; `train_batch_size` and `max_seq_len` can
also be set on the command line. The fields that interact:

| Field | Where |
|---|---|
| `n_gpus` | `training_config` |
| `sp_size` | `training_config` |
| `max_seq_len` | `training_config` |
| `train_batch_size` | `training_config` |
| `train_batch_size` (DeepSpeed's) | `training_config.ds_config` |
| `train_micro_batch_size_per_gpu` | `training_config.ds_config` |
| `gradient_accumulation_steps` | `training_config.ds_config` |
| `ep_size` | `training_config`, MoE only |

`training_config.train_batch_size` is the number of sequences in a training
step: the SFT example trains on that many per step. DeepSpeed's batch fields
follow the GPU layout instead. The server sets `train_micro_batch_size_per_gpu`
to 1, `gradient_accumulation_steps` to 1 and DeepSpeed's `train_batch_size` to
the data-parallel degree, `n_gpus / sp_size`, and then merges
`training_config.ds_config` on top. Some catalog files spell out these values.
Changing `train_batch_size`, on the command line or in an override, leaves them
as they are, and they stay correct: they depend only on `n_gpus` and `sp_size`.

When you set DeepSpeed's batch fields yourself, they must agree with each other
and with the GPU layout, counting the server's value for any you leave out:

```text
ds_config.train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps * n_gpus / sp_size
```

The examples do not check this before submitting, so an inconsistent config is
only caught server-side. For MoE training, `n_gpus` must also be divisible by
`ep_size`.

There is no validated sizing guide or GPU requirement matrix yet, so treat the
shipped configs as starting points and measure your own.
