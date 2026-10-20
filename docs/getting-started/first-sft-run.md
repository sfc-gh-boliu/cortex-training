# Run a Quick SFT Job

The conversational SFT example is the current shortest end-to-end training path.
It creates a training job, loads a chat dataset, submits forward/backward
batches, applies optimizer steps, and cancels the job when complete.

Install the example dependency:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Start with a short run:

```bash
python -m examples.sft.conversational.train \
  config=/path/to/config.json \
  max_steps=2
```

The default job is `Qwen/Qwen3-8B` at standard context on H200 with
full-parameter training, taken from that model's training file in the
[model catalog](../../model-catalog/README.md). It requests four GPUs, so check
capacity first. Pick another catalog file with `model=`, `context=`,
`hardware=` and `lora=`, for example
`lora=true` for the lighter LoRA path; `hardware=b200` (or `b300`) needs B200
files in the catalog and B200 capacity, so check
`cortex-training capacity --hardware B200` first. GPU count, parallelism and
the LoRA adapter come from the catalog file; `lr=`, `train_batch_size=` and
`max_seq_len=` override it on the command line. See
[sizing and batching](../concepts/sizing-and-batching.md) and
[GPU hardware](../concepts/hardware.md).

For a longer run, dataset changes, dense training, and MoE configuration, see
the [conversational SFT example](../../examples/sft/README.md).

There is no packaged before/after evaluation workflow yet. To confirm the run
worked, check that `train_mean_nll` and `test/nll` fall over a longer run, then
run the sampling command the example prints after it saves its checkpoint -- on
the default memorize task the answer should be `Snowflake AI Research`.
