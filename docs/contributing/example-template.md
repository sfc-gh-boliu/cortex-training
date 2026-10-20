# Example Template

Each runnable example has one folder, next to the shared loop it uses. An RL
task, for example, lives in `examples/rl/<task>/` and reuses
`examples/rl/runner.py`:

```text
examples/rl/
├── README.md               # the RL examples' README
├── runner.py               # the shared loop
├── overrides.json          # optional: config changes for every RL run
└── <task>/
    ├── __init__.py
    ├── train.py            # entry point: the chz Config and main()
    ├── task.py             # dataset, prompt, reward, evaluator
    ├── example.yaml        # metadata
    ├── data/               # optional bundled data
    └── *.ipynb             # optional notebooks
```

The entry point builds its job with `build_job` from `examples/utils.py`, so
the example ships no job configs. Give its `Config` the `model=`, `context=`,
`hardware=` and `lora=` arguments, `job_config=` (default `None`) and the
hyperparameter arguments (`lr=`, `train_batch_size=`, `max_seq_len=`, default
`None`). The job is then built in three layers, each overriding the one
before:

1. The model's file in the [model catalog](../../model-catalog/README.md),
   `model-catalog/models/<slug>/<context>/<hw>-<kind>-<mode>.json`, where
   `<hw>` is `h200`, `b200` or `b300` and the file's `hardware` field matches
   it.
2. The folder's `overrides.json`, applied to every run of that example on top of
   the catalog. Every example ships one; it may be empty apart from a `_comment`.
3. The command-line arguments.

Overrides merge key by key, `null` removes a key, `sub_job_configs` entries
are matched by `job_type`, and other lists are replaced whole. Each
`overrides.json` carries a reserved `_comment` key -- a free-text string (or
list of strings) describing the file -- which the loader strips at any depth
before merging, so it never reaches the job (JSON has no comments). A new model
or GPU type is a change to the model catalog, not to the example. Distillation
builds two jobs (student and teacher) from the catalog with
`make_distillation_config`; its `configs/` holds only the records each run
generates and is gitignored, like RL's.

Keep task code in `task.py` and import shared helpers from `runner.py` and
`examples/utils.py`. A helper another module uses is public, with no leading
underscore, and the entry point does not re-export it.

The README should cover:

1. Outcome and training method
2. Prerequisites and access
3. Dataset and model
4. Hardware requirements
5. Exact run command
6. Expected outputs and metrics
7. Configuration variations, and how the job is built
8. Checkpoint and evaluation workflow
9. Troubleshooting
10. Provenance and last validation date

`example.yaml` records the example's `id`, `title`, `status`, `task`,
`methods`, `datasets`, `precision`, `provenance` when adapted from another
project, and `last_validated`. The metadata should be machine-readable so
model compatibility and example catalog pages can eventually be generated.
For examples whose job comes from the model catalog, the catalog says which
models they support, so their metadata lists no `models` or
`default_training_gpus`.
