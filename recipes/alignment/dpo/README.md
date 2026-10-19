# Direct Preference Optimization

This experimental recipe submits complete preference pairs to native server DPO.
The caller renders response masks and caches frozen reference sequence sums;
the server computes the loss and backward. Evaluation uses sequence-sum DPO
math locally for checkpoint selection.

```bash
python -m recipes.alignment.dpo.train config=/path/to/connection.json job_config=configs/qwen3_8b_lora.json
```

Use `dataset=ultrafeedback n_train_pairs=1024 max_steps=None` for a larger run.
`pairs_per_batch`, `epochs`, `eval_every`, `dpo_beta`, `lr_schedule`, and
`warmup_ratio` control training. Each update needs at least one complete pair
per GPU, and each pair must fit the configured microbatch token budget. Development data selects checkpoints; the
held-out test split is evaluated after selection. Prompt overlap is rejected.
A smoke run checks initial loss near ln(2), decreasing loss, and held-out accuracy.

Checkpoints include optimizer state. Use `resume_state=/path/to/dpo-state.json`
to recover from the last acknowledged update with unchanged reference scores
and pair order. An unacknowledged update is replayed from that checkpoint.
The runner owns and cancels its job when it exits.

See [the REST contract](../../../docs/reference/rest-api.md#native-dpo-and-checkpoint-recovery).
Native service/GPU execution of this recipe has not yet been validated.
