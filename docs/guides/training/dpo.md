# Direct Preference Optimization

See the [DPO recipe](../../../recipes/alignment/dpo/README.md) for configuration,
frozen reference scoring, checkpoint selection, and recovery.

### Native DPO and checkpoint recovery

Submit complete adjacent chosen/rejected rows with `processing.loss_fn="dpo"`
with `processing.post=["compute_logprobs"]` and `processing.config={"beta": 0.1}`. Each row carries prediction-aligned
int64 `context.labels` and boolean `context.loss_mask`, selecting response tokens
only. `context.dpo_pair_ids` is an int64 row vector with each unique pair ID
repeated twice; boolean `context.dpo_is_chosen` alternates true/false.
`context.dpo_reference_logps` contains detached finite floating-point response
log-probability sums from a frozen reference, one per row. Reference `forward`
requests use the same explicit aligned labels. Forward returns token log
probabilities; the caller applies the same response mask locally when summing
those scores. The reference-forward frame need not send `loss_mask`.

The server preserves complete pairs during rank assignment and token-budget
packing, computes the global pair-mean DPO objective, and performs backward.
Do not supply `cu_seqlens` or training token gradients. A successful
`forward_backward` returns scalar `avg_loss`; call `step` afterward and require
`update_successful=1` before recording an update. On failed backward, do not step.
Recovery restores model and optimizer checkpoints while retaining the original
reference sums and deterministic pair schedule.
