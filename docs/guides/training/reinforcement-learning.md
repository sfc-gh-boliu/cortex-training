# Reinforcement Learning

The [Math GRPO recipe](../../../examples/rl/README.md) is the current
end-to-end RL example. It demonstrates:

- Colocated training and sampling sub-jobs
- Grouped rollout generation and reward centering
- GRPO loss configuration
- Training-to-sampling weight synchronization
- Held-out MATH-500 evaluation

## Integrated RL frameworks

[SkyRL](../../../examples/frameworks/skyrl/gsm8k_grpo/README.md) can also run GRPO against Cortex Training,
with its trainer on a CPU driver and training and sampling in Cortex sub-jobs.
Its GSM8K example lives in the Arctic Platform repository and has its own
install.

Additional code, tool-use, and multi-agent RL recipes are planned.
