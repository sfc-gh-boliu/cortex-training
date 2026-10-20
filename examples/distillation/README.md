# On-Policy Distillation

Train a small student model to match a larger teacher on the student's own
outputs. Each step, the student answers a batch of GSM8K questions, a frozen
teacher scores every token the student wrote, and the student is updated toward
the tokens the teacher prefers. The default run distills Qwen3-8B into
Qwen3-0.6B.

## How it works

The example creates two jobs:

- **Student**: a training sub-job and a sampling sub-job of the same model.
  The sampling sub-job generates rollouts, and weight sync copies each update
  from training to sampling.
- **Teacher**: a separate sampling-only job. It never trains and never
  receives weight syncs.

Each training step:

1. The student samples `group_size` answers to each of `problems_per_batch`
   questions.
2. Each answer is sent to the teacher as one pre-tokenized prompt (question
   tokens followed by the student's answer tokens) with `prompt_logprobs: 0`
   and `max_tokens: 1`. The teacher returns its log probability of every token
   the student wrote, so the student's exact tokens are scored without decoding
   or re-tokenizing (see
   [Per-token log probabilities](../../docs/reference/rest-api.md#per-token-log-probabilities)).
3. The teacher scores go in `context["teacher_log_probs_shifted"]`, aligned so
   that position `i` scores the token at `i + 1`, and the student trains with
   the `grpo` loss's teacher term (see
   [Teacher term for on-policy distillation](../../docs/reference/rest-api.md#teacher-term-for-on-policy-distillation)).
   Each token's advantage becomes
   `teacher_tau * clamp(log p_teacher - log p_student, -teacher_clip, teacher_clip)`,
   which on average follows the gradient of the reverse KL from student to
   teacher.

This is the formulation from Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
and tinker-cookbook's `distillation/train_on_policy.py`, with the KL advantage
clipped and the learning rate decayed.

Every teacher result is checked before training: a missing, mis-sized,
misaligned, or non-finite score raises an error. After each forward-backward
pass, and before the optimizer step, the example also checks that the server's
teacher term covered exactly the tokens the teacher scored.
The student and teacher must share a tokenizer; the example checks this before
creating any job, so pick a teacher from the student's model family.

The example does not send `old_log_probs_shifted`, so the server uses the
current, detached student log probabilities. Each update is fully on-policy
and `eps_clip` has no effect.

## Prerequisites

Install the client and example dependencies as described in
[examples/README.md](../README.md#prerequisites), and create a connection file
from `docs/getting-started/connection.json.template`. The student and teacher
models are both in the [model catalog](../../model-catalog/README.md).

## Dataset and Model

- Dataset: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k).
  200 training problems are held out as a validation set; the remaining 7,273
  are training prompts, and the 1,319-question test split is the benchmark.
  Each question ends with
  `Please reason step by step, and put your final answer within \boxed{}.`
- Student: `Qwen/Qwen3-0.6B`, full-parameter, non-thinking chat template
  (`renderer_name=qwen3_disable_thinking`).
- Teacher: `Qwen/Qwen3-8B`, sampling-only.

## Hardware

The small default runs on four H200 GPUs: one training and one sampling GPU for
the student (1 + 1), and the teacher's catalog inference file (two sampling GPUs
for Qwen3-8B). Check capacity before submitting:

```bash
cortex-training capacity --hardware H200
```

## Run

```bash
python -m examples.distillation.on_policy.train \
  config=/path/to/config.json
```

`config=` is the Snowflake connection file. Both jobs are cancelled when the run
ends, after the final checkpoint is saved.

## Choose the jobs

The example builds both jobs on every run, from the
[model catalog](../../model-catalog/README.md), through
`make_distillation_config` in `utils.py`:

1. The **student** is the catalog's training and inference files for
   `student=` (the exact model ID), `context=` (`standard` or `long`),
   `hardware=` (`h200`, `b200` or `b300`; default `h200`) and `lora=` (default
   `false`), combined into one job with a training and a sampling sub-job, like
   an RL job but with no router replay. On-policy distillation replays no
   routing.
2. The **teacher** is `teacher=`'s full-weights inference file (in
   `teacher_context=`, defaulting to `context=`) as a sampling-only job.
3. `distillation/overrides.json`, the example's small single-GPU student
   footprint (one training and one sampling GPU, `train_batch_size` 1,
   `mb_spec.max_tokens_per_mb` 10240, ZeRO `reduce_scatter`,
   `model_provider` `huggingface`), is applied to the student like
   `rl/overrides.json` is to an RL run. Edit it to change a default for every
   distillation run.
4. `student_max_seq_len=` (default 2048) sizes both of the student's sub-jobs
   and `teacher_max_seq_len=` (default 4096) the teacher's, so the catalog's
   full-length files shrink to a small, runnable job. The teacher's window must
   exceed the student's — the teacher scores the question and answer together —
   and the builder raises a clear error naming both windows when it does not.

Each generated run writes the jobs it built to
`examples/distillation/configs/<student-slug>-<context>-<hw>-<mode>-student.json`
and `<teacher-slug>-teacher.json`, before submitting them, so you can copy a
file, edit it, and pass it back. The folder is not committed. `job_config=FILE`
and `teacher_job_config=FILE` replace the student or teacher body with your own
file, used as is; `job_id=` and `teacher_job_id=` attach to running jobs instead
of creating new ones, and attached jobs are left running. To build the jobs
without running the example, pass the example's `overrides.json` with
`--override` so the output is the shipped small default (one training and one
sampling GPU for the student, two for the teacher) rather than the raw
full-size catalog composition:

```bash
python -m examples.utils --distillation \
  --student Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-8B --context standard \
  --student-max-seq-len 2048 --teacher-max-seq-len 4096 \
  --override examples/distillation/overrides.json -o dist.json
```

This writes the student to `dist.json` and the teacher to `dist.teacher.json`.
Without `--override`, the student is the catalog's full-size training
configuration (8 training + 8 sampling GPUs); the `--override` reproduces the
example's 4-GPU default. Check each with `ct submit --dry-run`, and submit, or
pass them back with `job_config=`/`teacher_job_config=`.

## Expected Results

The catalog-built example has not yet been validated on a live run, so its
`last_validated` is `null` (pending) and no accuracy numbers are quoted here.
Greedy accuracy on the full GSM8K test split, evaluated every `eval_every`
steps, will be measured on a live run and this section and `example.yaml`
refreshed then. Logged metrics include:

| Metric | Meaning |
|---|---|
| `test/correct`, `val/correct` | Greedy accuracy on the test split and the validation split |
| `teacher/test/correct` | The teacher's test accuracy, at step 0 |
| `train/kl/k1_per_token`, `train/kl/k3_per_token` | Reverse-KL estimates on the training rollouts, from the student's sampling log probabilities |
| `heldout/kl/k3_per_token` | The same estimate on `n_kl_eval` test prompts the student does not train on |
| `teacher_term_token_count`, `teacher_log_ratio_sum`, `teacher_clipped_log_ratio_sum` | The server's teacher-term metrics |
| `teacher/clipped_fraction` | Share of training tokens whose `log p_teacher - log p_student` falls outside the clip, using the student's sampling log probabilities |
| `rollout/correct`, `rollout/mean_tokens`, `rollout/frac_truncated` | Training-rollout accuracy, length, and how often answers hit `max_tokens` |

At the end, the run logs the step with the best validation accuracy and the test
accuracy at that step. Choose a checkpoint by validation accuracy, not test
accuracy.

## Configuration Variations

```bash
python -m examples.distillation.on_policy.train \
  config=/path/to/config.json \
  student=Qwen/Qwen3-0.6B teacher=Qwen/Qwen3-8B context=standard \
  student_max_seq_len=2048 teacher_max_seq_len=4096 \
  max_steps=MAX_STEPS \
  problems_per_batch=PROBLEMS_PER_BATCH \
  group_size=GROUP_SIZE \
  max_tokens=MAX_TOKENS \
  teacher_clip=TEACHER_CLIP \
  lr_schedule=cosine \
  reward_coef=REWARD_COEF \
  seed=SEED \
  wandb_project=WANDB_PROJECT
```

- `student=`, `teacher=`, `context=`, `hardware=`, `lora=`,
  `student_max_seq_len=` and `teacher_max_seq_len=` pick and size the jobs (see
  [Choose the jobs](#choose-the-jobs)). The teacher's window must exceed the
  student's.
- `lr_schedule` (default `cosine`, decaying to `min_lr_ratio` of the peak after
  `warmup_steps`). With `constant`, accuracy can peak early and then fall, so
  keep the decay unless you also stop early.
- `teacher_clip` (default `3`) limits how hard any one token can pull.
- `reward_coef` (default `0`, pure distillation) adds a group-centred GSM8K
  correctness reward to every token's advantage, alongside the teacher term.
- `teacher_tau=0` with `reward_coef=1` turns the teacher off and runs the same
  loop as plain GRPO, for comparison. No teacher job is created.
- `teacher_tau` (default `1`) weights the teacher term relative to the reward.
- The peak learning rate and GPU counts come from the catalog files and
  `overrides.json`; the sequence lengths from the sizing flags.
- `job_config=`/`teacher_job_config=` use a job file as is, and `job_id=`/
  `teacher_job_id=` attach to running jobs and leave them running.

## Checkpoint and Evaluation

The run saves a weights-only checkpoint of the student and logs its
`checkpoint_id`. Serve it with the inference example, picking the student's
catalog inference file the same way the other examples do:

```bash
python -m examples.inference.serve \
  config=/path/to/config.json \
  model=Qwen/Qwen3-0.6B context=standard \
  source_job_id=STUDENT_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

`examples.inference.evaluate task=gsm8k` uses a different prompt and answer
format (the bare question, graded on `#### n` or the last number), so its
numbers are not comparable with the accuracy this example reports.

## Troubleshooting

- `student and teacher tokenizers differ`: pick a teacher from the same model
  family; the teacher scores the student's token ids directly.
- `teacher max_seq_len ... must exceed the student's`: raise
  `teacher_max_seq_len=` above `student_max_seq_len=`.
- `generate result has no 'prompt_logprobs'`: the teacher job must be a
  sampling job on a server version that passes `prompt_logprobs` through.
- `Unknown config keys` naming `teacher_tau`: the training server predates the
  `grpo` teacher term.
- `the teacher term covered N tokens, expected M`: the server trained on
  different tokens than the teacher scored; check that the student job config
  has not been changed to pack or truncate sequences.
- Accuracy rises and then falls: keep `lr_schedule=cosine`, or choose the step
  with the best `val/correct`.

## Provenance

Adapted from tinker-cookbook's on-policy distillation
(`tinker_cookbook/distillation/train_on_policy.py`) and Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
The `last_validated` date in `example.yaml` records the most recent verified run.
