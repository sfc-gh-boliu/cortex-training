# API Request Examples

These JSON files are small, standalone API examples: use them to try one call
at a time, to see a format, or as a starting point for your own request. They
are not end-to-end training workflows; for those, see the
[runnable examples](../README.md).

The **create-job bodies** below are raw request bodies, the exact shapes the
REST API and the Python client take, submitted as given (aside from
`peft_config` canonicalization). The **operation bodies** further down are
readable specs — text and settings — that the CLI or a helper converts into the
wire frame before sending: `fwd-bwd.json`, for example, is tokenized into a
DSSST1 forward/backward frame, not posted as-is (passing it straight to
`create_job_from_body` would raise). Each section shows the right way to send
its files.

## Create-job bodies

| File | What it creates |
|---|---|
| [`training.json`](training.json) | A training sub-job for `Qwen/Qwen3.6-35B-A3B` with the `prime_rl` model provider |
| [`sampling.json`](sampling.json) | A sampling sub-job for `Qwen/Qwen3.6-35B-A3B` |
| [`rl.json`](rl.json) | Colocated training and sampling sub-jobs, the RL layout |
| [`external-weights-training.json`](external-weights-training.json) | A training sub-job initialized from a Hugging Face model directory uploaded to a stage |

Submit one with `ct`, the CLI's short name for `cortex-training`. `--dry-run`
runs the same checks without a connection and prints the body it would send:

```bash
ct submit --dry-run examples/api/training.json
ct submit examples/api/training.json
ct submit examples/api/training.json --wait   # wait until running
```

Or send it through the Python client:

```python
import json
from pathlib import Path

from cortex_training import CortexTrainingClient

client = CortexTrainingClient.from_connection_name("training")
body = json.loads(Path("examples/api/training.json").read_text())
job_id = client.create_job_from_body(body)["job_id"]
client.wait_for_job(job_id)
```

`create_job_from_body` checks the body the same way `submit --dry-run` does.
It also rewrites each `peft_config` in place into its canonical form, adding
`task_type` and `target_parameters` for example, so the dict you pass changes.
Everything else is posted as given. See
[REST API section 5.1](../../docs/reference/rest-api.md#51-create-job---post-)
for every field.

## Operation bodies

| File | What it sends |
|---|---|
| [`fwd-bwd.json`](fwd-bwd.json) | A readable forward/backward batch: text, tokenizer and label settings |
| [`generate.json`](generate.json) | Prompts and sampling parameters for generation |

Both need a running job:

```bash
ct --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
ct --job JOB_ID step
ct --job JOB_ID generate examples/api/generate.json
```

The CLI tokenizes the `fwd-bwd.json` text (this needs `transformers`), builds
the DSSST1 frame, submits it and polls the result; `"poll": false` prints only
the request id. From Python, build the same frame with
`build_forward_backward_payload` and send the prompts directly:

```python
import json
from pathlib import Path

from cortex_training import build_forward_backward_payload

spec = json.loads(Path("examples/api/fwd-bwd.json").read_text())
request_id = client.forward_backward(job_id, build_forward_backward_payload(spec))
loss = client.poll_request(job_id, request_id)["avg_loss"]

generate = json.loads(Path("examples/api/generate.json").read_text())
request_id = client.generate(job_id, prompts=generate["prompts"], sampling_params=generate["sampling_params"])
texts = [result["text"] for result in client.poll_request(job_id, request_id)["results"]]
```

See the [CLI reference](../../docs/reference/cli.md#run-a-forward-backward-smoke-test)
and [REST API section 6](../../docs/reference/rest-api.md#6-data-plane-endpoints)
for the operations.

## When to use make_config instead

The create-job bodies here are fixed examples with small sequence lengths, not
tuned configurations. For a job you intend to run, start from the
[model catalog](../../model-catalog/README.md), which holds the tuned training
and inference files for each supported model, context and GPU hardware.
`python -m examples.utils` writes one of those jobs, or an RL job built from
them, with your overrides applied; submit the result the same way. See
[Create a job config from the command line](../README.md#create-a-job-config-from-the-command-line).
