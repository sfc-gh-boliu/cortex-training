# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the TUI's pure view helpers and local store (no textual)."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from cortex_training.tui import views as v
from cortex_training.tui.stores import AliasStore

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(jid="11111111-aaaa", status="RUNNING", *, age=timedelta(hours=1), lora=None, lr=1e-5, gpus=8, hw="H200",
         model="Qwen/Qwen3-8B", sampling=False, comment=None):
    cfg = {"n_gpus": float(gpus), "optimizer": {"name": "AdamW", "lr": lr}, "train_batch_size": 8.0, "max_seq_len": 4096.0}
    if lora:
        cfg["peft_config"] = {"r": float(lora), "lora_alpha": float(2 * lora)}
    subs = [{"sub_job_id": f"{jid}:training:0", "job_type": "TRAINING", "status": status, "model_name": model,
             "dtype": "bfloat16", "training_config": cfg}]
    if sampling:
        subs.append({"sub_job_id": f"{jid}:sampling:0", "job_type": "SAMPLING", "status": status, "model_name": model,
                     "inference_config": {"n_gpus": 2.0, "max_seq_len": 4096.0}})
    return {"job_id": jid, "status": status, "created_at": _iso(NOW - age), "updated_at": _iso(NOW - age / 2),
            "state_changed_at": _iso(NOW - age / 2), "hardware": hw, "submitted_by": "USER_A", "owner_role": "ML",
            "comment": comment, "sub_jobs": subs}


# ─── names ───────────────────────────────────────────────────────────────────


def test_display_name_prefers_alias_then_comment():
    job = _job(comment="server comment")
    assert v.display_name(job, {}) == "server comment"
    assert v.display_name(job, {job["job_id"]: "my alias"}) == "my alias"
    assert v.display_name(_job(), {}) == ""


def test_alias_store_round_trip_and_clear(tmp_path):
    store = AliasStore(tmp_path)
    store.set("job-1", "  qwen   sweep  #3 ")
    assert store.get("job-1") == "qwen sweep #3"  # whitespace collapsed
    assert AliasStore(tmp_path).all() == {"job-1": "qwen sweep #3"}  # persisted
    store.set("job-1", "")
    assert store.get("job-1") is None


# ─── search ──────────────────────────────────────────────────────────────────


def test_name_and_partial_id_search_is_case_insensitive():
    job = _job("abcdef12-3456")
    assert v.job_matches_query(job, v.parse_query("SWEEP"), "Qwen LoRA Sweep")
    assert v.job_matches_query(job, v.parse_query("abcdef"), "")
    assert v.job_matches_query(job, v.parse_query("ABCDEF"), "")
    assert not v.job_matches_query(job, v.parse_query("nothere"), "Qwen LoRA Sweep")


def test_structured_field_filters():
    job = _job(lora=16, lr=1e-5, gpus=8)
    match = lambda q: v.job_matches_query(job, v.parse_query(q), "n")  # noqa: E731
    assert match("model:qwen3-8b")
    assert match("method:lora lora_r=16 lora_alpha>=32")
    assert match("gpus>=8 gpu:h200")
    assert match("lr<1e-4 optimizer=adamw")
    assert match("bs=8 seq:4096")
    assert not match("lr>1e-4")
    assert not match("method:full")
    assert not match("gpu:b200")
    assert v.job_matches_query(_job(), v.parse_query("method:full"), "")


def test_advanced_dotted_config_member():
    job = _job(lr=3e-5)
    assert v.job_matches_query(job, v.parse_query("training_config.optimizer.lr=3e-5"), "")
    assert v.job_matches_query(job, v.parse_query("optimizer.name:adam"), "")
    assert not v.job_matches_query(job, v.parse_query("training_config.optimizer.lr>1e-4"), "")


def test_unknown_field_matches_nothing_and_explains():
    q = v.parse_query("nope:1")
    assert q.errors == ["unknown field 'nope'"]
    assert not v.job_matches_query(_job(), q, "")
    msg = v.no_match_message("nope:1", q, 12)
    assert "No jobs match" in msg and "12 job(s)" in msg and "Fields:" in msg


# ─── listing ─────────────────────────────────────────────────────────────────


def test_live_vs_history_and_retention():
    assert v.is_live(_job(status="PLACING"))
    assert not v.is_live(_job(status="CANCELLED"))
    assert not v.is_live(_job(status="weird"))  # an unrecognized string is not shown as live
    # The service's UNKNOWN means it could not read the live state: the job may be running.
    assert v.is_live(_job(status="UNKNOWN")) and v.is_live(_job(status="JOB_STATE_UNKNOWN"))
    assert v.within_retention(_job(age=timedelta(days=180)), NOW)
    assert not v.within_retention(_job(age=timedelta(days=200)), NOW)


def test_sort_by_created_updated_status():
    a = _job("a", "FAILED", age=timedelta(hours=1))
    b = _job("b", "RUNNING", age=timedelta(hours=5))
    c = _job("c", "CANCELLED", age=timedelta(hours=3))
    c["updated_at"] = _iso(NOW)
    ids = lambda js: [j["job_id"] for j in js]  # noqa: E731
    assert ids(v.sort_job_list([a, b, c], "created")) == ["a", "c", "b"]
    assert ids(v.sort_job_list([a, b, c], "created", descending=False)) == ["b", "c", "a"]
    assert ids(v.sort_job_list([a, b, c], "updated"))[0] == "c"
    assert ids(v.sort_job_list([a, b, c], "status")) == ["b", "a", "c"]


def test_summary_fields_cover_the_preview():
    job = _job(sampling=True)
    fields = dict(v.summary_fields(job, "my job", NOW))
    assert fields["Name"] == "my job"
    assert fields["Job ID"] == job["job_id"]
    assert fields["Status"] == "RUNNING"
    assert fields["Model"] == "Qwen/Qwen3-8B"
    assert fields["Type"] == "train+sample"
    assert fields["Owner"] == "USER_A (ML)"
    assert fields["GPUs"].startswith("10 × H200")
    assert fields["Elapsed"].startswith("1h 00m")
    assert set(fields) >= {"Created", "Last update"}


def test_describe_error_shows_the_status_and_message_not_the_url():
    class Response:
        status_code = 500
        reason = "Internal Server Error"

        def json(self):
            return {"message": "cannot resolve the job's experiment"}

    class FakeHTTPError(Exception):
        response = Response()

    text = v.describe_error(FakeHTTPError("500 Server Error: Internal Server Error for url: http://host/a/b"))
    assert text == "HTTP 500 Internal Server Error: cannot resolve the job's experiment"
    assert v.describe_error(RuntimeError("boom")) == "RuntimeError: boom"


def test_search_matches_a_full_sub_job_id():
    job = _job("job-9")
    sub_job_id = job["sub_jobs"][0]["sub_job_id"]
    assert v.job_matches_query(job, v.parse_query(sub_job_id), "")
