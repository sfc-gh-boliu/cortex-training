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

"""The job list's search, labels and order: what the list shows is what search finds."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from cortex_training.tui import views as v

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
JID = "fake-job-0001"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _lora_job(jid=JID, *, modules=("q_proj", "v_proj"), betas=(0.9, 0.999), seed=42, ep_size=None, **extra):
    cfg = {"optimizer": {"name": "adamw", "lr": 1e-5, "betas": list(betas)}, "n_gpus": 8.0,
           "peft_config": {"r": 16, "lora_alpha": 32, "target_modules": list(modules)}}
    if ep_size is not None:
        cfg["ep_size"] = ep_size
    return {"job_id": jid, "status": "CANCELLED", "created_at": _iso(NOW - timedelta(days=1)),
            "comment": "lora sweep", "hardware": "H200", "owner_role": "TRAINING_ROLE", "submitted_by": "USER_A",
            "sub_jobs": [{"sub_job_id": f"{jid}:training:0", "job_type": "TRAINING", "model_name": "Qwen/Qwen3-8B",
                          "seed": seed, "dtype": "bfloat16", "training_config": cfg},
                         {"sub_job_id": f"{jid}:sampling:0", "job_type": "SAMPLING", "model_name": "Qwen/Qwen3-8B",
                          "inference_config": {"n_gpus": 2.0}}], **extra}


def _matches(job, text, jobs=None):
    query = v.resolve_query(v.parse_query(text), jobs or [job])
    return v.job_matches_query(job, query, v.search_name(job))


# ─── list-valued members, and sub-job members ────────────────────────────────


def test_a_list_valued_config_member_matches_when_one_element_does():
    job = _lora_job()
    other = _lora_job("other", modules=("k_proj",), betas=(0.8, 0.95))
    for text in ("training_config.peft_config.target_modules:q_proj",
                 "training_config.peft_config.target_modules=v_proj",
                 "training_config.optimizer.betas:0.9",
                 "training_config.optimizer.betas>=0.999"):
        assert _matches(job, text, [job, other]), text
        assert not _matches(other, text, [job, other]), text
    # != keeps only the jobs none of whose elements is that value.
    assert not _matches(job, "training_config.peft_config.target_modules!=q_proj")
    assert _matches(other, "training_config.peft_config.target_modules!=q_proj", [job, other])


def test_a_sub_job_member_is_reachable_by_its_own_name():
    job, other = _lora_job(seed=42), _lora_job("other", seed=7)
    jobs = [job, other]
    assert [j["job_id"] for j in jobs if _matches(j, "seed=42", jobs)] == [JID]
    assert [j["job_id"] for j in jobs if _matches(j, "seed>=7", jobs)] == [JID, "other"]
    # A member of a config's top level, too: what the preview's config shows.
    moe, dense = _lora_job("moe", ep_size=4), _lora_job("dense")
    assert [j["job_id"] for j in (moe, dense) if _matches(j, "ep_size>1", [moe, dense])] == ["moe"]


def test_a_key_no_job_has_is_still_an_unknown_field_with_a_hint():
    job = _lora_job()
    query = v.resolve_query(v.parse_query("modle:qwen"), [job])
    assert query.errors == ["unknown field 'modle'"]
    assert not v.job_matches_query(job, query, "")
    # Nested only: the dotted path that reads it.
    query = v.resolve_query(v.parse_query("lora_alpha_x>1 target_modules:q_proj"), [job])
    assert query.errors == ["unknown field 'lora_alpha_x'",
                            "unknown field 'target_modules' (did you mean training_config.peft_config.target_modules?)"]
    message = v.no_match_message("target_modules:q_proj", query, 1)
    assert "did you mean training_config.peft_config.target_modules?" in message
    assert "any other key reads that member" in message


# ─── search finds what the list shows ────────────────────────────────────────


def test_the_type_label_and_the_short_model_name_the_list_shows_are_found():
    job = _lora_job()
    assert v.job_type_label(job) == "train+sample"
    assert _matches(job, "train+sample")
    assert _matches(job, "model=Qwen3-8B") and _matches(job, "model=qwen/qwen3-8b") and _matches(job, "model:qwen3")
    assert not _matches(job, "model=Qwen3-32B")
    assert not _matches(job, "model!=Qwen3-8B") and _matches(job, "model!=Qwen3-32B")


# ─── Unicode ─────────────────────────────────────────────────────────────────


def test_search_matches_either_spelling_of_an_accent_and_fullwidth_letters():
    precomposed, decomposed = "café run", "café run"
    wide = {"job_id": "w", "status": "FAILED", "comment": "ｗｉｄｅ sweep"}
    for comment in (precomposed, decomposed):
        job = {"job_id": "j", "status": "FAILED", "comment": comment}
        for typed in ("café", "café", "CAFÉ"):
            assert _matches(job, typed), (comment, typed)
    assert _matches(wide, "wide") and _matches(wide, "WIDE")
    assert _matches({"job_id": "s", "status": "FAILED", "comment": "Straße"}, "strasse")


# ─── quoting, and the no-match message ───────────────────────────────────────


def test_a_quoted_word_with_a_colon_is_searched_as_text():
    job = _lora_job()
    query = v.parse_query('"training:0"')
    assert query.terms == ["training:0"] and not query.errors
    assert _matches(job, '"training:0"') and _matches(job, "'sampling:0'")
    # Unquoted, it is a field filter, and the message says how to search the text.
    query = v.resolve_query(v.parse_query("training:0"), [job])
    assert query.errors == ["unknown field 'training'"]
    assert 'quote it: "training:0"' in v.no_match_message("training:0", query, 1)
    # Quotes still group words, and a field's quoted value is still its value.
    assert v.parse_query('"lora sweep"').terms == ["lora sweep"]
    assert v.parse_query('name:"lora sweep"').filters == [("name", ":", "lora sweep")]
    assert v.parse_query("db='\"ml\"'").filters == [("db", "=", '"ml"')]


def test_every_line_after_the_first_of_the_no_match_message_is_indented():
    query = v.resolve_query(v.parse_query("nope:1 sweep"), [_lora_job()])
    lines = v.no_match_message("nope:1 sweep", query, 12).split("\n")
    assert lines[0].startswith("No jobs match")
    assert all(line.startswith("  ") for line in lines[1:])


# ─── jobs older than six months ──────────────────────────────────────────────


def test_the_no_match_message_names_an_older_job_that_matches():
    message = v.no_match_message(JID[:8], v.parse_query(JID[:8]), 623, older=[JID])
    assert "No jobs match" in message and "623 job(s) in the last 6 months" in message
    assert f"1 older job matches (the list hides jobs older than 6 months): cortex-training tui {JID} opens it" \
        in message
    message = v.no_match_message("sweep", v.parse_query("sweep"), 5, older=["b", "a"])
    assert "2 older jobs match" in message and "cortex-training tui b opens the newest" in message


# ─── status order ────────────────────────────────────────────────────────────


def test_other_spellings_of_a_status_sort_with_it():
    rows = [{"job_id": s, "status": s, "created_at": _iso(NOW - timedelta(minutes=i))}
            for i, s in enumerate(["SUCCEEDED", "TERMINATED", "DONE", "CANCELED", "FAILED", "COMPLETED",
                                   "CANCELLED", "STOPPING", "?"])]
    ordered = [j["status"] for j in v.sort_job_list(rows, "status", descending=True)]
    # done, succeeded and completed together, then cancelled and canceled, then terminated; unknown words last.
    assert ordered == ["FAILED", "SUCCEEDED", "DONE", "COMPLETED", "CANCELED", "CANCELLED", "TERMINATED",
                       "STOPPING", "?"]


# ─── one name for log_prob ───────────────────────────────────────────────────


def test_log_prob_sub_jobs_have_one_short_name():
    lp = {"sub_job_id": f"{JID}:log_prob:0", "job_type": "LOG_PROBABILITY", "inference_config": {"n_gpus": 1}}
    train = {"sub_job_id": f"{JID}:training:0", "job_type": "TRAINING", "training_config": {"n_gpus": 8}}
    job = {"job_id": JID, "status": "RUNNING", "sub_jobs": [train, lp]}
    assert v.sub_job_label(lp) == "log_prob #0" and v.sub_job_type(lp) == "log_prob"
    assert v.job_type_label(job) == "train+logprob"
    assert v.job_type_label({"job_id": "x", "sub_jobs": [lp]}) == "log_prob"
    # Search knows both spellings.
    assert _matches(job, "type:log_prob") and _matches(job, "type=log_probability") and _matches(job, "logprob")


# ─── an owner that is only a role ────────────────────────────────────────────


def test_an_owner_that_is_only_a_role_says_so():
    assert v.owner({"submitted_by": "USER_A", "owner_role": "ROLE_X"}) == "USER_A (ROLE_X)"
    assert v.owner({"submitted_by": "", "owner_role": "ROLE_X"}) == "role ROLE_X"
    assert v.owner({"owner_role": "ROLE_X"}) == "role ROLE_X"
    assert v.owner({"submitted_by": "USER_A"}) == "USER_A"
    assert v.owner({}) == "—"
    assert v.owner({"owner_role": "R\x1b]52;c;x\x07"}) == "role R�]52;c;x�"  # never raw controls


# ─── an UNKNOWN job the control plane has no record of ───────────────────────


def test_an_unknown_job_the_control_plane_has_no_record_of_is_not_live():
    gone = {"job_id": "g", "status": "UNKNOWN",
            "reason": "Job status unavailable: the job was not found in the Control Plane"}
    outage = {"job_id": "o", "status": "UNKNOWN", "reason": "Job status unavailable: Control Plane is unavailable"}
    assert v.job_not_found(gone) and not v.is_live(gone)
    assert not v.job_not_found(outage) and v.is_live(outage)  # may still be running
    assert not v.job_not_found({"job_id": "f", "status": "FAILED", "reason": "image not found"})
