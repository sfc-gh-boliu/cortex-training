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

"""Unit tests for the TUI's pure view helpers and local stores (no textual)."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from cortex_training.tui import views as v
from cortex_training.tui.stores import AliasStore
from cortex_training.tui.stores import LocalMetricsLogger
from cortex_training.tui.stores import MetricsStore

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


# ─── series stats and windows ────────────────────────────────────────────────
# GPU telemetry has its own parser and tests (resource.py, test_tui_resource.py).


def test_series_stats():
    assert v.series_stats([60.0, 80.0, 70.0]) == {"min": 60.0, "avg": 70.0, "max": 80.0, "latest": 70.0}
    assert v.series_stats([]) == {"min": None, "avg": None, "max": None, "latest": None}


def test_windows():
    assert v.parse_window("10m") == ("time", 600.0)
    assert v.parse_window("2h") == ("time", 7200.0)
    assert v.parse_window("all") == ("all",)
    assert v.parse_window("steps:500-100") == ("steps", 100, 500)
    assert v.parse_window("soon") is None


# ─── training metrics ────────────────────────────────────────────────────────


def test_metric_catalog_recognizes_sft_and_rl_names():
    recs = [
        {"step": 0, "ts": _iso(NOW), "train_nll": 2.0, "lr": 1e-4, "reward": 0.1, "progress": 0.0, "custom": 3},
        {"step": 1, "ts": _iso(NOW + timedelta(seconds=10)), "train_nll": 1.5, "lr": 1e-4, "reward": 0.4, "custom": 4},
        {"step": 2, "ts": _iso(NOW + timedelta(seconds=20)), "train_nll": 1.7, "kl_divergence": 0.02},
    ]
    cat = v.metric_catalog(recs)
    keys = [m.key for m in cat]
    assert keys == ["train_nll", "lr", "reward", "kl_divergence", "custom"]  # known first, bookkeeping dropped
    loss = v.primary_loss(cat)
    assert (loss.label, loss.latest, loss.minimum, loss.min_step) == ("training loss", 1.7, 1.5, 1)
    assert cat[2].family == "RL" and cat[-1].family == "other"
    pts = v.metric_points(recs, "train_nll")
    assert [p[1] for p in pts] == [0, 1, 2]
    assert len(v.window_points(pts, ("steps", 1, 2))) == 2
    assert len(v.window_points(pts, ("time", 10.0))) == 2


def test_metrics_store_and_recipe_logger(tmp_path):
    store = MetricsStore(tmp_path)
    LocalMetricsLogger("job-1", store=store).log_metrics({"train_nll": 1.25, "note": "x", "nan": float("nan")}, step=3)
    recs = store.load("job-1")
    assert len(recs) == 1
    assert recs[0]["step"] == 3 and recs[0]["train_nll"] == 1.25
    assert "note" not in recs[0] and "nan" not in recs[0]  # only finite numbers are kept
    assert store.load("missing") == []


# ─── charts ──────────────────────────────────────────────────────────────────


def test_render_chart_shape():
    rows = v.render_chart(list(range(100)), [float(i) for i in range(100)], width=60, height=6, x_fmt=lambda x: f"s{int(x)}")
    assert len(rows) == 8  # 6 plot rows + axis + labels
    assert all(len(r) <= 60 for r in rows[:-1])
    assert rows[-1].strip().startswith("s0") and rows[-1].strip().endswith("s99")
    assert "█" in rows[-3]  # the bottom plot row is filled at the high end
    assert v.render_chart([], [], width=40, height=4) == ["(no data in this window)"]
    flat = v.render_chart([0, 1], [5.0, 5.0], width=40, height=4)  # zero range must not divide by zero
    assert len(flat) == 6


def test_render_chart_accepts_unsorted_x():
    # A recipe re-attached to a job restarts its step counter.
    rows = v.render_chart([5, 0, 3, 1], [1.0, 2.0, 3.0, 4.0], width=40, height=4)
    labels = rows[-1].split()
    assert labels == ["0", "5"]


def test_recipe_metric_keys_are_recognized():
    rl = [{"step": 0, "train/avg_loss": 0.9, "reward/mean": 0.1, "optim/lr": 1e-6, "time/total": 5.0,
           "test/env/all/correct": 0.4, "progress/batch": 0}]
    cat = v.metric_catalog(rl)
    assert v.primary_loss(cat).key == "train/avg_loss"  # an RL job opens on its loss, not a constant lr
    labels = {m.key: (m.label, m.family) for m in cat}
    assert labels["reward/mean"] == ("reward", "RL")
    assert labels["test/env/all/correct"] == ("eval success rate", "RL")
    assert labels["time/total"][0] == "step time (s)"  # the recipes restart this timer every step
    assert "progress/batch" not in labels


def test_steps_prefer_global_steps_and_ignore_non_finite_values():
    recs = [
        {"step": 0, "global_steps": 101, "train_nll": 1.0},  # a re-attached recipe restarts its loop step
        {"step": float("nan"), "train_nll": 0.9},
        {"step": float("inf"), "train_nll": 0.8},
        {"step": 7, "train_nll": 0.7},
    ]
    assert [p[1] for p in v.metric_points(recs, "train_nll")] == [101, None, None, 7]
    assert v.metric_catalog(recs)[0].min_step == 7


def test_chart_draws_the_minimum_and_labels_a_day_window():
    rows = v.render_chart([0, 1, 2, 3], [1.0, 0.5, 1.0, 0.5], width=20, height=3)
    bottom = rows[2].split("┤", 1)[1]
    assert bottom.count(" ") < len(bottom)  # the minimum is drawn, not left blank like a gap
    assert v.window_label(("time", 86400.0)) == "last 1d"


def test_local_metrics_logger_never_raises(tmp_path):
    logger = LocalMetricsLogger("job-1", store=MetricsStore(tmp_path))
    for metrics, step in (({"x": 1.0}, float("nan")), ({"x": 1.0}, float("inf")), ({"x": 1.0}, "abc"),
                          ({"x": 10**400}, 1), ([("x", 1.0)], 1)):
        logger.log_metrics(metrics, step=step)  # must not reach the training loop


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


def test_huge_integers_in_metrics_are_skipped():
    recs = [{"step": 1, "train_nll": 10**400}, {"step": 10**400, "train_nll": 0.5}, {"step": 2, "train_nll": 0.4}]
    assert [m.key for m in v.metric_catalog(recs)] == ["train_nll"]
    assert [p[1:] for p in v.metric_points(recs, "train_nll")] == [(None, 0.5), (2, 0.4)]
