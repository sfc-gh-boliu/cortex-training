# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""--demo's telemetry: system rows in the service's shape beside the GPU rows, with nodes to pick."""

import json
from datetime import datetime, timezone

import pytest

from cortex_training.tui import views
from cortex_training.tui.demo import DemoClient
from cortex_training.tui.resource import GPU
from cortex_training.tui.resource import SYSTEM
from cortex_training.tui.resource import load_resource_data
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401

SYSTEM_FIELDS = ["kind", "record_version", "sub_job_id", "sample_ts", "worker_num", "seq", "seq_epoch",
                 "sample_interval_s", "cpu_util_pct", "cpu_count", "mem_used_bytes", "mem_total_bytes"]


def _rows(demo, job, root):
    demo.download_metrics(job["job_id"], root)
    return {path.parent.name: [json.loads(line) for line in path.read_text().splitlines()]
            for path in root.glob("*/gpu.jsonl")}


def _grpo(demo):
    return next(j for j in demo.list_jobs() if demo._kinds.get(j["job_id"]) == "rl" and j["status"] == "RUNNING")


def test_the_running_grpo_job_reports_two_nodes_per_sub_job_in_the_services_shape(tmp_path):
    demo = DemoClient()
    job = _grpo(demo)
    files = _rows(demo, job, tmp_path)
    assert set(files) == {sj["sub_job_id"] for sj in job["sub_jobs"]}
    for sub_job_id, rows in files.items():
        system = [row for row in rows if row.get("kind") == "system"]
        gpus = [row for row in rows if "kind" not in row]
        assert system and all(list(row) == SYSTEM_FIELDS for row in system)
        assert {row["record_version"] for row in system} == {3}
        assert {row["sub_job_id"] for row in system} == {sub_job_id}
        assert {row["worker_num"] for row in system} == {0, 1}
        assert {row["worker_num"] for row in gpus} == {0, 1}  # four GPUs to a node
        assert {(row["worker_num"], row["global_gpu_index"] // 4) for row in gpus} == {(0, 0), (1, 1)}
        assert {(row["cpu_count"], row["mem_total_bytes"]) for row in system} == {(88.0, 860.0 * 1024**3)}
        assert all(0.1 <= row["cpu_util_pct"] <= 100 and row["mem_used_bytes"] <= row["mem_total_bytes"]
                   for row in system)
        # CPU varies by node and over time.
        for node in (0, 1):
            assert len({row["cpu_util_pct"] for row in system if row["worker_num"] == node}) > 20
        mean = {node: sum(r["cpu_util_pct"] for r in system if r["worker_num"] == node)
                / sum(1 for r in system if r["worker_num"] == node) for node in (0, 1)}
        assert abs(mean[0] - mean[1]) > 1
        # Each system row shares its scrape's time with the node's GPU rows.
        assert {row["sample_ts"] for row in system} <= {row["sample_ts"] for row in gpus}


def test_nodes_miss_a_scrape_now_and_then_but_never_two_in_a_row(tmp_path):
    demo = DemoClient()
    files = _rows(demo, _grpo(demo), tmp_path)
    for rows in files.values():
        for node in (0, 1):
            times = sorted(views.parse_ts(r["sample_ts"]).timestamp() for r in rows
                           if r.get("kind") == "system" and r["worker_num"] == node)
            newest_hour = [t for t in times if t >= times[-1] - 3600]
            steps = {round(b - a) for a, b in zip(newest_hour, newest_hour[1:])}
            assert steps <= {5, 10} and 10 in steps


def test_one_running_job_reports_gpu_rows_only_and_every_other_gpu_sub_job_has_system_rows(tmp_path):
    demo = DemoClient()
    quiet = [j for j in demo.list_jobs() if j["job_id"] in demo._gpu_rows_only]
    assert len(quiet) == 1 and quiet[0]["status"] == "RUNNING" and len(quiet[0]["sub_jobs"]) == 1
    for job in demo.list_jobs():
        try:
            files = _rows(demo, job, tmp_path / job["job_id"])
        except Exception:  # noqa: BLE001 - a job whose Experiment this role can't read
            continue
        for sub_job_id, rows in files.items():
            has_system = any(row.get("kind") == "system" for row in rows)
            assert has_system is (job["job_id"] not in demo._gpu_rows_only), sub_job_id


def test_the_demo_telemetry_loads_as_both_kinds(tmp_path):
    demo = DemoClient()
    job = _grpo(demo)
    demo.download_metrics(job["job_id"], tmp_path)
    data = load_resource_data(tmp_path)
    training = f"{job['job_id']}:training:0"
    gpu, system = data.kind(training, GPU), data.kind(training, SYSTEM)
    assert data.ignored_rows == 0
    assert gpu.unit_ids == tuple(range(8)) and system.unit_ids == (0, 1)
    assert [m.key for m in system.metrics] == ["cpu_util_pct", "mem_used_pct", "cpu_count", "mem_used_bytes",
                                               "mem_total_bytes"]
    assert [m.key for m in gpu.metrics][:2] == ["gpu_util_pct", "fb_used_pct"]
    assert len(gpu.metrics) + len(system.metrics) >= 10  # every chart the data has
    assert system.view("10m").label == "mean of 2 nodes" and system.view("10m", 1).label == "node 2"
    now = datetime.now(timezone.utc)
    assert (now - system.latest_sample_ts).total_seconds() < 60
    assert 0 < system.latest["cpu_util_pct"] < 30
    assert system.view("10m").stats["mem_total_bytes"].latest == pytest.approx(860.0)
