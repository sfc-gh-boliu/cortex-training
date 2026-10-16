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

"""--demo agrees with itself and moves on with the clock, as the real service's jobs do."""

import json
import re
from datetime import timedelta

import pytest
import requests

from cortex_training.tui import views
from cortex_training.tui.demo import UNAVAILABLE
from cortex_training.tui.demo import DemoClient
from cortex_training.tui.log_cache import job_cache_dir
from cortex_training.tui.stores import AliasStore
from tests.r1_cross_cutting_support import isolated_tui_files  # noqa: F401

GIB = 1024 ** 3
RAW = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{9}Z \{")


def _seeded() -> DemoClient:
    demo = DemoClient()
    demo.seed_local(aliases=AliasStore())
    return demo


def _gpu_rows(demo, job, tmp_path) -> list:
    out = tmp_path / views.job_id_of(job)
    demo.for_job(job).download_metrics(job["job_id"], out) if "database_name" in job else (
        demo.download_metrics(job["job_id"], out))
    return [json.loads(line) for path in out.glob("*/gpu.jsonl") for line in path.read_text().splitlines()]


def _all_jobs(demo) -> list:
    return list(demo.iter_jobs())


def _job(demo, **match) -> dict:
    return next(j for j in _all_jobs(demo) if all(j.get(k) == v for k, v in match.items()))


# ─── one story per job (R1-32) ────────────────────────────────────────────────


@pytest.mark.parametrize("hardware,total_gib", [("H200", 141), ("B200", 179)])
def test_each_gpu_type_reports_its_own_memory(tmp_path, hardware, total_gib):
    demo = _seeded()
    job = next(j for j in _all_jobs(demo) if j.get("hardware") == hardware and j["status"] == "CANCELLED"
               and j["reason"] == "cancelled_by_client" and demo._kinds.get(j["job_id"]))
    rows = _gpu_rows(demo, job, tmp_path)
    assert rows and {round((r["fb_used_bytes"] + r["fb_free_bytes"]) / GIB) for r in rows} == {total_gib}


def test_a_job_that_was_never_placed_has_no_gpu_samples_and_no_console(tmp_path):
    demo = _seeded()
    for reason in ("pending_timeout", "placement_too_large"):
        job = _job(demo, reason=reason)
        assert _gpu_rows(demo, job, tmp_path) == [], reason
        client = demo.for_job(job) if job["database_name"] != "CORTEX_TRAINING_DB" else demo
        assert client.download_stdout_logs(job["job_id"], tmp_path / "console" / reason) == []
        assert not (tmp_path / "console" / reason).exists()


def test_a_job_that_ran_out_of_memory_loading_was_barely_busy(tmp_path):
    demo = _seeded()
    rows = _gpu_rows(demo, _job(demo, reason="out_of_memory"), tmp_path)
    assert rows and max(r["gpu_util_pct"] for r in rows) < 20


def test_the_logs_follow_the_jobs_config():
    demo = _seeded()
    rl = next(j for j in demo.list_jobs() if len(j["sub_jobs"]) == 2 and j["status"] == "RUNNING")
    training = rl["sub_jobs"][0]["sub_job_id"]
    lines = [json.loads(e["_raw"].split(" ", 1)[1])["msg"]
             for e in demo.tail_logs(rl["job_id"], max_lines=20, sub_job_id=training)["entries"]]
    steps = [line for line in lines if line.startswith("step")]
    assert steps and all("reward=" in line and "lr=1.00e-06" in line for line in steps)  # GRPO at its own lr


def test_the_demo_says_it_is_a_demo_in_the_title(monkeypatch):
    from cortex_training.tui import __main__ as tui_main

    seen = {}

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, schema_only=False):
            self.title = "cortex training"

        def run(self):
            seen["title"] = self.title

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    assert tui_main.run(["--demo"]) == 0
    assert seen["title"] == "cortex training (demo)"


# ─── the clock moves on (R1-31) ────────────────────────────────────────────────


def _later(demo, monkeypatch, seconds):
    """Move the demo's wall clock ``seconds`` past its start (its for_job views too)."""
    monkeypatch.setattr(DemoClient, "_clock", lambda self: self.now + timedelta(seconds=seconds))


def test_live_logs_come_in_the_services_shape_and_keep_coming(monkeypatch):
    demo = _seeded()
    _later(demo, monkeypatch, 0)
    job = next(j for j in demo.list_jobs() if j["status"] == "RUNNING" and len(j["sub_jobs"]) == 1
               and j["sub_jobs"][0]["job_type"] == "TRAINING")
    sid = job["sub_jobs"][0]["sub_job_id"]
    first = demo.tail_logs(job["job_id"], max_lines=50, sub_job_id=sid)
    assert len(first["entries"]) == 50 and all(set(e) == {"_raw"} and RAW.match(e["_raw"]) for e in first["entries"])
    assert demo.tail_logs(job["job_id"], cursor=first["next_cursor"], sub_job_id=sid)["entries"] == []
    _later(demo, monkeypatch, 60)
    more = demo.tail_logs(job["job_id"], cursor=first["next_cursor"], sub_job_id=sid)
    assert 5 <= len(more["entries"]) <= 12  # a step about every 6 seconds
    newest = views.parse_ts(more["entries"][-1]["_raw"].split(" ", 1)[0])
    assert 0 <= (demo.now + timedelta(seconds=60) - newest).total_seconds() < 10
    # The same index is always the same line.
    again = demo.tail_logs(job["job_id"], cursor=str(int(first["next_cursor"]) - 1), sub_job_id=sid)
    assert again["entries"][0] == first["entries"][-1]


def test_gpu_samples_stay_fresh_while_a_job_runs(monkeypatch, tmp_path):
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if j["status"] == "RUNNING" and demo._kinds.get(j["job_id"]) == "sft")
    _later(demo, monkeypatch, 300)
    rows = _gpu_rows(demo, job, tmp_path)
    newest = max(views.parse_ts(r["sample_ts"]) for r in rows)
    assert (demo.now + timedelta(seconds=300) - newest).total_seconds() < 60


def test_the_placing_job_starts_up_and_runs_and_the_team_job_is_cancelled(monkeypatch):
    demo = _seeded()
    placing = next(j for j in demo.list_jobs() if j["status"] == "PLACING")
    team = _job(demo, database_name="TEAM_DB", status="RUNNING")
    team_client = demo.for_job(team)
    sid = placing["sub_jobs"][0]["sub_job_id"]
    for seconds, status in ((10, "PLACING"), (60, "INITIALIZING"), (130, "RUNNING")):
        _later(demo, monkeypatch, seconds)
        assert demo.get_job(placing["job_id"])["status"] == status
    lines = demo.tail_logs(placing["job_id"], sub_job_id=sid)["entries"]
    assert lines and "placed on 16 × B300" in lines[0]["_raw"]  # its log starts when it started running
    assert team_client.get_job(team["job_id"])["status"] == "RUNNING"
    _later(demo, monkeypatch, 400)
    ended = team_client.get_job(team["job_id"])
    assert (ended["status"], ended["reason"]) == ("CANCELLED", "cancelled_by_client")
    with pytest.raises(requests.HTTPError) as error:
        team_client.tail_logs(team["job_id"], sub_job_id=team["sub_jobs"][0]["sub_job_id"])
    assert error.value.response.status_code == 409 and "has been cancelled" in views.describe_error(error.value)


def test_a_demo_built_with_now_stays_at_that_moment():
    from tests.r1_cross_cutting_support import NOW

    demo = DemoClient(now=NOW)
    placing = next(j for j in demo.list_jobs() if j["status"] == "PLACING")
    assert demo._clock() == demo.now and demo.get_job(placing["job_id"])["status"] == "PLACING"


def test_the_list_can_fail_for_the_lists_error_view():
    demo = DemoClient(account_list=UNAVAILABLE)
    with pytest.raises(requests.HTTPError) as error:
        demo.list_jobs_page()
    assert views.describe_error(error.value) == "HTTP 503 Service Unavailable: Control Plane unavailable"
    with pytest.raises(ValueError):
        DemoClient(account_list="sometimes")


# ─── second pass: one sampling interval, and what else real users meet ────────


def test_long_jobs_report_their_gpus_every_5_seconds_as_the_service_does(tmp_path):
    from datetime import datetime
    from datetime import timezone

    from cortex_training.tui.resource import load_resource_data
    from cortex_training.tui.resource import sampling_text

    demo = _seeded()
    long_jobs = [j for j in _all_jobs(demo) if j.get("reason") not in ("pending_timeout", "placement_too_large")
                 and views.job_elapsed(j, demo.now) and views.job_elapsed(j, demo.now) > 5 * 3600]
    assert len(long_jobs) >= 3  # the 9 h, 7 h, 6 h and 5½ h jobs
    for job in long_jobs:
        out = tmp_path / views.job_id_of(job)
        client = demo.for_job(job) if "database_name" in job and job["database_name"] != "CORTEX_TRAINING_DB" else demo
        client.download_metrics(job["job_id"], out)
        rows = [json.loads(line) for path in out.glob("*/gpu.jsonl") for line in path.read_text().splitlines()]
        assert {r["sample_interval_s"] for r in rows} == {5.0}, job.get("comment")
        stamps = sorted({views.parse_ts(r["sample_ts"]).timestamp() for r in rows})
        newest_hour = [t for t in stamps if t >= stamps[-1] - 3600]
        assert {round(b - a, 3) for a, b in zip(newest_hour, newest_hour[1:])} == {5.0}
        assert len(stamps) <= 2500  # a demo file stays small: older samples are thinned
        for sub_job in load_resource_data(out).sub_jobs.values():
            text = sampling_text(sub_job, datetime.now(timezone.utc))
            assert text.startswith("scraped every 5 s  ·") and "rows report" not in text, text


def test_a_job_with_more_than_20000_lines_shows_the_scrollback_label():
    import asyncio

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.screens import JobDetailScreen

    demo = _seeded()
    job = next(j for j in demo.list_jobs() if (j.get("comment") or "").startswith("batch eval"))
    sid = job["sub_jobs"][0]["sub_job_id"]
    out = job_cache_dir("x") / "console-check"
    demo.download_stdout_logs(job["job_id"], out)
    assert len((out / sid / "stdout.log").read_text().splitlines()) > 25_000

    async def go():
        app = CortexTrainingLogTUI(demo, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            app.push_screen(JobDetailScreen(demo, job["job_id"], job=job, initial_tab="logs"))
            for _ in range(600):
                await pilot.pause(0.02)
                screen = app.screen
                if isinstance(screen, JobDetailScreen) and "last 20,000 lines" in (screen._log_status_text or ""):
                    break
            status = app.screen._log_status_text
            assert status.startswith("last 20,000 lines · cortex-training ") and status.endswith(
                f"download-log {job['job_id']} --log-type stdout for the full log"), status
            app.workers.cancel_all()
            for _ in range(5):
                await pilot.pause()

    asyncio.run(go())


def test_a_stuck_line_stops_the_tail_and_a_read_of_the_end_gets_past_it(monkeypatch):
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if (j.get("comment") or "").startswith("debug run"))
    sid = job["sub_jobs"][0]["sub_job_id"]
    _later(demo, monkeypatch, 0)
    first = demo.tail_logs(job["job_id"], max_lines=50, sub_job_id=sid)
    cursor = first["next_cursor"]
    _later(demo, monkeypatch, 120)  # the 300 KiB line came at one minute
    page = demo.tail_logs(job["job_id"], cursor=cursor, sub_job_id=sid)
    assert 5 <= len(page["entries"]) <= 12  # the lines before it
    stuck = page["next_cursor"]
    _later(demo, monkeypatch, 240)
    assert demo.tail_logs(job["job_id"], cursor=stuck, sub_job_id=sid) == {
        "entries": [], "next_cursor": stuck, "eof": False}  # stuck, while the job logs on
    end = demo.tail_logs(job["job_id"], max_lines=10, sub_job_id=sid)  # the Logs tab's read of the end
    assert len(end["entries"]) == 10 and int(end["next_cursor"]) > int(stuck) + 10
    # A read of the end that still reaches back to the line stops before it, as the cursor does.
    assert demo.tail_logs(job["job_id"], max_lines=200, sub_job_id=sid)["next_cursor"] == stuck


def test_the_logs_tab_pauses_a_demo_tail_left_idle(monkeypatch):
    import asyncio

    from cortex_training.tui.app import CortexTrainingLogTUI
    from cortex_training.tui.app import LogScreen
    from cortex_training.tui.screens import JobDetailScreen

    monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 1.0)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.2)
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if j["status"] == "RUNNING" and demo._kinds.get(j["job_id"]) == "sft")

    async def go():
        app = CortexTrainingLogTUI(demo, poll_interval=0.05)
        async with app.run_test(size=(160, 45)) as pilot:
            app.push_screen(JobDetailScreen(demo, job["job_id"], job=job, initial_tab="logs"))
            for _ in range(300):
                await pilot.pause(0.02)
                if isinstance(app.screen, JobDetailScreen) and app.screen._idle_paused:
                    break
            assert app.screen._idle_paused and "paused: no input for" in app.screen._log_status_text
            app.workers.cancel_all()
            for _ in range(5):
                await pilot.pause()

    asyncio.run(go())


def test_the_placing_jobs_sampling_sub_job_is_not_routed_until_the_job_runs(monkeypatch):
    demo = _seeded()
    placing = next(j for j in demo.list_jobs() if j["status"] == "PLACING")
    training, sampling = (s["sub_job_id"] for s in placing["sub_jobs"])
    _later(demo, monkeypatch, 10)
    assert demo.tail_logs(placing["job_id"], sub_job_id=training)["entries"] == []  # its pod isn't up yet
    with pytest.raises(requests.HTTPError) as error:
        demo.tail_logs(placing["job_id"], sub_job_id=sampling)
    assert error.value.response.status_code == 400
    assert "does not belong to session" in views.describe_error(error.value)
    _later(demo, monkeypatch, 130)
    assert demo.tail_logs(placing["job_id"], sub_job_id=sampling)["entries"]


def test_a_role_that_cant_read_a_jobs_experiment_is_told_so(tmp_path):
    demo = _seeded()
    team = _job(demo, comment="team grpo · math v1")
    client = demo.for_job(team)
    for read in (lambda: client.download_metrics(team["job_id"], tmp_path / "m"),
                 lambda: client.download_stdout_logs(team["job_id"], tmp_path / "c"),
                 lambda: client.fetch_execution_logs(team["job_id"])):
        with pytest.raises(Exception) as error:
            read()
        assert views.experiment_access(error.value) == "sql"
    gone = next(j for j in demo.list_jobs() if demo._experiment.get(j["job_id"]) == "missing")
    with pytest.raises(requests.HTTPError) as error:
        demo.download_metrics(gone["job_id"], tmp_path / "m2")
    assert views.experiment_access(error.value) == "missing"


def test_a_log_prob_sub_job_requests_no_gpus_and_reports_none(tmp_path):
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if any(views.sub_job_kind(s) == "log_prob" for s in j["sub_jobs"]))
    log_prob = next(s for s in job["sub_jobs"] if views.sub_job_kind(s) == "log_prob")
    assert log_prob["sub_job_id"].endswith(":log_prob:0") and views.requested_gpus(log_prob) == 0
    assert views.total_gpus(job) == 4
    demo.download_metrics(job["job_id"], tmp_path)
    assert not (tmp_path / log_prob["sub_job_id"]).exists()
    demo.download_stdout_logs(job["job_id"], tmp_path / "console")
    console = (tmp_path / "console" / log_prob["sub_job_id"] / "stdout.log").read_text()
    assert "log_prob worker scheduled beside training:0" in console and "scored batch=" in console


@pytest.mark.parametrize("status,ended", [("SUCCEEDED", "has completed"), ("DONE", "has completed"),
                                          ("JOB_STATE_CANCELLED", "has been cancelled")])
def test_jobs_in_older_and_unlisted_states_end_as_the_service_says(status, ended):
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if j["status"] == status)
    assert not views.is_live(job)
    with pytest.raises(requests.HTTPError) as error:
        demo.tail_logs(job["job_id"], sub_job_id=job["sub_jobs"][0]["sub_job_id"])
    assert error.value.response.status_code == 409 and ended in views.describe_error(error.value)
    # A status filter on the account-wide list matches the customer states only.
    terminal = {r["job_id"] for r in demo.iter_jobs(status="terminal")}
    assert (job["job_id"] in terminal) is (status == "JOB_STATE_CANCELLED")


def test_an_unknown_job_its_control_plane_doesnt_know_has_no_sub_jobs():
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if j["job_id"] in demo._not_found)
    assert (job["status"], job["sub_jobs"]) == ("UNKNOWN", [])
    assert job["reason"] == "Job status unavailable: job not found on Control Plane"
    assert "created_at" not in job and _job(demo, job_id=job["job_id"])["created_at"]  # the account list dates it
    unknown = [views.status_unavailable_text(j) for j in demo.list_jobs() if views.is_unknown(j)]
    assert unknown[0].endswith("Control Plane is unavailable") and len(unknown) == 2


@pytest.mark.parametrize("kind, seconds, values", [("sft", 600, r"loss=\S+ +grad_norm=\S+"),
                                                   ("rl", 3600, r"reward=\S+ +kl=\S+ +policy_loss=\S+")])
def test_a_running_jobs_step_lines_have_values_of_their_own(monkeypatch, kind, seconds, values):
    demo = _seeded()
    job = next(j for j in demo.list_jobs() if j["status"] == "RUNNING" and demo._kinds.get(j["job_id"]) == kind)
    sid = job["sub_jobs"][0]["sub_job_id"]
    _later(demo, monkeypatch, seconds)
    lines = [json.loads(e["_raw"].split(" ", 1)[1])["msg"]
             for e in demo.tail_logs(job["job_id"], max_lines=40, sub_job_id=sid)["entries"]]
    steps = [line for line in lines if line.startswith("step")]
    found = {re.search(values, line).group(0) for line in steps}
    assert len(steps) >= 30 and len(found) == len(steps)  # no line repeats another's values
    numbers = [int(line.split()[1]) for line in steps]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)


def test_only_a_job_that_trains_sends_its_sampling_engine_weights():
    demo = _seeded()
    checked = set()
    for job in _all_jobs(demo):
        kinds = {views.sub_job_kind(s) for s in job["sub_jobs"]}
        for sj in job["sub_jobs"]:
            if views.sub_job_kind(sj) != "sampling" or job["status"] != "RUNNING":
                continue
            client = demo.for_job(job)
            msgs = [json.loads(e["_raw"].split(" ", 1)[1])["msg"]
                    for e in client.tail_logs(job["job_id"], max_lines=60, sub_job_id=sj["sub_job_id"])["entries"]]
            assert any("received weights" in m for m in msgs) is ("training" in kinds), job.get("comment")
            checked.add("training" in kinds)
    assert checked == {True, False}  # an RL job and an inference-only one
