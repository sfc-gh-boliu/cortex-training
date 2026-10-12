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

"""Synthetic, offline client for previewing the TUI (``cortex-training tui --demo``).

Implements the read-only client methods the TUI calls, returning records in
the shapes the service returns (job / sub-job metadata, checkpoint lists,
``gpu.jsonl`` telemetry, tail-logs pages). Every value is generated from a
fixed seed relative to "now"; nothing touches the network.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Optional

from cortex_training.tui.stores import AliasStore
from cortex_training.tui.stores import MetricsStore

_GIB = 1024**3
_H200_FB = 141 * _GIB


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _training(model, gpus, *, lr=2e-5, bs=8, seq=4096, lora=None, opt="AdamW"):
    cfg: dict = {"n_gpus": float(gpus), "max_seq_len": float(seq), "train_batch_size": float(bs),
                 "optimizer": {"name": opt, "lr": lr}}
    if lora:
        cfg["peft_config"] = {"peft_type": "Lora", "r": float(lora), "lora_alpha": float(lora * 2),
                              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    return {"job_type": "TRAINING", "model_name": model, "dtype": "bfloat16", "seed": 42, "training_config": cfg}


def _sampling(model, gpus, *, seq=4096, mem=0.85):
    return {"job_type": "SAMPLING", "model_name": model, "dtype": "bfloat16", "seed": 42,
            "inference_config": {"n_gpus": float(gpus), "max_seq_len": float(seq), "gpu_memory_utilization": mem}}


# (alias, status, reason, age, duration, hardware, sub-jobs, kind) — kind drives metrics.
_SPECS = [
    ("qwen3-8b lora · support-chat v4", "RUNNING", "", timedelta(hours=2, minutes=14), None, "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=1e-4, bs=16, seq=8192, lora=16)], "sft"),
    ("gsm8k grpo · qwen3.6-35b", "RUNNING", "", timedelta(hours=5, minutes=40), None, "H200",
     [_training("Qwen/Qwen3.6-35B-A3B", 8, lr=1e-6, bs=64, seq=4096), _sampling("Qwen/Qwen3.6-35B-A3B", 8)], "rl"),
    ("", "PLACING", "", timedelta(minutes=3), None, "B300",
     [_training("Qwen/Qwen3.8-27B", 16, lr=1e-5, bs=32, seq=16384)], None),
    ("qwen3-8b full sft · identity", "FAILED", "sub_job_readiness", timedelta(days=1, hours=3), timedelta(minutes=31), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=4, seq=32768)], "sft-fail"),
    ("support-chat v3 (baseline)", "CANCELLED", "cancelled_by_client", timedelta(days=2, hours=6), timedelta(hours=3, minutes=12), "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=1e-4, bs=16, seq=8192, lora=8)], "sft"),
    ("byo weights smoke · qwen3-0.6b", "CANCELLED", "cancelled_by_client", timedelta(days=4), timedelta(minutes=20), "H200",
     [_training("Qwen/Qwen3-0.6B", 1, lr=1e-5, bs=1, seq=128), _sampling("Qwen/Qwen3-0.6B", 1, seq=128, mem=0.8)], "sft"),
    ("", "TERMINATED", "idle_timeout", timedelta(days=9), timedelta(hours=1, minutes=2), "H200",
     [_sampling("Qwen/Qwen3-8B", 2, seq=32768)], None),
    ("qwen3.5-9b lora r32 sweep", "CANCELLED", "cancelled_by_client", timedelta(days=16), timedelta(hours=6), "B200",
     [_training("Qwen/Qwen3.5-9B", 8, lr=5e-5, bs=32, seq=8192, lora=32)], "sft"),
    ("qwen3.5-9b lora r8 sweep", "CANCELLED", "cancelled_by_client", timedelta(days=16, hours=1), timedelta(hours=5, minutes=30), "B200",
     [_training("Qwen/Qwen3.5-9B", 8, lr=5e-5, bs=32, seq=8192, lora=8)], "sft"),
    ("", "FAILED", "pending_timeout", timedelta(days=27), timedelta(hours=24), "B300",
     [_training("Qwen/Qwen3.6-35B-A3B", 32, lr=1e-6, bs=128, seq=4096)], None),
    ("math grpo · 8b (paper repro)", "CANCELLED", "cancelled_by_client", timedelta(days=41), timedelta(hours=9), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=1e-6, bs=32, seq=4096), _sampling("Qwen/Qwen3-8B", 4)], "rl"),
    ("", "CANCELLED", "cancelled_by_client", timedelta(days=58), timedelta(minutes=48), "H200",
     [_sampling("Qwen/Qwen3-8B", 2)], None),
    ("contract-extraction sft", "TERMINATED", "idle_timeout", timedelta(days=77), timedelta(hours=4), "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=2e-5, bs=16, seq=16384)], "sft"),
    ("", "FAILED", "out_of_memory", timedelta(days=96), timedelta(minutes=7), "H200",
     [_training("Qwen/Qwen3.8-27B", 8, lr=1e-5, bs=64, seq=32768)], None),
    ("", "CANCELLED", "cancelled_by_client", timedelta(days=120), timedelta(hours=2), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=3e-5, bs=8, seq=4096, lora=16)], None),
    ("first conversational sft", "CANCELLED", "cancelled_by_client", timedelta(days=171), timedelta(minutes=25), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=4, seq=32768)], None),
    ("", "CANCELLED", "cancelled_by_client", timedelta(days=201), timedelta(hours=1), "H200",
     [_training("Qwen/Qwen3-8B", 4)], None),  # older than six months: hidden
    ("", "FAILED", "sub_job_readiness", timedelta(days=230), timedelta(minutes=12), "H200",
     [_sampling("Qwen/Qwen3-8B", 2)], None),  # older than six months: hidden
]


class DemoClient:
    """Deterministic in-memory stand-in for :class:`CortexTrainingClient` (read paths only)."""

    def __init__(self, *, now: Optional[datetime] = None, seed: int = 7) -> None:
        self.now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
        self._rng = random.Random(seed)
        self._jobs: list = []
        self._kinds: dict = {}
        self._aliases: dict = {}
        for alias, status, reason, age, duration, hw, subs, kind in _SPECS:
            jid = _uid(self._rng)
            created = self.now - age
            ended = created + duration if duration else None
            updated = ended or self.now - timedelta(seconds=self._rng.randint(4, 40))
            counts: dict = {}
            sub_jobs = []
            for sj in subs:
                t = sj["job_type"].lower()
                i = counts.get(t, 0)
                counts[t] = i + 1
                sub_status = status
                if status == "FAILED" and reason == "sub_job_readiness" and len(subs) > 1 and t == "training":
                    sub_status = "CANCELLED"
                sub_jobs.append({"sub_job_id": f"{jid}:{t}:{i}", "status": sub_status, **json.loads(json.dumps(sj))})
            self._jobs.append({
                "job_id": jid, "status": status, "reason": reason, "created_at": _iso(created),
                "updated_at": _iso(updated), "state_changed_at": _iso(ended or updated), "sub_jobs": sub_jobs,
                "image_tag": "release_demo", "hardware": hw, "owner_role": "ML_RESEARCH",
                "submitted_by": self._rng.choice(["USER_A", "USER_B", "USER_A", "USER_C"]), "comment": None,
                "idle_timeout_seconds": 1800, "pending_timeout_seconds": 86400,
            })
            self._kinds[jid] = kind
            if alias:
                self._aliases[jid] = alias

    # ── seeding local stores (aliases + client-logged metrics) ──
    def seed_local(self, *, aliases: AliasStore, metrics: MetricsStore) -> None:
        for jid, name in self._aliases.items():
            aliases.set(jid, name)
        for job in self._jobs:
            kind = self._kinds.get(job["job_id"])
            if kind and not metrics.load(job["job_id"]):
                self._seed_metrics(job, kind, metrics)

    def _seed_metrics(self, job: dict, kind: str, store: MetricsStore) -> None:
        rng = random.Random(job["job_id"])
        start = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00")) + timedelta(minutes=6)
        end = datetime.fromisoformat(job["state_changed_at"].replace("Z", "+00:00"))
        if job["status"] == "RUNNING":
            end = self.now - timedelta(seconds=20)
        span = max(60.0, (end - start).total_seconds())
        if kind == "rl":
            steps = int(min(400, span / 45))
            for s in range(steps):
                frac = s / max(1, steps - 1)
                reward = 0.18 + 0.55 * (1 - math.exp(-4 * frac)) + rng.gauss(0, 0.025)
                store.append(job["job_id"], {
                    "policy_loss": 0.04 * math.exp(-2 * frac) + rng.gauss(0, 0.006),
                    "reward": reward,
                    "kl": 0.002 + 0.03 * frac + abs(rng.gauss(0, 0.003)),
                    "entropy": 0.92 - 0.45 * frac + rng.gauss(0, 0.02),
                    "clip_frac": 0.06 + 0.05 * math.sin(frac * 6) ** 2 + abs(rng.gauss(0, 0.01)),
                    "completion_length": 380 - 120 * frac + rng.gauss(0, 18),
                    "eval_success_rate": min(0.95, 0.22 + 0.6 * frac + rng.gauss(0, 0.02)) if s % 20 == 0 else None,
                    "learning_rate": 1e-6,
                    "grad_norm": 0.6 + 0.4 * math.exp(-3 * frac) + abs(rng.gauss(0, 0.08)),
                    "step_time": 38 + rng.gauss(0, 3),
                }, step=s, ts=(start + timedelta(seconds=span * frac)).isoformat())
            return
        steps = int(min(1500, span / 6))
        if kind == "sft-fail":
            steps = min(steps, 140)
        for s in range(steps):
            frac = s / max(1, steps - 1)
            loss = 0.38 + 1.9 * math.exp(-5.5 * frac) + rng.gauss(0, 0.035)
            if kind == "sft-fail" and s > 110:
                loss += (s - 110) * 0.09  # diverging before the failure
            warm = min(1.0, s / 50)
            rec = {
                "train_loss": loss,
                "learning_rate": 1e-4 * warm * (1 - 0.9 * frac),
                "grad_norm": 1.4 * math.exp(-3 * frac) + 0.25 + abs(rng.gauss(0, 0.06)) + (3.0 if kind == "sft-fail" and s > 120 else 0),
                "step_time": 5.6 + rng.gauss(0, 0.25),
                "tokens_per_second": 21500 + rng.gauss(0, 600),
            }
            if s % 50 == 0 and s:
                rec["val_loss"] = loss + 0.06 + rng.gauss(0, 0.01)
            store.append(job["job_id"], rec, step=s, ts=(start + timedelta(seconds=span * frac)).isoformat())

    # ── client surface used by the TUI ──
    def list_jobs(self, status: Optional[str] = None) -> list:
        jobs = [json.loads(json.dumps(j)) for j in self._jobs]
        return [j for j in jobs if status is None or j["status"].lower() == status.lower()]

    def get_job(self, job_id: str) -> dict:
        for j in self._jobs:
            if j["job_id"] == job_id:
                return json.loads(json.dumps(j))
        raise KeyError(job_id)

    def list_checkpoints(self, job_id: str) -> list:
        job = self.get_job(job_id)
        kind = self._kinds.get(job_id)
        if kind not in ("sft", "rl", "sft-fail"):
            return []
        rng = random.Random("cp" + job_id)
        created = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
        records = MetricsStore().load(job_id)
        last_step = max((r.get("step", 0) for r in records), default=600)
        every = 50 if kind in ("rl", "sft-fail") else 250
        out = []
        for step in range(every, last_step + 1, every):
            rec = next((r for r in records if r.get("step") == step), {})
            ts = created + timedelta(minutes=6 + 10 * step / every)
            if rec.get("ts"):
                ts = datetime.fromisoformat(rec["ts"])
            kind_cp = "resumable" if (step // every) % 2 else "weights-only"
            # The service's checkpoint records carry only these three fields.
            out.append({"checkpoint_id": f"cp_{_uid(rng)}", "checkpoint_type": kind_cp,
                        "created_at": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")})
        return out

    def download_metrics(self, job_id: str, output_dir, *, resume: bool = False) -> list:
        job = self.get_job(job_id)
        created = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00")) + timedelta(minutes=2)
        end = self.now if job["status"] in ("RUNNING",) else datetime.fromisoformat(job["state_changed_at"].replace("Z", "+00:00"))
        if job["status"] in ("PLACING", "PENDING"):
            return []
        written = []
        for sj in job["sub_jobs"]:
            sid = sj["sub_job_id"]
            n_gpus = int((sj.get("training_config") or sj.get("inference_config") or {}).get("n_gpus", 1))
            sampling = sj["job_type"].lower() == "sampling"
            rng = random.Random("gpu" + sid)
            path = Path(output_dir) / sid / "gpu.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            interval = 5.0
            total = (end - created).total_seconds()
            stride = max(interval, total / 2400)  # bound the demo file size on long jobs
            lines = []
            t = 0.0
            seq = 0
            while t <= total:
                ts = created + timedelta(seconds=t)
                phase = t / max(total, 1)
                for g in range(n_gpus):
                    if sampling:
                        busy = 0.5 + 0.5 * math.sin(t / 90.0 + g) ** 2
                        util = max(0.0, min(100.0, 35 + 50 * busy + rng.gauss(0, 6)))
                        used = 0.88 * _H200_FB + rng.gauss(0, 0.3) * _GIB
                    else:
                        dip = 35 if int(t / 600) % 7 == 3 else 0  # periodic checkpoint/eval stalls
                        util = max(0.0, min(100.0, 91 - dip + rng.gauss(0, 4) - 25 * math.exp(-phase * 40)))
                        used = (0.56 + 0.12 * min(1.0, phase * 30)) * _H200_FB + rng.gauss(0, 0.4) * _GIB
                    used = max(0.0, min(_H200_FB, used))
                    lines.append(json.dumps({
                        "record_version": 2, "sub_job_id": sid,
                        "sample_ts": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "worker_num": g // 8, "local_gpu_index": g % 8, "global_gpu_index": g,
                        "gpu_uuid": f"gpu-{sid[-12:]}-{g}", "gpu_util_pct": round(util, 1),
                        "fb_used_bytes": float(used), "fb_free_bytes": float(_H200_FB - used),
                        "power_watts": round(180 + 5 * util, 1), "gpu_temp_c": round(38 + util / 3, 1),
                        "sm_active_pct": None, "sm_occupancy_pct": None, "tensor_active_pct": round(util * 0.6, 1),
                        "dram_active_pct": round(util * 0.4, 1), "seq": seq, "seq_epoch": "demo", "sample_interval_s": stride,
                    }))
                seq += 1
                t += stride
            data = ("\n".join(lines) + "\n").encode("utf-8")
            path.write_bytes(data)
            # The resume manifest the real client writes; the Resources tab reads
            # only the bytes it commits.
            manifest = path.with_name("." + path.name + ".manifest.json")
            manifest.write_text(json.dumps({"version": 1, "names": [f"_metrics/{sid}/gpu.demo.jsonl.gz"], "bytes": len(data)}),
                                encoding="utf-8")
            written.append({"sub_job_id": sid, "path": str(path)})
        return written

    def _log_lines(self, job: dict, sj: dict) -> list:
        rng = random.Random("log" + sj["sub_job_id"])
        created = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
        t = created
        sampling = sj["job_type"].lower() == "sampling"
        model = sj["model_name"]
        n = int((sj.get("training_config") or sj.get("inference_config") or {}).get("n_gpus", 1))
        out = []

        def emit(level, logger, msg, dt=3.0):
            nonlocal t
            t += timedelta(seconds=dt * (0.5 + rng.random()))
            out.append({"ts": t.astimezone().strftime("%H:%M:%S"), "level": level, "logger": logger, "msg": msg})

        emit("INFO", "zone", f"placed on {n} × {job['hardware']} (zone ZONE_A)")
        emit("INFO", "ray", f"ray head ready; {max(1, n // 8)} worker node(s) joined", 40)
        emit("INFO", "loader", f"loading {model} weights (bfloat16)", 5)
        emit("INFO", "loader", "weights loaded in 74.2s", 70)
        if sampling:
            emit("INFO", "vllm", "engine ready: max_model_len=4096 gpu_memory_utilization=0.85", 20)
            for i in range(24):
                emit("INFO", "vllm", f"generate batch={rng.choice([32, 64, 128])} prompt_tokens={rng.randint(9000, 30000)} "
                     f"throughput={rng.randint(5200, 7400)} tok/s", 40)
                if i == 11:
                    emit("INFO", "weight-sync", "received weights from training:0 (broadcast, 4.1s); prefix cache reset")
        else:
            emit("INFO", "deepspeed", "ZeRO stage 2 · activation checkpointing on · flash_attention_3", 6)
            loss = 2.3
            for s in range(0, 220, 10):
                loss = max(0.4, loss * 0.9 + rng.gauss(0, 0.02))
                emit("INFO", "train", f"step {s:>4}  loss={loss:.4f}  grad_norm={rng.uniform(0.3, 1.2):.3f}  "
                     f"lr={1e-4 * min(1, s / 50):.2e}  tok/s={rng.randint(20500, 22800)}", 55)
                if s == 120:
                    emit("WARNING", "train", "dataloader stalled 11.8s waiting for next batch (consider num_workers>4)")
                if s == 200:
                    emit("INFO", "checkpoint", "saved weights-only checkpoint global_step200 → stage (38.2s)")
            if job["status"] == "FAILED":
                emit("ERROR", "worker", "rank 3: CUDA error: an illegal memory access was encountered")
                emit("ERROR", "zone", "sub-job readiness probe failed 3 times; tearing down")
        return out

    def tail_logs(self, job_id: str, *, cursor: Optional[str] = None, max_lines: Optional[int] = None,
                  sub_job_id: Optional[str] = None) -> dict:
        job = self.get_job(job_id)
        sj = next((s for s in job["sub_jobs"] if s["sub_job_id"] == sub_job_id), job["sub_jobs"][0])
        lines = self._log_lines(job, sj)
        pos = int(cursor or 0)
        if pos >= len(lines):
            return {"entries": [], "next_cursor": str(pos), "eof": False}
        return {"entries": lines[pos:], "next_cursor": str(len(lines)), "eof": False}

    def download_stdout_logs(self, job_id: str, output_dir, *, resume: bool = False) -> list:
        from cortex_training.tui.format import format_log_entry

        job = self.get_job(job_id)
        for sj in job["sub_jobs"]:
            path = Path(output_dir) / sj["sub_job_id"] / "stdout.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(format_log_entry(e) for e in self._log_lines(job, sj)) + "\n", encoding="utf-8")
        return []

    def close(self) -> None:
        pass
