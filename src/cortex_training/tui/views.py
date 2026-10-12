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

"""Pure view-model helpers for the TUI's job list, summary, and detail tabs.

Search, sorting, summary fields, GPU and training-metric series, checkpoint
details, and plain-text charts. Like ``format.py`` this never imports
``textual``, so every rule here is unit-testable without the TUI installed.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Iterable
from typing import Optional

from cortex_training.tui.format import _ACTIVE_STATUSES
from cortex_training.tui.format import _num
from cortex_training.tui.format import _subjob_config
from cortex_training.tui.format import source_label

RETENTION_DAYS = 183  # six months of job history

# ─── time ────────────────────────────────────────────────────────────────────


def parse_ts(raw: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (nanosecond precision allowed) as aware UTC."""
    if not raw:
        return None
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Python's fromisoformat takes at most 6 fractional digits.
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_ts(raw: Any) -> str:
    dt = raw if isinstance(raw, datetime) else parse_ts(raw)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M") if dt else "—"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_ago(raw: Any, now: datetime) -> str:
    dt = raw if isinstance(raw, datetime) else parse_ts(raw)
    if not dt:
        return "—"
    return f"{fmt_duration((now - dt).total_seconds())} ago"


# ─── job fields ──────────────────────────────────────────────────────────────


def status_word(status: Any) -> str:
    return (str(status or "").strip() or "?").removeprefix("JOB_STATE_").lower()


def is_live(job: Any) -> bool:
    """Strictly a recognized active state (unknown counts as historical here)."""
    return isinstance(job, dict) and status_word(job.get("status")) in _ACTIVE_STATUSES


def job_id_of(job: dict) -> str:
    return str(job.get("job_id") or job.get("id") or "")


def display_name(job: dict, aliases: Optional[dict] = None) -> str:
    """Local alias first, then a server-side ``comment``; empty when neither."""
    jid = job_id_of(job)
    alias = (aliases or {}).get(jid)
    if alias:
        return alias
    comment = job.get("comment")
    return str(comment).strip() if comment else ""


def sub_jobs(job: dict) -> list:
    return [s for s in job.get("sub_jobs") or [] if isinstance(s, dict)]


def sub_job_label(sj: Any) -> str:
    """``training #0`` style label; the API spells job types in upper case."""
    if isinstance(sj, dict):
        sj = {**sj, "job_type": str(sj.get("job_type") or sj.get("type") or "").lower() or None}
    return source_label(sj)


def job_types(job: dict) -> list:
    types = [str(s.get("job_type") or s.get("type") or "").lower() for s in sub_jobs(job)]
    return [t for t in types if t]


def job_type_label(job: dict) -> str:
    types = job_types(job)
    if "training" in types and "sampling" in types:
        return "train+sample"
    return ", ".join(dict.fromkeys(types)) or "—"


def sub_job_gpus(sj: dict) -> int:
    cfg = _subjob_config(sj)
    try:
        return int(float(cfg.get("n_gpus") or 0))
    except (TypeError, ValueError):
        return 0


def total_gpus(job: dict) -> int:
    return sum(sub_job_gpus(s) for s in sub_jobs(job))


def job_models(job: dict) -> list:
    return list(dict.fromkeys(str(s["model_name"]) for s in sub_jobs(job) if s.get("model_name")))


def owner(job: dict) -> str:
    user = job.get("submitted_by") or ""
    role = job.get("owner_role") or ""
    if user and role:
        return f"{user} ({role})"
    return str(user or role or "—")


def training_method(sj: dict) -> str:
    cfg = _subjob_config(sj)
    return "lora" if isinstance(cfg.get("peft_config"), dict) else "full"


def job_elapsed(job: dict, now: datetime) -> Optional[float]:
    start = parse_ts(job.get("created_at"))
    if not start:
        return None
    end = now if is_live(job) else parse_ts(job.get("state_changed_at") or job.get("updated_at")) or now
    return max(0.0, (end - start).total_seconds())


def within_retention(job: dict, now: datetime, days: int = RETENTION_DAYS) -> bool:
    created = parse_ts(job.get("created_at"))
    return created is None or (now - created).total_seconds() <= days * 86400


def subjob_config_summary(sj: dict) -> str:
    """Compact human summary of one sub-job's create-time config."""
    cfg = _subjob_config(sj)
    bits = []
    opt = cfg.get("optimizer")
    if isinstance(opt, dict):
        o = str(opt.get("name") or "")
        if opt.get("lr") is not None:
            o = f"{o} lr={_num(opt['lr'])}".strip()
        if o:
            bits.append(o)
    for key, label in (("train_batch_size", "bs"), ("max_seq_len", "seq")):
        if cfg.get(key) is not None:
            bits.append(f"{label}={_num(cfg[key])}")
    peft = cfg.get("peft_config")
    if isinstance(peft, dict):
        lora = "LoRA"
        if peft.get("r") is not None:
            lora += f" r={_num(peft['r'])}"
        if peft.get("lora_alpha") is not None:
            lora += f" α={_num(peft['lora_alpha'])}"
        bits.append(lora)
    elif str(sj.get("job_type") or "").lower() == "training":
        bits.append("full-param")
    if cfg.get("gpu_memory_utilization") is not None:
        bits.append(f"mem={_num(cfg['gpu_memory_utilization'])}")
    return " · ".join(bits)


def summary_fields(job: dict, name: str, now: datetime) -> list:
    """(label, value) rows for the job preview, in display order."""
    models = job_models(job)
    hw = job.get("hardware") or "—"
    gpus = total_gpus(job)
    split = " + ".join(f"{sub_job_gpus(s)} {sub_job_label(s)}" for s in sub_jobs(job) if sub_job_gpus(s))
    return [
        ("Name", name or "— (press n to name this job)"),
        ("Job ID", job_id_of(job)),
        ("Status", status_word(job.get("status")).upper()),
        ("Model", ", ".join(models) or "—"),
        ("Type", job_type_label(job)),
        ("Owner", owner(job)),
        ("GPUs", f"{gpus} × {hw}" + (f"  ({split})" if split and len(sub_jobs(job)) > 1 else "")),
        ("Created", fmt_ts(job.get("created_at"))),
        ("Elapsed", fmt_duration(job_elapsed(job, now)) + ("  (running)" if is_live(job) else "")),
        ("Last update", f"{fmt_ts(job.get('updated_at'))}  ({fmt_ago(job.get('updated_at'), now)})"),
    ]


# ─── search ──────────────────────────────────────────────────────────────────

# field -> help text; the field names users type in the search box.
SEARCH_FIELDS = {
    "name": "display name",
    "id": "job id",
    "status": "job status",
    "owner": "submitting user or owning role",
    "type": "training / sampling",
    "model": "model name",
    "method": "full / lora",
    "gpu": "GPU type (H200, B200, B300)",
    "gpus": "total GPU count",
    "optimizer": "optimizer name",
    "lr": "learning rate",
    "batch": "train batch size",
    "seq": "max sequence length",
    "lora_r": "LoRA rank",
    "lora_alpha": "LoRA alpha",
    "dtype": "dtype",
}
_FIELD_ALIASES = {
    "hardware": "gpu",
    "hw": "gpu",
    "gpu_type": "gpu",
    "n_gpus": "gpus",
    "learning_rate": "lr",
    "opt": "optimizer",
    "bs": "batch",
    "batch_size": "batch",
    "train_batch_size": "batch",
    "seq_len": "seq",
    "max_seq_len": "seq",
    "rank": "lora_r",
    "r": "lora_r",
    "alpha": "lora_alpha",
    "user": "owner",
    "job_type": "type",
    "model_name": "model",
}
_OPS = (">=", "<=", "!=", ":", "=", ">", "<")
_TOKEN = re.compile(r"^([A-Za-z_][\w.]*)(>=|<=|!=|:|=|>|<)(.+)$")


@dataclass
class Query:
    terms: list = field(default_factory=list)
    filters: list = field(default_factory=list)  # (field, op, value)
    errors: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.terms and not self.filters


def parse_query(text: str) -> Query:
    """Free-text words plus ``field<op>value`` filters (ops ``: = != > >= < <=``).

    A dotted field (``training_config.optimizer.lr``) is an advanced filter that
    reads that member of each sub-job's config directly.
    """
    q = Query()
    try:
        tokens = shlex.split(text or "")
    except ValueError:
        tokens = (text or "").split()
    for tok in tokens:
        m = _TOKEN.match(tok)
        if not m:
            q.terms.append(tok.lower())
            continue
        key, op, value = m.group(1).lower(), m.group(2), m.group(3)
        key = _FIELD_ALIASES.get(key, key)
        if key not in SEARCH_FIELDS and "." not in key:
            q.errors.append(f"unknown field '{m.group(1)}'")
            continue
        q.filters.append((key, op, value))
    return q


def _dig(obj: Any, path: list) -> list:
    if not path:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in _dig(item, path)]
    if isinstance(obj, dict):
        key = path[0]
        for k, v in obj.items():
            if str(k).lower() == key:
                return _dig(v, path[1:])
    return []


def field_values(job: dict, key: str, name: str = "") -> list:
    """All values a job carries for a search field (one per sub-job when relevant)."""
    if "." in key:
        path = key.split(".")
        found = _dig(job, path)
        for sj in sub_jobs(job):
            found.extend(_dig(sj, path))
            found.extend(_dig(_subjob_config(sj), path))
        return [v for v in found if not isinstance(v, (dict, list))]
    subs = sub_jobs(job)
    cfgs = [_subjob_config(s) for s in subs]
    if key == "name":
        return [name] if name else []
    if key == "id":
        return [job_id_of(job)] + [str(s.get("sub_job_id") or "") for s in subs]
    if key == "status":
        return [status_word(job.get("status"))]
    if key == "owner":
        return [str(v) for v in (job.get("submitted_by"), job.get("owner_role")) if v]
    if key == "type":
        return job_types(job)
    if key == "model":
        return job_models(job)
    if key == "method":
        return [training_method(s) for s in subs if str(s.get("job_type") or "").lower() == "training"]
    if key == "gpu":
        return [job["hardware"]] if job.get("hardware") else []
    if key == "gpus":
        return [total_gpus(job)]
    if key == "dtype":
        return [s["dtype"] for s in subs if s.get("dtype")]
    if key in ("optimizer", "lr"):
        opts = [c.get("optimizer") for c in cfgs if isinstance(c.get("optimizer"), dict)]
        return [o.get("name" if key == "optimizer" else "lr") for o in opts if o.get("name" if key == "optimizer" else "lr") is not None]
    if key == "batch":
        return [c["train_batch_size"] for c in cfgs if c.get("train_batch_size") is not None]
    if key == "seq":
        return [c["max_seq_len"] for c in cfgs if c.get("max_seq_len") is not None]
    if key in ("lora_r", "lora_alpha"):
        peft_key = "r" if key == "lora_r" else "lora_alpha"
        return [c["peft_config"][peft_key] for c in cfgs if isinstance(c.get("peft_config"), dict) and c["peft_config"].get(peft_key) is not None]
    return []


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _compare(actual: Any, op: str, wanted: str) -> bool:
    a, w = _as_float(actual), _as_float(wanted)
    if a is not None and w is not None:
        close = math.isclose(a, w, rel_tol=1e-9, abs_tol=1e-12)
        return {
            ":": close, "=": close, "!=": not close,
            ">": a > w and not close, ">=": a >= w or close,
            "<": a < w and not close, "<=": a <= w or close,
        }[op]
    sa, sw = str(actual).lower(), wanted.lower()
    if op == ":":
        return sw in sa
    if op == "=":
        return sa == sw
    if op == "!=":
        return sa != sw
    return False  # ordering needs numbers


def search_haystack(job: dict, name: str) -> str:
    bits = [name, job_id_of(job), status_word(job.get("status")), owner(job), job.get("hardware") or ""]
    bits += job_types(job) + job_models(job)
    return " ".join(str(b) for b in bits).lower()


def job_matches_query(job: dict, query: Query, name: str = "") -> bool:
    if query.errors:
        return False  # a typo'd field should surface, not silently match everything
    if query.empty:
        return True
    hay = search_haystack(job, name)
    if any(t not in hay for t in query.terms):
        return False
    for key, op, value in query.filters:
        values = field_values(job, key, name)
        if op == "!=":
            if any(not _compare(v, op, value) for v in values):
                return False
            continue
        if not any(_compare(v, op, value) for v in values):
            return False
    return True


def no_match_message(text: str, query: Query, searched: int) -> str:
    lines = [f"No jobs match “{text}” among {searched} job(s) in the last 6 months."]
    if query.errors:
        lines.append("  " + "; ".join(query.errors) + ".  Fields: " + ", ".join(SEARCH_FIELDS))
    if query.terms:
        lines.append("  Words match name, id, status, owner, GPU type, job type and model (case-insensitive).")
    if query.filters:
        lines.append("  Filters must all match; try fewer, or ':' for a partial match (model:qwen).")
    lines.append("  Esc clears the search.")
    return "\n".join(lines)


# ─── sorting ─────────────────────────────────────────────────────────────────

SORT_KEYS = ("created", "updated", "status")
_STATUS_ORDER = ["running", "initializing", "placing", "pending", "queued", "creating", "failed", "cancelled", "terminated"]


def _epoch(raw: Any) -> float:
    dt = parse_ts(raw)
    return dt.timestamp() if dt else float("-inf")


def sort_job_list(jobs: Iterable[dict], key: str = "created", descending: bool = True) -> list:
    jobs = [j for j in jobs if isinstance(j, dict)]
    if key == "status":
        rank = {s: i for i, s in enumerate(_STATUS_ORDER)}
        by_created = sorted(jobs, key=lambda j: _epoch(j.get("created_at")), reverse=True)
        return sorted(
            by_created,
            key=lambda j: rank.get(status_word(j.get("status")), len(rank)),
            reverse=not descending,
        )
    field_name = "updated_at" if key == "updated" else "created_at"
    return sorted(jobs, key=lambda j: _epoch(j.get(field_name)), reverse=descending)


# ─── GPU resources ───────────────────────────────────────────────────────────

GIB = 1024**3


@dataclass
class GpuSample:
    t: float  # epoch seconds
    util: float  # mean GPU compute utilization across GPUs, %
    mem_pct: float  # framebuffer used / total across GPUs, %
    mem_used: float  # bytes, summed
    mem_total: float  # bytes, summed
    gpus: int


def gpu_series(records: Iterable[dict]) -> list:
    """Aggregate per-GPU telemetry rows (``gpu.jsonl``) into one sample per tick."""
    buckets: dict = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        dt = parse_ts(r.get("sample_ts"))
        if not dt:
            continue
        t = round(dt.timestamp())
        b = buckets.setdefault(t, {"util": [], "used": 0.0, "total": 0.0, "gpus": set()})
        if (u := _as_float(r.get("gpu_util_pct"))) is not None:
            b["util"].append(u)
        used, free = _as_float(r.get("fb_used_bytes")), _as_float(r.get("fb_free_bytes"))
        if used is not None and free is not None:
            b["used"] += used
            b["total"] += used + free
        b["gpus"].add(r.get("gpu_uuid") or r.get("global_gpu_index"))
    out = []
    for t in sorted(buckets):
        b = buckets[t]
        util = sum(b["util"]) / len(b["util"]) if b["util"] else 0.0
        mem_pct = 100.0 * b["used"] / b["total"] if b["total"] else 0.0
        out.append(GpuSample(t, util, mem_pct, b["used"], b["total"], len(b["gpus"])))
    return out


def sample_interval(records: Iterable[dict]) -> Optional[float]:
    for r in records:
        if isinstance(r, dict) and (v := _as_float(r.get("sample_interval_s"))):
            return v
    return None


def series_stats(values: list) -> dict:
    if not values:
        return {"min": None, "avg": None, "max": None, "latest": None}
    return {"min": min(values), "avg": sum(values) / len(values), "max": max(values), "latest": values[-1]}


def window_samples(samples: list, seconds: Optional[float], end: Optional[float] = None) -> list:
    """Samples in the trailing window ending at ``end`` (default: last sample)."""
    if not samples or seconds is None:
        return list(samples)
    end = end if end is not None else samples[-1].t
    return [s for s in samples if s.t >= end - seconds]


def parse_window(text: str) -> Optional[tuple]:
    """``10m`` / ``2h`` / ``1d`` / ``90s`` → ("time", seconds); ``all`` → ("all",);
    ``steps:100-500`` → ("steps", 100, 500). ``None`` when unparseable."""
    s = (text or "").strip().lower()
    if s in ("all", "full", "job"):
        return ("all",)
    m = re.fullmatch(r"steps?\s*[:=]?\s*(\d+)\s*-\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return ("steps", min(lo, hi), max(lo, hi))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd])", s)
    if m:
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
        return ("time", float(m.group(1)) * mult)
    return None


def window_label(win: tuple) -> str:
    if win[0] == "all":
        return "full job"
    if win[0] == "steps":
        return f"steps {win[1]}–{win[2]}"
    return f"last {fmt_duration(win[1]).replace(' 00s', '').replace(' 00m', '')}"


# ─── training metrics ────────────────────────────────────────────────────────

# canonical key -> (label, family, recognized raw names). Order = display order.
KNOWN_METRICS = {
    "train_loss": ("training loss", "SFT · common", ("train_loss", "train_nll", "loss", "avg_loss", "train/loss", "train_mean_nll")),
    "val_loss": ("validation loss", "SFT · common", ("val_loss", "eval_loss", "validation_loss", "val_nll", "test/nll", "test_nll")),
    "learning_rate": ("learning rate", "SFT · common", ("learning_rate", "lr", "last_lr", "optim/lr")),
    "grad_norm": ("gradient norm", "SFT · common", ("grad_norm", "gradient_norm", "optim/grad_norm")),
    "step_time": ("step time (s)", "SFT · common", ("step_time", "time_total", "time/step", "time/total")),
    "tokens_per_s": ("token throughput (tok/s)", "SFT · common", ("tokens_per_s", "tokens_per_second", "throughput", "tok_per_s")),
    "policy_loss": ("policy loss", "RL", ("policy_loss", "pg_loss", "loss/policy")),
    "reward": ("reward", "RL", ("reward", "reward_mean", "env/all/reward/total", "mean_reward")),
    "kl": ("KL divergence", "RL", ("kl", "kl_divergence", "kl_div", "approx_kl", "optim/kl_sample_train_v1")),
    "entropy": ("entropy", "RL", ("entropy", "policy_entropy", "optim/entropy")),
    "clip_frac": ("clip fraction", "RL", ("clip_frac", "clip_fraction", "clipfrac")),
    "completion_len": ("completion length (tok)", "RL", ("completion_len", "completion_length", "response_length", "env/all/ac_tokens_per_turn")),
    "eval_success": ("eval success rate", "RL", ("eval_success", "eval_success_rate", "success_rate", "eval/accuracy", "test/correct")),
}
_RAW_TO_CANON = {raw: canon for canon, (_, _, raws) in KNOWN_METRICS.items() for raw in raws}
_BOOKKEEPING = {"step", "ts", "sub_job_id", "rank", "progress", "global_steps", "sft_model_calls", "sft_packed_rows"}
LOSS_KEYS = ("train_loss", "policy_loss", "val_loss")


@dataclass
class MetricInfo:
    key: str  # raw key in the records
    label: str
    family: str  # SFT · common | RL | other
    count: int
    latest: float
    minimum: float
    min_step: Optional[int]


def canonical_metric(raw: str) -> Optional[str]:
    return _RAW_TO_CANON.get(str(raw).lower())


def metric_catalog(records: Iterable[dict]) -> list:
    """Every numeric metric seen, known metrics first (in canonical order)."""
    stats: dict = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        step = rec.get("step")
        for k, v in rec.items():
            if k in _BOOKKEEPING or (f := _as_float(v)) is None:
                continue
            s = stats.setdefault(k, {"n": 0, "latest": f, "min": f, "min_step": step})
            s["n"] += 1
            s["latest"] = f
            if f < s["min"]:
                s["min"], s["min_step"] = f, step
    order = list(KNOWN_METRICS)

    def rank(k: str) -> tuple:
        canon = canonical_metric(k)
        return (order.index(canon), k) if canon else (len(order), k.lower())

    out = []
    for k in sorted(stats, key=rank):
        canon = canonical_metric(k)
        label, family, _ = KNOWN_METRICS[canon] if canon else (k, "other", ())
        s = stats[k]
        out.append(MetricInfo(k, label, family, s["n"], s["latest"], s["min"], s["min_step"]))
    return out


def primary_loss(catalog: list) -> Optional[MetricInfo]:
    for canon in LOSS_KEYS:
        for info in catalog:
            if canonical_metric(info.key) == canon:
                return info
    return None


def metric_points(records: Iterable[dict], key: str) -> list:
    """(epoch_seconds | None, step | None, value) for one metric, in record order."""
    pts = []
    for rec in records:
        if not isinstance(rec, dict) or (v := _as_float(rec.get(key))) is None:
            continue
        dt = parse_ts(rec.get("ts"))
        step = rec.get("step")
        pts.append((dt.timestamp() if dt else None, int(step) if isinstance(step, (int, float)) else None, v))
    return pts


def window_points(points: list, win: tuple) -> list:
    if win[0] == "all" or not points:
        return list(points)
    if win[0] == "steps":
        return [p for p in points if p[1] is not None and win[1] <= p[1] <= win[2]]
    times = [p[0] for p in points if p[0] is not None]
    if not times:
        return list(points)
    end = max(times)
    return [p for p in points if p[0] is not None and p[0] >= end - win[1]]


def fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v:.4g}"


# ─── checkpoints ─────────────────────────────────────────────────────────────

CHECKPOINT_SORTS = ("time", "step", "type")
_CP_KNOWN = {"checkpoint_id", "checkpoint_type", "created_at", "global_steps", "global_step", "step"}


def checkpoint_kind(cp: dict) -> str:
    t = str(cp.get("checkpoint_type") or "").lower().replace("_", "-")
    return {"weights-only": "weights-only", "weightsonly": "weights-only", "resumable": "resumable"}.get(t, t or "unknown")


def checkpoint_step(cp: dict) -> Optional[int]:
    for k in ("global_steps", "global_step", "step"):
        if (v := _as_float(cp.get(k))) is not None:
            return int(v)
    m = re.search(r"global_step(\d+)", str(cp.get("checkpoint_id") or "") + str(cp.get("checkpoint_tag") or ""))
    return int(m.group(1)) if m else None


def checkpoint_metric_keys(cps: Iterable[dict]) -> list:
    """Numeric quality metrics carried on checkpoint records (loss first)."""
    keys: dict = {}
    for cp in cps:
        for k, v in cp.items():
            if k not in _CP_KNOWN and _as_float(v) is not None and k not in ("version",):
                keys[k] = True
    return sorted(keys, key=lambda k: (0 if "loss" in k else 1, k))


def sort_checkpoints(cps: Iterable[dict], key: str = "time", descending: bool = True) -> list:
    cps = [c for c in cps if isinstance(c, dict)]
    if key == "step":
        return sorted(cps, key=lambda c: (checkpoint_step(c) if checkpoint_step(c) is not None else -1), reverse=descending)
    if key == "type":
        by_time = sorted(cps, key=lambda c: _epoch(c.get("created_at")), reverse=True)
        return sorted(by_time, key=lambda c: checkpoint_kind(c), reverse=not descending)
    return sorted(cps, key=lambda c: _epoch(c.get("created_at")), reverse=descending)


def _training_sub_job(job: dict) -> Optional[dict]:
    for sj in sub_jobs(job):
        if str(sj.get("job_type") or "").lower() == "training":
            return sj
    return None


def checkpoint_details(cp: dict, job: dict, name: str) -> list:
    """(section title, lines) for the expanded checkpoint view."""
    jid = job_id_of(job)
    cid = str(cp.get("checkpoint_id") or "CHECKPOINT_ID")
    kind = checkpoint_kind(cp)
    train = _training_sub_job(job) or {}
    fields = [f"{k}: {_num(v) if isinstance(v, float) else v}" for k, v in sorted(cp.items())]
    step = checkpoint_step(cp)
    if step is not None and "global_steps" not in cp:
        fields.append(f"global_step (from tag): {step}")
    origin = [
        f"job id:      {jid}",
        f"job name:    {name or '—'}",
        f"sub-job:     {train.get('sub_job_id') or '—'}  (parent job {jid})",
        f"model:       {train.get('model_name') or ', '.join(job_models(job)) or '—'}",
        f"job status:  {status_word(job.get('status')).upper()}",
    ]
    cfg = {k: v for k, v in train.items() if k not in ("status", "sub_job_id")}
    config_lines = json.dumps(cfg, indent=2, sort_keys=True).splitlines() if cfg else ["(training sub-job config unavailable)"]
    fetch = [
        "List this job's checkpoints:",
        f"  cortex-training checkpoints {jid}",
        "Export short-lived download URLs for every file (Python SDK):",
        f'  files = client.export_checkpoint("{jid}", "{cid}")["files"]',
        "  for f in files:  # each has filename, url, size_bytes",
        '      urllib.request.urlretrieve(f["url"], f["filename"])',
        "URLs expire (see expires_at); export again when they do.",
    ]
    if kind == "weights-only":
        deploy = [
            "Batch inference (spins up a sampling job, generates, cancels):",
            "  python -m recipes.inference.generate config=CONFIG.json \\",
            f"    job_config=JOB_CONFIG source_job_id={jid} \\",
            f'    checkpoint_id={cid} prompt="..."',
            "Streaming endpoint (stays up until you cancel it):",
            "  python -m recipes.inference.serve config=CONFIG.json \\",
            f"    job_config=JOB_CONFIG source_job_id={jid} checkpoint_id={cid}",
            "  then client.generate_stream(ENDPOINT_JOB_ID, prompts=[...]); cortex-training cancel ENDPOINT_JOB_ID",
            "Or set on any sampling sub-job:",
            f'  "source_checkpoint_info": {{"checkpoint_id": "{cid}", "source_job_id": "{jid}"}}',
        ]
    else:
        deploy = [
            "Resumable checkpoints hold weights + optimizer state and cannot serve inference directly.",
            "Resume training by setting on the new job's training sub-job:",
            f'  "source_checkpoint_info": {{"checkpoint_id": "{cid}", "source_job_id": "{jid}"}}',
            "Add training_config.load_optimizer_states=false for weights only (needed when DP size changes).",
            "For inference, save a weights-only checkpoint from a job resumed from this one.",
        ]
    return [
        ("Checkpoint", fields),
        ("Originating job", origin),
        ("Training configuration when saved", config_lines),
        ("Fetch / export the weights", fetch),
        ("Deploy for inference" if kind == "weights-only" else "Resume or deploy", deploy),
    ]


# ─── plain-text charts ───────────────────────────────────────────────────────

_BLOCKS = " ▁▂▃▄▅▆▇█"


def render_chart(
    xs: list,
    ys: list,
    *,
    width: int,
    height: int,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    y_fmt=fmt_metric,
    x_fmt=str,
) -> list:
    """Filled column chart with a y-axis and first/last x labels.

    ``xs`` must be ascending numbers. Columns are equal x-buckets; a bucket's
    value is the mean of its points, and empty buckets stay blank.
    """
    gutter = 9
    plot_w = max(4, width - gutter - 1)
    height = max(2, height)
    if not ys:
        return ["(no data in this window)"]
    lo = min(ys) if y_min is None else y_min
    hi = max(ys) if y_max is None else y_max
    if hi <= lo:
        hi, lo = lo + (abs(lo) * 0.1 or 1.0), lo - (abs(lo) * 0.1 if y_min is None else 0)
    x0, x1 = xs[0], xs[-1]
    span = (x1 - x0) or 1.0
    sums = [0.0] * plot_w
    counts = [0] * plot_w
    for x, y in zip(xs, ys):
        col = min(plot_w - 1, int((x - x0) / span * plot_w))
        sums[col] += y
        counts[col] += 1
    cols = [sums[i] / counts[i] if counts[i] else None for i in range(plot_w)]
    # Sparse series (fewer points than columns) leave regular holes; carry the
    # previous value across short gaps so a steady signal reads as steady.
    max_gap = max(1, int(plot_w / max(1, len(ys)) * 2) + 1)
    last, gap = None, 0
    for i, c in enumerate(cols):
        if c is not None:
            last, gap = c, 0
        elif last is not None:
            gap += 1
            if gap <= max_gap:
                cols[i] = last
    units = [None if v is None else max(0.0, min(1.0, (v - lo) / (hi - lo))) * height * 8 for v in cols]
    rows = []
    for r in range(height):
        level = height - 1 - r  # 0 = bottom row
        line = []
        for u in units:
            if u is None:
                line.append(" ")
                continue
            fill = u - level * 8
            line.append("█" if fill >= 8 else _BLOCKS[max(0, int(round(fill)))] if fill > 0 else " ")
        if r == 0:
            label = y_fmt(hi)
        elif r == height - 1:
            label = y_fmt(lo)
        elif r == height // 2:
            label = y_fmt((hi + lo) / 2)
        else:
            label = ""
        rows.append(f"{label:>{gutter - 1}} ┤" + "".join(line))
    rows.append(" " * (gutter - 1) + " └" + "─" * plot_w)
    left, right = x_fmt(x0), x_fmt(x1)
    rows.append(" " * (gutter + 1) + left + right.rjust(max(1, plot_w - len(left))))
    return rows


def fmt_clock(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M")


def fmt_pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.0f}%"
