# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry rows in the shapes the service writes, for the Resources data-layer tests.

GPU rows carry no ``kind`` and record version 2; system rows carry
``"kind": "system"`` and record version 3, one per node per scrape. Both come in
the field order the service writes them, and use placeholder ids only.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

GIB = 1024**3
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SUB = "JOB:training:0"


def stamp(seconds: float) -> str:
    """``sample_ts`` ``seconds`` after START, with nanosecond digits as the service writes them."""
    moment = START + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond:06d}123Z"


def gpu_row(seconds: float, gpu: int, util: float | None = 50.0, *, used: float = 60 * GIB, free: float = 80 * GIB,
            per_node: int = 4, interval: float = 5.0, sub: str = SUB, **extra) -> dict:
    """One GPU at one scrape, in the service's field order (``extra`` adds or replaces fields)."""
    row = {
        "record_version": 2,
        "sub_job_id": sub,
        "sample_ts": stamp(seconds),
        "worker_num": gpu // per_node,
        "local_gpu_index": gpu % per_node,
        "global_gpu_index": gpu,
        "gpu_uuid": f"GPU-placeholder-{gpu}",
        "gpu_util_pct": util,
        "fb_used_bytes": float(used),
        "fb_free_bytes": float(free),
        "power_watts": 100.0 + 3 * (util or 0.0),
        "gpu_temp_c": 30.0 + (util or 0.0) / 4,
        "sm_active_pct": None,
        "sm_occupancy_pct": None,
        "tensor_active_pct": None if util is None else util / 2,
        "dram_active_pct": None if util is None else util / 4,
        "seq": int(seconds // 5),
        "seq_epoch": "0" * 32,
        "sample_interval_s": interval,
    }
    row.update(extra)
    return row


def system_row(seconds: float, node: int, cpu: float = 4.0, *, used: float = 100 * GIB, total: float = 860 * GIB,
               cpus: float = 88.0, interval: float = 5.0, sub: str = SUB, **extra) -> dict:
    """One node's CPU and memory at one scrape, in the service's field order."""
    row = {
        "kind": "system",
        "record_version": 3,
        "sub_job_id": sub,
        "sample_ts": stamp(seconds),
        "worker_num": node,
        "seq": int(seconds // 5),
        "seq_epoch": "0" * 32,
        "sample_interval_s": interval,
        "cpu_util_pct": cpu,
        "cpu_count": cpus,
        "mem_used_bytes": float(used),
        "mem_total_bytes": float(total),
    }
    row.update(extra)
    return row


def lines(rows) -> bytes:
    return "".join((row if isinstance(row, str) else json.dumps(row)) + "\n" for row in rows).encode()


def manifest(path: Path, names=("c0",), size: int | None = None) -> None:
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": list(names), "bytes": path.stat().st_size if size is None else size}))


def write(root: Path, rows, *, sub: str = SUB, names=("c0",)) -> Path:
    """The cache file download_metrics leaves for ``sub``, with its manifest."""
    path = root / sub / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lines(rows))
    manifest(path, names)
    return path


def append(path: Path, rows, names) -> int:
    """Rows appended as download_metrics appends a chunk: the same file, a longer chunk list."""
    payload = lines(rows)
    with path.open("ab") as handle:
        handle.write(payload)
    manifest(path, names)
    return len(payload)


def scrapes(seconds: float, step: float = 5.0, start: float = 0.0):
    """The scrape times from ``start`` for ``seconds`` (inclusive of the first, exclusive of the end)."""
    t = start
    while t < start + seconds:
        yield t
        t += step
