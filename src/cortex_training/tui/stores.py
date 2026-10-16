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

"""Client-side stores for job display names and client-logged training metrics.

A job's server-side name is its ``comment``: set at creation (``create_job``,
``submit --comment``, the recipes), at most 1,024 bytes, and never editable.
Local aliases name a job on this machine, override the comment in the TUI, and
can name jobs created without one. The service also does not retain the
per-step metrics a recipe computes. Both live next to the log cache on the
client:

    <cache_root>/aliases.json                          job_id -> display name
    <cache_root>/jobs/<safe(job_id)>/metrics.jsonl      {"ts", "step", ...metrics}

Pure stdlib, NO ``textual`` import, so recipes can write metrics without the
TUI and the stores are unit-testable like ``format.py``.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Optional

from cortex_training.tui.log_cache import _atomic_write_text
from cortex_training.tui.log_cache import cache_root
from cortex_training.tui.log_cache import safe_source_name

# A display name keeps at most this many characters (the rename prompt takes no more).
ALIAS_MAX_CHARS = 80
_MAX_ALIAS = ALIAS_MAX_CHARS


class AliasStore:
    """Persistent ``job_id -> display name`` map, shared by every TUI session.

    An alias wins over the job's server ``comment`` as its display name, on
    this machine only; the comment stays visible and searchable beside it.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        base = root if root is not None else cache_root()
        self._path = base / "aliases.json"

    def _read(self) -> dict:
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        names = doc.get("aliases") if isinstance(doc, dict) else None
        return {str(k): str(v) for k, v in names.items() if v} if isinstance(names, dict) else {}

    def all(self) -> dict:
        return self._read()

    def get(self, job_id: str) -> Optional[str]:
        return self._read().get(job_id)

    def set(self, job_id: str, name: Optional[str]) -> None:
        """Set (or, with an empty name, remove) a job's display name."""
        names = self._read()
        clean = " ".join(str(name or "").split())[:_MAX_ALIAS]
        if clean:
            names[job_id] = clean
        else:
            names.pop(job_id, None)
        _atomic_write_text(self._path, json.dumps({"version": 1, "aliases": names}, indent=1, sort_keys=True) + "\n")


def _finite(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


class MetricsStore:
    """Append-only per-job training metrics written by the client."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._base = root if root is not None else cache_root()

    def path(self, job_id: str) -> Path:
        return self._base / "jobs" / safe_source_name(job_id) / "metrics.jsonl"

    def append(
        self,
        job_id: str,
        metrics: dict,
        *,
        step: Optional[int] = None,
        sub_job_id: Optional[str] = None,
        ts: Optional[str] = None,
    ) -> None:
        """Record the numeric members of ``metrics``; non-numeric values are dropped."""
        numeric = {str(k): f for k, v in (metrics or {}).items() if (f := _finite(v)) is not None}
        if not numeric:
            return
        record: dict = {"ts": ts or datetime.now(timezone.utc).isoformat()}
        if (finite_step := _finite(step)) is not None:
            record["step"] = int(finite_step)
        if sub_job_id:
            record["sub_job_id"] = sub_job_id
        record.update(numeric)
        path = self.path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def load(self, job_id: str) -> list:
        """The job's records, oldest first. A torn or corrupt line is skipped; bytes
        that aren't UTF-8 (the file was damaged outside the client) read as U+FFFD."""
        path = self.path(job_id)
        out = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # torn last line
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError:
            return []
        return out


class LocalMetricsLogger:
    """``log_metrics(metrics, step)`` sink that feeds the TUI's Metrics tab.

    Shaped like the recipe loggers so it composes with them; write failures are
    swallowed because metrics display must never break a training run.
    """

    def __init__(self, job_id: str, *, store: Optional[MetricsStore] = None) -> None:
        self._job_id = job_id
        self._store = store or MetricsStore()

    def log_metrics(self, metrics: dict, step: int = 0, **kwargs: Any) -> None:
        try:
            self._store.append(self._job_id, metrics, step=step)
        except Exception:  # noqa: BLE001 - odd input or a full disk must not stop training
            pass

    def close(self) -> None:
        pass
