# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPU telemetry for the TUI's Resources tab: incremental parsing, views and charts.

``CortexTrainingClient.download_metrics`` keeps one ``<sub_job_id>/gpu.jsonl``
per sub-job in the TUI cache. It appends new chunks to that file and records the
committed length (and the chunk names it holds) in a manifest beside it, or
rebuilds the file when the chunk list changed. :class:`ResourceLoader`
remembers how far it read each file and keeps running aggregates, so a refresh
reads only the bytes appended since the previous load. A file that was rebuilt
(another inode, a chunk list that no longer extends the one it read, or changed
bytes) is read again from the start. Each load returns an immutable
:class:`ResourceData` snapshot that the UI thread can render while the next
load runs.

Compute and memory are kept apart. ``gpu_util_pct`` is the share of time a
kernel was running on the GPU ("GPU busy", not how saturated it was); memory is
``fb_used / (fb_used + fb_free)`` per GPU.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from array import array
from bisect import bisect_left
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from cortex_training.tui import views

RANGES: dict[str, timedelta | None] = {
    "10m": timedelta(minutes=10),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "all": None,
}
# Every job opens on this window; the stats beside it always cover the whole job too.
DEFAULT_RANGE = "10m"
_MAX_BUCKETS = 240
_MAX_SAMPLE_INTERVAL_S = 24 * 60 * 60
_MAX_FB_BYTES = float(2**63)
_MIN_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()
_MAX_EPOCH = datetime(9998, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc).timestamp()
# RFC 3339 with a "T" and an explicit zone; the fraction may carry more digits than microseconds.
_TS_PARTS_RE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)")
# Raw per-GPU samples are kept for the longest finite window plus slack for rows
# that arrive late (workers upload their chunks independently).
_RAW_RETENTION_S = 7 * 60 * 60.0
# Past the retention the average across GPUs at each sampling interval is folded
# into the whole-job bins' extremes (see _Series.trim). While a file is read in
# order, only samples this far behind the newest row read are folded: rows from
# different workers interleave by up to a chunk. The fold keeps this much before
# its cutoff as the next fold's context.
_FOLD_MARGIN_S = 20 * 60.0
_FOLD_CONTEXT_S = 10 * 60.0
# The whole-job series past that retention: per-GPU bins whose width doubles
# whenever the job outgrows _MAX_BINS of them.
_BIN_START_S = 60.0
_MAX_BINS = 480
_READ_BLOCK = 4 << 20
_FINGERPRINT = 64
_PEEK_BYTES = 64 * 1024
# First load of the selected sub-job: read its newest bytes first, starting with
# this much and doubling until the window on screen is covered, and publish that
# before reading the rest. Smaller files are read in one go.
_NEWEST_FIRST_STEP = 1 << 20
_NEWEST_FIRST_MIN_BYTES = 4 << 20
# Rows from different workers interleave in the file by up to a chunk. The file
# holds the chunks in name order, and each worker uploads one about every 30 s,
# so reading this far past a window's left edge reaches every worker's rows in
# it. A worker whose chunks span longer can still have rows in the window that a
# partial read hasn't reached; the full read that follows moments later has them.
_NEWEST_FIRST_SLACK_S = 120.0
# A GPU counts as reporting while its newest sample is this recent (or three
# sampling intervals, if longer) relative to the sub-job's newest sample. Workers
# upload a chunk about every 30 s, and an upload can take about 75 s to land, so
# one worker's newest rows can trail another's by more than a minute.
_REPORTING_TOLERANCE_S = 120.0
_INTERVAL_DELTAS = 64
# Each GPU's newest samples looked at for how often its values change (about 21 minutes at 5 s).
_VALUE_RUN_SAMPLES = 256
# Min and max come from the average across GPUs at each sampling interval; a
# finer interval than this many buckets over the raw retention is coarsened.
_MAX_NATIVE_BUCKETS = 20_000
# A row stamped further ahead of this computer's clock than this is unsupported:
# one clock-skewed row would otherwise become the newest sample, which anchors
# the windows, the freshness and the reporting GPUs.
_MAX_AHEAD_S = 10 * 60.0
# A row without a usable sample_interval_s counts for the time since its GPU's
# previous row, up to this long; a longer gap is a pause, not an interval.
_FALLBACK_INTERVAL_MAX_S = 5 * 60.0
_NAN = float("nan")
_INF = math.inf
_OK, _SKIP, _IGNORED = 0, 1, 2
_MISSING = object()
_SECOND_CACHE: dict[str, float | None] = {}


@dataclass(frozen=True)
class SeriesStats:
    """Min, average, max and latest of one series over one scope.

    ``min`` and ``max`` are the lowest and highest averages across GPUs at one
    sampling interval, so a scope's extremes always include those of the windows
    inside it. ``avg`` weights GPU-time: every sample counts for its sampling
    interval, so a GPU that reported for part of the scope counts for that part
    (for busy, the share of the GPUs' time a kernel was running). ``latest`` is
    the average across the GPUs in the newest capture. ``None`` means no data.

    ``min_bound`` / ``max_bound``: part of the scope is known only as averages
    over the whole-job bins (rows that arrived after their time had left the
    raw samples), and a sample there lies far enough below ``min`` (above
    ``max``) that the true min may be lower (the true max higher) than shown,
    at the precision shown. History that left the raw samples in order keeps
    exact extremes (see :meth:`_Series.trim`).
    """

    min: float | None = None
    avg: float | None = None
    max: float | None = None
    latest: float | None = None
    min_bound: bool = False
    max_bound: bool = False


@dataclass(frozen=True)
class GPURow:
    global_gpu_index: int
    current_util_pct: float | None
    window_avg_util_pct: float | None
    fb_used_bytes: float | None
    fb_total_bytes: float | None
    current_mem_pct: float | None = None
    window_avg_mem_pct: float | None = None
    # DCGM profiling gauges from the GPU's newest row; often absent (None).
    sm_active_pct: float | None = None
    tensor_active_pct: float | None = None
    dram_active_pct: float | None = None
    # False once the GPU has dropped out of the newest capture.
    reporting: bool = True


@dataclass(frozen=True)
class RangeView:
    name: str
    start: datetime
    end: datetime
    chart: tuple[float | None, ...]
    rows: tuple[GPURow, ...]
    mem_chart: tuple[float | None, ...] = ()
    busy: SeriesStats = SeriesStats()
    memory: SeriesStats = SeriesStats()
    # GPUs with a sample in the range: the charts and stats average across them.
    gpus: int = 0
    # Each GPU's own chart values, aligned with ``chart``: gpu id -> (busy, memory).
    gpu_series: dict = field(default_factory=dict, hash=False)
    # "GPU 3" on a view narrowed to one GPU of a sub-job with more (see gpu_view).
    gpu_label: str = ""


@dataclass(frozen=True)
class SubJobResource:
    sub_job_id: str
    latest_sample_ts: datetime
    views: dict[str, RangeView]
    first_sample_ts: datetime | None = None
    # Spacing of consecutive samples, derived from the data (None if unknown).
    sample_interval_s: float | None = None
    # The interval the rows report in ``sample_interval_s`` (most common value).
    reported_interval_s: float | None = None
    gpu_ids: tuple[int, ...] = ()
    reporting_gpu_ids: tuple[int, ...] = ()
    # Older history is still being read (a first load shows the newest data first).
    partial: bool = False
    # How often the busy values change (DCGM's refresh), derived from the data (None if unknown).
    value_interval_s: float | None = None


@dataclass(frozen=True)
class ResourceData:
    sub_jobs: dict[str, SubJobResource]
    ignored_rows_by_sub_job: dict[str, int] = field(default_factory=dict)

    @property
    def ignored_rows(self) -> int:
        return sum(self.ignored_rows_by_sub_job.values())


def _finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_number(value, lower: float, upper: float) -> float | None:
    result = _finite_number(value)
    return result if result is not None and lower <= result <= upper else None


def _gauge(value, upper: float) -> float | None:
    """``_bounded_number(value, 0, upper)`` with a fast path for JSON floats."""
    if type(value) is float:
        return value if 0.0 <= value <= upper else None  # NaN and infinities fail
    if value is None:
        return None
    return _bounded_number(value, 0, upper)


def _sample_epoch(value: object) -> float | None:
    """``sample_ts`` (RFC 3339 with a zone) as epoch seconds, or None.

    Fractions finer than a microsecond are truncated, and times outside
    1970-01-01 to 9998-12-31 are rejected. This is the hot path of a load, so
    the whole seconds of each timestamp are cached: every GPU in a capture
    shares them.
    """
    if not isinstance(value, str):
        return None
    match = _TS_PARTS_RE.fullmatch(value)
    if match is None:
        return None
    base, fraction, zone = match.groups()
    key = base + zone
    seconds = _SECOND_CACHE.get(key, _MISSING)
    if seconds is _MISSING:
        try:
            seconds = datetime.fromisoformat(base + ("+00:00" if zone == "Z" else zone)).timestamp()
        except (OverflowError, ValueError, OSError):
            seconds = None
        if len(_SECOND_CACHE) > 8192:
            _SECOND_CACHE.clear()
        _SECOND_CACHE[key] = seconds
    if seconds is None:
        return None
    if fraction:
        seconds += int(fraction[:6].ljust(6, "0")) / 1_000_000
    return seconds if _MIN_EPOCH <= seconds <= _MAX_EPOCH else None


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _read_manifest(path: Path) -> tuple[list[str], int, bool] | None:
    """(chunk names, committed bytes, partial) from the manifest beside ``path``, or None.

    ``partial``: the download committed only the sub-job's newest chunks so far,
    and its older history is still to come.
    """
    manifest = path.with_name("." + path.name + ".manifest.json")
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        size = path.stat().st_size
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    names = doc.get("names")
    committed = doc.get("bytes")
    partial = doc.get("partial", False)
    if (
        doc.get("version") != 1
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or isinstance(committed, bool)
        or not isinstance(committed, int)
        or committed < 0
        or size < committed
        or not isinstance(partial, bool)
    ):
        return None
    return names, committed, partial


def _manifest_bytes(path: Path) -> int | None:
    manifest = _read_manifest(path)
    return None if manifest is None else manifest[1]


def cached_metric_sources(metrics_root: Path) -> list[str]:
    try:
        paths = sorted(metrics_root.glob("*/gpu.jsonl"))
    except OSError:
        return []
    return [path.parent.name for path in paths if _manifest_bytes(path) is not None]


# ─── per-GPU and per-sub-job state ───────────────────────────────────────────


class _Gpu:
    """One GPU: its samples inside the raw retention, its newest row, running totals."""

    __slots__ = ("ts", "util", "mem", "weight", "unsorted", "latest", "busy_wsum", "busy_w", "mem_wsum", "mem_w",
                 "previous")

    def __init__(self) -> None:
        # Columns of samples; NaN marks an unmeasured value or a missing interval.
        self.ts = array("d")
        self.util = array("d")
        self.mem = array("d")
        self.weight = array("d")
        self.unsorted = False
        # (ts, util, mem %, fb used, fb total, the row) of the newest row.
        self.latest: tuple | None = None
        self.busy_wsum = 0.0
        self.busy_w = 0.0
        self.mem_wsum = 0.0
        self.mem_w = 0.0
        # The time of the row read just before (in file order), for rows that carry no interval.
        self.previous = _NAN

    def sort(self) -> None:
        if not self.unsorted:
            return
        order = sorted(range(len(self.ts)), key=self.ts.__getitem__)
        for name in ("ts", "util", "mem", "weight"):
            column = getattr(self, name)
            setattr(self, name, array("d", (column[i] for i in order)))
        self.unsorted = False

    def drop_before(self, cutoff: float) -> None:
        self.sort()
        index = bisect_left(self.ts, cutoff)
        if index:
            for column in (self.ts, self.util, self.mem, self.weight):
                del column[:index]


class _Series:
    """Everything read so far from one sub-job's ``gpu.jsonl``."""

    def __init__(self, sub_job_id: str) -> None:
        self.sub_job_id = sub_job_id
        self.gpus: dict[int, _Gpu] = {}
        # Any accepted row, measured or not: freshness and the reporting GPUs.
        self.first: float | None = None
        self.latest: float | None = None
        # Rows with a value to chart (an interval and a busy or memory reading).
        # The charts and the whole-job span cover these, so a row that carries
        # nothing to plot (a malformed or clock-skewed one) can't stretch them.
        self.data_first: float | None = None
        self.data_last: float | None = None
        self.min_interval: float | None = None
        self.reported: dict[float, int] = {}
        # Raw samples are complete from this time on (older ones live only in bins).
        self.raw_start = -math.inf
        self.bin_width = _BIN_START_S
        # bin index -> GPU -> [busy sum, busy weight, memory sum, memory weight,
        # lowest busy, highest busy, lowest memory, highest memory]
        self.bins: dict[int, dict[int, list[float]]] = {}
        # bin index -> [lowest busy, highest busy, lowest memory, highest memory]
        # average across GPUs at one sampling interval, folded in from the raw
        # samples before they leave the retention (see trim()).
        self.extremes: dict[int, list[float]] = {}
        # Bins holding rows that arrived after their time had been folded: those
        # rows never reached the raw samples, so their bins' extremes are bounds.
        self.inexact: set[int] = set()
        # The newest sample time of the read in progress (see ResourceLoader._read).
        self.pass_newest = -_INF
        self.ignored = 0
        self.version = 0
        # Rows stamped after this (epoch seconds) are unsupported; each load sets it.
        self.horizon = _INF

    @property
    def chart_start(self) -> float | None:
        return self.data_first if self.data_first is not None else self.first

    def ingest(self, doc: object) -> int:
        if type(doc) is not dict:
            return _IGNORED
        kind = doc.get("kind")
        if kind is not None and kind != "gpu":
            return _SKIP
        if doc.get("record_version") != 2 or doc.get("sub_job_id") != self.sub_job_id:
            return _IGNORED
        index = doc.get("global_gpu_index")
        if type(index) is not int or index < 0:
            return _IGNORED
        ts = _sample_epoch(doc.get("sample_ts"))
        if ts is None or ts > self.horizon:
            return _IGNORED
        interval = _gauge(doc.get("sample_interval_s"), _MAX_SAMPLE_INTERVAL_S) or None
        util = _gauge(doc.get("gpu_util_pct"), 100.0)
        used = _gauge(doc.get("fb_used_bytes"), _MAX_FB_BYTES)
        free = _gauge(doc.get("fb_free_bytes"), _MAX_FB_BYTES)
        total = used + free if used is not None and free is not None else None
        mem = 100.0 * used / total if total else None

        gpu = self.gpus.get(index)
        if gpu is None:
            gpu = self.gpus[index] = _Gpu()
        if interval is None:
            # No usable sample_interval_s (absent, 0 or malformed): the time since this
            # GPU's row before it, when that looks like a sampling interval.
            gap = ts - gpu.previous
            if 0 < gap <= _FALLBACK_INTERVAL_MAX_S:
                interval = gap
        else:
            if interval in self.reported or len(self.reported) < 64:
                self.reported[interval] = self.reported.get(interval, 0) + 1
        gpu.previous = ts
        if ts > self.pass_newest:
            self.pass_newest = ts
        if self.first is None or ts < self.first:
            self.first = ts
        if self.latest is None or ts > self.latest:
            self.latest = ts
        if gpu.latest is None or ts >= gpu.latest[0]:
            gpu.latest = (ts, util, mem, used, total, doc)
        if interval is not None:
            if self.min_interval is None or interval < self.min_interval:
                self.min_interval = interval
            if util is not None or mem is not None:
                if self.data_first is None or ts < self.data_first:
                    self.data_first = ts
                if self.data_last is None or ts > self.data_last:
                    self.data_last = ts
                key = int(ts // self.bin_width)
                cells = self.bins.get(key)
                if cells is None:
                    cells = self.bins[key] = {}
                cell = cells.get(index)
                if cell is None:
                    cell = cells[index] = [0.0, 0.0, 0.0, 0.0, _INF, -_INF, _INF, -_INF]
                if util is not None:
                    gpu.busy_wsum += util * interval
                    gpu.busy_w += interval
                    cell[0] += util * interval
                    cell[1] += interval
                    if util < cell[4]:
                        cell[4] = util
                    if util > cell[5]:
                        cell[5] = util
                if mem is not None:
                    gpu.mem_wsum += mem * interval
                    gpu.mem_w += interval
                    cell[2] += mem * interval
                    cell[3] += interval
                    if mem < cell[6]:
                        cell[6] = mem
                    if mem > cell[7]:
                        cell[7] = mem
                if ts < self.raw_start:
                    self.inexact.add(key)  # its interval was folded without it
        if ts >= self.raw_start:
            if gpu.ts and ts < gpu.ts[-1]:
                gpu.unsorted = True
            gpu.ts.append(ts)
            gpu.util.append(_NAN if util is None else util)
            gpu.mem.append(_NAN if mem is None else mem)
            gpu.weight.append(_NAN if interval is None else interval)
        return _OK

    def rebin(self) -> None:
        """Double the bin width until the whole job fits in _MAX_BINS bins."""
        if not self.bins:
            return
        low, high = min(self.bins), max(self.bins)
        while high - low + 1 > _MAX_BINS:
            self.bin_width *= 2
            merged: dict[int, dict[int, list[float]]] = {}
            for key, cells in self.bins.items():
                target = merged.setdefault(key // 2, {})
                for gpu_id, cell in cells.items():
                    existing = target.get(gpu_id)
                    if existing is None:
                        target[gpu_id] = cell
                    else:
                        for i in range(4):
                            existing[i] += cell[i]
                        for i in (4, 6):
                            existing[i] = min(existing[i], cell[i])
                            existing[i + 1] = max(existing[i + 1], cell[i + 1])
            self.bins = merged
            # Extremes merge as extremes: a wider bin never smooths them.
            extremes: dict[int, list[float]] = {}
            for key, values in self.extremes.items():
                existing = extremes.get(key // 2)
                if existing is None:
                    extremes[key // 2] = values
                else:
                    for i in (0, 2):
                        existing[i] = min(existing[i], values[i])
                        existing[i + 1] = max(existing[i + 1], values[i + 1])
            self.extremes = extremes
            self.inexact = {key // 2 for key in self.inexact}
            low, high = low // 2, high // 2

    def trim(self, upto: float | None = None) -> None:
        """Keep raw samples only for the retention before the newest one (or before ``upto``).

        Before samples leave, the average across GPUs at each sampling interval
        among them is folded into their bins' extremes, so the whole job's min and
        max stay exact once its history is older than the retention. A short
        stretch before the cutoff stays as the next fold's context: the value a
        GPU last had carries into the intervals just after it.
        """
        if self.latest is None:
            return
        cutoff = self.latest - _RAW_RETENTION_S
        if upto is not None:
            cutoff = min(cutoff, upto)
        if cutoff <= self.raw_start:
            return
        self._fold(cutoff)
        for gpu in self.gpus.values():
            gpu.drop_before(cutoff - _FOLD_CONTEXT_S)
        self.raw_start = cutoff

    def _fold(self, right: float) -> None:
        """Fold the per-interval averages across GPUs from ``raw_start`` to ``right`` into ``extremes``."""
        start = self.chart_start
        if self.min_interval is None or start is None:
            return
        left = max(self.raw_start, start)
        if right < left:
            return
        # One sampling interval per bucket, coarsened as _Native's are when that would
        # be more than _MAX_NATIVE_BUCKETS (a malformed row can report a tiny interval).
        width = max(self.min_interval, (right - left) / (_MAX_NATIVE_BUCKETS - 1))
        count = math.floor((right - left) / width + 1e-9) + 1
        hold_s = max(_REPORTING_TOLERANCE_S, 3 * self.min_interval)
        busy, memory, *_ = _accumulate(self, left, right, width, count, hold_s, until=right)
        last = count - 1
        for k in range(count):
            b, m = busy[k], memory[k]
            if b is None and m is None:
                continue
            key = int((right - (last - k) * width) // self.bin_width)
            values = self.extremes.get(key)
            if values is None:
                values = self.extremes[key] = [_INF, -_INF, _INF, -_INF]
            if b is not None:
                values[0] = min(values[0], b)
                values[1] = max(values[1], b)
            if m is not None:
                values[2] = min(values[2], m)
                values[3] = max(values[3], m)

    def plan(self, data_first: float | None, data_last: float | None) -> None:
        """Before a first read: size the bins by rebin()'s own rule.

        A first read then ends at the width that loading the same rows a chunk at
        a time would reach.
        """
        if data_first is not None and data_last is not None:
            while int(data_last // self.bin_width) - int(data_first // self.bin_width) + 1 > _MAX_BINS:
                self.bin_width *= 2


# ─── views ───────────────────────────────────────────────────────────────────


def _grid(left: float, latest: float, min_interval: float | None) -> tuple[float, int]:
    """(bucket width, bucket count) for a chart from ``left`` to ``latest``."""
    span = max(0.0, latest - left)
    if span <= 0:
        return 1.0, 1
    interval = min_interval or span
    width = max(span / (_MAX_BUCKETS - 1), interval)
    count = min(_MAX_BUCKETS, math.floor(span / width + 1e-9) + 1)
    return width, max(1, count)


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _accumulate(series: _Series, left: float, latest: float, width: float, count: int, hold_s: float,
                until: float | None = None, per_gpu: dict | None = None):
    """Bucket the raw samples from ``left`` to ``latest`` on ``count`` buckets ``width`` wide.

    ``latest`` is the newest sample unless ``until`` is given: then samples after
    ``until`` (which must be ``latest``) are left out.

    Returns ``(busy, memory, busy_avg, mem_avg, pooled)``: the average across
    GPUs in each bucket (None where no GPU has a sample), each GPU's
    time-weighted mean over the range, and the (busy, memory) mean over every
    GPU's samples in the range, each weighted by its interval. Each bucket
    averages its GPUs' means, so every GPU weighs the same there.

    Buckets are anchored on ``latest``, so sub-interval scrape jitter doesn't
    turn a regular series into alternating double and empty buckets. Workers
    scrape and upload on their own clocks, so a bucket can still miss a GPU: its
    scrape landed in the next bucket, or its worker's newest chunk hasn't been
    uploaded yet. In a bucket where other GPUs have samples, such a GPU counts
    with its previous value while that is at most ``hold_s`` old (the time a
    GPU still counts as reporting), so no bucket is one worker's average. With
    nothing before it, a GPU's first value also counts in the bucket just
    before its first sample, where another worker's first scrape may have
    landed. A bucket with no samples at all stays empty.

    ``per_gpu``, when given, receives each GPU's own bucket values: gpu id ->
    (busy, memory) lists, None where the GPU has no sample (held values fill a
    bucket it missed, as in the averages).

    The first bucket is centred on ``left``, so a sample up to half a bucket
    before ``left`` is charted there, though it stays out of the averages.
    ``left`` is the newest sample time less the range, and scrape jitter puts
    the capture at that edge a few milliseconds either side of it.
    """
    last = count - 1
    per_bucket = 1.0 / width
    busy_sum = [0.0] * count
    busy_n = [0] * count
    mem_sum = [0.0] * count
    mem_n = [0] * count
    # (first bucket, last bucket, busy, memory): a GPU's value carried into buckets it missed.
    holds: list[tuple[int, int, float | None, float | None]] = []
    busy_avg: dict[int, float | None] = {}
    mem_avg: dict[int, float | None] = {}
    pool = [0.0, 0.0, 0.0, 0.0]  # busy sum, busy weight, memory sum, memory weight

    def reach(sample_ts: float) -> int:
        """The newest bucket a value sampled at ``sample_ts`` can be carried into."""
        behind = latest - sample_ts - hold_s
        return last if behind <= 0 else last - math.ceil(behind * per_bucket - 1e-9)

    for gpu_id, gpu in series.gpus.items():
        gpu.sort()
        g_busy: list = [None] * count
        g_mem: list = [None] * count
        if per_gpu is not None:
            per_gpu[gpu_id] = (g_busy, g_mem)
        ts, util, mem, weight = gpu.ts, gpu.util, gpu.mem, gpu.weight
        begin = bisect_left(ts, left - 1e-6)
        edge = bisect_left(ts, left - 0.5 * width) if last else begin
        # The GPU's previous value: its bucket (-2: none), busy, memory and sample
        # time. The sample just before the range carries into its first buckets.
        held_bucket, held_busy, held_mem, held_ts = -2, None, None, 0.0
        if edge and weight[edge - 1] > 0 and ts[edge - 1] >= left - hold_s:
            u, m = util[edge - 1], mem[edge - 1]
            held_bucket, held_busy, held_mem, held_ts = -1, (u if u == u else None), (m if m == m else None), ts[edge - 1]
        current = -1
        current_ts = 0.0
        # Samples in [edge, begin) open the first bucket; the totals start without them.
        xbu = xbw = xmu = xmw = 0.0
        for i in range(edge, begin):
            w = weight[i]
            if not w > 0:
                continue
            current, current_ts = 0, ts[i]
            u = util[i]
            if u == u:
                xbu += u * w
                xbw += w
            m = mem[i]
            if m == m:
                xmu += m * w
                xmw += w
        bu, bw, mu, mw = xbu, xbw, xmu, xmw
        tbu, tbw, tmu, tmw = -xbu, -xbw, -xmu, -xmw
        end = len(ts) if until is None else bisect_right(ts, until + 1e-6)
        for i in range(begin, end):
            w = weight[i]
            if not w > 0:  # NaN: the row carried no usable interval
                continue
            t = ts[i]
            bucket = last - int((latest - t) * per_bucket + 0.5) if last else 0  # latest >= t
            if bucket < 0:
                bucket = 0
            if bucket != current:
                if current >= 0:
                    held_busy = held_mem = None
                    if bw > 0:
                        held_busy = bu / bw
                        busy_sum[current] += held_busy
                        busy_n[current] += 1
                        g_busy[current] = held_busy
                    if mw > 0:
                        held_mem = mu / mw
                        mem_sum[current] += held_mem
                        mem_n[current] += 1
                        g_mem[current] = held_mem
                    tbu += bu
                    tbw += bw
                    tmu += mu
                    tmw += mw
                    if held_bucket == -2 and current > 0:  # its first values, with nothing before them
                        holds.append((current - 1, current - 1, held_busy, held_mem, gpu_id))
                    held_bucket, held_ts = current, current_ts
                if held_bucket > -2 and bucket > held_bucket + 1:
                    stop = min(bucket - 1, reach(held_ts))
                    if stop > held_bucket:
                        holds.append((held_bucket + 1, stop, held_busy, held_mem, gpu_id))
                current = bucket
                bu = bw = mu = mw = 0.0
            current_ts = t
            u = util[i]
            if u == u:
                bu += u * w
                bw += w
            m = mem[i]
            if m == m:
                mu += m * w
                mw += w
        if current >= 0:
            held_busy = held_mem = None
            if bw > 0:
                held_busy = bu / bw
                busy_sum[current] += held_busy
                busy_n[current] += 1
                g_busy[current] = held_busy
            if mw > 0:
                held_mem = mu / mw
                mem_sum[current] += held_mem
                mem_n[current] += 1
                g_mem[current] = held_mem
            tbu += bu
            tbw += bw
            tmu += mu
            tmw += mw
            if held_bucket == -2 and current > 0:
                holds.append((current - 1, current - 1, held_busy, held_mem, gpu_id))
            held_bucket, held_ts = current, current_ts
        if held_bucket > -2 and held_bucket < last:
            stop = reach(held_ts)
            if stop > held_bucket:
                holds.append((held_bucket + 1, stop, held_busy, held_mem, gpu_id))
        busy_avg[gpu_id] = tbu / tbw if tbw > 0 else None
        mem_avg[gpu_id] = tmu / tmw if tmw > 0 else None
        if tbw > 0:
            pool[0] += tbu
            pool[1] += tbw
        if tmw > 0:
            pool[2] += tmu
            pool[3] += tmw
    # Every GPU's own samples are in, so a bucket's count says whether any GPU sampled it.
    for first, stop, busy_value, mem_value, hold_gpu in holds:
        own = per_gpu.get(hold_gpu) if per_gpu is not None else None
        for k in range(first, stop + 1):
            if busy_value is not None and busy_n[k]:
                busy_sum[k] += busy_value
                busy_n[k] += 1
                if own is not None and own[0][k] is None:
                    own[0][k] = busy_value
            if mem_value is not None and mem_n[k]:
                mem_sum[k] += mem_value
                mem_n[k] += 1
                if own is not None and own[1][k] is None:
                    own[1][k] = mem_value
    busy = [busy_sum[k] / busy_n[k] if busy_n[k] else None for k in range(count)]
    memory = [mem_sum[k] / mem_n[k] if mem_n[k] else None for k in range(count)]
    return busy, memory, busy_avg, mem_avg, _pooled(pool)


def _pooled(pool) -> tuple[float | None, float | None]:
    busy_sum, busy_weight, mem_sum, mem_weight = pool
    return (busy_sum / busy_weight if busy_weight > 0 else None, mem_sum / mem_weight if mem_weight > 0 else None)


_NO_OLDER: tuple[float | None, float | None] = (None, None)


def _stats(low: float | None, high: float | None, avg: float | None, latest: float | None,
           older: tuple[float | None, float | None] = _NO_OLDER) -> SeriesStats:
    """``avg``: the GPU-time mean; ``older``: the lowest and highest sample of the history known only as bins."""
    # The newest capture is a point in every scope.
    lows = [value for value in (low, latest) if value is not None]
    highs = [value for value in (high, latest) if value is not None]
    shown_min = min(lows) if lows else None
    shown_max = max(highs) if highs else None
    floor, ceiling = older
    # An average across GPUs lies between the lowest and highest samples it
    # averages, so that history can't hide a lower min than its lowest sample, or
    # a higher max than its highest. Compared at the whole percent shown.
    return SeriesStats(
        min=shown_min, avg=avg, max=shown_max, latest=latest,
        min_bound=floor is not None and shown_min is not None and round(floor) < round(shown_min),
        max_bound=ceiling is not None and shown_max is not None and round(ceiling) > round(shown_max),
    )


class _Native:
    """The average across GPUs at each sampling interval over the raw samples.

    Buckets are anchored on the newest sample, one sampling interval wide.
    ``extremes(left)`` gives the busy and memory min and max over the buckets
    from ``left`` to the newest; nested scopes therefore nest their extremes.
    """

    def __init__(self, series: _Series, hold_s: float) -> None:
        self.ready = series.min_interval is not None and series.latest is not None
        if not self.ready:
            return
        self.latest = series.latest
        left = max(series.chart_start, series.raw_start)
        span = max(0.0, self.latest - left)
        self.width = max(series.min_interval, span / (_MAX_NATIVE_BUCKETS - 1)) if span > 0 else 1.0
        count = math.floor(span / self.width + 1e-9) + 1 if span > 0 else 1
        self.last = count - 1
        busy, memory, _busy_avg, _mem_avg, _pooled_avg = _accumulate(
            series, left, self.latest, self.width, count, hold_s)
        self.busy = self._suffix(busy)
        self.mem = self._suffix(memory)

    @staticmethod
    def _suffix(values: list[float | None]) -> list[tuple[float | None, float | None]]:
        """(min, max) of values[k:] for every k."""
        out: list[tuple[float | None, float | None]] = [(None, None)] * len(values)
        low = high = None
        for k in range(len(values) - 1, -1, -1):
            value = values[k]
            if value is not None:
                low = value if low is None or value < low else low
                high = value if high is None or value > high else high
            out[k] = (low, high)
        return out

    def extremes(self, left: float) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None]]:
        if not self.ready:
            return (None, None), (None, None)
        k = self.last - round((self.latest - left) / self.width) if self.last else 0
        k = min(self.last, max(0, k))
        return self.busy[k], self.mem[k]


def _merge(pair: tuple[float | None, float | None], values) -> tuple[float | None, float | None]:
    points = [value for value in (*pair, *values) if value is not None]
    return (min(points), max(points)) if points else (None, None)


def _value_interval(series: _Series) -> float | None:
    """How often the GPUs' busy values change, from their recent samples (None if unknown).

    DCGM refreshes a field on its own schedule, often far less often than the
    sampler scrapes, so consecutive rows repeat a value. A value can also stay
    the same across a refresh, so each GPU's time between changes is a multiple
    of the refresh: its lower quartile estimates it, and the median over GPUs
    leaves out an idle GPU's long runs. Runs cut short by the samples' ends
    don't count.
    """
    per_gpu: list[float] = []
    for gpu in series.gpus.values():
        gpu.sort()
        ts = gpu.ts[-(_VALUE_RUN_SAMPLES + 1):]
        util = gpu.util[-(_VALUE_RUN_SAMPLES + 1):]
        changes = [ts[i] for i in range(1, len(ts))
                   if util[i] == util[i] and util[i - 1] == util[i - 1] and util[i] != util[i - 1]]
        runs = sorted(b - a for a, b in zip(changes, changes[1:]))
        if len(runs) >= 2:
            per_gpu.append(runs[len(runs) // 4])
    if not per_gpu:
        return None
    per_gpu.sort()
    middle = len(per_gpu) // 2
    return per_gpu[middle] if len(per_gpu) % 2 else (per_gpu[middle - 1] + per_gpu[middle]) / 2


def _derived_interval(series: _Series) -> float | None:
    """The median spacing of each GPU's recent consecutive samples."""
    deltas: list[float] = []
    for gpu in series.gpus.values():
        gpu.sort()
        ts = gpu.ts[-(_INTERVAL_DELTAS + 1):]
        deltas.extend(b - a for a, b in zip(ts, ts[1:]) if b - a > 1e-3)
    if not deltas:
        return None
    deltas.sort()
    middle = len(deltas) // 2
    return deltas[middle] if len(deltas) % 2 else (deltas[middle - 1] + deltas[middle]) / 2


class _Capture:
    """The newest capture: the GPUs reporting in it and their average busy and memory."""

    def __init__(self, series: _Series, tolerance: float) -> None:
        newest = series.latest or 0.0
        self.reporting = tuple(
            sorted(gpu_id for gpu_id, gpu in series.gpus.items() if gpu.latest and gpu.latest[0] >= newest - tolerance)
        )
        rows = [series.gpus[gpu_id].latest for gpu_id in self.reporting]
        self.busy = _mean(row[1] for row in rows)
        self.mem = _mean(row[2] for row in rows)


def _gpu_rows(series: _Series, capture: _Capture, busy_avg: dict, mem_avg: dict) -> tuple[GPURow, ...]:
    rows = []
    reporting = set(capture.reporting)
    for gpu_id in sorted(series.gpus):
        _ts, util, mem, used, total, doc = series.gpus[gpu_id].latest
        rows.append(
            GPURow(
                global_gpu_index=gpu_id,
                current_util_pct=util,
                window_avg_util_pct=busy_avg.get(gpu_id),
                fb_used_bytes=used,
                fb_total_bytes=total,
                current_mem_pct=mem,
                window_avg_mem_pct=mem_avg.get(gpu_id),
                sm_active_pct=_gauge(doc.get("sm_active_pct"), 100.0),
                tensor_active_pct=_gauge(doc.get("tensor_active_pct"), 100.0),
                dram_active_pct=_gauge(doc.get("dram_active_pct"), 100.0),
                reporting=gpu_id in reporting,
            )
        )
    return tuple(rows)


def _finish(series: _Series, name: str, left: float, chart, mem_chart, busy_avg, mem_avg, capture,
            extremes, pooled, older=(_NO_OLDER, _NO_OLDER), per_gpu=None) -> RangeView:
    (busy_low, busy_high), (mem_low, mem_high) = extremes
    return RangeView(
        name=name,
        start=_utc(left),
        end=_utc(series.latest),
        chart=tuple(chart),
        rows=_gpu_rows(series, capture, busy_avg, mem_avg),
        mem_chart=tuple(mem_chart),
        busy=_stats(busy_low, busy_high, pooled[0], capture.busy, older[0]),
        memory=_stats(mem_low, mem_high, pooled[1], capture.mem, older[1]),
        gpus=sum(1 for gpu_id in series.gpus if busy_avg.get(gpu_id) is not None or mem_avg.get(gpu_id) is not None),
        gpu_series={gpu_id: (tuple(busy), tuple(memory)) for gpu_id, (busy, memory) in (per_gpu or {}).items()},
    )


def _raw_view(series: _Series, name: str, duration: timedelta | None, capture: _Capture, native: _Native,
              hold_s: float) -> RangeView:
    """A window (or a job within the raw retention) from the raw samples."""
    latest = series.latest
    first = series.chart_start
    left = first if duration is None else max(first, latest - duration.total_seconds())
    width, count = _grid(left, latest, series.min_interval)
    per_gpu: dict = {}
    chart, mem_chart, busy_avg, mem_avg, pooled = _accumulate(series, left, latest, width, count, hold_s,
                                                              per_gpu=per_gpu)
    return _finish(series, name, left, chart, mem_chart, busy_avg, mem_avg, capture, native.extremes(left), pooled,
                   per_gpu=per_gpu)


def _binned_view(series: _Series, name: str, capture: _Capture, native: _Native) -> RangeView:
    """The whole job from the running bins, once it is longer than the raw retention."""
    width = series.bin_width
    chart: list[float | None] = []
    mem_chart: list[float | None] = []
    per_gpu: dict = {gpu_id: ([], []) for gpu_id in series.gpus}
    # Lowest busy, highest busy, lowest memory, highest memory average across GPUs
    # at one interval in the history the raw samples no longer hold.
    folded = [_INF, -_INF, _INF, -_INF]
    # Lowest and highest sample in that history's inexact bins (see _Series.inexact).
    limits = [_INF, -_INF, _INF, -_INF]
    if series.bins:
        # rebin() keeps the bins' keys within _MAX_BINS of each other, and every bin
        # holds a charted row, so no row can stretch this walk.
        for key in range(min(series.bins), max(series.bins) + 1):
            cells = series.bins.get(key, {}).values()
            busy = _mean(cell[0] / cell[1] if cell[1] > 0 else None for cell in cells)
            memory = _mean(cell[2] / cell[3] if cell[3] > 0 else None for cell in cells)
            chart.append(busy)
            mem_chart.append(memory)
            bin_cells = series.bins.get(key, {})
            for gpu_id, (own_busy, own_mem) in per_gpu.items():
                cell = bin_cells.get(gpu_id)
                own_busy.append(cell[0] / cell[1] if cell is not None and cell[1] > 0 else None)
                own_mem.append(cell[2] / cell[3] if cell is not None and cell[3] > 0 else None)
            if key * width >= series.raw_start:
                continue
            values = series.extremes.get(key)
            if values is not None:
                for i in (0, 2):
                    folded[i] = min(folded[i], values[i])
                    folded[i + 1] = max(folded[i + 1], values[i + 1])
            if key in series.inexact:
                # Rows that never reached the raw samples: only their bin's average is
                # known, and its lowest and highest sample bound the true extremes.
                for i, value in ((0, busy), (2, memory)):
                    if value is not None:
                        folded[i] = min(folded[i], value)
                        folded[i + 1] = max(folded[i + 1], value)
                for cell in cells:
                    limits[0] = min(limits[0], cell[4])
                    limits[1] = max(limits[1], cell[5])
                    limits[2] = min(limits[2], cell[6])
                    limits[3] = max(limits[3], cell[7])
    busy_avg = {gpu_id: (gpu.busy_wsum / gpu.busy_w if gpu.busy_w > 0 else None) for gpu_id, gpu in series.gpus.items()}
    mem_avg = {gpu_id: (gpu.mem_wsum / gpu.mem_w if gpu.mem_w > 0 else None) for gpu_id, gpu in series.gpus.items()}
    pooled = _pooled([sum(gpu.busy_wsum for gpu in series.gpus.values()), sum(gpu.busy_w for gpu in series.gpus.values()),
                      sum(gpu.mem_wsum for gpu in series.gpus.values()), sum(gpu.mem_w for gpu in series.gpus.values())])
    # The raw samples give exact extremes for the retention, and folded ones for the
    # history before it; an inexact bin's share may be a bound, which its lowest
    # and highest samples tell.
    busy_extremes, mem_extremes = native.extremes(series.chart_start)
    extremes = (_merge(busy_extremes, [v for v in folded[:2] if math.isfinite(v)]),
                _merge(mem_extremes, [v for v in folded[2:] if math.isfinite(v)]))
    found = [value if math.isfinite(value) else None for value in limits]
    older = ((found[0], found[1]), (found[2], found[3]))
    return _finish(series, name, series.chart_start, chart, mem_chart, busy_avg, mem_avg, capture, extremes, pooled,
                   older, per_gpu=per_gpu)


def _snapshot(series: _Series, partial: bool) -> SubJobResource:
    interval = _derived_interval(series)
    # How long a GPU counts as reporting without a new sample.
    hold_s = max(_REPORTING_TOLERANCE_S, 3 * (interval or series.min_interval or 0))
    capture = _Capture(series, hold_s)
    native = _Native(series, hold_s)
    raw_covers_job = series.raw_start <= series.chart_start
    range_views = {
        name: (
            _raw_view(series, name, duration, capture, native, hold_s)
            if duration is not None or raw_covers_job
            else _binned_view(series, name, capture, native)
        )
        for name, duration in RANGES.items()
    }
    reported = max(series.reported, key=series.reported.get) if series.reported else None
    return SubJobResource(
        sub_job_id=series.sub_job_id,
        latest_sample_ts=_utc(series.latest),
        views=range_views,
        first_sample_ts=_utc(series.first),
        sample_interval_s=interval,
        reported_interval_s=reported,
        gpu_ids=tuple(sorted(series.gpus)),
        reporting_gpu_ids=capture.reporting,
        partial=partial,
        value_interval_s=_value_interval(series),
    )


# ─── loading ─────────────────────────────────────────────────────────────────


def _line_start(handle, position: int) -> int:
    """The offset of the first line that starts at or after ``position``."""
    if position <= 0:
        return 0
    handle.seek(position - 1)
    scanned = position - 1
    while True:
        block = handle.read(64 * 1024)
        if not block:
            return scanned
        newline = block.find(b"\n")
        if newline >= 0:
            return scanned + newline + 1
        scanned += len(block)


@dataclass
class _FileState:
    series: _Series
    ino: tuple[int, int] = (0, 0)
    names: list[str] = field(default_factory=list)
    offset: int = 0  # bytes read: the end of the last complete line
    head: bytes = b""
    tail: bytes = b""
    snapshot: SubJobResource | None = None
    snapshot_version: int = -1
    # The manifest says the file holds only the newest chunks so far.
    older_pending: bool = False


class ResourceLoader:
    """Incremental reader for one job's telemetry cache directory.

    Use one loader from one thread at a time (the TUI holds the cache
    directory's lock around each load). ``bytes_read`` and ``documents_read``
    count the bytes and rows parsed by all loads so far. ``last`` is the
    snapshot the latest load returned; it is immutable, so another thread may
    read it without the lock.
    """

    def __init__(self, metrics_root: Path) -> None:
        self.metrics_root = Path(metrics_root)
        self._files: dict[str, _FileState] = {}
        self.bytes_read = 0
        self.documents_read = 0
        self.last: ResourceData | None = None

    def load(
        self,
        *,
        priority: str | None = None,
        window: timedelta | None = None,
        on_partial: Callable[[ResourceData], None] | None = None,
    ) -> ResourceData:
        """Read what changed since the last load and return a snapshot.

        ``priority`` (the sub-job on screen) is read first. When it has to be
        read from the start and the file is large, it is read newest-first until
        ``window`` is covered, and ``on_partial`` gets that snapshot before the
        older bytes are read.
        """
        present: dict[str, tuple[Path, list[str], int, bool]] = {}
        try:
            paths = sorted(self.metrics_root.glob("*/gpu.jsonl"))
        except OSError:
            paths = []
        for path in paths:
            manifest = _read_manifest(path)
            if manifest is not None:
                present[path.parent.name] = (path, *manifest)
        for sub_job_id in list(self._files):
            if sub_job_id not in present:
                del self._files[sub_job_id]
        horizon = time.time() + _MAX_AHEAD_S
        for sub_job_id in sorted(present, key=lambda sid: sid != priority):
            path, names, committed, older_pending = present[sub_job_id]
            partial = on_partial if sub_job_id == priority else None
            try:
                self._update(sub_job_id, path, names, committed, window or RANGES[DEFAULT_RANGE], partial, horizon)
                self._files[sub_job_id].older_pending = older_pending
            except OSError:
                self._files.pop(sub_job_id, None)
            except BaseException:
                # A failure part-way through a read (the on_partial callback raising,
                # say) leaves rows ingested past the recorded offset; reading on top of
                # them would count them twice, so the next load starts this file afresh.
                self._files.pop(sub_job_id, None)
                raise
        data = self._data(set())
        self.last = data
        return data

    def _data(self, partial: set[str]) -> ResourceData:
        sub_jobs: dict[str, SubJobResource] = {}
        ignored: dict[str, int] = {}
        for sub_job_id in sorted(self._files):
            state = self._files[sub_job_id]
            series = state.series
            if series.ignored:
                ignored[sub_job_id] = series.ignored
            if series.latest is None:
                continue
            if sub_job_id in partial:
                sub_jobs[sub_job_id] = _snapshot(series, True)
                continue
            if (state.snapshot is None or state.snapshot_version != series.version
                    or state.snapshot.partial != state.older_pending):
                state.snapshot = _snapshot(series, state.older_pending)
                state.snapshot_version = series.version
            sub_jobs[sub_job_id] = state.snapshot
        return ResourceData(sub_jobs=sub_jobs, ignored_rows_by_sub_job=ignored)

    def _update(self, sub_job_id: str, path: Path, names: list[str], committed: int, window: timedelta,
                on_partial: Callable[[ResourceData], None] | None, horizon: float) -> None:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            ino = (stat.st_dev, stat.st_ino)
            state = self._files.get(sub_job_id)
            if state is not None and self._extends(state, handle, ino, names, committed):
                if committed > state.offset:
                    state.series.horizon = horizon
                    state.offset = self._read(state, handle, state.offset, committed)
                    state.series.trim()
                    state.tail = self._fingerprint(handle, state.offset)
                state.names = list(names)
                return
            # New, rebuilt or rewritten: read it from the start.
            state = self._files[sub_job_id] = _FileState(series=_Series(sub_job_id), ino=ino, names=list(names))
            series = state.series
            series.horizon = horizon
            series.plan(*self._peek(handle, committed, sub_job_id, horizon))
            if on_partial is not None and committed >= _NEWEST_FIRST_MIN_BYTES:
                end, rest = self._read_newest(state, handle, committed, window)
                if rest > 0:
                    on_partial(self._data({sub_job_id}))
                    self._read(state, handle, 0, rest)
            else:
                end = self._read(state, handle, 0, committed)
            state.offset = end
            series.trim()
            state.head = self._fingerprint(handle, min(_FINGERPRINT, end))
            state.tail = self._fingerprint(handle, end)

    def _read_newest(self, state: _FileState, handle, committed: int, window: timedelta) -> tuple[int, int]:
        """Read slices back from the end until ``window`` is covered.

        Returns (the offset after the last complete line, the start of what is left).
        """
        series = state.series
        position, step = committed, _NEWEST_FIRST_STEP
        end: int | None = None
        while position > 0:
            begin = _line_start(handle, position - step) if position > step else 0
            if begin >= position:  # one line longer than the step
                step *= 2
                continue
            stop = self._read(state, handle, begin, position, fold=False)
            if end is None:
                end = stop
            position = begin
            if series.first is not None and series.first <= (
                series.latest - window.total_seconds() - _NEWEST_FIRST_SLACK_S
            ):
                break
            step *= 2
        return (end or 0), position

    def _extends(self, state: _FileState, handle, ino: tuple[int, int], names: list[str], committed: int) -> bool:
        """Whether this is the file read before, with bytes only appended."""
        if ino != state.ino or committed < state.offset or names[: len(state.names)] != state.names:
            return False
        return (
            self._fingerprint(handle, len(state.head)) == state.head
            and self._fingerprint(handle, state.offset) == state.tail
        )

    @staticmethod
    def _fingerprint(handle, end: int) -> bytes:
        start = max(0, end - _FINGERPRINT)
        handle.seek(start)
        return handle.read(end - start)

    @staticmethod
    def _peek(handle, committed: int, sub_job_id: str, horizon: float) -> tuple[float | None, float | None]:
        """(first charted, newest charted) sample time, from the two ends of the file.

        Scratch series ingest the rows there, so they are judged exactly as the
        read will judge them: a file that ends in rows a load skips (system
        rows, say) still sizes the bins from its newest charted GPU row.
        """

        def scan(lines) -> _Series:
            scratch = _Series(sub_job_id)
            scratch.horizon = horizon
            for raw in lines:
                try:
                    scratch.ingest(json.loads(raw))
                except (RecursionError, UnicodeDecodeError, ValueError):
                    continue
            return scratch

        handle.seek(0)
        head = scan(handle.read(min(committed, _PEEK_BYTES)).split(b"\n")[:-1])
        start = max(0, committed - _PEEK_BYTES)
        handle.seek(start)
        lines = handle.read(committed - start).split(b"\n")[:-1]
        tail = scan(lines[1:] if start else lines)  # the first may be the end of an earlier line
        return head.data_first, tail.data_last

    def _read(self, state: _FileState, handle, start: int, end: int, *, fold: bool = True) -> int:
        """Ingest the complete lines in [start, end); return the offset after the last one.

        Rows in the file run oldest first, so after each block the samples well
        behind the newest row read so far leave the raw retention (``fold``):
        their per-interval extremes go into the bins (see :meth:`_Series.trim`).
        A newest-first slice (``fold=False``) is followed by older rows, so it
        folds nothing.
        """
        series = state.series
        series.pass_newest = -_INF
        ingest = series.ingest
        loads = json.loads
        handle.seek(start)
        position = start
        remaining = end - start
        carry = b""
        documents = 0
        while remaining > 0:
            block = handle.read(min(remaining, _READ_BLOCK))
            if not block:
                break
            remaining -= len(block)
            self.bytes_read += len(block)
            if carry:
                block = carry + block
            lines = block.split(b"\n")
            carry = lines.pop()  # an unterminated line is not complete yet
            for raw in lines:
                position += len(raw) + 1
                if not raw or raw.isspace():
                    continue
                documents += 1
                try:
                    doc = loads(raw)
                except (RecursionError, UnicodeDecodeError, ValueError):
                    series.ignored += 1
                    continue
                if ingest(doc) == _IGNORED:
                    series.ignored += 1
            series.rebin()
            if fold:
                series.trim(upto=series.pass_newest - _FOLD_MARGIN_S)
        self.documents_read += documents
        if documents:
            series.version += 1
        return position


def load_resource_data(metrics_root: Path) -> ResourceData:
    """One full read of a telemetry cache directory, keeping no state."""
    return ResourceLoader(metrics_root).load()


_LOADERS: OrderedDict[str, ResourceLoader] = OrderedDict()
_LOADERS_GUARD = threading.Lock()
_MAX_LOADERS = 4


def resource_loader(metrics_root: Path) -> ResourceLoader:
    """The shared loader for a cache directory, so reopening a job reuses what was read.

    Loaders for the few most recently used directories are kept.
    """
    key = str(Path(metrics_root).resolve())
    with _LOADERS_GUARD:
        loader = _LOADERS.pop(key, None) or ResourceLoader(Path(metrics_root))
        _LOADERS[key] = loader
        while len(_LOADERS) > _MAX_LOADERS:
            _LOADERS.popitem(last=False)
        return loader


# ─── GPU type and allocation ─────────────────────────────────────────────────
# The control plane's rule for what a sub-job adds to its job's GPU request: its
# config's ``n_gpus``, except that a log_prob sub-job uses none, and a sub-job
# with ``multiplex_job_id`` runs on that sub-job's GPUs and adds none. The job
# list and the preview apply it through the same helpers in ``views``.


@dataclass(frozen=True)
class GpuAllocation:
    """What a sub-job asked for: the job's GPU type and the sub-job's GPUs."""

    hardware: str | None = None
    # The sub-job's own GPUs (None: its config doesn't say).
    requested: int | None = None
    # ``multiplex_job_id``: the sub-job whose GPUs this one runs on.
    shared_with: str | None = None
    log_prob: bool = False


def gpu_allocation(job: dict | None, sub_job_id: str | None) -> GpuAllocation:
    """What ``sub_job_id`` of ``job`` asked for, read as the job list and the preview read it.

    The sub-job is looked up among :func:`views.sub_jobs` (which derives the ids
    of an UNKNOWN job's configs), and its type, the sub-job it is multiplexed
    onto and its GPUs come from :func:`views.sub_job_kind`,
    :func:`views.multiplex_target` and :func:`views.requested_gpus`.
    """
    job = job if isinstance(job, dict) else {}
    hardware = views.collapse(job.get("hardware")) or None
    sub_job = next(
        (sj for sj in views.sub_jobs(job) if sub_job_id and (sj.get("sub_job_id") or sj.get("id")) == sub_job_id),
        None,
    )
    if sub_job is None:
        return GpuAllocation(hardware=hardware)
    kind = views.sub_job_kind(sub_job)
    shared = views.multiplex_target(sub_job)
    try:
        count = views.requested_gpus(sub_job)
    except (OverflowError, ValueError):  # a count no int holds, such as an infinite n_gpus
        count = 0
    # requested_gpus reads a missing or unusable n_gpus as 0: unknown here.
    requested = 0 if kind == "log_prob" or shared else (count if count > 0 else None)
    return GpuAllocation(hardware=hardware, requested=requested, shared_with=shared or None,
                         log_prob=kind == "log_prob")


def allocation_text(allocation: GpuAllocation, reporting: int | None = None, seen: int | None = None) -> str:
    """``H200  ·  GPUs: 8 requested, 8 reporting`` for the Resources header."""
    if allocation.log_prob:
        requested = "0 requested (log_prob)"  # the sub-job id's spelling, as the list and Logs use
    elif allocation.shared_with:
        requested = "0 requested (multiplexed)"
    elif allocation.requested is None:
        requested = "requested count unknown"
    else:
        requested = f"{allocation.requested} requested"
    text = f"{allocation.hardware or 'GPU type unknown'}  ·  GPUs: {requested}"
    if reporting is not None:
        text += f", {reporting} reporting"
        if seen is not None and seen != reporting:
            text += f" ({seen} seen)"
    return text


def no_metrics_text(allocation: GpuAllocation, *, waiting: str | None = None, who: str = "this sub-job") -> str:
    """Why a sub-job's Resources tab is empty once its telemetry has loaded.

    ``waiting``: the status word while the job or sub-job (``who``) has not
    started running yet, such as ``placing``; its samples are still to come.
    """
    if allocation.shared_with:
        # Only the sub-job that owns the GPUs reports them, covering everything that runs on them.
        return (f"no GPU metrics for this sub-job: it runs on the GPUs of {allocation.shared_with}, "
                "whose metrics include it")
    if waiting:
        return (f"no GPU metrics yet: {who} is {waiting.upper()} (the first samples arrive about a minute "
                "after it starts running)")
    return "no GPU metrics for this sub-job"


def skipped_rows_text(count: int, *, short: bool = False) -> str:
    """``3 rows skipped (malformed or unsupported)``: rows the loader could not use (``short``: without the why)."""
    text = f"{count} {'row' if count == 1 else 'rows'} skipped"
    return text if short else f"{text} (malformed or unsupported)"


# ─── text ────────────────────────────────────────────────────────────────────

BUSY_LABEL = "GPU busy (kernel-active %)"
MEMORY_LABEL = "GPU memory used %"
SHORT_LABELS = ("GPU busy", "GPU memory")  # the two labels where the long ones don't fit
_BLOCKS = "▁▂▃▄▅▆▇█"


def _collapse(values: tuple[float | None, ...], width: int) -> list[float | None]:
    if width <= 0 or not values:
        return []
    if len(values) <= width:
        return list(values)
    collapsed: list[float | None] = []
    for column in range(width):
        start = math.floor(column * len(values) / width)
        end = max(start + 1, math.floor((column + 1) * len(values) / width))
        present = [value for value in values[start:end] if value is not None]
        collapsed.append(sum(present) / len(present) if present else None)
    return collapsed


def _glyph(value: float | None) -> str:
    if value is None:
        return " "
    return _BLOCKS[min(len(_BLOCKS) - 1, max(0, math.ceil(value / 100 * len(_BLOCKS)) - 1))]


def _series(view: RangeView, series: str) -> tuple[float | None, ...]:
    return view.mem_chart if series == "memory" else view.chart


def _bar_row(values: list[float | None], height: int, level: int) -> str:
    """Row ``level`` (0 at the bottom) of a ``height``-row bar chart, drawn in eighths of a row."""
    cells = []
    for value in values:
        if value is None:
            cells.append(" ")
            continue
        eighths = min(height * 8, max(1, math.ceil(value / 100 * height * 8)))
        fill = eighths - 8 * level
        cells.append("█" if fill >= 8 else " " if fill <= 0 else _BLOCKS[fill - 1])
    return "".join(cells)


def render_chart(view: RangeView, width: int, *, series: str = "busy", height: int = 1) -> str:
    """A 0-100% chart of ``view``'s busy series (or ``series="memory"``), ``height`` text rows tall.

    Each bar is drawn in eighths of a row, so a taller chart tells 77% from 87%. Two
    rows or more label 100%, 50% (the row whose top is 50%, or around it) and 0%.
    """
    if width <= 0:
        return ""
    if width < 7:
        return "".join(_glyph(value) for value in _collapse(_series(view, series), width))
    height = max(1, height)
    inner = max(1, width - 6)
    columns = _collapse(_series(view, series), inner)
    chart_width = len(columns)
    times = (view.start, view.start + (view.end - view.start) / 2, view.end)
    # A day or more apart, the times alone would repeat: date them when they fit.
    dated = view.end - view.start >= timedelta(days=1) and chart_width >= 2 * len("10-13 20:59") + 1
    start, middle, end = (moment.astimezone().strftime("%m-%d %H:%M" if dated else "%H:%M") for moment in times)
    size = len(end)
    padding = " " * (inner - chart_width)
    if chart_width >= 3 * size + 2:
        left_gap = (chart_width - 3 * size) // 2
        right_gap = chart_width - 3 * size - left_gap
        labels = padding + start + (" " * left_gap) + middle + (" " * right_gap) + end
    elif chart_width >= 2 * size + 1:
        labels = padding + start + (" " * (chart_width - 2 * size)) + end
    else:
        # Too few columns for a label: the newest sample's time ends under it, reaching
        # into the empty space before the bars when it fits there.
        labels = end.rjust(inner) if inner >= size else ""
    lines = []
    for index in range(height):
        if index == 0:
            axis = "100% ┤"
        elif index == height // 2:
            axis = " 50% ┤"
        else:
            axis = "     │"
        lines.append(axis + padding + _bar_row(columns, height, height - 1 - index))
    lines.append(f"  0% └{padding}{'─' * chart_width}")
    lines.append(f"      {labels}".rstrip())
    return "\n".join(lines)


def has_samples(view: RangeView, *, series: str = "busy") -> bool:
    return any(value is not None for value in _series(view, series))


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}%"


def format_bytes(value: float | None) -> str:
    if value is None:
        return "—"
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"


def format_interval(seconds: float | None) -> str:
    """``5 s``, ``2.5 s``, ``1m 30s``; ``—`` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "—"
    if seconds < 10 and abs(seconds - round(seconds)) >= 0.05:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{round(seconds)} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m {rest:02d}s" if rest else f"{minutes}m"


def format_age(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 86400}d {seconds % 86400 // 3600}h"


def sampling_text(sub_job: SubJobResource, now: datetime) -> str:
    """``scraped every 5 s (values change about every 1m)  ·  newest 40 s ago (12:34:56)``."""
    return f"{scrape_text(sub_job)}  ·  {freshness_text(sub_job, now)}"


def scrape_text(sub_job: SubJobResource, *, notes: bool = True) -> str:
    """``scraped every 5 s (values change about every 1m)``; ``notes=False``: ``scraped every 5 s``.

    "Scraped": the sampler reads DCGM at this interval, but DCGM may refresh a
    field less often, so consecutive samples can repeat a value. When the busy
    values change less often than every two scrapes, the text says how often:
    the newest value can be that much older than the newest row.
    """
    derived, reported = sub_job.sample_interval_s, sub_job.reported_interval_s
    scraped = derived or reported
    text = f"scraped every {format_interval(scraped)}"
    if not notes:
        return text
    found = []
    if derived and reported and abs(derived - reported) > 0.25 * reported:
        found.append(f"rows report {format_interval(reported)}")
    changes = sub_job.value_interval_s
    if changes and scraped and changes > 2 * scraped:
        found.append(f"values change about every {format_interval(changes)}")
    if found:
        text += f" ({'; '.join(found)})"
    return text


def freshness_text(sub_job: SubJobResource, now: datetime, *, stale: bool = False, clock: bool = True) -> str:
    """``newest 40 s ago (12:34:56)``; ``stale``: the sub-job should be sending samples and isn't.

    A stale one reads ``no new GPU samples for 41m 56s; the windows end at 12:34:56``:
    the windows and the table's latest values are measured back from that sample.
    ``clock=False`` leaves the sample's time out: ``newest 40 s ago``.
    """
    age = max(0.0, (now - sub_job.latest_sample_ts).total_seconds())
    when = sub_job.latest_sample_ts.astimezone().strftime("%H:%M:%S" if age < 86400 else "%Y-%m-%d %H:%M")
    if stale:
        text = f"no new GPU samples for {format_age(age)}"
        return f"{text}; the windows end at {when}" if clock else text
    text = f"newest {format_age(age)} ago"
    return f"{text} ({when})" if clock else text


def age_slack(sub_job: SubJobResource, now: datetime) -> int:
    """Cells the age in :func:`freshness_text` can still grow by (``5 s`` to ``59m 59s``).

    A header keeps that much room, so the age ticking past a minute (or an
    hour) never makes the line longer than the width it was fitted to.
    """
    age = max(0.0, (now - sub_job.latest_sample_ts).total_seconds())
    return max(0, len("59m 59s") - len(format_age(age)))


def window_label(name: str) -> str:
    return "whole job" if name == "all" else f"last {name}"


def across_label(gpus: int) -> str:
    """``mean of 8 GPUs``: the stats and charts average across the GPUs (``""`` for one GPU)."""
    return f"mean of {gpus} GPUs" if gpus > 1 else ""


def chart_title(view: RangeView, label: str, width: int | None = None) -> str:
    """``GPU busy (kernel-active %) · last 10m · mean of 8 GPUs``; without the GPUs when that is wider than ``width``."""
    base = f"{label} · {window_label(view.name)}"
    across = view.gpu_label or across_label(view.gpus)
    full = f"{base} · {across}" if across else base
    return full if width is None or len(full) <= width else base


def gpu_view(view: RangeView, gpu_id: int) -> RangeView | None:
    """``view`` narrowed to one GPU: its own chart, stats and row; None when the GPU isn't in it.

    ``avg`` is the GPU's time-weighted mean over the range and ``latest`` its
    newest capture, as in the per-GPU table; ``min`` and ``max`` are its lowest
    and highest chart values (one value per sampling interval in a short range).
    """
    row = next((r for r in view.rows if r.global_gpu_index == gpu_id), None)
    own = view.gpu_series.get(gpu_id)
    if row is None or own is None:
        return None
    busy, memory = own

    def stats(values, avg, latest):
        found = [v for v in values if v is not None]
        return _stats(min(found) if found else None, max(found) if found else None, avg, latest)

    return replace(
        view,
        chart=busy,
        mem_chart=memory,
        rows=(row,),
        busy=stats(busy, row.window_avg_util_pct, row.current_util_pct),
        memory=stats(memory, row.window_avg_mem_pct, row.current_mem_pct),
        gpus=1,
        gpu_series={gpu_id: own},
        gpu_label=f"GPU {gpu_id}",
    )


_STAT_NAMES = ("min", "avg", "max", "latest")
# In place of the stats a sub-job's older samples are needed for: while they download,
# and once a download has ended without them (it failed; the header says why).
LOADING_OLDER = "loading older samples…"
OLDER_NOT_LOADED = "older samples not loaded"


def window_complete(sub_job: SubJobResource, view: RangeView) -> bool:
    """Whether every sample of ``view``'s window has been read (False while older data loads)."""
    if not sub_job.partial:
        return True
    duration = RANGES.get(view.name)
    return duration is not None and sub_job.first_sample_ts is not None and (
        sub_job.first_sample_ts <= view.end - duration
    )


def stats_text(view: RangeView, whole: RangeView, *, width: int, partial: bool = False,
               window_partial: bool = False, missing: str = LOADING_OLDER) -> str:
    """Min / avg / max / latest of busy and memory, for the window and for the whole job.

    ``partial``: the whole job's history is not read yet; ``window_partial``: nor is the window's.
    Their stats read ``missing`` instead (:data:`LOADING_OLDER`, or :data:`OLDER_NOT_LOADED`).
    A min or max that only bounds the true value (see :attr:`SeriesStats.min_bound`) reads ``≤5%`` or ``≥90%``.
    The widest layout that fits ``width`` is used: long labels, short labels, or a row per scope.
    """
    window_partial = window_partial or (partial and view.name == "all")
    scopes = [(window_label(view.name), None if window_partial else (view.busy, view.memory),
               view.gpu_label or view.gpus)]
    if view.name != "all":
        scopes.append(("whole job", None if partial else (whole.busy, whole.memory), whole.gpu_label or whole.gpus))

    def title(name: str, gpus) -> str:
        # Every value is an average across the GPUs (or one GPU's): say so above each scope when it fits.
        across = gpus if isinstance(gpus, str) else across_label(gpus)
        full = f"{name} · {across}"
        return full if across and len(full) <= 28 else name

    def cells(stats: SeriesStats | None) -> str:
        if stats is None:
            return f"{missing:>28}"
        low, avg, high, latest = (format_percent(value) for value in (stats.min, stats.avg, stats.max, stats.latest))
        if stats.min_bound:
            low = "≤" + low
        if stats.max_bound:
            high = "≥" + high
        return "".join(f"{value:>7}" for value in (low, avg, high, latest))

    header = "".join(f"{name:>7}" for name in _STAT_NAMES)
    for labels in ((BUSY_LABEL, MEMORY_LABEL), SHORT_LABELS):
        label_width = max(len(label) for label in labels) + 2
        if width >= label_width + 30 * len(scopes):
            lines = [
                (" " * label_width + "".join(f"  {title(name, gpus):^28}" for name, _, gpus in scopes)).rstrip(),
                " " * label_width + "".join("  " + header for _ in scopes),
            ]
            for index, label in enumerate(labels):
                lines.append(f"{label:<{label_width}}" + "".join(
                    "  " + cells(None if pair is None else pair[index]) for _, pair, _gpus in scopes))
            return "\n".join(lines)
    # Narrow: a row per series and scope, with short labels (the chart titles carry the long ones).
    across = across_label(view.gpus)
    lines = [(across if len(across) <= 16 else "").ljust(16) + header]
    for index, label in enumerate(("busy", "memory")):
        for position, (name, pair, _gpus) in enumerate(scopes):
            lines.append(f"{(label if position == 0 else ''):<7}{name[:9]:<9}"
                         + cells(None if pair is None else pair[index]))
    return "\n".join(lines)
