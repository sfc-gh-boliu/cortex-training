# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Resources tab's incremental telemetry loader (``ResourceLoader``).

A refresh must read only the bytes ``download_metrics`` appended, fall back to a
full read whenever the file was rebuilt or rewritten, and give the same result
as reading everything again.
"""

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cortex_training.tui.resource as resource
from cortex_training.tui.resource import RANGES
from cortex_training.tui.resource import ResourceLoader
from cortex_training.tui.resource import load_resource_data
from cortex_training.tui.resource import resource_loader
from cortex_training.tui.resource import GPU
from tests.resource_legacy import gpu_jobs

SUB = "job:training:0"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "tui-cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _row(seconds: float, gpu: int, util, *, used=40.0, free=60.0, sub_job=SUB, interval=5.0) -> dict:
    return {
        "record_version": 2,
        "sub_job_id": sub_job,
        "sample_ts": (START + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z"),
        "worker_num": gpu // 4,
        "local_gpu_index": gpu % 4,
        "global_gpu_index": gpu,
        "gpu_util_pct": util,
        "fb_used_bytes": used,
        "fb_free_bytes": free,
        "sm_active_pct": None,
        "tensor_active_pct": None if util is None else util / 2,
        "dram_active_pct": None,
        "sample_interval_s": interval,
    }


def _lines(rows) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _manifest(path: Path, names, size=None) -> None:
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": list(names), "bytes": path.stat().st_size if size is None else size})
    )


def _write(root: Path, rows, *, sub_job=SUB, names=("c0",)) -> Path:
    path = root / sub_job / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_lines(rows))
    _manifest(path, names)
    return path


def _append(path: Path, rows, names) -> int:
    """Append rows the way download_metrics does: same file, a longer chunk list."""
    payload = _lines(rows)
    with path.open("ab") as handle:
        handle.write(payload)
    _manifest(path, names)
    return len(payload)


def _series(seconds: int, gpus: int = 2, step: float = 5.0, offset: float = 0.0, rng=None):
    rows = []
    t = offset
    while t < offset + seconds:
        for gpu in range(gpus):
            util = 50 + 40 * math.sin(t / 300 + gpu) + (rng.uniform(-5, 5) if rng else 0)
            rows.append(_row(t, gpu, round(max(0.0, min(100.0, util)), 2), used=30.0 + gpu, free=70.0 - gpu))
        t += step
    return rows


def _same(a, b) -> None:
    """Two snapshots describe the same data."""
    assert set(gpu_jobs(a)) == set(gpu_jobs(b))
    assert a.ignored_rows_by_sub_job == b.ignored_rows_by_sub_job
    for sub_job_id, left in gpu_jobs(a).items():
        right = gpu_jobs(b)[sub_job_id]
        assert left.latest_sample_ts == right.latest_sample_ts
        assert left.first_sample_ts == right.first_sample_ts
        assert left.gpu_ids == right.gpu_ids and left.reporting_gpu_ids == right.reporting_gpu_ids
        assert left.sample_interval_s == pytest.approx(right.sample_interval_s)
        for name in RANGES:
            x, y = left.views[name], right.views[name]
            assert (x.start, x.end) == (y.start, y.end), name
            assert x.chart == pytest.approx(y.chart, nan_ok=True), name
            assert x.mem_chart == pytest.approx(y.mem_chart, nan_ok=True), name
            for field in ("busy", "memory"):
                assert getattr(x, field) == pytest.approx(getattr(y, field)), (name, field)
            assert x.rows == y.rows, name


def test_second_load_reads_only_the_appended_bytes(tmp_path):
    path = _write(tmp_path, _series(1800))
    loader = ResourceLoader(tmp_path)
    first = loader.load()
    assert loader.bytes_read == path.stat().st_size
    rows_before, bytes_before = loader.documents_read, loader.bytes_read

    new_rows = _series(30, offset=1800)
    appended = _append(path, new_rows, ["c0", "c1"])
    second = loader.load()

    assert loader.bytes_read - bytes_before == appended
    assert loader.documents_read - rows_before == len(new_rows)
    assert gpu_jobs(second)[SUB].latest_sample_ts > gpu_jobs(first)[SUB].latest_sample_ts
    _same(second, load_resource_data(tmp_path))


def test_a_load_with_nothing_new_reads_nothing_and_reuses_the_snapshot(tmp_path):
    _write(tmp_path, _series(600))
    loader = ResourceLoader(tmp_path)
    first = loader.load()
    read = loader.bytes_read
    second = loader.load()
    assert loader.bytes_read == read
    assert second.kind(SUB, GPU) is first.kind(SUB, GPU)  # the snapshot is reused


def test_incremental_loads_match_a_full_reload(tmp_path):
    """Many appends of different sizes, with two workers' rows interleaving late."""
    rng = random.Random(7)
    path = _write(tmp_path, _series(300, gpus=4, rng=rng))
    loader = ResourceLoader(tmp_path)
    loader.load()
    names = ["c0"]
    t = 300.0
    for index in range(1, 25):
        span = rng.choice([5, 30, 60, 600])
        rows = _series(span, gpus=4, offset=t, rng=rng)
        # Worker 1's GPUs (2, 3) arrive one chunk late, as separate uploads do.
        early = [row for row in rows if row["global_gpu_index"] < 2]
        late = [row for row in rows if row["global_gpu_index"] >= 2]
        names.append(f"c{index}a")
        _append(path, early, names)
        _same(loader.load(), load_resource_data(tmp_path))
        names.append(f"c{index}b")
        _append(path, late, names)
        _same(loader.load(), load_resource_data(tmp_path))
        t += span


@pytest.mark.parametrize("change", ["replaced", "names-not-a-prefix", "rewritten-in-place", "truncated"])
def test_a_rebuilt_or_rewritten_file_is_read_again_from_the_start(tmp_path, change):
    path = _write(tmp_path, _series(900))
    loader = ResourceLoader(tmp_path)
    loader.load()
    other = _series(900, offset=5000)  # different data, about the same size
    if change == "replaced":  # a rebuild: a new file moved into place (new inode)
        temporary = path.with_name(".gpu-metrics-x.tmp")
        temporary.write_bytes(_lines(other))
        os.replace(temporary, path)
        _manifest(path, ["c0"])
    elif change == "names-not-a-prefix":
        path.write_bytes(_lines(_series(900)) + _lines(_series(60, offset=900)))
        _manifest(path, ["b0", "c0"])
    elif change == "rewritten-in-place":  # same inode, longer chunk list, but other bytes before the offset
        payload = _lines(_series(1200, offset=5000))
        assert len(payload) > loader._files[SUB].offset
        with path.open("r+b") as handle:
            handle.write(payload)
            handle.truncate(len(payload))
        _manifest(path, ["c0", "c1"])
    else:
        path.write_bytes(_lines(_series(300)))
        _manifest(path, ["c0", "c1"])
    read = loader.bytes_read
    reloaded = loader.load()
    assert loader.bytes_read - read == path.stat().st_size  # all of it, from the start
    _same(reloaded, load_resource_data(tmp_path))


def test_manifest_changes_drop_and_restore_a_sub_job(tmp_path):
    path = _write(tmp_path, _series(300))
    loader = ResourceLoader(tmp_path)
    assert SUB in gpu_jobs(loader.load())
    manifest = path.with_name(".gpu.jsonl.manifest.json")
    saved = manifest.read_text()
    manifest.unlink()
    assert gpu_jobs(loader.load()) == {}
    manifest.write_text(saved)
    _same(loader.load(), load_resource_data(tmp_path))


def test_an_unterminated_committed_line_is_read_once_it_is_complete(tmp_path):
    rows = _series(60)
    path = tmp_path / SUB / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    whole = _lines(rows)
    cut = len(whole) - 20  # the last row is torn
    path.write_bytes(whole[:cut])
    _manifest(path, ["c0"])
    loader = ResourceLoader(tmp_path)
    data = loader.load()
    assert data.ignored_rows == 0
    assert loader.documents_read == len(rows) - 1
    with path.open("ab") as handle:  # the rest of that row, then one more chunk
        handle.write(whole[cut:] + _lines(_series(10, offset=60)))
    _manifest(path, ["c0", "c1"])
    data = loader.load()
    assert data.ignored_rows == 0
    assert loader.documents_read == len(rows) + len(_series(10, offset=60))
    _same(data, load_resource_data(tmp_path))


def test_first_load_of_the_selected_sub_job_publishes_its_newest_data_first(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    path = _write(tmp_path, _series(3 * 3600, gpus=2))
    _write(tmp_path, _series(600, gpus=1, offset=3 * 3600 - 600), sub_job="job:sampling:0")
    for row_file in (tmp_path / "job:sampling:0" / "gpu.jsonl",):
        rows = [json.loads(line) for line in row_file.read_text().splitlines()]
        row_file.write_text("".join(json.dumps({**row, "sub_job_id": "job:sampling:0"}) + "\n" for row in rows))
        _manifest(row_file, ["c0"])
    loader = ResourceLoader(tmp_path)
    partials = []

    def on_partial(data):
        partials.append((data, loader.bytes_read))

    final = loader.load(priority=SUB, window=timedelta(minutes=10), on_partial=on_partial)
    assert len(partials) == 1
    partial, read_by_then = partials[0]
    assert read_by_then < path.stat().st_size / 4  # only the newest bytes were read
    assert set(gpu_jobs(partial)) == {SUB}  # the selected sub-job comes before the others
    early = gpu_jobs(partial)[SUB]
    assert early.partial and not gpu_jobs(final)[SUB].partial
    assert early.first_sample_ts > gpu_jobs(final)[SUB].first_sample_ts
    # The window on screen was complete in the partial snapshot.
    assert early.views["10m"].chart == gpu_jobs(final)[SUB].views["10m"].chart
    assert early.views["10m"].busy == gpu_jobs(final)[SUB].views["10m"].busy
    _same(final, load_resource_data(tmp_path))


def test_a_job_longer_than_the_raw_retention_bins_the_rest(tmp_path):
    hours = 9
    rows = [_row(t, 0, 20.0 if (t // 3600) % 2 else 80.0) for t in range(0, hours * 3600, 30)]
    for row in rows:
        row["sample_interval_s"] = 30.0
    _write(tmp_path, rows)
    loader = ResourceLoader(tmp_path)
    data = loader.load()
    series = loader._files[SUB].series
    sub_job = gpu_jobs(data)[SUB]
    newest = series.latest
    # Older samples are only binned (a short stretch stays as the next fold's context).
    assert min(series.gpus[0].ts) >= newest - resource._RAW_RETENTION_S - resource._FOLD_CONTEXT_S
    whole = sub_job.views["all"]
    assert whole.start == sub_job.first_sample_ts and len(whole.chart) <= resource._MAX_BINS
    assert whole.busy.avg == pytest.approx(sum(r["gpu_util_pct"] for r in rows) / len(rows))
    assert (whole.busy.min, whole.busy.max) == (pytest.approx(20.0), pytest.approx(80.0))
    six_hours = sub_job.views["6h"]
    in_window = [r["gpu_util_pct"] for r in rows if (START + timedelta(seconds=newest - START.timestamp())
                                                     - datetime.fromisoformat(r["sample_ts"].replace("Z", "+00:00"))
                                                     ) <= timedelta(hours=6)]
    assert six_hours.busy.avg == pytest.approx(sum(in_window) / len(in_window))


def test_cache_directories_share_one_loader(tmp_path):
    first = resource_loader(tmp_path / "a")
    assert resource_loader(tmp_path / "a" / ".." / "a") is first
    for name in "bcde":
        resource_loader(tmp_path / name)
    assert resource_loader(tmp_path / "a") is not first  # only the most recent few are kept


def test_a_file_that_starts_tiny_still_grows_incrementally(tmp_path):
    path = tmp_path / SUB / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\n")  # shorter than a fingerprint
    _manifest(path, ["c0"])
    loader = ResourceLoader(tmp_path)
    loader.load()
    for index, offset in enumerate((0, 30, 60), start=1):
        read = loader.bytes_read
        appended = _append(path, _series(30, offset=offset), [f"c{i}" for i in range(index + 1)])
        data = loader.load()
        assert loader.bytes_read - read == appended
    _same(data, load_resource_data(tmp_path))


def test_a_load_that_fails_part_way_leaves_nothing_half_read(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "_NEWEST_FIRST_MIN_BYTES", 16 * 1024)
    monkeypatch.setattr(resource, "_NEWEST_FIRST_STEP", 8 * 1024)
    _write(tmp_path, _series(3 * 3600, gpus=2))
    loader = ResourceLoader(tmp_path)

    def fail(_data):
        raise RuntimeError("the screen went away")

    with pytest.raises(RuntimeError):
        loader.load(priority=SUB, window=timedelta(minutes=10), on_partial=fail)
    # The rows read before the failure must not be counted again on top of a re-read.
    _same(loader.load(), load_resource_data(tmp_path))


def test_a_file_that_ends_in_rows_a_load_skips_keeps_every_gpu_sample_in_the_windows(tmp_path):
    """GPU rows for 8 hours, then 3 hours of system rows only: the GPU sampler stopped."""
    gpu_rows = [_row(t, 0, float(t // 5 % 100)) for t in range(0, 8 * 3600, 5)]
    system = [{"record_version": 2, "kind": "system", "sub_job_id": SUB, "cpu_util_pct": 10.0,
               "sample_ts": (START + timedelta(seconds=t)).isoformat().replace("+00:00", "Z")}
              for t in range(8 * 3600, 11 * 3600, 5)]
    _write(tmp_path, gpu_rows + system)
    sub_job = gpu_jobs(load_resource_data(tmp_path))[SUB]
    newest = 8 * 3600 - 5
    for name in ("1h", "6h"):
        seconds = RANGES[name].total_seconds()
        expected = [row["gpu_util_pct"] for row in gpu_rows
                    if newest - seconds <= (datetime.fromisoformat(row["sample_ts"].replace("Z", "+00:00"))
                                            - START).total_seconds()]
        view = sub_job.views[name]
        assert view.busy.avg == pytest.approx(sum(expected) / len(expected)), name
        assert all(value is not None for value in view.chart), name


def test_incremental_loads_across_the_raw_retention_match_a_full_reload(tmp_path):
    rng = random.Random(11)
    path = _write(tmp_path, _series(int(6.5 * 3600), gpus=2, rng=rng))
    loader = ResourceLoader(tmp_path)
    loader.load()
    names = ["c0"]
    t = 6.5 * 3600
    while t < 8.5 * 3600:
        rows = _series(900, gpus=2, offset=t, rng=rng)
        names.append(f"c{len(names)}")
        _append(path, rows, names)
        _same(loader.load(), load_resource_data(tmp_path))
        t += 900
    whole = gpu_jobs(loader.load())[SUB].views["all"]
    # Part of the job is older than the raw samples; its intervals' averages across
    # GPUs were folded in before they left, so min and max are exact, not bounds.
    assert not (whole.busy.min_bound or whole.busy.max_bound)
    by_time: dict = {}
    for row in [json.loads(line) for line in path.read_bytes().splitlines()]:
        by_time.setdefault(row["sample_ts"], []).append(row["gpu_util_pct"])
    averages = [sum(values) / len(values) for values in by_time.values() if len(values) == 2]
    assert (whole.busy.min, whole.busy.max) == (pytest.approx(min(averages)), pytest.approx(max(averages)))
