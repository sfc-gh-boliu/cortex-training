# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""download_metrics: the sub-job on screen first, and a cold sub-job's newest chunks before its history."""

import gzip
import json
from pathlib import Path
from pathlib import PurePosixPath
from unittest.mock import MagicMock
from urllib.parse import unquote
from urllib.parse import urlparse

import pytest

from cortex_training.client import CortexTrainingClient
from cortex_training.tui.resource import ResourceLoader

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"


class _Connection:
    """Experiment-stage SQL: LIST answers the listed paths, GET writes the gzip body."""

    def __init__(self, listed: dict[str, list[str]], bodies: dict[str, bytes]):
        self.listed = listed
        self.bodies = bodies
        self.gets: list[str] = []

    class _Cursor:
        def __init__(self, connection):
            self.connection = connection
            self.rows = []

        @staticmethod
        def _unquote(value):
            return value[1:-1].replace("''", "'")

        def execute(self, statement):
            if statement.startswith("LIST "):
                uri = self._unquote(statement.removeprefix("LIST "))
                self.rows = [(f"/versions/RUN_ABC/{path}",) for path in self.connection.listed.get(uri, [])]
            else:
                source_literal, target_literal = statement.removeprefix("GET ").split(" ", 1)
                source = self._unquote(source_literal)
                target = Path(unquote(urlparse(self._unquote(target_literal)).path))
                target.mkdir(parents=True, exist_ok=True)
                (target / PurePosixPath(source).name).write_bytes(gzip.compress(self.connection.bodies[source]))
                self.connection.gets.append(source.removeprefix(RUN_URI))
                self.rows = [(source, "DOWNLOADED")]
            return self

        def fetchall(self):
            return self.rows

        def close(self):
            pass

    def cursor(self):
        return self._Cursor(self)

    def close(self):
        pass


def _chunk(sub, minute):
    return f"_metrics/{sub}/gpu.20261015-10{minute:02d}00.c{minute:02d}.gz"


def _client(monkeypatch, chunks_by_sub: dict[str, list[str]]):
    bodies = {RUN_URI + path: f"{path}\n".encode() for paths in chunks_by_sub.values() for path in paths}
    connection = _Connection({RUN_URI + "_metrics/": [p for paths in chunks_by_sub.values() for p in paths],
                              RUN_URI + "checkpoints/_metrics/": []}, bodies)
    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    client._session = MagicMock()
    monkeypatch.setattr(client, "_experiment_run_uri", lambda _job: (RUN_URI, "RUN_ABC"))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: connection)
    return client, connection


def _manifest(path: Path) -> dict:
    return json.loads(path.with_name("." + path.name + ".manifest.json").read_text())


def test_priority_downloads_the_named_sub_job_first(tmp_path, monkeypatch):
    sampling = [_chunk("job-1:sampling:0", m) for m in range(3)]
    training = [_chunk("job-1:training:0", m) for m in range(3)]
    client, connection = _client(monkeypatch, {"s": sampling, "t": training})
    committed = []

    client.download_metrics("job-1", tmp_path, priority=["job-1:training:0"],
                            on_commit=lambda result: committed.append((result["sub_job_id"], result["complete"])))

    assert connection.gets == training + sampling  # id order alone would fetch sampling first
    assert committed == [("job-1:training:0", True), ("job-1:sampling:0", True)]
    connection.gets.clear()
    client.download_metrics("job-1", tmp_path / "again", priority="job-1:training:0")  # one id as a string
    assert connection.gets == training + sampling


def test_a_cold_sub_job_commits_its_newest_chunks_before_its_history(tmp_path, monkeypatch):
    sub = "job-1:training:0"
    chunks = [_chunk(sub, m) for m in range(0, 60, 2)]  # 30 chunks, one every 2 minutes
    client, connection = _client(monkeypatch, {"t": chunks})
    path = tmp_path / sub / "gpu.jsonl"
    seen = []

    def on_commit(result):
        manifest = _manifest(path)
        seen.append((result["complete"], result["chunk_count"], manifest.get("partial", False),
                     path.read_text().splitlines()))

    results = client.download_metrics("job-1", tmp_path, resume=True, newest_first_seconds=600,
                                      on_commit=on_commit)

    # The chunks named within 10 minutes of the newest (10:48 to 10:58), and the one before them.
    newest = chunks[-7:]
    assert seen[0] == (False, len(newest), True, newest)
    assert seen[1] == (True, len(chunks), False, chunks)
    assert path.read_text().splitlines() == chunks
    assert "partial" not in _manifest(path)
    assert sorted(connection.gets) == sorted(chunks)  # every chunk fetched once: the newest ones were reused
    assert connection.gets[: len(newest)] == newest
    assert results == [{
        "sub_job_id": sub, "filename": "gpu.jsonl", "saved_path": str(path), "chunk_count": len(chunks),
        "first_artifact_uri": RUN_URI + chunks[0], "last_artifact_uri": RUN_URI + chunks[-1],
    }]


def test_a_short_sub_job_or_a_warm_cache_is_downloaded_in_one_commit(tmp_path, monkeypatch):
    sub = "job-1:training:0"
    chunks = [_chunk(sub, m) for m in range(4)]
    client, connection = _client(monkeypatch, {"t": chunks})
    commits = []
    client.download_metrics("job-1", tmp_path, resume=True, newest_first_seconds=600,
                            on_commit=lambda result: commits.append(result["complete"]))
    assert commits == [True]  # every chunk is inside the window: nothing to put first

    longer = chunks + [_chunk(sub, m) for m in range(20, 50)]
    connection.listed[RUN_URI + "_metrics/"] = longer
    connection.bodies.update({RUN_URI + p: f"{p}\n".encode() for p in longer})
    connection.gets.clear()
    commits.clear()
    client.download_metrics("job-1", tmp_path, resume=True, newest_first_seconds=60,
                            on_commit=lambda result: commits.append(result["complete"]))
    assert commits == [True]  # a cached prefix is appended to, not reread newest-first
    assert connection.gets == longer[4:]
    assert (tmp_path / sub / "gpu.jsonl").read_text().splitlines() == longer


def test_a_partial_file_left_by_an_interrupted_download_is_completed_from_its_bytes(tmp_path, monkeypatch):
    sub = "job-1:training:0"
    chunks = [_chunk(sub, m) for m in range(0, 40, 2)]
    client, connection = _client(monkeypatch, {"t": chunks})

    class Stop(Exception):
        pass

    def stop_after_the_newest(result):
        if not result["complete"]:
            raise Stop

    with pytest.raises(Stop):
        client.download_metrics("job-1", tmp_path, resume=True, newest_first_seconds=300,
                                on_commit=stop_after_the_newest)
    path = tmp_path / sub / "gpu.jsonl"
    newest = path.read_text().splitlines()
    assert _manifest(path)["partial"] is True and newest == chunks[-4:]

    connection.gets.clear()
    client.download_metrics("job-1", tmp_path, resume=True)
    assert connection.gets == chunks[:-4]  # only the history it was missing
    assert path.read_text().splitlines() == chunks
    assert "partial" not in _manifest(path)


def test_a_chunk_that_sorts_before_the_cached_ones_reuses_their_bytes(tmp_path, monkeypatch):
    sub = "job-1:training:0"
    chunks = [_chunk(sub, m) for m in range(10, 20)]
    client, connection = _client(monkeypatch, {"t": chunks})
    client.download_metrics("job-1", tmp_path, resume=True)
    late = _chunk(sub, 5)
    connection.listed[RUN_URI + "_metrics/"] = [late, *chunks]
    connection.bodies[RUN_URI + late] = f"{late}\n".encode()
    connection.gets.clear()

    client.download_metrics("job-1", tmp_path, resume=True)

    assert connection.gets == [late]
    assert (tmp_path / sub / "gpu.jsonl").read_text().splitlines() == [late, *chunks]


def _gpu_rows(sub, start_minute, minutes):
    rows = []
    for second in range(start_minute * 60, (start_minute + minutes) * 60, 5):
        rows.append(json.dumps({
            "record_version": 2, "sub_job_id": sub, "global_gpu_index": 0,
            "sample_ts": f"2026-01-15T10:{second // 60:02d}:{second % 60:02d}Z",
            "gpu_util_pct": 50.0, "fb_used_bytes": 1.0, "fb_free_bytes": 3.0, "sample_interval_s": 5.0,
        }))
    return "\n".join(rows) + "\n"


def test_the_loader_marks_a_file_that_holds_only_the_newest_chunks_as_partial(tmp_path):
    sub = "job-1:training:0"
    path = tmp_path / sub / "gpu.jsonl"
    path.parent.mkdir(parents=True)
    manifest = path.with_name(".gpu.jsonl.manifest.json")
    path.write_text(_gpu_rows(sub, 40, 20))
    manifest.write_text(json.dumps({"version": 1, "names": ["newest"], "bytes": path.stat().st_size,
                                    "partial": True}))
    loader = ResourceLoader(tmp_path)

    first = loader.load().sub_jobs[sub]
    assert first.partial  # the whole job is still loading: its stats must not be shown as complete
    assert first.views["10m"].busy.avg == pytest.approx(50.0)

    path.write_text(_gpu_rows(sub, 0, 60))
    manifest.write_text(json.dumps({"version": 1, "names": ["older", "newest"], "bytes": path.stat().st_size}))
    complete = loader.load().sub_jobs[sub]
    assert not complete.partial
    assert complete.first_sample_ts.minute == 0
