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

"""Tests for the download lock (``cortex_training._download_lock``)."""

import threading
import time

import pytest

from cortex_training._download_lock import DownloadLock


def test_download_lock_serializes_threads(tmp_path):
    root = tmp_path / "stdout"
    first = DownloadLock(root)
    assert first.acquire(lambda: True)
    waits = []
    acquired = threading.Event()

    def contender():
        second = DownloadLock(root)
        if second.acquire(lambda: True, on_wait=lambda: waits.append(1), poll_seconds=0.01):
            acquired.set()
            second.release()

    thread = threading.Thread(target=contender)
    thread.start()
    time.sleep(0.1)
    assert not acquired.is_set()
    assert waits == [1]
    first.release()
    thread.join(timeout=5)
    assert acquired.is_set()


def test_download_lock_gives_up_when_told_to_stop(tmp_path):
    root = tmp_path / "stdout"
    holder = DownloadLock(root)
    assert holder.acquire(lambda: True)
    tries = iter([True, True, False])
    assert DownloadLock(root).acquire(lambda: next(tries), poll_seconds=0.01) is False
    holder.release()
    again = DownloadLock(root)
    assert again.acquire(lambda: True)
    again.release()


def test_download_lock_waits_on_a_file_lock_held_elsewhere(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    root = tmp_path / "stdout"
    lock_path = tmp_path / ".stdout.lock"
    # A separate open file description stands in for another TUI process.
    with open(lock_path, "a+b") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        tries = iter([True] * 5 + [False])
        waits = []
        lock = DownloadLock(root)
        assert lock.acquire(lambda: next(tries), on_wait=lambda: waits.append(1), poll_seconds=0.01) is False
        assert waits == [1]
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    lock = DownloadLock(root)
    assert lock.acquire(lambda: True)
    lock.release()
