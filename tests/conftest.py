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

"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_tui_files(request, tmp_path, monkeypatch):
    """Keep TUI tests out of the real TUI cache and home directory.

    The TUI writes its log cache, telemetry, saved consoles and display names
    under CORTEX_TRAINING_TUI_CACHE_DIR, and its error log and saved logs under
    the home directory. A test that needs other values sets them itself.
    """
    if not request.node.path.name.startswith("test_tui_"):
        return
    home = tmp_path / "isolated-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-tui-cache"))
    monkeypatch.setenv("HOME", str(home))
    from cortex_training.tui import log_cache

    # Resolved at import time, so the HOME change above doesn't reach it.
    monkeypatch.setattr(log_cache, "_ERROR_LOG", str(home / ".cortex-training-errors.log"))
