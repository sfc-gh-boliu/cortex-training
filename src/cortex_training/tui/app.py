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

"""Read-only Textual TUI for Cortex Training jobs.

Flow: open → every job from the last six months (live first, then history,
with sub-job rows; searchable by name, id, or training config) → a job preview
→ tabbed details (Metrics | Logs | Resources | Checkpoints) for the job or one
sub-job. Pass a job_id on the command line to jump straight to its Logs tab.

Imports ``textual`` at module load. The interface never mutates job state; the
only writes are local (display names, log and telemetry caches).
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from collections import deque
from datetime import datetime
from datetime import timezone
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import ListItem
from textual.widgets import ListView
from textual.widgets import Log
from textual.widgets import Static
from textual.worker import get_current_worker

from cortex_training.tui import views
from cortex_training.tui.format import entry_at_level
from cortex_training.tui.format import entry_matches
from cortex_training.tui.format import format_created_at
from cortex_training.tui.format import format_job_config
from cortex_training.tui.format import format_log_entry
from cortex_training.tui.format import is_active_status
from cortex_training.tui.format import source_label
from cortex_training.tui.format import wrap_log_line
from cortex_training.tui.log_cache import LogCache
from cortex_training.tui.log_cache import cached_log_pages

_ERROR_LOG = "~/.cortex-training-errors.log"


class JobListScreen(Screen):
    """Every job in the retention window: live first, then history, with sub-job rows.

    Search matches display names and partial ids (case-insensitive) and takes
    ``field<op>value`` filters over the training configuration; see
    :func:`cortex_training.tui.views.parse_query`.
    """

    BINDINGS = [
        ("/", "filter", "Search"),
        ("escape", "clear_filter", "Clear search"),
        ("s", "cycle_sort", "Sort"),
        ("o", "toggle_order", "Order"),
        ("space", "toggle_expand", "Sub-jobs"),
        ("e", "expand_all", "Expand all"),
        ("n", "rename", "Name job"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]
    _REFRESH_SECONDS = 20
    _COLUMNS = [(" ", 1), ("Name", 34), ("Job ID", 12), ("Type", 12), ("Status", 11), ("Model", 20),
                ("GPUs", 9), ("Created", 16), ("Updated", 16), ("Owner", 12)]

    def __init__(self, client, *, sub_job_id=None, poll_interval=1.0, aliases=None, now=None):
        super().__init__()
        from cortex_training.tui.stores import AliasStore

        self._client = client
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._aliases = aliases or AliasStore()
        self._now = now  # fixed clock for tests/screenshots; None = wall clock
        self._job_by_item: dict[str, str] = {}
        self._status_by_item: dict[str, str] = {}
        self._jobs_by_id: dict[str, dict] = {}
        self._sub_by_item: dict[str, tuple] = {}
        self._all_jobs: list = []
        self._filter = ""
        self._sort = "created"
        self._descending = True
        self._expanded: set = set()
        self._collapsed: set = set()  # live jobs the user folded
        self._last_refreshed = None
        self._refreshing = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="listinfo")
        yield DataTable(id="jobs", cursor_type="row", zebra_stripes=False)
        yield Static("", id="nomatch")
        yield Input(
            placeholder="search names / ids…  or fields: model:qwen lr<1e-4 gpus>=8 method:lora gpu:H200  (/ focus · Esc clear)",
            id="jobfilter",
        )
        yield Static(
            "[dim]fields:[/dim] " + " ".join(views.SEARCH_FIELDS)
            + "   [dim]ops:[/dim] :  =  !=  >  >=  <  <=   [dim]any config member:[/dim] training_config.optimizer.lr<1e-4",
            id="searchhelp",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#jobs", DataTable)
        for label, width in self._COLUMNS:
            table.add_column(label, width=width)
        self.query_one("#nomatch", Static).display = False
        self._refresh_jobs()
        # Gentle auto-refresh so status changes appear without pressing r.
        self.set_interval(self._REFRESH_SECONDS, self._auto_refresh)

    def on_screen_resume(self) -> None:
        # Detail screens own the same app-level subtitle while they are active.
        # Restore the list status (and any renamed job) on return.
        self._update_subtitle()
        self._render_jobs()

    def _clock(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    def _update_subtitle(self) -> None:
        if self._refreshing:
            refresh_status = "refreshing…"
            if self._last_refreshed:
                refresh_status += f" (last refreshed {self._last_refreshed})"
        else:
            refresh_status = f"last refreshed {self._last_refreshed or '—'}"
        self.app.sub_title = (
            "select a job (↑/↓, Enter · / search · s sort · space sub-jobs · r refresh)"
            f"  ·  {refresh_status}"
        )

    def _refresh_jobs(self) -> None:
        self._refreshing = True
        self._update_subtitle()
        self._load_jobs()

    def _auto_refresh(self) -> None:
        # Only refetch while the list is the active screen — don't poll in the
        # background while the user is down in a job.
        if self.app.screen is self:
            self._refresh_jobs()

    def action_refresh(self) -> None:
        self._refresh_jobs()

    def action_filter(self) -> None:
        self.set_focus(self.query_one("#jobfilter", Input))

    def action_clear_filter(self) -> None:
        inp = self.query_one("#jobfilter", Input)
        if self.focused is inp or self._filter:
            inp.value = ""  # fires on_input_changed → clears the filter
            self.set_focus(self.query_one("#jobs", DataTable))

    def action_cycle_sort(self) -> None:
        keys = views.SORT_KEYS
        self._sort = keys[(keys.index(self._sort) + 1) % len(keys)]
        self._render_jobs()

    def action_toggle_order(self) -> None:
        self._descending = not self._descending
        self._render_jobs()

    def _highlighted(self) -> tuple:
        table = self.query_one("#jobs", DataTable)
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            return None, None
        if key in self._job_by_item:
            return key, self._job_by_item[key]
        if key in self._sub_by_item:
            return key, self._sub_by_item[key][0]
        return key, None

    def _is_expanded(self, job: dict) -> bool:
        jid = views.job_id_of(job)
        if jid in self._collapsed:
            return False
        return jid in self._expanded or views.is_live(job)

    def action_toggle_expand(self) -> None:
        _key, jid = self._highlighted()
        job = self._jobs_by_id.get(jid or "")
        if not job:
            return
        if self._is_expanded(job):
            self._expanded.discard(jid)
            self._collapsed.add(jid)
        else:
            self._collapsed.discard(jid)
            self._expanded.add(jid)
        self._render_jobs(keep=jid)

    def action_expand_all(self) -> None:
        if self._expanded:
            self._expanded.clear()
            self._collapsed = {views.job_id_of(j) for j in self._all_jobs if views.is_live(j)}
        else:
            self._collapsed.clear()
            self._expanded = {views.job_id_of(j) for j in self._all_jobs}
        self._render_jobs()

    def action_rename(self) -> None:
        from cortex_training.tui.screens import PromptScreen

        _key, jid = self._highlighted()
        if not jid:
            return
        current = views.display_name(self._jobs_by_id.get(jid, {}), self._aliases.all())

        def done(value):
            if value is not None:
                self._aliases.set(jid, value)
                self._render_jobs(keep=jid)

        self.app.push_screen(PromptScreen(f"Display name for {jid}", current, "e.g. qwen3-8b lora sweep #3"), done)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "jobfilter":
            self._filter = event.value.strip()
            self._render_jobs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "jobfilter":
            self.set_focus(self.query_one("#jobs", DataTable))

    @work(thread=True, exclusive=True, group="jobs")
    def _load_jobs(self) -> None:
        worker = get_current_worker()
        try:
            jobs = self._client.list_jobs()
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.app.call_from_thread(self._on_jobs, None, f"list_jobs failed: {type(exc).__name__}: {exc}")
            return
        if worker.is_cancelled:
            return
        self.app.call_from_thread(self._on_jobs, jobs, None)

    def _on_jobs(self, jobs, error) -> None:
        self._refreshing = False
        if error:
            self._update_subtitle()
            self._all_jobs = []
            self._render_jobs(error=error)
            return
        self._all_jobs = [j for j in (jobs or []) if isinstance(j, dict) and views.job_id_of(j)]
        self._jobs_by_id = {views.job_id_of(j): j for j in self._all_jobs}
        self._last_refreshed = time.strftime("%H:%M:%S")
        self._update_subtitle()
        self._render_jobs()

    def _job_row(self, job: dict, name: str) -> list:
        live = views.is_live(job)
        word = views.status_word(job.get("status"))
        status_style = {"running": "bold green", "failed": "bold red"}.get(word, "yellow" if live else "grey62")
        jid = views.job_id_of(job)
        models = views.job_models(job)
        model = models[0].split("/")[-1] if models else "—"
        if len(models) > 1:
            model += f" +{len(models) - 1}"
        marker = Text("●", style="bold green") if live else Text("○", style="grey50")
        name_cell = Text(name, style="bold") if name else Text("—", style="grey50")
        if views.sub_jobs(job):
            name_cell = Text.assemble(("▾ " if self._is_expanded(job) else "▸ ", "grey50"), name_cell)
        return [
            marker,
            name_cell,
            jid[:8] + "…" if len(jid) > 9 else jid,
            views.job_type_label(job),
            Text(word.upper(), style=status_style),
            model,
            f"{views.total_gpus(job) or '—'}" + (f" {job['hardware']}" if job.get("hardware") else ""),
            format_created_at(job.get("created_at")) or "—",
            format_created_at(job.get("updated_at")) or "—",
            str(job.get("submitted_by") or job.get("owner_role") or "—"),
        ]

    @staticmethod
    def _sub_row(sj: dict, last: bool) -> list:
        word = views.status_word(sj.get("status"))
        style = {"running": "green", "failed": "red"}.get(word, "grey62")
        branch = "└─" if last else "├─"
        return [
            "",
            Text(f"   {branch} {views.sub_job_label(sj)}", style="grey70"),
            Text(":" + ":".join(str(sj.get("sub_job_id") or "").split(":")[1:]), style="grey50"),
            Text(str(sj.get("job_type") or "").lower(), style="grey70"),
            Text(word.upper(), style=style),
            Text(str(sj.get("model_name") or "—").split("/")[-1], style="grey70"),
            Text(str(views.sub_job_gpus(sj) or "—"), style="grey70"),
            "",
            "",
            "",
        ]

    def _render_jobs(self, *, keep: str | None = None, error: str | None = None) -> None:
        table = self.query_one("#jobs", DataTable)
        nomatch = self.query_one("#nomatch", Static)
        if keep is None:
            _k, keep = self._highlighted()
        table.clear()
        self._job_by_item = {}
        self._status_by_item = {}
        self._sub_by_item = {}
        if error:
            self.query_one("#listinfo", Static).update(f"[red][error] {escape(error)}[/red]")
            nomatch.display = False
            return
        now = self._clock()
        aliases = self._aliases.all()
        recent = [j for j in self._all_jobs if views.within_retention(j, now)]
        hidden = len(self._all_jobs) - len(recent)
        query = views.parse_query(self._filter)
        names = {views.job_id_of(j): views.display_name(j, aliases) for j in recent}
        visible = [j for j in recent if views.job_matches_query(j, query, names[views.job_id_of(j)])]
        ordered = views.sort_job_list(visible, self._sort, self._descending)
        live = [j for j in ordered if views.is_live(j)]
        history = [j for j in ordered if not views.is_live(j)]

        arrow = "↓" if self._descending else "↑"
        info = (
            f"  [bold green]● {len(live)} live[/]   [grey62]○ {len(history)} historical[/]"
            f"   [dim]·  last 6 months ({len(recent)} jobs"
            + (f", {hidden} older hidden" if hidden else "")
            + f")  ·  sort: [/dim][b]{self._sort} {arrow}[/b]"
        )
        if self._filter:
            info += f"   [dim]·  search:[/dim] [b]{escape(self._filter)}[/b] → {len(visible)} match(es)"
        self.query_one("#listinfo", Static).update(info)

        if not visible:
            if self._filter:
                nomatch.update(views.no_match_message(self._filter, query, len(recent)))
            else:
                nomatch.update("No jobs in the last 6 months for this role.")
            nomatch.display = True
            table.display = False
            return
        nomatch.display = False
        table.display = True

        idx = 0
        target_row = None
        for group, jobs in (("LIVE", live), ("HISTORY", history)):
            if not jobs:
                continue
            style = "bold green" if group == "LIVE" else "bold grey62"
            label = "running or queued" if group == "LIVE" else "ended or failed"
            table.add_row("", Text.assemble((f"{group} ({len(jobs)})  ", style), (label, "dim")),
                          *[""] * 8, key=f"hdr-{group.lower()}")
            for job in jobs:
                jid = views.job_id_of(job)
                item_id = f"job-{idx}"
                idx += 1
                self._job_by_item[item_id] = jid
                self._status_by_item[item_id] = job.get("status")
                table.add_row(*self._job_row(job, names[jid]), key=item_id)
                if jid == keep and target_row is None:
                    target_row = table.row_count - 1
                if self._is_expanded(job):
                    subs = views.sub_jobs(job)
                    for si, sj in enumerate(subs):
                        sub_key = f"sub-{item_id}-{si}"
                        self._sub_by_item[sub_key] = (jid, sj.get("sub_job_id"))
                        table.add_row(*self._sub_row(sj, si == len(subs) - 1), key=sub_key)
        if target_row is None:
            target_row = 1  # first job row (row 0 is a group header)
        table.move_cursor(row=min(target_row, table.row_count - 1), animate=False)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from cortex_training.tui.screens import JobDetailScreen
        from cortex_training.tui.screens import JobSummaryScreen

        key = event.row_key.value
        if key in self._job_by_item:
            jid = self._job_by_item[key]
            self.app.push_screen(
                JobSummaryScreen(self._client, jid, job=self._jobs_by_id.get(jid), aliases=self._aliases,
                                 poll_interval=self._poll_interval)
            )
        elif key in self._sub_by_item:
            jid, sid = self._sub_by_item[key]
            self.app.push_screen(
                JobDetailScreen(self._client, jid, job=self._jobs_by_id.get(jid), sub_job_id=sid,
                                initial_tab="tab-metrics", aliases=self._aliases, poll_interval=self._poll_interval)
            )


class LogScreen(Screen):
    """Source list + live tail for one job. Read-only."""

    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
        ("escape", "dismiss_or_back", "Back/clear"),
        ("/", "filter", "Filter"),
        ("L", "cycle_level", "Level"),
        ("p", "toggle_pause", "Pause"),
        ("s", "save_log", "Save"),
        ("y", "copy_log", "Copy all"),
        ("c", "copy_text", "Copy sel"),
        ("r", "refresh_sources", "Refresh"),
        ("[", "shrink_sources", "Narrower"),
        ("]", "grow_sources", "Wider"),
        ("q", "quit", "Quit"),
    ]

    _SOURCES_MIN = 24
    _SOURCES_MAX = 140
    _SOURCES_STEP = 8
    _LEVEL_CYCLE = [None, "INFO", "WARNING", "ERROR"]

    def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0, job_status=None):
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._job_status = job_status
        self._job_reason = None  # failure/cancel reason, when terminal
        self._sources_width = 48
        self._cache = LogCache(job_id)
        self._source_by_item: dict[str, str] = {}
        self._logview: Log | None = None
        self._filter = ""  # grep substring for the log pane
        self._min_level = None  # None | INFO | WARNING | ERROR
        self._paused = False  # auto-scroll paused?
        self._current_source = None  # source currently being tailed
        self._shown_lines: list[str] = []  # unwrapped lines on screen (for re-wrap)
        self._wrap_width = -1  # content width the buffer is wrapped to
        self._last_update = None  # local HH:MM:SS of the last live line
        self._screen_open = True
        self._stage: tempfile.TemporaryDirectory[str] | None = None
        self._stdout_inflight = False
        self._stdout_ready = False
        self._stdout_failed = False
        self._held: tuple[list[str], bool] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        with Horizontal():
            yield ListView(id="sources")
            # Log (not RichLog): it supports native text selection + copy. It
            # does not soft-wrap, so we pre-wrap each line to the pane width.
            yield Log(id="logview", highlight=False, max_lines=20000)
        yield Input(placeholder="filter logs…  (/ focus · Esc clear)", id="logfilter")
        yield Footer()

    def on_mount(self) -> None:
        self._logview = self.query_one("#logview", Log)
        # We drive scrolling ourselves (scroll to the tail on every write unless
        # paused), so disable the widget's blanket auto-scroll.
        self._logview.auto_scroll = False
        self._apply_sources_width()
        self._update_subtitle()
        self._load_sources()

    def on_unmount(self) -> None:
        # Stop the tail worker when leaving this job (it polls follow=True).
        # A download still inside download_stdout_logs recreates its output
        # directory, so the temp dir is removed only after that call returns.
        self._screen_open = False
        self.workers.cancel_all()
        if not self._stdout_inflight:
            self._cleanup_stage()

    def _apply_sources_width(self) -> None:
        try:
            self.query_one("#sources", ListView).styles.width = self._sources_width
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def action_grow_sources(self) -> None:
        self._sources_width = min(self._SOURCES_MAX, self._sources_width + self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def action_shrink_sources(self) -> None:
        self._sources_width = max(self._SOURCES_MIN, self._sources_width - self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def on_resize(self, event) -> None:
        # Terminal resize changes the log pane width → re-wrap the buffer.
        self.call_after_refresh(self._rewrap)

    def action_refresh_sources(self) -> None:
        self._load_sources()

    def _update_subtitle(self) -> None:
        # Dynamic status line: everything that changes over the session lives
        # here, leaving the #summary line for static config only.
        bits = [self._job_id]
        st = (self._job_status or "").strip()
        if st:
            bits.append(st.removeprefix("JOB_STATE_"))
        if self._job_reason:
            bits.append(f"reason: {self._job_reason}")
        if self._filter:
            bits.append(f"filter:{self._filter}")
        if self._min_level:
            bits.append(f"≥{self._min_level}")
        if self._paused:
            bits.append("PAUSED")
        if self._last_update:
            bits.append(f"updated {self._last_update}")
        self.app.sub_title = "  ·  ".join(bits)

    def _set_summary(self, text: str) -> None:
        try:
            self.query_one("#summary", Static).update(text)
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    # ─── filter / level / pause ──────────────────────────────────────────
    def action_filter(self) -> None:
        self.set_focus(self.query_one("#logfilter", Input))

    def action_dismiss_or_back(self) -> None:
        # Esc clears an active filter (and unfocuses the input); otherwise backs out.
        inp = self.query_one("#logfilter", Input)
        if self.focused is inp or self._filter:
            inp.value = ""  # fires on_input_changed → clears the filter
            self.set_focus(self.query_one("#sources", ListView))
        else:
            self.app.pop_screen()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "logfilter":
            self._filter = event.value.strip()
            self._update_subtitle()
            self._refilter()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "logfilter":
            self.set_focus(self.query_one("#sources", ListView))

    def action_cycle_level(self) -> None:
        i = self._LEVEL_CYCLE.index(self._min_level) if self._min_level in self._LEVEL_CYCLE else 0
        self._min_level = self._LEVEL_CYCLE[(i + 1) % len(self._LEVEL_CYCLE)]
        self._update_subtitle()
        self._refilter()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        # Resuming jumps to the end and follows again; pausing just stops
        # following (the user can scroll freely). A saved console that arrived
        # while paused replaces the pane now.
        if not self._paused:
            held = self._held
            self._held = None
            if held is not None:
                lines, empty_note = held
                self._shown_lines = []
                if self._logview is not None:
                    self._logview.clear()
                if lines:
                    self._write_lines(lines)
                elif empty_note:
                    self._write_line("No log output is available.")
            if self._logview is not None:
                self._logview.scroll_end(animate=False)
        self._update_subtitle()

    def _refilter(self) -> None:
        # Re-run the tail so the filter/level apply to the replayed cache too.
        if self._current_source:
            self._start_tail(self._current_source)

    # ─── export / copy ───────────────────────────────────────────────────
    def _export_target(self) -> tuple[str, str] | None:
        """UI thread. ``path`` is the saved console; empty means export the cache."""
        src = self._current_source
        if not src:
            return None
        if self._stdout_ready:
            path = self._stdout_path(src)
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return src, str(path)
            except OSError:
                pass
        return src, ""

    def _exported_text(self, src: str, path: str) -> str:
        if path:
            return Path(path).read_bytes().decode("utf-8", errors="replace")
        entries = self._cache.load_entries(src)
        return "\n".join(format_log_entry(entry) for entry in entries)

    def _take_export(self) -> tuple[str, str, str] | None:
        try:
            target = self.app.call_from_thread(self._export_target)
        except RuntimeError:
            return None
        if target is None:
            return None
        src, path = target
        return src, path, self._exported_text(src, path)

    @work(thread=True, exclusive=True, group="export")
    def action_save_log(self) -> None:
        exported = self._take_export()
        if exported is None:
            return
        src, file_path, text = exported
        count = len(text.splitlines())
        safe = src.replace("/", "_").replace(":", "-")
        path = os.path.expanduser(f"~/cortex-training-{self._job_id[:8]}-{safe}.log")
        try:
            if file_path:
                with open(path, "wb") as handle:
                    handle.write(Path(file_path).read_bytes())
            else:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text + ("\n" if text else ""))
            msg = f"[saved] {count} line(s) → {path}"
        except OSError as exc:  # noqa: BLE001
            msg = f"[error] save failed: {exc}"
        self.app.call_from_thread(self._write_line, msg)

    @work(thread=True, exclusive=True, group="copy")
    def action_copy_log(self) -> None:
        exported = self._take_export()
        if exported is None:
            return
        _src, _path, text = exported
        count = len(text.splitlines())
        self.app.call_from_thread(self.app.copy_to_clipboard, text)
        self.app.call_from_thread(self._write_line, f"[copied] {count} line(s) to clipboard")

    @work(thread=True, exclusive=True, group="sources")
    def _load_sources(self) -> None:
        worker = get_current_worker()
        # Fetch the job once (GS metadata — survives after the zone is gone) for
        # the summary header, status (drives follow/cache), and the sub-job list.
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception:  # noqa: BLE001 - best-effort
            job = {}
        if job.get("status"):
            self._job_status = job.get("status")
        if job:
            self._job_reason = job.get("reason")  # clears a stale reason on refresh
            # The summary line is reserved for static config; status/reason go
            # on the subtitle line instead.
            self._post(self._set_summary, format_job_config(job))
            self._post(self._on_job_loaded, job)
        self._post(self._update_subtitle)

        # Each sub-job is a tailable source (its zone-manager / Ray-head pod,
        # whose stdout already includes worker output via Ray log_to_driver).
        sources = []
        for sj in job.get("sub_jobs") or []:
            if isinstance(sj, dict):
                sid = sj.get("sub_job_id") or sj.get("id")
                if sid:
                    sources.append({"sub_job_id": sid, "job_type": sj.get("job_type") or sj.get("type")})
        if not sources:
            # get_job failed/empty (e.g. offline): fall back to cached sub-jobs.
            sources = [{"sub_job_id": sid} for sid in self._cache.cached_sources()]
        if not sources:
            self._post(
                self._write_line,
                "[no logs] this job has no sub-jobs and no cached logs are available",
            )
        if worker.is_cancelled:
            return
        self._post(self._populate_sources, sources)

    async def _populate_sources(self, sources) -> None:
        lv = self.query_one("#sources", ListView)
        await lv.clear()  # await removal before re-appending src-* ids (see _populate)
        self._source_by_item = {}
        items = [s for s in sources if s.get("sub_job_id")]
        for i, s in enumerate(items):
            sid = s["sub_job_id"]
            item_id = f"src-{i}"
            self._source_by_item[item_id] = sid
            lv.append(ListItem(Label(views.sub_job_label(s)), id=item_id))
        if items:
            # --sub-job-id (or a sub-job drill-down) picks the source to open.
            first = next((i for i, s in enumerate(items) if s["sub_job_id"] == self._sub_job_id), 0)
            lv.index = first
            self._start_tail(items[first]["sub_job_id"])

    def _on_job_loaded(self, job: dict) -> None:
        """UI thread. Hook for subclasses that render more of the job."""

    def _display(self, line: str) -> str:
        """Hook: how a buffered line is shown. The buffer (and save/copy) keep it raw."""
        return line

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        sid = self._source_by_item.get(event.item.id or "")
        # Re-clicking the source you're already viewing shouldn't reload/reset
        # the pane (that's what caused the jump-to-bottom on every click).
        if sid and sid != self._current_source:
            self._start_tail(sid)

    def _post(self, fn, *args, **kwargs) -> bool:
        """Run ``fn`` on the UI thread from a worker; return ``False`` instead of
        raising once the app/event loop is gone (teardown), so background workers
        stop cleanly rather than surfacing a NoActiveAppError as a WorkerError.

        Only teardown-class errors (``RuntimeError`` — NoActiveAppError and
        "event loop is closed" both derive from it) are swallowed; a real bug in
        ``fn`` (Attribute/Type/Key/…) still propagates to ``_tail``'s handler so
        it gets logged rather than silently stopping the tail."""
        try:
            self.app.call_from_thread(fn, *args, **kwargs)
            return True
        except RuntimeError:
            return False

    def _stdout_path(self, sub_job_id: str) -> Path:
        root = "" if self._stage is None else self._stage.name
        return Path(root) / sub_job_id / "stdout.log"

    def _cleanup_stage(self) -> None:
        stage = self._stage
        self._stage = None
        if stage is not None:
            stage.cleanup()

    def _claim_stdout(self) -> bool:
        """UI thread. One whole-job download for this view; a failure retries on the next open."""
        if not self._screen_open or self._stdout_ready or self._stdout_inflight or self._stdout_failed:
            return False
        if self._stage is None:
            self._stage = tempfile.TemporaryDirectory(prefix="cortex-training-stage-")
        self._stdout_inflight = True
        return True

    def _filtered_stdout(self, sub_job_id: str) -> list[str] | None:
        path = self._stdout_path(sub_job_id)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        query = self._filter
        kept: deque[str] = deque(maxlen=20000)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                text = raw.rstrip("\n")
                if entry_matches(text, query):
                    kept.append(text)
        return list(kept)

    def _replace_body(self, lines: list[str], *, empty_note: bool) -> None:
        if self._paused:
            self._held = (list(lines), empty_note)
            return
        self._held = None
        self._shown_lines = []
        if self._logview is not None:
            self._logview.clear()
        if lines:
            self._write_lines(lines)
        elif empty_note:
            self._write_line("No log output is available.")

    def _paint_saved(self) -> None:
        source = self._current_source
        if not source:
            return
        lines = self._filtered_stdout(source)
        if lines is None:
            entries = self._cache.load_entries(source, limit=2000)
            cached = [
                format_log_entry(entry)
                for entry in entries
                if entry_matches(entry, self._filter) and entry_at_level(entry, self._min_level)
            ]
            self._replace_body(cached, empty_note=not entries)
            return
        self._replace_body(lines, empty_note=False)

    def _finish_download(self, ok: bool) -> None:
        if not self._screen_open:
            self._stdout_inflight = False
            self._cleanup_stage()
            return
        self._stdout_inflight = False
        if not ok:
            self._stdout_failed = True
            return
        self._stdout_ready = True
        self._paint_saved()

    def _apply_terminal_status(self, status: str, reason: str | None) -> None:
        self._job_status = status
        if reason is not None:
            self._job_reason = reason
        self._update_subtitle()

    def _start_tail(self, sub_job_id: str) -> None:
        self._current_source = sub_job_id
        self._held = None
        self._shown_lines = []
        # A fresh tail context (new source, or a re-tail after a filter change)
        # starts following the tail — otherwise a pause left over from a prior
        # source would strand this one at the top, the very bug we're fixing.
        self._paused = False
        self._update_subtitle()
        if self._logview is not None:
            self._logview.clear()
        self._write_line(f"— tailing {sub_job_id} —")
        if self._stdout_ready:
            self._paint_saved()
            return
        download = not is_active_status(self._job_status) and self._claim_stdout()
        live = is_active_status(self._job_status)
        self._tail(sub_job_id, download, live)

    @work(thread=True, exclusive=True, group="tail")
    def _tail(self, sub_job_id: str, download: bool, live: bool) -> None:
        worker = get_current_worker()
        # A terminal job's zone is gone, so the operation API would error: serve
        # cache, then one saved-console download. A running/unknown job follows live.
        try:
            self._stream(worker, sub_job_id, live=live)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            if worker.is_cancelled:
                pass
            elif live:
                if not self._handoff_after_tail_error(worker):
                    self._report_tail_error(sub_job_id, exc)
                return
            elif not download:
                self._report_tail_error(sub_job_id, exc)
                return
        if download:
            self._run_download(sub_job_id)

    def _stream(self, worker, sub_job_id: str, *, live: bool) -> None:
        # Replay the local cache instantly, then (only for a live job) resume
        # the tail from the saved cursor so only new lines are fetched. Each
        # yield is a whole page (or the cache batch); render it in one shot
        # so high-velocity logs don't saturate the UI thread per line.
        saw_live = False
        for kind, entries in cached_log_pages(
            self._cache,
            self._client,
            self._job_id,
            sub_job_id,
            follow=live,
            live=live,
            poll_interval=self._poll_interval,
            replay_limit=2000,
            is_cancelled=lambda: worker.is_cancelled,
        ):
            if worker.is_cancelled:
                return
            lines = [
                format_log_entry(entry)
                for entry in entries
                if entry_matches(entry, self._filter) and entry_at_level(entry, self._min_level)
            ]
            if not lines:
                continue
            if kind == "live" and not saw_live:
                saw_live = True
                lines = ["— live —"] + lines
            if not self._post(self._write_replay, lines):
                return
            if kind == "live":
                self._last_update = time.strftime("%H:%M:%S")
                if not self._post(self._update_subtitle):
                    return

    def _report_tail_error(self, sub_job_id: str, exc: BaseException) -> None:
        try:
            with open(os.path.expanduser(_ERROR_LOG), "a", encoding="utf-8") as handle:
                handle.write(f"tail {sub_job_id} failed:\n{traceback.format_exc()}\n")
        except Exception:  # noqa: BLE001
            pass
        self._post(
            self._write_line,
            f"[error] tail {sub_job_id} failed: {type(exc).__name__}: {exc}  (full traceback: {_ERROR_LOG})",
        )

    def _handoff_after_tail_error(self, worker) -> bool:
        """Return False when the tail error should stay: get_job failed, or the job is still active."""
        if worker.is_cancelled or not self._screen_open:
            return True
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception:  # noqa: BLE001 - a running job keeps the tail error
            return False
        status = job.get("status") or self._job_status
        if is_active_status(status):
            return False
        if worker.is_cancelled or not self._screen_open:
            return True
        reason = job.get("reason") if isinstance(job, dict) and "reason" in job else None
        if not self._post(self._apply_terminal_status, status, reason):
            return True
        if worker.is_cancelled or not self._screen_open:
            return True
        try:
            claimed = self.app.call_from_thread(self._claim_stdout)
        except RuntimeError:
            return True
        if not claimed:
            if self._stdout_ready:
                self._post(self._paint_saved)
            return True
        self._run_download(self._current_source or "")
        return True

    def _run_download(self, sub_job_id: str) -> None:
        if not self._screen_open:
            if not self._post(self._finish_download, False):
                self._cleanup_stage()
            return
        ok = False
        try:
            root = "" if self._stage is None else self._stage.name
            self._client.download_stdout_logs(self._job_id, root)
            ok = True
        except Exception:  # noqa: BLE001 - keep the cache; record the traceback
            try:
                with open(os.path.expanduser(_ERROR_LOG), "a", encoding="utf-8") as handle:
                    handle.write(
                        f"saved log {sub_job_id} failed:\n{traceback.format_exc()}\n"
                    )
            except Exception:  # noqa: BLE001
                pass
        if not self._post(self._finish_download, ok):
            self._cleanup_stage()

    def _content_width(self) -> int:
        """Usable text columns of the log pane (accounts for padding/scrollbar)."""
        lv = self._logview
        if lv is None:
            return 0
        try:
            w = lv.scrollable_content_region.width
        except Exception:  # noqa: BLE001 - not laid out yet
            w = 0
        if w <= 0:
            w = getattr(lv.size, "width", 0)
        return max(0, w)

    def _write_replay(self, lines: list[str]) -> None:
        # Checked on the UI thread so a replay started before the saved console
        # cannot append after that console replaces the pane.
        if self._stdout_ready:
            return
        self._write_lines(lines)

    def _write_line(self, line: str) -> None:
        self._write_lines([line])

    def _write_lines(self, lines: list[str]) -> None:
        lv = self._logview
        if lv is None or not lines:
            return
        self._shown_lines.extend(lines)
        # Bound the re-wrap buffer to the widget's own max_lines budget.
        if len(self._shown_lines) > 20000:
            del self._shown_lines[: len(self._shown_lines) - 20000]
        # Always stick to the tail so every viewer sees the newest lines on
        # open and as they stream — cache replay and live alike. Pause is the
        # one escape hatch: it stops following so the user can scroll back.
        follow = not self._paused
        w = self._content_width()
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(wrap_log_line(self._display(line), w))
        # One write_lines call per page keeps the UI responsive under
        # high-velocity logs (vs. a cross-thread hop + write per line).
        lv.write_lines(wrapped, scroll_end=follow)

    def _rewrap(self) -> None:
        """Re-wrap the on-screen buffer after the pane width changes."""
        lv = self._logview
        if lv is None:
            return
        w = self._content_width()
        if w <= 0 or w == self._wrap_width:
            return
        self._wrap_width = w
        lv.clear()
        for line in self._shown_lines:
            for piece in wrap_log_line(self._display(line), w):
                lv.write_line(piece)
        # Re-wrapping clears + repaints (resetting scroll to the top), so snap
        # back to the tail unless the user has paused to read history.
        if not self._paused:
            lv.scroll_end(animate=False)


class CortexTrainingLogTUI(App):
    """Entry app: show the job list, or jump straight to a job's Logs tab when a
    job_id is supplied."""

    CSS = """
    #jobs { height: 1fr; }
    #listinfo { padding: 0 1; height: 1; }
    #nomatch { padding: 1 2; color: $warning; height: auto; }
    #searchhelp { padding: 0 1; height: auto; }
    #sources { width: 48; border-right: solid $panel; }
    #logview { padding: 0 1; }
    #summary { color: $text-muted; padding: 0 1; height: 1; }
    #logfilter { border: tall $panel; }
    #jobfilter { border: tall $panel; }
    #tabs { height: 1fr; }
    TabPane { padding: 0; }
    #metric-head, #cp-head { padding: 0 1; height: auto; margin-bottom: 1; }
    #metriclist { width: 40; border-right: solid $panel; }
    #metric-pane, #gpu-pane { padding: 0 1; }
    #cptable { height: auto; max-height: 70%; }
    #cp-foot { padding: 1 1; height: auto; }
    #summary-body { padding: 1 1; }
    #preview { height: auto; }
    #health { height: auto; margin: 1 0; }
    #subjobs-title { height: auto; margin-top: 1; }
    #subjobs { height: auto; max-height: 12; margin: 0 1; }
    #glance { height: auto; padding: 0 1; }
    #cp-detail { padding: 1 2; }
    .cp-title { margin-top: 1; color: $accent; }
    .cp-section { padding: 0 2; color: $text; }
    PromptScreen { align: center middle; }
    #prompt { width: 80; height: auto; border: thick $accent; background: $surface; padding: 1 2; }
    #prompt-hint { color: $text-muted; }
    /* Text-selection highlight: reuse the picker's subtle "blurred cursor"
       shade so it's visible but not a heavy saturated block. */
    LogScreen > .screen--selection {
        background: $block-cursor-blurred-background;
        color: $block-cursor-blurred-foreground;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, client, job_id=None, *, sub_job_id=None, poll_interval=1.0, now=None):
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._now = now
        self.title = "cortex training"

    def on_mount(self) -> None:
        if self._job_id:
            from cortex_training.tui.screens import JobDetailScreen

            self.push_screen(
                JobDetailScreen(
                    self._client,
                    self._job_id,
                    sub_job_id=self._sub_job_id,
                    initial_tab="tab-logs",
                    poll_interval=self._poll_interval,
                )
            )
        else:
            self.push_screen(
                JobListScreen(self._client, sub_job_id=self._sub_job_id, poll_interval=self._poll_interval, now=self._now)
            )
