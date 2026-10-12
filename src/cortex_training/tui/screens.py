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

"""Job preview, tabbed job detail, and checkpoint screens for the TUI.

Flow: job list → :class:`JobSummaryScreen` (preview + sub-jobs) →
:class:`JobDetailScreen` (Metrics | Logs | Resources | Checkpoints) for the
whole job or one sub-job → :class:`CheckpointScreen` for one checkpoint.

Everything here is read-only against the service. The only writes are local:
job display names (:class:`~cortex_training.tui.stores.AliasStore`) and the GPU
telemetry download cache.
"""

from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
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
from textual.widgets import TabbedContent
from textual.widgets import TabPane

from cortex_training.tui import views as v
from cortex_training.tui.app import LogScreen
from cortex_training.tui.format import source_label
from cortex_training.tui.log_cache import cache_root
from cortex_training.tui.log_cache import safe_source_name
from cortex_training.tui.stores import AliasStore
from cortex_training.tui.stores import MetricsStore

STATUS_STYLE = {
    "running": "bold green",
    "initializing": "yellow",
    "placing": "yellow",
    "pending": "yellow",
    "queued": "yellow",
    "creating": "yellow",
    "failed": "bold red",
    "cancelled": "grey62",
    "canceled": "grey62",
    "terminated": "grey62",
}


def status_text(status) -> Text:
    word = v.status_word(status)
    return Text(word.upper(), style=STATUS_STYLE.get(word, ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PromptScreen(ModalScreen):
    """One-line text prompt; dismisses with the entered text, or ``None`` on Esc."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "", placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._value = value
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt"):
            yield Label(self._title, id="prompt-title")
            yield Input(value=self._value, placeholder=self._placeholder, id="prompt-input")
            yield Label("Enter to save · Esc to cancel", id="prompt-hint")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#prompt-input", Input))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ─── job preview ─────────────────────────────────────────────────────────────


class JobSummaryScreen(Screen):
    """Preview of one job: identity, config, status, and a sub-job breakdown."""

    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
        ("escape", "app.pop_screen", "Back"),
        ("m", "open('tab-metrics')", "Metrics"),
        ("l", "open('tab-logs')", "Logs"),
        ("g", "open('tab-resources')", "Resources"),
        ("c", "open('tab-checkpoints')", "Checkpoints"),
        ("n", "rename", "Name job"),
        ("r", "refresh", "Refresh"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self, client, job_id: str, *, job: dict | None = None, aliases: AliasStore | None = None,
                 metrics: MetricsStore | None = None, poll_interval: float = 1.0) -> None:
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._job = job or {"job_id": job_id}
        self._aliases = aliases or AliasStore()
        self._metrics = metrics or MetricsStore()
        self._poll_interval = poll_interval
        self._checkpoints: list = []
        self._scope_by_row: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="summary-body"):
            yield Static("", id="preview")
            yield Static("", id="health")
            yield Static("[b]Sub-jobs[/b]  [dim](Enter opens details for the highlighted row)[/dim]", id="subjobs-title")
            yield DataTable(id="subjobs", cursor_type="row", zebra_stripes=True)
            yield Static("", id="glance")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#subjobs", DataTable)
        table.add_columns("Sub-job", "Type", "Status", "Model", "GPUs", "Configuration")
        self._refresh_view()
        self._load()
        self.set_focus(table)

    def on_screen_resume(self) -> None:
        self._refresh_view()

    @property
    def _display_name(self) -> str:
        return v.display_name(self._job, self._aliases.all())

    def _refresh_view(self) -> None:
        job = self._job
        self.app.sub_title = f"job preview  ·  {self._display_name or self._job_id}"
        now = _now()
        rows = []
        for label, value in v.summary_fields(job, self._display_name, now):
            if label == "Status":
                word = v.status_word(job.get("status"))
                style = STATUS_STYLE.get(word, "")
                live = "  [green]● live[/green]" if v.is_live(job) else "  [dim]historical[/dim]"
                value_markup = f"[{style}]{escape(value)}[/]{live}" if style else escape(value) + live
            elif label == "Name" and not self._display_name:
                value_markup = f"[dim]{escape(value)}[/dim]"
            else:
                value_markup = f"[b]{escape(value)}[/b]" if label in ("Name", "Job ID") else escape(value)
            rows.append(f"  [dim]{label:<12}[/dim] {value_markup}")
        self.query_one("#preview", Static).update("\n".join(rows))
        self.query_one("#health", Static).update(self._health_markup())

        table = self.query_one("#subjobs", DataTable)
        table.clear()
        self._scope_by_row = {"all": None}
        table.add_row(Text("▸ entire job", style="bold"), v.job_type_label(job), status_text(job.get("status")),
                      ", ".join(v.job_models(job)) or "—", str(v.total_gpus(job) or "—"),
                      Text("whole job: every sub-job's metrics, logs, GPUs and checkpoints", style="dim"), key="all")
        for i, sj in enumerate(v.sub_jobs(job)):
            key = f"sj-{i}"
            self._scope_by_row[key] = sj.get("sub_job_id")
            table.add_row(f"  {v.sub_job_label(sj)}", str(sj.get("job_type") or "").lower(), status_text(sj.get("status")),
                          str(sj.get("model_name") or "—"), str(v.sub_job_gpus(sj) or "—"),
                          v.subjob_config_summary(sj), key=key)
        self.query_one("#glance", Static).update(self._glance_markup())

    def _health_markup(self) -> str:
        job = self._job
        word = v.status_word(job.get("status"))
        reason = str(job.get("reason") or "").strip()
        if word == "failed":
            return f"  [bold red]✖ Job failed[/]  reason: [b]{escape(reason or 'not reported')}[/b]  — check Logs (l) for the failing sub-job"
        if word in ("cancelled", "canceled"):
            return f"  [grey62]■ Cancelled[/]  {escape(reason)}"
        if word == "terminated":
            return f"  [grey62]■ Terminated by the platform[/]  {escape(reason)}"
        if word in ("placing", "pending", "queued", "initializing", "creating"):
            return f"  [yellow]◔ {word.capitalize()}[/] — waiting for GPUs or loading the model"
        if word == "running":
            return "  [green]● Running[/] — logs stream live; metrics, resources and checkpoints refresh automatically"
        return ""

    def _glance_markup(self) -> str:
        bits = []
        catalog = v.metric_catalog(self._metrics.load(self._job_id))
        loss = v.primary_loss(catalog)
        if loss:
            at = f" @ step {loss.min_step}" if loss.min_step is not None else ""
            bits.append(f"[dim]{loss.label}[/dim] latest [b]{v.fmt_metric(loss.latest)}[/b] · min [b]{v.fmt_metric(loss.minimum)}[/b]{at}")
        elif catalog:
            bits.append(f"[dim]{len(catalog)} client-logged metric(s)[/dim]")
        else:
            bits.append("[dim]no client-logged training metrics on this machine[/dim]")
        if self._checkpoints:
            latest = v.sort_checkpoints(self._checkpoints, "time")[0]
            step = v.checkpoint_step(latest)
            kinds = {v.checkpoint_kind(c) for c in self._checkpoints}
            bits.append(f"[dim]checkpoints[/dim] [b]{len(self._checkpoints)}[/b] ({', '.join(sorted(kinds))}) · latest "
                        f"{escape(str(latest.get('checkpoint_id')))}" + (f" @ step {step}" if step is not None else ""))
        else:
            bits.append("[dim]no checkpoints saved[/dim]")
        return "\n[b]At a glance[/b]\n  " + "\n  ".join(bits) + (
            "\n\n[dim]m metrics · l logs · g resources · c checkpoints · n name this job · b back[/dim]"
        )

    @work(thread=True, exclusive=True, group="summary")
    def _load(self) -> None:
        try:
            job = self._client.get_job(self._job_id)
        except Exception:  # noqa: BLE001 - keep the list row's copy
            job = None
        try:
            cps = self._client.list_checkpoints(self._job_id)
        except Exception:  # noqa: BLE001 - optional
            cps = []
        try:
            self.app.call_from_thread(self._apply, job if isinstance(job, dict) and job else None,
                                      cps if isinstance(cps, list) else [])
        except RuntimeError:
            pass

    def _apply(self, job, cps) -> None:
        if job:
            self._job = {**self._job, **job}
        self._checkpoints = cps
        self._refresh_view()

    def action_refresh(self) -> None:
        self._load()

    def _highlighted_scope(self):
        table = self.query_one("#subjobs", DataTable)
        try:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001 - empty table
            key = "all"
        return self._scope_by_row.get(key)

    def action_open(self, tab: str) -> None:
        self._open_detail(self._highlighted_scope(), tab)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._open_detail(self._scope_by_row.get(event.row_key.value), "tab-metrics")

    def _open_detail(self, sub_job_id, tab: str) -> None:
        self.app.push_screen(
            JobDetailScreen(self._client, self._job_id, job=self._job, sub_job_id=sub_job_id, initial_tab=tab,
                            aliases=self._aliases, metrics=self._metrics, poll_interval=self._poll_interval)
        )

    def action_rename(self) -> None:
        def done(value):
            if value is not None:
                self._aliases.set(self._job_id, value)
                self._refresh_view()

        self.app.push_screen(PromptScreen(f"Display name for {self._job_id}", self._display_name, "e.g. qwen3-8b lora sweep #3"), done)


# ─── job detail (tabs) ───────────────────────────────────────────────────────

_METRIC_WINDOWS = [("time", 3600.0), ("time", 6 * 3600.0), ("time", 86400.0), ("all",)]
_GPU_WINDOWS = [("time", 600.0), ("time", 1800.0), ("time", 3600.0), ("time", 6 * 3600.0), ("all",)]
_LOG_ONLY = {"save_log", "copy_log", "copy_text", "cycle_level", "toggle_pause", "shrink_sources", "grow_sources", "filter"}
_TAB_IDS = ("tab-metrics", "tab-logs", "tab-resources", "tab-checkpoints")


class JobDetailScreen(LogScreen):
    """Metrics | Logs | Resources | Checkpoints for a job, optionally scoped to one sub-job.

    Extends :class:`LogScreen`, so the Logs tab keeps its live tail, cache,
    saved-console fallback, filter, level, pause, save, and copy behavior.
    """

    BINDINGS = [
        Binding("1", "show_tab('tab-metrics')", "Metrics"),
        Binding("2", "show_tab('tab-logs')", "Logs"),
        Binding("3", "show_tab('tab-resources')", "Resources"),
        Binding("4", "show_tab('tab-checkpoints')", "Checkpoints"),
        Binding("w", "cycle_window", "Window"),
        Binding("W", "custom_window", "Custom window"),
        Binding("x", "toggle_axis", "Step/time"),
        Binding("o", "cycle_cp_sort", "Sort"),
        # Keep the footer readable next to the tab keys; these still work.
        Binding("[", "shrink_sources", "Narrower", show=False),
        Binding("]", "grow_sources", "Wider", show=False),
        Binding("y", "copy_log", "Copy all", show=False),
        Binding("c", "copy_text", "Copy sel", show=False),
    ]
    _DATA_REFRESH_SECONDS = 30

    def __init__(self, client, job_id, *, job: dict | None = None, sub_job_id=None, initial_tab: str = "tab-logs",
                 aliases: AliasStore | None = None, metrics: MetricsStore | None = None, poll_interval=1.0,
                 job_status=None) -> None:
        super().__init__(client, job_id, sub_job_id=sub_job_id, poll_interval=poll_interval,
                         job_status=job_status or (job or {}).get("status"))
        self._job = job or {"job_id": job_id}
        self._scope = sub_job_id
        self._initial_tab = initial_tab if initial_tab in _TAB_IDS else "tab-logs"
        self._aliases = aliases or AliasStore()
        self._metrics_store = metrics or MetricsStore()
        self._metric_records: list = []
        self._metric_key: str | None = None
        self._metric_window: tuple = _METRIC_WINDOWS[0]
        self._metric_axis = "step"
        self._gpu: dict = {}  # sub_job_id -> raw gpu.jsonl records
        self._gpu_window: tuple = _GPU_WINDOWS[0]
        self._gpu_error: str | None = None
        self._gpu_loaded = False
        self._checkpoints: list = []
        self._cp_loaded = False
        self._cp_sort = "time"
        self._cp_rows: dict = {}

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        with TabbedContent(initial=self._initial_tab, id="tabs"):
            with TabPane("Metrics", id="tab-metrics"):
                yield Static("", id="metric-head")
                with Horizontal():
                    yield ListView(id="metriclist")
                    with VerticalScroll(id="metric-pane"):
                        yield Static("", id="metric-chart")
            with TabPane("Logs", id="tab-logs"):
                with Horizontal():
                    yield ListView(id="sources")
                    yield Log(id="logview", highlight=False, max_lines=20000)
                yield Input(placeholder="filter logs…  (/ focus · Esc clear)", id="logfilter")
            with TabPane("Resources", id="tab-resources"):
                with VerticalScroll(id="gpu-pane"):
                    yield Static("Loading GPU telemetry…", id="gpu-body")
            with TabPane("Checkpoints", id="tab-checkpoints"):
                yield Static("", id="cp-head")
                yield DataTable(id="cptable", cursor_type="row", zebra_stripes=True)
                yield Static("", id="cp-foot")
        yield Footer()

    def on_mount(self) -> None:
        super().on_mount()
        self._render_summary_line()
        self._load_metrics()
        self._load_gpu()
        self._load_checkpoints()
        self.set_interval(self._DATA_REFRESH_SECONDS, self._auto_refresh_data)
        self.set_interval(10, self._auto_refresh_metrics)

    def _render_summary_line(self) -> None:
        name = v.display_name(self._job, self._aliases.all())
        scope = f"sub-job {self._scope}" if self._scope else "all sub-jobs"
        head = f"[b]{escape(name)}[/b]  ·  " if name else ""
        self._set_summary(f"{head}{escape(v.job_id_of(self._job) or self._job_id)}  ·  {scope}  ·  "
                          f"{escape(v.job_type_label(self._job))}  ·  {escape(', '.join(v.job_models(self._job)) or '—')}")

    def _on_job_loaded(self, job: dict) -> None:
        self._job = {**self._job, **job}
        self._render_summary_line()
        if self._gpu_loaded:
            self._render_gpu()

    def _scoped(self, sub_job_id: str) -> bool:
        return self._scope is None or sub_job_id == self._scope

    # ── tabs & bindings ──
    def _active_tab(self) -> str:
        try:
            return self.query_one("#tabs", TabbedContent).active
        except Exception:  # noqa: BLE001 - not mounted
            return self._initial_tab

    def action_show_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self.refresh_bindings()
        if self._active_tab() == "tab-logs":
            self.call_after_refresh(self._rewrap)  # the log pane had no width while hidden

    def check_action(self, action: str, parameters):
        tab = self._active_tab()
        if action in _LOG_ONLY:
            return tab == "tab-logs"
        if action in ("cycle_window", "custom_window"):
            return tab in ("tab-metrics", "tab-resources")
        if action == "toggle_axis":
            return tab == "tab-metrics"
        if action == "cycle_cp_sort":
            return tab == "tab-checkpoints"
        return True

    def _display(self, line: str) -> str:
        # Tag every log line with the sub-job it came from; banners stay as-is.
        if not self._current_source or line.startswith(("— ", "[")) or line == "No log output is available.":
            return line
        tag = source_label(self._current_source).replace(" #", ":")
        return f"{tag} │ {line}"

    def action_refresh_sources(self) -> None:
        super().action_refresh_sources()
        self._load_metrics()
        self._load_gpu()
        self._load_checkpoints()

    def _auto_refresh_data(self) -> None:
        if self.app.screen is self and v.is_live(self._job):
            self._load_gpu()
            self._load_checkpoints()

    def _auto_refresh_metrics(self) -> None:
        if self.app.screen is self and v.is_live(self._job):
            self._load_metrics()

    # ── windows ──
    def action_cycle_window(self) -> None:
        if self._active_tab() == "tab-resources":
            i = _GPU_WINDOWS.index(self._gpu_window) if self._gpu_window in _GPU_WINDOWS else -1
            self._gpu_window = _GPU_WINDOWS[(i + 1) % len(_GPU_WINDOWS)]
            self._render_gpu()
        else:
            i = _METRIC_WINDOWS.index(self._metric_window) if self._metric_window in _METRIC_WINDOWS else -1
            self._metric_window = _METRIC_WINDOWS[(i + 1) % len(_METRIC_WINDOWS)]
            self._render_metric_chart()

    def action_custom_window(self) -> None:
        on_gpu = self._active_tab() == "tab-resources"
        hint = "10m, 45m, 2h, all" if on_gpu else "30m, 2h, 1d, all, steps:100-500"

        def done(value):
            if value is None:
                return
            win = v.parse_window(value)
            if win is None or (on_gpu and win[0] == "steps"):
                self.notify(f"Couldn't read window “{value}”. Try {hint}.", severity="warning")
                return
            if on_gpu and win[0] == "time" and win[1] < 600:
                win = ("time", 600.0)  # the shortest GPU window is 10 minutes
                self.notify("GPU windows start at 10 minutes.")
            if on_gpu:
                self._gpu_window = win
                self._render_gpu()
            else:
                self._metric_window = win
                self._render_metric_chart()

        self.app.push_screen(PromptScreen("Time window", "", hint), done)

    def action_toggle_axis(self) -> None:
        self._metric_axis = "time" if self._metric_axis == "step" else "step"
        self._render_metric_chart()

    # ── metrics tab ──
    @work(thread=True, exclusive=True, group="metrics")
    def _load_metrics(self) -> None:
        records = self._metrics_store.load(self._job_id)
        if self._scope:
            records = [r for r in records if not r.get("sub_job_id") or r.get("sub_job_id") == self._scope]
        self._post(self._apply_metrics, records)

    async def _apply_metrics(self, records: list) -> None:
        self._metric_records = records
        catalog = v.metric_catalog(records)
        lv = self.query_one("#metriclist", ListView)
        keys = [m.key for m in catalog]
        if self._metric_key not in keys:
            loss = v.primary_loss(catalog)
            self._metric_key = loss.key if loss else (keys[0] if keys else None)
        await lv.clear()
        family = None
        index = None
        for i, info in enumerate(catalog):
            if info.family != family:
                family = info.family
                lv.append(ListItem(Label(f"[b dim]{family}[/]"), disabled=True))
            lv.append(ListItem(Label(f"{escape(info.label):<26} [b]{v.fmt_metric(info.latest):>9}[/b]"), id=f"metric-{i}"))
            if info.key == self._metric_key:
                index = len(lv) - 1
        if index is not None:
            lv.index = index
        self._render_metric_head(catalog)
        self._render_metric_chart()

    def _render_metric_head(self, catalog: list) -> None:
        head = self.query_one("#metric-head", Static)
        if not catalog:
            head.update("")
            return
        loss = v.primary_loss(catalog)
        bits = []
        if loss:
            at = f" (step {loss.min_step})" if loss.min_step is not None else ""
            bits.append(f"[b]{loss.label}[/b]  latest [b green]{v.fmt_metric(loss.latest)}[/]  ·  min [b]{v.fmt_metric(loss.minimum)}[/]{at}")
        bits.append(f"{len(catalog)} metric(s) logged by the client  ·  {len(self._metric_records)} record(s)")
        head.update("  " + "     ".join(bits))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "metriclist" or event.item is None or not (event.item.id or "").startswith("metric-"):
            return
        catalog = v.metric_catalog(self._metric_records)
        idx = int(event.item.id.split("-", 1)[1])
        if idx < len(catalog) and catalog[idx].key != self._metric_key:
            self._metric_key = catalog[idx].key
            self._render_metric_chart()

    def _render_metric_chart(self) -> None:
        chart = self.query_one("#metric-chart", Static)
        if not self._metric_records or not self._metric_key:
            chart.update(
                "[b]No client-logged metrics for this job on this machine.[/b]\n\n"
                "The service does not keep per-step training metrics. Recipes run from this\n"
                "machine record them automatically; from your own loop call:\n\n"
                "  [b]from cortex_training.tui.stores import MetricsStore[/b]\n"
                f'  [b]MetricsStore().append("{escape(self._job_id)}", {{"train_loss": loss}}, step=step)[/b]\n\n'
                "Checkpoint losses are listed on the Checkpoints tab (4)."
            )
            return
        catalog = {m.key: m for m in v.metric_catalog(self._metric_records)}
        info = catalog.get(self._metric_key)
        pts = v.window_points(v.metric_points(self._metric_records, self._metric_key), self._metric_window)
        if self._metric_axis == "step":
            pts = [p for p in pts if p[1] is not None]
            xs, x_fmt, axis = [p[1] for p in pts], (lambda s: f"step {int(s)}"), "training step"
        else:
            pts = [p for p in pts if p[0] is not None]
            xs, x_fmt, axis = [p[0] for p in pts], v.fmt_clock, "wall-clock time"
        ys = [p[2] for p in pts]
        try:
            width = max(40, self.query_one("#metric-pane").size.width - 2)
            height = max(8, self.query_one("#metric-pane").size.height - 7)
        except Exception:  # noqa: BLE001 - not laid out yet
            width, height = 90, 14
        stats = v.series_stats(ys)
        title = f"[b]{escape(info.label if info else self._metric_key)}[/b]  [dim]({escape(self._metric_key)} · {info.family if info else ''})[/dim]"
        sub = (f"window: [b]{v.window_label(self._metric_window)}[/b]  ·  x: [b]{axis}[/b]  ·  {len(ys)} point(s)  ·  "
               f"latest [b]{v.fmt_metric(stats['latest'])}[/b]  min {v.fmt_metric(stats['min'])}  max {v.fmt_metric(stats['max'])}")
        body = v.render_chart(xs, ys, width=width, height=height, x_fmt=x_fmt) if ys else ["(no points in this window — press w to widen it)"]
        chart.update(title + "\n" + sub + "\n\n[cyan]" + escape("\n".join(body)) + "[/cyan]\n\n"
                     "[dim]↑/↓ pick a metric · w cycle 1h / 6h / 24h / full job · W custom window · x step ↔ time[/dim]")

    def on_resize(self, event) -> None:
        super().on_resize(event)
        self.call_after_refresh(self._render_metric_chart)
        self.call_after_refresh(self._render_gpu)

    # ── resources tab ──
    def _gpu_dir(self) -> Path:
        return cache_root() / "jobs" / safe_source_name(self._job_id) / "gpu"

    @work(thread=True, exclusive=True, group="gpu")
    def _load_gpu(self) -> None:
        out = self._gpu_dir()
        error = None
        try:
            out.mkdir(parents=True, exist_ok=True)
            self._client.download_metrics(self._job_id, str(out), resume=True)
        except Exception as exc:  # noqa: BLE001 - surfaced in the tab; keep any earlier download
            error = f"{type(exc).__name__}: {exc}"
        data: dict = {}
        for path in sorted(out.glob("*/gpu.jsonl")) if out.exists() else []:
            records = []
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        records.append(rec)
            sid = str(records[0].get("sub_job_id")) if records and records[0].get("sub_job_id") else path.parent.name
            data[sid] = records
        self._post(self._apply_gpu, data, error)

    def _apply_gpu(self, data: dict, error) -> None:
        self._gpu = data
        self._gpu_error = error
        self._gpu_loaded = True
        self._render_gpu()

    def _render_gpu(self) -> None:
        try:
            body = self.query_one("#gpu-body", Static)
        except Exception:  # noqa: BLE001 - not mounted
            return
        if not self._gpu_loaded:
            return
        try:
            width = max(50, self.query_one("#gpu-pane").size.width - 4)
        except Exception:  # noqa: BLE001
            width = 100
        subs = [s for s in v.sub_jobs(self._job) if self._scoped(str(s.get("sub_job_id") or ""))]
        sids = [str(s.get("sub_job_id")) for s in subs] or [s for s in self._gpu if self._scoped(s)]
        hw = self._job.get("hardware") or "GPU"
        parts = []
        now = _now().timestamp()
        for sid in sids:
            sj = next((s for s in subs if s.get("sub_job_id") == sid), {"sub_job_id": sid})
            records = self._gpu.get(sid) or []
            samples = v.gpu_series(records)
            alloc = v.sub_job_gpus(sj)
            label = v.sub_job_label(sj)
            if not samples:
                parts.append(f"[b]{escape(label)}[/b]  ·  {alloc or '?'} × {escape(hw)}\n  [dim]no GPU samples yet[/dim]")
                continue
            interval = v.sample_interval(records)
            last = samples[-1]
            fresh = now - last.t
            stale = fresh > 6 * (interval or 5) and v.is_live(self._job)
            freshness = f"{v.fmt_duration(fresh)} ago" + ("  [yellow](stale)[/yellow]" if stale else "")
            u = v.series_stats([s.util for s in samples])
            m = v.series_stats([s.mem_pct for s in samples])
            win = v.window_samples(samples, None if self._gpu_window[0] == "all" else self._gpu_window[1])
            head = (f"[b]{escape(label)}[/b]  ·  {last.gpus or alloc} × {escape(hw)}  ·  sampled every "
                    f"{v._num(interval) if interval else '?'}s  ·  latest sample "
                    f"{datetime.fromtimestamp(last.t).strftime('%H:%M:%S')} ({freshness})")
            table = [
                f"  {'':<26}{'latest':>9}{'min':>9}{'avg':>9}{'max':>9}   [dim](whole job)[/dim]",
                f"  [cyan]{'GPU compute utilization %':<26}[/cyan]{v.fmt_pct(u['latest']):>9}{v.fmt_pct(u['min']):>9}"
                f"{v.fmt_pct(u['avg']):>9}{v.fmt_pct(u['max']):>9}",
                f"  [magenta]{'GPU memory used %':<26}[/magenta]{v.fmt_pct(m['latest']):>9}{v.fmt_pct(m['min']):>9}"
                f"{v.fmt_pct(m['avg']):>9}{v.fmt_pct(m['max']):>9}   "
                f"{last.mem_used / v.GIB:,.1f} / {last.mem_total / v.GIB:,.1f} GiB now",
            ]
            per_gpu = self._per_gpu_latest(records)
            if per_gpu:
                table.append("  [dim]per GPU now (compute / memory): " + escape(per_gpu) + "[/dim]")
            xs = [s.t for s in win]
            chart_u = v.render_chart(xs, [s.util for s in win], width=width, height=5, y_min=0, y_max=100,
                                     y_fmt=v.fmt_pct, x_fmt=v.fmt_clock)
            chart_m = v.render_chart(xs, [s.mem_pct for s in win], width=width, height=5, y_min=0, y_max=100,
                                     y_fmt=v.fmt_pct, x_fmt=v.fmt_clock)
            wl = v.window_label(self._gpu_window)
            parts.append(
                head + "\n" + "\n".join(table)
                + f"\n\n  [cyan]GPU compute utilization (% of SM time busy) — {wl}[/cyan]\n[cyan]" + escape("\n".join(chart_u)) + "[/cyan]"
                + f"\n  [magenta]GPU memory utilization (% of framebuffer allocated) — {wl}[/magenta]\n[magenta]" + escape("\n".join(chart_m)) + "[/magenta]"
            )
        if not parts:
            parts.append("[dim]No GPU telemetry for this job.[/dim]")
        foot = "[dim]w cycle window 10m / 30m / 1h / 6h / full job · W custom window · r refresh"
        if v.is_live(self._job):
            foot += f" · auto-refresh every {self._DATA_REFRESH_SECONDS}s"
        foot += "[/dim]"
        if self._gpu_error and not self._gpu:
            parts.append(f"[yellow]Telemetry download failed: {escape(self._gpu_error)}[/yellow]")
        body.update(("\n\n" + "─" * min(width, 120) + "\n\n").join(parts) + "\n\n" + foot)

    @staticmethod
    def _per_gpu_latest(records: list) -> str:
        latest: dict = {}
        for r in records:
            idx = r.get("global_gpu_index", r.get("local_gpu_index"))
            if idx is not None:
                latest[idx] = r
        if len(latest) < 2:
            return ""
        out = []
        for idx in sorted(latest, key=lambda k: (str(type(k)), k)):
            r = latest[idx]
            used, free = r.get("fb_used_bytes") or 0, r.get("fb_free_bytes") or 0
            mem = 100 * used / (used + free) if (used + free) else 0
            out.append(f"{v._num(idx)}:{(r.get('gpu_util_pct') or 0):.0f}/{mem:.0f}")
        return "  ".join(out)

    # ── checkpoints tab ──
    @work(thread=True, exclusive=True, group="checkpoints")
    def _load_checkpoints(self) -> None:
        try:
            cps = self._client.list_checkpoints(self._job_id)
            error = None
        except Exception as exc:  # noqa: BLE001 - surfaced in the tab
            cps, error = None, f"{type(exc).__name__}: {exc}"
        self._post(self._apply_checkpoints, cps if isinstance(cps, list) else None, error)

    def _apply_checkpoints(self, cps, error) -> None:
        if cps is not None:
            self._checkpoints = cps
        self._cp_loaded = True
        self._render_checkpoints(error)

    def action_cycle_cp_sort(self) -> None:
        i = v.CHECKPOINT_SORTS.index(self._cp_sort)
        self._cp_sort = v.CHECKPOINT_SORTS[(i + 1) % len(v.CHECKPOINT_SORTS)]
        self._render_checkpoints(None)

    def _render_checkpoints(self, error) -> None:
        table = self.query_one("#cptable", DataTable)
        cps = v.sort_checkpoints(self._checkpoints, self._cp_sort)
        metric_keys = v.checkpoint_metric_keys(cps)
        table.clear(columns=True)
        table.add_columns("Checkpoint ID", "Type", "Step", "Created", *metric_keys)
        self._cp_rows = {}
        for i, cp in enumerate(cps):
            kind = v.checkpoint_kind(cp)
            kind_cell = Text("◆ resumable", style="bold green") if kind == "resumable" else (
                Text("○ weights-only", style="bold cyan") if kind == "weights-only" else Text(kind))
            step = v.checkpoint_step(cp)
            key = f"cp-{i}"
            self._cp_rows[key] = cp
            table.add_row(str(cp.get("checkpoint_id") or "?"), kind_cell, "—" if step is None else str(step),
                          v.fmt_ts(cp.get("created_at")), *[v.fmt_metric(v._as_float(cp.get(k))) for k in metric_keys], key=key)
        counts = {k: sum(1 for c in cps if v.checkpoint_kind(c) == k) for k in ("resumable", "weights-only")}
        head = (f"  [b]{len(cps)}[/b] checkpoint(s)  ·  [green]◆ {counts['resumable']} resumable[/green] "
                f"[dim](weights + optimizer, resume training)[/dim]  ·  [cyan]○ {counts['weights-only']} weights-only[/cyan] "
                f"[dim](HF weights, serve / export)[/dim]  ·  sorted by [b]{self._cp_sort}[/b]")
        self.query_one("#cp-head", Static).update(head)
        foot = "[dim]Enter inspect · o sort by time / step / type · r refresh"
        if v.is_live(self._job):
            foot += f" · new checkpoints appear automatically (every {self._DATA_REFRESH_SECONDS}s)"
        foot += "[/dim]"
        if not cps:
            foot = ("  [dim]No checkpoints saved by this job yet.[/dim]\n" if self._cp_loaded else "  Loading…\n") + foot
        if error:
            foot = f"  [yellow]list_checkpoints failed: {escape(error)}[/yellow]\n" + foot
        self.query_one("#cp-foot", Static).update(foot)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "cptable":
            return
        cp = self._cp_rows.get(event.row_key.value)
        if cp:
            self.app.push_screen(CheckpointScreen(cp, self._job, v.display_name(self._job, self._aliases.all())))


# ─── checkpoint detail ───────────────────────────────────────────────────────


class CheckpointScreen(Screen):
    """Expanded view of one checkpoint: fields, origin, config, fetch and deploy."""

    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
        ("escape", "app.pop_screen", "Back"),
        ("y", "copy_instructions", "Copy commands"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self, checkpoint: dict, job: dict, name: str) -> None:
        super().__init__()
        self._cp = checkpoint
        self._job = job
        self._job_name = name
        self._sections = v.checkpoint_details(checkpoint, job, name)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="cp-detail"):
            for title, lines in self._sections:
                yield Static(f"[b]{escape(title)}[/b]", classes="cp-title")
                yield Static(escape("\n".join(lines)), classes="cp-section")
        yield Footer()

    def on_mount(self) -> None:
        kind = v.checkpoint_kind(self._cp)
        self.app.sub_title = f"checkpoint {self._cp.get('checkpoint_id')}  ·  {kind}"

    def action_copy_instructions(self) -> None:
        text = "\n".join(line for title, lines in self._sections[3:] for line in [f"# {title}", *lines])
        self.app.copy_to_clipboard(text)
        self.notify("Fetch and deploy commands copied to the clipboard.")
