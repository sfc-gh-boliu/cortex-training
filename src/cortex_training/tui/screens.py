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

from datetime import datetime
from datetime import timezone

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
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane

from cortex_training.tui import views as v
from cortex_training.tui.app import LogScreen
from cortex_training.tui.format import source_label
from cortex_training.tui.resource import RANGES
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
        ("m", "open('metrics')", "Metrics"),
        ("l", "open('logs')", "Logs"),
        ("g", "open('resource')", "Resources"),
        ("c", "open('checkpoints')", "Checkpoints"),
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
        self._open_detail(self._scope_by_row.get(event.row_key.value), "metrics")

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
_LOG_ONLY = {"save_log", "copy_log", "copy_text", "cycle_level", "toggle_pause", "shrink_sources", "grow_sources", "filter"}
# Pane ids match LogScreen's ("logs", "resource"), so its tab logic applies as-is.
_TAB_IDS = ("metrics", "logs", "resource", "checkpoints")
_TAB_FOCUS = {"metrics": "#metriclist", "logs": "#sources", "resource": "#resource-table", "checkpoints": "#cptable"}


class JobDetailScreen(LogScreen):
    """Metrics | Logs | Resources | Checkpoints for a job, optionally scoped to one sub-job.

    Extends :class:`LogScreen`, so the Logs tab keeps its live tail, cache,
    saved-console fallback, filter, level, pause, save, and copy behavior, and
    the Resources tab is LogScreen's GPU view of the sub-job selected in Logs.

    Textual runs a handler such as ``on_mount`` for every class in the MRO, so
    the handlers here never call ``super()`` for them.
    """

    BINDINGS = [
        Binding("1", "show_tab('metrics')", "Metrics"),
        Binding("2", "show_tab('logs')", "Logs"),
        Binding("3", "show_tab('resource')", "Resources"),
        Binding("4", "show_tab('checkpoints')", "Checkpoints"),
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
    _RESOURCE_KEYS = "w window 5m · 15m · 1h · 6h · all   r refresh   ·   shows the sub-job selected in Logs (2)"

    def __init__(self, client, job_id, *, job: dict | None = None, sub_job_id=None, initial_tab: str = "logs",
                 aliases: AliasStore | None = None, metrics: MetricsStore | None = None, poll_interval=1.0,
                 job_status=None) -> None:
        super().__init__(client, job_id, sub_job_id=sub_job_id, poll_interval=poll_interval,
                         job_status=job_status or (job or {}).get("status"))
        self._job = job or {"job_id": job_id}
        self._scope = sub_job_id
        self._initial_tab = initial_tab if initial_tab in _TAB_IDS else "logs"
        self._active_tab = self._initial_tab
        self._aliases = aliases or AliasStore()
        self._metrics_store = metrics or MetricsStore()
        self._metric_records: list = []
        self._metric_key: str | None = None
        self._metric_window: tuple = _METRIC_WINDOWS[0]
        self._metric_axis = "step"
        self._checkpoints: list = []
        self._cp_loaded = False
        self._cp_loading = False
        self._cp_error: str | None = None
        self._cp_sort = "time"
        self._cp_rows: dict = {}

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        with TabbedContent(initial=self._initial_tab, id="job-tabs"):
            with TabPane("Metrics", id="metrics"):
                yield Static("", id="metric-head")
                with Horizontal(id="metrics-body"):
                    yield ListView(id="metriclist")
                    with VerticalScroll(id="metric-pane"):
                        yield Static("", id="metric-chart")
            with TabPane("Logs", id="logs"):
                yield from self._compose_logs()
            with TabPane("Resources", id="resource"):
                yield from self._compose_resource()
            with TabPane("Checkpoints", id="checkpoints"):
                yield Static("", id="cp-head")
                yield DataTable(id="cptable", cursor_type="row", zebra_stripes=True)
                yield Static("", id="cp-foot")
        yield Footer(id="footer")

    def on_mount(self) -> None:
        self._render_summary_line()
        self._load_metrics()
        if self._initial_tab == "checkpoints":
            self._load_checkpoints()
        self.set_interval(self._DATA_REFRESH_SECONDS, self._auto_refresh_data)
        self.set_interval(10, self._auto_refresh_metrics)

    def on_screen_resume(self) -> None:
        self._update_subtitle()  # a pushed checkpoint view or prompt replaced it

    def _render_summary_line(self) -> None:
        name = v.display_name(self._job, self._aliases.all())
        scope = f"sub-job {self._scope}" if self._scope else "all sub-jobs"
        head = f"[b]{escape(name)}[/b]  ·  " if name else ""
        self._set_summary(f"{head}{escape(v.job_id_of(self._job) or self._job_id)}  ·  {scope}  ·  "
                          f"{escape(v.job_type_label(self._job))}  ·  {escape(', '.join(v.job_models(self._job)) or '—')}")

    def _on_job_loaded(self, job: dict) -> None:
        self._job = {**self._job, **job}
        self._render_summary_line()

    def _live(self) -> bool:
        # LogScreen keeps _job_status current (get_job, and the tail seeing the job end).
        return v.is_live({"status": self._job_status or self._job.get("status")})

    # ── tabs & bindings ──
    def action_show_tab(self, tab: str) -> None:
        if tab in _TAB_IDS:
            self.query_one("#job-tabs", TabbedContent).active = tab

    def action_cycle_tab(self) -> None:
        i = _TAB_IDS.index(self._active_tab) if self._active_tab in _TAB_IDS else 0
        self.action_show_tab(_TAB_IDS[(i + 1) % len(_TAB_IDS)])

    def _sync_footer(self) -> None:
        # The footer stays on every tab; check_action scopes its keys to the tab.
        self.refresh_bindings()

    def _focus_active_tab(self) -> None:
        try:
            self.set_focus(self.query_one(_TAB_FOCUS.get(self._active_tab, "#sources")))
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def _other_tab_shown(self, tab: str) -> None:
        self._focus_active_tab()
        if tab == "metrics":
            self.call_after_refresh(self._render_metric_chart)  # it was sized while hidden
        elif tab == "checkpoints" and not self._cp_loaded:
            self._load_checkpoints()

    def check_action(self, action: str, parameters):
        tab = self._active_tab
        if action in _LOG_ONLY:
            return tab == "logs"
        if action == "resource_range":
            return False  # 1-4 pick tabs here; w cycles the GPU window
        if action == "cycle_window":
            return tab in ("metrics", "resource")
        if action in ("custom_window", "toggle_axis"):
            return tab == "metrics"
        if action == "cycle_cp_sort":
            return tab == "checkpoints"
        return True

    def _display(self, line: str) -> str:
        # Tag every log line with the sub-job it came from; banners stay as-is.
        if not self._current_source or line.startswith(("— ", "[")) or line == "No log output is available.":
            return line
        tag = source_label(self._current_source).replace(" #", ":")
        return f"{tag} │ {line}"

    def action_refresh_sources(self) -> None:
        # r refreshes what the visible tab shows; Logs and Resources keep LogScreen's behavior.
        if self._active_tab == "metrics":
            self._load_metrics()
        elif self._active_tab == "checkpoints":
            self._load_checkpoints()
        else:
            super().action_refresh_sources()

    def _auto_refresh_data(self) -> None:
        # Only the visible tab polls, and only while the job is live.
        if self.app.screen is not self or not self._live():
            return
        if self._resource_active():
            self._start_resource_refresh(force=True)
        elif self._active_tab == "checkpoints":
            self._load_checkpoints()

    def _auto_refresh_metrics(self) -> None:
        if self.app.screen is self and self._active_tab == "metrics" and self._live():
            self._load_metrics()

    # ── windows ──
    def action_cycle_window(self) -> None:
        if self._resource_active():
            names = list(RANGES)
            i = names.index(self._resource_range) if self._resource_range in names else -1
            self.action_resource_range(names[(i + 1) % len(names)])
            return
        i = _METRIC_WINDOWS.index(self._metric_window) if self._metric_window in _METRIC_WINDOWS else -1
        self._metric_window = _METRIC_WINDOWS[(i + 1) % len(_METRIC_WINDOWS)]
        self._render_metric_chart()

    def action_custom_window(self) -> None:
        hint = "30m, 2h, 1d, all, steps:100-500"

        def done(value):
            if value is None:
                return
            win = v.parse_window(value)
            if win is None:
                self.notify(f"Couldn't read window “{escape(value)}”. Try {hint}.", severity="warning")
                return
            self._metric_window = win
            self._render_metric_chart()

        self.app.push_screen(PromptScreen("Metrics window", "", hint), done)

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
        bits.append(f"{len(catalog)} metric(s) logged on this machine  ·  {len(self._metric_records)} record(s)")
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
        try:
            chart = self.query_one("#metric-chart", Static)
        except Exception:  # noqa: BLE001 - not mounted
            return
        if not self._metric_records or not self._metric_key:
            chart.update(
                "[b]No client-logged metrics for this job on this machine.[/b]\n\n"
                "The service does not keep per-step training metrics. Recipes run from this\n"
                "machine record them automatically; from your own loop call:\n\n"
                "  [b]from cortex_training.tui.stores import MetricsStore[/b]\n"
                f'  [b]MetricsStore().append("{escape(self._job_id)}", {{"train_loss": loss}}, step=step)[/b]'
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
        # LogScreen.on_resize also runs (log re-wrap, Resources); redraw the chart here.
        if self._active_tab == "metrics":
            self.call_after_refresh(self._render_metric_chart)

    # ── checkpoints tab ──
    def _load_checkpoints(self) -> None:
        # A thread can't be cancelled, so never stack a second listing on a slow one.
        if self._cp_loading:
            return
        self._cp_loading = True
        self._fetch_checkpoints()

    @work(thread=True, group="checkpoints")
    def _fetch_checkpoints(self) -> None:
        try:
            cps = self._client.list_checkpoints(self._job_id)
            error = None
        except Exception as exc:  # noqa: BLE001 - surfaced in the tab
            cps, error = None, f"{type(exc).__name__}: {exc}"
        self._post(self._apply_checkpoints, cps if isinstance(cps, list) else None, error)

    def _apply_checkpoints(self, cps, error) -> None:
        self._cp_loading = False
        if cps is not None:
            self._checkpoints = cps
        self._cp_loaded = True
        self._cp_error = error
        self._render_checkpoints()

    def _cp_sorts(self) -> tuple:
        # Sorting by step only means something when some checkpoint carries one.
        if any(v.checkpoint_step(c) is not None for c in self._checkpoints):
            return v.CHECKPOINT_SORTS
        return tuple(k for k in v.CHECKPOINT_SORTS if k != "step")

    def action_cycle_cp_sort(self) -> None:
        sorts = self._cp_sorts()
        i = sorts.index(self._cp_sort) if self._cp_sort in sorts else -1
        self._cp_sort = sorts[(i + 1) % len(sorts)]
        self._render_checkpoints()

    def _render_checkpoints(self) -> None:
        table = self.query_one("#cptable", DataTable)
        cps = v.sort_checkpoints(self._checkpoints, self._cp_sort)
        metric_keys = v.checkpoint_metric_keys(cps)
        has_step = any(v.checkpoint_step(c) is not None for c in cps)
        selected = None
        if table.row_count:
            try:
                selected = self._cp_rows.get(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
            except Exception:  # noqa: BLE001 - no cursor row
                selected = None
        table.clear(columns=True)
        table.add_columns("Checkpoint ID", "Type", *(["Step"] if has_step else []), "Created", *metric_keys)
        self._cp_rows = {}
        cursor = 0
        for i, cp in enumerate(cps):
            kind = v.checkpoint_kind(cp)
            kind_cell = Text("◆ resumable", style="bold green") if kind == "resumable" else (
                Text("○ weights-only", style="bold cyan") if kind == "weights-only" else Text(kind))
            step = v.checkpoint_step(cp)
            key = f"cp-{i}"
            self._cp_rows[key] = cp
            if selected is not None and cp.get("checkpoint_id") == selected.get("checkpoint_id"):
                cursor = i
            table.add_row(str(cp.get("checkpoint_id") or "?"), kind_cell,
                          *(["—" if step is None else str(step)] if has_step else []),
                          v.fmt_ts(cp.get("created_at")), *[v.fmt_metric(v._as_float(cp.get(k))) for k in metric_keys], key=key)
        if cps:
            table.move_cursor(row=cursor, animate=False)
        counts = {k: sum(1 for c in cps if v.checkpoint_kind(c) == k) for k in ("resumable", "weights-only")}
        head = (f"  [b]{len(cps)}[/b] checkpoint(s)  ·  [green]◆ {counts['resumable']} resumable[/green] "
                f"[dim](weights + optimizer, resume training)[/dim]  ·  [cyan]○ {counts['weights-only']} weights-only[/cyan] "
                f"[dim](HF weights, serve / export)[/dim]  ·  sorted by [b]{self._cp_sort}[/b]")
        self.query_one("#cp-head", Static).update(head)
        foot = f"[dim]Enter inspect · o sort by {' / '.join(self._cp_sorts())} · r refresh"
        if self._live():
            foot += f" · new checkpoints appear automatically (every {self._DATA_REFRESH_SECONDS}s)"
        foot += "[/dim]"
        if not cps:
            foot = ("  [dim]No checkpoints saved by this job yet.[/dim]\n" if self._cp_loaded else "  Loading…\n") + foot
        if self._cp_error:
            foot = f"  [yellow]list_checkpoints failed: {escape(self._cp_error)}[/yellow]\n" + foot
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
