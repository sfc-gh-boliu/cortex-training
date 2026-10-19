# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The job screen's Training metrics and Checkpoints tabs.

Each tab is one widget that reads and draws its own data, so the job screen
only places it, tells it when it is shown (:meth:`activate`) and when a live
job's periodic refresh is due (:meth:`refresh_live`). A tab reads the job as
the screen knows it through the ``job`` callable it is given, and the Training
metrics tab fetches its rows through the screen's ``download``, which writes
into the same cache directory as the Resources tab, under the same lock.

The panes' keys (``w``, ``Space``, ``r``) are their own bindings, which win over
the screen's while the focus is inside the pane.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from datetime import timezone
from pathlib import Path
import threading
from typing import Callable

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.containers import VerticalScroll
from textual.widgets import DataTable
from textual.widgets import Static
from textual.worker import get_current_worker

from cortex_training.tui import checkpoints
from cortex_training.tui import training
from cortex_training.tui import views
from cortex_training.tui.resource import CHART_AXIS
from cortex_training.tui.resource import CHART_MUTED
from cortex_training.tui.resource import chart_hue

_WARNING = "#fbbf24"


class _Pane(Vertical):
    """What both tabs share: posting a worker's answer to the UI thread while the pane is up."""

    def __init__(self, client, job_id: str, *, job: Callable[[], dict], id: str) -> None:
        super().__init__(id=id)
        self._client = client
        self._job_id = job_id
        self._job = job
        self._shown = False  # activated at least once: the first show loads
        self._loading = False
        self._error: str | None = None

    def _post(self, fn, *args) -> None:
        """Worker thread: run ``fn(*args)`` on the UI thread, dropped once the pane is gone."""

        def run() -> None:
            if self.is_attached:
                fn(*args)

        try:
            if self.app.is_running:
                self.app.call_from_thread(run)
        except RuntimeError:  # the app is shutting down
            pass

    def _live(self) -> bool:
        return views.is_live(self._job())

    def activate(self) -> None:
        """UI thread. The tab was shown: the first time, read its data."""
        if not self._shown:
            self._shown = True
            self.reload()
        else:
            self.redraw()

    def refresh_live(self) -> None:
        """UI thread. A live job's periodic refresh, while the tab is shown."""
        if self._shown and self._live():
            self.reload()

    def action_refresh(self) -> None:
        self._shown = True
        self.reload()

    def reload(self) -> None:
        raise NotImplementedError

    def redraw(self) -> None:
        raise NotImplementedError


# ─── Checkpoints ─────────────────────────────────────────────────────────────


class CheckpointsPane(_Pane):
    """A job's checkpoints, newest first, and the highlighted one's details and use."""

    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def __init__(self, client, job_id: str, *, job: Callable[[], dict], id: str = "checkpoints") -> None:
        super().__init__(client, job_id, job=job, id=id)
        self._records: list[dict] | None = None  # None: not read yet
        self._metrics: list[str] = []
        self._read_at: datetime | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="checkpoints-status")
        table = DataTable(id="checkpoints-table", cursor_type="row", zebra_stripes=True)
        table.display = False
        yield table
        with VerticalScroll(id="checkpoints-detail-pane"):
            yield Static("", id="checkpoints-detail")

    def focus_target(self):
        table = self.query_one("#checkpoints-table", DataTable)
        return table if table.display else self.query_one("#checkpoints-detail-pane")

    def _has_training(self) -> bool:
        return any(views.sub_job_kind(sj) == "training" for sj in views.sub_jobs(self._job()))

    def reload(self) -> None:
        if self._loading:
            return
        if not self._has_training() and self._job().get("sub_jobs"):
            self._records = []  # a sampling job saves none: nothing to ask
            self.redraw()
            return
        self._loading = True
        self.redraw()
        self._read()

    @work(thread=True, exclusive=True, group="checkpoints")
    def _read(self) -> None:
        worker = get_current_worker()
        try:
            records = self._client.list_checkpoints(self._job_id)
        except Exception as exc:  # noqa: BLE001 - said in the tab's header
            if not worker.is_cancelled:
                self._post(self._finish, None, views.access_hint(exc, "checkpoints") or views.describe_error(exc))
            return
        if not worker.is_cancelled:
            self._post(self._finish, records if isinstance(records, list) else [], None)

    def _finish(self, records: list | None, error: str | None) -> None:
        self._loading = False
        self._error = error
        if records is not None:
            self._records = checkpoints.sort_checkpoints(records)
            self._read_at = datetime.now(timezone.utc)
            self._fill_table()
        self.redraw()

    def _fill_table(self) -> None:
        table = self.query_one("#checkpoints-table", DataTable)
        records = self._records or []
        picked = self._highlighted()
        picked_id = picked.get("checkpoint_id") if picked else None
        self._metrics = checkpoints.metric_columns(records)
        table.clear(columns=True)
        table.add_columns("checkpoint", "type", "step", *self._metrics, "saved", "age")
        now = datetime.now(timezone.utc)
        for index, record in enumerate(records):
            table.add_row(*(Text(cell) for cell in checkpoints.table_row(record, self._metrics, now)), key=str(index))
        table.display = bool(records)
        # The highlight stays on the checkpoint it was on, when a refresh still lists it.
        for index, record in enumerate(records):
            if picked_id is not None and record.get("checkpoint_id") == picked_id:
                table.move_cursor(row=index)
                break
        # The table is hidden until the first read answers, so the tab's focus went to
        # the detail pane (see focus_target); move it to the table now that there are rows
        # to arrow through. Only from that placeholder or from nothing: anywhere else means
        # the focus was deliberately put there (another tab, say), and is not ours to take.
        if records:
            focused = self.screen.focused
            if focused is None or focused is self.query_one("#checkpoints-detail-pane"):
                table.focus()

    def _highlighted(self) -> dict | None:
        if not self._records:
            return None
        table = self.query_one("#checkpoints-table", DataTable)
        row = table.cursor_row
        return self._records[row] if 0 <= row < len(self._records) else self._records[0]

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        event.stop()
        self._draw_detail()

    def redraw(self) -> None:
        status = self.query_one("#checkpoints-status", Static)
        now = datetime.now(timezone.utc)
        if self._records is None:
            text = Text(f"checkpoints unavailable: {self._error}" if self._error else "loading checkpoints…",
                        style=_WARNING if self._error else "")
        else:
            text = Text(checkpoints.summary_text(self._records, now) if self._records
                        else checkpoints.no_checkpoints_text(self._job()))
            if self._loading:
                text.append("  ·  refreshing…", style=CHART_MUTED)
            elif self._error:
                text.append(f"  ·  refresh failed: {self._error}", style=_WARNING)
            elif self._read_at is not None and self._live():
                text.append(f"  ·  read {views.fmt_ago(self._read_at, now)} (r refreshes)", style=CHART_MUTED)
        status.update(text)
        self._draw_detail()

    def _draw_detail(self) -> None:
        detail = self.query_one("#checkpoints-detail", Static)
        record = self._highlighted()
        if record is None:
            detail.update("")
            return
        text = Text()
        for title, lines in checkpoints.detail_lines(record, self._job_id):
            if text:
                text.append("\n")
            text.append(title + "\n", style="bold")
            for line in lines:
                text.append(line + "\n", style="" if not line.startswith("  ") else CHART_AXIS)
        detail.update(text)


# ─── Training metrics ────────────────────────────────────────────────────────

_LOADERS: OrderedDict[str, training.TrainingLoader] = OrderedDict()
_LOADERS_GUARD = threading.Lock()
_MAX_LOADERS = 4


def training_loader(metrics_root: Path) -> training.TrainingLoader:
    """The shared loader for a cache directory, so reopening a job reads only what is new."""
    key = str(Path(metrics_root).resolve())
    with _LOADERS_GUARD:
        loader = _LOADERS.pop(key, None) or training.TrainingLoader(Path(metrics_root))
        _LOADERS[key] = loader
        while len(_LOADERS) > _MAX_LOADERS:
            _LOADERS.popitem(last=False)
        return loader


# Two charts to a row in a pane at least this wide, as on the Resources tab.
_TWO_CHARTS_WIDTH = 90
_CHART_HEIGHT = 6


class TrainingPane(_Pane):
    """One sub-job's training metrics: stats over the window, and a chart per metric against the step."""

    BINDINGS = [
        Binding("w", "cycle_window", "Window"),
        Binding("space", "next_sub_job", "Sub-job", key_display="space"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, client, job_id: str, metrics_root: Path, *, job: Callable[[], dict],
                 download: Callable[..., bool], id: str = "training") -> None:
        super().__init__(client, job_id, job=job, id=id)
        self._metrics_root = Path(metrics_root)
        self._download = download
        self._data = training.TrainingData()
        self._downloaded = False  # a download finished: a sub-job without rows has none
        self._window = training.DEFAULT_WINDOW
        self._pick: str | None = None  # the sub-job Space last showed

    def compose(self) -> ComposeResult:
        yield Static("", id="training-status")
        with VerticalScroll(id="training-pane"):
            yield Static("", id="training-stats")
            yield Static("", id="training-charts")

    def focus_target(self):
        return self.query_one("#training-pane")

    # ── which sub-job ──
    def _sub_job_ids(self) -> list[str]:
        """The job's training sub-jobs, then any other sub-job with training rows."""
        ids = [str(sj.get("sub_job_id")) for sj in views.sub_jobs(self._job())
               if views.sub_job_kind(sj) == "training" and sj.get("sub_job_id")]
        return list(dict.fromkeys(ids + list(self._data.sub_jobs)))

    def _source(self) -> str | None:
        ids = self._sub_job_ids()
        if self._pick in ids:
            return self._pick
        return ids[0] if ids else None

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "next_sub_job":
            return len(self._sub_job_ids()) > 1
        if action == "cycle_window":
            return self._source() is not None
        return True

    def action_next_sub_job(self) -> None:
        ids = self._sub_job_ids()
        if len(ids) < 2:
            return
        current = self._source()
        self._pick = ids[(ids.index(current) + 1) % len(ids)] if current in ids else ids[0]
        self.redraw()

    def action_cycle_window(self) -> None:
        self._window = training.next_window(self._window)
        self.redraw()

    # ── reading ──
    def reload(self) -> None:
        if self._loading:
            return
        if self._job().get("sub_jobs") and not self._sub_job_ids():
            self._downloaded = True  # no training sub-job: nothing to download for this tab
            self.redraw()
            return
        self._loading = True
        self.redraw()
        self._refresh()

    @work(thread=True, exclusive=True, group="training")
    def _refresh(self) -> None:
        worker = get_current_worker()
        loader = training_loader(self._metrics_root)

        def show() -> None:
            data = loader.load()
            if not worker.is_cancelled:
                self._post(self._accept, data)

        try:
            show()  # what is cached, while the download runs
            source = self._source()
            ok = self._download(priority=[source] if source else [], on_commit=lambda _result: show(),
                                cancelled=lambda: worker.is_cancelled)
            if not ok or worker.is_cancelled:
                return
            data = loader.load()
            if not worker.is_cancelled:
                self._post(self._finish, data, None)
        except Exception as exc:  # noqa: BLE001 - said in the tab's header
            if not worker.is_cancelled:
                self._post(self._finish, None, views.access_hint(exc, "training metrics") or views.describe_error(exc))
        finally:
            if worker.is_cancelled:
                self._post(self._cancelled)

    def _accept(self, data: training.TrainingData) -> None:
        self._data = data
        self.redraw()

    def _finish(self, data: training.TrainingData | None, error: str | None) -> None:
        self._loading = False
        self._error = error
        if data is not None:
            self._data = data
        if error is None:
            self._downloaded = True
        self.redraw()

    def _cancelled(self) -> None:
        self._loading = False

    # ── drawing ──
    def _width(self) -> int:
        pane = self.query_one("#training-pane", VerticalScroll)
        return max(20, pane.scrollable_content_region.width - 1)

    def on_resize(self, event) -> None:
        if self._shown:
            self.call_after_refresh(self.redraw)

    def redraw(self) -> None:
        if not self.is_attached:
            return
        status = self.query_one("#training-status", Static)
        stats = self.query_one("#training-stats", Static)
        charts = self.query_one("#training-charts", Static)
        source = self._source()
        sub_job = self._data.get(source)
        status.update(self._status_text(source, sub_job))
        if sub_job is None or not sub_job.metrics:
            stats.update(self._empty_text(source))
            charts.update("")
            return
        first = sub_job.first_step(self._window)
        width = self._width()
        stats.update(self._stats_text(sub_job, first, width))
        charts.update(self._charts_text(sub_job, first, width))

    def _status_text(self, source: str | None, sub_job: training.SubJobTraining | None) -> Text:
        text = Text()
        ids = self._sub_job_ids()
        if source is None:
            text.append("training metrics")
        else:
            text.append(views.sub_job_label(source), style="bold")
            if len(ids) > 1:
                text.append(f" ({ids.index(source) + 1} of {len(ids)}, space for next)", style=CHART_MUTED)
        if sub_job is not None and sub_job.last_step is not None:
            text.append(f"  ·  step {training.format_step(sub_job.last_step)}")
            if sub_job.newest is not None:
                text.append(f"  ·  newest row {views.fmt_ago(sub_job.newest, datetime.now(timezone.utc))}")
        if source is not None:
            text.append(f"  ·  {training.window_label(self._window)} (w)")
        notes = []
        if self._loading:
            notes.append(("cached · refreshing…" if sub_job else "loading training metrics…", CHART_MUTED))
        elif self._error:
            notes.append((f"{'cached · refresh failed' if sub_job else 'training metrics unavailable'}: "
                          f"{self._error}", _WARNING))
        if sub_job is not None and sub_job.skipped:
            notes.append((f"{sub_job.skipped:,} training row{'s' if sub_job.skipped != 1 else ''} skipped "
                          "(malformed or unsupported)", _WARNING))
        text.append("\n")
        for index, (note, style) in enumerate(notes):
            text.append(("  ·  " if index else "") + note, style=style)
        return text

    def _empty_text(self, source: str | None) -> Text:
        job = self._job()
        if source is None:
            if not job.get("sub_jobs"):
                return Text("loading the job…" if not self._downloaded else "this job has no sub-jobs")
            return Text("this job has no training sub-job, so it saves no training metrics")
        entry = next((sj for sj in views.sub_jobs(job) if sj.get("sub_job_id") == source), {})
        status = views.status_word(entry.get("status") or job.get("status"))
        if not views.is_live(job) and status not in ("unknown", "?"):
            status = views.status_word(job.get("status"))
        is_training = not entry or views.sub_job_kind(entry) == "training"
        return Text(training.no_metrics_text(views.sub_job_label(source), status, training=is_training,
                                             downloaded=self._downloaded))

    def _stats_text(self, sub_job: training.SubJobTraining, first: int | None, width: int) -> Text:
        rows = []
        for series in sub_job.metrics:
            found = training.window_stats(*series.window(first))
            if found is None:
                rows.append((series.name, "—", "—", "—"))
                continue
            rows.append((series.name,
                         f"{training.format_metric(found.latest)} @ {training.format_step(found.latest_step)}",
                         f"{training.format_metric(found.low)} @ {training.format_step(found.low_step)}",
                         f"{training.format_metric(found.high)} @ {training.format_step(found.high_step)}"))
        header = ("metric", "latest @ step", "min @ step", "max @ step")
        widths = [max(len(row[i]) for row in [header, *rows]) for i in range(4)]
        text = Text()
        text.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)).rstrip()[:width] + "\n",
                    style=CHART_AXIS)
        for index, row in enumerate(rows):
            text.append(row[0].ljust(widths[0]), style=chart_hue(index))
            text.append(("  " + "  ".join(cell.ljust(widths[i + 1]) for i, cell in enumerate(row[1:]))).rstrip())
            text.append("\n")
        return text

    def _charts_text(self, sub_job: training.SubJobTraining, first: int | None, width: int) -> Text:
        per_row = 2 if width >= _TWO_CHARTS_WIDTH else 1
        gap = 3
        chart_width = (width - gap * (per_row - 1)) // per_row
        blocks = [self._chart_block(series, first, chart_width, chart_hue(index))
                  for index, series in enumerate(sub_job.metrics)]
        text = Text()
        for start in range(0, len(blocks), per_row):
            row = blocks[start:start + per_row]
            height = max(len(block) for block in row)
            text.append("\n")
            for line_index in range(height):
                for column, block in enumerate(row):
                    spans = block[line_index] if line_index < len(block) else []
                    used = sum(len(chunk) for chunk, _ in spans)
                    for chunk, style in spans:
                        text.append(chunk, style=style)
                    if column < len(row) - 1:
                        text.append(" " * (chart_width - used + gap))
                text.append("\n")
        return text

    def _chart_block(self, series: training.MetricSeries, first: int | None, width: int,
                     hue: str) -> list[list[tuple[str, str]]]:
        steps, values = series.window(first)
        latest = training.format_metric(values[-1]) if values else "—"
        title = f"{series.name} · {training.window_label(self._window)}"
        room = width - len(latest) - 1
        title = title if len(title) <= room else title[: max(0, room - 1)] + "…"
        head = [(title, hue), (" " * max(1, width - len(title) - len(latest)), ""), (latest, f"bold {hue}")]
        return [head, *training.chart_spans(series, first, width, height=_CHART_HEIGHT, hue=hue)]
