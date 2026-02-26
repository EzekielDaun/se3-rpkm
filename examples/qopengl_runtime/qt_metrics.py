from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable

import pyqtgraph as pg
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from qopengl_runtime.qt_simulation_runtime import SimulationStepPayload

"""Qt widgets for streaming simulation metrics with per-episode tabs."""

pg.setConfigOptions(background="w", foreground="k")


@dataclass(frozen=True)
class MetricProvider:
    """Named metric callback used by generic plotting widgets."""

    name: str
    compute_fn: Callable[[SimulationStepPayload], float]

    def metric_names(self) -> list[str]:
        return [self.name]

    def compute_metrics(self, payload: SimulationStepPayload) -> dict[str, float]:
        return {self.name: float(self.compute_fn(payload))}


class MetricsPlotWidget(QWidget):
    def __init__(
        self,
        providers: list[MetricProvider],
        *,
        window_seconds: float,
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self.providers = providers
        self.window_seconds = window_seconds

        self._times: list[float] = []
        self._series: dict[str, list[float]] = {}
        self._twist_active_flags: list[bool] = []

        self.plot = pg.PlotWidget(parent=self)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        self.plot.setLabel("bottom", "wall time", units="s")
        self.plot.setLabel("left", "metric")

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot)

        colors = [
            (26, 188, 156),
            (52, 152, 219),
            (231, 76, 60),
            (243, 156, 18),
            (46, 204, 113),
            (155, 89, 182),
        ]

        self._curves: dict[str, pg.PlotDataItem] = {}
        all_names: list[str] = []
        for provider in providers:
            all_names.extend(provider.metric_names())

        if len(all_names) != len(set(all_names)):
            raise ValueError("Metric names must be unique across providers.")

        self._loss_metric_name: str | None = "loss" if "loss" in all_names else None

        for idx, metric_name in enumerate(all_names):
            color = colors[idx % len(colors)]
            self._series[metric_name] = []
            self._curves[metric_name] = self.plot.plot(
                [],
                [],
                name=metric_name,
                pen=pg.mkPen(color=color, width=2),
            )

        self._twist_curve = self.plot.plot(
            [],
            [],
            name="twist_active",
            pen=pg.mkPen(color=(22, 160, 133, 0), width=1),
            fillLevel=0.0,
            brush=pg.mkBrush(22, 160, 133, 50),
        )

    def on_step_finished(self, payload: SimulationStepPayload) -> None:
        self._times.append(payload.wall_time)

        combined_metrics: dict[str, float] = {}
        for provider in self.providers:
            values = provider.compute_metrics(payload)
            for key, value in values.items():
                if key in combined_metrics:
                    raise ValueError(f"Duplicated metric key: {key}")
                combined_metrics[key] = float(value)

        for metric_name, series in self._series.items():
            series.append(combined_metrics[metric_name])

        if self._loss_metric_name is None:
            self._twist_active_flags.append(False)
        else:
            self._twist_active_flags.append(payload.twist_active)

        newest_time = self._times[-1]
        min_time = newest_time - self.window_seconds
        trim = bisect_left(self._times, min_time)
        if trim > 0:
            self._times = self._times[trim:]
            self._twist_active_flags = self._twist_active_flags[trim:]
            for metric_name in self._series:
                self._series[metric_name] = self._series[metric_name][trim:]

        for metric_name, curve in self._curves.items():
            curve.setData(self._times, self._series[metric_name])
        if self._loss_metric_name is None:
            twist_fill_level = 0.0
            twist_ceiling = 0.0
        else:
            loss_series = self._series[self._loss_metric_name]
            finite_loss_values = [value for value in loss_series if math.isfinite(value)]
            twist_fill_level = min(finite_loss_values) if finite_loss_values else 0.0
            twist_ceiling = max(finite_loss_values) if finite_loss_values else 0.0
        twist_active_series = [
            twist_ceiling if is_active else math.nan
            for is_active in self._twist_active_flags
        ]
        self._twist_curve.setFillLevel(twist_fill_level)
        self._twist_curve.setData(self._times, twist_active_series)


class EpisodeMetricsTabsWidget(QWidget):
    def __init__(
        self,
        providers: list[MetricProvider],
        *,
        window_seconds: float,
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self.providers = providers
        self.window_seconds = window_seconds

        self.tabs = QTabWidget(self)
        self._episode_widgets: dict[int, MetricsPlotWidget] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def _get_or_create_episode_tab(self, episode_id: int) -> MetricsPlotWidget:
        if episode_id in self._episode_widgets:
            return self._episode_widgets[episode_id]

        # Each reset/episode draws into an independent tab.
        episode_widget = MetricsPlotWidget(
            providers=self.providers,
            window_seconds=self.window_seconds,
            parent=self.tabs,
        )
        self._episode_widgets[episode_id] = episode_widget
        self.tabs.addTab(episode_widget, f"ep {episode_id}")
        self.tabs.setCurrentWidget(episode_widget)
        return episode_widget

    def on_step_finished(self, payload: SimulationStepPayload) -> None:
        episode_widget = self._get_or_create_episode_tab(payload.episode_id)
        episode_widget.on_step_finished(payload)

        if payload.just_reset:
            self.tabs.setCurrentWidget(episode_widget)
