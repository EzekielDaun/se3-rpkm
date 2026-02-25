from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Generic, TypeVar

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from simulation_runtime import (
    KinematicState,
    SimulationCore,
    TwistInput,
)
from teleop_types import Twist

"""Qt-driven simulation loop helpers and cross-widget step signals."""

_PayloadX = TypeVar("_PayloadX")
_PayloadQ = TypeVar("_PayloadQ")
TWIST_ZERO_EPS: float = 5e-2


def _is_idle_twist(maybe_twist: Twist | None) -> bool:
    if maybe_twist is None:
        return True

    values = (
        maybe_twist.vx,
        maybe_twist.vy,
        maybe_twist.vz,
        maybe_twist.wx,
        maybe_twist.wy,
        maybe_twist.wz,
    )
    return all(abs(float(value)) <= TWIST_ZERO_EPS for value in values)


@dataclass(frozen=True)
class SimulationStepPayload(Generic[_PayloadX, _PayloadQ]):
    step_idx: int
    sim_time: float
    wall_time: float
    elapsed: float
    twist_active: bool
    x: _PayloadX
    q: _PayloadQ
    episode_id: int
    just_reset: bool


class QtSimulationDriver(QObject, Generic[_PayloadX, _PayloadQ]):
    step_finished = Signal(object)

    def __init__(
        self,
        core: SimulationCore,
        twist_source: TwistInput,
        *,
        state_getter: Callable[[], KinematicState[_PayloadX, _PayloadQ]],
        parent: QObject | None,
    ) -> None:
        super().__init__(parent)
        self.core = core
        self.twist_source = twist_source
        self.state_getter = state_getter
        self.step_idx = 0
        self._latest_elapsed: float | None = None
        self._running = False
        self._wall_start: float | None = None
        self._paused_total: float = 0.0
        self._paused_at: float | None = None

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._on_timer)

    def start(self) -> None:
        if self._running:
            return
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        elif self._paused_at is not None:
            self._paused_total += now - self._paused_at
            self._paused_at = None

        self._running = True
        self._schedule_next(0)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.timer.stop()
        self._paused_at = time.perf_counter()

    @property
    def mujoco_timestep(self) -> float:
        return float(self.core.model.opt.timestep)

    @property
    def latest_elapsed(self) -> float | None:
        return self._latest_elapsed

    def _schedule_next(self, delay_ms: int) -> None:
        if not self._running:
            return
        self.timer.start(max(0, delay_ms))

    def _current_wall_time(self) -> float:
        if self._wall_start is None:
            return 0.0

        now = time.perf_counter()
        paused_total = self._paused_total
        if self._paused_at is not None:
            paused_total += now - self._paused_at
        return max(0.0, now - self._wall_start - paused_total)

    def _on_timer(self) -> None:
        if not self._running:
            return

        maybe_twist = self.twist_source.latest_twist()
        elapsed = self.core.step_once(maybe_twist)
        self._latest_elapsed = elapsed
        twist_active = not _is_idle_twist(maybe_twist)

        state = self.state_getter()
        # Emit one payload per finished step for renderers/metrics/widgets.
        payload = SimulationStepPayload(
            step_idx=self.step_idx,
            sim_time=float(self.core.data.time),
            wall_time=self._current_wall_time(),
            elapsed=elapsed,
            twist_active=twist_active,
            x=state.x,
            q=state.q,
            episode_id=state.episode_id,
            just_reset=state.just_reset,
        )
        self.step_idx += 1
        self.step_finished.emit(payload)

        remaining = self.mujoco_timestep - elapsed
        delay_ms = int(remaining * 1000.0) if remaining > 0.0 else 0
        self._schedule_next(delay_ms)
