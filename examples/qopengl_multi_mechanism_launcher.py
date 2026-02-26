from __future__ import annotations

from io import TextIOBase
import sys
from dataclasses import dataclass
from typing import Callable, TextIO

import jax.numpy as jnp
from jaxlie import SE3, SO2, SO3
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qopengl_runtime.qopengl_renderer import CameraConfig, MuJoCoOpenGLWidget
from qopengl_runtime.qt_metrics import EpisodeMetricsTabsWidget, MetricProvider
from qopengl_runtime.qt_simulation_runtime import (
    QtSimulationDriver,
)
from se3r3_stewart import DimensionJIT as SE3R3Dimension
from se3r3_stewart import JOINT_LIMIT_FACTOR as SE3R3_JOINT_LIMIT_FACTOR
from se3r3_stewart import SE3R3StewartController
from se3so3_5pss_s_4pss import JITDimension as SE3SO3Dimension
from se3so3_5pss_s_4pss import SE3SO3_5PSS_4PSSController
from se3so22_stewart import DimensionJIT as SE3SO22Dimension
from se3so22_stewart import SE3SO22StewartController
from se3so23_sr_platform_basic_continuous import (
    DimensionJIT as SRPlatformBasicDimension,
)
from se3so23_sr_platform_basic_continuous import (
    SRPlatformBasicContinuousController,
)
from se3so23_sr_platform_rrr_serial import (
    JITDimension as SRPlatformRRRSerialDimension,
)
from se3so23_sr_platform_rrr_serial import (
    SRPlatformRRRSerialController,
)
from se3so23_stewart import DimensionJIT as SE3SO23Dimension
from se3so23_stewart import SE3SO23StewartController
from simulation_runtime import KinematicState, SimulationCore, TwistInput

from se3_rpkm.data_types import SE3SO3, SE3SO22, SE3SO23
from se3_rpkm.linear_redundant_stewart import SE3R3
from se3_rpkm.sr_platform import (
    RRRSerialArmKinematics,
    SE3SO23SRPlatformKinematics,
)

METRIC_WINDOW_SECONDS: float = 10.0


class LogTextEmitter(QObject):
    text_written = Signal(str)


class DockTextStream(TextIOBase):
    def __init__(
        self,
        emitter: LogTextEmitter,
        original_stream: TextIO,
        stream_label: str,
    ) -> None:
        super().__init__()
        self.emitter = emitter
        self.original_stream = original_stream
        self.stream_label = stream_label

    def _decorate_text(self, text: str) -> str:
        if self.stream_label == "":
            return text

        lines = text.splitlines(keepends=True)
        return "".join(
            f"{self.stream_label}{line}" if line.strip() != "" else line
            for line in lines
        )

    def write(self, text: str) -> int:
        if text == "":
            return 0

        decorated = self._decorate_text(text)
        self.emitter.text_written.emit(decorated)

        try:
            self.original_stream.write(text)
            self.original_stream.flush()
        except Exception:
            pass

        return len(text)

    def flush(self) -> None:
        try:
            self.original_stream.flush()
        except Exception:
            pass

    def writable(self) -> bool:
        return True


class EpisodeStateTracker:
    def __init__(
        self,
        core: SimulationCore,
        x_getter,
        q_getter,
    ) -> None:
        self.core = core
        self.x_getter = x_getter
        self.q_getter = q_getter
        self.episode_id = 0
        self.last_sim_time = float(core.data.time)

    def __call__(self) -> KinematicState:
        sim_time = float(self.core.data.time)
        just_reset = sim_time + 1e-12 < self.last_sim_time
        if just_reset:
            self.episode_id += 1
        self.last_sim_time = sim_time

        return KinematicState(
            x=self.x_getter(),
            q=self.q_getter(),
            episode_id=self.episode_id,
            just_reset=just_reset,
        )


@dataclass
class MechanismSession:
    core: SimulationCore
    widget: QWidget
    driver: QtSimulationDriver


SessionBuilder = Callable[[TwistInput], MechanismSession]


@dataclass(frozen=True)
class MechanismDescriptor:
    name: str
    build: SessionBuilder


def _build_session_widget(
    core: SimulationCore,
    driver: QtSimulationDriver,
    metric_providers: list[MetricProvider],
) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    splitter = QSplitter(Qt.Orientation.Vertical)
    opengl_widget = MuJoCoOpenGLWidget(
        core=core,
        camera=CameraConfig(
            azimuth=90.0,
            elevation=-20.0,
            distance_scale=2.0,
            lookat=None,
        ),
        enable_mouse_controls=True,
    )
    metric_tabs_widget = EpisodeMetricsTabsWidget(
        metric_providers,
        window_seconds=METRIC_WINDOW_SECONDS,
        parent=container,
    )

    splitter.addWidget(opengl_widget)
    splitter.addWidget(metric_tabs_widget)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([750, 250])

    driver.step_finished.connect(opengl_widget.on_step_finished)
    driver.step_finished.connect(metric_tabs_widget.on_step_finished)

    layout.addWidget(splitter)
    return container


def _build_se3so23_stewart_session(twist_input: TwistInput) -> MechanismSession:
    l_j = 80e-3 * jnp.ones(3)
    unit = 250e-3

    a21_xyz = jnp.array([unit, 0.75 * -(3**0.5) / 2 * unit, 0])
    a22_xyz = jnp.array([unit, 0.75 * (3**0.5) / 2 * unit, 0])
    a2_xyz = jnp.array([unit, -unit * 3**0.5, 0])

    v2x_xyz = jnp.array([unit, 0.25 * unit, 0])
    v2_xyz = jnp.array([unit, -0.25 * unit, 0])

    so3_z_120_dup = SO3.from_z_radians(2 * jnp.pi * jnp.array([1 / 3, 0 / 3, 2 / 3]))

    ai_xyz = so3_z_120_dup.apply(a2_xyz)
    aj1_xyz = so3_z_120_dup.apply(a21_xyz)
    aj2_xyz = so3_z_120_dup.apply(a22_xyz)
    vi_xyz = so3_z_120_dup.apply(v2_xyz)
    vj_xyz = so3_z_120_dup.apply(v2x_xyz)

    dimension = SE3SO23Dimension(
        a_i=ai_xyz,
        v_i=vi_xyz,
        a_j1=aj1_xyz,
        a_j2=aj2_xyz,
        v_j=vj_xyz,
        l_j=l_j,
    )

    x0 = SE3SO23(
        SE3.from_translation(jnp.array([0.0, 0.0, unit * 1.5])),
        SO2.from_radians(jnp.deg2rad(jnp.array([45.0, 45.0, 45.0]))),
    )

    x = x0
    print("[se3so23_stewart] Warming up JIT...")
    for _ in range(100):
        (_, _loss), x = dimension.damped_newton_step_fn((x, 0.0), x0.pose, factor=1e-2)

    _spec, model, data = dimension.mj_spec_model_data(x0)
    q = dimension.ik(x)
    data.ctrl = q

    controller = SE3SO23StewartController(
        dimension=dimension,
        initial_x=x,
        initial_q=q,
        x0=x0,
    )
    core = SimulationCore(model=model, data=data, controller=controller)

    state_getter = EpisodeStateTracker(
        core,
        x_getter=lambda: controller.x,
        q_getter=lambda: controller.q,
    )
    driver = QtSimulationDriver(
        core,
        twist_input,
        state_getter=state_getter,
        parent=None,
    )

    providers = [
        MetricProvider("loss", lambda p: dimension.loss_func(p.x)),
    ]
    widget = _build_session_widget(core, driver, providers)
    return MechanismSession(core=core, widget=widget, driver=driver)


def _build_se3so22_stewart_session(twist_input: TwistInput) -> MechanismSession:
    beta = 2.25 * 2e-1
    rot_90 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
    rot_180 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 2, endpoint=False))
    dimension = SE3SO22Dimension(
        a_i=rot_90.inverse().apply(jnp.array([beta, beta, 0.0])),
        a_j1=rot_180.apply(jnp.array([beta, beta, 0.0])),
        a_j2=rot_180.apply(jnp.array([beta, -beta, 0.0])),
        v_i=jnp.array(
            [
                [0.0, beta, 0.0],
                [0.0, -beta, 0.0],
                [0.0, -beta, 0.0],
                [0.0, beta, 0.0],
            ]
        )
        / 2.25,
        v_j=rot_180.apply(jnp.array([beta, 0.0, 0.0]) / 2.25),
        l_j=jnp.array([beta, beta]) * 0.15,
    )
    x0 = SE3SO22(
        SE3.from_translation(jnp.array([0.0, 0.0, beta * 1.5])),
        SO2.from_radians(jnp.deg2rad(jnp.array([45.0, 135.0]))),
    )

    x = x0
    print("[se3so22_stewart] Warming up JIT...")
    for _ in range(100):
        (_, _loss), x = dimension.damped_newton_step_fn((x, 0.0), x0.pose, factor=1e-2)

    _spec, model, data = dimension.mj_spec_model_data(
        x0,
        act_lower_length=0.4,
        act_upper_length=0.7,
    )
    data.ctrl = dimension.ik(x)

    controller = SE3SO22StewartController(dimension=dimension, initial_x=x, x0=x0)
    core = SimulationCore(model=model, data=data, controller=controller)

    state_getter = EpisodeStateTracker(
        core,
        x_getter=lambda: controller.x,
        q_getter=lambda: jnp.array(core.data.ctrl),
    )
    driver = QtSimulationDriver(
        core,
        twist_input,
        state_getter=state_getter,
        parent=None,
    )

    providers = [
        MetricProvider("loss", lambda p: dimension.loss_func(p.x)),
    ]
    widget = _build_session_widget(core, driver, providers)
    return MechanismSession(core=core, widget=widget, driver=driver)


def _build_se3r3_stewart_session(twist_input: TwistInput) -> MechanismSession:
    alpha = 70.3e-3
    beta_deg = 45
    h = 10e-3

    deg_120_3 = jnp.array([0.0, 120.0, 240.0])
    rad_120_3 = jnp.deg2rad(deg_120_3)

    dimension = SE3R3Dimension(
        v_i1=SO3.from_z_radians(jnp.deg2rad(50.0 + deg_120_3)).apply(
            jnp.array([[alpha, 0.0, 0.0]])
        ),
        v_i2=SO3.from_z_radians(jnp.deg2rad(-50.0 + deg_120_3)).apply(
            jnp.array([[alpha, 0.0, 0.0]])
        ),
        r_i_se3=SE3.from_rotation(SO3.from_z_radians(rad_120_3))
        @ SE3.from_rotation_and_translation(
            SO3.from_y_radians(jnp.deg2rad(-beta_deg)),
            jnp.array([100e-3, 0.0, 0.0]),
        ),
        a_i1_in_r=jnp.array([[0, h, 0]] * 3),
        a_i2_in_r=jnp.array([[0, -h, 0]] * 3),
        r_i_lower_limits=-0.1 * jnp.ones(3),
        r_i_upper_limits=0.1 * jnp.ones(3),
    )

    x0 = SE3R3(
        pose=SE3.from_translation(jnp.array([0, 0, 0.2])),
        rdof=jnp.ones(3) * 0,
    )

    x = x0
    print("[se3r3_stewart] Warming up JIT...")
    for _ in range(int(1e3)):
        grad = dimension.loss_grad(x, SE3R3_JOINT_LIMIT_FACTOR)
        x = SE3R3(pose=x.pose, rdof=x.rdof - 1e-3 * grad.rdof)

    _spec, model, data = dimension.mj_spec_model_data(x)
    data.ctrl = dimension.ik(x)

    controller = SE3R3StewartController(dimension=dimension, initial_x=x, x0=x0)
    core = SimulationCore(model=model, data=data, controller=controller)

    state_getter = EpisodeStateTracker(
        core,
        x_getter=lambda: controller.x,
        q_getter=lambda: jnp.array(core.data.ctrl),
    )
    driver = QtSimulationDriver(
        core,
        twist_input,
        state_getter=state_getter,
        parent=None,
    )

    providers = [
        MetricProvider(
            "loss", lambda p: dimension.loss(p.x, SE3R3_JOINT_LIMIT_FACTOR)
        ),
    ]
    widget = _build_session_widget(core, driver, providers)
    return MechanismSession(core=core, widget=widget, driver=driver)


def _build_sr_platform_basic_session(twist_input: TwistInput) -> MechanismSession:
    dimension = SRPlatformBasicDimension(
        revolute_se3=(
            SE3.from_rotation(
                SO3.from_z_radians(jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3]))
            )
            @ SE3.from_translation(jnp.array([0.5, 0.0, 0.0]))
        ),
        redundant_links=jnp.array([0.2] * 3),
    )

    x0 = SE3SO23(
        pose=SE3.identity(),
        rdof=SO2.from_radians(jnp.deg2rad(jnp.array([90.0, 90.0, 90.0]))),
    )

    _spec, model, data = dimension.mj_spec_model_data(x0)
    data.ctrl = dimension.ik(x0)

    controller = SRPlatformBasicContinuousController(dimension=dimension, x0=x0)
    core = SimulationCore(model=model, data=data, controller=controller)

    state_getter = EpisodeStateTracker(
        core,
        x_getter=lambda: controller.x,
        q_getter=lambda: jnp.array(core.data.ctrl),
    )
    driver = QtSimulationDriver(
        core,
        twist_input,
        state_getter=state_getter,
        parent=None,
    )

    providers = [
        MetricProvider("loss", lambda p: dimension.loss(p.x)),
    ]
    widget = _build_session_widget(core, driver, providers)
    return MechanismSession(core=core, widget=widget, driver=driver)


def _build_sr_platform_rrr_serial_session(
    twist_input: TwistInput,
) -> MechanismSession:
    arm_dimension_1 = RRRSerialArmKinematics(
        t01=SE3.from_rotation_and_translation(
            rotation=SO3.from_y_radians(-jnp.deg2rad(120)),
            translation=jnp.array([250e-3, 0.0, 0.0]),
        ),
        t12=SE3.from_rotation(SO3.from_x_radians(jnp.pi / 2)),
        t23=SE3.from_translation(jnp.array([300e-3, 0.0, 0.0])),
        t3e=SE3.from_translation(jnp.array([150e-3, 0.0, 0.0])),
    )
    arm_dimension_2 = RRRSerialArmKinematics(
        t01=SE3.from_rotation(SO3.from_z_radians(2 * jnp.pi / 3)) @ arm_dimension_1.t01,
        t12=arm_dimension_1.t12,
        t23=arm_dimension_1.t23,
        t3e=arm_dimension_1.t3e,
    )
    arm_dimension_3 = RRRSerialArmKinematics(
        t01=SE3.from_rotation(SO3.from_z_radians(4 * jnp.pi / 3)) @ arm_dimension_1.t01,
        t12=arm_dimension_1.t12,
        t23=arm_dimension_1.t23,
        t3e=arm_dimension_1.t3e,
    )

    platform_dimension = SE3SO23SRPlatformKinematics(
        revolute_se3=(
            SE3.from_rotation(
                SO3.from_z_radians(jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3]))
            )
            @ SE3.from_translation(jnp.array([125e-3, 0.0, 0.0]))
        ),
        redundant_links=jnp.array([50e-3] * 3),
    )

    dimension = SRPlatformRRRSerialDimension(
        platform=platform_dimension,
        serial_arm=(arm_dimension_1, arm_dimension_2, arm_dimension_3),
    )

    x0 = SE3SO23(
        pose=SE3.from_translation(jnp.array([0.0, 0.0, 250e-3])),
        rdof=SO2.from_radians(jnp.deg2rad(jnp.array([45.0, 45.0, 45.0]))),
    )
    q0 = SO2.from_radians(jnp.array([0.0, -jnp.pi / 2, 0.0] * 3))
    for _ in range(10):
        q0 = dimension.ik_lm_optx(x0, q0).normalize()

    _spec, model, data = dimension.mj_spec_model_data(x0, q0)
    data.ctrl = q0.as_radians().flatten()

    controller = SRPlatformRRRSerialController(dimension=dimension, x0=x0, q0=q0)
    core = SimulationCore(model=model, data=data, controller=controller)

    state_getter = EpisodeStateTracker(
        core,
        x_getter=lambda: controller.x,
        q_getter=lambda: controller.q,
    )
    driver = QtSimulationDriver(
        core,
        twist_input,
        state_getter=state_getter,
        parent=None,
    )

    providers = [
        MetricProvider("loss", lambda p: dimension.loss_jitted(p.x, p.q)),
    ]
    widget = _build_session_widget(core, driver, providers)
    return MechanismSession(core=core, widget=widget, driver=driver)


def _build_se3so3_5pss_4pss_session(twist_input: TwistInput) -> MechanismSession:
    dimension = SE3SO3Dimension(
        slider_axis=SE3.from_translation(
            SO3.from_z_radians(
                jnp.deg2rad(jnp.array([-50, -40, 40, 50, 130, 140, 180, 220, 230]))
            ).apply(jnp.array([[250e-3, 0.0, 0.0]]))
        ),
        link_length=0.3 * jnp.ones(9),
        a1_a5=SO3.from_z_radians(jnp.deg2rad(jnp.array([-80, -5, 5, 80, 100]))).apply(
            jnp.array([[150e-3, 0.0, 0.0]])
        ),
        a6_a9=SO3.from_z_radians(jnp.deg2rad(jnp.array([110, 180, 185, 240]))).apply(
            jnp.array([[150e-3, 0.0, 0.0]])
        ),
    )

    x0 = SE3SO3(pose=SE3.identity(), rdof=SO3.identity())
    q0 = jnp.ones(9) * 0.1
    for _ in range(10):
        q0 = dimension.ik_lm_optx(x0, q0)

    _spec, model, data = dimension.mj_spec_model_data(x0, q0)
    data.ctrl = q0

    controller = SE3SO3_5PSS_4PSSController(dimension=dimension, x0=x0, q0=q0)
    core = SimulationCore(model=model, data=data, controller=controller)

    state_getter = EpisodeStateTracker(
        core,
        x_getter=lambda: controller.x,
        q_getter=lambda: controller.q,
    )
    driver = QtSimulationDriver(
        core,
        twist_input,
        state_getter=state_getter,
        parent=None,
    )

    providers = [
        MetricProvider("loss", lambda p: dimension.loss(p.x, p.q)),
    ]
    widget = _build_session_widget(core, driver, providers)
    return MechanismSession(core=core, widget=widget, driver=driver)


class QOpenGLMechanismLauncher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RPKM QOpenGL Launcher")

        self.twist_input = TwistInput.create()

        self.registry: list[MechanismDescriptor] = [
            MechanismDescriptor("se3so23_stewart", _build_se3so23_stewart_session),
            MechanismDescriptor("se3so22_stewart", _build_se3so22_stewart_session),
            MechanismDescriptor("se3r3_stewart", _build_se3r3_stewart_session),
            MechanismDescriptor(
                "se3so23_sr_platform_basic_continuous",
                _build_sr_platform_basic_session,
            ),
            MechanismDescriptor(
                "se3so23_sr_platform_rrr_serial",
                _build_sr_platform_rrr_serial_session,
            ),
            MechanismDescriptor("se3so3_5pss_s_4pss", _build_se3so3_5pss_4pss_session),
        ]

        self.tabs = QTabWidget(self)
        self._tab_containers: list[QWidget] = []
        self._sessions: dict[int, MechanismSession] = {}

        for descriptor in self.registry:
            container = QWidget(self.tabs)
            layout = QVBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            placeholder = QLabel(
                f"Select tab to initialize: {descriptor.name}",
                parent=container,
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder)

            self._tab_containers.append(container)
            self.tabs.addTab(container, descriptor.name)

        self.setCentralWidget(self.tabs)
        self.resize(1800, 1000)
        self.statusBar().showMessage("Initializing...")
        self._setup_log_dock()
        self.manual_reset_button = QPushButton("Reset", self)
        self.manual_reset_button.clicked.connect(self._on_manual_reset_clicked)
        self.statusBar().addPermanentWidget(self.manual_reset_button)

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self._refresh_status_bar)
        self.status_timer.start()

        self.tabs.currentChanged.connect(self._on_current_tab_changed)
        self._on_current_tab_changed(self.tabs.currentIndex())

    def _setup_log_dock(self) -> None:
        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_output.document().setMaximumBlockCount(5000)

        self.log_dock = QDockWidget("Console", self)
        self.log_dock.setObjectName("console_dock")
        self.log_dock.setWidget(self.log_output)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        self.log_emitter = LogTextEmitter(self)
        self.log_emitter.text_written.connect(self._append_log_text)

        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        self.stdout_redirect = DockTextStream(
            self.log_emitter,
            self.stdout_original,
            "",
        )
        self.stderr_redirect = DockTextStream(
            self.log_emitter,
            self.stderr_original,
            "[stderr] ",
        )
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect

    @Slot(str)
    def _append_log_text(self, text: str) -> None:
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _on_current_tab_changed(self, index: int) -> None:
        if index < 0:
            return

        # Build a mechanism session on first activation to reduce startup cost.
        if index not in self._sessions:
            self._initialize_session(index)

        # Only advance the active tab; inactive tabs keep their last plotted history.
        for tab_index, session in self._sessions.items():
            if tab_index == index:
                session.driver.start()
            else:
                session.driver.stop()
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        index = self.tabs.currentIndex()
        if index < 0:
            self.statusBar().showMessage("No active simulation.")
            return

        session = self._sessions.get(index)
        if session is None:
            descriptor_name = self.registry[index].name
            self.statusBar().showMessage(f"{descriptor_name}: initializing...")
            return

        descriptor_name = self.registry[index].name
        mujoco_dt_ms = session.driver.mujoco_timestep * 1000.0
        latest_elapsed = session.driver.latest_elapsed
        if latest_elapsed is None:
            elapsed_text = "--"
        else:
            elapsed_text = f"{latest_elapsed * 1000.0:.3f} ms"

        self.statusBar().showMessage(
            (
                f"{descriptor_name} | mujoco dt: {mujoco_dt_ms:.3f} ms | "
                f"actual step dt: {elapsed_text}"
            )
        )

    def _on_manual_reset_clicked(self) -> None:
        index = self.tabs.currentIndex()
        if index < 0:
            return

        session = self._sessions.get(index)
        if session is None:
            return

        session.core.reset()
        self._refresh_status_bar()

    def _initialize_session(self, index: int) -> None:
        descriptor = self.registry[index]
        session = descriptor.build(self.twist_input)
        self._sessions[index] = session

        container = self._tab_containers[index]
        layout = container.layout()
        if layout is None:
            layout = QVBoxLayout(container)

        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()  # type: ignore
            if widget is not None:
                widget.deleteLater()

        layout.addWidget(session.widget)

    def closeEvent(self, event: QCloseEvent) -> None:
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original
        self.status_timer.stop()
        for session in self._sessions.values():
            session.driver.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QOpenGLMechanismLauncher()
    window.show()
    sys.exit(app.exec())
