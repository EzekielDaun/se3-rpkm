from __future__ import annotations

from dataclasses import dataclass

import mujoco
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy
from simulation_runtime import SimulationCore

"""MuJoCo rendering widget based on Qt's QOpenGLWidget."""
# pyright: reportAttributeAccessIssue=false


@dataclass
class CameraConfig:
    azimuth: float
    elevation: float
    distance_scale: float
    lookat: tuple[float, float, float] | None


class MuJoCoOpenGLWidget(QOpenGLWidget):
    """OpenGL widget that renders the current MuJoCo scene from SimulationCore."""

    def __init__(
        self,
        core: SimulationCore,
        *,
        camera: CameraConfig,
        enable_mouse_controls: bool,
    ) -> None:
        super().__init__()
        self.core = core
        self.camera_config = camera
        self.enable_mouse_controls = enable_mouse_controls

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(self.core.model, maxgeom=10_000)
        self.context = None

        self._drag_button: Qt.MouseButton | None = None
        self._last_mouse_pos: tuple[float, float] | None = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def initializeGL(self) -> None:
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        mujoco.mjv_defaultPerturb(self.pert)
        mujoco.mjv_defaultFreeCamera(self.core.model, self.cam)

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = self.camera_config.azimuth
        self.cam.elevation = self.camera_config.elevation
        self.cam.distance = (
            self.camera_config.distance_scale * self.core.model.stat.extent
        )
        if self.camera_config.lookat is None:
            self.cam.lookat[:] = self.core.model.stat.center
        else:
            self.cam.lookat[:] = self.camera_config.lookat

        self.context = mujoco.MjrContext(
            self.core.model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW.value, self.context)

    def paintGL(self) -> None:
        if self.context is None:
            return

        # MuJoCo expects framebuffer pixels, while Qt gives logical widget size.
        dpr = self.devicePixelRatioF()
        viewport = mujoco.MjrRect(
            0,
            0,
            int(round(self.width() * dpr)),
            int(round(self.height() * dpr)),
        )
        mujoco.mjv_updateScene(
            self.core.model,
            self.core.data,
            self.opt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)

    def resizeGL(self, _w: int, _h: int) -> None:
        self.update()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.context is not None:
            self.makeCurrent()
            self.context.free()  # type: ignore[attr-defined]
            self.doneCurrent()
            self.context = None
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.enable_mouse_controls:
            return super().mousePressEvent(event)

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._drag_button = event.button()
            pos = event.position()
            self._last_mouse_pos = (pos.x(), pos.y())
            self.setFocus()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.enable_mouse_controls:
            return super().mouseMoveEvent(event)

        if self._drag_button is None or self._last_mouse_pos is None:
            super().mouseMoveEvent(event)
            return

        pos = event.position()
        x, y = pos.x(), pos.y()
        lx, ly = self._last_mouse_pos
        self._last_mouse_pos = (x, y)

        reldx = (x - lx) / max(1.0, float(self.width()))
        reldy = (y - ly) / max(1.0, float(self.height()))

        if self._drag_button == Qt.MouseButton.LeftButton:
            mujoco.mjv_moveCamera(
                self.core.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_H.value,
                reldx,
                0.0,
                self.scene,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.core.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V.value,
                0.0,
                reldy,
                self.scene,
                self.cam,
            )
        elif self._drag_button == Qt.MouseButton.RightButton:
            mujoco.mjv_moveCamera(
                self.core.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_H.value,
                reldx,
                0.0,
                self.scene,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.core.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_V.value,
                0.0,
                reldy,
                self.scene,
                self.cam,
            )
        elif self._drag_button == Qt.MouseButton.MiddleButton:
            mujoco.mjv_moveCamera(
                self.core.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM.value,
                0.0,
                reldy,
                self.scene,
                self.cam,
            )

        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self.enable_mouse_controls:
            return super().mouseReleaseEvent(event)

        if event.button() == self._drag_button:
            self._drag_button = None
            self._last_mouse_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.enable_mouse_controls:
            return super().wheelEvent(event)

        wheel_steps = event.angleDelta().y() / 120.0
        if wheel_steps != 0.0:
            mujoco.mjv_moveCamera(
                self.core.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM.value,
                0.0,
                -0.05 * wheel_steps,
                self.scene,
                self.cam,
            )
            self.update()
        event.accept()

    def on_step_finished(self, _payload) -> None:
        self.update()
