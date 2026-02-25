from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

import iceoryx2 as iox2
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer as viewer
from mujoco import mjx
from teleop_types import Pose, Twist

_XStateT = TypeVar("_XStateT")
_QStateT = TypeVar("_QStateT")


@jax.jit(donate_argnums=(0,), keep_unused=True)
def mjx_set_data(
    dx,
    ctrl,
    act,
    xfrc_applied,
    qpos,
    qvel,
    time_,
):
    return dx.tree_replace(
        {
            "ctrl": jnp.array(ctrl),
            "act": jnp.array(act),
            "xfrc_applied": jnp.array(xfrc_applied),
            "qpos": jnp.array(qpos),
            "qvel": jnp.array(qvel),
            "time": jnp.array(time_),
        }
    )


@jax.jit(donate_argnums=(1,), keep_unused=True)
def mjx_step(*args, **kwargs):
    return mjx.step(*args, **kwargs)


@dataclass(frozen=True)
class KinematicState(Generic[_XStateT, _QStateT]):
    x: _XStateT
    q: _QStateT
    episode_id: int
    just_reset: bool


class StepControllerTrait:
    """Trait-like base: implement one simulation step update."""

    __slots__ = ()

    def step_control(
        self,
        maybe_twist: Twist | None,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        raise NotImplementedError


class TwistInput:
    def __init__(
        self,
        node,
        twist_subscriber,
        pose_subscriber,
    ) -> None:
        self.node = node
        self.twist_subscriber = twist_subscriber
        self.pose_subscriber = pose_subscriber

    @classmethod
    def create(cls) -> "TwistInput":
        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore
        twist_subscriber = (
            node.service_builder(iox2.ServiceName.new("/twist"))  # type: ignore
            .publish_subscribe(Twist)
            .open_or_create()
            .subscriber_builder()
            .create()
        )

        pose_subscriber = (
            node.service_builder(iox2.ServiceName.new("/pose"))  # type: ignore
            .publish_subscribe(Pose)
            .open_or_create()
            .subscriber_builder()
            .create()
        )
        return cls(
            node=node,
            twist_subscriber=twist_subscriber,
            pose_subscriber=pose_subscriber,
        )

    def latest_twist(self) -> Twist | None:
        maybe_twist_msg = None
        while True:
            temp = self.twist_subscriber.receive()
            if temp is None:
                break
            maybe_twist_msg = temp

        if maybe_twist_msg is None:
            return None
        return maybe_twist_msg.payload().contents


class SimulationCore:
    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
        controller: StepControllerTrait,
    ) -> None:
        self.model = model
        self.data = data
        self.controller = controller
        self.mx = mjx.put_model(model)
        self.dx = mjx.put_data(model, data)

    def step_once(self, maybe_twist: Twist | None) -> float:
        start = time.perf_counter()
        self.controller.step_control(maybe_twist, self.model, self.data)

        self.dx = mjx_set_data(
            self.dx,
            self.data.ctrl,
            self.data.act,
            self.data.xfrc_applied,
            self.data.qpos,
            self.data.qvel,
            self.data.time,
        )
        self.dx = mjx_step(self.mx, self.dx)
        mjx.get_data_into(self.data, self.model, self.dx)
        return time.perf_counter() - start


def sleep_to_timestep(
    timestep: float,
    elapsed: float,
    *,
    log_sleep: bool,
) -> None:
    if elapsed < timestep:
        sleep_duration = timestep - elapsed
        if log_sleep:
            print(f"Sleeping for {sleep_duration:.6f} seconds")
        time.sleep(sleep_duration)


def run_with_mujoco_viewer(
    core: SimulationCore,
    twist_input: TwistInput,
    *,
    log_sleep: bool,
) -> None:
    with viewer.launch_passive(core.model, core.data) as sim_viewer:
        while sim_viewer.is_running():
            maybe_twist = twist_input.latest_twist()
            elapsed = core.step_once(maybe_twist)
            sim_viewer.sync()
            sleep_to_timestep(
                core.model.opt.timestep,
                elapsed,
                log_sleep=log_sleep,
            )
