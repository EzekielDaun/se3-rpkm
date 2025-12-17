import time

import iceoryx2 as iox2
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
import mujoco.viewer as viewer
from jaxlie import SE3, SO2, SO3
from mujoco import mjx
from teleop_types import Pose, Twist

from se3_rpkm.se3so23.data_types import SE3SO23, Vec9
from se3_rpkm.se3so23.sr_platform.core import SE3SO23SRPlatformKinematics
from se3_rpkm.se3so23.sr_platform.mujoco_sim import (
    mjcf_spec_platform_and_gantry_model_data,
)


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(SE3SO23SRPlatformKinematics):
    @jax.jit
    def ik(self, task_coord: SE3SO23) -> Vec9:
        return super().ik(task_coord)

    @jax.jit
    def loss_grad(self, x: SE3SO23) -> SE3SO23:
        return super().loss_grad(x)

    @jax.jit
    def loss(self, x: SE3SO23) -> float:
        return super().loss(x)


# MJX Simulation
@jax.jit(donate_argnums=(0,), keep_unused=True)
def mjx_set_data(dx, ctrl, act, xfrc_applied, qpos, qvel, time_):
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


if __name__ == "__main__":
    dimension = DimensionJIT(
        revolute_se3_transforms=tuple(
            (
                SE3.from_rotation(
                    SO3.from_z_radians(jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3]))
                )
                @ SE3.from_translation(jnp.array([0.5, 0.0, 0.0]))
            )
            .parameters()
            .flatten()
            .tolist()
        ),
        redundant_links_tuple=tuple([0.2] * 3),
    )

    x0 = SE3SO23(
        pose=SE3.identity(),
        rdof=SO2.from_radians(jnp.deg2rad(jnp.array([90.0, 90.0, 90.0]))),
    )

    spec, model, data = mjcf_spec_platform_and_gantry_model_data(dimension, x0)

    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)

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

    x = x0
    se3_log = jnp.zeros(6)
    with viewer.launch_passive(model, data) as viewer:
        last_update_instant = time.perf_counter()
        while viewer.is_running():
            start = time.perf_counter()
            # twist control logic
            maybe_twist = None
            while True:
                temp = twist_subscriber.receive()
                if temp is None:
                    break
                else:
                    maybe_twist = temp
            if maybe_twist is None:
                pass
            else:
                twist: Twist = maybe_twist.payload().contents
                se3_log = (
                    jnp.array(
                        [
                            twist.vx,
                            twist.vy,
                            twist.vz,
                            twist.wx,
                            twist.wy,
                            twist.wz,
                        ],
                        dtype=jnp.float64,
                    )
                    * model.opt.timestep
                )
                x = SE3SO23(pose=x.pose @ SE3.exp(se3_log), rdof=x.rdof)

            # rdof 1 increment
            x = SE3SO23(
                pose=x.pose,
                rdof=x.rdof
                @ SO2.exp(
                    model.opt.timestep
                    * jnp.deg2rad(jnp.array([60, 0, 0])).reshape(3, 1)
                ),
            )

            # redundancy resolution
            grad_rdof = dimension.loss_grad(x).rdof.flatten()
            update_rdof = -1e-3 * jnp.array(
                [
                    0,  # first rdof is externally controlled
                    grad_rdof[1] * 50,  # magic: imbalance makes it better
                    grad_rdof[2],
                ]
            )

            x = SE3SO23(
                pose=x.pose @ SE3.exp(se3_log),
                rdof=x.rdof @ SO2.exp(update_rdof.reshape(3, 1)),
            )

            data.ctrl = dimension.ik(x)

            if (
                # redundancy failed
                jnp.any(jnp.isnan(jnp.array(data.ctrl)))
                # translational error
                or jnp.linalg.norm(x.pose.translation() - data.qpos[:3]) > 20e-3
                # rotational error
                or jnp.linalg.norm(
                    (x.pose.rotation().inverse() @ SO3(data.qpos[3:7])).log()
                )
                > jnp.deg2rad(10)
            ):
                print("Resetting to initial position.")
                mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
                x = x0

            # MuJoCo step
            ## MJX
            dx = mjx_set_data(
                dx,
                data.ctrl,
                data.act,
                data.xfrc_applied,
                data.qpos,
                data.qvel,
                data.time,
            )
            dx = mjx_step(mx, dx)
            mjx.get_data_into(data, model, dx)
            viewer.sync()

            ## MuJoCo
            # mujoco.mj_step(model, data)  # type: ignore
            # viewer.sync()

            # timing control
            elapsed = time.perf_counter() - start
            if elapsed < model.opt.timestep:
                print(f"Sleeping for {model.opt.timestep - elapsed:.6f} seconds")
                time.sleep(model.opt.timestep - elapsed)
            else:
                print(
                    f"Step took {elapsed:.6f} seconds, which is longer than timestep."
                )
                pass
