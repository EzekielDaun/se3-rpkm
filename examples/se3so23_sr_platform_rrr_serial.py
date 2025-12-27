import time

import iceoryx2 as iox2
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import mujoco
import mujoco.viewer as viewer
from jaxlie import SE3, SO2, SO3
from mujoco import mjx
from teleop_types import Pose, Twist

from se3_rpkm.data_types import SE3SO23, SO29
from se3_rpkm.sr_platform import (
    RRRSerialArmKinematics,
    SE3SO23SRPlatform3RSerialArmKinematics,
    SE3SO23SRPlatformKinematics,
)


@jdc.pytree_dataclass(frozen=True)
class JITDimension(SE3SO23SRPlatform3RSerialArmKinematics):
    @jax.jit
    def ik_lm_optx(self, task_coord: SE3SO23, joint_coordinate: SO29) -> SO29:
        return super().ik_lm_optx(task_coord, joint_coordinate)

    def loss(self, x: SE3SO23, q: SO29):
        ik_jac = self.ik_jacobian(x, q)
        # return jnp.linalg.cond(ik_jac)
        return -jnp.linalg.slogdet((ik_jac.T) @ ik_jac)[1]

    @jax.jit
    def loss_grad(self, x: SE3SO23, q: SO29) -> SE3SO23:
        return jaxlie.manifold.grad(self.loss, argnums=0)(x, q)

    @jax.jit
    def platform_loss_grad(self, x: SE3SO23) -> SE3SO23:
        return self.platform.loss_grad(x)


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

    dimension = JITDimension(
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

    # spec, model, data = mjcf_spec_platform_and_rrr_serial_model_data(dimension, x0, q0)
    spec, model, data = dimension.mj_spec_model_data(x0, q0)

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
    q = q0
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
                        ]
                    )
                    * model.opt.timestep
                )
                x = SE3SO23(pose=x.pose @ SE3.exp(se3_log), rdof=x.rdof)

            # redundancy resolution
            grad_rdof = dimension.loss_grad(x, q).rdof.flatten()
            x = SE3SO23(
                pose=x.pose @ SE3.exp(se3_log),
                rdof=x.rdof @ SO2.exp(-1e-3 * grad_rdof.reshape(3, 1)),
            )
            q = dimension.ik_lm_optx(x, q).normalize()
            data.ctrl = q.as_radians().flatten()

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
                q = q0

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

            # ## MuJoCo
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
