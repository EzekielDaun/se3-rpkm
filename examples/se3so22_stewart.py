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

from se3_rpkm.data_types import SE3SO22, Vec8
from se3_rpkm.so2_redundant_stewart import SE3SO22StewartKinematics


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(SE3SO22StewartKinematics):
    @jax.jit
    def damped_newton_step_fn(
        self, carry: tuple[SE3SO22, float], pose: SE3, factor: float
    ):
        return super().damped_newton_step_fn(carry, pose, factor)

    @jax.jit
    def ik(self, task_coord: SE3SO22) -> Vec8:
        return super().ik(task_coord)


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
    beta = 2.25 * 2e-1
    rot_90 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
    rot_180 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 2, endpoint=False))
    DIMENSION = DimensionJIT(
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
        SE3.from_translation(jnp.array([0.0, 0.0, beta * 2.5])),
        SO2.from_radians(jnp.deg2rad(jnp.array([45.0, 135.0]))),
    )

    x = x0
    # JIT warm up
    print("Warming up JIT...")
    for _ in range(100):
        (_, loss), x = DIMENSION.damped_newton_step_fn((x, 0.0), x0.pose, factor=1e-2)
    print("JIT warm up done.")

    spec, model, data = DIMENSION.mj_spec_model_data(x0)

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

    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
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
                        [twist.vx, twist.vy, twist.vz, twist.wx, twist.wy, twist.wz]
                    )
                    * model.opt.timestep
                )
                (_, loss), x = DIMENSION.damped_newton_step_fn(
                    (x, 0.0), x.pose @ SE3.exp(se3_log), factor=3e-2
                )
                data.ctrl = DIMENSION.ik(x)

                if (
                    jnp.isnan(loss)
                    or jnp.any(jnp.isnan(jnp.array(data.ctrl)))
                    or jnp.linalg.norm(x.pose.translation() - data.qpos[:3]) > 0.1
                    or jnp.linalg.norm(x.pose.rotation().parameters() - data.qpos[3:7])
                    > 0.1
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
            # dx_batch = jax.vmap(lambda _: dx.replace())(jnp.zeros(1000))
            # dx_batch = jax.vmap(lambda _: dx.replace())(jnp.zeros(1))
            dx = mjx_step(mx, dx)
            # dx_batch = jit_step(mx, dx_batch)
            # mjx.get_data_into(data, model, dx_batch[0])
            mjx.get_data_into(data, model, dx)
            viewer.sync()

            ## MuJoCo
            # mujoco.mj_step(model, data)  # type: ignore
            # viewer.sync()

            elapsed = time.perf_counter() - start
            if elapsed < model.opt.timestep:
                print(f"Sleeping for {model.opt.timestep - elapsed:.6f} seconds")
                time.sleep(model.opt.timestep - elapsed)
            else:
                print(
                    f"Step took {elapsed:.6f} seconds, which is longer than timestep."
                )
