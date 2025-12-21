import time

import iceoryx2 as iox2
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
import mujoco.viewer as viewer
from jaxlie import SE3, SO3
from mujoco import mjx
from teleop_types import Pose, Twist

from se3_rpkm.data_types import Vec9
from se3_rpkm.linear_redundant_stewart import SE3R3, RedundantR3LegStewartKinematics


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(RedundantR3LegStewartKinematics):
    @jax.jit
    def ik(self, task_coord: SE3R3) -> Vec9:
        return super().ik(task_coord)

    @jax.jit
    def loss_grad(self, x: SE3R3):
        return super().loss_grad(x)


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
    alpha = 70.3e-3
    beta_deg = 45
    h = 10e-3

    deg_120_3 = jnp.array([0.0, 120.0, 240.0])
    rad_120_3 = jnp.deg2rad(deg_120_3)

    DIMENSION = DimensionJIT(
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
    # JIT warm up
    print("Warming up JIT...")
    for _ in range(int(1e3)):
        grad = DIMENSION.loss_grad(x)
        x = SE3R3(pose=x.pose, rdof=x.rdof - 1e-3 * grad.rdof)
    print("JIT warm up done.")

    spec, model, data = DIMENSION.mj_spec_model_data(x)

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

                grad = DIMENSION.loss_grad(x)
                x = SE3R3(
                    pose=x.pose @ SE3.exp(se3_log), rdof=x.rdof - 1e-3 * grad.rdof
                )

                data.ctrl = DIMENSION.ik(x)
                # print(DIMENSION.loss_func(x))
                if (
                    # jnp.isnan(loss)
                    jnp.any(jnp.isnan(jnp.array(data.ctrl)))
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
