import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import mujoco
import numpy as np
from jaxlie import SE3, SO3
from simulation_runtime import (
    SimulationCore,
    StepControllerTrait,
    TwistInput,
    run_with_mujoco_viewer,
)
from teleop_types import Twist

from se3_rpkm.data_types import SE3SO3, Vec9
from se3_rpkm.se3so3_5pss_s_4pss import SE3SO3_5PSS_S_4PSS_Kinematics


@jdc.pytree_dataclass(frozen=True)
class JITDimension(SE3SO3_5PSS_S_4PSS_Kinematics):
    @jax.jit
    def ik_lm_optx(self, task_coord: SE3SO3, joint_coordinate: Vec9) -> Vec9:
        return super().ik_lm_optx(task_coord, joint_coordinate)

    @jax.jit
    def loss(self, x: SE3SO3, q: Vec9) -> float:
        ik_jac = self.ik_jacobian(x, q)
        # return jnp.linalg.cond(ik_jac)
        return -jnp.linalg.slogdet((ik_jac.T) @ ik_jac)[1]

    @jax.jit
    def loss_grad(self, x: SE3SO3, q: Vec9) -> SE3SO3:
        return jaxlie.manifold.grad(self.loss, argnums=0)(x, q)


class SE3SO3_5PSS_4PSSController(StepControllerTrait):
    def __init__(self, dimension: JITDimension, x0: SE3SO3, q0: Vec9) -> None:
        self.dimension = dimension
        self.x0 = x0
        self.q0 = q0
        self.x = x0
        self.q = q0
        self.se3_log = jnp.zeros(6)
        self.rdof_grad_step = 2e-4
        self.enable_stochastic_rdof = True
        self.rdof_noise_init = 3e-3
        self.rdof_noise_decay = 5e-4
        self.rdof_update_norm_clip = 8e-3
        self.step_count = 0
        self.rng: np.random.Generator = np.random.default_rng(0)

    def reset(
        self,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        print("Resetting to initial position.")
        mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
        self.x = self.x0
        self.q = self.q0
        self.step_count = 0
        data.ctrl = self.q

    def step_control(
        self,
        maybe_twist: Twist | None,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        if maybe_twist is not None:
            self.se3_log = (
                jnp.array(
                    [
                        maybe_twist.vx,
                        maybe_twist.vy,
                        maybe_twist.vz,
                        maybe_twist.wx,
                        maybe_twist.wy,
                        maybe_twist.wz,
                    ]
                )
                * model.opt.timestep
            )
            self.x = SE3SO3(pose=self.x.pose @ SE3.exp(self.se3_log), rdof=self.x.rdof)

        grad_rdof = self.dimension.loss_grad(self.x, self.q).rdof.flatten()
        rdof_update = -self.rdof_grad_step * grad_rdof
        if self.enable_stochastic_rdof:
            noise_scale = self.rdof_noise_init / jnp.sqrt(
                1.0 + self.rdof_noise_decay * self.step_count
            )
            noise = jnp.asarray(self.rng.standard_normal(3), dtype=grad_rdof.dtype)
            rdof_update = rdof_update + noise_scale * noise

        update_norm = jnp.linalg.norm(rdof_update)
        update_scale = jnp.minimum(
            1.0,
            self.rdof_update_norm_clip / (update_norm + 1e-12),
        )
        rdof_update = rdof_update * update_scale

        self.x = SE3SO3(
            pose=self.x.pose @ SE3.exp(self.se3_log),
            rdof=self.x.rdof @ SO3.exp(rdof_update),
        )
        self.step_count += 1
        self.q = self.dimension.ik_lm_optx(self.x, self.q)
        data.ctrl = self.q

        if (
            jnp.any(jnp.isnan(jnp.array(data.ctrl)))
            or jnp.linalg.norm(self.x.pose.translation() - data.qpos[:3]) > 25e-3
            or jnp.linalg.norm((self.x.pose.rotation().inverse() @ SO3(data.qpos[3:7])).log())
            > jnp.deg2rad(15)
        ):
            self.reset(model, data)


if __name__ == "__main__":
    dimension = JITDimension(
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

    controller = SE3SO3_5PSS_4PSSController(dimension=dimension, x0=x0, q0=q0)
    core = SimulationCore(model=model, data=data, controller=controller)
    twist_input = TwistInput.create()
    run_with_mujoco_viewer(core, twist_input, log_sleep=False)
