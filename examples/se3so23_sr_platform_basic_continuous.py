import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
from jaxlie import SE3, SO2, SO3
from teleop_types import Twist

from se3_rpkm.data_types import SE3SO23, Vec9
from se3_rpkm.sr_platform import SE3SO23SRPlatformGantryKinematics
from simulation_runtime import (
    SimulationCore,
    StepControllerTrait,
    TwistInput,
    run_with_mujoco_viewer,
)


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(SE3SO23SRPlatformGantryKinematics):
    @jax.jit
    def ik(self, task_coord: SE3SO23) -> Vec9:
        return super().ik(task_coord)

    @jax.jit
    def loss_grad(self, x: SE3SO23) -> SE3SO23:
        return super().loss_grad(x)

    @jax.jit
    def loss(self, x: SE3SO23) -> float:
        return super().loss(x)


class SRPlatformBasicContinuousController(StepControllerTrait):
    def __init__(self, dimension: DimensionJIT, x0: SE3SO23) -> None:
        self.dimension = dimension
        self.x0 = x0
        self.x = x0
        self.se3_log = jnp.zeros(6)

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
            self.x = SE3SO23(pose=self.x.pose @ SE3.exp(self.se3_log), rdof=self.x.rdof)

        self.x = SE3SO23(
            pose=self.x.pose,
            rdof=self.x.rdof
            @ SO2.exp(model.opt.timestep * jnp.deg2rad(jnp.array([60, 0, 0])).reshape(3, 1)),
        )

        grad_rdof = self.dimension.loss_grad(self.x).rdof.flatten()
        update_rdof = -1e-3 * jnp.array([0, grad_rdof[1] * 50, grad_rdof[2]])

        self.x = SE3SO23(
            pose=self.x.pose @ SE3.exp(self.se3_log),
            rdof=self.x.rdof @ SO2.exp(update_rdof.reshape(3, 1)),
        )

        data.ctrl = self.dimension.ik(self.x)

        if (
            jnp.any(jnp.isnan(jnp.array(data.ctrl)))
            or jnp.linalg.norm(self.x.pose.translation() - data.qpos[:3]) > 20e-3
            or jnp.linalg.norm((self.x.pose.rotation().inverse() @ SO3(data.qpos[3:7])).log())
            > jnp.deg2rad(10)
        ):
            print("Resetting to initial position.")
            mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
            self.x = self.x0


if __name__ == "__main__":
    dimension = DimensionJIT(
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

    controller = SRPlatformBasicContinuousController(dimension=dimension, x0=x0)
    core = SimulationCore(model=model, data=data, controller=controller)
    twist_input = TwistInput.create()
    run_with_mujoco_viewer(core, twist_input, log_sleep=True)
