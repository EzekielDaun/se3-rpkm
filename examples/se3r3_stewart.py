import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
from jaxlie import SE3, SO3
from simulation_runtime import (
    SimulationCore,
    StepControllerTrait,
    TwistInput,
    run_with_mujoco_viewer,
)
from teleop_types import Twist

from se3_rpkm.data_types import Vec9
from se3_rpkm.linear_redundant_stewart import SE3R3, RedundantR3LegStewartKinematics

JOINT_LIMIT_FACTOR: float = 1e-1


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(RedundantR3LegStewartKinematics):
    @jax.jit
    def ik(self, task_coord: SE3R3) -> Vec9:
        return super().ik(task_coord)

    @jax.jit
    def loss_grad(self, x: SE3R3, joint_limit_factor: float) -> SE3R3:
        return super().loss_grad(x, joint_limit_factor=joint_limit_factor)

    @jax.jit
    def loss(self, x: SE3R3, joint_limit_factor: float) -> float:
        return super().loss(x, joint_limit_factor=joint_limit_factor)


class SE3R3StewartController(StepControllerTrait):
    def __init__(self, dimension: DimensionJIT, initial_x: SE3R3, x0: SE3R3) -> None:
        self.dimension = dimension
        self.x = initial_x
        self.x0 = x0

    def step_control(
        self,
        maybe_twist: Twist | None,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        if maybe_twist is None:
            return

        se3_log = (
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

        grad = self.dimension.loss_grad(self.x, JOINT_LIMIT_FACTOR)
        self.x = SE3R3(
            pose=self.x.pose @ SE3.exp(se3_log),
            rdof=self.x.rdof - 1e-3 * grad.rdof,
        )

        data.ctrl = self.dimension.ik(self.x)

        if (
            jnp.any(jnp.isnan(jnp.array(data.ctrl)))
            or jnp.linalg.norm(self.x.pose.translation() - data.qpos[:3]) > 0.1
            or jnp.linalg.norm(self.x.pose.rotation().parameters() - data.qpos[3:7])
            > 0.1
        ):
            print("Resetting to initial position.")
            mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
            self.x = self.x0


if __name__ == "__main__":
    alpha = 70.3e-3
    beta_deg = 45
    h = 10e-3

    deg_120_3 = jnp.array([0.0, 120.0, 240.0])
    rad_120_3 = jnp.deg2rad(deg_120_3)

    dimension = DimensionJIT(
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
    print("Warming up JIT...")
    for _ in range(int(1e3)):
        grad = dimension.loss_grad(x, JOINT_LIMIT_FACTOR)
        x = SE3R3(pose=x.pose, rdof=x.rdof - 1e-3 * grad.rdof)
    print("JIT warm up done.")

    _spec, model, data = dimension.mj_spec_model_data(x)

    controller = SE3R3StewartController(dimension=dimension, initial_x=x, x0=x0)
    core = SimulationCore(model=model, data=data, controller=controller)
    twist_input = TwistInput.create()
    run_with_mujoco_viewer(core, twist_input, log_sleep=True)
