import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
from jaxlie import SE3, SO2, SO3
from simulation_runtime import (
    SimulationCore,
    StepControllerTrait,
    TwistInput,
    run_with_mujoco_viewer,
)
from teleop_types import Twist

from se3_rpkm.data_types import SE3SO22, Vec8
from se3_rpkm.so2_redundant_stewart import SE3SO22StewartKinematics


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(SE3SO22StewartKinematics):
    @jax.jit
    def damped_newton_step_fn(
        self, carry: tuple[SE3SO22, float], pose: SE3, factor: float
    ) -> tuple[tuple[SE3SO22, float], SE3SO22]:
        return super().damped_newton_step_fn(carry, pose, factor)

    @jax.jit
    def ik(self, task_coord: SE3SO22) -> Vec8:
        return super().ik(task_coord)

    @jax.jit
    def loss_func(self, x0: SE3SO22) -> float:
        return super().loss_func(x0)


class SE3SO22StewartController(StepControllerTrait):
    def __init__(
        self, dimension: DimensionJIT, initial_x: SE3SO22, x0: SE3SO22
    ) -> None:
        self.dimension = dimension
        self.x = initial_x
        self.x0 = x0

    def reset(
        self,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        print("Resetting to initial position.")
        mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
        self.x = self.x0
        data.ctrl = self.dimension.ik(self.x)

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
        (_, loss), self.x = self.dimension.damped_newton_step_fn(
            (self.x, 0.0), self.x.pose @ SE3.exp(se3_log), factor=3e-2
        )
        data.ctrl = self.dimension.ik(self.x)

        if (
            jnp.isnan(loss)
            or jnp.any(jnp.isnan(jnp.array(data.ctrl)))
            or jnp.linalg.norm(self.x.pose.translation() - data.qpos[:3]) > 0.1
            or jnp.linalg.norm(self.x.pose.rotation().parameters() - data.qpos[3:7])
            > 0.1
        ):
            self.reset(model, data)


if __name__ == "__main__":
    beta = 2.25 * 2e-1
    rot_90 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
    rot_180 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 2, endpoint=False))
    dimension = DimensionJIT(
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
    print("Warming up JIT...")
    for _ in range(100):
        (_, _loss), x = dimension.damped_newton_step_fn((x, 0.0), x0.pose, factor=1e-2)
    print("JIT warm up done.")

    _spec, model, data = dimension.mj_spec_model_data(
        x0, act_lower_length=0.4, act_upper_length=0.7
    )

    controller = SE3SO22StewartController(dimension=dimension, initial_x=x, x0=x0)
    core = SimulationCore(model=model, data=data, controller=controller)
    twist_input = TwistInput.create()
    run_with_mujoco_viewer(core, twist_input, log_sleep=True)
