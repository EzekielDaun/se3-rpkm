import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
from jaxlie import SE3, SO2, SO3
from teleop_types import Twist

from se3_rpkm.data_types import SE3SO23, Vec9
from se3_rpkm.so2_redundant_stewart import SE3SO23StewartKinematics
from simulation_runtime import (
    KinematicState,
    SimulationCore,
    StepControllerTrait,
    TwistInput,
    run_with_mujoco_viewer,
)


@jdc.pytree_dataclass(frozen=True)
class DimensionJIT(SE3SO23StewartKinematics):
    @jax.jit
    def damped_newton_step_fn(
        self, carry: tuple[SE3SO23, float], pose: SE3, factor: float
    ) -> tuple[tuple[SE3SO23, float], SE3SO23]:
        return super().damped_newton_step_fn(carry, pose, factor)

    @jax.jit
    def ik(self, task_coord: SE3SO23) -> Vec9:
        return super().ik(task_coord)

    @jax.jit
    def loss_func(self, x0: SE3SO23) -> float:
        return super().loss_func(x0)


class SE3SO23StewartController(StepControllerTrait):
    def __init__(
        self,
        dimension: DimensionJIT,
        initial_x: SE3SO23,
        initial_q: Vec9,
        x0: SE3SO23,
    ) -> None:
        self.dimension = dimension
        self.x = initial_x
        self.q = initial_q
        self.x0 = x0
        self.q0 = initial_q
        self.episode_id = 0
        self.just_reset = False

    def reset(
        self,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        print("Resetting to initial position.")
        mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
        self.x = self.x0
        self.q = self.q0
        data.ctrl = self.q
        self.episode_id += 1
        self.just_reset = True

    def step_control(
        self,
        maybe_twist: Twist | None,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        self.just_reset = False
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
            (self.x, 0.0), self.x.pose @ SE3.exp(se3_log), factor=1e-2
        )
        data.ctrl = self.dimension.ik(self.x)
        self.q = jnp.array(data.ctrl)

        if (
            jnp.isnan(loss)
            or jnp.any(jnp.isnan(jnp.array(data.ctrl)))
            or jnp.linalg.norm(self.x.pose.translation() - data.qpos[:3]) > 0.1
            or jnp.linalg.norm(self.x.pose.rotation().parameters() - data.qpos[3:7])
            > 0.1
        ):
            self.reset(model, data)

    def get_kinematic_state(self) -> KinematicState[SE3SO23, Vec9]:
        return KinematicState(
            x=self.x,
            q=self.q,
            episode_id=self.episode_id,
            just_reset=self.just_reset,
        )


if __name__ == "__main__":
    l_j = 80e-3 * jnp.ones(3)
    unit = 250e-3

    a21_xyz = jnp.array([unit, 0.75 * -(3**0.5) / 2 * unit, 0])
    a22_xyz = jnp.array([unit, 0.75 * (3**0.5) / 2 * unit, 0])
    a2_xyz = jnp.array([unit, -unit * 3**0.5, 0])

    v2x_xyz = 1 * jnp.array([unit, 0.25 * unit, 0])
    v2_xyz = 1 * jnp.array([unit, -0.25 * unit, 0])

    so3_z_120_dup = SO3.from_z_radians(2 * jnp.pi * jnp.array([1 / 3, 0 / 3, 2 / 3]))

    ai_xyz = so3_z_120_dup.apply(a2_xyz)
    aj1_xyz = so3_z_120_dup.apply(a21_xyz)
    aj2_xyz = so3_z_120_dup.apply(a22_xyz)
    vi_xyz = so3_z_120_dup.apply(v2_xyz)
    vj_xyz = so3_z_120_dup.apply(v2x_xyz)

    dimension = DimensionJIT(
        a_i=ai_xyz,
        v_i=vi_xyz,
        a_j1=aj1_xyz,
        a_j2=aj2_xyz,
        v_j=vj_xyz,
        l_j=l_j,
    )

    x0 = SE3SO23(
        SE3.from_translation(jnp.array([0.0, 0.0, unit * 1.5])),
        SO2.from_radians(jnp.deg2rad(jnp.array([45.0, 45.0, 45.0]))),
    )

    x = x0
    print("Warming up JIT...")
    for _ in range(100):
        (_, _loss), x = dimension.damped_newton_step_fn((x, 0.0), x0.pose, factor=1e-2)
    print("JIT warm up done.")

    _spec, model, data = dimension.mj_spec_model_data(x0)
    q = dimension.ik(x)
    data.ctrl = q

    controller = SE3SO23StewartController(
        dimension=dimension,
        initial_x=x,
        initial_q=q,
        x0=x0,
    )
    core = SimulationCore(model=model, data=data, controller=controller)
    twist_input = TwistInput.create()
    run_with_mujoco_viewer(core, twist_input, log_sleep=True)
