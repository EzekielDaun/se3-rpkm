import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import mujoco
from jaxlie import SE3, SO2, SO3
from simulation_runtime import (
    SimulationCore,
    StepControllerTrait,
    TwistInput,
    run_with_mujoco_viewer,
)
from teleop_types import Twist

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

    def loss(self, x: SE3SO23, q: SO29) -> float:
        ik_jac = self.ik_jacobian(x, q)
        # return -jnp.linalg.slogdet((ik_jac.T) @ ik_jac)[1]
        return jnp.linalg.cond(ik_jac)

    @jax.jit
    def loss_jitted(self, x: SE3SO23, q: SO29) -> float:
        return self.loss(x, q)

    @jax.jit
    def loss_grad(self, x: SE3SO23, q: SO29) -> SE3SO23:
        return jaxlie.manifold.grad(self.loss, argnums=0)(x, q)

    @jax.jit
    def platform_loss_grad(self, x: SE3SO23) -> SE3SO23:
        return self.platform.loss_grad(x)


class SRPlatformRRRSerialController(StepControllerTrait):
    def __init__(self, dimension: JITDimension, x0: SE3SO23, q0: SO29) -> None:
        self.dimension = dimension
        self.x0 = x0
        self.q0 = q0
        self.x = x0
        self.q = q0
        self.se3_log = jnp.zeros(6)

    def reset(
        self,
        model: mujoco.MjModel,  # type: ignore
        data: mujoco.MjData,  # type: ignore
    ) -> None:
        print("Resetting to initial position.")
        mujoco.mj_resetDataKeyframe(model, data, 0)  # type: ignore
        self.x = self.x0
        self.q = self.q0
        data.ctrl = self.q.as_radians().flatten()

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

        grad_rdof = self.dimension.loss_grad(self.x, self.q).rdof.flatten()
        self.x = SE3SO23(
            pose=self.x.pose @ SE3.exp(self.se3_log),
            rdof=self.x.rdof @ SO2.exp(-1e-3 * grad_rdof.reshape(3, 1)),
        )
        self.q = self.dimension.ik_lm_optx(self.x, self.q).normalize()
        data.ctrl = self.q.as_radians().flatten()

        if (
            jnp.any(jnp.isnan(jnp.array(data.ctrl)))
            or jnp.linalg.norm(self.x.pose.translation() - data.qpos[:3]) > 20e-3
            or jnp.linalg.norm(
                (self.x.pose.rotation().inverse() @ SO3(data.qpos[3:7])).log()
            )
            > jnp.deg2rad(10)
        ):
            self.reset(model, data)


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

    _spec, model, data = dimension.mj_spec_model_data(x0, q0)

    controller = SRPlatformRRRSerialController(dimension=dimension, x0=x0, q0=q0)
    core = SimulationCore(model=model, data=data, controller=controller)
    twist_input = TwistInput.create()
    run_with_mujoco_viewer(core, twist_input, log_sleep=True)
