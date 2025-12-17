from typing import override

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jax.tree_util import Partial
from jaxlie import SE3, SO2, SO3
from jaxtyping import Float

from ...lie_group_kinematics import AbstractManipulator
from ..data_types import SE3SO23, SO23, SO29, Mat3x3, Vec3, Vec9


@jdc.pytree_dataclass(frozen=True)
class SE3SO23SRPlatformKinematics(AbstractManipulator[SE3SO23, Vec9]):
    # 3*7 floats, x-axis as angle 0, z-axis as revolute axis
    revolute_se3_transforms: tuple[float, ...]
    redundant_links_tuple: tuple[float, ...]  # 3 floats

    @property
    def revolute_se3(self) -> SE3:
        return SE3(jnp.array(self.revolute_se3_transforms).reshape(3, 7))

    @property
    def redundant_links(self) -> Vec3:
        return jnp.array(self.redundant_links_tuple)

    @property
    def b_i(self) -> Mat3x3:
        return self.revolute_se3.translation()

    def s_i(self, x: SE3SO23) -> Mat3x3:
        return x.pose.apply(
            (
                self.revolute_se3
                @ SE3.from_rotation(SO3.from_z_radians(x.rdof.log().flatten()))
            ).apply(
                jnp.column_stack(
                    [
                        self.redundant_links,
                        jnp.zeros_like(self.redundant_links),
                        jnp.zeros_like(self.redundant_links),
                    ]
                )
            )
        )

    def ik(self, task_coord: SE3SO23) -> Vec9:
        return self.s_i(task_coord).flatten()

    @override
    def kinematic_constraints(self, task_coord: SE3SO23, joint_coord: Vec9):
        return self.ik(task_coord) - joint_coord

    def loss(self, x: SE3SO23) -> Float:
        jac = self.ik_jacobian(x, self.ik(x))
        return -jnp.log(jnp.linalg.det(jac @ jac.T))

    def loss_grad(self, x: SE3SO23):
        return jaxlie.manifold.grad(Partial(self.loss))(x)


@jdc.pytree_dataclass(frozen=True)
class RRRSerialArmKinematics(AbstractManipulator[Vec3, SO23]):
    """
    4 SE3 to define 3 revolute joints and the end-effector in their previous frame, revolute axes are about z-axis
    """

    t01: SE3
    t12: SE3
    t23: SE3
    t3e: SE3

    def fk(self, joint_coord: SO23) -> Vec3:
        theta1, theta2, theta3 = joint_coord.as_radians().flatten()
        return (
            self.t01
            @ (SE3.from_rotation(SO3.from_z_radians(theta1)))
            @ self.t12
            @ (SE3.from_rotation(SO3.from_z_radians(theta2)))
            @ self.t23
            @ (SE3.from_rotation(SO3.from_z_radians(theta3)))
            @ self.t3e
        ).translation()

    @override
    def kinematic_constraints(self, task_coord: Vec3, joint_coord: SO23) -> Vec3:
        return self.fk(joint_coord) - task_coord


@jdc.pytree_dataclass(frozen=True)
class SE3SO23SRPlatform3RSerialArmKinematics(AbstractManipulator[SE3SO23, SO29]):
    platform: SE3SO23SRPlatformKinematics
    serial_arm: tuple[
        RRRSerialArmKinematics, RRRSerialArmKinematics, RRRSerialArmKinematics
    ]

    @override
    def kinematic_constraints(self, task_coord: SE3SO23, joint_coord: SO29) -> Vec9:
        platform_ik_r9 = self.platform.ik(task_coord)

        leg_fk_r9 = jnp.concatenate(
            [
                arm.fk(SO2(joint_coord.parameters()[i * 3 : (i + 1) * 3]))
                for i, arm in enumerate(self.serial_arm)
            ]
        )

        return platform_ik_r9 - leg_fk_r9
