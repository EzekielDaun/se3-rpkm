from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jax.tree_util import Partial
from jaxlie import SE3, SO2, SO3
from jaxtyping import Float

from .data_types import SE3SO23, SE33, SO23, SO29, Mat3x3, MuJoCoMixin, Vec3, Vec9
from .lie_group_kinematics import AbstractManipulator

try:
    import mujoco

    REVOLUTE_AXIS_CAPSULE_SIZE = [0.01, 0.02, 0.0]
    REVOLUTE_LINK_RADIUS = 0.01
    SITE_RADIUS = 0.015
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    import mujoco as mujoco_t


@jdc.pytree_dataclass(frozen=True)
class SE3SO23SRPlatformKinematics(AbstractManipulator[SE3SO23, Vec9]):
    """
    Kinematics model for an $\\mathrm{SE}(3) \\times \\mathrm{SO}(2)^3$ spatial redundancy platform.

    Coordinates are row vectors.
    """

    # 3*7 floats, x-axis as angle 0, z-axis as revolute axis
    revolute_se3: SE33
    """Batch of 3 $\\mathrm{SE}(3)$ SE3 transforms for 3 revolute joint frames, with local +Z axis as revolute axis, and +X direction as angle 0."""
    redundant_links: Vec3
    """Lengths of the 3 redundant links."""

    @property
    def v_i(self) -> Mat3x3:
        """End-effector revolute axis center points in end-effector frame."""
        return self.revolute_se3.translation()

    def s_i(self, x: SE3SO23) -> Mat3x3:
        """
        Args:
            x (SE3SO23): Task coordinates $\\mathrm{SE}(3) \\times \\mathrm{SO}(2)^3$.

        Returns:
            Mat3x3: Redundant link tip positions in world frame.
        """
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
        """
        Args:
            task_coord (SE3SO23): Task coordinates $\\mathrm{SE}(3) \\times \\mathrm{SO}(2)^3$.

        Returns:
            Vec9: Joint coordinates (stacked redundant link tip positions, sequence: x1, y1, z1, x2, y2, z2, x3, y3, z3).
        """
        return self.s_i(task_coord).flatten()

    @override
    def kinematic_constraints(self, task_coord: SE3SO23, joint_coord: Vec9):
        return self.ik(task_coord) - joint_coord

    def loss(self, x: SE3SO23) -> Float:
        """
        Args:
            x (SE3SO23): Task coordinates.

        Returns:
            Float: Negative log-determinant of the IK Jacobian metric.
        """
        jac = self.ik_jacobian(x, self.ik(x))
        return -jnp.log(jnp.linalg.det(jac @ jac.T))

    def loss_grad(self, x: SE3SO23):
        """
        Args:
            x (SE3SO23): Task coordinates.

        Returns:
            SE3SO23: Gradient of the loss with respect to task coordinates.
        """
        return jaxlie.manifold.grad(Partial(self.loss))(x)


def mjcf_spec_platform(dimension: SE3SO23SRPlatformKinematics) -> "mujoco_t.MjSpec":  # type: ignore
    """
    Args:
        dimension (SE3SO23SRPlatformKinematics): Platform kinematic parameters.

    Returns:
        mujoco_t.MjSpec: MJCF spec for the platform and a list of link sites.
    """
    MuJoCoMixin._check_mujoco_availability()

    spec = mujoco.MjSpec()  # type: ignore
    spec.modelname = "se3so23_sr_platform"

    # Globally Disable Contact
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

    # End-Effector
    ee_body = spec.worldbody.add_body(name="body_ee")
    ee_body.add_site(
        size=[SITE_RADIUS] * 3,
        rgba=[0.0, 1.0, 0.0, 1],
    )
    # ee_body.add_freejoint()

    ## give EE a triangle plate mesh
    vertices_top = dimension.v_i + jnp.array([0, 0, 0.01])
    vertices_bottom = dimension.v_i - jnp.array([0, 0, 0])
    vertices = jnp.vstack([vertices_top, vertices_bottom]).astype(jnp.float32)
    mesh = spec.add_mesh(name="triangle_mesh")
    mesh.uservert = vertices.flatten().tolist()
    triangle_geom = ee_body.add_geom()
    triangle_geom.type = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
    triangle_geom.meshname = mesh.name

    ## attach redundant links to ee
    site_list = []
    for i in range(3):
        link_body = ee_body.add_body(
            name=f"body_link{i + 1}",
            pos=dimension.revolute_se3.translation()[i].tolist(),
            quat=dimension.revolute_se3.rotation().parameters()[i].tolist(),
        )
        link_body.add_joint(
            name=f"joint_link{i + 1}",
            type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
            pos=[0, 0, 0],
            axis=[0, 0, 1],
        )
        link_body.add_geom(  # revolute axis
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=REVOLUTE_AXIS_CAPSULE_SIZE,
            rgba=[0.5, 0.5, 0.5, 1],
        )

        link_body.add_geom(  # revolute link
            name=f"link_geom{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=[REVOLUTE_LINK_RADIUS, dimension.redundant_links[i] / 2, 0.0],
            rgba=[0.5, 0.5, 0.5, 1],
            pos=[dimension.redundant_links[i] / 2, 0, 0],
            quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
        )
        site_list.append(
            link_body.add_site(
                name=f"site_link{i + 1}",
                pos=[dimension.redundant_links[i], 0, 0],
                size=[SITE_RADIUS] * 3,
                rgba=[0, 0.5, 0, 1],
            )
        )
    return spec, site_list


@jdc.pytree_dataclass(frozen=True)
class SE3SO23SRPlatformGantryKinematics(SE3SO23SRPlatformKinematics, MuJoCoMixin):
    """SR platform kinematics with 3 gantry legs."""

    @override
    def mj_spec(self) -> "mujoco_t.MjSpec":  # type: ignore
        """
        Returns:
            mujoco_t.MjSpec: The MJCF specification of the mechanism.
        """
        MuJoCoMixin._check_mujoco_availability()

        spec, site_list = mjcf_spec_platform(self)

        # Origin Site
        origin_site = spec.worldbody.add_site(
            name="site_origin",
            pos=[0.0, 0.0, 0.0],
            size=[SITE_RADIUS] * 3,
            rgba=[0.5, 0.5, 0.5, 1.0],
        )

        # Actuated Sites
        for index, s in enumerate(site_list):
            GAIN = 1.2e5
            free_body = spec.worldbody.add_body(
                mass=1,
                inertia=jnp.array([1.0, 1.0, 1.0]),  # type: ignore
            )
            free_body.add_freejoint()
            leg_site = free_body.add_site(
                name=f"leg_site_{index + 1}",
                size=[SITE_RADIUS] * 3,
                rgba=[0.0, 0.0, 1.0, 1],
            )

            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = s.name
            eq.name2 = leg_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore

            spec.add_actuator(
                name=f"act_{index + 1}_x",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_SITE,  # type: ignore
                gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                target=leg_site.name,
                refsite=origin_site.name,
            )
            spec.add_actuator(
                name=f"act_{index + 1}_y",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_SITE,  # type: ignore
                gear=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                target=leg_site.name,
                refsite=origin_site.name,
            )
            spec.add_actuator(
                name=f"act_{index + 1}_z",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_SITE,  # type: ignore
                gear=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                target=leg_site.name,
                refsite=origin_site.name,
            )

        return spec

    @override
    def mj_spec_model_data(
        self, x0: SE3SO23
    ) -> tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]:  # type: ignore
        """
        Args:
            x0 (SE3SO23): Initial task coordinates.

        Returns:
            tuple[mujoco_t.MjSpec, mujoco_t.MjModel, mujoco_t.MjData]: MJCF spec, model,
            and data initialized at the provided configuration.
        """
        MuJoCoMixin._check_mujoco_availability()

        spec = self.mj_spec()
        spec.body("body_ee").pos = x0.pose.translation()
        spec.body("body_ee").quat = x0.pose.rotation().parameters()
        model = spec.compile()
        data = mujoco.MjData(model)  # type: ignore
        data.ctrl = jnp.array(self.ik(x0))
        for i in range(int(1e5)):
            mujoco.mj_step(model, data)  # type: ignore
            if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-6:
                break
        if i >= int(1e5) - 1:
            raise RuntimeError("Failed to settle the simulation for initial position.")

        spec.body("body_ee").add_freejoint()

        key_frame = spec.add_key()  # this should be keyframe 0
        key_frame.name = "home"
        model, data0 = spec.recompile(model, data)
        key_frame.qpos = data0.qpos.copy()
        key_frame.ctrl = data0.ctrl.copy()
        model, data0 = spec.recompile(model, data0)
        return spec, model, data0


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
class SE3SO23SRPlatform3RSerialArmKinematics(
    AbstractManipulator[SE3SO23, SO29], MuJoCoMixin
):
    """Kinematic of SR platform with three 3R serial arms as legs."""

    platform: SE3SO23SRPlatformKinematics
    serial_arm: tuple[
        RRRSerialArmKinematics, RRRSerialArmKinematics, RRRSerialArmKinematics
    ]

    @override
    def kinematic_constraints(self, task_coord: SE3SO23, joint_coord: SO29) -> Vec9:
        """
        Args:
            task_coord (SE3SO23): Task coordinates.
            joint_coord (SO29): Joint coordinates for the three 3R arms.

        Returns:
            Vec9: Constraint residuals between platform IK and arm FK.
        """
        platform_ik_r9 = self.platform.ik(task_coord)

        leg_fk_r9 = jnp.concatenate(
            [
                arm.fk(SO2(joint_coord.parameters()[i * 3 : (i + 1) * 3]))
                for i, arm in enumerate(self.serial_arm)
            ]
        )

        return platform_ik_r9 - leg_fk_r9

    @override
    def mj_spec(self) -> "mujoco_t.MjSpec":  # type: ignore
        """
        Returns:
            mujoco_t.MjSpec: The MJCF specification of the mechanism.
        """
        MuJoCoMixin._check_mujoco_availability()
        spec, site_list = mjcf_spec_platform(self.platform)
        GAIN = 70

        for index_arm, (arm, platform_link_site) in enumerate(
            zip(self.serial_arm, site_list)
        ):
            # Body 1
            body_1 = spec.worldbody.add_body(
                pos=arm.t01.translation(),
                quat=arm.t01.rotation().parameters(),
            )
            ## Revolute Axis 1
            body_1.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=REVOLUTE_AXIS_CAPSULE_SIZE,
                rgba=[0.0, 0.0, 1.0, 1.0],
            )
            body_1.add_joint(
                name=f"arm_{index_arm + 1}_joint_1",
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                pos=[0, 0, 0],
                axis=[0, 0, 1],
            )
            spec.add_actuator(
                name=f"act_{index_arm + 1}1",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                target=f"arm_{index_arm + 1}_joint_1",
            )
            ## Revolute Link 1
            body_1.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                pos=[arm.t12.translation()[0] / 2, 0, 0],
                quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
                size=[
                    REVOLUTE_LINK_RADIUS,
                    jnp.linalg.norm(arm.t12.translation()) / 2 + mujoco.mjMINVAL,  # type: ignore
                    0.0,
                ],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )

            # Body 2
            body_2 = body_1.add_body(
                pos=arm.t12.translation(),
                quat=arm.t12.rotation().parameters(),
            )
            ## Revolute Axis 2
            body_2.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=REVOLUTE_AXIS_CAPSULE_SIZE,
                rgba=[0.0, 0.0, 1.0, 1.0],
            )
            body_2.add_joint(
                name=f"arm_{index_arm + 1}_joint_2",
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                pos=[0, 0, 0],
                axis=[0, 0, 1],
            )
            spec.add_actuator(
                name=f"act_{index_arm + 1}2",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                target=f"arm_{index_arm + 1}_joint_2",
            )
            ## Revolute Link 2
            body_2.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                pos=[arm.t23.translation()[0] / 2, 0, 0],
                quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
                size=[
                    REVOLUTE_LINK_RADIUS,
                    jnp.linalg.norm(arm.t23.translation()) / 2 + mujoco.mjMINVAL,  # type: ignore
                    0.0,
                ],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )

            # Body 3
            body_3 = body_2.add_body(
                pos=arm.t23.translation(),
                quat=arm.t23.rotation().parameters(),
            )
            ## Revolute Axis 3
            body_3.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=REVOLUTE_AXIS_CAPSULE_SIZE,
                rgba=[0.0, 0.0, 1.0, 1.0],
            )
            body_3.add_joint(
                name=f"arm_{index_arm + 1}_joint_3",
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                pos=[0, 0, 0],
                axis=[0, 0, 1],
            )
            spec.add_actuator(
                name=f"act_{index_arm + 1}3",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                target=f"arm_{index_arm + 1}_joint_3",
            )
            ## Revolute Link 3
            body_3.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                pos=[arm.t3e.translation()[0] / 2, 0, 0],
                quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
                size=[
                    REVOLUTE_LINK_RADIUS,
                    jnp.linalg.norm(arm.t3e.translation()) / 2 + mujoco.mjMINVAL,  # type: ignore
                    0.0,
                ],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            s = body_3.add_site(
                name=f"arm_{index_arm + 1}_ee_site",
                pos=arm.t3e.translation(),
                size=[SITE_RADIUS] * 3,
                rgba=[1.0, 0.0, 0.0, 1.0],
            )

            # Connect to Platform
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = s.name
            eq.name2 = platform_link_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore
        return spec

    @override
    def mj_spec_model_data(
        self, x0: SE3SO23, q0: SO29
    ) -> tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]:  # type:ignore
        """
        Args:
            x0 (SE3SO23): Initial task coordinates.
            q0 (SO29): Initial joint coordinates.

        Returns:
            tuple[mujoco_t.MjSpec, mujoco_t.MjModel, mujoco_t.MjData]: MJCF spec, model,
            and data initialized at the provided configuration.
        """
        MuJoCoMixin._check_mujoco_availability()

        spec = self.mj_spec()
        spec.body("body_ee").pos = x0.pose.translation()
        spec.body("body_ee").quat = x0.pose.rotation().parameters()
        model = spec.compile()
        data = mujoco.MjData(model)  # type: ignore
        data.ctrl = q0.as_radians().flatten()
        for i in range(int(1e5)):
            mujoco.mj_step(model, data)  # type: ignore
            if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-6:
                break
        if i >= int(1e5) - 1:
            raise RuntimeError("Failed to settle the simulation for initial position.")

        spec.body("body_ee").add_freejoint()

        key_frame = spec.add_key()  # this should be keyframe 0
        key_frame.name = "home"
        model, data0 = spec.recompile(model, data)
        key_frame.qpos = data0.qpos.copy()
        key_frame.ctrl = data0.ctrl.copy()
        model, data0 = spec.recompile(model, data0)
        return spec, model, data0
