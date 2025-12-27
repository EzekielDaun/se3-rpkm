from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jax.tree_util import Partial
from jaxlie import SE3
from jaxtyping import Float

from .data_types import (
    SE3R3,
    SE33,
    Mat3x3,
    MuJoCoMixin,
    Vec3,
    Vec9,
)
from .lie_group_kinematics import AbstractManipulator

try:
    import mujoco

    STRUCTURAL_CAPSULE_RADIUS = 2.5e-3
    STRUCTURAL_CAPSULE_SIZE = [STRUCTURAL_CAPSULE_RADIUS, 0.02, 0.0]
    REVOLUTE_LINK_RADIUS = 10e-3
    SITE_RADIUS = 15e-3
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    import mujoco as mujoco_t


@jdc.pytree_dataclass(frozen=True, slots=True)
class RedundantR3LegStewartKinematics(AbstractManipulator[SE3R3, Vec9], MuJoCoMixin):
    """
    Kinematics model for a Stewart platform with 3 redundant linear actuators, each connects two Stewart legs, namely, suffix `i1` and suffix `i2`.

    Coordinates are row vectors.
    """

    v_i1: Mat3x3
    """Attachment points on the end-effector for leg set `i1`, in end-effector frame."""
    v_i2: Mat3x3
    """Attachment points on the end-effector for leg set `i2`, in end-effector frame."""
    r_i_se3: SE33
    """Batch of 3 SE3 transforms from the base frame to the redundant prismatic joint frames, the redundant sliding is along local Z-axis."""
    a_i1_in_r: Mat3x3
    """Attachment points on the redundant prismatic joint frames for leg set `i1`, in redundant prismatic joint frames."""
    a_i2_in_r: Mat3x3
    """Attachment points on the redundant prismatic joint frames for leg set `i2`, in redundant prismatic joint frames."""
    r_i_lower_limits: Vec3
    """Lower limits for the redundant prismatic joints."""
    r_i_upper_limits: Vec3
    """Upper limits for the redundant prismatic joints."""

    def r_i(self, rdof: Vec3) -> SE33:
        """
        Args:
            rdof (Vec3): Redundant DOF for the 3 redundant prismatic joints.

        Returns:
            SE33: Batch of 3 SE3 transforms from the base frame to the redundant prismatic joint frames at given redundant DOF.
        """
        return self.r_i_se3 @ SE3.from_translation(jnp.zeros((3, 3)).at[:, 2].set(rdof))

    def a_i1(self, rdof: Vec3) -> Mat3x3:
        """
        Args:
            rdof (Vec3): Redundant DOF for the 3 redundant prismatic joints.

        Returns:
            Mat3x3: Root attachment points for leg set `i1` at given redundant DOF, in world frame.
        """
        return (self.r_i(rdof) @ SE3.from_translation(self.a_i1_in_r)).translation()

    def a_i2(self, rdof: Vec3) -> Mat3x3:
        """
        Args:
            rdof (Vec3): Redundant DOF for the 3 redundant prismatic joints.

        Returns:
            Mat3x3: Root attachment points for leg set `i2` at given redundant DOF, in world frame.
        """
        return (self.r_i(rdof) @ SE3.from_translation(self.a_i2_in_r)).translation()

    def b_i1(self, pose: SE3) -> Mat3x3:
        """
        Args:
            pose (SE3): End-effector pose.

        Returns:
            Mat3x3: End-effector attachment points for leg set `i1` at given end-effector pose, in world frame.
        """
        return pose.apply(self.v_i1)

    def b_i2(self, pose: SE3) -> Mat3x3:
        """
        Args:
            pose (SE3): End-effector pose.

        Returns:
            Mat3x3: End-effector attachment points for leg set `i2` at given end-effector pose, in world frame.
        """
        return pose.apply(self.v_i2)

    def l_i1(self, x: SE3R3) -> Vec3:
        """
        Args:
            x (SE3R3): Task coordinates $\\mathrm{SE}(3) \\times \\mathbb{R}^3$

        Returns:
            Vec3: Lengths of leg set `i1`.
        """
        a_i1 = self.a_i1(x.rdof)
        b_i1 = self.b_i1(x.pose)
        return jnp.linalg.norm(a_i1 - b_i1, axis=1)

    def l_i2(self, x: SE3R3) -> Vec3:
        """
        Args:
            x (SE3R3): Task coordinates $\\mathrm{SE}(3) \\times \\mathbb{R}^3$

        Returns:
            Vec3: Lengths of leg set `i2`.
        """
        a_i2 = self.a_i2(x.rdof)
        b_i2 = self.b_i2(x.pose)
        return jnp.linalg.norm(a_i2 - b_i2, axis=1)

    def ik(self, x: SE3R3) -> Vec9:
        """
        Args:
            x (SE3R3): Task coordinates $\\mathrm{SE}(3) \\times \\mathbb{R}^3$

        Returns:
            Vec9: Joint coordinates $\\mathbb{R}^9$ (3 lengths of leg set `i1` + 3 lengths of leg set `i2` + 3 redundant prismatic joints).
        """
        l1 = self.l_i1(x)
        l2 = self.l_i2(x)
        return jnp.concatenate([l1, l2, x.rdof])

    @override
    def kinematic_constraints(self, x: SE3R3, q: Vec9) -> Vec9:
        return self.ik(x) - q

    def loss(self, x: SE3R3, joint_limit_factor=1e-1) -> Float:
        """
        Args:
            x (SE3R3): Task coordinates $\\mathrm{SE}(3) \\times \\mathbb{R}^3$
            joint_limit_factor (float): Factor for joint limit penalty.

        Returns:
            Float: Scalar loss value. Negative Log-det of IK jacobian penalized by log-barrier on joint limits.
        """
        jac = self.ik_jacobian(x, self.ik(x))

        joint_limit_loss = joint_limit_factor * (
            -jnp.log(x.rdof - self.r_i_lower_limits).sum()
            - jnp.log(self.r_i_upper_limits - x.rdof).sum()
        )
        return -jnp.log(jnp.linalg.det(jac @ jac.T)) + joint_limit_loss

    def loss_grad(self, x: SE3R3, joint_limit_factor=1e-1) -> SE3R3:
        """
        Args:
            x (SE3R3): Task coordinates $\\mathrm{SE}(3) \\times \\mathbb{R}^3$
            joint_limit_factor (float): Factor for joint limit penalty.

        Returns:
            SE3R3: Gradient of the loss with respect to task coordinates, in `SE3R3` pytree format.
        """
        return jaxlie.manifold.grad(
            Partial(self.loss, joint_limit_factor=joint_limit_factor)
        )(x)

    @override
    def mj_spec(
        self,
        act_lower_radius=5e-3,
        act_lower_length=100e-3,
        act_upper_radius=2.5e-3,
        act_upper_length=200e-3,
    ) -> "mujoco_t.MjSpec":  # type: ignore
        """
        Args:
            act_lower_radius (float): Radius of the actuator lower capsule radius. Defaults to 5e-3.
            act_lower_length (float): Length of the actuator lower capsule. Defaults to 100e-3.
            act_upper_radius (float): Radius of the actuator upper capsule radius. Defaults to 2.5e-3.
            act_upper_length (float): Length of the actuator upper capsule. Defaults to 200e-3.

        Returns:
            mujoco_t.MjSpec: The MJCF specification of the mechanism.
        """
        self._check_mujoco_availability()

        spec = mujoco.MjSpec()  # type: ignore
        spec.modelname = "se3r3_stewart"

        # Globally Disable Contact
        spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

        # End-Effector
        ee_body = spec.worldbody.add_body(name="body_ee")
        # ee_body.add_freejoint()

        ## give EE a hex plate mesh
        vertices = jnp.vstack([self.v_i1, self.v_i2])
        vertices_top = vertices + jnp.array([0, 0, 2.5e-3])
        vertices_bottom = vertices - jnp.array([0, 0, 0])
        vertices = jnp.vstack([vertices_top, vertices_bottom]).astype(jnp.float32)
        mesh = spec.add_mesh(name="hex_mesh")
        mesh.uservert = vertices.flatten().tolist()
        hex_geom = ee_body.add_geom()
        hex_geom.type = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
        hex_geom.meshname = mesh.name

        ## EE sites
        site_Bi1_ee = [
            ee_body.add_site(
                name=f"site_B{i + 1}1_ee",
                pos=v_i,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            for i, v_i in enumerate(self.v_i1)
        ]
        site_Bi2_ee = [
            ee_body.add_site(
                name=f"site_B{j + 1}2_ee",
                pos=v_j,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            for j, v_j in enumerate(self.v_i2)
        ]

        # Slider Redundant Legs
        slider_body_list = []
        slider_joint_list = []
        for wxyz_xyz, lower_limit, upper_limit in zip(
            self.r_i_se3.parameters(), self.r_i_lower_limits, self.r_i_upper_limits
        ):
            slider_body = spec.worldbody.add_body(
                name=f"body_slider_{len(slider_body_list) + 1}",
                pos=SE3(wxyz_xyz).translation(),
                quat=SE3(wxyz_xyz).rotation().parameters(),
            )
            slider_joint = slider_body.add_joint(
                name=f"joint_slider_{len(slider_joint_list) + 1}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                axis=[0.0, 0.0, 1.0],
            )
            slider_body_list.append(slider_body)
            slider_joint_list.append(slider_joint)

            # Slider initial position site
            spec.worldbody.add_site(
                pos=wxyz_xyz[4:7],
            )

            # Slider range capsule site
            spec.worldbody.add_site(
                fromto=jnp.concatenate(
                    [
                        SE3(wxyz_xyz)
                        @ SE3.from_translation(
                            jnp.array([0, 0, lower_limit])
                        ).translation(),
                        SE3(wxyz_xyz)
                        @ SE3.from_translation(
                            jnp.array([0, 0, upper_limit])
                        ).translation(),
                    ]
                ),
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[STRUCTURAL_CAPSULE_RADIUS, 0, 0],
                rgba=[0.5, 0.5, 0.5, 0.3],
            )

        # Linear Legs
        leg_i1_joint_list: list["mujoco_t.MjsJoint"] = []  # type: ignore
        leg_i2_joint_list: list["mujoco_t.MjsJoint"] = []  # type: ignore
        leg_i1_site_list: list["mujoco_t.MjsSite"] = []  # type: ignore
        leg_i2_site_list: list["mujoco_t.MjsSite"] = []  # type: ignore

        for index12, (
            a_ix_in_r,
            slider_bodies,
            joint_list,
            site_list,
            ee_site_list,
        ) in enumerate(
            [
                [
                    self.a_i1_in_r,
                    slider_body_list,
                    leg_i1_joint_list,
                    leg_i1_site_list,
                    site_Bi1_ee,
                ],
                [
                    self.a_i2_in_r,
                    slider_body_list,
                    leg_i2_joint_list,
                    leg_i2_site_list,
                    site_Bi2_ee,
                ],
            ]
        ):
            for i, (a_i_in_r, slider_body, ee_site) in enumerate(
                zip(a_ix_in_r, slider_bodies, ee_site_list)
            ):
                slider_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                    size=[STRUCTURAL_CAPSULE_RADIUS, 0, 0],
                    fromto=jnp.concatenate([jnp.zeros(3), a_i_in_r]),
                    rgba=[0.0, 0.0, 1.0, 0.9],
                )

                slider_rev_body1 = slider_body.add_body(
                    pos=a_i_in_r,
                    mass=mujoco.mjMINVAL,  # type: ignore
                    inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
                )
                slider_rev_body1.add_joint(
                    name=f"joint_revolute_{i}{index12 + 1}1",
                    type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                    axis=[0.0, 1.0, 0.0],
                    group=5,
                )

                slider_rev_body2 = slider_rev_body1.add_body(
                    pos=jnp.array([0.0, 0.0, 0.0]),
                    mass=mujoco.mjMINVAL,  # type: ignore
                    inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
                )
                slider_rev_body2.add_joint(
                    name=f"joint_revolute_{i}{index12 + 1}2",
                    type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                    axis=[1.0, 0.0, 0.0],
                    group=5,
                )

                slider_rev_body2.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                    size=[act_lower_radius, 0, 0],
                    fromto=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, act_lower_length]),
                )

                leg_upper_body = slider_rev_body2.add_body(
                    pos=jnp.zeros(3),
                )
                leg_joint = leg_upper_body.add_joint(
                    name=f"joint_slide_leg_{i}{index12 + 1}",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                    axis=[0.0, 0.0, 1.0],
                )
                leg_upper_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                    size=[act_upper_radius, 0, 0],
                    fromto=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, -act_upper_length]),
                    rgba=[0.0, 0.0, 1.0, 0.9],
                )
                site_joint = leg_upper_body.add_site(
                    name=f"site_A{i + 1}{index12 + 1}_leg_upper",
                    # pos=jnp.array([0.0, 0.0, -act_upper_length]),
                    pos=jnp.zeros(3),
                    size=[act_upper_radius * 1.5, 0.0, 0.0],
                    rgba=[1.0, 0.0, 0.0, 1.0],
                )
                eq = spec.add_equality()
                eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
                eq.name1 = site_joint.name
                eq.name2 = ee_site.name
                eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore
                joint_list.append(leg_joint)  # type: ignore
                site_list.append(site_joint)  # type: ignore

        for i, leg_i1_joint in enumerate(leg_i1_joint_list, 1):
            spec.add_actuator(
                name=f"act_{i}1",
                target=leg_i1_joint.name,
                **(
                    act_params := {
                        "gainprm": [2e2] + [0.0] * 9,
                        "biastype": mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                        "biasprm": [0.0, -2e2, 1] + [0.0] * 7,
                        "trntype": mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                    }
                ),
            )
        for i, leg_i2_joint in enumerate(leg_i2_joint_list, 1):
            spec.add_actuator(
                name=f"act_{i}2",
                target=leg_i2_joint.name,
                **act_params,
            )
        for i, slider_joint in enumerate(slider_joint_list, 1):
            spec.add_actuator(
                name=f"act_slider_{i}",
                target=slider_joint.name,
                **act_params,
            )

        return spec

    @override
    def mj_spec_model_data(
        self, x0: SE3R3, *args, **kwargs
    ) -> tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]:  # type: ignore
        """
        Args:
            x0 (SE3R3): Initial task coordinates $\\mathrm{SE}(3) \\times \\mathbb{R}^3$
            *args: Additional arguments for `mj_spec`.
            **kwargs: Additional keyword arguments for `mj_spec`.

        Returns:
            tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]: `MjModel` and `MjData` initialized at the given task coordinates, along with `MjSpec`.
        """

        spec = self.mj_spec(*args, **kwargs)
        spec.body("body_ee").pos = x0.pose.translation().tolist()
        spec.body("body_ee").quat = x0.pose.rotation().parameters().tolist()
        model = spec.compile()  # type: ignore
        data = mujoco.MjData(model)  # type: ignore

        data.ctrl = jnp.array(self.ik(x0))
        for i in range(int(1e5)):
            mujoco.mj_step(model, data)  # type: ignore
            if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-8:
                break
        if i >= int(1e5) - 1:
            raise RuntimeError("Failed to stabilize the model.")

        spec.body("body_ee").add_freejoint()

        key_frame = spec.add_key()  # this should be keyframe 0
        key_frame.name = "home"
        model, data0 = spec.recompile(model, data)
        key_frame.qpos = data0.qpos.copy()
        key_frame.ctrl = data0.ctrl.copy()
        model, data0 = spec.recompile(model, data0)
        return spec, model, data0
