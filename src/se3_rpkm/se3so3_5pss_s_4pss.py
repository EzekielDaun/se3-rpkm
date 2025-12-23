from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxlie import SE3

from .data_types import SE3SO3, SE39, Mat4x3, Mat5x3, Mat9x3, MuJoCoMixin, Vec9
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
class SE3SO3_5PSS_S_4PSS_Kinematics(AbstractManipulator[SE3SO3, Vec9], MuJoCoMixin):
    slider_axis: SE39  # Batch of 9 SE3, sliding along Z-axis
    link_length: Vec9
    a1_a5: Mat5x3  # (5, 3), 5 row vectors
    a6_a9: Mat4x3  # (4, 3), 4 row vectors

    def a_i(self, x: SE3SO3) -> Mat9x3:
        return jnp.concatenate(
            [
                x.pose.apply(self.a1_a5),
                (x.pose @ SE3.from_rotation(x.rdof)).apply(self.a6_a9),
            ]
        )

    def b_i(self, q: Vec9) -> Mat9x3:
        return (
            self.slider_axis
            @ SE3.from_translation(jnp.zeros((9, 3)).at[:, 2].set(q)).translation()
        )

    @override
    def kinematic_constraints(self, task_coord: SE3SO3, joint_coord: Vec9):
        return (
            jnp.linalg.norm(self.a_i(task_coord) - self.b_i(joint_coord), axis=1)
            - self.link_length
        )

    @override
    def mj_spec(self) -> "mujoco_t.MjSpec":  # type: ignore
        self._check_mujoco_availability()

        spec = mujoco.MjSpec()  # type: ignore
        spec.modelname = "se3r3_stewart"

        # Globally Disable Contact
        spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

        # End-Effector
        ee_body = spec.worldbody.add_body(name="body_ee")
        # ee_body.add_freejoint()

        ## give EE a hex plate mesh
        vertices = jnp.vstack([self.a1_a5, jnp.zeros(3)])
        vertices_top = vertices + jnp.array([0, 0, 2.5e-3])
        vertices_bottom = vertices - jnp.array([0, 0, 0])
        vertices = jnp.vstack([vertices_top, vertices_bottom]).astype(jnp.float32)
        mesh = spec.add_mesh(name="ee_mesh")
        mesh.uservert = vertices.flatten().tolist()
        hex_geom = ee_body.add_geom(rgba=[1, 0, 0, 1])
        hex_geom.type = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
        hex_geom.meshname = mesh.name

        ## EE Site
        ee_site_list = [
            ee_body.add_site(name=f"site_ee_{i}", pos=ai)
            for i, ai in enumerate(self.a1_a5)
        ]
        ee_body.add_site()  # rdof SO3 ball joint visualization

        # RDoF End-Effector
        rdof_ee_body = ee_body.add_body(name="body_rdof_ee")
        # rdof_ee_body.add_joint(type=mujoco.mjtJoint.mjJNT_BALL)  # type: ignore

        ## give EE a hex plate mesh
        vertices = jnp.vstack([self.a6_a9, jnp.zeros(3)])
        vertices_top = vertices + jnp.array([0, 0, 2.5e-3])
        vertices_bottom = vertices - jnp.array([0, 0, 0])
        vertices = jnp.vstack([vertices_top, vertices_bottom]).astype(jnp.float32)
        mesh = spec.add_mesh(name="rdof_ee_mesh")
        mesh.uservert = vertices.flatten().tolist()
        hex_geom = rdof_ee_body.add_geom(rgba=[0, 1, 0, 1])
        hex_geom.type = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
        hex_geom.meshname = mesh.name

        ## RDoF EE Site
        rdof_ee_site_list = [
            rdof_ee_body.add_site(name=f"site_rdof_ee_{i}", pos=ai)
            for i, ai in enumerate(self.a6_a9)
        ]
        rdof_ee_body.add_site()  # rdof SO3 ball joint visualization

        slider_joint_list = []
        # Legs to EE
        for i, (wxyz_xyz, li, ee_site) in enumerate(
            zip(self.slider_axis.parameters()[:5], self.link_length[:5], ee_site_list)
        ):
            slider_base_body = spec.worldbody.add_body(
                pos=wxyz_xyz[-3:], quat=wxyz_xyz[:4]
            )
            slider_base_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,  # type: ignore
                size=[5e-3] * 3,
                fromto=[0, 0, -1e1, 0, 0, 1e1],
            )

            slider_body = slider_base_body.add_body(
                mass=mujoco.mjMINVAL,  # type: ignore
                inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
            )
            slider_body.add_site(  # visualize slider joint
                size=[10e-3] * 3, rgba=[0, 0, 1, 1]
            )
            slider_joint_list.append(
                slider_body.add_joint(
                    type=mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                    name=f"leg_{i + 1}",
                    axis=[0, 0, 1],
                )
            )
            slider_hinge_body1 = slider_body.add_body(
                mass=mujoco.mjMINVAL,  # type: ignore
                inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
            )
            slider_hinge_body1.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                axis=[0, 0, 1],
                group=5,
            )

            link_body = slider_hinge_body1.add_body()
            link_body.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                axis=[0, 1, 0],
                group=5,
            )
            link_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[2.5e-3] * 3,
                fromto=[0, 0, 0, 0, 0, li],
            )

            leg_site = link_body.add_site(pos=[0, 0, li], name=f"site_leg_{i}")
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = leg_site.name
            eq.name2 = ee_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore

        # Legs to RDoF EE
        for i, (wxyz_xyz, li, rdof_ee_site) in enumerate(
            zip(
                self.slider_axis.parameters()[5:],
                self.link_length[5:],
                rdof_ee_site_list,
            )
        ):
            slider_base_body = spec.worldbody.add_body(
                pos=wxyz_xyz[-3:], quat=wxyz_xyz[:4]
            )
            slider_base_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,  # type: ignore
                size=[5e-3] * 3,
                fromto=[0, 0, -1e1, 0, 0, 1e1],
            )

            slider_body = slider_base_body.add_body(
                mass=mujoco.mjMINVAL,  # type: ignore
                inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
            )

            slider_body.add_site(  # visualize slider joint
                size=[10e-3] * 3, rgba=[0, 0, 1, 1]
            )

            slider_joint_list.append(
                slider_body.add_joint(
                    type=mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                    name=f"leg_{i + 5 + 1}",
                    axis=[0, 0, 1],
                )
            )

            slider_hinge_body1 = slider_body.add_body(
                mass=mujoco.mjMINVAL,  # type: ignore
                inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
            )
            slider_hinge_body1.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                axis=[0, 0, 1],
                group=5,
            )

            link_body = slider_hinge_body1.add_body()
            link_body.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                axis=[0, 1, 0],
                group=5,
            )
            link_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[2.5e-3] * 3,
                fromto=[0, 0, 0, 0, 0, li],
            )

            leg_site = link_body.add_site(pos=[0, 0, li], name=f"site_leg_{i + 5}")
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = leg_site.name
            eq.name2 = rdof_ee_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore

        for index, slide_joint in enumerate(slider_joint_list):
            GAIN = 500.0
            spec.add_actuator(
                name=f"act_{index + 1}",
                gainprm=[GAIN] + [0.0] * 9,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                target=slide_joint.name,
            )

        return spec

    @override
    def mj_spec_model_data(
        self, x0: SE3SO3, q0: Vec9, *args, **kwargs
    ) -> tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]:  # type: ignore
        spec = self.mj_spec()
        spec.body("body_ee").pos = x0.pose.translation().tolist()
        spec.body("body_ee").quat = x0.pose.rotation().parameters().tolist()
        pose_rdof = x0.pose @ SE3.from_rotation(x0.rdof)
        spec.body("body_rdof_ee").quat = pose_rdof.rotation().parameters().tolist()
        model = spec.compile()  # type: ignore
        data = mujoco.MjData(model)  # type: ignore

        data.ctrl = jnp.array(self.ik_optx(x0, q0))
        for i in range(int(1e5)):
            mujoco.mj_step(model, data)  # type: ignore
            if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-6:
                break
        if i >= int(1e5) - 1:
            raise RuntimeError("Failed to stabilize the model.")

        spec.body("body_ee").add_freejoint()
        spec.body("body_rdof_ee").add_joint(type=mujoco.mjtJoint.mjJNT_BALL)  # type: ignore

        key_frame = spec.add_key()  # this should be keyframe 0
        key_frame.name = "home"
        model, data0 = spec.recompile(model, data)
        key_frame.qpos = data0.qpos.copy()
        key_frame.ctrl = data0.ctrl.copy()
        model, data0 = spec.recompile(model, data0)
        return spec, model, data0
