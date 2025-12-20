from typing import TYPE_CHECKING, Generic, TypeVar, override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jax.tree_util import Partial
from jaxlie import SE3, SO2, SO3
from jaxtyping import Float

from .data_types import (
    SE3SO22,
    SE3SO23,
    MuJoCoMixin,
    RedundantSE3TaskCoordinate,
    Vec8,
    Vec9,
)
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


def _jax_normalize_rows(X: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
    return X / norms


T = TypeVar("T", bound=RedundantSE3TaskCoordinate[SO2])
J = TypeVar("J", bound=Float)


@jdc.pytree_dataclass(frozen=True, slots=True)
class RedundantSO2LegStewartKinematicsCommon(
    Generic[T, J], AbstractManipulator[T, J], MuJoCoMixin
):
    # M non-redundant legs, N redundant legs, M + 2N linear actuators
    a_i: Float  # (M, 3), 3 row vectors
    v_i: Float  # (M, 3)
    a_j1: Float  # (N, 3)
    a_j2: Float  # (N, 3)
    v_j: Float  # (N, 3)
    l_j: Float  # (N,)

    def b_i(self, pose: SE3) -> Float:  # (M, 3)
        return pose.apply(self.v_i)

    def b_j(self, pose: SE3) -> Float:  # (N, 3)
        return pose.apply(self.v_j)

    @property
    def e_j(self) -> Float:  # (N, 3)
        return _jax_normalize_rows(self.a_j2 - self.a_j1)

    def k_j(self, pose: SE3) -> Float:  # (N, 3)
        e_j = self.e_j
        b_j = self.b_j(pose)
        z_j = _jax_normalize_rows(jnp.cross(e_j, (b_j - self.a_j1), axis=1))
        return jnp.cross(z_j, e_j, axis=1)

    def r_j(self, task_coordinate: T) -> Float:  # (N, 3)
        e_j = self.e_j
        k_j = self.k_j(task_coordinate.pose)
        b_j = self.b_j(task_coordinate.pose)

        cos_sin = task_coordinate.rdof.apply(jnp.array([1, 0]))
        return (
            b_j + (e_j * cos_sin[:, 0:1] + -k_j * cos_sin[:, 1:2]) * self.l_j[:, None]
        )

    def ik(self, task_coordinate: T) -> Float:  # (M + 2N,)
        r_j = self.r_j(task_coordinate)
        rho_i = jnp.linalg.norm(self.b_i(task_coordinate.pose) - self.a_i, axis=1)
        rho_j1 = jnp.linalg.norm(r_j - self.a_j1, axis=1)
        rho_j2 = jnp.linalg.norm(r_j - self.a_j2, axis=1)
        return jnp.concatenate([rho_i, rho_j1, rho_j2]).flatten()

    @override
    def kinematic_constraints(self, task_coord: T, joint_coord: J) -> J:
        return self.ik(task_coord) - joint_coord

    def loss_func(self, x0: T):
        ik_jac = self.ik_jacobian(x0, self.ik(x0))
        return -jnp.linalg.slogdet(ik_jac.T @ ik_jac)[1]

    def loss_grad(self, x0: T):
        return jaxlie.manifold.grad(Partial(self.loss_func))(x0)

    def loss_hessian(self, x0: T):
        def flat_func(x: T) -> jnp.ndarray:
            grad = self.loss_grad(x)
            return jnp.concatenate([grad.pose, grad.rdof.ravel()])

        def hess_block(i):
            return jaxlie.manifold.grad(lambda x: flat_func(x)[i])(x0)

        rdof_tangent_dim = x0.rdof.tangent_dim * x0.rdof.get_batch_axes()[0]
        hess_rows = jax.vmap(hess_block)(
            jnp.arange(x0.pose.tangent_dim + rdof_tangent_dim)
        )
        pose_block = hess_rows.pose
        rdof_block = hess_rows.rdof.squeeze(-1)
        return jnp.hstack([pose_block, rdof_block])

    def damped_newton_step_fn(self, carry: tuple[T, float], pose: SE3, factor: float):
        x_last, loss_accum = carry
        x = type(x_last)(pose=pose, rdof=x_last.rdof)  # type: ignore

        grads = self.loss_grad(x)
        hessian = self.loss_hessian(x)
        rdof_tangent_dim = x.rdof.tangent_dim * x.rdof.get_batch_axes()[0]
        delta = jnp.linalg.solve(
            hessian[-rdof_tangent_dim:, -rdof_tangent_dim:]  # type: ignore
            + factor * jnp.eye(rdof_tangent_dim),  # type: ignore
            grads.rdof,
        )

        x_new = type(x)(pose=pose, rdof=jaxlie.manifold.rplus(x.rdof, -delta))  # type: ignore
        loss = self.loss_func(x_new)
        return (x_new, loss_accum + loss), x_new

    @override
    def mj_spec(
        self,
        act_lower_radius=0.02,
        act_lower_length=0.15,
        act_upper_radius=0.01,
        act_upper_length=0.5,
    ) -> "mujoco_t.MjSpec":  # type: ignore
        MuJoCoMixin._check_mujoco_availability()
        spec = mujoco.MjSpec()  # type: ignore
        spec.modelname = "se3so2x_stewart"

        # Globally Disable Contact
        spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

        # End-Effector
        ee_body = spec.worldbody.add_body(name="body_ee")
        # ee_body.add_freejoint()

        ## give EE a hex plate mesh
        vertices = jnp.vstack([self.v_i, self.v_j])
        vertices_top = vertices + jnp.array([0, 0, 0.01])
        vertices_bottom = vertices - jnp.array([0, 0, 0])
        vertices = jnp.vstack([vertices_top, vertices_bottom]).astype(jnp.float32)
        mesh = spec.add_mesh(name="hex_mesh")
        mesh.uservert = vertices.flatten().tolist()
        hex_geom = ee_body.add_geom()
        hex_geom.type = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
        hex_geom.meshname = mesh.name

        ## EE sites
        site_Bi_ee = [
            ee_body.add_site(
                name=f"site_B{i + 1}_ee",
                pos=v_i,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            for i, v_i in enumerate(self.v_i)
        ]
        site_Bjx_ee = [
            ee_body.add_site(
                name=f"site_B{j + 1}x_ee",
                pos=v_j,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            for j, v_j in enumerate(self.v_j)
        ]

        # Base
        ## Redundant Legs Attachment Sites
        site_Aj1_base = []
        site_Aj2_base = []
        for j, a_j1 in enumerate(self.a_j1):
            site_Aj1_base.append(
                spec.worldbody.add_site(
                    name=f"site_A{j + 1}1_base",
                    pos=a_j1,
                    size=[act_upper_radius * 1.5, 0.0, 0.0],
                    rgba=[1.0, 0.0, 0.0, 1.0],
                )
            )
        for j, a_j2 in enumerate(self.a_j2):
            site_Aj2_base.append(
                spec.worldbody.add_site(
                    name=f"site_A{j + 1}2_base",
                    pos=a_j2,
                    size=[act_upper_radius * 1.5, 0.0, 0.0],
                    rgba=[1.0, 0.0, 0.0, 1.0],
                )
            )

        # non-redundant legs
        site_nr_leg_i = []
        actuator_nr_leg_i = []
        for i, a_i in enumerate(self.a_i):
            spec.worldbody.add_site(
                pos=a_i,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[1.0, 0.0, 0.0, 1.0],
            )
            nr_leg_body0 = spec.worldbody.add_body(
                pos=a_i,
                mass=mujoco.mjMINVAL,  # type: ignore
                inertia=mujoco.mjMINVAL * jnp.ones(3),  # type: ignore
            )
            nr_leg_body0.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                axis=[1, 0, 0],
                group=5,
            )

            nr_leg_body1 = nr_leg_body0.add_body()
            nr_leg_body1.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                axis=[0, 1, 0],
                group=5,
            )
            nr_leg_body1.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[act_lower_radius, act_lower_length / 2, 0],
                rgba=[0.5, 0.5, 0.5, 0.5],
                pos=[0, 0, act_lower_length / 2],
                # quat=[0.7071, 0.0, 0.7071, 0.0],
            )

            nr_leg_body2 = nr_leg_body1.add_body()
            slide_joint = nr_leg_body2.add_joint(
                name=f"slide_joint_{i + 1}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                axis=[0, 0, 1],
                range=[0, mujoco.mjMAXVAL],  # type: ignore
            )
            actuator_nr_leg_i.append(
                spec.add_actuator(
                    name=f"act_{i + 1}",
                    gainprm=[5000.0] + [0.0] * 9,
                    biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                    biasprm=[0.0, -5000, 1] + [0.0] * 7,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                    target=slide_joint.name,
                )
            )
            nr_leg_body2.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[act_upper_radius, act_upper_length / 2, 0],
                rgba=[0.2, 0.2, 0.8, 0.9],
                pos=[0, 0, -act_upper_length / 2],
            )
            site_nr_leg_i.append(nr_leg_body2.add_site(name=f"site_B{i + 1}_leg"))

        # redundant legs
        revolute_body_list = []
        site_r_leg_jx = []
        for j, (_a_j1, _a_j2, l_j) in enumerate(zip(self.a_j1, self.a_j2, self.l_j)):
            r_revolute_body = spec.worldbody.add_body()
            r_revolute_body.add_freejoint()
            r_revolute_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[act_upper_radius, l_j / 2, 0],
                rgba=[0.5, 0.5, 0.5, 1],
                pos=[0, 0, l_j / 2],
            )
            r_revolute_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                size=[act_upper_radius, l_j / 2, 0],
                rgba=[0.5, 0.5, 0.5, 1],
                quat=SO3.from_x_radians(jnp.pi / 2).parameters(),
            )
            site_r_leg_jx.append(
                r_revolute_body.add_site(name=f"site_B{j + 1}x_leg", pos=[0, 0, l_j])
            )
            revolute_body_list.append(r_revolute_body)

        site_r_leg_j1 = []
        site_r_leg_j2 = []
        for index12, site_lst in enumerate([site_r_leg_j1, site_r_leg_j2]):
            for j, revolute_body in enumerate(revolute_body_list):
                leg_body = revolute_body.add_body()
                leg_body.add_joint(
                    type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
                    axis=[0, 1, 0],
                    group=5,
                )
                leg_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                    size=[act_upper_radius, act_upper_length / 2, 0],
                    rgba=[0.2, 0.2, 0.8, 0.9],
                    pos=[0, 0, -act_upper_length / 2],
                )

                leg_body_lower = leg_body.add_body()
                slide_joint = leg_body_lower.add_joint(
                    name=f"slide_joint_{j + 1}{index12 + 1}",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                    axis=[0, 0, -1],
                    range=[0, mujoco.mjMAXVAL],  # type: ignore
                )
                leg_body_lower.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
                    size=[act_lower_radius, act_lower_length / 2, 0],
                    rgba=[0.5, 0.5, 0.5, 0.5],
                    pos=[0, 0, act_lower_length / 2],
                )
                site_lst.append(
                    leg_body_lower.add_site(
                        name=f"site_A{j + 1}{index12 + 1}_leg", pos=[0, 0, 0]
                    )
                )
                spec.add_actuator(
                    name=f"act_{j + 1}{index12 + 1}",
                    gainprm=[5000.0] + [0.0] * 9,
                    biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
                    biasprm=[0.0, -5000, 1] + [0.0] * 7,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
                    target=slide_joint.name,
                )

        # equality on sites for connection
        ## non-redundant legs
        for ee_site, leg_site in zip(site_Bi_ee, site_nr_leg_i):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = ee_site.name
            eq.name2 = leg_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore
        ## redundant legs
        ### EE and redundant link
        for ee_site, leg_site in zip(site_Bjx_ee, site_r_leg_jx):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = ee_site.name
            eq.name2 = leg_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore
        ### redundant leg and base
        for ee_site, leg_site in zip(site_Aj1_base, site_r_leg_j1):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = ee_site.name
            eq.name2 = leg_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore
        for ee_site, leg_site in zip(site_Aj2_base, site_r_leg_j2):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
            eq.name1 = ee_site.name
            eq.name2 = leg_site.name
            eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore

        return spec

    @override
    def mj_spec_model_data(
        self, x0: T
    ) -> tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]:  # type: ignore
        spec = self.mj_spec()
        spec.body("body_ee").pos = x0.pose.translation().tolist()
        spec.body("body_ee").quat = x0.pose.rotation().parameters().tolist()
        model = spec.compile()  # type: ignore
        data = mujoco.MjData(model)  # type: ignore

        data.ctrl = jnp.array(self.ik(x0))
        for i in range(int(1e5)):
            mujoco.mj_step(model, data)  # type: ignore
            if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-6:
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


class SE3SO22StewartKinematics(RedundantSO2LegStewartKinematicsCommon[SE3SO22, Vec8]):
    pass


class SE3SO23StewartKinematics(RedundantSO2LegStewartKinematicsCommon[SE3SO23, Vec9]):
    pass
