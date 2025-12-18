from typing import TYPE_CHECKING, Callable, override

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
    SO23,
    Mat2x3,
    Mat3x3,
    Mat3x15,
    Mat4x3,
    Mat6x6,
    Mat9x9,
    MuJoCoMixin,
    Vec3,
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


def _jax_normalize_columns(X: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(X, axis=0, keepdims=True)  # shape (1, m)
    return X / norms


def _jax_normalize_rows(X: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
    return X / norms


@jdc.pytree_dataclass(frozen=True)
class SE3SO23StewartDimension(AbstractManipulator[SE3SO23, Vec9], MuJoCoMixin):
    passive_joints_tuple: tuple[float, ...]  # 45 floats
    redundant_links_tuple: tuple[float, ...]  # 3 floats

    @property
    def passive_joints(self) -> Mat3x15:
        """Convert the tuple to a 3x15 matrix."""
        return jnp.array(self.passive_joints_tuple).reshape(15, 3).T

    @property
    def redundant_links(self) -> Vec3:
        """Convert the tuple to a 3-vector."""
        return jnp.array(self.redundant_links_tuple)

    # ===== Begin of Constructor Methods =====

    @classmethod
    def from_passive_joints_and_config(
        cls,
        a_i: tuple[Vec3, ...],  # 3 vectors
        v_i: tuple[Vec3, ...],  # 3 vectors
        a_j1: tuple[Vec3, ...],  # 3 vectors
        a_j2: tuple[Vec3, ...],  # 3 vectors
        v_j: tuple[Vec3, ...],  # 3 vectors
        l_j: Vec3,  # 3 positive floats
    ) -> "SE3SO23StewartDimension":
        return cls(
            passive_joints_tuple=tuple(
                jnp.column_stack([*a_i, *v_i, *a_j1, *a_j2, *v_j]).T.flatten().tolist()
            ),
            redundant_links_tuple=tuple(l_j.tolist()),
        )

    # ===== End of Constructor Methods =====

    # ===== Begin of Passive Joint Positions =====
    @property
    def a_i(self) -> Mat3x3:
        return self.passive_joints[:, :3]

    @property
    def v_i(self) -> Mat3x3:
        return self.passive_joints[:, 3:6]

    @property
    def a_j1(self) -> Mat3x3:
        return self.passive_joints[:, 6:9]

    @property
    def a_j2(self) -> Mat3x3:
        return self.passive_joints[:, 9:12]

    @property
    def v_j(self) -> Mat3x3:
        return self.passive_joints[:, 12:15]

    def b_i(self, pose: SE3) -> Mat3x3:
        return jax.vmap(pose.apply, in_axes=1, out_axes=1)(self.v_i)

    def b_j(self, pose: SE3) -> Mat3x3:
        return jax.vmap(pose.apply, in_axes=1, out_axes=1)(self.v_j)

    @property
    def e_j(self) -> Mat3x3:
        return _jax_normalize_columns(self.a_j2 - self.a_j1)

    def k_j(self, pose: SE3) -> Mat3x3:
        e_j = self.e_j
        b_j = self.b_j(pose)
        z_j = _jax_normalize_columns(jnp.cross(e_j, (b_j - self.a_j1), axis=0))
        return jnp.cross(z_j, e_j, axis=0)

    def r_j(self, ext_coord: SE3SO23) -> Mat3x3:
        e_j = self.e_j
        k_j = self.k_j(ext_coord.pose)
        b_j = self.b_j(ext_coord.pose)

        cos_sin = ext_coord.rdof.apply(jnp.array([1, 0]))
        return b_j + (e_j * cos_sin[:, 0] + -k_j * cos_sin[:, 1]) * self.redundant_links

    # ===== End of Passive Joint Positions =====

    # ===== Begin of Actuated Joint Positions =====
    def ik(self, task_coord: SE3SO23) -> Vec9:
        r_j = self.r_j(task_coord)
        rho_i = jnp.linalg.norm(self.b_i(task_coord.pose) - self.a_i, axis=0)
        rho_j1 = jnp.linalg.norm(r_j - self.a_j1, axis=0)
        rho_j2 = jnp.linalg.norm(r_j - self.a_j2, axis=0)
        return jnp.concatenate([rho_i, rho_j1, rho_j2]).flatten()

    # ===== End of Actuated Joint Positions =====

    # ===== Begin of Kinematic Constraints =====
    def kinematic_constraints(self, task_coord: SE3SO23, joint_coord: Vec9) -> Vec9:
        return self.ik(task_coord) - joint_coord

    # ===== End of Kinematic Constraints =====

    # ===== Begin of Jacobian Matrices =====
    def ext_jacobian(self, task_coord: SE3SO23) -> Mat9x9:
        return self.ik_jacobian(task_coord, self.ik(task_coord))

    def base_jacobian(self, ext_coord: SE3SO23) -> Mat6x6:
        j_ext = self.ext_jacobian(ext_coord)
        j1, j2 = j_ext[:, :6], j_ext[:, 6:]
        q, _ = jnp.linalg.qr(j2, mode="complete")
        q2 = q[:, 3:]
        return q2.T @ j1

    # ===== End of Jacobian Matrices =====

    # ===== Begin of Jacobian Determinants =====

    def ext_jacobian_det(self, ext_coord: SE3SO23) -> float:
        return jnp.linalg.det(self.ext_jacobian(ext_coord))

    def base_jacobian_det(self, ext_coord: SE3SO23) -> float:
        return jnp.linalg.det(self.base_jacobian(ext_coord))

    # ===== End of Jacobian Determinants =====

    # ===== Begin of Loss Functions =====
    def loss_func(self, x0: SE3SO23):
        singularity_cost = -jnp.log(
            jnp.linalg.det(self.ext_jacobian(x0).T @ self.ext_jacobian(x0))
        )
        return singularity_cost

    def loss_grad(self, x0: SE3SO23):
        return jaxlie.manifold.grad(Partial(self.loss_func))(x0)

    def loss_val_and_grad(self, x0: SE3SO23):
        return jaxlie.manifold.value_and_grad(Partial(self.loss_func))(x0)

    def loss_hessian(self, x0: SE3SO23):
        def flat_func(x: SE3SO23) -> jnp.ndarray:
            grad = self.loss_grad(x)
            return jnp.concatenate([grad.pose, grad.rdof.ravel()])

        def hess_block(i):
            return jaxlie.manifold.grad(lambda x: flat_func(x)[i])(x0)

        hess_rows = jax.vmap(hess_block)(jnp.arange(9))

        pose_block = hess_rows.pose
        rdof_block = hess_rows.rdof.squeeze(-1)
        return jnp.hstack([pose_block, rdof_block])

    # ===== End of Loss Functions =====

    # ===== Begin of RDoF Trajectory Optimization with Given SE3 Trajectory =====
    def pseudo_inverse_step_fn(
        self, carry: tuple[SE3SO23, float], pose: SE3, factor: float
    ):
        x_last, loss_accum = carry
        x = SE3SO23(pose=pose, rdof=x_last.rdof)

        se3log = (pose @ x_last.pose.inverse()).log()
        jac = self.ext_jacobian(x)
        u_full = jnp.linalg.inv(jac)
        u = u_full[:6, :]
        rho = jnp.linalg.pinv(u) @ se3log
        dx = jnp.linalg.inv(jac) @ rho
        rdof_new = x_last.rdof @ SO2.exp(dx[6:].reshape(3, 1))
        x_new = SE3SO23(pose=pose, rdof=rdof_new)
        loss = self.loss_func(x_new)
        return (x_new, loss_accum + loss), x_new

    def normalized_gradient_descent_step_fn(
        self, carry: tuple[SE3SO23, float], pose: SE3, factor: float
    ):
        x_last, loss_accum = carry
        x = SE3SO23(pose=pose, rdof=x_last.rdof)

        grads = self.loss_grad(x)
        grads_rdof = grads.rdof
        grads_rdof_norm = jnp.linalg.norm(grads_rdof)
        scale = jnp.minimum(1.0, factor / (grads_rdof_norm + jnp.finfo(float).eps))
        rdof_update = -grads_rdof * scale
        update = SE3SO23(pose=jnp.zeros_like(grads.pose), rdof=rdof_update)  # type: ignore
        x_new = jaxlie.manifold.rplus(x, update)
        loss = self.loss_func(x_new)
        return (x_new, loss_accum + loss), x_new

    def damped_newton_step_fn(
        self, carry: tuple[SE3SO23, float], pose: SE3, factor: float
    ):
        x_last, loss_accum = carry
        x = SE3SO23(pose=pose, rdof=x_last.rdof)

        grads = self.loss_grad(x).rdof
        hessian = self.loss_hessian(x)
        delta = jnp.linalg.solve(hessian[-3:, -3:] + factor * jnp.eye(3), grads)

        x_new = SE3SO23(pose=pose, rdof=jaxlie.manifold.rplus(x.rdof, -delta))
        loss = self.loss_func(x_new)
        return (x_new, loss_accum + loss), x_new

    def hessian_mix_step_fn(
        self, carry: tuple[SE3SO23, float], pose: SE3, factor: float
    ):
        x_last, loss_accum = carry
        x = SE3SO23(pose=pose, rdof=x_last.rdof)

        grads = self.loss_grad(x).rdof
        hessian = self.loss_hessian(x)
        hess_block = hessian[-3:, -3:]

        # Attempt Cholesky for positive definiteness
        def use_newton():
            delta = jnp.linalg.solve(hess_block + factor * jnp.eye(3), grads)
            return jaxlie.manifold.rplus(x.rdof, -delta)

        def use_gradient():
            eigvals, eigvecs = jnp.linalg.eigh(hess_block)
            min_dir = eigvecs[:, 0]
            direction = (
                grads.squeeze() / (jnp.linalg.norm(grads) + 1e-8) + 0.1 * min_dir
            )
            direction /= jnp.linalg.norm(direction) + 1e-8
            return jaxlie.manifold.rplus(x.rdof, (-direction * (1e-2)).reshape(3, 1))

        new_rdof = jax.lax.cond(
            jnp.all(jnp.linalg.eigvalsh(hess_block) > 0),
            use_newton,
            use_gradient,
        )

        x_new = SE3SO23(pose=pose, rdof=new_rdof)
        loss = self.loss_func(x_new)
        return (x_new, loss_accum + loss), x_new

    # sample call to step_fn
    def plan_rdof_trajectory_hessian_mix(
        self,
        trajectory_fn: Callable[[float], SE3],
        r0: SO23,
        ts: jnp.ndarray,
        factor: float = 5e2,
    ) -> tuple[SE3SO23, float]:
        poses_batched = jax.vmap(trajectory_fn)(ts)  # type: ignore

        (carry_final, xs) = jax.lax.scan(
            Partial(self.hessian_mix_step_fn, factor=factor),
            (SE3SO23(pose=SE3(poses_batched.parameters()[0]), rdof=r0), 0.0),  # type: ignore
            poses_batched,
        )
        return xs, carry_final[1]

    # ===== End of RDoF Trajectory Optimization with Given SE3 Trajectory =====

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
        spec.modelname = "se3so23_stewart"

        # Globally Disable Contact
        spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

        # End-Effector
        ee_body = spec.worldbody.add_body(name="body_ee")
        # ee_body.add_freejoint()

        ## give EE a hex plate mesh
        vertices = jnp.hstack([self.v_i, self.v_j]).T
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
            for i, v_i in enumerate(self.v_i.T)
        ]
        site_Bjx_ee = [
            ee_body.add_site(
                name=f"site_B{j + 1}x_ee",
                pos=v_j,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
            )
            for j, v_j in enumerate(self.v_j.T)
        ]

        # Base
        ## Redundant Legs Attachment Sites
        site_Aj1_base = []
        site_Aj2_base = []
        for j, a_j1 in enumerate(self.a_j1.T):
            site_Aj1_base.append(
                spec.worldbody.add_site(
                    name=f"site_A{j + 1}1_base",
                    pos=a_j1,
                    size=[act_upper_radius * 1.5, 0.0, 0.0],
                    rgba=[1.0, 0.0, 0.0, 1.0],
                )
            )
        for j, a_j2 in enumerate(self.a_j2.T):
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
        for i, a_i in enumerate(self.a_i.T):
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
        for j, (a_j1, a_j2, l_j) in enumerate(
            zip(self.a_j1.T, self.a_j2.T, self.redundant_links)
        ):
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
        self, x0: SE3SO23
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


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO22StewartKinematics(AbstractManipulator[SE3SO22, Vec8]):
    a_i: Float  # (4, 3)
    a_j1: Float  # (2, 3)
    a_j2: Float  # (2, 3)
    v_i: Float  # (4, 3)
    v_j: Float  # (2, 3)
    l_j: Float  # (2,)

    def b_i(self, pose: SE3) -> Mat4x3:
        return pose.apply(self.v_i)

    def b_j(self, pose: SE3) -> Mat4x3:
        return pose.apply(self.v_j)

    @property
    def e_j(self) -> Mat2x3:
        return _jax_normalize_rows(self.a_j2 - self.a_j1)

    def k_j(self, pose: SE3) -> Mat2x3:
        e_j = self.e_j
        b_j = self.b_j(pose)
        z_j = _jax_normalize_rows(jnp.cross(e_j, (b_j - self.a_j1), axis=1))
        return jnp.cross(z_j, e_j, axis=1)

    def r_j(self, task_coord: SE3SO22) -> Mat2x3:
        e_j = self.e_j
        k_j = self.k_j(task_coord.pose)
        b_j = self.b_j(task_coord.pose)

        cos_sin = task_coord.rdof.apply(jnp.array([1, 0]))
        return (
            b_j + (e_j * cos_sin[:, 0:1] + -k_j * cos_sin[:, 1:2]) * self.l_j[:, None]
        )

    def ik(self, task_coord: SE3SO22) -> Vec8:
        r_j = self.r_j(task_coord)
        rho_i = jnp.linalg.norm(self.b_i(task_coord.pose) - self.a_i, axis=1)
        rho_j1 = jnp.linalg.norm(r_j - self.a_j1, axis=1)
        rho_j2 = jnp.linalg.norm(r_j - self.a_j2, axis=1)
        return jnp.concatenate([rho_i, rho_j1, rho_j2]).flatten()

    @override
    def kinematic_constraints(self, task_coord: SE3SO22, joint_coord: Vec8) -> Vec8:
        return self.ik(task_coord) - joint_coord
