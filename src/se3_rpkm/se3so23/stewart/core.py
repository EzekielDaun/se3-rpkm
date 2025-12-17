from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jax.tree_util import Partial
from jaxlie import SE3, SO2

from ...lie_group_kinematics import AbstractManipulator
from ..data_types import (
    SE3SO23,
    SO23,
    Mat3x3,
    Mat3x15,
    Mat6x6,
    Mat9x9,
    Vec3,
    Vec9,
)


def _jax_normalize_columns(X: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(X, axis=0, keepdims=True)  # shape (1, m)
    return X / norms


@jdc.pytree_dataclass(frozen=True)
class SE3SO23StewartDimension(AbstractManipulator[SE3SO23, Vec9]):
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
