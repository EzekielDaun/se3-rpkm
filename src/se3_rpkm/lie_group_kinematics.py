from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

import jax
import jax.numpy as jnp
import jaxlie
import optimistix as optx
from jax.tree_util import tree_reduce
from jax_dataclasses import pytree_dataclass
from jaxtyping import Float


@pytree_dataclass(frozen=True, slots=True)
class AbstractLieGroupTree(ABC):
    """Abstract base class for coordinate composed by Lie groups.

    Members must be MatrixLieGroup or 1D JAX arrays.
    """

    pass


T = TypeVar("T", bound=Union[AbstractLieGroupTree, jaxlie.MatrixLieGroup, Float])
J = TypeVar("J", bound=Union[AbstractLieGroupTree, jaxlie.MatrixLieGroup, Float])


class AbstractManipulator(Generic[T, J], ABC):
    """
    Abstract base class for manipulators defined by kinematic constraints between task coordinates and joint coordinates.

    - `T`: Type of the task coordinate
    - `J`: Type of the joint coordinate
    """

    @abstractmethod
    def kinematic_constraints(self, x: T, q: J) -> Float:
        """Implicit kinematics constraint function $f(x,q) = \\mathbf{0}_{\\dim{(x)} + \\dim{(q)}}$"""
        raise NotImplementedError

    def _jacobian_wrt_joint_tangent_tree(self, x: T, q: J):
        def tangent_func(delta):
            return self.kinematic_constraints(x, jaxlie.manifold.rplus(q, delta))

        return jax.jacobian(tangent_func)(jaxlie.manifold.zero_tangents(q))

    def jacobian_wrt_joint_tangent_matrix(self, x: T, q: J):
        """
        Effectively $\\frac{\\partial f}{\\partial q}$ if $q$ is in linear space.

        If $q$ is a Lie group element, this partial is with respect to the tangent space of $q$.
        """
        return tree_reduce(
            lambda acc, xx: jnp.hstack([acc.squeeze(), xx.squeeze()]),
            self._jacobian_wrt_joint_tangent_tree(x, q),
            is_leaf=lambda xx: isinstance(xx, jaxlie.MatrixLieGroup),
        ).squeeze()

    def _jacobian_wrt_task_tangent_tree(self, x: T, q: J):
        def tangent_func(delta):
            return self.kinematic_constraints(jaxlie.manifold.rplus(x, delta), q)

        return jax.jacobian(tangent_func)(jaxlie.manifold.zero_tangents(x))

    def jacobian_wrt_task_tangent_matrix(self, x: T, q: J):
        """
        Effectively $\\frac{\\partial f}{\\partial x}$ if $x$ is in linear space.

        If $x$ is a Lie group element, this partial is with respect to the tangent space of $x$.
        """
        return tree_reduce(
            lambda acc, xx: jnp.hstack([acc.squeeze(), xx.squeeze()]),
            self._jacobian_wrt_task_tangent_tree(x, q),
            is_leaf=lambda xx: isinstance(xx, jaxlie.MatrixLieGroup),
        ).squeeze()

    def ik_jacobian(self, x: T, q: J):
        """
        Effectively $\\frac{\\partial q}{\\partial x} \\approx \\frac{\\partial f}{\\partial q}^\\dagger \\frac{\\partial f}{\\partial x}$,
        if both are in linear space.

        Replace with tangent space partials if either is a Lie group element.
        """
        jacobian_wrt_joint_log_matrix = self.jacobian_wrt_joint_tangent_matrix(x, q)
        jacobian_wrt_task_log_matrix = self.jacobian_wrt_task_tangent_matrix(x, q)

        return (
            -jnp.linalg.pinv(jacobian_wrt_joint_log_matrix)
            @ jacobian_wrt_task_log_matrix
        )

    def fk_jacobian(self, x: T, q: J):
        """
        Effectively $\\frac{\\partial x}{\\partial q} \\approx \\frac{\\partial f}{\\partial x}^\\dagger \\frac{\\partial f}{\\partial q}$,
        if both are in linear space.

        Replace with tangent space partials if either is a Lie group element.
        """
        jacobian_wrt_joint_log_matrix = self.jacobian_wrt_joint_tangent_matrix(x, q)
        jacobian_wrt_task_log_matrix = self.jacobian_wrt_task_tangent_matrix(x, q)

        return (
            -jnp.linalg.pinv(jacobian_wrt_task_log_matrix)
            @ jacobian_wrt_joint_log_matrix
        )

    def ik_lm_optx(self, x: T, q0: J, **kwargs) -> J:
        """
        Levenberg-Marquardt (LM) inverse kinematics solver using Optimistix. Effectively $q(x, q_0).$

        Args:
            x: The desired task coordinate
            q0: Initial guess for the joint coordinate
            **kwargs: Additional keyword arguments for Optimistix root finder

        Returns:
            J: The computed joint coordinate that achieves the desired task coordinate
        """

        def f(xx, _args):
            return self.kinematic_constraints(x, xx)

        solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(f, solver, q0, **kwargs)
        return sol.value

    def fk_lm_optx(self, q: J, x0: T, **kwargs) -> T:
        """
        Levenberg-Marquardt (LM) forward kinematics solver using Optimistix. Effectively $x(q, x_0).$

        Args:
            q: The given joint coordinate
            x0: Initial guess for the task coordinate
            **kwargs: Additional keyword arguments for Optimistix root finder

        Returns:
            T: The computed task coordinate that corresponds to the given joint coordinate
        """

        def f(xx, _args):
            return self.kinematic_constraints(xx, q)

        solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(f, solver, x0, **kwargs)
        return sol.value
