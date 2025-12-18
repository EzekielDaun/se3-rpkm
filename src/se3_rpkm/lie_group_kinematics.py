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
    @abstractmethod
    def kinematic_constraints(self, task_coord: T, joint_coord: J) -> Float:
        raise NotImplementedError

    def jacobian_wrt_joint_log_tree(self, task_coord: T, joint_coord: J):
        def tangent_func(delta):
            return self.kinematic_constraints(
                task_coord, jaxlie.manifold.rplus(joint_coord, delta)
            )

        return jax.jacobian(tangent_func)(jaxlie.manifold.zero_tangents(joint_coord))

    def jacobian_wrt_joint_log_matrix(self, task_coord: T, joint_coord: J):
        return tree_reduce(
            lambda acc, x: jnp.hstack([acc.squeeze(), x.squeeze()]),
            self.jacobian_wrt_joint_log_tree(task_coord, joint_coord),
            is_leaf=lambda x: isinstance(x, jaxlie.MatrixLieGroup),
        ).squeeze()

    def jacobian_wrt_task_log_tree(self, task_coord: T, joint_coord: J):
        def tangent_func(delta):
            return self.kinematic_constraints(
                jaxlie.manifold.rplus(task_coord, delta), joint_coord
            )

        return jax.jacobian(tangent_func)(jaxlie.manifold.zero_tangents(task_coord))

    def jacobian_wrt_task_log_matrix(self, task_coord: T, joint_coord: J):
        return tree_reduce(
            lambda acc, x: jnp.hstack([acc.squeeze(), x.squeeze()]),
            self.jacobian_wrt_task_log_tree(task_coord, joint_coord),
            is_leaf=lambda x: isinstance(x, jaxlie.MatrixLieGroup),
        ).squeeze()

    def ik_jacobian(self, task_coord: T, joint_coord: J):
        jacobian_wrt_joint_log_matrix = self.jacobian_wrt_joint_log_matrix(
            task_coord, joint_coord
        )
        jacobian_wrt_task_log_matrix = self.jacobian_wrt_task_log_matrix(
            task_coord, joint_coord
        )

        return (
            -jnp.linalg.pinv(jacobian_wrt_joint_log_matrix)
            @ jacobian_wrt_task_log_matrix
        )

    def fk_jacobian(self, task_coord: T, joint_coord: J):
        jacobian_wrt_joint_log_matrix = self.jacobian_wrt_joint_log_matrix(
            task_coord, joint_coord
        )
        jacobian_wrt_task_log_matrix = self.jacobian_wrt_task_log_matrix(
            task_coord, joint_coord
        )

        return (
            -jnp.linalg.pinv(jacobian_wrt_task_log_matrix)
            @ jacobian_wrt_joint_log_matrix
        )

    def ik_optx(self, task_coord: T, initial_joint_coord: J, **kwargs) -> J:
        """ik_optx
        Inverse kinematics solver using Optimistix

        Args:
            task_coord (T): The desired task coordinate
            initial_joint_coord (J): Initial guess for the joint coordinate
            **kwargs: Additional keyword arguments for Optimistix root finder

        Returns:
            J: The computed joint coordinate that achieves the desired task coordinate
        """

        def f(x, args):
            return self.kinematic_constraints(task_coord, x)

        solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(f, solver, initial_joint_coord, **kwargs)
        return sol.value

    def fk_optx(self, joint_coord: J, initial_task_coord: T, **kwargs) -> T:
        """fk_optx

        Args:
            joint_coord (J): The given joint coordinate
            initial_task_coord (T): Initial guess for the task coordinate
            **kwargs: Additional keyword arguments for Optimistix root finder

        Returns:
            T: The computed task coordinate that corresponds to the given joint coordinate
        """

        def f(x, args):
            return self.kinematic_constraints(x, joint_coord)

        solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(f, solver, initial_task_coord, **kwargs)
        return sol.value
