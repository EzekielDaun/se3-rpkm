from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Generic, Protocol, TypeVar, Union

import jax_dataclasses as jdc
from jaxlie import SE3, SO2, SO3, MatrixLieGroup
from jaxtyping import Array, Float

from .lie_group_kinematics import (
    AbstractLieGroupTree,
)

try:
    import mujoco
except ImportError:
    mujoco = None

if TYPE_CHECKING:
    import mujoco as mujoco_t

SO22 = Annotated[SO2, "Batch of 2 SO2"]
SO23 = Annotated[SO2, "Batch of 3 SO2"]
SO29 = Annotated[SO2, "Batch of 9 SO2"]
SE33 = Annotated[SE3, "Batch of 3 SE3"]
SE39 = Annotated[SE3, "Batch of 9 SE3"]
Vec3 = Annotated[Float[Array, "3"], ""]
Vec8 = Annotated[Float[Array, "8"], ""]
Vec9 = Annotated[Float[Array, "9"], ""]
Mat2x3 = Annotated[Float[Array, "2 3"], ""]  # noqa: F722
Mat3x3 = Annotated[Float[Array, "3 3"], ""]  # noqa: F722
Mat4x3 = Annotated[Float[Array, "4 3"], ""]  # noqa: F722
Mat5x3 = Annotated[Float[Array, "5 3"], ""]  # noqa: F722
Mat9x3 = Annotated[Float[Array, "9 3"], ""]  # noqa: F722
Mat8x8 = Annotated[Float[Array, "8 8"], ""]  # noqa: F722

T = TypeVar("T", bound=Union[MatrixLieGroup, Float])


class RedundantSE3TaskCoordinate(Generic[T], Protocol):
    pose: SE3
    rdof: T


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO22(AbstractLieGroupTree, RedundantSE3TaskCoordinate[SO22]):
    pose: SE3
    rdof: SO22


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO23(AbstractLieGroupTree, RedundantSE3TaskCoordinate[SO23]):
    pose: SE3
    rdof: SO23


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO3(AbstractLieGroupTree, RedundantSE3TaskCoordinate[SO3]):
    pose: SE3
    rdof: SO3


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3R3(AbstractLieGroupTree, RedundantSE3TaskCoordinate[Vec3]):
    pose: SE3
    rdof: Vec3


class MuJoCoMixin(ABC):
    @staticmethod
    def _check_mujoco_availability():
        if mujoco is None:
            raise ImportError(
                "MuJoCo is not installed. Please install mujoco-python to use this functionality."
            )

    @abstractmethod
    def mj_spec(self, *args, **kwargs) -> "mujoco_t.MjSpec":  # type: ignore
        """mj_spec

        Convert mechanism geometry information (dimension, joints, constraints etc.) into a mujoco.MjSpec object.
        No joint position data included, only the model specification. Implementation can take any arguments as needed.

        Returns:
            mujoco.MjSpec: The MJCF specification of the mechanism.
        """
        self._check_mujoco_availability()
        raise NotImplementedError

    @abstractmethod
    def mj_spec_model_data(
        self, *args, **kwargs
    ) -> tuple["mujoco_t.MjSpec", "mujoco_t.MjModel", "mujoco_t.MjData"]:  # type: ignore
        """mjcf_spec_model_data

        Convert mechanism geometry information (dimension, joints, constraints etc.) into a mujoco.MjSpec object,
        and also create the corresponding MjModel and MjData objects, where the MjData can be a safe initial configuration. Implementation can take any arguments as needed.

        Returns:
            tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]: The MJCF specification of the mechanism,
            along with the corresponding MjModel and MjData objects.
        """
        self._check_mujoco_availability()
        raise NotImplementedError
