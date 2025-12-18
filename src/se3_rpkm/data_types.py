from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated

import jax_dataclasses as jdc
from jaxlie import SE3, SO2, SO3
from jaxtyping import Array, Float

from .lie_group_kinematics import (
    AbstractLieGroupTree,
)

if TYPE_CHECKING:
    import mujoco

SO22 = Annotated[SO2, "Batch of 2 SO2"]
SO23 = Annotated[SO2, "Batch of 3 SO2"]
SO29 = Annotated[SO2, "Batch of 9 SO2"]
Vec3 = Annotated[Float[Array, "3"], ""]
Vec8 = Annotated[Float[Array, "8"], ""]
Vec9 = Annotated[Float[Array, "9"], ""]
Mat2x3 = Annotated[Float[Array, "2 3"], ""]  # noqa: F722
Mat3x3 = Annotated[Float[Array, "3 3"], ""]  # noqa: F722
Mat4x3 = Annotated[Float[Array, "4 3"], ""]  # noqa: F722
Mat6x6 = Annotated[Float[Array, "6 6"], ""]  # noqa: F722
Mat8x8 = Annotated[Float[Array, "8 8"], ""]  # noqa: F722
Mat9x9 = Annotated[Float[Array, "9 9"], ""]  # noqa: F722
Mat3x15 = Annotated[Float[Array, "3 15"], ""]  # noqa: F722


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO22(AbstractLieGroupTree):
    pose: SE3
    rdof: SO22


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO23(AbstractLieGroupTree):
    pose: SE3
    rdof: SO23


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO3(AbstractLieGroupTree):
    pose: SE3
    rdof: SO3


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3R3(AbstractLieGroupTree):
    pose: SE3
    rdof: Vec3
