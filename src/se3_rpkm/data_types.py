from typing import Annotated

import jax_dataclasses as jdc
from jaxlie import SE3, SO2
from jaxtyping import Array, Float

from .lie_group_kinematics import (
    AbstractLieGroupTree,
)

SO23 = Annotated[SO2, "Batch of 3 SO2"]
SO29 = Annotated[SO2, "Batch of 9 SO2"]
Vec3 = Annotated[Float[Array, "3"], ""]
Vec9 = Annotated[Float[Array, "9"], ""]
Mat3x3 = Annotated[Float[Array, "3 3"], ""]  # noqa: F722
Mat6x6 = Annotated[Float[Array, "6 6"], ""]  # noqa: F722
Mat3x15 = Annotated[Float[Array, "3 15"], ""]  # noqa: F722
Mat9x9 = Annotated[Float[Array, "9 9"], ""]  # noqa: F722


@jdc.pytree_dataclass(frozen=True, slots=True)
class SE3SO23(AbstractLieGroupTree):
    pose: SE3
    rdof: SO23
