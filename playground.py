import jax.numpy as jnp
from jaxlie import SE3, SO2, SO3

from se3_rpkm.data_types import SE3SO22
from se3_rpkm.stewart import SE3SO22StewartKinematics

if __name__ == "__main__":
    beta = 2.25
    rot_90 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
    rot_180 = SO3.from_z_radians(jnp.linspace(0, 2 * jnp.pi, 2, endpoint=False))
    dimension = SE3SO22StewartKinematics(
        a_i=rot_90.inverse().apply(jnp.array([beta, beta, 0.0])),
        a_j1=rot_180.apply(jnp.array([beta, beta, 0.0])),
        a_j2=rot_180.apply(jnp.array([beta, -beta, 0.0])),
        v_i=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        v_j=rot_180.apply(jnp.array([1.0, 0.0, 0.0])),
        l_j=jnp.array([0.1, 0.1]),
    )
    x = SE3SO22(
        pose=SE3.identity(), rdof=SO2.from_radians(jnp.array([jnp.pi / 2, jnp.pi / 2]))
    )
    print(dimension.ik(x))
