import jax.numpy as jnp
import mujoco
import mujoco.viewer
from jaxlie import SE3, SO3

from se3_rpkm.data_types import SE3R3
from se3_rpkm.linear_redundant_stewart import RedundantR3LegStewartKinematics

if __name__ == "__main__":
    alpha = 70.3e-3
    beta_deg = 45
    h = 10e-3

    deg_120_3 = jnp.array([0.0, 120.0, 240.0])
    rad_120_3 = jnp.deg2rad(deg_120_3)

    dimension = RedundantR3LegStewartKinematics(
        v_i1=SO3.from_z_radians(jnp.deg2rad(50.0 + deg_120_3)).apply(
            jnp.array([[alpha, 0.0, 0.0]])
        ),
        v_i2=SO3.from_z_radians(jnp.deg2rad(-50.0 + deg_120_3)).apply(
            jnp.array([[alpha, 0.0, 0.0]])
        ),
        r_i_se3=SE3.from_rotation(SO3.from_z_radians(rad_120_3))
        @ SE3.from_rotation_and_translation(
            SO3.from_y_radians(jnp.deg2rad(-beta_deg)),
            jnp.array([100e-3, 0.0, 0.0]),
        ),
        a_i1_in_r=jnp.array([[0, h, 0]] * 3),
        a_i2_in_r=jnp.array([[0, -h, 0]] * 3),
    )

    # x = SE3R3(pose=SE3.identity(), rdof=jnp.array([0.1, 0.1, 0.1]))
    # print(dimension.ik(x))

    spec = dimension.mj_spec()
    model = spec.compile()
    mujoco.viewer.launch(model)

