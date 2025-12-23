import jax.numpy as jnp
import mujoco
import mujoco.viewer
from jaxlie import SE3, SO3

from se3_rpkm.data_types import SE3SO3
from se3_rpkm.se3so3_5pss_s_4pss import SE3SO3_5PSS_S_4PSS_Kinematics

if __name__ == "__main__":
    dimension = SE3SO3_5PSS_S_4PSS_Kinematics(
        slider_axis=SE3.from_translation(
            SO3.from_z_radians(
                jnp.deg2rad(jnp.array([-50, -40, 40, 50, 130, 140, 180, 220, 230]))
            ).apply(jnp.array([[250e-3, 0.0, 0.0]]))
        ),
        link_length=0.3 * jnp.ones(9),
        a1_a5=SO3.from_z_radians(jnp.deg2rad(jnp.array([-80, -5, 5, 80, 100]))).apply(
            jnp.array([[150e-3, 0.0, 0.0]])
        ),
        a6_a9=SO3.from_z_radians(jnp.deg2rad(jnp.array([110, 180, 185, 240]))).apply(
            jnp.array([[150e-3, 0.0, 0.0]])
        ),
    )

    # print(dimension.a_i(SE3SO3(SE3.identity(), SO3.identity())))
    # print(dimension.b_i(0.1 * jnp.ones(9)))
    # print(
    #     dimension.kinematic_constraints(
    #         SE3SO3(SE3.identity(), SO3.identity()), jnp.zeros((9, 3))
    #     )
    # )

    x0 = SE3SO3(
        pose=SE3.from_translation(jnp.array([0.0, 0.0, 250e-3])), rdof=SO3.identity()
    )
    q0 = jnp.ones(9) * 0.1

    spec = dimension.mj_spec()
    model = spec.compile()

    mujoco.viewer.launch(model)
