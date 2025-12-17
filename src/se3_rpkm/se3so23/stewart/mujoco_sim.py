import jax.numpy as jnp
import mujoco
from jaxlie import SO3

from ..data_types import SE3SO23
from .core import SE3SO23StewartDimension


def mjcf_spec(
    dimension: SE3SO23StewartDimension,
    act_lower_radius=0.02,
    act_lower_length=0.15,
    act_upper_radius=0.01,
    act_upper_length=0.5,
) -> mujoco.MjSpec:  # type: ignore
    spec = mujoco.MjSpec()  # type: ignore
    spec.modelname = "se3so23_stewart"

    # Globally Disable Contact
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

    # End-Effector
    ee_body = spec.worldbody.add_body(name="body_ee")
    # ee_body.add_freejoint()

    ## give EE a hex plate mesh
    vertices = jnp.hstack([dimension.v_i, dimension.v_j]).T
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
        for i, v_i in enumerate(dimension.v_i.T)
    ]
    site_Bjx_ee = [
        ee_body.add_site(
            name=f"site_B{j + 1}x_ee",
            pos=v_j,
            size=[act_upper_radius * 1.5, 0.0, 0.0],
            rgba=[0.0, 1.0, 0.0, 1.0],
        )
        for j, v_j in enumerate(dimension.v_j.T)
    ]

    # Base
    ## Redundant Legs Attachment Sites
    site_Aj1_base = []
    site_Aj2_base = []
    for j, a_j1 in enumerate(dimension.a_j1.T):
        site_Aj1_base.append(
            spec.worldbody.add_site(
                name=f"site_A{j + 1}1_base",
                pos=a_j1,
                size=[act_upper_radius * 1.5, 0.0, 0.0],
                rgba=[1.0, 0.0, 0.0, 1.0],
            )
        )
    for j, a_j2 in enumerate(dimension.a_j2.T):
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
    for i, a_i in enumerate(dimension.a_i.T):
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
        zip(dimension.a_j1.T, dimension.a_j2.T, dimension.redundant_links)
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


def mjcf_model_data(
    dimension: SE3SO23StewartDimension,
    x0: SE3SO23,
) -> tuple[mujoco.MjModel, mujoco.MjData]:  # type: ignore
    spec = mjcf_spec(dimension)
    spec.body("body_ee").pos = x0.pose.translation().tolist()
    spec.body("body_ee").quat = x0.pose.rotation().parameters().tolist()
    model = spec.compile()  # type: ignore
    data = mujoco.MjData(model)  # type: ignore

    data.ctrl = jnp.array(dimension.ik(x0))
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
    return model, data0
