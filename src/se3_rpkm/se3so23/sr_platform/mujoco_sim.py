import jax.numpy as jnp
import mujoco
from jaxlie import SO3

from se3_rpkm.se3so23.data_types import SO29

from .core import (
    SE3SO23,
    SE3SO23SRPlatform3RSerialArmKinematics,
    SE3SO23SRPlatformKinematics,
)

REVOLUTE_AXIS_CAPSULE_SIZE = [0.01, 0.02, 0.0]
REVOLUTE_LINK_RADIUS = 0.01
SITE_RADIUS = 0.015


def mjcf_spec_platform(dimension: SE3SO23SRPlatformKinematics) -> mujoco.MjSpec:  # type: ignore
    spec = mujoco.MjSpec()  # type: ignore
    spec.modelname = "se3so23_sr_platform"

    # Globally Disable Contact
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT  # type: ignore

    # End-Effector
    ee_body = spec.worldbody.add_body(name="body_ee")
    ee_body.add_site(
        size=[SITE_RADIUS] * 3,
        rgba=[0.0, 1.0, 0.0, 1],
    )
    # ee_body.add_freejoint()

    ## give EE a triangle plate mesh
    vertices_top = dimension.b_i + jnp.array([0, 0, 0.01])
    vertices_bottom = dimension.b_i - jnp.array([0, 0, 0])
    vertices = jnp.vstack([vertices_top, vertices_bottom]).astype(jnp.float32)
    mesh = spec.add_mesh(name="triangle_mesh")
    mesh.uservert = vertices.flatten().tolist()
    triangle_geom = ee_body.add_geom()
    triangle_geom.type = mujoco.mjtGeom.mjGEOM_MESH  # type: ignore
    triangle_geom.meshname = mesh.name

    ## attach redundant links to ee
    site_list = []
    for i in range(3):
        link_body = ee_body.add_body(
            name=f"body_link{i + 1}",
            pos=dimension.revolute_se3.translation()[i].tolist(),
            quat=dimension.revolute_se3.rotation().parameters()[i].tolist(),
        )
        link_body.add_joint(
            name=f"joint_link{i + 1}",
            type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
            pos=[0, 0, 0],
            axis=[0, 0, 1],
        )
        link_body.add_geom(  # revolute axis
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=REVOLUTE_AXIS_CAPSULE_SIZE,
            rgba=[0.5, 0.5, 0.5, 1],
        )

        link_body.add_geom(  # revolute link
            name=f"link_geom{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=[REVOLUTE_LINK_RADIUS, dimension.redundant_links[i] / 2, 0.0],
            rgba=[0.5, 0.5, 0.5, 1],
            pos=[dimension.redundant_links[i] / 2, 0, 0],
            quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
        )
        site_list.append(
            link_body.add_site(
                name=f"site_link{i + 1}",
                pos=[dimension.redundant_links[i], 0, 0],
                size=[SITE_RADIUS] * 3,
                rgba=[0, 0.5, 0, 1],
            )
        )
    return spec, site_list


def mjcf_spec_platform_and_gantry(
    dimension: SE3SO23SRPlatformKinematics,
) -> mujoco.MjSpec:  # type: ignore
    spec, site_list = mjcf_spec_platform(dimension)

    # Origin Site
    origin_site = spec.worldbody.add_site(
        name="site_origin",
        pos=[0.0, 0.0, 0.0],
        size=[SITE_RADIUS] * 3,
        rgba=[0.5, 0.5, 0.5, 1.0],
    )

    # Actuated Sites
    for index, s in enumerate(site_list):
        GAIN = 1.2e5
        free_body = spec.worldbody.add_body(
            mass=1,
            inertia=jnp.array([1.0, 1.0, 1.0]),  # type: ignore
        )
        free_body.add_freejoint()
        leg_site = free_body.add_site(
            name=f"leg_site_{index + 1}",
            size=[SITE_RADIUS] * 3,
            rgba=[0.0, 0.0, 1.0, 1],
        )

        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
        eq.name1 = s.name
        eq.name2 = leg_site.name
        eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore

        spec.add_actuator(
            name=f"act_{index + 1}_x",
            gainprm=[GAIN] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
            biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
            trntype=mujoco.mjtTrn.mjTRN_SITE,  # type: ignore
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target=leg_site.name,
            refsite=origin_site.name,
        )
        spec.add_actuator(
            name=f"act_{index + 1}_y",
            gainprm=[GAIN] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
            biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
            trntype=mujoco.mjtTrn.mjTRN_SITE,  # type: ignore
            gear=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            target=leg_site.name,
            refsite=origin_site.name,
        )
        spec.add_actuator(
            name=f"act_{index + 1}_z",
            gainprm=[GAIN] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
            biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
            trntype=mujoco.mjtTrn.mjTRN_SITE,  # type: ignore
            gear=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            target=leg_site.name,
            refsite=origin_site.name,
        )

    return spec


def mjcf_spec_platform_and_gantry_model_data(
    dimension: SE3SO23SRPlatformKinematics, x0: SE3SO23
) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:  # type: ignore
    spec = mjcf_spec_platform_and_gantry(dimension)
    spec.body("body_ee").pos = x0.pose.translation()
    spec.body("body_ee").quat = x0.pose.rotation().parameters()
    model = spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    data.ctrl = jnp.array(dimension.ik(x0))
    for i in range(int(1e5)):
        mujoco.mj_step(model, data)  # type: ignore
        if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-6:
            break
    if i >= int(1e5) - 1:
        raise RuntimeError("Failed to settle the simulation for initial position.")

    spec.body("body_ee").add_freejoint()

    key_frame = spec.add_key()  # this should be keyframe 0
    key_frame.name = "home"
    model, data0 = spec.recompile(model, data)
    key_frame.qpos = data0.qpos.copy()
    key_frame.ctrl = data0.ctrl.copy()
    model, data0 = spec.recompile(model, data0)
    return spec, model, data0


def mjcf_spec_platform_and_rrr_serial(
    dimension: SE3SO23SRPlatform3RSerialArmKinematics,
) -> mujoco.MjSpec:  # type: ignore
    spec, site_list = mjcf_spec_platform(dimension.platform)
    GAIN = 70

    for index_arm, (arm, platform_link_site) in enumerate(
        zip(dimension.serial_arm, site_list)
    ):
        # Body 1
        body_1 = spec.worldbody.add_body(
            pos=arm.t01.translation(),
            quat=arm.t01.rotation().parameters(),
        )
        ## Revolute Axis 1
        body_1.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=REVOLUTE_AXIS_CAPSULE_SIZE,
            rgba=[0.0, 0.0, 1.0, 1.0],
        )
        body_1.add_joint(
            name=f"arm_{index_arm + 1}_joint_1",
            type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
            pos=[0, 0, 0],
            axis=[0, 0, 1],
        )
        spec.add_actuator(
            name=f"act_{index_arm + 1}1",
            gainprm=[GAIN] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
            biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
            target=f"arm_{index_arm + 1}_joint_1",
        )
        ## Revolute Link 1
        body_1.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            pos=[arm.t12.translation()[0] / 2, 0, 0],
            quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
            size=[
                REVOLUTE_LINK_RADIUS,
                jnp.linalg.norm(arm.t12.translation()) / 2 + mujoco.mjMINVAL,  # type: ignore
                0.0,
            ],
            rgba=[0.0, 1.0, 0.0, 1.0],
        )

        # Body 2
        body_2 = body_1.add_body(
            pos=arm.t12.translation(),
            quat=arm.t12.rotation().parameters(),
        )
        ## Revolute Axis 2
        body_2.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=REVOLUTE_AXIS_CAPSULE_SIZE,
            rgba=[0.0, 0.0, 1.0, 1.0],
        )
        body_2.add_joint(
            name=f"arm_{index_arm + 1}_joint_2",
            type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
            pos=[0, 0, 0],
            axis=[0, 0, 1],
        )
        spec.add_actuator(
            name=f"act_{index_arm + 1}2",
            gainprm=[GAIN] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
            biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
            target=f"arm_{index_arm + 1}_joint_2",
        )
        ## Revolute Link 2
        body_2.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            pos=[arm.t23.translation()[0] / 2, 0, 0],
            quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
            size=[
                REVOLUTE_LINK_RADIUS,
                jnp.linalg.norm(arm.t23.translation()) / 2 + mujoco.mjMINVAL,  # type: ignore
                0.0,
            ],
            rgba=[0.0, 1.0, 0.0, 1.0],
        )

        # Body 3
        body_3 = body_2.add_body(
            pos=arm.t23.translation(),
            quat=arm.t23.rotation().parameters(),
        )
        ## Revolute Axis 3
        body_3.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            size=REVOLUTE_AXIS_CAPSULE_SIZE,
            rgba=[0.0, 0.0, 1.0, 1.0],
        )
        body_3.add_joint(
            name=f"arm_{index_arm + 1}_joint_3",
            type=mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
            pos=[0, 0, 0],
            axis=[0, 0, 1],
        )
        spec.add_actuator(
            name=f"act_{index_arm + 1}3",
            gainprm=[GAIN] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,  # type: ignore
            biasprm=[0.0, -GAIN, 1] + [0.0] * 7,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,  # type: ignore
            target=f"arm_{index_arm + 1}_joint_3",
        )
        ## Revolute Link 3
        body_3.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,  # type: ignore
            pos=[arm.t3e.translation()[0] / 2, 0, 0],
            quat=SO3.from_y_radians(jnp.pi / 2).parameters(),
            size=[
                REVOLUTE_LINK_RADIUS,
                jnp.linalg.norm(arm.t3e.translation()) / 2 + mujoco.mjMINVAL,  # type: ignore
                0.0,
            ],
            rgba=[0.0, 1.0, 0.0, 1.0],
        )
        s = body_3.add_site(
            name=f"arm_{index_arm + 1}_ee_site",
            pos=arm.t3e.translation(),
            size=[SITE_RADIUS] * 3,
            rgba=[1.0, 0.0, 0.0, 1.0],
        )

        # Connect to Platform
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_CONNECT  # type: ignore
        eq.name1 = s.name
        eq.name2 = platform_link_site.name
        eq.objtype = mujoco.mjtObj.mjOBJ_SITE  # type: ignore
    return spec


def mjcf_spec_platform_and_rrr_serial_model_data(
    dimension: SE3SO23SRPlatform3RSerialArmKinematics, x0: SE3SO23, q0: SO29
) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:  # type:ignore
    spec = mjcf_spec_platform_and_rrr_serial(dimension)
    spec.body("body_ee").pos = x0.pose.translation()
    spec.body("body_ee").quat = x0.pose.rotation().parameters()
    model = spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    data.ctrl = q0.as_radians().flatten()
    for i in range(int(1e5)):
        mujoco.mj_step(model, data)  # type: ignore
        if jnp.linalg.norm(data.qvel, ord=jnp.inf) < 1e-6:
            break
    if i >= int(1e5) - 1:
        raise RuntimeError("Failed to settle the simulation for initial position.")

    spec.body("body_ee").add_freejoint()

    key_frame = spec.add_key()  # this should be keyframe 0
    key_frame.name = "home"
    model, data0 = spec.recompile(model, data)
    key_frame.qpos = data0.qpos.copy()
    key_frame.ctrl = data0.ctrl.copy()
    model, data0 = spec.recompile(model, data0)
    return spec, model, data0
