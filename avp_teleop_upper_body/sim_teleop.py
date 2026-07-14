"""sim_teleop (upper body): drive the Astribot S1 upper body in MuJoCo.

Pipeline (this process):

    UDP {head, left, right}  ->  merged whole-body IK + finger retarget
                             ->  data.ctrl  ->  mj_step  ->  viewer

A single :class:`WholeBodyIK` solves the 23-DOF whole body (mobile base + torso
+ neck + both arms) so the robot head camera tracks the AVP head pose while the
two tool frames track the hands. The arms automatically compensate for torso
motion, and the base translates/yaws to extend reach for out-of-reach targets,
because all three end-effector tasks share one configuration.

Run inside the ``AVP`` conda env (after starting the publisher in another shell):

    python -m avp_teleop_upper_body.sim_teleop
    python -m avp_teleop_upper_body.sim_teleop --orientation

Viewer keys:
    c      (re)calibrate: anchor head + both hands to the robot's current poses
    space  pause / resume teleop (sim keeps running)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

# Reuse the dual-arm package's robot interface, calibration and finger retarget.
from avp_teleop.robot_interface import SimRobot
from avp_teleop.retarget.hand_retarget import HandRetargeter
from avp_teleop.retarget.frames import WristCalibration, wrist_to_tool_target

from avp_teleop_upper_body.config import (
    default_config,
    MJCF_PATH,
    BODY_JOINTS,
    IK_KEEP_JOINTS,
    BODY_HOME,
    HEAD_FRAME_BODY,
    CHASSIS_BASE_FRAME,
    TORSO_LEAN_JOINTS,
    NEURAL_WAIST_JOINT,
    NEURAL_PITCH_JOINT,
    CHEST_HEAD_FRAME,
    CHEST_HIP_FRAME,
    CHEST_ANKLE_FRAME,
    TOOL_BODY,
    all_finger_joints,
    finger_specs,
)
from avp_teleop_upper_body.transport import UpperBodySubscriber
from avp_teleop_upper_body.whole_body_ik import WholeBodyIK
from avp_teleop_upper_body.pose_filter import PoseFilter
from avp_teleop_upper_body.trajectory_io import (
    AvpTrajectoryRecorder,
    RetargetTrajectoryRecorder,
    FileAvpSource,
    load_avp_trajectory,
    load_retarget_trajectory,
)
from avp_teleop_upper_body import pose_io

Target = Tuple[np.ndarray, Optional[np.ndarray]]  # (world_p, world_R_or_None)


def _body_pose(model, data, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """(R, p) world pose of a body, read from current MuJoCo data."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy()


def _prior_anchor_pose(model, data, hip_joint_name: str,
                       waist_body_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Upright, heading-aligned anchor frame for the EgoPoser prior wireframe.

    The prior skeleton is built world-aligned (+Z up, forward lean toward the
    robot's -Y face), so it must NOT be anchored in a torso link body frame --
    those are rotated 90 degrees (their local +Z points to world +X), which is
    what sprayed the spine horizontally. Instead we anchor at the *hip* joint
    (``astribot_torso_joint_3``, the top of the sagittal lean spine ~= the human
    pelvis; joint_1/2/3 are ankle/knee/hip on the wheeled base), with a frame
    that keeps world +Z up and only yaws by the robot's waist heading so the
    skeleton turns with the robot.

    Returns ``(R, p)``: ``p`` = hip joint world anchor, ``R`` = rotation about
    world Z by the waist yaw.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint_name)
    p = data.xanchor[jid].copy()
    # Waist heading: the waist link's local +X axis is the robot's forward, which
    # points to world -Y at neutral. Yaw the skeleton (its own forward is also
    # -Y) about world Z to follow it: Rz(yaw)@[0,-1,0] = forward_xy.
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, waist_body_name)
    fwd = data.xmat[bid].reshape(3, 3)[:, 0]       # local +X in world
    yaw = float(np.arctan2(fwd[0], -fwd[1]))
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R, p


def _T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _set_body_home(model, data, robot: SimRobot, home) -> None:
    """Place the body joints at home and hold them via the actuators."""
    for adr, qi in zip(robot._arm_qpos_adr, home):
        data.qpos[adr] = qi
    mujoco.mj_forward(model, data)
    robot.command_arm(np.asarray(home))


# Colours (RGBA) for the EgoPoser prior wireframe. The full SMPL body is drawn;
# joints/bones that actually feed the robot retargeting (the sagittal spine) are
# DARK/saturated, the rest (legs, feet, arms, collars, neck, head -- hallucinated
# but unused by the map) are LIGHT/desaturated so the two are visually distinct.
_PRIOR_BONE_MAPPED_RGBA   = np.array([0.00, 0.55, 0.80, 1.00])  # deep cyan (mapped spine)
_PRIOR_NODE_MAPPED_RGBA   = np.array([0.95, 0.35, 0.00, 1.00])  # deep orange (mapped joints)
_PRIOR_BONE_UNMAPPED_RGBA = np.array([0.55, 0.85, 0.95, 0.55])  # pale cyan (unused bones)
_PRIOR_NODE_UNMAPPED_RGBA = np.array([1.00, 0.78, 0.55, 0.55])  # pale orange (unused joints)
# Waist link body (child of the waist yaw joint) -- used to read the robot's
# heading so the prior wireframe turns with the robot.
_WAIST_LINK_BODY = "astribot_torso_link_4"

# --- IK command-target marker colours (RGBA) ----------------------------- #
# One coloured sphere per commanded end-effector target (the incremental,
# scaled, robot-world pose the IK is solving toward), plus an RGB orientation
# triad (X=red, Y=green, Z=blue). A thin line from the robot's ACTUAL tool frame
# to the target sphere shows the live tracking error.
_TGT_HEAD_RGBA  = np.array([1.00, 0.90, 0.20, 1.0])   # yellow = head command
_TGT_LEFT_RGBA  = np.array([0.60, 0.30, 0.95, 1.0])   # purple = left  command
_TGT_RIGHT_RGBA = np.array([0.20, 0.90, 0.85, 1.0])   # teal   = right command
_ERR_LINE_RGBA  = np.array([0.95, 0.95, 0.95, 0.7])   # pale white = tracking-error line
_AXIS_X_RGBA = np.array([0.90, 0.20, 0.20, 1.0])      # red   = +X
_AXIS_Y_RGBA = np.array([0.20, 0.80, 0.20, 1.0])      # green = +Y
_AXIS_Z_RGBA = np.array([0.25, 0.45, 0.95, 1.0])      # blue  = +Z
_TGT_MARKER_RGBA = {"head": _TGT_HEAD_RGBA, "left": _TGT_LEFT_RGBA,
                    "right": _TGT_RIGHT_RGBA}


def _scn_sphere(scn, pos, rgba, r) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([r, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(9), rgba.astype(np.float32))
    scn.ngeom += 1


def _scn_connector(scn, gtype, a, b, rgba, width) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(gtype), np.zeros(3), np.zeros(3),
                        np.eye(3).reshape(9), rgba.astype(np.float32))
    mujoco.mjv_connector(g, int(gtype), width,
                         np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))
    scn.ngeom += 1


def _draw_target_marker(scn, p, R, marker_rgba, actual_p=None, *,
                        sphere_r=0.02, axis_len=0.12, axis_width=0.008,
                        err_width=0.004) -> None:
    """Draw one IK command target: sphere at ``p``, RGB orient triad if ``R``.

    ``p`` / ``R`` are the commanded target in robot world frame (``R`` may be
    None -> no triad). If ``actual_p`` (the robot's current tool position) is
    given, a thin line target->actual visualises the tracking error.
    """
    _scn_sphere(scn, p, marker_rgba, sphere_r)
    if R is not None:
        for col, rgba in ((0, _AXIS_X_RGBA), (1, _AXIS_Y_RGBA), (2, _AXIS_Z_RGBA)):
            _scn_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, p,
                           p + axis_len * np.asarray(R)[:, col], rgba, axis_width)
    if actual_p is not None:
        _scn_connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, actual_p, p,
                       _ERR_LINE_RGBA, err_width)


def _draw_pose_frame(scn, R, p, *, sphere_rgba=None, sphere_r=0.03,
                     axis_len=0.15, axis_width=0.01) -> None:
    """Draw a coordinate frame: RGB=XYZ axis triad at ``p`` oriented by ``R``,
    with an optional coloured sphere at the origin. Used to show the world frame
    and the raw AVP head / hand poses (see ``--show-input-frames``)."""
    if sphere_rgba is not None:
        _scn_sphere(scn, p, sphere_rgba, sphere_r)
    Rm = np.asarray(R, dtype=np.float64)
    for col, rgba in ((0, _AXIS_X_RGBA), (1, _AXIS_Y_RGBA), (2, _AXIS_Z_RGBA)):
        _scn_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, p,
                       p + axis_len * Rm[:, col], rgba, axis_width)


def _draw_prior_skeleton(scn, anchor_R, anchor_p, skeleton, parents, mapped, *,
                         width=0.006, node_r=0.012) -> None:
    """Draw the full EgoPoser SMPL body skeleton into a viewer user scene.

    ``skeleton`` is the (22, 3) pelvis-local joint set from :class:`TrunkPrior`
    (pelvis + the 21 pose_body joints), ``parents`` its SMPL kinematic tree
    (``_SMPL_PARENTS``) and ``mapped`` the joint ids that feed the robot
    retargeting (``_SMPL_MAPPED_JOINTS``). ``anchor_R``/``anchor_p`` place the
    pelvis at a robot frame (world). Bones are thin capsules, joints small
    spheres, appended to ``scn`` (e.g. the passive viewer's ``user_scn``).

    Joints/bones whose (child) joint is in ``mapped`` are drawn DARK, the rest
    LIGHT, so the spine that drives the trunk map stands out from the merely
    hallucinated limbs. Silently no-ops when the scene's geom buffer is full.
    """
    world = anchor_p[None, :] + skeleton @ anchor_R.T   # (22, 3) world points

    def _add_sphere(pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([node_r, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
            np.eye(3).reshape(9), rgba.astype(np.float32))
        scn.ngeom += 1

    def _add_bone(a, b, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.zeros(3),
            np.zeros(3), np.eye(3).reshape(9), rgba.astype(np.float32))
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), width,
                             np.asarray(a, dtype=np.float64),
                             np.asarray(b, dtype=np.float64))
        scn.ngeom += 1

    # Bones: one per non-root joint (parent -> joint). A bone is "mapped" iff its
    # child joint feeds the retargeting.
    for jid in range(len(world)):
        par = int(parents[jid])
        if par < 0:
            continue
        is_mapped = jid in mapped and par in mapped
        _add_bone(world[par], world[jid],
                  _PRIOR_BONE_MAPPED_RGBA if is_mapped else _PRIOR_BONE_UNMAPPED_RGBA)
    # Joints on top of the bones.
    for jid, pt in enumerate(world):
        _add_sphere(pt, _PRIOR_NODE_MAPPED_RGBA if jid in mapped
                    else _PRIOR_NODE_UNMAPPED_RGBA)


# Flat gray applied to every robot/scene material when textures are stripped.
_FLAT_GRAY = np.array([0.55, 0.55, 0.57, 1.0], dtype=np.float32)


def _apply_simple_render(model, *, body_alpha: float = 1.0) -> None:
    """Strip textures/shadows and flatten the scene to gray (cheaper to render).

    - Detaches textures from every material and paints materials + geoms a flat
      gray, so no texture upload/sampling happens.
    - Disables shadow casting on all lights.
    - Optionally makes the robot body translucent (``body_alpha`` < 1) so the
      EgoPoser prior wireframe drawn inside it stays visible. The ground plane
      keeps full opacity.

    Modifies ``model`` in place; combine with the per-frame render flags set on
    the viewer (skybox/reflection/haze) in :func:`main`.
    """
    # Flatten materials: drop texture, reset colour to gray, kill reflectance.
    if model.nmat:
        model.mat_texid[:] = -1
        model.mat_rgba[:] = _FLAT_GRAY
        model.mat_reflectance[:] = 0.0
    # Paint geoms gray too (covers geoms with no material) and set alpha. The
    # floor plane (geom_type PLANE) stays opaque; robot geoms get body_alpha.
    for gi in range(model.ngeom):
        model.geom_rgba[gi] = _FLAT_GRAY
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            model.geom_rgba[gi, 3] = body_alpha
    # No shadows.
    if model.nlight:
        model.light_castshadow[:] = 0


def main() -> None:
    cfg = default_config()
    parser = argparse.ArgumentParser(
        description="AVP -> MuJoCo upper-body teleop (head + torso + dual arm)."
    )
    parser.add_argument("--host", default=cfg.network.host)
    parser.add_argument("--port", type=int, default=cfg.network.port)
    parser.add_argument("--mjcf", default=None, help="Override teleop MJCF path.")
    parser.add_argument("--orientation", action="store_true",
                        help="Also track hand (arm) orientation (6-DOF arms).")
    parser.add_argument("--no-orientation", action="store_true",
                        help="Track hand position only (default).")
    parser.add_argument("--head-no-orientation", action="store_true",
                        help="Track head position only (default tracks head gaze).")
    parser.add_argument("--position-scale", type=float,
                        default=cfg.retarget.position_scale)
    parser.add_argument("--head-position-scale", type=float,
                        default=cfg.head_position_scale)
    parser.add_argument("--init-pose", default=None,
                        help="Saved pose name/path (see pose_editor) to use as "
                             "the initial + rest posture instead of BODY_HOME.")
    parser.add_argument("--alpha-translation", type=float,
                        default=cfg.filter.alpha_translation,
                        help="EMA smoothing of target translation, in (0,1]: "
                             "1=off, smaller=smoother but laggier (default %(default)s).")
    parser.add_argument("--alpha-rotation", type=float,
                        default=cfg.filter.alpha_rotation,
                        help="SLERP smoothing of target rotation, in (0,1]: "
                             "1=off, smaller=smoother but laggier (default %(default)s).")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable target pose smoothing (alpha=1.0 on both).")
    parser.add_argument("--egoposer", action="store_true",
                        help="Enable the EgoPoser neural trunk-posture prior "
                             "(hallucinates a natural trunk pitch + waist yaw "
                             "from head/hand poses; low-cost soft QP task, "
                             "balance/precision always dominate). Needs torch + "
                             "weights; degrades gracefully to off if missing.")
    parser.add_argument("--egoposer-weights", default=None,
                        help="Path to the EgoPoser checkpoint (.pth). Overrides "
                             "the config; enables the prior if a file is given.")
    parser.add_argument("--neural-posture-cost", type=float, default=None,
                        help="Override the NeuralPostureTask cost (default "
                             f"{cfg.ik.neural_posture_cost}); higher = stronger "
                             "human trunk bias. Only used with --egoposer.")
    parser.add_argument("--visualize-prior", action="store_true",
                        help="Draw the EgoPoser-hallucinated FULL SMPL body "
                             "skeleton (legs, spine, arms, neck, head) as a "
                             "wireframe in the viewer, anchored upright at the "
                             "robot's hip. The sagittal spine that drives the "
                             "trunk map is drawn DARK, the rest (unused, "
                             "hallucinated limbs) LIGHT. Implies --egoposer.")
    parser.add_argument("--rich-render", action="store_true",
                        help="Keep the full textured/shadowed rendering. By "
                             "default the scene is flattened to gray with no "
                             "textures/shadows/skybox (cheaper to render).")
    parser.add_argument("--body-alpha", type=float, default=None,
                        help="Robot body opacity in [0,1] (default 1.0, or 0.35 "
                             "with --visualize-prior so the skeleton shows "
                             "through). Ignored with --rich-render.")
    parser.add_argument("--no-target-markers", action="store_true",
                        help="Hide the IK command-target markers (shown by "
                             "default: yellow=head, purple=left, teal=right "
                             "sphere + RGB orientation triad, plus a thin line to "
                             "the robot's actual tool frame = tracking error).")
    # --- record / replay (see avp_teleop_upper_body.trajectory_io) --------- #
    parser.add_argument("--replay-avp", default=None, metavar="NAME",
                        help="Replay a recorded raw-AVP input clip from "
                             "avp_trajectory/ (bare name or path) instead of "
                             "listening on UDP. Runs the full retarget+IK+render "
                             "with no headset; auto-calibrates on the first "
                             "valid frame and stops at the end.")
    parser.add_argument("--replay-loop", action="store_true",
                        help="Loop the --replay-avp clip instead of stopping "
                             "at the end.")
    parser.add_argument("--record-avp", default=None, metavar="NAME",
                        help="Record the raw AVP input stream to "
                             "avp_trajectory/NAME.json (replayable via "
                             "--replay-avp). Works with a live headset or "
                             "alongside --replay-avp.")
    parser.add_argument("--record-retarget", default=None, metavar="NAME",
                        help="Record the FULL retarget trace (CLI args + AVP "
                             "input + targets + joint angles + fingers + prior) "
                             "to retargetting_trajectory/NAME.json for pure "
                             "replay (replay_retarget.py) and offline analysis.")
    parser.add_argument("--record-trim", type=float, default=2.0, metavar="SEC",
                        help="Trim this many seconds off BOTH ends of a recording "
                             "on save (default 2.0), to discard the keyboard-"
                             "fumble while you start/stop recording. 0 = keep the "
                             "full clip. Applies to --record-avp and "
                             "--record-retarget alike.")
    parser.add_argument("--show-input-frames", action="store_true",
                        help="Draw the raw AVP head + hand poses and the world "
                             "origin as coordinate triads (RGB = XYZ), in the "
                             "robot-world frame (via align_R). Auto-enabled while "
                             "--record-avp so you can verify the input you record.")
    parser.add_argument("--no-input-frames", action="store_true",
                        help="Force the raw-AVP input frames OFF even while "
                             "recording (overrides the --record-avp auto-on).")
    args = parser.parse_args()


    cfg.retarget.position_scale = args.position_scale
    cfg.head_position_scale = args.head_position_scale
    cfg.filter.alpha_translation = args.alpha_translation
    cfg.filter.alpha_rotation = args.alpha_rotation
    if args.no_filter:
        cfg.filter.alpha_translation = 1.0
        cfg.filter.alpha_rotation = 1.0
    if args.orientation:
        cfg.retarget.track_orientation = True
    if args.no_orientation:
        cfg.retarget.track_orientation = False
    if args.head_no_orientation:
        cfg.head_track_orientation = False

    # EgoPoser neural trunk-posture prior (opt-in). --egoposer or supplying a
    # weights path enables it; --visualize-prior also implies it. The task cost
    # may be overridden on the CLI.
    if args.egoposer or args.egoposer_weights or args.visualize_prior:
        cfg.egoposer.enabled = True
    if args.egoposer_weights:
        cfg.egoposer.weights_path = args.egoposer_weights
    if args.neural_posture_cost is not None:
        cfg.ik.neural_posture_cost = args.neural_posture_cost
    # If the prior is off, force the task cost to 0 so the solver never builds
    # the NeuralPostureTask (byte-for-byte the non-neural pipeline). If it is on
    # but the cost is still 0, give it a sensible default so the flag does work.
    if not cfg.egoposer.enabled:
        cfg.ik.neural_posture_cost = 0.0
    elif cfg.ik.neural_posture_cost <= 0.0:
        cfg.ik.neural_posture_cost = 5.0

    # The neural prior tracks its forward-lean target on the HIP joint alone
    # (NEURAL_PITCH_JOINT = torso_joint_3), leaving torso_joint_1/2 free for the
    # balance task. TrunkUprightTask, however, pulls the SUM theta_1+theta_2+theta_3
    # -> 0; left active it would drive torso_joint_1/2 to cancel the hip bias back
    # toward a ground-vertical trunk, defeating the point. So while the prior is on,
    # disable trunk-upright and let ChestOverAnkleTask own balance entirely.
    if cfg.ik.neural_posture_cost > 0.0 and cfg.ik.trunk_upright_cost > 0.0:
        print(f"[SIM] EgoPoser prior ON -> disabling trunk_upright_cost "
              f"({cfg.ik.trunk_upright_cost:g} -> 0) so the hip-joint lean bias is "
              f"not cancelled; balance stays with ChestOverAnkleTask.")
        cfg.ik.trunk_upright_cost = 0.0

    mjcf_path = args.mjcf or MJCF_PATH
    print(f"[SIM] Loading {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Rendering: by default flatten to gray (no textures/shadows) for a cheaper,
    # cleaner scene; --rich-render keeps the stock look. Body opacity defaults to
    # translucent when visualising the prior so the wireframe shows through.
    if not args.rich_render:
        default_alpha = 0.35 if args.visualize_prior else 1.0
        body_alpha = float(np.clip(args.body_alpha, 0.0, 1.0)) \
            if args.body_alpha is not None else default_alpha
        _apply_simple_render(model, body_alpha=body_alpha)
        print(f"[SIM] Simple render: gray, no textures/shadows/skybox "
              f"(body alpha={body_alpha:g}). Use --rich-render for full detail.")

    # Initial + rest posture: a saved pose (pose_editor) if given, else home.
    if args.init_pose:
        body_home = pose_io.body_vector(pose_io.load_pose(args.init_pose))
        print(f"[SIM] Initial pose: '{args.init_pose}'")
    else:
        body_home = np.array(BODY_HOME, dtype=np.float64)
        print("[SIM] Initial pose: BODY_HOME (default)")

    # One robot owns the 23 body joints (chassis + torso + neck + arms) + all
    # finger joints; its "tool" is the
    # head camera (used for head calibration). Tool-link poses are read directly.
    body_robot = SimRobot(model, data, BODY_JOINTS, all_finger_joints(),
                          HEAD_FRAME_BODY)
    ranges = body_robot.joint_ranges()
    hand_retarget = {
        "left": HandRetargeter(finger_specs("left"), cfg.retarget),
        "right": HandRetargeter(finger_specs("right"), cfg.retarget),
    }

    # Orientation costs gate on the track flags (0 -> task ignores rotation).
    arm_ori = cfg.ik.arm_orientation_cost if cfg.track_orientation else 0.0
    head_ori = cfg.ik.head_orientation_cost if cfg.head_track_orientation else 0.0
    ik = WholeBodyIK(
        mjcf_path,
        body_joint_names=IK_KEEP_JOINTS,   # reduced-model names (base = composite)
        dof_names=BODY_JOINTS,             # per-DOF command order (base = x/y/yaw)
        head_frame_name=HEAD_FRAME_BODY,
        left_tool_name=TOOL_BODY["left"],
        right_tool_name=TOOL_BODY["right"],
        home=body_home,
        arm_position_cost=cfg.ik.arm_position_cost,
        arm_orientation_cost=arm_ori,
        head_position_cost=cfg.ik.head_position_cost,
        head_orientation_cost=head_ori,
        posture_cost=cfg.ik.posture_cost,
        lm_damping=cfg.ik.lm_damping,
        damping_cost=cfg.ik.damping_costs(),   # per-DOF: lean high, base mid, upper low
        low_accel_cost=cfg.ik.low_accel_cost,
        com_cost=cfg.ik.com_cost,
        com_cost_vertical=cfg.ik.com_cost_vertical,
        com_lm_damping=cfg.ik.com_lm_damping,
        base_frame_name=CHASSIS_BASE_FRAME,
        trunk_upright_cost=cfg.ik.trunk_upright_cost,
        trunk_lean_joint_names=TORSO_LEAN_JOINTS,
        chest_over_ankle_cost=cfg.ik.chest_over_ankle_cost,
        chest_head_frame_name=CHEST_HEAD_FRAME,
        chest_hip_frame_name=CHEST_HIP_FRAME,
        chest_ankle_frame_name=CHEST_ANKLE_FRAME,
        neural_posture_cost=cfg.ik.neural_posture_cost,
        neural_waist_joint_name=NEURAL_WAIST_JOINT,
        neural_pitch_joint_names=[NEURAL_PITCH_JOINT],  # hip only (torso_joint_3)
        max_velocity=cfg.ik.max_velocity(),
        max_acceleration=cfg.ik.max_acceleration(),
        config_limit_gain=cfg.ik.config_limit_gain,
        enforce_limits=cfg.ik.enforce_limits,
        control_dt=cfg.ik.control_dt,
        solver=cfg.ik.solver,
    )

    _set_body_home(model, data, body_robot, body_home)

    # Input source: a recorded AVP clip (--replay-avp) or the live UDP stream.
    # FileAvpSource matches UpperBodySubscriber's poll() contract, so the loop
    # below is identical either way.
    if args.replay_avp:
        sub = FileAvpSource(args.replay_avp, loop=args.replay_loop)
        print(f"[SIM] Replaying AVP input '{args.replay_avp}' "
              f"({sub.n_frames} frames, dt={sub.nominal_dt*1e3:.1f} ms"
              f"{', looping' if args.replay_loop else ''}); no headset needed.")
    else:
        sub = UpperBodySubscriber(args.host, args.port,
                                  timeout_s=cfg.network.recv_timeout_s)

    # Optional recorders (independent): raw AVP input and/or the full retarget
    # trace. Both flush to disk on exit (see the end of main()).
    avp_rec = (AvpTrajectoryRecorder(1.0 / 60.0, note=" ".join(sys.argv[1:]))
               if args.record_avp else None)
    retarget_rec = None
    if args.record_retarget:
        retarget_rec = RetargetTrajectoryRecorder(
            argv=sys.argv[1:], model_path=mjcf_path, body_joints=BODY_JOINTS,
            nominal_dt=1.0 / 60.0, track_orientation=cfg.track_orientation,
            head_track_orientation=cfg.head_track_orientation,
            extra_meta={"position_scale": cfg.position_scale,
                        "head_position_scale": cfg.head_position_scale,
                        "alpha_translation": cfg.filter.alpha_translation,
                        "alpha_rotation": cfg.filter.alpha_rotation,
                        "egoposer": bool(cfg.egoposer.enabled),
                        "neural_posture_cost": cfg.ik.neural_posture_cost,
                        "replay_avp": args.replay_avp or None})
    if avp_rec is not None:
        print(f"[SIM] Recording raw AVP input -> avp_trajectory/{args.record_avp}"
              f" (press 'q' or close viewer to save).")
    if retarget_rec is not None:
        print(f"[SIM] Recording retarget trace -> "
              f"retargetting_trajectory/{args.record_retarget}.")

    # Raw-AVP input frames overlay: on by request, or automatically while
    # recording AVP (so you can watch the input you're capturing), unless
    # explicitly suppressed. Draws head/hands (align_R-transformed to robot
    # world) + the world origin as RGB coordinate triads.
    show_input_frames = ((args.show_input_frames or avp_rec is not None)
                         and not args.no_input_frames)
    align_R_mat = np.asarray(cfg.align_R, dtype=np.float64)
    if show_input_frames:
        print("[SIM] Input frames ON: world origin + raw AVP head/hand poses "
              "as RGB triads (yellow=head, purple=left, teal=right).")

    # EgoPoser neural trunk-posture prior (opt-in). Built only when enabled;
    # constructs lazily and reports itself unavailable (rather than crashing) if
    # torch or the weights are missing, in which case the prior stays off.
    estimator = None
    if cfg.egoposer.enabled:
        from avp_teleop_upper_body.egoposer import (
            EgoPoserEstimator, _SMPL_PARENTS, _SMPL_MAPPED_JOINTS,
        )
        estimator = EgoPoserEstimator(
            weights_path=cfg.egoposer.weights_path or None,
            window_size=cfg.egoposer.window_size,
            spatial_normalization=cfg.egoposer.spatial_normalization,
            pitch_gain=cfg.egoposer.pitch_gain,
            yaw_gain=cfg.egoposer.yaw_gain,
            max_pitch=cfg.egoposer.max_pitch,
            max_yaw=cfg.egoposer.max_yaw,
            align_R=cfg.align_R,
            device=cfg.egoposer.device,
        )
        if estimator.available():
            print(f"[SIM] EgoPoser prior: ON (neural_posture_cost="
                  f"{cfg.ik.neural_posture_cost:g}, weights="
                  f"{cfg.egoposer.weights_path or '<none>'}).")
        else:
            print(f"[SIM] EgoPoser prior: requested but UNAVAILABLE "
                  f"({estimator.reason}); running without it.")
            estimator = None
    # EMA state for the (pitch, yaw) prior across ticks.
    neural_prior = {"pitch": 0.0, "yaw": 0.0, "init": False}
    # --visualize-prior: draw the hallucinated full SMPL body in the viewer,
    # anchored at the robot hip (torso_joint_3 ~= human pelvis).
    visualize_prior = bool(args.visualize_prior) and estimator is not None
    prior_skeleton = {"pts": None}     # latest (22, 3) pelvis-local body
    if visualize_prior:
        print(f"[SIM] EgoPoser prior visualisation: ON (full SMPL body; mapped "
              f"spine DARK, unused limbs LIGHT; anchored upright at hip "
              f"{CHEST_HIP_FRAME}).")

    # The three end-effector frames and how to read the robot's current pose.
    end_frames = {
        "head": HEAD_FRAME_BODY,
        "left": TOOL_BODY["left"],
        "right": TOOL_BODY["right"],
    }
    # IK command-target markers (on by default): draw each end's commanded target
    # (incremental, scaled, robot-world) as a sphere + orientation triad, with a
    # line to the actual tool frame showing tracking error. viz_rot holds a
    # viz-only orientation recomputed with track=True so the triad shows the live
    # AVP command orientation even when the arm IK tracks position only.
    show_targets = not args.no_target_markers
    viz_rot: Dict[str, Optional[np.ndarray]] = {"head": None, "left": None, "right": None}
    if show_targets:
        print("[SIM] Target markers ON: yellow=head, purple=left, teal=right "
              "(sphere+RGB triad); thin line to the tool = tracking error. "
              "--no-target-markers to hide.")
    # --- control state mutated by the key callback ---
    state = {"paused": False, "needs_calib": True}
    calib: Dict[str, Optional[WristCalibration]] = {"head": None, "left": None, "right": None}
    targets: Dict[str, Optional[Target]] = {"head": None, "left": None, "right": None}
    # One smoothing filter per end-effector target (head / left / right). Reset
    # on (re)calibration so we never smooth across the re-anchor discontinuity.
    filters: Dict[str, PoseFilter] = {
        end: PoseFilter(cfg.filter.alpha_translation, cfg.filter.alpha_rotation)
        for end in end_frames
    }
    q_body = body_home.copy()

    def key_callback(keycode: int) -> None:
        if keycode == 67:      # 'c'
            state["needs_calib"] = True
            print("[SIM] Recalibration requested (head + both hands).")
        elif keycode == 32:    # space
            state["paused"] = not state["paused"]
            print(f"[SIM] {'Paused' if state['paused'] else 'Resumed'}.")

    print(f"[SIM] Subscribing udp://{args.host}:{args.port} | "
          f"arms orientation={'on' if cfg.track_orientation else 'off'}, "
          f"head orientation={'on' if cfg.head_track_orientation else 'off'}")
    print(f"[SIM] Pose filter: alpha_t={cfg.filter.alpha_translation:.2f}, "
          f"alpha_R={cfg.filter.alpha_rotation:.2f} "
          f"(1.0=off; smaller=smoother)")
    if cfg.ik.enforce_limits:
        print(f"[SIM] QP rate limits: v_max base_lin={cfg.ik.chassis_max_linear_velocity:.1f} m/s, "
              f"base_yaw={cfg.ik.chassis_max_yaw_velocity:.1f}, "
              f"torso/neck={cfg.ik.torso_neck_max_velocity:.1f}, "
              f"arm={cfg.ik.arm_max_velocity:.1f} rad/s")
    else:
        print("[SIM] QP rate limits: DISABLED (enforce_limits=False)")
    smooth = [n for n, c in (("damping", cfg.ik.damping_cost),
                             ("low-accel", cfg.ik.low_accel_cost)) if c > 0]
    print(f"[SIM] Soft smoothing tasks: "
          f"{', '.join(smooth) if smooth else 'none'} "
          f"(damping_cost={cfg.ik.damping_cost:g}, low_accel_cost={cfg.ik.low_accel_cost:g})")
    print(f"[SIM] Whole-body priority: arm first, then torso/neck/lean, then base "
          f"(damping: base={cfg.ik.damping_cost_chassis:g} > "
          f"torso/neck/lean={cfg.ik.damping_cost:g}/{cfg.ik.damping_cost_lean:g} > "
          f"arm={cfg.ik.damping_cost_arm:g}).")
    if cfg.ik.chest_over_ankle_cost > 0:
        print(f"[SIM] Balance: chest (head/hip midpoint) kept over the ankle "
              f"(chest_over_ankle_cost={cfg.ik.chest_over_ankle_cost:g}); "
              f"trunk-pitch sum softly biased to ~0 "
              f"(trunk_upright_cost={cfg.ik.trunk_upright_cost:g}).")
    elif cfg.ik.trunk_upright_cost > 0:
        print(f"[SIM] Balance: trunk kept vertical via lean-angle sum -> 0 "
              f"(trunk_upright_cost={cfg.ik.trunk_upright_cost:g}); "
              f"the two remaining spine DOFs stay free to squat.")
    if cfg.ik.com_cost > 0:
        print(f"[SIM] Balance (legacy): CoM kept over the base (com_cost={cfg.ik.com_cost:g} "
              f"on horizontal, vertical={cfg.ik.com_cost_vertical:g} -> squat free).")
    print("[SIM] Waiting for AVP frames... (press 'c' to calibrate, space to pause)")

    n_steps_per_frame = max(1, int(round((1.0 / 60.0) / model.opt.timestep)))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Shadows/reflections/textures are already disabled at the model level in
        # _apply_simple_render (light_castshadow, mat_reflectance, mat_texid); the
        # model has no skybox texture, so nothing more is needed here. (The passive
        # viewer Handle does not expose the internal render scene's mjRND_* flags.)
        last_status = time.time()
        frames_seen = 0
        while viewer.is_running():
            hands, head = sub.poll()
            # Stop at the end of a (non-looping) replayed clip.
            if isinstance(sub, FileAvpSource) and sub.done:
                print("\n[SIM] Replay finished.")
                break
            # Record the raw AVP input verbatim (before any retargeting), so the
            # clip replays with identical timing / dropouts.
            if avp_rec is not None:
                avp_rec.record(hands, head)
            # Per-end live source pose (4x4) for this tick, if present.
            live: Dict[str, Optional[np.ndarray]] = {
                "head": head.head if (head is not None and head.valid) else None,
                "left": hands["left"].wrist if "left" in hands and hands["left"].valid else None,
                "right": hands["right"].wrist if "right" in hands and hands["right"].valid else None,
            }
            if any(v is not None for v in live.values()):
                frames_seen += 1

            # (Re)calibrate: anchor each available end to the robot's current pose.
            if state["needs_calib"]:
                ik.reset()                           # ramp from rest after re-anchor
                if estimator is not None:
                    estimator.reset()                # clear the EgoPoser window
                    neural_prior["init"] = False
                for end, body in end_frames.items():
                    R, p = _body_pose(model, data, body)
                    targets[end] = (p.copy(), None)  # hold current pose by default
                    filters[end].reset()             # don't smooth across re-anchor
                    if live[end] is not None:
                        calib[end] = WristCalibration.capture(live[end], _T(R, p))
                got = [e for e in end_frames if calib[e] is not None]
                q_body = body_robot.get_arm_qpos()
                state["needs_calib"] = False
                print(f"[SIM] Calibrated: {', '.join(got) if got else '(none yet)'}.")

            # Update targets from fresh frames (per-end scale / orientation),
            # smoothed by a per-end EMA(translation) + SLERP(rotation) filter.
            for end in end_frames:
                if live[end] is None or calib[end] is None:
                    continue
                if end == "head":
                    scale, track = cfg.head_position_scale, cfg.head_track_orientation
                else:
                    scale, track = cfg.position_scale, cfg.track_orientation
                tR, tp = wrist_to_tool_target(live[end], calib[end], scale, track,
                                              cfg.align_R)
                tp, tR = filters[end].filter(tp, tR if track else None)
                targets[end] = (tp, tR)
                # Visualisation-only orientation: recompute with track=True so the
                # marker triad shows the live AVP-derived command orientation even
                # when the arm IK tracks position only (tR above would be None).
                if show_targets:
                    vR, _ = wrist_to_tool_target(live[end], calib[end], scale,
                                                 True, cfg.align_R)
                    viz_rot[end] = vR

            calibrated = any(c is not None for c in calib.values())

            # EgoPoser prior: when all three live source poses are present, run
            # inference on the RAW AVP poses (not the retargeted robot targets),
            # EMA-smooth the (pitch, yaw) result, and hand it to the QP as a
            # low-cost trunk reference. Skipped silently if any pose is missing
            # or the estimator is unavailable -- the task then holds its last
            # target (default upright), so teleop is unaffected.
            if estimator is not None and all(live[e] is not None for e in end_frames):
                prior = estimator.predict(live["head"], live["left"], live["right"],
                                          with_skeleton=visualize_prior)
                if prior is not None:
                    a = cfg.egoposer.alpha
                    if not neural_prior["init"]:
                        neural_prior["pitch"], neural_prior["yaw"] = prior.pitch, prior.yaw
                        neural_prior["init"] = True
                    else:
                        neural_prior["pitch"] += a * (prior.pitch - neural_prior["pitch"])
                        neural_prior["yaw"] += a * (prior.yaw - neural_prior["yaw"])
                    ik.set_neural_target(neural_prior["pitch"], neural_prior["yaw"])
                    if visualize_prior and prior.skeleton is not None:
                        prior_skeleton["pts"] = prior.skeleton

            finger_cmd: Dict[str, Dict[str, float]] = {}
            if not state["paused"] and calibrated:
                q_body = ik.solve(q_body, targets["head"], targets["left"],
                                  targets["right"])
                body_robot.command_arm(q_body)
                for side in ("left", "right"):
                    if live[side] is not None:
                        ft = hand_retarget[side].joint_targets(
                            hands[side].keypoints, ranges)
                        body_robot.command_fingers(ft)
                        finger_cmd[side] = ft

                # Record the full retarget trace (after solve + fingers): raw
                # input, solved targets, joint angles, fingers, prior, skeleton.
                if retarget_rec is not None:
                    retarget_rec.record(
                        hands=hands, head=head, targets=targets, viz_rot=viz_rot,
                        q_body=q_body, fingers=finger_cmd or None,
                        neural=((neural_prior["pitch"], neural_prior["yaw"])
                                if estimator is not None else None),
                        skeleton=prior_skeleton["pts"] if visualize_prior else None)

            for _ in range(n_steps_per_frame):
                mujoco.mj_step(model, data)

            # --- viewer overlays (rebuilt each frame; one ngeom reset) --------
            # Both the target markers and the EgoPoser skeleton draw into
            # viewer.user_scn, so reset the geom count ONCE here, then append.
            scn = viewer.user_scn
            scn.ngeom = 0

            # IK command-target markers (default on): the incremental, scaled,
            # robot-world target the IK is solving toward, drawn as a coloured
            # sphere + RGB orientation triad, with a thin line to the robot's
            # ACTUAL tool frame = the live tracking error.
            if show_targets and calibrated:
                for end, body in end_frames.items():
                    tgt = targets[end]
                    if tgt is None:
                        continue
                    tp, _tr = tgt
                    _, actual_p = _body_pose(model, data, body)
                    _draw_target_marker(scn, tp, viz_rot[end],
                                        _TGT_MARKER_RGBA[end], actual_p=actual_p)

            # Raw AVP input frames: the world origin (long triad, no sphere) plus
            # each live head/hand 6-DoF pose rotated into robot world by align_R,
            # as a coloured sphere + RGB axis triad. Shows the actual input stream
            # (independent of calibration / retargeting) so a recording can be
            # eyeballed as it is captured.
            if show_input_frames:
                _draw_pose_frame(scn, np.eye(3), np.zeros(3),
                                 axis_len=0.3, axis_width=0.012)   # world origin
                for end in end_frames:
                    T = live[end]
                    if T is None:
                        continue
                    Rw = align_R_mat @ np.asarray(T)[:3, :3]
                    pw = align_R_mat @ np.asarray(T)[:3, 3]
                    _draw_pose_frame(scn, Rw, pw, sphere_rgba=_TGT_MARKER_RGBA[end])

            # Draw the EgoPoser prior wireframe. Anchored at the robot HIP
            # (torso_joint_3 ~= human pelvis) in an upright, heading-aligned frame
            # so the world-aligned SMPL body stands vertically and turns with the
            # robot. Mapped sagittal spine dark, rest light (see _draw_prior_skeleton).
            if visualize_prior and prior_skeleton["pts"] is not None:
                R_anchor, p_anchor = _prior_anchor_pose(
                    model, data, CHEST_HIP_FRAME, _WAIST_LINK_BODY)
                _draw_prior_skeleton(scn, R_anchor, p_anchor,
                                     prior_skeleton["pts"],
                                     _SMPL_PARENTS, _SMPL_MAPPED_JOINTS)
            viewer.sync()

            now = time.time()
            if now - last_status >= 2.0:
                rate = frames_seen / (now - last_status)
                if isinstance(sub, FileAvpSource):
                    i, n = sub.progress
                    print(f"[SIM] replaying | frame {i}/{n} | {rate:.0f}/s",
                          end="\r", flush=True)
                else:
                    tag = "tracking" if frames_seen else "NO DATA (is the publisher running?)"
                    print(f"[SIM] {tag} | frames {rate:.0f}/s", end="\r", flush=True)
                frames_seen, last_status = 0, now

            time.sleep(max(0.0, 1.0 / 60.0 - 0.001))

    sub.close()
    # Flush any recorders (viewer closed or replay finished). --record-trim
    # drops the first/last N seconds (default 2) to cut the start/stop fumble.
    trim = max(0.0, args.record_trim)
    if avp_rec is not None and len(avp_rec):
        raw_n = len(avp_rec)
        path = avp_rec.save(args.record_avp, trim_seconds=trim)
        kept = len(load_avp_trajectory(path)["frames"])
        print(f"[SIM] Saved raw AVP input ({kept}/{raw_n} frames kept, "
              f"trim={trim:g}s/end) -> {path}")
    if retarget_rec is not None and len(retarget_rec):
        raw_n = len(retarget_rec)
        path = retarget_rec.save(args.record_retarget, trim_seconds=trim)
        kept = load_retarget_trajectory(path)["metadata"]["n_frames"]
        print(f"[SIM] Saved retarget trace ({kept}/{raw_n} frames kept, "
              f"trim={trim:g}s/end) -> {path}")
    print("\n[SIM] Stopped.")


if __name__ == "__main__":
    main()
