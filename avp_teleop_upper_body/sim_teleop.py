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
    CHASSIS_IK_JOINT,
    TORSO_LEAN_JOINTS,
    NEURAL_WAIST_JOINT,
    NEURAL_PITCH_JOINT,
    CHEST_HEAD_FRAME,
    CHEST_HIP_FRAME,
    CHEST_ANKLE_FRAME,
    WAIST_LINK_BODY,
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


def _head_world(head_T: np.ndarray, align_R: np.ndarray):
    """Map a raw AVP head 4x4 into robot world: (position(3), yaw(rad)).

    Position is ``align_R @ p`` (same map wrist_to_tool_target uses for the
    hands). NOTE: the yaw returned here is from the head's local +Z axis, which
    is actually near-VERTICAL on the AVP headset (mean|z|~0.98), so it DEGENERATES
    when looking down -- do NOT use it for a heading. It is only kept for the
    (default-off) Phase-2 base translation follow, which does not use the yaw.
    For a pitch-robust turn heading use :func:`_head_interaural_yaw`."""
    p = align_R @ np.asarray(head_T)[:3, 3]
    fwd = align_R @ np.asarray(head_T)[:3, 2]      # head local +Z -> world
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    return p, yaw


def _head_interaural_yaw(head_T: np.ndarray, align_R: np.ndarray) -> float:
    """Pitch-robust turn heading (rad) from the head's INTERAURAL (left-right)
    axis. That axis (head local +X) stays horizontal even when the operator
    looks down -- measured mean|z|=0.03 / min ground-projection 0.99 across the
    recorded clips incl. the deep squat -- so its ground-plane ``atan2`` never
    degenerates, unlike the near-vertical gaze axis (which broke Phase 2). Used
    to drive the waist yaw (Phase 2b): track the head's turn since calibration."""
    ax = align_R @ np.asarray(head_T)[:3, 0]       # head local +X -> world
    return float(np.arctan2(ax[1], ax[0]))


def _gear_tag(name, frozen, scheduling, damp, static, floor):
    """Format one mode-switch axis as a P/N/D gear tag for the status line.

    P = frozen (velocityLimit 0). N = unfrozen at static damping (scheduling off,
    or on but signal below the D threshold). D = scheduling active and damping
    ramped below static -> show the live damping + a release percentage
    (0% = just left N / static, 100% = at the floor). ``damp`` is the live cost,
    ``static``/``floor`` the ramp endpoints. When ``scheduling`` is False the axis
    tops out at N (no D gear)."""
    if frozen:
        return f"{name} P"
    if not scheduling or damp >= static - 1e-6:
        return f"{name} N {static:g}"
    span = static - floor
    pct = 0.0 if span <= 1e-9 else 100.0 * (static - damp) / span
    return f"{name} D {damp:.1f}[{pct:.0f}%]"


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
    parser.add_argument("--max-target-jump", type=float,
                        default=cfg.filter.max_translation_jump,
                        help="Outlier rejection: max per-tick jump (m) of the raw "
                             "translation target before it is clamped, ahead of "
                             "the EMA. Rejects corrupt/post-dropout AVP frames "
                             "that would twitch the robot. 0 = disable "
                             "(default %(default)s ~= 4.8 m/s).")
    parser.add_argument("--no-base-deadzone", action="store_true",
                        help="Disable the mobile-base dead-zone (which freezes "
                             "the base while the head is not translating, to stop "
                             "in-place base spin / jitter). On by default.")
    parser.add_argument("--base-freeze-speed", type=float,
                        default=cfg.ik.base_freeze_speed,
                        help="Head horizontal speed (m/s) BELOW which the base is "
                             "frozen (default %(default)s).")
    parser.add_argument("--base-unfreeze-speed", type=float,
                        default=cfg.ik.base_unfreeze_speed,
                        help="Head horizontal speed (m/s) ABOVE which the base is "
                             "released; must exceed --base-freeze-speed "
                             "(hysteresis; default %(default)s).")
    parser.add_argument("--lean-freeze-speed", type=float,
                        default=cfg.ik.lean_freeze_speed,
                        help="Head VERTICAL speed (m/s) BELOW which the trunk lean "
                             "spine (torso_1/2/3) is frozen -- stops knee jitter "
                             "when not squatting (default %(default)s).")
    parser.add_argument("--lean-unfreeze-speed", type=float,
                        default=cfg.ik.lean_unfreeze_speed,
                        help="Head VERTICAL speed (m/s) ABOVE which the lean spine "
                             "is released to track a squat/bend; must exceed "
                             "--lean-freeze-speed (default %(default)s).")
    parser.add_argument("--enable-yaw-scheduling", action="store_true",
                        help="Enable Phase-4 continuous damping scheduling: "
                             "chassis-yaw damping lowers as combined yaw rate "
                             "(hands-head + head) rises, making the base cheaper "
                             "to recruit for a turn instead of contorting arms/waist. "
                             "Active only when base unfrozen. Off by default "
                             "(experimental, pending data).")
    parser.add_argument("--yaw-schedule-floor", type=float,
                        default=cfg.ik.yaw_schedule_floor,
                        help="Chassis-yaw damping floor at high combined yaw rate "
                             "(default %(default)s).")
    parser.add_argument("--yaw-schedule-rate-low", type=float,
                        default=cfg.ik.yaw_schedule_rate_low,
                        help="Combined yaw rate (rad/s) at which static damping applies "
                             "(default %(default)s).")
    parser.add_argument("--yaw-schedule-rate-high", type=float,
                        default=cfg.ik.yaw_schedule_rate_high,
                        help="Combined yaw rate (rad/s) at which floor damping applies "
                             "(default %(default)s).")
    parser.add_argument("--yaw-schedule-alpha", type=float,
                        default=cfg.ik.yaw_schedule_alpha,
                        help="EMA alpha for smoothing the damping cost itself (lower "
                             "= slower, reduces shift shock; default %(default)s).")
    parser.add_argument("--base-yaw-freeze-rate", type=float,
                        default=cfg.ik.base_yaw_freeze_rate,
                        help="Combined yaw rate (rad/s) BELOW which the base yaw "
                             "freezes (Phase-4b, decoupled from xy; default "
                             "%(default)s). Only with --enable-yaw-scheduling.")
    parser.add_argument("--base-yaw-unfreeze-rate", type=float,
                        default=cfg.ik.base_yaw_unfreeze_rate,
                        help="Combined yaw rate (rad/s) ABOVE which the base yaw "
                             "unfreezes to turn in place; must exceed "
                             "--base-yaw-freeze-rate (default %(default)s).")
    parser.add_argument("--enable-trans-scheduling", action="store_true",
                        help="Enable Phase-4b TRANSLATION damping scheduling: base "
                             "xy damping eases as head horizontal speed rises "
                             "(gentle; base carries the arms more when walking "
                             "fast). Off by default.")
    parser.add_argument("--trans-schedule-floor", type=float,
                        default=cfg.ik.trans_schedule_floor,
                        help="Base xy damping floor at high walk speed (default "
                             "%(default)s; static is --damping-chassis-linear).")
    parser.add_argument("--trans-schedule-speed-low", type=float,
                        default=cfg.ik.trans_schedule_speed_low,
                        help="Head horiz speed (m/s) at which xy damping is still "
                             "static (N->D boundary; default %(default)s).")
    parser.add_argument("--trans-schedule-speed-high", type=float,
                        default=cfg.ik.trans_schedule_speed_high,
                        help="Head horiz speed (m/s) at which xy damping hits the "
                             "floor (default %(default)s).")
    parser.add_argument("--no-base-follow", action="store_true",
                        help="Disable base-follows-head (Phase 2): the base then "
                             "only moves reactively via hand-reach error. On by "
                             "default (drives the base from head walk/turn).")
    parser.add_argument("--base-follow-scale", type=float,
                        default=cfg.ik.base_follow_scale,
                        help="Metres of base motion per metre of head motion "
                             "(1.0 = 1:1 walking; default %(default)s).")
    parser.add_argument("--base-track-cost", type=float,
                        default=cfg.ik.base_track_cost,
                        help="Weight of the base-tracking task (0 disables; "
                             "default %(default)s).")
    parser.add_argument("--waist-yaw-follow", action="store_true",
                        help="Phase-2b: turn the WAIST (torso_joint_4) with the "
                             "operator's head interaural yaw, so an in-place twist "
                             "rotates the waist instead of twisting the arms. "
                             "Ignored when --egoposer is on (the prior owns the "
                             "waist). Off by default.")
    parser.add_argument("--waist-yaw-follow-cost", type=float, default=None,
                        help="Weight of the waist-yaw-follow task (overrides the "
                             "default when --waist-yaw-follow is set).")
    parser.add_argument("--turn-follow-scale", type=float,
                        default=cfg.ik.turn_follow_scale,
                        help="Head-yaw -> waist-yaw gain (<1 attenuates; head yaw "
                             "over-estimates torso twist; default %(default)s).")
    parser.add_argument("--hand-front", action="store_true",
                        help="Phase-3: soft guard nudging a hand forward once it "
                             "goes behind the pelvis, to stop the arms crossing / "
                             "twisting behind the back on walk/turn. Inert while "
                             "hands are in front. Off by default.")
    parser.add_argument("--hand-front-cost", type=float, default=None,
                        help="Weight of the hand-in-front guard (overrides the "
                             "default when --hand-front is set).")
    parser.add_argument("--hand-front-margin", type=float,
                        default=cfg.ik.hand_front_margin,
                        help="Guard plane offset (m) behind the hip; negative = "
                             "behind (default %(default)s, so the natural rest "
                             "pose is not penalised).")
    # --- Per-DOF damping (whole-body movement-priority) overrides ------------ #
    # A higher velocity cost makes the QP move that joint LESS / LATER, so these
    # set which DOF absorbs a hand motion in the REACTIVE solve (they re-price
    # existing DOFs, they do NOT add competing target tasks). Cheapest moves
    # first: arm < waist < neck/lean < chassis. Lower a tier to recruit it sooner
    # (e.g. --damping-waist 0.2 turns the waist instead of contorting the arms on
    # an in-place twist; --damping-chassis 12 makes the base carry the arms sooner
    # on a walk); raise it to keep that joint planted.
    parser.add_argument("--damping-arm", type=float,
                        default=cfg.ik.damping_cost_arm,
                        help="Velocity cost on the 14 arm DOFs (cheapest tier; "
                             "default %(default)s). Lower = more arm-only reach.")
    parser.add_argument("--damping-waist", type=float,
                        default=cfg.ik.damping_cost_waist,
                        help="Velocity cost on the waist yaw (torso_joint_4; "
                             "default %(default)s). Lower to turn the WAIST rather "
                             "than the arms on an in-place twist (json7/8).")
    parser.add_argument("--damping-neck", type=float,
                        default=cfg.ik.damping_cost,
                        help="Velocity cost on the 2 neck DOFs (default "
                             "%(default)s).")
    parser.add_argument("--damping-lean", type=float,
                        default=cfg.ik.damping_cost_lean,
                        help="Velocity cost on the sagittal lean spine "
                             "(torso_joint_1/2/3; default %(default)s). Balance is "
                             "enforced by the balance tasks, not this.")
    parser.add_argument("--damping-chassis-linear", type=float,
                        default=cfg.ik.damping_cost_chassis_linear,
                        help="Velocity cost on the base TRANSLATION x, y (per m/s; "
                             "default %(default)s). Lower for a more eager base "
                             "that walks to carry the arms (json9-11); raise to "
                             "keep it planted for in-reach targets.")
    parser.add_argument("--damping-chassis-yaw", type=float,
                        default=cfg.ik.damping_cost_chassis_yaw,
                        help="Velocity cost on the base YAW / turn (per rad/s; "
                             "default %(default)s). Lower to let the base TURN to "
                             "face a target instead of contorting the arms/waist.")
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
    cfg.filter.max_translation_jump = args.max_target_jump
    if args.no_filter:
        cfg.filter.alpha_translation = 1.0
        cfg.filter.alpha_rotation = 1.0
    # Per-DOF damping overrides (whole-body movement priority). Defaults come from
    # the config, so these are no-ops unless the user passes a value.
    cfg.ik.damping_cost_arm = args.damping_arm
    cfg.ik.damping_cost_waist = args.damping_waist
    cfg.ik.damping_cost = args.damping_neck
    cfg.ik.damping_cost_lean = args.damping_lean
    cfg.ik.damping_cost_chassis_linear = args.damping_chassis_linear
    cfg.ik.damping_cost_chassis_yaw = args.damping_chassis_yaw
    if args.no_base_deadzone:
        cfg.ik.base_deadzone = False
    cfg.ik.base_freeze_speed = args.base_freeze_speed
    cfg.ik.base_unfreeze_speed = args.base_unfreeze_speed
    cfg.ik.lean_freeze_speed = args.lean_freeze_speed
    cfg.ik.lean_unfreeze_speed = args.lean_unfreeze_speed
    cfg.ik.enable_yaw_scheduling = args.enable_yaw_scheduling
    cfg.ik.yaw_schedule_floor = args.yaw_schedule_floor
    cfg.ik.yaw_schedule_rate_low = args.yaw_schedule_rate_low
    cfg.ik.yaw_schedule_rate_high = args.yaw_schedule_rate_high
    cfg.ik.yaw_schedule_alpha = args.yaw_schedule_alpha
    cfg.ik.base_yaw_freeze_rate = args.base_yaw_freeze_rate
    cfg.ik.base_yaw_unfreeze_rate = args.base_yaw_unfreeze_rate
    cfg.ik.enable_trans_scheduling = args.enable_trans_scheduling
    cfg.ik.trans_schedule_floor = args.trans_schedule_floor
    cfg.ik.trans_schedule_speed_low = args.trans_schedule_speed_low
    cfg.ik.trans_schedule_speed_high = args.trans_schedule_speed_high
    cfg.ik.base_follow_scale = args.base_follow_scale
    cfg.ik.base_track_cost = 0.0 if args.no_base_follow else args.base_track_cost
    cfg.ik.turn_follow_scale = args.turn_follow_scale
    # Waist-yaw follow: on when --waist-yaw-follow (or an explicit cost) is given.
    # Give it a sensible default cost when enabled but none specified.
    if args.waist_yaw_follow_cost is not None:
        cfg.ik.waist_yaw_follow_cost = args.waist_yaw_follow_cost
    elif args.waist_yaw_follow:
        if cfg.ik.waist_yaw_follow_cost <= 0.0:
            cfg.ik.waist_yaw_follow_cost = 2.0
    else:
        cfg.ik.waist_yaw_follow_cost = 0.0
    # Phase-3 hand-in-front guard: on when --hand-front (or explicit cost) given.
    cfg.ik.hand_front_margin = args.hand_front_margin
    if args.hand_front_cost is not None:
        cfg.ik.hand_front_cost = args.hand_front_cost
    elif args.hand_front:
        if cfg.ik.hand_front_cost <= 0.0:
            cfg.ik.hand_front_cost = 5.0
    else:
        cfg.ik.hand_front_cost = 0.0
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
        base_joint_name=CHASSIS_IK_JOINT,   # enables the base dead-zone freeze
        base_track_cost=cfg.ik.base_track_cost,   # Phase-2 base-follows-head
        waist_yaw_follow_cost=cfg.ik.waist_yaw_follow_cost,  # Phase-2b turn follow
        waist_joint_name=NEURAL_WAIST_JOINT,      # torso_joint_4
        hand_front_cost=cfg.ik.hand_front_cost,   # Phase-3 hand-in-front guard
        hand_front_margin=cfg.ik.hand_front_margin,
        hand_front_pelvis_frame_name=CHEST_HIP_FRAME,
        hand_front_waist_frame_name=WAIST_LINK_BODY,
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
    recording = avp_rec is not None or retarget_rec is not None
    if avp_rec is not None:
        print(f"[SIM] Recording raw AVP input -> avp_trajectory/{args.record_avp}.json")
    if retarget_rec is not None:
        print(f"[SIM] Recording retarget trace -> "
              f"retargetting_trajectory/{args.record_retarget}.json")
    if recording:
        trim = max(0.0, args.record_trim)
        print("[SIM] ==================== RECORDING GUIDE ====================")
        print("[SIM]  START : recording begins AUTOMATICALLY now (no key). Every")
        print("[SIM]          tick is captured, including invalid/no-data ticks.")
        print("[SIM]  STOP  : close the viewer window, OR press Ctrl+C in this")
        print("[SIM]          terminal -- BOTH stop AND SAVE the clip. (A second")
        print("[SIM]          Ctrl+C force-quits without saving.)")
        print(f"[SIM]  TRIM  : the first & last {trim:g}s are dropped on save "
              f"(--record-trim,")
        print("[SIM]          set 0 to keep everything) to cut the start/stop fumble.")
        print("[SIM]  TIP   : press 'c' to (re)calibrate, space to pause TELEOP")
        print("[SIM]          (recording of raw AVP input continues while paused).")
        print("[SIM] =========================================================")

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
    state = {"paused": False, "needs_calib": True, "stop": False}
    calib: Dict[str, Optional[WristCalibration]] = {"head": None, "left": None, "right": None}
    targets: Dict[str, Optional[Target]] = {"head": None, "left": None, "right": None}
    # One smoothing filter per end-effector target (head / left / right). Reset
    # on (re)calibration so we never smooth across the re-anchor discontinuity.
    filters: Dict[str, PoseFilter] = {
        end: PoseFilter(cfg.filter.alpha_translation, cfg.filter.alpha_rotation,
                        max_translation_jump=cfg.filter.max_translation_jump)
        for end in end_frames
    }
    q_body = body_home.copy()

    # Base dead-zone + follow state.
    #   speed/prev_xy      : EMA head horizontal speed for the freeze hysteresis.
    #   head0_xy/head0_z   : head position (robot world) at calibration -- anchor
    #                        for the walk displacement.
    #   base0_xy/base0_yaw : chassis pose at calibration -- the base reference the
    #                        head displacement/yaw is added to.
    #   prev_hyaw/yaw_accum: previous head yaw + unwrapped cumulative yaw (so a
    #                        360 deg turn does not wrap at +/-pi).
    # Start FROZEN (safest at rest); reset on (re)calibration.
    base_dz = {"prev_xy": None, "speed": 0.0,
               "head0_xy": None, "head0_z": None, "ref_head_xy": None,
               "base0_xy": None, "base0_yaw": 0.0,
               "prev_hyaw": None, "yaw_accum": 0.0,
               # Phase-2b waist-yaw follow: interaural heading tracking.
               "iy0": None, "iy_prev": None, "iy_accum": 0.0,
               # Phase-1b lean dead-zone: head vertical speed for the freeze
               # hysteresis (prev_z = last raw head world-Z, vspeed = EMA).
               "prev_z": None, "vspeed": 0.0,
               # Phase-4b translation scheduling: xy damping EMA.
               "trans_damp_ema": cfg.ik.damping_cost_chassis_linear,
               # Phase-4 yaw scheduling: combined yaw rate (hands-head + head) + damping.
               "yaw_sched": {"head_yaw_prev": None, "hands_yaw_prev": None,
                             "rate_ema": 0.0, "damp_ema": cfg.ik.damping_cost_chassis_yaw}}
    if cfg.ik.base_deadzone:
        ik.set_base_frozen(True)
    # Lean-spine dead-zone (Phase 1b) also starts FROZEN (stationary at calib).
    ik.set_lean_frozen(True)

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
          f"(1.0=off; smaller=smoother); "
          f"outlier clamp={cfg.filter.max_translation_jump:g} m/tick"
          f"{' (off)' if cfg.filter.max_translation_jump <= 0 else ''}")
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
    print(f"[SIM] Whole-body priority (damping cost, higher = moves later): "
          f"arm={cfg.ik.damping_cost_arm:g} < waist={cfg.ik.damping_cost_waist:g} "
          f"< neck={cfg.ik.damping_cost:g} / lean={cfg.ik.damping_cost_lean:g} "
          f"< base[xy={cfg.ik.damping_cost_chassis_linear:g}, "
          f"yaw={cfg.ik.damping_cost_chassis_yaw:g}].")
    if cfg.ik.base_deadzone:
        print(f"[SIM] Base dead-zone: ON (freeze base below head speed "
              f"{cfg.ik.base_freeze_speed:g} m/s, release above "
              f"{cfg.ik.base_unfreeze_speed:g} m/s). --no-base-deadzone to disable.")
    else:
        print("[SIM] Base dead-zone: OFF (base free to move at any head speed).")
    print(f"[SIM] Lean-spine dead-zone: freeze torso_1/2/3 below head vertical "
          f"speed {cfg.ik.lean_freeze_speed:g} m/s, release above "
          f"{cfg.ik.lean_unfreeze_speed:g} m/s (squat detection).")
    if cfg.ik.enable_yaw_scheduling:
        print(f"[SIM] Phase-4 yaw scheduling: ON (chassis-yaw damping "
              f"{cfg.ik.damping_cost_chassis_yaw:g} -> {cfg.ik.yaw_schedule_floor:g} "
              f"as combined yaw rate (hands-head + head) {cfg.ik.yaw_schedule_rate_low:g} -> "
              f"{cfg.ik.yaw_schedule_rate_high:g} rad/s; active only when base unfrozen).")
    if cfg.ik.enable_trans_scheduling:
        print(f"[SIM] Phase-4b trans scheduling: ON (chassis-xy damping "
              f"{cfg.ik.damping_cost_chassis_linear:g} -> {cfg.ik.trans_schedule_floor:g} "
              f"as head horiz speed {cfg.ik.trans_schedule_speed_low:g} -> "
              f"{cfg.ik.trans_schedule_speed_high:g} m/s; active only when base xy unfrozen).")
    if cfg.ik.base_track_cost > 0:
        print(f"[SIM] Base follows head: ON (scale={cfg.ik.base_follow_scale:g}, "
              f"track cost={cfg.ik.base_track_cost:g}, lean-drop gate="
              f"{cfg.ik.base_lean_drop:g} m). --no-base-follow to disable.")
    else:
        print("[SIM] Base follows head: OFF (base moves only reactively).")
    if cfg.ik.waist_yaw_follow_cost > 0 and cfg.ik.neural_posture_cost <= 0:
        print(f"[SIM] Waist yaw follows head: ON (interaural-axis turn, "
              f"scale={cfg.ik.turn_follow_scale:g}, soft limit "
              f"±{cfg.ik.waist_soft_limit:g} rad, cost={cfg.ik.waist_yaw_follow_cost:g}).")
    elif cfg.ik.waist_yaw_follow_cost > 0:
        print("[SIM] Waist yaw follows head: OFF (EgoPoser owns the waist yaw).")
    if cfg.ik.hand_front_cost > 0:
        print(f"[SIM] Hand-in-front guard: ON (cost={cfg.ik.hand_front_cost:g}, "
              f"plane {cfg.ik.hand_front_margin:g} m behind the hip). "
              f"--hand-front to toggle.")
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

    # Ctrl+C sets a stop flag so the loop exits cleanly and the recorder-save
    # block below still runs (a raw KeyboardInterrupt would skip it and lose the
    # clip). A second Ctrl+C still hard-kills via the default handler.
    import signal
    _prev_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):
        state["stop"] = True
        signal.signal(signal.SIGINT, _prev_sigint)   # next Ctrl+C = hard kill
        print("\n[SIM] Ctrl+C: stopping and saving...")
    signal.signal(signal.SIGINT, _on_sigint)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Shadows/reflections/textures are already disabled at the model level in
        # _apply_simple_render (light_castshadow, mat_reflectance, mat_texid); the
        # model has no skybox texture, so nothing more is needed here. (The passive
        # viewer Handle does not expose the internal render scene's mjRND_* flags.)
        last_status = time.time()
        frames_seen = 0
        while viewer.is_running() and not state["stop"]:
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
                # Reset the base dead-zone: forget head-speed history and start
                # frozen (the operator is stationary at the calibration instant).
                base_dz["prev_xy"] = None
                base_dz["speed"] = 0.0
                base_dz["trans_damp_ema"] = cfg.ik.damping_cost_chassis_linear
                if cfg.ik.base_deadzone:
                    ik.set_base_frozen(True)
                # Reset the lean dead-zone (Phase 1b): forget the vertical-speed
                # history and start frozen (stationary at calibration).
                base_dz["prev_z"] = None
                base_dz["vspeed"] = 0.0
                ik.set_lean_frozen(True)
                # Reset the Phase-4 yaw scheduler: forget yaw-rate history and
                # reset damping EMA to the static value (so it doesn't carry stale
                # low values into the next unfreeze).
                base_dz["yaw_sched"]["head_yaw_prev"] = None
                base_dz["yaw_sched"]["hands_yaw_prev"] = None
                base_dz["yaw_sched"]["rate_ema"] = 0.0
                base_dz["yaw_sched"]["damp_ema"] = cfg.ik.damping_cost_chassis_yaw
                # Anchor the Phase-2 base-follow: head pose (robot world) + the
                # base pose at calibration are the references head displacement /
                # yaw are added to. yaw_accum unwraps the head heading over time.
                base_dz["head0_xy"] = None
                base_dz["base0_xy"] = q_body[:2].copy()
                base_dz["base0_yaw"] = float(q_body[2])
                base_dz["prev_hyaw"] = None
                base_dz["yaw_accum"] = 0.0
                base_dz["iy0"] = None
                base_dz["iy_prev"] = None
                base_dz["iy_accum"] = 0.0
                if live["head"] is not None:
                    hp, hyaw = _head_world(live["head"], align_R_mat)
                    base_dz["head0_xy"] = hp[:2].copy()
                    base_dz["head0_z"] = float(hp[2])
                    base_dz["ref_head_xy"] = hp[:2].copy()
                    base_dz["prev_hyaw"] = hyaw
                    iy = _head_interaural_yaw(live["head"], align_R_mat)
                    base_dz["iy0"] = iy
                    base_dz["iy_prev"] = iy
                if ik.base_task is not None:
                    ik.set_base_target(base_dz["base0_xy"][0], base_dz["base0_xy"][1],
                                       base_dz["base0_yaw"])
                if ik.waist_task is not None:
                    ik.set_waist_yaw_target(0.0)
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

            # --- Base dead-zone: freeze the mobile base while the operator's
            # head is not translating horizontally, release it once they walk.
            # Uses the RAW AVP head xy (horizontal only, so a pure squat/bend
            # does not release the base). EMA-smoothed speed + hysteresis
            # (freeze below base_freeze_speed, release above base_unfreeze_speed)
            # so it does not chatter. Head missing this tick -> hold last state.
            if cfg.ik.base_deadzone and live["head"] is not None:
                head_xy = np.asarray(live["head"])[:2, 3]
                if base_dz["prev_xy"] is not None:
                    inst = float(np.linalg.norm(head_xy - base_dz["prev_xy"])) / (1.0 / 60.0)
                    a = cfg.ik.base_speed_alpha
                    base_dz["speed"] += a * (inst - base_dz["speed"])
                base_dz["prev_xy"] = head_xy.copy()
                # Phase-4b: this gates the base TRANSLATION (xy) only. Base yaw
                # has its own gate below (combined yaw rate) when scheduling is on;
                # otherwise yaw mirrors xy for byte-identical pre-4b behaviour.
                if ik.base_xy_frozen and base_dz["speed"] > cfg.ik.base_unfreeze_speed:
                    ik.set_base_xy_frozen(False)
                elif not ik.base_xy_frozen and base_dz["speed"] < cfg.ik.base_freeze_speed:
                    ik.set_base_xy_frozen(True)

                # Phase-4b TRANSLATION scheduling (D gear): once unfrozen, ease
                # the xy damping from static (damping_cost_chassis_linear) toward
                # trans_schedule_floor as head horizontal speed rises, so the base
                # carries the arms more eagerly when walking fast. Gentle by
                # design. EMA-smoothed to avoid shift shock. Only when enabled and
                # xy unfrozen (else static value / frozen dominates).
                if cfg.ik.enable_trans_scheduling and not ik.base_xy_frozen:
                    target_xy_damp = float(np.interp(
                        base_dz["speed"],
                        [cfg.ik.trans_schedule_speed_low, cfg.ik.trans_schedule_speed_high],
                        [cfg.ik.damping_cost_chassis_linear, cfg.ik.trans_schedule_floor]))
                    base_dz["trans_damp_ema"] += cfg.ik.base_speed_alpha * (
                        target_xy_damp - base_dz["trans_damp_ema"])
                    ik.set_chassis_xy_damping(base_dz["trans_damp_ema"])
                elif cfg.ik.enable_trans_scheduling:
                    # xy frozen: reset the damping EMA to the static value.
                    base_dz["trans_damp_ema"] = cfg.ik.damping_cost_chassis_linear

            # --- Lean-spine dead-zone (Phase 1b): freeze the sagittal lean spine
            # (torso_joint_1/2/3) while the operator is NOT squatting/bending, so
            # the knees/ankles stop jittering during stationary fine manipulation
            # (json1-3). Gate on head VERTICAL speed (squat detection) -- raw AVP
            # head world-Z (align_R is a yaw, so Z is unchanged), EMA-smoothed,
            # with hysteresis (freeze below lean_freeze_speed, release above
            # lean_unfreeze_speed). Orthogonal to the base's horizontal gate.
            # Head missing this tick -> hold last state.
            if live["head"] is not None:
                head_z = float(np.asarray(live["head"])[2, 3])
                if base_dz["prev_z"] is not None:
                    inst = abs(head_z - base_dz["prev_z"]) / (1.0 / 60.0)
                    a = cfg.ik.base_speed_alpha
                    base_dz["vspeed"] += a * (inst - base_dz["vspeed"])
                base_dz["prev_z"] = head_z
                if ik.lean_frozen and base_dz["vspeed"] > cfg.ik.lean_unfreeze_speed:
                    ik.set_lean_frozen(False)
                elif not ik.lean_frozen and base_dz["vspeed"] < cfg.ik.lean_freeze_speed:
                    ik.set_lean_frozen(True)

            # --- Phase-4 continuous damping scheduling: chassis-yaw damping lowers
            # as COMBINED yaw rate rises (hands-head yaw rate + head yaw rate),
            # making the base cheaper to recruit for a turn (instead of contorting
            # arms/waist). The combined signal captures both "turn waist" (hands
            # sweep around body, head static as in clip7/8) and "turn body" (head +
            # hands both turn as in clip11). Active ONLY when the base is unfrozen
            # (the dead-zone handles stationary). Damping cost itself is EMA-smoothed
            # to avoid QP objective discontinuities ("shift shock" / 换挡冲击).
            if cfg.ik.enable_yaw_scheduling and live["head"] is not None:
                # 1. Head yaw rate (interaural, pitch-robust).
                head_yaw = _head_interaural_yaw(live["head"], align_R_mat)
                head_yaw_rate = 0.0
                if base_dz["yaw_sched"]["head_yaw_prev"] is not None:
                    d_head = (head_yaw - base_dz["yaw_sched"]["head_yaw_prev"] + np.pi) % (2*np.pi) - np.pi
                    head_yaw_rate = abs(d_head) / (1.0 / 60.0)
                base_dz["yaw_sched"]["head_yaw_prev"] = head_yaw

                # 2. Hands-head yaw rate: yaw angle of (hands_midpoint - head) vector.
                hands_yaw_rate = 0.0
                if live["left"] is not None and live["right"] is not None:
                    hands_mid_xy = 0.5 * (np.asarray(live["left"])[:2, 3] + np.asarray(live["right"])[:2, 3])
                    head_xy = np.asarray(live["head"])[:2, 3]
                    vec = hands_mid_xy - head_xy
                    dist = float(np.linalg.norm(vec))
                    if dist > 0.25:  # Hands far enough from head for stable yaw angle.
                        hands_yaw = float(np.arctan2(vec[1], vec[0]))
                        if base_dz["yaw_sched"]["hands_yaw_prev"] is not None:
                            d_hands = (hands_yaw - base_dz["yaw_sched"]["hands_yaw_prev"] + np.pi) % (2*np.pi) - np.pi
                            hands_yaw_rate = abs(d_hands) / (1.0 / 60.0)
                        base_dz["yaw_sched"]["hands_yaw_prev"] = hands_yaw
                    else:
                        base_dz["yaw_sched"]["hands_yaw_prev"] = None  # Too close, reset.

                # 3. Combined signal: total turning intent strength.
                combined_rate = head_yaw_rate + hands_yaw_rate
                a = cfg.ik.yaw_schedule_alpha
                base_dz["yaw_sched"]["rate_ema"] += a * (combined_rate - base_dz["yaw_sched"]["rate_ema"])
                r = base_dz["yaw_sched"]["rate_ema"]

                # 4a. Phase-4b independent base-YAW dead-zone: gate the base yaw
                # freeze on the combined yaw rate (hysteresis), DECOUPLED from the
                # xy dead-zone above. So an in-place turn (low translation, high
                # yaw rate) releases the base yaw even while xy stays frozen
                # (fixes clip11). Only when base_deadzone is on.
                if cfg.ik.base_deadzone:
                    if ik.base_yaw_frozen and r > cfg.ik.base_yaw_unfreeze_rate:
                        ik.set_base_yaw_frozen(False)
                    elif not ik.base_yaw_frozen and r < cfg.ik.base_yaw_freeze_rate:
                        ik.set_base_yaw_frozen(True)

                # 4b. Map combined rate -> damping (only when base yaw unfrozen).
                if not ik.base_yaw_frozen:
                    target_damp = float(np.interp(
                        r,
                        [cfg.ik.yaw_schedule_rate_low, cfg.ik.yaw_schedule_rate_high],
                        [cfg.ik.damping_cost_chassis_yaw, cfg.ik.yaw_schedule_floor]
                    ))
                    # Smooth the damping cost itself (EMA) to avoid QP jumps.
                    base_dz["yaw_sched"]["damp_ema"] += a * (target_damp - base_dz["yaw_sched"]["damp_ema"])
                    ik.set_chassis_yaw_damping(base_dz["yaw_sched"]["damp_ema"])
                else:
                    # Frozen: reset the damping EMA to the static value so it doesn't
                    # carry stale low values into the next unfreeze.
                    base_dz["yaw_sched"]["damp_ema"] = cfg.ik.damping_cost_chassis_yaw
            elif cfg.ik.base_deadzone:
                # Scheduling OFF: base yaw mirrors the xy freeze (byte-identical
                # to the pre-4b behaviour -- the whole base freezes/thaws together).
                if ik.base_yaw_frozen != ik.base_xy_frozen:
                    ik.set_base_yaw_frozen(ik.base_xy_frozen)

            # --- Phase 2: base follows head. Drive the chassis reference from
            # the operator's head horizontal displacement (walk) + head yaw
            # (turn) since calibration, both in robot world. The base task then
            # carries the arms along instead of the arms twisting to reach.
            # xy is LEAN-GATED (a forward bend drops the head -> hold xy, only a
            # near-level step advances it); yaw is unwrapped so a 360 deg turn
            # tracks continuously. While the base is frozen (head not walking)
            # the hard velocity limit pins it regardless, and an in-place torso
            # twist stays frozen -> handled by the waist/arms (json7/8).
            if ik.base_task is not None and live["head"] is not None:
                hp, hyaw = _head_world(live["head"], align_R_mat)
                # Lazy anchor if the head was absent at calibration.
                if base_dz["head0_xy"] is None:
                    base_dz["head0_xy"] = hp[:2].copy()
                    base_dz["head0_z"] = float(hp[2])
                    base_dz["ref_head_xy"] = hp[:2].copy()
                    base_dz["base0_xy"] = q_body[:2].copy()
                    base_dz["base0_yaw"] = float(q_body[2])
                    base_dz["prev_hyaw"] = hyaw
                # Advance the xy reference only when the head is near its
                # calibration height (a step, not a bend).
                if abs(hp[2] - base_dz["head0_z"]) <= cfg.ik.base_lean_drop:
                    base_dz["ref_head_xy"] = hp[:2].copy()
                # Unwrap head yaw into a continuous heading.
                if base_dz["prev_hyaw"] is not None:
                    d = (hyaw - base_dz["prev_hyaw"] + np.pi) % (2 * np.pi) - np.pi
                    base_dz["yaw_accum"] += d
                base_dz["prev_hyaw"] = hyaw
                s = cfg.ik.base_follow_scale
                dxy = s * (base_dz["ref_head_xy"] - base_dz["head0_xy"])
                tx = base_dz["base0_xy"][0] + dxy[0]
                ty = base_dz["base0_xy"][1] + dxy[1]
                tyaw = base_dz["base0_yaw"] + base_dz["yaw_accum"]
                ik.set_base_target(tx, ty, tyaw)

            # --- Phase 2b: waist yaw follows the operator's turn. Drive
            # torso_joint_4 toward the head INTERAURAL-axis yaw since calibration
            # (pitch-robust -- stable when looking down, see _head_interaural_yaw),
            # unwrapped and scaled, clamped to a soft limit below the joint's hard
            # +/-1.53 rad. So an in-place twist turns the waist instead of twisting
            # the arms. Only active when the waist task exists (waist_yaw_follow_cost
            # > 0 AND EgoPoser off -- EgoPoser owns the waist otherwise).
            if ik.waist_task is not None and live["head"] is not None:
                iy = _head_interaural_yaw(live["head"], align_R_mat)
                if base_dz["iy0"] is None:                 # lazy anchor
                    base_dz["iy0"] = iy
                    base_dz["iy_prev"] = iy
                # Unwrap into a continuous accumulated turn since calibration.
                d = (iy - base_dz["iy_prev"] + np.pi) % (2 * np.pi) - np.pi
                base_dz["iy_accum"] += d
                base_dz["iy_prev"] = iy
                tgt = cfg.ik.turn_follow_scale * base_dz["iy_accum"]
                tgt = float(np.clip(tgt, -cfg.ik.waist_soft_limit,
                                    cfg.ik.waist_soft_limit))
                ik.set_waist_yaw_target(tgt)

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
                base_tag = ""
                if cfg.ik.base_deadzone:
                    # Three orthogonal mode-switch axes as P/N/D gears.
                    trans = _gear_tag(
                        "base_trans", ik.base_xy_frozen,
                        cfg.ik.enable_trans_scheduling, ik.chassis_xy_damping,
                        cfg.ik.damping_cost_chassis_linear, cfg.ik.trans_schedule_floor)
                    yaw = _gear_tag(
                        "base_yaw", ik.base_yaw_frozen,
                        cfg.ik.enable_yaw_scheduling, ik.chassis_yaw_damping,
                        cfg.ik.damping_cost_chassis_yaw, cfg.ik.yaw_schedule_floor)
                    # body_pitch (lean spine) has no damping schedule -> P/N only.
                    pitch = f"body_pitch {'P' if ik.lean_frozen else 'N'}"
                    base_tag = f" | {trans} | {yaw} | {pitch}"
                if isinstance(sub, FileAvpSource):
                    i, n = sub.progress
                    print(f"[SIM] replaying | frame {i}/{n} | {rate:.0f}/s{base_tag}",
                          end="\r", flush=True)
                else:
                    tag = "tracking" if frames_seen else "NO DATA (is the publisher running?)"
                    print(f"[SIM] {tag} | frames {rate:.0f}/s{base_tag}", end="\r", flush=True)
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
