"""Standalone EgoPoser skeleton viewer -- NO robot, NO teleop, NO IK.

Run this to look at *only* the EgoPoser-predicted human skeleton, decoupled from
the robot entirely. It subscribes to the AVP head + hand stream, runs EgoPoser on
the RAW poses, and draws the full-body SMPL skeleton in a minimal MuJoCo scene
(just a floor + a world-frame XYZ arrow triad -- no robot model is loaded).

Two things are shown:
  * POSTURE comes from EgoPoser: the pelvis-local skeleton (spine bend, limb
    articulation) hallucinated from head + 2 hands. Mapped sagittal spine is drawn
    DARK, the rest (unused hallucinated limbs) LIGHT.
  * TRANSLATION comes from the raw AVP HEAD pose, not EgoPoser (EgoPoser outputs a
    pelvis-local skeleton with no global translation). On the first frame we latch
    the head world position as the origin; every frame after, the whole skeleton is
    translated by (current head - origin), so you can watch accumulated displacement
    against the fixed world-axis triad at the origin. Use --fixed to pin the pelvis
    at the origin instead (pure posture, no drift).

Run inside the ``AVP`` conda env (after starting the publisher in another shell):

    python -m avp_teleop_upper_body.egoposer.preview
    python -m avp_teleop_upper_body.egoposer.preview --print
    python -m avp_teleop_upper_body.egoposer.preview --fixed

No calibration is needed -- EgoPoser consumes the raw AVP world poses directly.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from avp_teleop_upper_body.config import default_config
from avp_teleop_upper_body.transport import UpperBodySubscriber
from avp_teleop_upper_body.egoposer import (
    EgoPoserEstimator, _SMPL_PARENTS, _SMPL_MAPPED_JOINTS,
)

# SMPL joint id of the head (its skeleton point is placed at the AVP head world
# position in head-follow mode). pelvis=0, head=15 in the 22-joint body.
_HEAD_JID = 15

# --- colours (RGBA) ------------------------------------------------------- #
# Skeleton: mapped sagittal spine DARK, unused hallucinated limbs LIGHT.
_BONE_MAPPED   = np.array([0.00, 0.55, 0.80, 1.00])   # deep cyan
_NODE_MAPPED   = np.array([0.95, 0.35, 0.00, 1.00])   # deep orange
_BONE_UNMAPPED = np.array([0.55, 0.85, 0.95, 0.55])   # pale cyan
_NODE_UNMAPPED = np.array([1.00, 0.78, 0.55, 0.55])   # pale orange
# World-frame axis arrows.
_AXIS_X_RGBA = np.array([0.90, 0.20, 0.20, 1.0])      # red   = +X
_AXIS_Y_RGBA = np.array([0.20, 0.80, 0.20, 1.0])      # green = +Y
_AXIS_Z_RGBA = np.array([0.25, 0.45, 0.95, 1.0])      # blue  = +Z
# Ground-truth AVP marker sphere colours (position), one per tracked frame. The
# orientation is drawn as an RGB triad (X=red, Y=green, Z=blue) at each marker.
_GT_HEAD_RGBA  = np.array([1.00, 0.90, 0.20, 1.0])    # yellow  = head
_GT_LEFT_RGBA  = np.array([0.60, 0.30, 0.95, 1.0])    # purple  = left hand
_GT_RIGHT_RGBA = np.array([0.20, 0.90, 0.85, 1.0])    # teal    = right hand

# A minimal scene: a checker floor and one light. No robot. The skeleton and the
# world axes are drawn as user-scene geoms on top of this.
_SCENE_XML = """
<mujoco model="egoposer_preview">
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.4 0.4 0.4"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.35"
             rgb2="0.22 0.22 0.27" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"/>
  </worldbody>
</mujoco>
"""


def _add_sphere(scn, pos, rgba, r):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([r, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(9), rgba.astype(np.float32))
    scn.ngeom += 1


def _add_capsule(scn, a, b, rgba, width):
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


def _draw_world_axes(scn, *, length=0.5, width=0.012, origin=(0.0, 0.0, 0.0)):
    """Draw a fixed world-frame XYZ triad at the origin (R=X, G=Y, B=Z).

    A static reference so accumulated skeleton translation is easy to read off.
    """
    o = np.asarray(origin, dtype=np.float64)
    for vec, rgba in ((np.array([length, 0, 0]), _AXIS_X_RGBA),
                      (np.array([0, length, 0]), _AXIS_Y_RGBA),
                      (np.array([0, 0, length]), _AXIS_Z_RGBA)):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, int(mujoco.mjtGeom.mjGEOM_ARROW), np.zeros(3),
            np.zeros(3), np.eye(3).reshape(9), rgba.astype(np.float32))
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), width,
                             o, o + vec)
        scn.ngeom += 1


def _draw_pose(scn, R, p, marker_rgba, *, sphere_r=0.02, axis_len=0.12,
               axis_width=0.008):
    """Draw a ground-truth pose: a coloured position sphere + an RGB orient triad.

    ``R`` (3,3) / ``p`` (3,) are in the viewer world frame. The sphere (in
    ``marker_rgba``) marks the position; three short arrows along the frame's own
    X/Y/Z columns (red/green/blue) show its orientation, so head/hand pose is
    directly comparable to the world axes and the skeleton.
    """
    _add_sphere(scn, p, marker_rgba, sphere_r)
    for col, rgba in ((0, _AXIS_X_RGBA), (1, _AXIS_Y_RGBA), (2, _AXIS_Z_RGBA)):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, int(mujoco.mjtGeom.mjGEOM_ARROW), np.zeros(3),
            np.zeros(3), np.eye(3).reshape(9), rgba.astype(np.float32))
        mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), axis_width,
                             p, p + axis_len * R[:, col])
        scn.ngeom += 1


def _draw_skeleton(scn, points, parents, mapped, *, width=0.006, node_r=0.012):
    """Draw the full SMPL body: mapped spine dark, unused limbs light."""
    for jid in range(len(points)):
        par = int(parents[jid])
        if par < 0:
            continue
        is_mapped = jid in mapped and par in mapped
        _add_capsule(scn, points[par], points[jid],
                     _BONE_MAPPED if is_mapped else _BONE_UNMAPPED, width)
    for jid, pt in enumerate(points):
        _add_sphere(scn, pt, _NODE_MAPPED if jid in mapped else _NODE_UNMAPPED,
                    node_r)


def main() -> None:
    cfg = default_config()
    parser = argparse.ArgumentParser(
        description="Standalone EgoPoser skeleton viewer (no robot, no teleop)."
    )
    parser.add_argument("--host", default=cfg.network.host)
    parser.add_argument("--port", type=int, default=cfg.network.port)
    parser.add_argument("--egoposer-weights", default=None,
                        help="Override the EgoPoser checkpoint path (defaults to "
                             "the bundled one in egoposer/model_zoo/).")
    parser.add_argument("--fixed", action="store_true",
                        help="Pin the pelvis at the world origin (pure posture, no "
                             "translation). Default follows the AVP head so you can "
                             "watch accumulated displacement against the axes.")
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="Print pitch/yaw + head displacement once per second.")
    parser.add_argument("--alpha", type=float, default=cfg.egoposer.alpha,
                        help="EMA smoothing of the drawn skeleton in (0,1]: "
                             "1=off, smaller=smoother (default %(default)s).")
    parser.add_argument("--axis-length", type=float, default=0.5,
                        help="World-axis arrow length in metres (default %(default)s).")
    parser.add_argument("--no-gt", action="store_true",
                        help="Hide the raw-AVP ground-truth head/hand pose markers "
                             "(shown by default: yellow=head, purple=left, teal=right, "
                             "each with an RGB orientation triad).")
    args = parser.parse_args()

    if args.egoposer_weights:
        cfg.egoposer.weights_path = args.egoposer_weights

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
    if not estimator.available():
        print(f"[EGOVIEW] EgoPoser UNAVAILABLE: {estimator.reason}")
        print("[EGOVIEW] Install torch and/or pass --egoposer-weights, then retry.")
        return
    print(f"[EGOVIEW] EgoPoser ready (weights="
          f"{cfg.egoposer.weights_path or '<none>'}, window={cfg.egoposer.window_size}).")

    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sub = UpperBodySubscriber(args.host, args.port,
                              timeout_s=cfg.network.recv_timeout_s)

    align_R = cfg.align_R
    alpha = float(np.clip(args.alpha, 1e-3, 1.0))
    skel_ema: Optional[np.ndarray] = None    # smoothed (22, 3) skeleton (pelvis-local)
    head0: Optional[np.ndarray] = None       # latched head world origin (viewer frame)
    disp = np.zeros(3)                        # current head displacement from origin
    gt_head = gt_left = gt_right = None       # latched raw AVP poses (4x4) for GT markers
    prior_pitch = prior_yaw = 0.0
    last_status = time.time()
    frames_seen = 0

    mode = "FIXED pelvis at origin" if args.fixed else "head-follow (shows displacement)"
    print(f"[EGOVIEW] Subscribing udp://{args.host}:{args.port} | mode: {mode}")
    print("[EGOVIEW] World axes: RED=+X, GREEN=+Y, BLUE=+Z at the origin.")
    if not args.no_gt:
        print("[EGOVIEW] GT markers (raw AVP): YELLOW=head, PURPLE=left, TEAL=right "
              "(sphere=position, RGB triad=orientation). --no-gt to hide.")
    print("[EGOVIEW] Waiting for AVP frames (need head+L+R)...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            hands, head = sub.poll()
            live_head = head.head if (head is not None and head.valid) else None
            live_l = hands["left"].wrist if "left" in hands and hands["left"].valid else None
            live_r = hands["right"].wrist if "right" in hands and hands["right"].valid else None

            if live_head is not None and live_l is not None and live_r is not None:
                frames_seen += 1
                prior = estimator.predict(live_head, live_l, live_r,
                                          with_skeleton=True)
                if prior is not None and prior.skeleton is not None:
                    if skel_ema is None:
                        skel_ema = prior.skeleton.copy()
                    else:
                        skel_ema += alpha * (prior.skeleton - skel_ema)
                    prior_pitch, prior_yaw = prior.pitch, prior.yaw

                # Latch the raw GT poses so the markers hold between datagrams
                # (live_* go None on ticks with no fresh frame).
                gt_head, gt_left, gt_right = live_head, live_l, live_r

                # Head world position in the viewer frame (align AVP world -> ours).
                head_w = align_R @ live_head[:3, 3]
                if head0 is None:
                    head0 = head_w.copy()
                disp = head_w - head0

            if skel_ema is not None:
                # Place the skeleton: head-follow translates the pelvis so the
                # skeleton's HEAD sits at (origin height + head displacement);
                # --fixed pins the pelvis at the world origin.
                if args.fixed:
                    anchor = -skel_ema[0]          # pelvis -> origin
                else:
                    # skeleton head should land at [disp_xy, head0_z + disp_z].
                    head_target = np.array([disp[0], disp[1],
                                            (head0[2] if head0 is not None else 1.5) + disp[2]])
                    anchor = head_target - skel_ema[_HEAD_JID]
                pts = skel_ema + anchor[None, :]

                viewer.user_scn.ngeom = 0
                _draw_world_axes(viewer.user_scn, length=args.axis_length)
                _draw_skeleton(viewer.user_scn, pts, _SMPL_PARENTS, _SMPL_MAPPED_JOINTS)

                # Ground-truth AVP poses (head + both hands), if enabled. Each raw
                # AVP 4x4 is mapped into the viewer frame (rotate basis by align_R)
                # and shifted horizontally by the SAME latched-head offset the
                # skeleton uses, so GT and skeleton share one frame: in head-follow
                # the GT head sphere sits on the skeleton head, and the GT hand
                # spheres show where the real hands are vs EgoPoser's guessed wrists.
                if not args.no_gt and head0 is not None:
                    shift = np.array([-head0[0], -head0[1], 0.0])
                    for T, rgba in ((gt_head, _GT_HEAD_RGBA),
                                    (gt_left, _GT_LEFT_RGBA),
                                    (gt_right, _GT_RIGHT_RGBA)):
                        if T is None:
                            continue
                        R_v = align_R @ T[:3, :3]
                        p_v = align_R @ T[:3, 3] + shift
                        _draw_pose(viewer.user_scn, R_v, p_v, rgba)
            viewer.sync()

            now = time.time()
            if now - last_status >= 1.0:
                rate = frames_seen / (now - last_status)
                if args.do_print and skel_ema is not None:
                    print(f"[EGOVIEW] {rate:4.0f} fps | pitch={prior_pitch:+.3f} "
                          f"yaw={prior_yaw:+.3f} rad | head disp "
                          f"[{disp[0]:+.2f} {disp[1]:+.2f} {disp[2]:+.2f}] m "
                          f"|disp|={np.linalg.norm(disp):.2f}")
                else:
                    tag = "tracking" if frames_seen else "NO DATA (publisher up? all 3 poses valid?)"
                    print(f"[EGOVIEW] {tag} | frames {rate:.0f}/s", end="\r", flush=True)
                frames_seen, last_status = 0, now

            time.sleep(max(0.0, 1.0 / 60.0 - 0.001))

    sub.close()
    print("\n[EGOVIEW] Stopped.")


if __name__ == "__main__":
    main()
