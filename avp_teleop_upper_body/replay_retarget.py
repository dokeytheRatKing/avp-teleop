"""Pure playback of a recorded retarget trajectory -- NO computation.

Unlike ``sim_teleop --replay-avp`` (which re-runs the full retarget + IK on a
raw AVP clip), this player does *zero* solving. It loads a
``retarget_trajectory`` file (see :mod:`avp_teleop_upper_body.trajectory_io`),
sets the robot to each stored joint configuration, and redraws the stored
overlays (target markers + tracking-error lines + EgoPoser skeleton). It never
imports the IK solver, the config task weights, or the estimator.

    python -m avp_teleop_upper_body.replay_retarget run1
    python -m avp_teleop_upper_body.replay_retarget run1 --no-overlays

Because every frame is a complete posture snapshot, scrubbing is instant (just
re-index and set qpos) and works backwards as well as forwards.

Viewer keys:
    space      pause / resume
    . , (>< )  step one frame forward / back (while paused)
    ] [        jump +/- 1 second
    0          jump to the start
    l          toggle looping
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np

from avp_teleop.robot_interface import SimRobot
from avp_teleop_upper_body.config import (
    BODY_JOINTS, HEAD_FRAME_BODY, TOOL_BODY, CHEST_HIP_FRAME,
    all_finger_joints,
)
from avp_teleop_upper_body import trajectory_io
from avp_teleop_upper_body.trajectory_io import frame_target
# Reuse the EXACT overlay drawing from the live teleop so replay looks identical.
from avp_teleop_upper_body.sim_teleop import (
    _body_pose, _prior_anchor_pose, _apply_simple_render,
    _draw_target_marker, _draw_prior_skeleton, _set_body_home,
    _TGT_MARKER_RGBA, _WAIST_LINK_BODY,
)

_END_FRAMES = {"head": HEAD_FRAME_BODY, "left": TOOL_BODY["left"],
               "right": TOOL_BODY["right"]}


def _progress_bar(i: int, n: int, t: float, *, width: int = 30) -> str:
    frac = (i + 1) / n if n else 0.0
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {frac*100:3.0f}%  frame {i+1}/{n}  t={t:5.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure playback of a recorded retarget trajectory (no IK).")
    parser.add_argument("trajectory",
                        help="Name (in retargetting_trajectory/) or path of a "
                             "retarget_trajectory JSON file.")
    parser.add_argument("--mjcf", default=None,
                        help="Override the MJCF path (default: the model the "
                             "trajectory was recorded on, from its metadata).")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Hide the target markers + EgoPoser skeleton "
                             "(show only the robot motion).")
    parser.add_argument("--rich-render", action="store_true",
                        help="Keep full textures/shadows (default: flat gray).")
    parser.add_argument("--body-alpha", type=float, default=None,
                        help="Robot body opacity in [0,1]; default 0.35 when a "
                             "skeleton is present, else 1.0.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop playback from the start when it ends.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override playback rate (default: the recorded "
                             "nominal_dt).")
    args = parser.parse_args()

    payload = trajectory_io.load_retarget_trajectory(args.trajectory)
    meta = payload["metadata"]
    frames: List[dict] = payload["frames"]
    n = len(frames)
    if n == 0:
        print("[REPLAY] Trajectory has no frames; nothing to play.")
        return

    body_joints = meta.get("body_joints", BODY_JOINTS)
    dt = (1.0 / args.fps) if args.fps else float(meta.get("nominal_dt", 1.0 / 60.0))
    has_skeleton = any(f.get("skeleton") is not None for f in frames)
    show_overlays = not args.no_overlays

    # Resolve the model path to ABSOLUTE (MuJoCo nested <include> breaks on a
    # relative path -- see the record/replay notes) and prefer the recorded one.
    mjcf_path = args.mjcf or meta.get("model_path")
    if not mjcf_path or not os.path.isfile(mjcf_path):
        raise FileNotFoundError(
            f"MJCF model not found: {mjcf_path!r}. Pass --mjcf to override "
            f"(the trajectory was recorded on '{meta.get('model_path')}').")
    mjcf_path = os.path.abspath(mjcf_path)
    print(f"[REPLAY] {args.trajectory}: {n} frames, dt={dt*1e3:.1f} ms, "
          f"model={os.path.basename(mjcf_path)}")
    print(f"[REPLAY] recorded argv: {' '.join(meta.get('argv', [])) or '(none)'}")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    if not args.rich_render:
        alpha = (args.body_alpha if args.body_alpha is not None
                 else (0.35 if (has_skeleton and show_overlays) else 1.0))
        _apply_simple_render(model, body_alpha=float(np.clip(alpha, 0.0, 1.0)))

    robot = SimRobot(model, data, body_joints, all_finger_joints(), HEAD_FRAME_BODY)

    # --- playback state mutated by the key callback ---
    st = {"paused": False, "i": 0, "loop": bool(args.loop), "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == 32:            # space
            st["paused"] = not st["paused"]
            print(f"\n[REPLAY] {'Paused' if st['paused'] else 'Resumed'}.")
        elif keycode in (46, 190):   # '.' / '>'
            st["step"] = 1
        elif keycode in (44, 188):   # ',' / '<'
            st["step"] = -1
        elif keycode == 93:          # ']'  -> +1 s
            st["i"] = min(n - 1, st["i"] + int(round(1.0 / dt)))
        elif keycode == 91:          # '['  -> -1 s
            st["i"] = max(0, st["i"] - int(round(1.0 / dt)))
        elif keycode == 48:          # '0'  -> start
            st["i"] = 0
        elif keycode == 76:          # 'l'  -> toggle loop
            st["loop"] = not st["loop"]
            print(f"\n[REPLAY] Loop {'ON' if st['loop'] else 'OFF'}.")

    def _apply_frame(idx: int) -> None:
        """Set the robot to frame ``idx`` (joints + fingers) and forward-kinematics."""
        frame = frames[idx]
        q = frame.get("q_body")
        if q is not None:
            robot.command_arm(np.asarray(q, dtype=np.float64))
            for adr, qi in zip(robot._arm_qpos_adr, q):   # place qpos too (static hold)
                data.qpos[adr] = float(qi)
        fingers = frame.get("fingers") or {}
        for _side, ft in fingers.items():
            robot.command_fingers({k: float(v) for k, v in ft.items()})
        mujoco.mj_forward(model, data)

    def _draw_overlays(scn, idx: int) -> None:
        scn.ngeom = 0
        if not show_overlays:
            return
        frame = frames[idx]
        for end, body in _END_FRAMES.items():
            tgt = frame_target(frame, end)
            if tgt is None:
                continue
            tp, _tr = tgt
            vr = frame.get("viz_rot", {}).get(end)
            vr = None if vr is None else np.asarray(vr, dtype=np.float64)
            _, actual_p = _body_pose(model, data, body)
            _draw_target_marker(scn, tp, vr, _TGT_MARKER_RGBA[end], actual_p=actual_p)
        skel = frame.get("skeleton")
        if skel is not None:
            from avp_teleop_upper_body.egoposer import _SMPL_PARENTS, _SMPL_MAPPED_JOINTS
            R_anchor, p_anchor = _prior_anchor_pose(
                model, data, CHEST_HIP_FRAME, _WAIST_LINK_BODY)
            _draw_prior_skeleton(scn, R_anchor, p_anchor,
                                 np.asarray(skel, dtype=np.float64),
                                 _SMPL_PARENTS, _SMPL_MAPPED_JOINTS)

    print("[REPLAY] space=pause  .,=step  ][=+/-1s  0=start  l=loop")
    _set_body_home(model, data, robot, np.asarray(frames[0]["q_body"],
                                                  dtype=np.float64))

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_callback) as viewer:
        last_print = time.time()
        while viewer.is_running():
            i = st["i"]
            _apply_frame(i)
            _draw_overlays(viewer.user_scn, i)
            viewer.sync()

            now = time.time()
            if now - last_print >= 0.1:
                tag = " PAUSED" if st["paused"] else ""
                print("[REPLAY] " + _progress_bar(i, n, frames[i].get("t", i * dt))
                      + tag, end="\r", flush=True)
                last_print = now

            # advance: manual single-step while paused, else play forward.
            if st["step"] != 0:
                st["i"] = int(np.clip(i + st["step"], 0, n - 1))
                st["step"] = 0
            elif not st["paused"]:
                nxt = i + 1
                if nxt >= n:
                    if st["loop"]:
                        nxt = 0
                    else:
                        st["paused"] = True
                        nxt = n - 1
                        print("\n[REPLAY] Reached end (paused). "
                              "'0' to restart, 'l' to loop.")
                st["i"] = nxt

            time.sleep(dt)

    print("\n[REPLAY] Stopped.")


if __name__ == "__main__":
    main()
