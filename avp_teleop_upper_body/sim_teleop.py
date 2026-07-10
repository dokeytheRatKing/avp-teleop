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
from avp_teleop_upper_body import pose_io

Target = Tuple[np.ndarray, Optional[np.ndarray]]  # (world_p, world_R_or_None)


def _body_pose(model, data, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """(R, p) world pose of a body, read from current MuJoCo data."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy()


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

    mjcf_path = args.mjcf or MJCF_PATH
    print(f"[SIM] Loading {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

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
        max_velocity=cfg.ik.max_velocity(),
        max_acceleration=cfg.ik.max_acceleration(),
        config_limit_gain=cfg.ik.config_limit_gain,
        enforce_limits=cfg.ik.enforce_limits,
        control_dt=cfg.ik.control_dt,
        solver=cfg.ik.solver,
    )

    _set_body_home(model, data, body_robot, body_home)
    sub = UpperBodySubscriber(args.host, args.port,
                              timeout_s=cfg.network.recv_timeout_s)

    # The three end-effector frames and how to read the robot's current pose.
    end_frames = {
        "head": HEAD_FRAME_BODY,
        "left": TOOL_BODY["left"],
        "right": TOOL_BODY["right"],
    }
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
    print(f"[SIM] Whole-body priority: arms/waist/lean first, then mobile base "
          f"(damping: base={cfg.ik.damping_cost_chassis:g} > "
          f"upper/lean={cfg.ik.damping_cost:g}).")
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
        last_status = time.time()
        frames_seen = 0
        while viewer.is_running():
            hands, head = sub.poll()
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

            calibrated = any(c is not None for c in calib.values())
            if not state["paused"] and calibrated:
                q_body = ik.solve(q_body, targets["head"], targets["left"],
                                  targets["right"])
                body_robot.command_arm(q_body)
                for side in ("left", "right"):
                    if live[side] is not None:
                        ft = hand_retarget[side].joint_targets(
                            hands[side].keypoints, ranges)
                        body_robot.command_fingers(ft)

            for _ in range(n_steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()

            now = time.time()
            if now - last_status >= 2.0:
                rate = frames_seen / (now - last_status)
                tag = "tracking" if frames_seen else "NO DATA (is the publisher running?)"
                print(f"[SIM] {tag} | frames {rate:.0f}/s", end="\r", flush=True)
                frames_seen, last_status = 0, now

            time.sleep(max(0.0, 1.0 / 60.0 - 0.001))

    sub.close()
    print("\n[SIM] Stopped.")


if __name__ == "__main__":
    main()
