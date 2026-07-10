"""sim_teleop: drive the Astribot S1 in MuJoCo from AVP hand frames.

Pipeline (this process):

    UDP HandFrame  ->  arm IK + finger retarget  ->  data.ctrl  ->  mj_step  ->  viewer

One control loop drives one *or both* arms: with ``--side both`` the loop holds
a list of two per-side controllers (left + right), each with its own IK solver,
hand retargeter and calibration. The arms are kinematically independent in this
model (the torso/chassis are held at neutral inside the IK), so solving them
separately is exact — there is no shared DOF to couple.

Run inside the `AVP` conda env (after starting avp_publisher in another shell):

    python -m avp_teleop.sim_teleop                 # single arm (default side)
    python -m avp_teleop.sim_teleop --side both     # dual-arm teleop
    python -m avp_teleop.sim_teleop --side right --no-orientation

Viewer keys:
    c      (re)calibrate ALL arms: anchor each hand pose to its tool pose
    space  pause / resume teleop (sim keeps running)
    r      start / stop trajectory recording (only when --record is set)

Recording:
    Pass ``--record PATH.json`` to capture the retargeted joint-space command
    stream in the Astribot ROS2 joint-space format (see recording.py). Frames
    are captured only while teleop is calibrated, not paused, and recording is
    armed (toggle with 'r', or auto-armed with --record-autostart). The file is
    written when the viewer closes. Replay it offline with ``replay_sim.py``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np

from avp_teleop.config import default_config
from avp_teleop.transport import HandFrameSubscriber
from avp_teleop.robot_interface import SimRobot
from avp_teleop.retarget.arm_ik import ArmIK
from avp_teleop.retarget.hand_retarget import HandRetargeter
from avp_teleop.retarget.frames import WristCalibration, wrist_to_tool_target
from avp_teleop.recording import TrajectoryRecorder


@dataclass
class ArmController:
    """Everything needed to drive one arm + hand from one side's frames."""

    side: str
    robot: SimRobot
    ik: object
    hand: HandRetargeter
    ranges: Dict[str, tuple]
    home: np.ndarray
    q_arm: np.ndarray
    calib: Optional[WristCalibration] = None
    needs_calib: bool = True


def _build_ik(cfg, model, mjcf_path, side):
    """Construct the configured IK backend for one side.

    Both backends expose the same ``solve(q_init, target_p, target_R,
    base_qpos)`` contract, so the control loop is identical for either.
    """
    arm_joints = cfg.arm_joints_for(side)
    tool_body = cfg.tool_body_for(side)
    home = np.array(cfg.arm_home_for(side))
    if cfg.retarget.ik_backend == "pink":
        # Imported lazily so the legacy backend works without pinocchio/pink.
        from avp_teleop.retarget.arm_ik_pink import PinkArmIK
        print(f"[SIM] IK backend ({side}): pink (Pinocchio + Pink)")
        return PinkArmIK(
            mjcf_path,
            arm_joint_names=arm_joints,
            tool_frame_name=tool_body,
            home=home,
            position_cost=cfg.retarget.pink_position_cost,
            orientation_cost=(cfg.retarget.pink_orientation_cost
                              if cfg.retarget.track_orientation else 0.0),
            posture_cost=cfg.retarget.pink_posture_cost,
            lm_damping=cfg.retarget.pink_lm_damping,
            max_joint_step=cfg.retarget.max_joint_step,
            control_dt=cfg.retarget.control_dt,
            solver_iters=cfg.retarget.pink_solver_iters,
            solver=cfg.retarget.pink_solver,
        )
    print(f"[SIM] IK backend ({side}): mujoco (legacy DLS)")
    return ArmIK(
        model,
        arm_joint_names=arm_joints,
        tool_body_name=tool_body,
        damping=cfg.retarget.dls_damping,
        max_iters=cfg.retarget.ik_iters,
        pos_tol=cfg.retarget.ik_pos_tol,
        max_joint_step=cfg.retarget.max_joint_step,
    )


def _build_controller(cfg, model, data, mjcf_path, side) -> ArmController:
    """Assemble the SimRobot + IK + hand retargeter for one side."""
    finger_joints = [jn for spec in cfg.finger_specs_for(side) for (jn, _w) in spec.joints]
    robot = SimRobot(
        model, data,
        arm_joint_names=cfg.arm_joints_for(side),
        finger_joint_names=finger_joints,
        tool_body_name=cfg.tool_body_for(side),
    )
    home = np.array(cfg.arm_home_for(side))
    ik = _build_ik(cfg, model, mjcf_path, side)
    hand = HandRetargeter(cfg.finger_specs_for(side), cfg.retarget)
    return ArmController(
        side=side, robot=robot, ik=ik, hand=hand,
        ranges=robot.joint_ranges(), home=home, q_arm=home.copy(),
    )


def _set_arms_home(model, data, controllers: List[ArmController]) -> None:
    """Place every arm at its home pose and hold it there via the actuators."""
    for ctrl in controllers:
        for adr, qi in zip(ctrl.robot._arm_qpos_adr, ctrl.home):
            data.qpos[adr] = qi
    mujoco.mj_forward(model, data)
    for ctrl in controllers:
        ctrl.robot.command_arm(ctrl.home)


def main() -> None:
    cfg = default_config()
    parser = argparse.ArgumentParser(description="AVP -> MuJoCo teleop for Astribot S1.")
    parser.add_argument("--side", default=cfg.avp.side,
                        choices=["left", "right", "both"],
                        help="Drive one arm or 'both' for dual-arm teleop.")
    parser.add_argument("--host", default=cfg.network.host)
    parser.add_argument("--port", type=int, default=cfg.network.port)
    parser.add_argument("--mjcf", default=None,
                        help="Override path to the teleop MJCF.")
    parser.add_argument("--no-orientation", action="store_true",
                        help="Track wrist position only (ignore orientation).")
    parser.add_argument("--orientation", action="store_true",
                        help="Also track wrist orientation (6-DOF).")
    parser.add_argument("--position-scale", type=float,
                        default=cfg.retarget.position_scale)
    parser.add_argument("--ik-backend", default=cfg.retarget.ik_backend,
                        choices=["pink", "mujoco"],
                        help="IK solver: 'pink' (Pinocchio+Pink, default) or "
                             "'mujoco' (legacy DLS fallback).")
    parser.add_argument("--record", default=None, metavar="PATH.json",
                        help="Record the retargeted joint-space trajectory to "
                             "this JSON file (Astribot ROS2 format).")
    parser.add_argument("--record-autostart", action="store_true",
                        help="Arm recording immediately (else press 'r' to "
                             "start). Only relevant with --record.")
    args = parser.parse_args()

    cfg.avp.side = args.side
    cfg.retarget.position_scale = args.position_scale
    cfg.retarget.ik_backend = args.ik_backend
    if args.orientation:
        cfg.retarget.track_orientation = True
    if args.no_orientation:
        cfg.retarget.track_orientation = False

    from avp_teleop.config import MJCF_PATH
    mjcf_path = args.mjcf or MJCF_PATH

    print(f"[SIM] Loading {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    sides = cfg.sides()
    controllers = [_build_controller(cfg, model, data, mjcf_path, s) for s in sides]
    _set_arms_home(model, data, controllers)

    sub = HandFrameSubscriber(args.host, args.port, timeout_s=cfg.network.recv_timeout_s)

    # --- optional trajectory recorder ---
    recorder: Optional[TrajectoryRecorder] = None
    if args.record:
        finger_names = {
            s: [jn for spec in cfg.finger_specs_for(s) for (jn, _w) in spec.joints]
            for s in ("left", "right")
        }
        recorder = TrajectoryRecorder(
            sides=sides,
            source_model=mjcf_path,
            nominal_dt=1.0 / 60.0,
            finger_joint_names=finger_names,
        )
        print(f"[SIM] Recording -> {args.record} "
              f"({'armed' if args.record_autostart else 'press r to start'})")

    # --- shared control state, mutated by the viewer key callback ---
    state = {"paused": False, "recording": bool(args.record_autostart and recorder)}

    def key_callback(keycode: int) -> None:
        # 'c' = 67, space = 32, 'r' = 82
        if keycode == 67:
            for ctrl in controllers:
                ctrl.needs_calib = True
            print("[SIM] Recalibration requested (all arms).")
        elif keycode == 32:
            state["paused"] = not state["paused"]
            print(f"[SIM] {'Paused' if state['paused'] else 'Resumed'}.")
        elif keycode == 82 and recorder is not None:
            state["recording"] = not state["recording"]
            print(f"[SIM] Recording {'STARTED' if state['recording'] else 'STOPPED'} "
                  f"({recorder.n_frames} frames so far).")

    print(f"[SIM] Subscribing udp://{args.host}:{args.port}, sides={'+'.join(sides)}, "
          f"orientation={'on' if cfg.retarget.track_orientation else 'off'}")
    print("[SIM] Waiting for AVP frames... (press 'c' to calibrate, space to pause)")

    n_steps_per_frame = max(1, int(round((1.0 / 60.0) / model.opt.timestep)))

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        last_status = time.time()
        frames_seen = 0
        while viewer.is_running():
            frames = sub.latest_by_side()
            if frames:
                frames_seen += 1

            # Record one frame per control tick while armed + actively driving.
            capturing = (recorder is not None and state["recording"]
                         and not state["paused"])
            if capturing:
                recorder.begin_frame(time.time())
            captured_any = False

            for ctrl in controllers:
                frame = frames.get(ctrl.side)
                if frame is None or not frame.valid:
                    continue

                # (Re)calibrate: anchor this hand's pose to this tool's pose.
                if ctrl.needs_calib:
                    R_tool, p_tool = ctrl.robot.get_tool_pose()
                    tool_pose = np.eye(4)
                    tool_pose[:3, :3] = R_tool
                    tool_pose[:3, 3] = p_tool
                    ctrl.calib = WristCalibration.capture(frame.wrist, tool_pose)
                    ctrl.q_arm = ctrl.robot.get_arm_qpos()
                    ctrl.needs_calib = False
                    print(f"[SIM] Calibrated ({ctrl.side}).")

                if not state["paused"] and ctrl.calib is not None:
                    target_R, target_p = wrist_to_tool_target(
                        frame.wrist,
                        ctrl.calib,
                        cfg.retarget.position_scale,
                        cfg.retarget.track_orientation,
                        cfg.retarget.align_R,
                    )
                    q = ctrl.ik.solve(
                        ctrl.q_arm,
                        target_p,
                        target_R if cfg.retarget.track_orientation else None,
                        ctrl.robot.base_qpos(),
                    )
                    ctrl.q_arm = q
                    ctrl.robot.command_arm(q)

                    finger_targets = ctrl.hand.joint_targets(frame.keypoints, ctrl.ranges)
                    ctrl.robot.command_fingers(finger_targets)

                    if capturing:
                        recorder.set_arm(ctrl.side, q)
                        recorder.set_hand(ctrl.side, finger_targets, ctrl.ranges)
                        captured_any = True

            if capturing:
                if captured_any:
                    recorder.commit_frame()
                else:
                    recorder._pending = None  # nothing driven this tick; drop it

            # Always step physics so the sim stays live even without data.
            for _ in range(n_steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()

            now = time.time()
            if now - last_status >= 2.0:
                rate = frames_seen / (now - last_status)
                tag = "tracking" if frames_seen else "NO DATA (is avp_publisher running?)"
                rec_tag = ""
                if recorder is not None:
                    rec_tag = (f" | REC {recorder.n_frames}f"
                               if state["recording"]
                               else f" | rec paused ({recorder.n_frames}f)")
                print(f"[SIM] {tag} | frames {rate:.0f}/s{rec_tag}",
                      end="\r", flush=True)
                frames_seen, last_status = 0, now

            time.sleep(max(0.0, 1.0 / 60.0 - 0.001))

    sub.close()
    if recorder is not None:
        recorder.save(args.record)
        print(f"\n[SIM] Saved {recorder.n_frames} frames -> {args.record}")
    print("\n[SIM] Stopped.")


if __name__ == "__main__":
    main()
