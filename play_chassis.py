#!/usr/bin/env python3
"""Manually drive the S1 mobile base's 3 chassis actuators in MuJoCo.

A teaching / observation tool: it loads the teleop model, holds the whole upper
body at a saved pose (default ``my_pose1``), and lets you drive ONLY the three
chassis DOFs from the keyboard so you can see what each one does:

    astribot_chassis_x    : slide along WORLD +X   (metres)
    astribot_chassis_y    : slide along WORLD +Y   (metres)
    astribot_chassis_zrot : rotate about WORLD +Z  (radians, "yaw" / heading)

The point to observe: x and y are WORLD-frame slides that come BEFORE the yaw
hinge in the kinematic chain, so they translate along fixed world axes no matter
which way the robot faces -- this is an OMNIDIRECTIONAL (holonomic) base. Press
only the translate keys and the robot slides sideways WITHOUT turning; press only
yaw and it spins in place. Toggle BODY-frame mode ('m') to instead drive
"forward / strafe" relative to the current heading -- you will see the script has
to move BOTH x and y actuators at once to go "forward" once the robot has turned,
which is exactly why two translation DOFs are needed.

Run (from the repo root, with the project's python env):

    python play_chassis.py                 # default pose my_pose1
    python play_chassis.py --init-pose home
    python play_chassis.py --step 0.02 --yaw-step 0.05
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import mujoco
import mujoco.viewer

# Make the sibling packages importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from avp_teleop_upper_body.config import (
    MJCF_PATH, BODY_JOINTS, BODY_HOME, CHASSIS_JOINTS, all_finger_joints,
)
from avp_teleop_upper_body import pose_io
from avp_teleop.robot_interface import SimRobot

# GLFW key codes: printable keys equal the ASCII of the UPPERCASE character.
KEY_SPACE = 32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-pose", default="my_pose1",
                    help="Saved pose name (or path) to hold the upper body at, "
                         "or 'home' for BODY_HOME. Default: my_pose1.")
    ap.add_argument("--mjcf", default=None, help="Override teleop MJCF path.")
    ap.add_argument("--step", type=float, default=0.02,
                    help="Metres the base moves per key press (default %(default)s).")
    ap.add_argument("--yaw-step", type=float, default=0.05,
                    help="Radians the base yaws per key press (default %(default)s).")
    args = ap.parse_args()

    mjcf_path = args.mjcf or MJCF_PATH
    print(f"[CHASSIS] Loading {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # Upper-body hold pose: a saved pose, or BODY_HOME.
    if args.init_pose and args.init_pose.lower() != "home":
        body_home = pose_io.body_vector(pose_io.load_pose(args.init_pose))
        print(f"[CHASSIS] Holding upper body at pose '{args.init_pose}'.")
    else:
        body_home = np.array(BODY_HOME, dtype=np.float64)
        print("[CHASSIS] Holding upper body at BODY_HOME.")

    # SimRobot owns the arm + finger actuators (it holds the upper body still); we
    # drive the 3 chassis actuators ourselves, by name (actuator name == joint name).
    robot = SimRobot(model, data, BODY_JOINTS, all_finger_joints(), "chassis_base")

    def act_id(name: str) -> int:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise SystemExit(f"[CHASSIS] no actuator named '{name}' in this model")
        return aid

    ax_x, ax_y, ax_yaw = (act_id(n) for n in CHASSIS_JOINTS)
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in CHASSIS_JOINTS}

    # Seat the whole body at the pose and hold it via the position actuators. The
    # chassis targets start at 0 (origin, facing world -Y at neutral).
    for adr, qi in zip(robot._arm_qpos_adr, body_home):
        data.qpos[adr] = qi
    mujoco.mj_forward(model, data)
    robot.command_arm(np.asarray(body_home))

    # Our commanded chassis targets (position servos track these).
    tgt = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    body_frame = False  # False: drive world X/Y; True: drive forward/strafe

    def apply_targets() -> None:
        data.ctrl[ax_x] = tgt["x"]
        data.ctrl[ax_y] = tgt["y"]
        data.ctrl[ax_yaw] = tgt["yaw"]

    apply_targets()

    def key_callback(key: int) -> None:
        nonlocal body_frame
        s, ys = args.step, args.yaw_step
        # Movement input in the chosen frame. In WORLD mode fwd/strafe map straight
        # to world X/Y; in BODY mode they are rotated by the current heading (yaw),
        # so "forward" recruits BOTH x and y once the robot has turned.
        fwd = strafe = 0.0
        if key in (ord("W"), 265):      # W / Up
            fwd = +s
        elif key in (ord("S"), 264):    # S / Down
            fwd = -s
        elif key in (ord("A"), 263):    # A / Left
            strafe = +s
        elif key in (ord("D"), 262):    # D / Right
            strafe = -s
        elif key == ord("Q"):           # yaw left (+)
            tgt["yaw"] += ys
        elif key == ord("E"):           # yaw right (-)
            tgt["yaw"] -= ys
        elif key == ord("M"):           # toggle world / body frame
            body_frame = not body_frame
            print(f"[CHASSIS] frame = {'BODY (forward/strafe)' if body_frame else 'WORLD (X/Y)'}")
            return
        elif key == ord("R"):           # reset base to origin
            tgt["x"] = tgt["y"] = tgt["yaw"] = 0.0
            print("[CHASSIS] base reset to origin")
            apply_targets()
            return
        else:
            return

        if fwd or strafe:
            if body_frame:
                # Chain order is x -> y -> yaw, so the base heading is +yaw about
                # world Z. Forward is that heading; strafe is 90 deg left of it.
                c, sn = math.cos(tgt["yaw"]), math.sin(tgt["yaw"])
                tgt["x"] += fwd * c - strafe * sn
                tgt["y"] += fwd * sn + strafe * c
            else:
                tgt["x"] += fwd
                tgt["y"] += strafe
        apply_targets()

    print(__doc__.split("Run (")[0].strip())
    print("\n[CHASSIS] Keys:  W/S = +/-X (or forward)   A/D = +/-Y (or strafe)")
    print("               Q/E = yaw left/right    m = toggle world/body frame")
    print("               r = reset base           space = pause (viewer)")
    print(f"[CHASSIS] step={args.step} m, yaw_step={args.yaw_step} rad. "
          f"Frame = WORLD (X/Y).\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            q = data.qpos
            # Overlay the live base state so you can read x/y/yaw as you drive.
            viewer.user_scn.ngeom = 0
            viewer.sync()
            print(f"\r[CHASSIS] x={q[qadr['astribot_chassis_x']]:+.3f}  "
                  f"y={q[qadr['astribot_chassis_y']]:+.3f}  "
                  f"yaw={math.degrees(q[qadr['astribot_chassis_zrot']]):+6.1f} deg   ",
                  end="", flush=True)
    print("\n[CHASSIS] viewer closed.")


if __name__ == "__main__":
    main()
