"""Interactive MuJoCo pose editor for the Astribot S1 upper body.

Open a live viewer, hand-pose the upper-body joints (4-DOF torso, 2-DOF neck,
two 7-DOF arms) with the keyboard, watch the result render in real time, then
save it. The 3 mobile-base DOFs are part of the body vector but are pinned at
the origin here (an initial *rest* pose never displaces the base). Saved poses
(:mod:`avp_teleop_upper_body.pose_io`) can be loaded as the teleop *initial /
rest* posture via
``python -m avp_teleop_upper_body.sim_teleop --init-pose <name>``.

The robot is fixed-base and every body joint is position-actuated, so the pose
is set purely kinematically (``mj_forward``, no physics) -- what you set is
exactly what is saved, with zero drift or fall-over.

Run inside the ``AVP`` conda env:

    # start from home, save to poses/my_pose.json
    python -m avp_teleop_upper_body.pose_editor --outname my_pose
    # open my_pose, save back to it (edit in place)
    python -m avp_teleop_upper_body.pose_editor --inname my_pose
    # open my_pose, save to a new my_pose1
    python -m avp_teleop_upper_body.pose_editor --inname my_pose --outname my_pose1

Keys (focus the viewer window):
    Up / Down      select previous / next joint
    Left / Right   decrease / increase the selected joint by the current step
    [  /  ]        halve / double the step size
    0              reset the selected joint to its home value
    M              mirror the left arm onto the right arm (symmetric pose)
    R              reset ALL joints to home
    S              save the current pose to the --name file
    P              print the current pose to the terminal
    H              print this help
    Space          print the selected joint + value
"""

from __future__ import annotations

import argparse
import time
from typing import List

import mujoco
import mujoco.viewer
import numpy as np

from avp_teleop_upper_body.config import (
    MJCF_PATH,
    BODY_JOINTS,
    BODY_HOME,
    CHASSIS_JOINTS,
    TORSO_JOINTS,
    TORSO_LEAN_JOINTS,
    NECK_JOINTS,
    ARM_JOINTS,
)
from avp_teleop_upper_body import pose_io


def _group_of(joint: str) -> str:
    if joint in CHASSIS_JOINTS:
        return "base "
    if joint in TORSO_LEAN_JOINTS:
        return "lean "   # sagittal lean spine (torso_joint_1/2/3)
    if joint in TORSO_JOINTS:
        return "waist"   # torso_joint_4: pure waist yaw
    if joint in NECK_JOINTS:
        return "neck "
    if joint in ARM_JOINTS["left"]:
        return "L-arm"
    if joint in ARM_JOINTS["right"]:
        return "R-arm"
    return "?????"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo editor for the upper-body initial pose."
    )
    parser.add_argument("--inname", "--from", dest="inname", default=None,
                        help="Pose name/path to OPEN and start editing from "
                             "(default: home). Alias: --from.")
    parser.add_argument("--outname", "--name", dest="outname", default=None,
                        help="Pose name/path to SAVE to on 'S' (default: same as "
                             "--inname, else custom_init). Alias: --name.")
    parser.add_argument("--mjcf", default=None, help="Override teleop MJCF path.")
    parser.add_argument("--note", default="", help="Optional note saved in the file.")
    args = parser.parse_args()

    # Save target: explicit --outname, else edit-in-place on --inname, else default.
    out_name = args.outname or args.inname or "custom_init"

    mjcf_path = args.mjcf or MJCF_PATH
    print(f"[EDIT] Loading {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # qpos address + range + actuator id for each body joint. The 3 base joints
    # are unlimited (jnt_range [0,0]) so they clamp to the origin -- intended for
    # a rest pose; the editable joints are torso/neck/arms.
    qadr: List[int] = []
    lo = np.empty(len(BODY_JOINTS))
    hi = np.empty(len(BODY_JOINTS))
    act_by_joint = {}
    for aid in range(model.nu):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                               model.actuator_trnid[aid, 0])
        if jn is not None:
            act_by_joint[jn] = aid
    act: List[int] = []
    for i, n in enumerate(BODY_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr.append(int(model.jnt_qposadr[jid]))
        lo[i], hi[i] = model.jnt_range[jid]
        act.append(act_by_joint[n])

    home = np.array(BODY_HOME, dtype=np.float64)
    if args.inname:
        start = pose_io.body_vector(pose_io.load_pose(args.inname))
        print(f"[EDIT] Opened pose '{args.inname}'.")
    else:
        start = home.copy()
        print("[EDIT] Starting from home.")
    q = np.clip(start, lo, hi)

    # `dirty` asks the main loop to push q into mjData. Only the MAIN thread is
    # allowed to touch mjData / call mj_forward -- the key_callback runs on the
    # viewer's UI thread, and concurrent mj_forward on one mjData segfaults.
    state = {"sel": 0, "step": 0.05, "dirty": True}

    def apply() -> None:
        """Write the body joint angles into qpos+ctrl and refresh kinematics.

        MAIN THREAD ONLY (called from the render loop under viewer.lock())."""
        for adr, ai, qi in zip(qadr, act, q):
            data.qpos[adr] = qi
            data.ctrl[ai] = qi
        mujoco.mj_forward(model, data)

    def show_sel() -> None:
        i = state["sel"]
        n = BODY_JOINTS[i]
        print(f"[EDIT] >> [{i:2d}] {_group_of(n)} {n:30s} "
              f"= {q[i]:+.3f} rad  (range [{lo[i]:+.2f},{hi[i]:+.2f}], "
              f"step {state['step']:.3f})")

    def print_pose() -> None:
        print("[EDIT] current pose:")
        for i, n in enumerate(BODY_JOINTS):
            print(f"         {_group_of(n)} {n:30s} = {q[i]:+.4f}")

    HELP = (
        "[EDIT] keys: Up/Down select joint | Left/Right -/+ by step | "
        "[ ] step/2 *2 | 0 joint->home | M mirror L-arm->R-arm | R all->home | "
        "S save | P print | H help"
    )

    # Index ranges of the two 7-DOF arms inside BODY_JOINTS (base 0-2, torso 3-6,
    # neck 7-8, L-arm 9-15, R-arm 16-22).
    _L0 = len(CHASSIS_JOINTS) + len(TORSO_JOINTS) + len(NECK_JOINTS)   # 9
    _R0 = _L0 + len(ARM_JOINTS["left"])                                # 16
    _NARM = len(ARM_JOINTS["right"])                                   # 7
    # Left->right MIRROR signs (per arm joint 1..7). Derived from a hand-made
    # symmetric pose: joints 1,3,5,7 flip sign, joints 2,4,6 keep it, so
    # right[k] = sign[k] * left[k] yields a sagittally symmetric posture.
    _ARM_MIRROR_SIGN = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    def key_callback(keycode: int) -> None:
        # UI THREAD: mutate only plain Python state (q / sel / step) and request
        # a redraw via state["dirty"]; never touch mjData or call mj_forward here.
        i = state["sel"]
        if keycode == 265:        # Up
            state["sel"] = (i - 1) % len(BODY_JOINTS); show_sel()
        elif keycode == 264:      # Down
            state["sel"] = (i + 1) % len(BODY_JOINTS); show_sel()
        elif keycode == 263:      # Left
            q[i] = float(np.clip(q[i] - state["step"], lo[i], hi[i]))
            state["dirty"] = True; show_sel()
        elif keycode == 262:      # Right
            q[i] = float(np.clip(q[i] + state["step"], lo[i], hi[i]))
            state["dirty"] = True; show_sel()
        elif keycode == 91:       # [
            state["step"] = max(0.005, state["step"] / 2); show_sel()
        elif keycode == 93:       # ]
            state["step"] = min(0.5, state["step"] * 2); show_sel()
        elif keycode == 48:       # 0
            q[i] = float(np.clip(home[i], lo[i], hi[i]))
            state["dirty"] = True; show_sel()
        elif keycode == 82:       # R
            q[:] = np.clip(home, lo, hi)
            state["dirty"] = True
            print("[EDIT] reset ALL joints to home.")
        elif keycode == 77:       # M : mirror left arm -> right arm
            for k in range(_NARM):
                q[_R0 + k] = float(np.clip(_ARM_MIRROR_SIGN[k] * q[_L0 + k],
                                           lo[_R0 + k], hi[_R0 + k]))
            state["dirty"] = True
            print("[EDIT] mirrored L-arm -> R-arm (symmetric).")
        elif keycode == 83:       # S
            joints = {n: float(q[k]) for k, n in enumerate(BODY_JOINTS)}
            path = pose_io.save_pose(out_name, joints, note=args.note)
            print(f"[EDIT] saved pose -> {path}")
        elif keycode == 80:       # P
            print_pose()
        elif keycode == 72:       # H
            print(HELP)
        elif keycode == 32:       # Space
            show_sel()

    apply()  # main thread: seed the initial pose before the viewer starts
    state["dirty"] = False
    print(HELP)
    print(f"[EDIT] will save to: {pose_io.resolve_path(out_name)}")
    show_sel()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            if state["dirty"]:
                with viewer.lock():
                    apply()
                state["dirty"] = False
            viewer.sync()
            time.sleep(1.0 / 60.0)

    print("[EDIT] viewer closed. (press S in the window to save before closing)")


if __name__ == "__main__":
    main()
