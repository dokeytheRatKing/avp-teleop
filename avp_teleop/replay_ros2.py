"""On-robot replay: publish a recorded trajectory to the Astribot S1 over ROS2.

This is the hardware-side player, the counterpart to ``replay_sim.py``. It takes
the same ``astribot_joint_trajectory`` JSON clip and republishes each frame as
per-component ``RobotJointController`` messages on the real robot's joint-space
command topics, paced to the recorded timing.

    /astribot_torso/joint_space_command        (4 dof)
    /astribot_head/joint_space_command          (2 dof)
    /astribot_arm_left/joint_space_command       (7 dof)
    /astribot_gripper_left/joint_space_command   (1 dof)
    /astribot_arm_right/joint_space_command      (7 dof)
    /astribot_gripper_right/joint_space_command  (1 dof)

msg RobotJointController{ header, int8 mode, string[] name, float64[] command },
mode = 1 (position). See astribot_simulation/src/simu_utils/robot_ros_interface.py.

!!! IMPORTANT / UNTESTED ON HARDWARE !!!
  * Requires ROS2 (rclpy) and the built ``astribot_msgs`` package on PYTHONPATH.
    These are NOT in the `AVP` conda env; run this on the robot / a ROS2 host
    (e.g. over ssh) after sourcing the workspace, not on the teleop laptop.
  * The clip records the dexterous BrainCo hand ('hand_*' components); the real
    S1 uses the 1-DOF gripper. By default this player publishes ONLY the arm +
    gripper (+ optionally torso/head) components and drops 'hand_*'. Verify the
    gripper sign/scale (derive_gripper_from_fingers maps to [0,100]) on the
    robot before trusting it.
  * SAFETY: start with --rate-limit and --dry-run. Ensure an e-stop is in reach.
    The first frame is NOT ramped from the robot's current pose — position the
    robot near the clip's first frame, or add a homing move first.

Usage (on a ROS2 host):

    python3 replay_ros2.py recording.json --dry-run          # print, don't publish
    python3 replay_ros2.py recording.json                     # arms + grippers
    python3 replay_ros2.py recording.json --include-torso-head
    python3 replay_ros2.py recording.json --speed 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

# recording.py is pure-stdlib+numpy, safe to import anywhere.
from avp_teleop.recording import load_trajectory


# Components published to the real robot. 'hand_*' is intentionally excluded
# (no dexterous hand on the gripper S1); include-torso-head is opt-in.
ARM_GRIPPER_COMPONENTS = [
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
]
TORSO_HEAD_COMPONENTS = ["astribot_torso", "astribot_head"]


def _select_components(meta: dict, include_torso_head: bool) -> List[str]:
    available = meta["components"]
    comps = [c for c in ARM_GRIPPER_COMPONENTS if c in available]
    if include_torso_head:
        comps = [c for c in TORSO_HEAD_COMPONENTS if c in available] + comps
    return comps


def _dry_run(doc: dict, comps: List[str], speed: float) -> None:
    frames = doc["frames"]
    print(f"[ROS2-DRY] {len(frames)} frames, components: {comps}")
    for c in comps:
        print(f"[ROS2-DRY]   {c}: {doc['metadata']['components'][c]}")
    for i in (0, len(frames) // 2, len(frames) - 1):
        f = frames[i]
        print(f"[ROS2-DRY] frame {i} t={f['t']:.3f}")
        for c in comps:
            vals = f["commands"].get(c)
            if vals is not None:
                pretty = ", ".join(f"{v:.3f}" for v in vals)
                print(f"[ROS2-DRY]     {c} <- [{pretty}]")
    print("[ROS2-DRY] No messages published (--dry-run).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded Astribot trajectory on the real S1 (ROS2).")
    parser.add_argument("trajectory")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--include-torso-head", action="store_true",
                        help="Also publish torso (4) + head (2) commands.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without importing ROS2 or publishing.")
    parser.add_argument("--mode", type=int, default=1,
                        help="RobotJointController.mode (1=position).")
    args = parser.parse_args()

    doc = load_trajectory(args.trajectory)
    meta = doc["metadata"]
    frames = doc["frames"]
    comps = _select_components(meta, args.include_torso_head)
    if not frames:
        print("[ROS2] Empty clip; nothing to publish.")
        return

    if args.dry_run:
        _dry_run(doc, comps, args.speed)
        return

    # ---- ROS2 imports deferred so --dry-run works without a ROS2 install ----
    try:
        import rclpy
        from rclpy.node import Node
        from astribot_msgs.msg import RobotJointController
    except Exception as e:  # ImportError or rclpy init issues
        print("[ROS2] Could not import rclpy / astribot_msgs. Run this on a "
              "ROS2 host with the astribot workspace sourced.\n"
              f"       Import error: {e}", file=sys.stderr)
        sys.exit(1)

    component_joint_names: Dict[str, List[str]] = {
        c: meta["components"][c] for c in comps
    }

    rclpy.init()
    node = Node("avp_trajectory_replayer")
    pubs = {
        c: node.create_publisher(RobotJointController,
                                 f"/{c}/joint_space_command", 10)
        for c in comps
    }
    print(f"[ROS2] Publishing components {comps} (mode={args.mode}). "
          f"{len(frames)} frames @ speed x{args.speed}. Ctrl+C to abort.")

    nominal_dt = float(meta.get("nominal_dt", 1.0 / 60.0))
    wall0 = time.time()
    try:
        for i, frame in enumerate(frames):
            if not rclpy.ok():
                break
            for c in comps:
                vals = frame["commands"].get(c)
                if vals is None:
                    continue
                msg = RobotJointController()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.mode = args.mode
                msg.name = list(component_joint_names[c])
                msg.command = [float(v) for v in vals]
                pubs[c].publish(msg)

            target_t = frame.get("t", i * nominal_dt)
            if args.speed > 0:
                sleep = target_t / args.speed - (time.time() - wall0)
                if sleep > 0:
                    time.sleep(sleep)
        print(f"\n[ROS2] Finished publishing {len(frames)} frames.")
    except KeyboardInterrupt:
        print("\n[ROS2] Aborted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
