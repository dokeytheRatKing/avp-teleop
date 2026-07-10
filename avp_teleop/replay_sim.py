"""Offline replay: drive the MuJoCo Astribot S1 from a recorded trajectory.

This is the sim-side half of the record/replay validation loop. It takes a
JSON clip written by ``sim_teleop.py --record`` (schema
``astribot_joint_trajectory``) and plays it back frame-by-frame into a MuJoCo
model, with NO Apple Vision Pro and NO network in the loop. If the robot
reproduces the recorded motion here, the retargeting + command stream are sound
and the same clip can be pushed to the real S1 over ROS2 (``replay_ros2.py``).

The clip stores commands keyed by joint name (grouped into robot components).
This replayer flattens all components to a single joint-name -> value map per
frame and writes each value onto the actuator that drives that joint in the
loaded model. That makes it model-agnostic:

  * default (the teleop MJCF, dexterous hand): the ``hand_*`` finger joints map
    to the finger actuators; the derived ``gripper`` joints have no actuator and
    are skipped.
  * a gripper MJCF (``--mjcf .../astribot_s1_with_gripper.xml``): the
    ``gripper_*_joint_L1`` joints map to the gripper actuators; the finger
    joints are skipped.

Run inside the `AVP` conda env:

    python -m avp_teleop.replay_sim recording.json
    python -m avp_teleop.replay_sim recording.json --speed 0.5 --loop
    python -m avp_teleop.replay_sim recording.json --no-render      # headless check
    python -m avp_teleop.replay_sim recording.json --mjcf <gripper_model.xml>
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from avp_teleop.recording import load_trajectory


def _actuator_by_joint(model: "mujoco.MjModel") -> Dict[str, int]:
    """Map joint name -> actuator id for every actuator that drives a joint."""
    out: Dict[str, int] = {}
    for aid in range(model.nu):
        trnid = model.actuator_trnid[aid, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        if jname is not None:
            out[jname] = aid
    return out


def _flatten_frame(components: Dict[str, List[str]], commands: Dict[str, list]
                   ) -> Dict[str, float]:
    """Zip each component's ordered joint names with its command vector."""
    flat: Dict[str, float] = {}
    for comp, names in components.items():
        vals = commands.get(comp)
        if vals is None:
            continue
        for jn, v in zip(names, vals):
            flat[jn] = float(v)
    return flat


def _build_ctrl_plan(
    model: "mujoco.MjModel", doc: dict
) -> Tuple[List[Tuple[int, str]], List[str]]:
    """Resolve which recorded joints have actuators in this model.

    Returns (mapped, skipped) where mapped is a list of (actuator_id, joint_name)
    and skipped is the joint names present in the clip but absent in the model.
    """
    act_by_joint = _actuator_by_joint(model)
    components = doc["metadata"]["components"]
    all_joints: List[str] = []
    for names in components.values():
        all_joints.extend(names)
    mapped = [(act_by_joint[jn], jn) for jn in all_joints if jn in act_by_joint]
    skipped = [jn for jn in all_joints if jn not in act_by_joint]
    return mapped, skipped


def replay(path: str, mjcf: str | None, speed: float, render: bool,
           loop: bool) -> None:
    doc = load_trajectory(path)
    meta = doc["metadata"]
    frames = doc["frames"]
    if not frames:
        print(f"[REPLAY] {path} has no frames; nothing to do.")
        return

    # Absolute path: MuJoCo resolves nested <include> paths relative to the
    # top-level file's directory, which only works reliably from an abs path.
    import os
    model_path = os.path.abspath(mjcf or meta.get("source_model"))
    print(f"[REPLAY] Loading model {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    components = meta["components"]
    mapped, skipped = _build_ctrl_plan(model, doc)
    if not mapped:
        raise RuntimeError(
            "No recorded joint matches an actuator in this model. "
            "Is --mjcf the right robot?"
        )
    print(f"[REPLAY] {len(mapped)} joints mapped to actuators, "
          f"{len(skipped)} skipped (no actuator in this model).")
    if skipped:
        # Grippers-when-hand-model or fingers-when-gripper-model are expected.
        preview = ", ".join(skipped[:6]) + (" ..." if len(skipped) > 6 else "")
        print(f"[REPLAY]   skipped: {preview}")

    nominal_dt = float(meta.get("nominal_dt", 1.0 / 60.0))
    n_steps = max(1, int(round(nominal_dt / model.opt.timestep)))

    def apply_frame(frame: dict) -> None:
        flat = _flatten_frame(components, frame["commands"])
        for aid, jn in mapped:
            if jn in flat:
                lo, hi = model.actuator_ctrlrange[aid]
                v = flat[jn]
                if hi > lo:  # respect ctrlrange when the actuator is limited
                    v = float(np.clip(v, lo, hi))
                data.ctrl[aid] = v

    def run_once(viewer=None) -> bool:
        """Play the clip once. Returns False if the viewer was closed."""
        wall0 = time.time()
        for i, frame in enumerate(frames):
            if viewer is not None and not viewer.is_running():
                return False
            apply_frame(frame)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            # Pace playback to the recorded timestamps, scaled by 1/speed:
            # sleep until this frame's scheduled wall-clock time.
            target_t = frame.get("t", i * nominal_dt)
            if speed > 0:
                sleep = target_t / speed - (time.time() - wall0)
                if sleep > 0:
                    time.sleep(sleep)
        return True

    # Seed the model at the first frame so it doesn't fall before playback.
    apply_frame(frames[0])
    mujoco.mj_forward(model, data)

    dur = frames[-1].get("t", len(frames) * nominal_dt)
    print(f"[REPLAY] {len(frames)} frames, ~{dur:.1f}s clip, speed x{speed}, "
          f"{'looping' if loop else 'once'}.")

    if not render:
        # Headless: run as fast as possible, no pacing.
        keep = True
        while keep:
            for frame in frames:
                apply_frame(frame)
                for _ in range(n_steps):
                    mujoco.mj_step(model, data)
            keep = loop
            if not loop:
                break
        print("[REPLAY] Headless replay complete.")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            if not run_once(viewer):
                break
            if not loop:
                break
    print("\n[REPLAY] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded Astribot joint trajectory in MuJoCo.")
    parser.add_argument("trajectory", help="Path to the recorded .json clip.")
    parser.add_argument("--mjcf", default=None,
                        help="Override the model to replay into (default: the "
                             "clip's source_model).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real time).")
    parser.add_argument("--loop", action="store_true", help="Loop forever.")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless (validation only, no viewer).")
    args = parser.parse_args()
    replay(args.trajectory, args.mjcf, args.speed,
           render=not args.no_render, loop=args.loop)


if __name__ == "__main__":
    main()
