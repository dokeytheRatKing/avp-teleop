"""Trajectory recording for AVP -> Astribot S1 teleop.

Captures the *retargeted* command stream (joint-space) during a live teleop
session and serialises it in a format aligned with the real robot's ROS2
joint-space command interface, so the same clip can be:

  * replayed offline in MuJoCo (``replay_sim.py``) to validate motion, and
  * played back on the physical S1 over ROS2 (``replay_ros2.py``).

Format
------
JSON, ``schema = "astribot_joint_trajectory"``, ``version = 1``:

    {
      "schema": "astribot_joint_trajectory",
      "version": 1,
      "metadata": {
        "created": <unix time>,
        "source_model": <mjcf path the commands were generated on>,
        "sides": ["right"] | ["left"] | ["left", "right"],
        "control_mode": 1,               # RobotJointController.mode (position)
        "nominal_dt": 0.01667,           # target seconds between frames
        "components": {                  # ordered joint names per component
            "astribot_torso": [...4...],
            "astribot_head": [...2...],
            "astribot_arm_left": [...7...],
            "astribot_gripper_left": ["astribot_gripper_left_joint_L1"],
            "astribot_arm_right": [...7...],
            "astribot_gripper_right": [...],
            "hand_left": [...15 BrainCo finger joints...],
            "hand_right": [...]
        }
      },
      "frames": [
        {"t": 0.0,     "commands": {"astribot_arm_right": [...], ...}},
        {"t": 0.0166,  "commands": {...}},
        ...
      ]
    }

Every frame carries a *complete* S1 joint-space command (arms + grippers +
torso + head), plus the 15-DOF BrainCo finger joints per side under
``hand_left`` / ``hand_right`` so the clip can be replayed faithfully in the
teleop MJCF (which uses the dexterous hand) as well as on the real gripper.

The single-DOF gripper value is *derived* from the finger curl so that a clip
recorded with the dexterous-hand model can still drive the real gripper. The
mapping (mean finger closure -> [0, gripper_max]) is a reasonable default but
the sign/scale is UNVERIFIED on hardware; tune ``gripper_max`` / invert if the
real gripper opens/closes the wrong way.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# Real-S1 component joint names (order matches simulation_mujoco_param.yaml).
TORSO_JOINTS = [f"astribot_torso_joint_{i}" for i in range(1, 5)]
HEAD_JOINTS = [f"astribot_head_joint_{i}" for i in range(1, 3)]
GRIPPER_JOINT = {
    "left": "astribot_gripper_left_joint_L1",
    "right": "astribot_gripper_right_joint_L1",
}
ARM_COMPONENT = {"left": "astribot_arm_left", "right": "astribot_arm_right"}
GRIPPER_COMPONENT = {"left": "astribot_gripper_left", "right": "astribot_gripper_right"}
HAND_COMPONENT = {"left": "hand_left", "right": "hand_right"}

# Real gripper joint command range is [0, 100] (see astribot_s1 gripper actuator).
# 0 = open, 100 = closed by convention here.
GRIPPER_MAX = 100.0

# Which finger proximal joints define "closure" for the derived gripper value.
# We average the normalised angle of the four non-thumb proximal joints.
_CLOSURE_JOINTS = ("index", "middle", "ring", "pinky")


def derive_gripper_from_fingers(
    side: str,
    finger_targets: Dict[str, float],
    finger_ranges: Dict[str, tuple],
    gripper_max: float = GRIPPER_MAX,
) -> float:
    """Collapse the dexterous finger curl into a single [0, gripper_max] value.

    closure = mean over {index,middle,ring,pinky} proximal joints of
    (angle - low) / (high - low), then scaled to the gripper command range.
    """
    vals: List[float] = []
    for finger in _CLOSURE_JOINTS:
        jn = f"{side}_{finger}_proximal_joint"
        if jn not in finger_targets:
            continue
        lo, hi = finger_ranges.get(jn, (0.0, 1.0))
        rng = hi - lo
        if rng <= 1e-6:
            continue
        vals.append(float(np.clip((finger_targets[jn] - lo) / rng, 0.0, 1.0)))
    closure = float(np.mean(vals)) if vals else 0.0
    return closure * gripper_max


@dataclass
class TrajectoryRecorder:
    """Accumulates per-frame joint-space commands and writes them to JSON.

    Usage per control tick (only while actively teleoperating):

        rec.begin_frame(t)
        rec.set_arm(side, q_arm)                       # (7,) rad
        rec.set_hand(side, finger_targets, ranges)     # dict joint->rad
        rec.commit_frame()

    Components not set in a frame reuse the previous frame's value (or the
    configured hold pose for torso/head), so partial updates are safe.
    """

    sides: List[str]
    source_model: str
    nominal_dt: float
    finger_joint_names: Dict[str, List[str]]  # side -> ordered finger joints
    gripper_max: float = GRIPPER_MAX
    control_mode: int = 1

    _frames: List[dict] = field(default_factory=list)
    _pending: Optional[dict] = None
    _pending_t: float = 0.0
    # Last-known command per component, so every frame is complete.
    _last: Dict[str, List[float]] = field(default_factory=dict)
    _t0: Optional[float] = None

    def __post_init__(self) -> None:
        # Torso/head are held at their neutral pose by the teleop loop.
        self._last["astribot_torso"] = [0.0] * len(TORSO_JOINTS)
        self._last["astribot_head"] = [0.0] * len(HEAD_JOINTS)
        for side in ("left", "right"):
            self._last[ARM_COMPONENT[side]] = [0.0] * 7
            self._last[GRIPPER_COMPONENT[side]] = [0.0]
            self._last[HAND_COMPONENT[side]] = [0.0] * len(
                self.finger_joint_names.get(side, [])
            )

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def begin_frame(self, t_wall: float) -> None:
        if self._t0 is None:
            self._t0 = t_wall
        self._pending = {}
        self._pending_t = t_wall - self._t0

    def set_arm(self, side: str, q_arm: np.ndarray) -> None:
        if self._pending is None:
            return
        self._pending[ARM_COMPONENT[side]] = [float(x) for x in np.asarray(q_arm).ravel()]

    def set_hand(
        self,
        side: str,
        finger_targets: Dict[str, float],
        finger_ranges: Dict[str, tuple],
    ) -> None:
        """Record the full BrainCo finger vector AND the derived 1-DOF gripper."""
        if self._pending is None:
            return
        names = self.finger_joint_names.get(side, [])
        self._pending[HAND_COMPONENT[side]] = [
            float(finger_targets.get(n, 0.0)) for n in names
        ]
        gripper = derive_gripper_from_fingers(
            side, finger_targets, finger_ranges, self.gripper_max
        )
        self._pending[GRIPPER_COMPONENT[side]] = [gripper]

    def commit_frame(self) -> None:
        """Finalise the pending frame, filling unset components from history."""
        if self._pending is None:
            return
        for comp, val in self._pending.items():
            self._last[comp] = val
        # Emit a complete command (all known components) for this frame.
        commands = {comp: list(val) for comp, val in self._last.items()}
        self._frames.append({"t": round(self._pending_t, 6), "commands": commands})
        self._pending = None

    def _components_metadata(self) -> Dict[str, List[str]]:
        from avp_teleop.config import ARM_JOINTS

        comps: Dict[str, List[str]] = {
            "astribot_torso": list(TORSO_JOINTS),
            "astribot_head": list(HEAD_JOINTS),
        }
        for side in ("left", "right"):
            comps[ARM_COMPONENT[side]] = list(ARM_JOINTS[side])
            comps[GRIPPER_COMPONENT[side]] = [GRIPPER_JOINT[side]]
            comps[HAND_COMPONENT[side]] = list(self.finger_joint_names.get(side, []))
        return comps

    def save(self, path: str) -> None:
        doc = {
            "schema": "astribot_joint_trajectory",
            "version": 1,
            "metadata": {
                "created": time.time(),
                "source_model": self.source_model,
                "sides": list(self.sides),
                "control_mode": self.control_mode,
                "nominal_dt": self.nominal_dt,
                "gripper_max": self.gripper_max,
                "components": self._components_metadata(),
            },
            "frames": self._frames,
        }
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)


def load_trajectory(path: str) -> dict:
    """Load and lightly validate a recorded trajectory JSON."""
    with open(path, "r") as f:
        doc = json.load(f)
    if doc.get("schema") != "astribot_joint_trajectory":
        raise ValueError(f"{path}: not an astribot_joint_trajectory file.")
    if "frames" not in doc or "metadata" not in doc:
        raise ValueError(f"{path}: missing frames/metadata.")
    return doc
