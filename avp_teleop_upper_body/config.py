"""Configuration for the upper-body (head + torso + dual-arm) teleop.

Most coordinate/finger/calibration settings are *reused* from the dual-arm
package (:mod:`avp_teleop.config`) so the two stay consistent; this module only
adds what the merged whole-upper-body solver needs: the torso/neck joint names,
the head end-effector frame, the 20-DOF body joint order + home pose, and the
merged-IK task weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# --- reuse the dual-arm package's coordinate / path / finger definitions ---
from avp_teleop.config import (
    NetworkConfig,
    AVPConfig,
    RetargetConfig,
    FingerSpec,
    ARM_JOINTS,
    TOOL_BODY,
    ARM_HOME,
    MJCF_PATH,
    AVP_TO_ROBOT_R,
    _finger_specs,
)

__all__ = [
    "MJCF_PATH",
    "AVP_TO_ROBOT_R",
    "ARM_JOINTS",
    "TOOL_BODY",
    "TORSO_JOINTS",
    "NECK_JOINTS",
    "HEAD_FRAME_BODY",
    "BODY_JOINTS",
    "BODY_HOME",
    "all_finger_joints",
    "finger_specs",
    "WholeBodyIKConfig",
    "PoseFilterConfig",
    "UpperBodyConfig",
    "default_config",
]


# --------------------------------------------------------------------------- #
# Upper-body kinematic chain (verified against astribot_s1_teleop.xml)
# --------------------------------------------------------------------------- #
# torso_link_4 is the SHARED root of both arms AND the head:
#   base -> torso_joint_1..4 -> torso_link_4 -> {head_joint_1,2 ; left arm ; right arm}
# Moving the torso moves both arm bases, which is exactly why the arms must be
# solved *together* with the torso/head in one configuration (see whole_body_ik).
TORSO_JOINTS: List[str] = [f"astribot_torso_joint_{i}" for i in range(1, 5)]
NECK_JOINTS: List[str] = ["astribot_head_joint_1", "astribot_head_joint_2"]

# Robot "head" end-effector frame: the head camera body is the natural analogue
# of the AVP headset (eye/camera) pose. It is driven by both neck joints.
HEAD_FRAME_BODY: str = "astribot_head_camera_base_link"

# The 20 actuated DOFs solved as one body, in a FIXED order used everywhere
# (IK qpos vector, SimRobot.command_arm, BODY_HOME). Order: torso, neck, L, R.
BODY_JOINTS: List[str] = (
    TORSO_JOINTS + NECK_JOINTS + ARM_JOINTS["left"] + ARM_JOINTS["right"]
)

# Home / reference posture for the 20 body joints. torso + neck rest at 0
# (upright, looking forward) -- this matches the neutral configuration the
# reduced Pinocchio model locks the non-body joints at, keeping frames aligned.
BODY_HOME: List[float] = (
    [0.0] * len(TORSO_JOINTS)
    + [0.0] * len(NECK_JOINTS)
    + list(ARM_HOME["left"])
    + list(ARM_HOME["right"])
)


def all_finger_joints() -> List[str]:
    """Every finger joint name across both hands (left then right)."""
    return [
        jn
        for side in ("left", "right")
        for spec in _finger_specs(side)
        for (jn, _w) in spec.joints
    ]


def finger_specs(side: str) -> List[FingerSpec]:
    """Per-finger chains for one hand (delegates to the dual-arm package)."""
    return _finger_specs(side)


# --------------------------------------------------------------------------- #
# Merged whole-upper-body IK knobs
# --------------------------------------------------------------------------- #
@dataclass
class WholeBodyIKConfig:
    """Task weights and integration knobs for the merged Pink solver.

    The three end-effector tasks (head, left tool, right tool) and one posture
    task are solved together. Hands are weighted higher than the head so grasp
    precision wins when they compete for the shared torso DOFs; the posture task
    keeps the torso near upright and the elbows natural (resolves the large
    redundancy of a 20-DOF chain tracking <=18 task dimensions).
    """

    # End-effector tracking weights.
    arm_position_cost: float = 10.0
    arm_orientation_cost: float = 1.0      # used only when track_orientation
    head_position_cost: float = 3.0        # moderate: head should not yank torso
    head_orientation_cost: float = 1.0     # used only when head_track_orientation
    # Null-space regularisation toward BODY_HOME.
    #posture_cost: float = 1e-2
    posture_cost: float = 1e-1

    # Levenberg-Marquardt damping on the frame tasks (singularity robustness).
    #lm_damping: float = 1e-3
    lm_damping: float = 1e-1

    # One differential-IK (QP) step per control tick. With hard QP velocity /
    # acceleration limits (below), the per-tick step is already bounded inside
    # the optimiser, so we no longer iterate-and-clip (see whole_body_ik.solve).
    solver: str = "quadprog"
    control_dt: float = 1.0 / 60.0

    # --- Hard velocity / acceleration limits (enforced inside the QP) ------- #
    # The teleop MJCF declares no velocity limits, so we inject them here and let
    # pink.limits add them as QP inequalities. This *supersedes* the old manual
    # per-tick np.clip: the solver now trades off rate limits against task
    # tracking instead of truncating the solution after the fact. torso/neck are
    # heavy, slow joints -> tighter caps than the arms (same intent as the old
    # torso_neck_max_step / arm_max_step split).
    torso_neck_max_velocity: float = 1.8       # rad/s  (~ old 0.03 rad/tick @60Hz)
    arm_max_velocity: float = 3.0              # rad/s  (~ old 0.05 rad/tick @60Hz)
    torso_neck_max_acceleration: float = 60.0  # rad/s^2
    arm_max_acceleration: float = 100.0        # rad/s^2
    # ConfigurationLimit steer-away gain in [0, 1] (Kanoun 2012); keeps joints
    # off their position stops. This was Pink's default limit already.
    config_limit_gain: float = 0.5
    # Master switch: False falls back to an unconstrained solve + a legacy clip.
    enforce_limits: bool = True

    # --- Soft smoothing tasks (low-cost QP objectives, NOT hard constraints) - #
    # Two extra Pink tasks shape the motion *inside* the hard velocity/accel
    # caps above, so the robot moves smoothly rather than just "as fast as the
    # cap allows":
    #   * DampingTask          penalizes |v|        -> bleeds off joint velocity
    #                          (brings the arm to rest when nothing drives it).
    #   * LowAccelerationTask  penalizes |v - v_prev| -> penalizes frame-to-frame
    #                          acceleration -> less jerk / fewer abrupt changes.
    # Costs are kept FAR below the end-effector tracking costs (arm=10, head=3)
    # so smoothing never visibly fights tracking -- they only smooth the slack
    # the redundant 20-DOF chain has left over. Pink's docs recommend pairing
    # LowAccelerationTask with a DampingTask (the former alone tends to
    # oscillate, as it does not dissipate energy). Set either to 0 to disable
    # that task (e.g. to A/B compare smoothness).
    damping_cost: float = 1e-1        # cost on |v|          (units: s/rad)
    low_accel_cost: float = 1e-1      # cost on |v - v_prev| (units: s^2/rad)

    def _body_split(self, tn: float, arm: float) -> np.ndarray:
        n_tn = len(TORSO_JOINTS) + len(NECK_JOINTS)
        n_arm = 2 * len(ARM_JOINTS["right"])
        return np.array([tn] * n_tn + [arm] * n_arm)

    def max_velocity(self) -> np.ndarray:
        """Per-joint velocity cap (rad/s) aligned with BODY_JOINTS order."""
        return self._body_split(self.torso_neck_max_velocity, self.arm_max_velocity)

    def max_acceleration(self) -> np.ndarray:
        """Per-joint acceleration cap (rad/s^2) aligned with BODY_JOINTS order."""
        return self._body_split(
            self.torso_neck_max_acceleration, self.arm_max_acceleration
        )


@dataclass
class PoseFilterConfig:
    """EMA / SLERP smoothing applied to each AVP target pose before the IK.

    A first-order exponential filter (see :mod:`avp_teleop_upper_body.pose_filter`)
    tames AVP tracking jitter: translation gets a 3D vector EMA, rotation gets a
    SO(3) SLERP. Translation and rotation are smoothed independently.

    Each coefficient is in ``(0, 1]``:
        ``1.0`` -> pass-through (no smoothing);
        smaller -> smoother but laggier (averages more past frames).
    The same two coefficients apply to all three targets (head / left / right).
    """

    alpha_translation: float = 0.5
    alpha_rotation: float = 0.5


@dataclass
class UpperBodyConfig:
    """Top-level config for the upper-body teleop pipeline."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    avp: AVPConfig = field(default_factory=AVPConfig)
    ik: WholeBodyIKConfig = field(default_factory=WholeBodyIKConfig)
    # Reused for finger retargeting (smoothing / reach band) and for the
    # AVP->robot world rotation (align_R). Arm position_scale / track_orientation
    # also live here.
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    # Smoothing of the AVP target poses (head + both hands) before the solver.
    filter: PoseFilterConfig = field(default_factory=PoseFilterConfig)

    # Head-specific mapping knobs (the arms use retarget.position_scale /
    # retarget.track_orientation).
    head_position_scale: float = 1.0
    head_track_orientation: bool = True

    @property
    def align_R(self) -> np.ndarray:
        return self.retarget.align_R

    @property
    def position_scale(self) -> float:
        return self.retarget.position_scale

    @property
    def track_orientation(self) -> bool:
        return self.retarget.track_orientation


def default_config() -> UpperBodyConfig:
    import os

    cfg = UpperBodyConfig()
    env_ip = os.environ.get("AVP_IP")
    if env_ip:
        cfg.avp.connection_id = env_ip
    return cfg
