"""Central configuration for the AVP -> Astribot teleoperation pipeline.

Everything that an operator might reasonably want to tune lives here so the
rest of the code stays generic. Joint *ranges* are read from the MuJoCo model
at runtime; the values below are names, home poses, gains and mapping knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Coordinate alignment: AVP world frame -> robot world frame
# --------------------------------------------------------------------------- #
# The AVP streamer reports poses in a Z-up world frame with
#   +X = operator's right, +Y = operator's forward, +Z = up.
# The Astribot world frame (measured from the model) has
#   +X = robot's LEFT, +Y = robot's backward, +Z = up.
# When the operator faces the robot these differ by a 180 deg rotation about Z,
# i.e. diag(-1, -1, +1): left/right and forward/back flip, up stays.
#
# This single rotation matrix maps AVP-world vectors into robot-world vectors.
# If the operator stands *beside* the robot or the mounting changes, override
# this (e.g. a 90 deg Z rotation) instead of abusing position_scale's sign.
def _rotz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# 180 deg about Z == diag(-1, -1, +1). Operator facing the robot.
AVP_TO_ROBOT_R = _rotz(180.0)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)

# Teleop MJCF (adds finger position actuators on top of the stock model).
MJCF_PATH = os.path.join(
    REPO_DIR,
    "astribot_simulation",
    "astribot_descriptions",
    "mjcf",
    "astribot_s1_mjcf",
    "astribot_s1_teleop.xml",
)


# --------------------------------------------------------------------------- #
# Network transport (UDP). The publisher binds nothing; it sends to HOST:PORT.
# The subscriber (sim_teleop) binds HOST:PORT and receives.
# --------------------------------------------------------------------------- #
@dataclass
class NetworkConfig:
    host: str = "127.0.0.1"   # loopback: publisher and sim on the same machine
    port: int = 9870
    recv_timeout_s: float = 0.5


# --------------------------------------------------------------------------- #
# Apple Vision Pro source
# --------------------------------------------------------------------------- #
@dataclass
class AVPConfig:
    # IP address (same network) or 6-char room code (cross network).
    connection_id: str = "10.200.177.142"
    # Which human hand(s) to track: "left" | "right" | "both".
    side: str = "right"
    publish_rate_hz: float = 60.0


# --------------------------------------------------------------------------- #
# Arm: names + a non-singular "home" pose used to seed IK and as the rest pose.
# Joint order MUST match the kinematic chain order in the MJCF.
# --------------------------------------------------------------------------- #
ARM_JOINTS: Dict[str, List[str]] = {
    "right": [f"astribot_arm_right_joint_{i}" for i in range(1, 8)],
    "left": [f"astribot_arm_left_joint_{i}" for i in range(1, 8)],
}

TOOL_BODY: Dict[str, str] = {
    "right": "astribot_arm_right_tool_link",
    "left": "astribot_arm_left_tool_link",
}

# Slightly bent elbow so IK does not start at a fully-extended singularity.
ARM_HOME: Dict[str, List[float]] = {
    "right": [0.0, -0.30, 0.0, 0.80, 0.0, 0.0, 0.0],
    "left": [0.0, -0.30, 0.0, 0.80, 0.0, 0.0, 0.0],
}


# --------------------------------------------------------------------------- #
# Hand: per-finger joint chains. Each finger is driven by a single curl scalar
# in [0, 1] (0 = open, 1 = fully closed). `weight` scales how much of each
# joint's range the curl spans (tip joints curl a bit less for a natural look).
# --------------------------------------------------------------------------- #
@dataclass
class FingerSpec:
    name: str
    # MANO/SMPL-X keypoint indices (in the GeoRT 21-point order) for this finger:
    # [mcp, pip, dip, tip]. Used to estimate curl from the human hand.
    human_chain: List[int]
    # Robot joints to drive, with a per-joint range weight.
    joints: List[Tuple[str, float]]


def _finger_specs(side: str) -> List[FingerSpec]:
    s = side
    return [
        FingerSpec(
            "thumb",
            human_chain=[1, 2, 3, 4],
            joints=[
                (f"{s}_thumb_proximal_joint", 1.0),
                (f"{s}_thumb_distal_joint", 1.0),
                (f"{s}_thumb_tip_joint", 0.8),
            ],
        ),
        FingerSpec(
            "index",
            human_chain=[5, 6, 7, 8],
            joints=[
                (f"{s}_index_proximal_joint", 1.0),
                (f"{s}_index_distal_joint", 1.0),
                (f"{s}_index_tip_joint", 0.8),
            ],
        ),
        FingerSpec(
            "middle",
            human_chain=[9, 10, 11, 12],
            joints=[
                (f"{s}_middle_proximal_joint", 1.0),
                (f"{s}_middle_distal_joint", 1.0),
                (f"{s}_middle_tip_joint", 0.8),
            ],
        ),
        FingerSpec(
            "ring",
            human_chain=[13, 14, 15, 16],
            joints=[
                (f"{s}_ring_proximal_joint", 1.0),
                (f"{s}_ring_distal_joint", 1.0),
                (f"{s}_ring_tip_joint", 0.8),
            ],
        ),
        FingerSpec(
            "pinky",
            human_chain=[17, 18, 19, 20],
            joints=[
                (f"{s}_pinky_proximal_joint", 1.0),
                (f"{s}_pinky_distal_joint", 1.0),
                (f"{s}_pinky_tip_joint", 0.8),
            ],
        ),
    ]


# --------------------------------------------------------------------------- #
# Retargeting knobs
# --------------------------------------------------------------------------- #
@dataclass
class RetargetConfig:
    # --- arm IK ---
    # Which IK solver to use:
    #   "pink"   -> Pinocchio + Pink differential IK (default). Precise,
    #               singularity-robust, posture-regularized (natural elbow).
    #   "mujoco" -> the legacy hand-rolled DLS solver (arm_ik.py). Kept as a
    #               zero-dependency fallback and for A/B comparison.
    ik_backend: str = "pink"
    # Rotation mapping AVP-world vectors into robot-world vectors. Fixes the
    # left/right + forward/back flip when facing the robot. Keep position_scale
    # POSITIVE; use this matrix (not a negative scale) for axis directions.
    align_R: np.ndarray = field(default_factory=lambda: AVP_TO_ROBOT_R.copy())
    # Human wrist translation is scaled before being applied to the robot tool.
    # 1.0 = 1:1 motion. Larger = robot reaches further for the same hand motion.
    position_scale: float = 1 #defalt 1.4
    # Whether to also track wrist orientation (vs position only).
    track_orientation: bool = False
    # Per-step joint change clamp (rad) to keep motion smooth. Shared by both
    # backends.
    max_joint_step: float = 0.05 #default 0.15

    # --- legacy MuJoCo DLS backend ("mujoco") ---
    # Damped least squares damping factor (larger = more stable, less precise).
    dls_damping: float = 0.08
    # IK iterations per control tick and convergence tolerance (metres).
    ik_iters: int = 12
    ik_pos_tol: float = 2e-3

    # --- Pink backend ("pink") ---
    # Task weights: position vs posture trade off tracking precision against
    # staying near ARM_HOME (raise posture_cost for a more "natural" arm that
    # tracks slightly less tightly; raise position_cost for tighter tracking).
    pink_position_cost: float = 10.0
    pink_orientation_cost: float = 1.0   # only used when track_orientation=True
    pink_posture_cost: float = 1e-2
    # Levenberg-Marquardt damping on the frame task; larger = smoother / more
    # stable near singularities, smaller = snappier. This replaces dls_damping.
    pink_lm_damping: float = 1e-3
    # Differential-IK steps integrated per control tick.
    pink_solver_iters: int = 8
    # QP backend used by Pink (installed: quadprog, osqp, proxqp, daqp, scs).
    pink_solver: str = "quadprog"
    # Control period used to integrate Pink velocities (matches the 60 Hz loop).
    control_dt: float = 1.0 / 60.0

    # --- finger curl ---
    # Curl is estimated as 1 - reach, where reach = |tip-mcp| / extended_length.
    # These map the raw reach range to [open, closed]; tune per operator.
    finger_open_reach: float = 0.95   # reach value treated as "fully open"
    finger_closed_reach: float = 0.45  # reach value treated as "fully closed"
    # Exponential smoothing on the final command (0 = none, ->1 = very smooth).
    command_smoothing: float = 0.5


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    # Position actuator gains for the finger joints we add in the teleop MJCF.
    finger_kp: float = 8.0
    render: bool = True
    # How long to hold still at startup so the operator can get into pose.
    startup_settle_s: float = 1.0


@dataclass
class TeleopConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    avp: AVPConfig = field(default_factory=AVPConfig)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @property
    def side(self) -> str:
        return self.avp.side

    @property
    def primary_side(self) -> str:
        """A single concrete side, used by the single-arm convenience
        accessors below. Resolves ``"both"`` to ``"right"`` so those never
        raise; dual-arm code should use :meth:`sides` instead."""
        return "right" if self.avp.side == "both" else self.avp.side

    def sides(self) -> List[str]:
        """The list of hands to drive: ``["left", "right"]`` for ``both``,
        otherwise the single configured side."""
        if self.avp.side == "both":
            return ["left", "right"]
        return [self.avp.side]

    # -- per-side accessors (used by dual-arm code) --------------------------
    def arm_joints_for(self, side: str) -> List[str]:
        return ARM_JOINTS[side]

    def tool_body_for(self, side: str) -> str:
        return TOOL_BODY[side]

    def arm_home_for(self, side: str) -> List[float]:
        return ARM_HOME[side]

    def finger_specs_for(self, side: str) -> List[FingerSpec]:
        return _finger_specs(side)

    # -- single-side convenience (default/single-arm paths) ------------------
    @property
    def arm_joints(self) -> List[str]:
        return ARM_JOINTS[self.primary_side]

    @property
    def tool_body(self) -> str:
        return TOOL_BODY[self.primary_side]

    @property
    def arm_home(self) -> List[float]:
        return ARM_HOME[self.primary_side]

    @property
    def finger_specs(self) -> List[FingerSpec]:
        return _finger_specs(self.primary_side)


def default_config() -> TeleopConfig:
    cfg = TeleopConfig()
    # Allow overriding the AVP connection id without editing code.
    env_ip = os.environ.get("AVP_IP")
    if env_ip:
        cfg.avp.connection_id = env_ip
    return cfg
