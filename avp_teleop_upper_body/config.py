"""Configuration for the upper-body (head + torso + dual-arm) teleop.

Most coordinate/finger/calibration settings are *reused* from the dual-arm
package (:mod:`avp_teleop.config`) so the two stay consistent; this module only
adds what the merged whole-upper-body solver needs: the torso/neck joint names,
the head end-effector frame, the 23-DOF whole-body joint order (chassis + torso
+ neck + both arms) + home pose, and the merged-IK task weights.
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
    "CHASSIS_JOINTS",
    "CHASSIS_IK_JOINT",
    "CHASSIS_BASE_FRAME",
    "TORSO_JOINTS",
    "TORSO_LEAN_JOINTS",
    "NECK_JOINTS",
    "HEAD_FRAME_BODY",
    "CHEST_HEAD_FRAME",
    "CHEST_HIP_FRAME",
    "CHEST_ANKLE_FRAME",
    "BODY_JOINTS",
    "IK_KEEP_JOINTS",
    "BODY_HOME",
    "all_finger_joints",
    "finger_specs",
    "WholeBodyIKConfig",
    "PoseFilterConfig",
    "UpperBodyConfig",
    "default_config",
]


# --------------------------------------------------------------------------- #
# Whole-body kinematic chain (verified against astribot_s1_teleop.xml)
# --------------------------------------------------------------------------- #
# The S1's "lower body" is a 3-DOF wheeled MOBILE BASE, not legs -- there are no
# hip/knee joints in this model. The base is:
#   astribot_chassis_x, _y   : planar translation (PRISMATIC, metres)
#   astribot_chassis_zrot    : base yaw           (REVOLUTE,  rad)
# All three are actuated position servos. Unlocking them turns the merged
# upper-body solver into a true WHOLE-BODY solver: the base can translate/rotate
# to help the hands and head reach, exactly like the arms compensate the torso.
#
# NOTE (Pinocchio): its MJCF parser MERGES these three consecutive joints on one
# body into a single JointModelComposite whose Pinocchio name is
# "Composite_<first-subjoint>", i.e. CHASSIS_IK_JOINT below (nq=nv=3, DOF order
# x, y, yaw -- matching the MuJoCo qpos order). So the *command* side (MuJoCo /
# SimRobot / BODY_JOINTS) names the three DOFs separately, while the *model*
# side (the reduced Pinocchio model / IK_KEEP_JOINTS) names them as one joint.
# WholeBodyIK expands the composite back into its 3 tangent DOFs internally.
CHASSIS_JOINTS: List[str] = [
    "astribot_chassis_x",
    "astribot_chassis_y",
    "astribot_chassis_zrot",
]
CHASSIS_IK_JOINT: str = "Composite_astribot_chassis_x"
# Base link body frame, used as the balance reference: the ComTask keeps the
# whole-robot CoM horizontally above this frame. It rides with the mobile base,
# so translating the base to extend reach moves the balance target too (only
# leaning the trunk is penalised, not driving the base).
CHASSIS_BASE_FRAME: str = "chassis_base"

# torso_link_4 is the SHARED root of both arms AND the head:
#   base -> torso_joint_1..4 -> torso_link_4 -> {head_joint_1,2 ; left arm ; right arm}
# Moving the torso moves both arm bases, which is exactly why the arms must be
# solved *together* with the torso/head in one configuration (see whole_body_ik).
#
# IMPORTANT (verified by kinematic sweep on astribot_s1_teleop.xml): the four
# torso joints are NOT a single "waist". torso_joint_1/2/3 form a 3-link SAGITTAL
# LEAN SPINE -- they are this wheeled robot's hip/knee equivalent: three parallel
# sagittal-PITCH joints stacked ankle/knee/hip (z = 0.217 / 0.597 / 0.987 m). Two
# consequences follow from the parallel axes: (1) the trunk's forward PITCH equals
# the SUM of the three angles, so keeping that sum ~0 stands the trunk upright at
# any squat depth (this is what WholeBodyIKConfig.trunk_upright_cost enforces), and
# (2) leaning them (a large sum) shifts the CoM off the wheel base -- the tip-over
# / over-compensation failure mode. torso_joint_4 is the pure WAIST YAW (no CoM
# shift, no height change -- safe to move freely). Balance is therefore enforced by
# the trunk-upright task (sum -> 0), NOT by per-joint damping: the lean joints must
# stay free to make large INDIVIDUAL moves (whose sum stays 0) to fold into a
# squat. See WholeBodyIKConfig.trunk_upright_cost / damping_cost_lean.
TORSO_JOINTS: List[str] = [f"astribot_torso_joint_{i}" for i in range(1, 5)]
# The sagittal lean spine (hip/knee equivalent), a strict prefix of TORSO_JOINTS;
# torso_joint_4 (the remainder) is the safe waist yaw.
TORSO_LEAN_JOINTS: List[str] = [f"astribot_torso_joint_{i}" for i in range(1, 4)]
NECK_JOINTS: List[str] = ["astribot_head_joint_1", "astribot_head_joint_2"]

# Robot "head" end-effector frame: the head camera body is the natural analogue
# of the AVP headset (eye/camera) pose. It is driven by both neck joints.
HEAD_FRAME_BODY: str = "astribot_head_camera_base_link"

# --- Balance reference frames for the chest-over-ankle task --------------- #
# The chest-over-ankle balance task (see WholeBodyIKConfig.chest_over_ankle_cost)
# approximates the upper-body CoM as the midpoint of the HEAD joint and the HIP
# joint, and keeps its ground projection over the ANKLE joint. These are the
# JOINT-frame names as parsed from the teleop MJCF by Pinocchio (each revolute
# joint yields a like-named frame). The three sagittal lean joints stack
# ankle/knee/hip at z = 0.217 / 0.597 / 0.987 m; the head joint sits at z = 1.455.
CHEST_HEAD_FRAME: str = "astribot_head_joint_1"    # top of the chest segment
CHEST_HIP_FRAME: str = "astribot_torso_joint_3"    # hip (top of the lean spine)
CHEST_ANKLE_FRAME: str = "astribot_torso_joint_1"  # ankle (base of the lean spine)

# The 23 actuated DOFs solved as one body, in a FIXED order used everywhere on
# the *command* side (IK output vector, SimRobot.command_arm, BODY_HOME, the
# per-DOF weight/rate vectors). Order: chassis, torso, neck, L, R. The chassis
# leads because the reduced Pinocchio model places the composite base joint
# first, so this order matches the model's flat tangent (velocity) order 1:1.
BODY_JOINTS: List[str] = (
    CHASSIS_JOINTS + TORSO_JOINTS + NECK_JOINTS
    + ARM_JOINTS["left"] + ARM_JOINTS["right"]
)

# The joints to KEEP unlocked when building the reduced Pinocchio model (every
# other joint is locked at neutral). Identical to BODY_JOINTS except the three
# chassis DOFs collapse into the single composite joint (see note above). Passed
# to WholeBodyIK; expanding it DOF-by-DOF reproduces BODY_JOINTS' 23-slot order.
IK_KEEP_JOINTS: List[str] = (
    [CHASSIS_IK_JOINT] + TORSO_JOINTS + NECK_JOINTS
    + ARM_JOINTS["left"] + ARM_JOINTS["right"]
)

# Home / reference posture for the 23 body joints. chassis + torso + neck rest at
# 0 (base at origin, upright, looking forward) -- this matches the neutral
# configuration the reduced Pinocchio model locks the non-body joints at, keeping
# frames aligned. The posture task pulls toward this home, which also RECENTERS
# the (position-unbounded) base to the origin when the arms no longer need it.
BODY_HOME: List[float] = (
    [0.0] * len(CHASSIS_JOINTS)
    + [0.0] * len(TORSO_JOINTS)
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
    task are solved together over the whole body (chassis + torso + neck + both
    arms). Hands are weighted higher than the head so grasp precision wins when
    they compete for the shared torso DOFs; the posture task keeps the torso
    near upright and the elbows natural (resolves the large redundancy of a
    23-DOF chain tracking <=18 task dimensions).

    Whole-body MOVEMENT PRIORITY is encoded through per-DOF DampingTask costs
    (see ``damping_costs``): a higher velocity cost makes the QP use that joint
    *less*, so it is recruited only when higher-priority joints cannot do the
    job. The intended order, cheapest (moves first) to most expensive (moves
    last), is:

        arms + neck + waist yaw   (upper body does the work first)
        chassis / mobile base     (translate/yaw the base to extend reach)

    so the base translates/yaws to extend reach before recruiting other DOFs.
    Balance / anti-tip is handled separately and directly by a
    :class:`TrunkUprightTask` (see ``trunk_upright_cost``): it penalises the SUM
    of the sagittal lean-joint angles (= trunk pitch) toward zero, keeping the
    trunk vertical at any squat depth while leaving the two remaining spine DOFs
    free to fold into a genuine squat. This is base-invariant (a pure function of
    the lean angles) and replaces the older mass-based :class:`ComTask` (see
    ``com_cost``, now off by default), which could slowly creep the mobile base.
    The posture task then lets the base drift back to home when the arms no longer
    need it.
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
    # torso_neck_max_step / arm_max_step split). The chassis is a mobile base:
    # its x/y DOFs are PRISMATIC (metres -> m/s, m/s^2) while zrot is REVOLUTE
    # (rad/s, rad/s^2). pink.limits apply each cap per tangent DOF regardless of
    # unit, so metric and angular caps coexist in one limit.
    chassis_max_linear_velocity: float = 0.6   # m/s    (base translation x, y)
    chassis_max_yaw_velocity: float = 1.2      # rad/s  (base yaw)
    torso_neck_max_velocity: float = 1.8       # rad/s  (~ old 0.03 rad/tick @60Hz)
    arm_max_velocity: float = 3.0              # rad/s  (~ old 0.05 rad/tick @60Hz)
    chassis_max_linear_acceleration: float = 4.0   # m/s^2
    chassis_max_yaw_acceleration: float = 20.0     # rad/s^2
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
    # the redundant 23-DOF chain has left over. Pink's docs recommend pairing
    # LowAccelerationTask with a DampingTask (the former alone tends to
    # oscillate, as it does not dissipate energy). Set either to 0 to disable
    # that task (e.g. to A/B compare smoothness).
    #
    # NOTE: ``damping_cost`` is the UPPER-BODY (arms + torso + neck) velocity
    # cost. The DampingTask actually receives a per-DOF *vector* (see
    # ``damping_costs``): the chassis DOFs get the much larger
    # ``damping_cost_chassis`` instead, which is what encodes whole-body movement
    # priority (base moves only as a last resort). So this one task does double
    # duty: gentle smoothing on the upper body, strong "don't move me" on the
    # base.
    damping_cost: float = 1e-1        # cost on |v|, upper body (units: s/rad)
    low_accel_cost: float = 1e-1      # cost on |v - v_prev|    (units: s^2/rad)

    # --- Whole-body movement priority (per-DOF DampingTask cost on the base) - #
    # Velocity cost on the 3 chassis DOFs. Much larger than damping_cost so the
    # QP prefers to reach with the arms/torso and only translates/yaws the base
    # when they can't (the base frame Jacobian is ~unit, i.e. the base is
    # "efficient" at moving the hands, so it needs a heavy cost to stay put for
    # in-reach targets). At the default, a normal in-workspace reach moves the
    # base <1 cm while an out-of-reach target recruits it (see check_whole_body).
    # RAISE it (e.g. 40) for an even more reluctant / sluggish base; LOWER it
    # (e.g. 10) for a more eager, responsive base; set it to damping_cost for a
    # "flat" (no-priority) whole body; freeze the base entirely by dropping the
    # chassis joints from BODY_JOINTS. The chassis carries the HIGHEST damping so
    # the base is the last-resort reach DOF; the lean spine no longer sits below it
    # (its anti-tip role moved to trunk_upright_cost, see damping_cost_lean).
    damping_cost_chassis: float = 20.0   # cost on |v| for chassis x, y, yaw

    # --- Lowest tier: the sagittal lean spine (torso_joint_1/2/3) ----------- #
    # These three joints are the robot's hip/knee equivalent (see TORSO_JOINTS
    # note): leaning them shifts the whole-robot CoM horizontally off the wheel
    # base and is the main cause of the over-compensation / tip-forward problem.
    # They get the LARGEST damping cost of all -- below even the base -- so the QP
    # reaches with the arms, yaws the waist, and drives the base around before it
    # ever leans the trunk. A *sustained* target the others cannot satisfy (a real
    # height change, e.g. crouching to reach something low) still forces the lean
    # through, because the persistent frame error outweighs the damping cost; a
    # transient over-reach does not. RAISE it for an even stiffer trunk (more
    # upright, less vertical reach); LOWER it toward damping_cost_chassis to let
    # the trunk lean more readily.
    #
    # Balance is now handled DIRECTLY by trunk_upright_cost (keep the lean angle
    # SUM = trunk pitch near 0) rather than by heavily damping each lean joint, so
    # this reverts to a light value on par with the rest of the upper body. A high
    # value here is actively harmful now: a genuine squat needs the lean joints to
    # make LARGE individual moves (e.g. knee ~ -1.15 rad) while their SUM stays 0
    # (an "accordion" fold that lowers the body but keeps the trunk vertical); a
    # big per-joint damping cost blocks exactly that fold (measured: at 40 a deep
    # squat fails to track by 36-58 mm, at 0.1 it tracks to < 3 mm). The
    # trunk-upright task -- not per-joint damping -- is what now stops the trunk
    # from tipping forward, and it does so without penalising the fold.
    damping_cost_lean: float = 0.1       # cost on |v| for torso_joint_1, 2, 3

    # --- Balance (PRIMARY): keep the chest over the ankle in the ground plane - #
    # ChestOverAnkleTask is the primary anti-tip constraint (confirmed on the real
    # robot to be the necessary one). It approximates the upper-body CoM as the
    # midpoint of the head joint and the hip joint (~the chest) and keeps its
    # GROUND (x, y) projection over the ankle joint (torso_joint_1). Measured on
    # the poses the user cited: the balanced squat [0.56,-1.15,0.65] gives ~0.1 cm
    # chest-ankle offset, while the tip-forward [0.911,-0.47,0.51] gives ~66 cm --
    # so penalising this offset directly and sharply prevents the tip. It uses
    # only joint FRAMES (no inertial data), and because head/hip/ankle all ride
    # with the mobile base, a pure base translation leaves the offset unchanged:
    # base-invariant, so it cannot cause the base-creep the mass ComTask did. It
    # does NOT fight a genuine squat (lowering with the trunk vertical keeps the
    # chest over the ankle). Set to a high weight so balance wins the lean
    # redundancy; 0 disables. Tuned to ~50: chest stays < ~1 cm off the ankle on
    # reaches/squats while end-effector tracking stays sub-cm.
    chest_over_ankle_cost: float = 50.0  # cost on horizontal chest-ankle offset (1/m)
    # Joint frames defining chest (head/hip midpoint) and ankle. Parsed from the
    # teleop MJCF (JOINT frames); see CHEST_* names below.
    # (frame names supplied from config module constants, see sim_teleop.py)

    # --- Balance (SECONDARY): softly bias the trunk upright (lean sum -> 0) ---- #
    # The three sagittal lean joints (torso_joint_1/2/3) are parallel pitch joints
    # stacked ankle/knee/hip, so the trunk's forward PITCH equals the SUM of their
    # angles. A TrunkUprightTask pulls that sum toward 0. After real-robot testing
    # this is DEMOTED to a soft secondary regulariser: the sum only needs to be
    # ROUGHLY 0 (chest_over_ankle_cost does the real balancing), so this just tidies
    # the lean redundancy toward a natural posture. It is base-invariant (a pure
    # function of the lean angles) and leaves the two free spine DOFs available to
    # fold for a squat. RAISE it for a stiffer/straighter trunk, LOWER (or 0) to let
    # the chest-over-ankle task shape the lean freely. Kept LOW (~0.5) so it does
    # not over-constrain the fold.
    trunk_upright_cost: float = 0.5      # cost on trunk pitch (sum of lean angles)

    # --- Balance: keep the CoM over the wheel base (soft constraint) -------- #
    # A ComTask pins the whole-robot centre of mass to a target in the world
    # frame. We only penalise the HORIZONTAL (x, y) CoM offset -- the coordinate
    # that decides tip-over -- and leave the vertical (z) cost at 0 so the robot
    # is still free to raise/lower its CoM (squat) when a task needs it. The
    # target is the CoM's home projection, i.e. "stay balanced over the base",
    # and it rides with the base: when the chassis translates to extend reach,
    # the balance target translates with it (recomputed each solve from the base
    # frame), so leaning is penalised but *walking* the base is not. Cost is a
    # middle weight: well above the posture/damping regularisers (so it actually
    # governs the lean redundancy) but below the end-effector tracking costs (so a
    # target that truly requires shifting weight can still win). Set to 0 to
    # disable the balance task entirely. Tuned to ~3: it trims the residual
    # off-base CoM lean of a forward reach (~4.5 -> ~1.5 cm) without blocking a
    # squat (the squat's CoM naturally stays near the base as the whole trunk
    # folds down over it).
    # DISABLED by default (0): the mass-based ComTask has been superseded by the
    # base-invariant trunk_upright_cost above. It re-aimed its horizontal target
    # at the current base each tick and nulled the arm-mass CoM residual by
    # translating the base, which slowly CREPT the mobile base backward even with
    # no operator input (and it never constrained trunk pitch, so the trunk could
    # still tip forward). Left in the code as a fallback / for A-B comparison; set
    # > 0 to re-enable, but prefer trunk_upright_cost.
    com_cost: float = 0.0                # cost on horizontal CoM error (1/m)
    com_cost_vertical: float = 0.0       # 0 -> vertical CoM (squat height) free
    com_lm_damping: float = 1e-3

    def _assemble(self, chassis_lin, chassis_yaw, lean, waist, neck, arm) -> np.ndarray:
        """Build a per-DOF vector aligned with BODY_JOINTS (length 23).

        Order matches BODY_JOINTS exactly:
            chassis x, y            -> chassis_lin
            chassis yaw             -> chassis_yaw
            torso_joint_1, 2, 3     -> lean   (the sagittal lean spine)
            torso_joint_4           -> waist  (pure waist yaw, safe)
            head_joint_1, 2         -> neck
            left arm (7), right (7) -> arm
        Splitting the four torso joints into ``lean`` (3) and ``waist`` (1) lets
        the destabilising lean spine carry a different movement priority than the
        harmless waist yaw. The chassis splits into linear (x, y) and yaw so the
        mobile base can carry different (and differently-united) caps.
        """
        n_lean = len(TORSO_LEAN_JOINTS)                    # 3
        n_waist = len(TORSO_JOINTS) - n_lean               # 1 (torso_joint_4)
        n_neck = len(NECK_JOINTS)                          # 2
        n_arm = 2 * len(ARM_JOINTS["right"])               # 14
        chassis = [chassis_lin, chassis_lin, chassis_yaw]  # x, y, zrot
        return np.array(
            chassis
            + [lean] * n_lean
            + [waist] * n_waist
            + [neck] * n_neck
            + [arm] * n_arm
        )

    def max_velocity(self) -> np.ndarray:
        """Per-joint velocity cap aligned with BODY_JOINTS order.

        Units are per DOF: chassis x/y in m/s, everything else in rad/s. The
        lean, waist and neck joints share the (slow, heavy) torso/neck cap.
        """
        return self._assemble(
            self.chassis_max_linear_velocity, self.chassis_max_yaw_velocity,
            self.torso_neck_max_velocity, self.torso_neck_max_velocity,
            self.torso_neck_max_velocity, self.arm_max_velocity,
        )

    def max_acceleration(self) -> np.ndarray:
        """Per-joint acceleration cap aligned with BODY_JOINTS order.

        Units are per DOF: chassis x/y in m/s^2, everything else in rad/s^2.
        """
        return self._assemble(
            self.chassis_max_linear_acceleration, self.chassis_max_yaw_acceleration,
            self.torso_neck_max_acceleration, self.torso_neck_max_acceleration,
            self.torso_neck_max_acceleration, self.arm_max_acceleration,
        )

    def damping_costs(self) -> np.ndarray:
        """Per-DOF DampingTask velocity cost aligned with BODY_JOINTS (length 23).

        This is the whole-body movement-priority vector; a higher cost makes the
        QP move that DOF *less* / *later*. Tiers, cheapest (moves first) to most
        expensive (moves last):

            arms + neck + waist yaw + lean   ``damping_cost`` / ``damping_cost_lean``
            chassis / mobile base            ``damping_cost_chassis``  (last resort)

        The lean spine (torso_joint_1/2/3) now shares the light upper-body damping
        (``damping_cost_lean`` defaults to ~``damping_cost``): its ANTI-TIP
        behaviour is enforced directly by the trunk-upright task (sum of lean
        angles -> 0), NOT by heavy per-joint damping. Damping it heavily is
        actively harmful because a genuine squat needs large individual lean moves
        whose SUM stays 0 (an accordion fold), which heavy per-joint damping would
        block. The chassis keeps the highest cost so the base stays a last-resort
        reach DOF.
        """
        return self._assemble(
            self.damping_cost_chassis, self.damping_cost_chassis,
            self.damping_cost_lean, self.damping_cost,
            self.damping_cost, self.damping_cost,
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
