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
    "NEURAL_WAIST_JOINT",
    "NEURAL_PITCH_JOINT",
    "HEAD_FRAME_BODY",
    "CHEST_HEAD_FRAME",
    "CHEST_HIP_FRAME",
    "CHEST_ANKLE_FRAME",
    "WAIST_LINK_BODY",
    "BODY_JOINTS",
    "IK_KEEP_JOINTS",
    "BODY_HOME",
    "all_finger_joints",
    "finger_specs",
    "WholeBodyIKConfig",
    "PoseFilterConfig",
    "EgoPoserConfig",
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
# The pure waist-yaw joint (remainder of TORSO_JOINTS after the lean spine).
# The EgoPoser trunk prior maps the operator's spine axial twist onto this DOF
# (see WholeBodyIKConfig.neural_posture_cost / NeuralPostureTask).
NEURAL_WAIST_JOINT: str = "astribot_torso_joint_4"
# The joint the EgoPoser trunk prior tracks its FORWARD-LEAN (pitch) target on.
# We use the HIP joint (torso_joint_3, top of the sagittal lean spine) ALONE --
# not the full lean sum theta_1+theta_2+theta_3. torso_joint_3 is the one hinge
# that pitches the (rigid) upper trunk over the lower limb, so mapping the
# operator's chest-over-pelvis flexion here is frame-consistent (human spine vs
# pelvis  <->  robot upper trunk vs thigh) and, crucially, leaves torso_joint_1/2
# FREE for ChestOverAnkleTask to counter-rotate for balance -- the prior and the
# balance task then act on near-orthogonal DOFs instead of fighting over the
# same "trunk-over-ground" angle (see NeuralPostureTask). sim_teleop disables the
# trunk-upright pitch regulariser while the prior is active so this hip bias is
# not cancelled back toward a ground-vertical trunk.
NEURAL_PITCH_JOINT: str = "astribot_torso_joint_3"
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
# Waist link (shared root of both arms + head); its local +X is the robot's
# forward heading (world -Y at neutral). Used by the Phase-3 hand-in-front guard
# for the "front" direction, and by the EgoPoser prior anchor in sim_teleop.
WAIST_LINK_BODY: str = "astribot_torso_link_4"

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
    chassis_max_linear_velocity: float = 1.2   # m/s    (base translation x, y) 0.6
    chassis_max_yaw_velocity: float = 1     # rad/s  (base yaw) 1.2
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
    # NOTE: ``damping_cost`` is now the NECK-only velocity cost; the waist yaw, the
    # lean spine and the arms have their OWN knobs -- ``damping_cost_waist``,
    # ``damping_cost_lean`` and ``damping_cost_arm`` -- so the five tiers can differ.
    # The DampingTask
    # actually receives a per-DOF *vector* (see ``damping_costs``): the chassis
    # DOFs get the much larger ``damping_cost_chassis_{linear,yaw}`` instead, which is what
    # encodes whole-body movement priority (base moves only as a last resort). So
    # this one task does double duty: smoothing on the upper body, strong
    # "don't move me" on the base.
    #
    # WHY torso/neck > arm (raised 0.1 -> 2.0): with a FLAT upper-body cost, the
    # least-norm QP spreads a small hand motion across every equally-cheap DOF, so
    # the torso/neck/waist visibly drift along with the hand ("whole body follows a
    # tiny wrist wiggle"). Making the arm the cheapest DOF (``damping_cost_arm``,
    # below) and the torso/neck several times dearer confines small in-reach motions
    # to the arm; the torso/neck are recruited only when the arm alone cannot do it.
    damping_cost: float = 1.0         # cost on |v|, NECK only (units: s/rad)
    # Waist yaw (torso_joint_4) velocity cost. SPLIT OUT of ``damping_cost`` (which
    # now covers only the neck) so the safe waist yaw can be priced independently of
    # the neck. The waist shifts NO CoM and changes NO height (unlike the lean spine),
    # so it is safe to move freely: LOWER this (toward ``damping_cost_arm``) to make
    # the QP rotate the WAIST rather than contort the ARMS on an in-place torso twist
    # (json7/8) -- a reactive-priority fix, NOT a competing target task like the old
    # Phase-2b WaistYawTask. RAISE it toward ``damping_cost`` for a stiffer waist that
    # leaves twisting to the arms. Tuned to 0.5 (from 1.0) after real-robot
    # testing: a lower waist cost turns the waist rather than contorting the arms
    # on an in-place twist, which felt better to the operator.
    damping_cost_waist: float = 0.5   # cost on |v| for torso_joint_4 (waist yaw)
    # Arm (both 7-DOF arms) velocity cost. Kept the LOWEST of the upper body so the
    # QP reaches with the arm first and leaves the torso/neck/base still for small
    # in-reach hand motions. Raise it toward ``damping_cost`` for a stiffer arm that
    # shares more work with the torso; lower it for an even more arm-only response.
    damping_cost_arm: float = 1e-1    # cost on |v| for the 14 arm DOFs (units: s/rad)
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
    # Raised 20 -> 25: at 20 the base's ~unit Jacobian still made it "cheap" enough
    # that small in-reach hand motions (and AVP wrist jitter) slid the base around.
    # 25 (250x the arm's 0.1) keeps it more planted for in-reach targets while still
    # converging to a far target within the control loop (40+ started to lag a step
    # target in the self-checks). The bigger lever against "whole body follows a
    # tiny wrist wiggle" is actually the arm/torso split below (arm cheapest); this
    # just trims residual base drift. Raise cautiously for a more planted base
    # (watch far-reach tracking); lower toward 20 for a more eager base.
    # SPLIT into linear (x, y translation) vs yaw (base turn) because the two are
    # different in unit AND in role: x/y are PRISMATIC (cost is per m/s), yaw is
    # REVOLUTE (cost is per rad/s), so one shared number prices "translate 1 m" and
    # "turn 1 rad" identically even though 1 rad/s ~= 57 deg/s is a fast spin while
    # 1 m/s is a walk. And the roles differ: LOWER the yaw cost to let the base TURN
    # to face a target (an alternative to contorting the arms / the waist on a twist),
    # while keeping the LINEAR cost high so the base stays planted for in-reach
    # targets. The hard velocity caps are already split this way
    # (chassis_max_linear_velocity vs chassis_max_yaw_velocity); this makes the soft
    # damping consistent. Tuned down from 25 after real-robot testing: a less
    # reluctant base (yaw 8, linear 12) turns/walks to carry the arms sooner on a
    # walk/turn (json9-11), which felt better; the Phase-1 dead-zone still keeps it
    # planted while the operator is stationary.
    damping_cost_chassis_linear: float = 12.0  # cost on |v| for chassis x, y (per m/s)
    damping_cost_chassis_yaw: float = 8.0      # cost on |v| for chassis yaw (per rad/s)

    # --- Mobile-base DEAD-ZONE (Phase 1: freeze the base while standing still) - #
    # High chassis damping (above) only makes the base *reluctant*; AVP wrist
    # jitter and near-singular solves can still slide/spin it a little while the
    # operator is not actually walking (the "base spins in place / knee jitter /
    # hand drags the base" failures). The dead-zone HARD-freezes the 3 chassis
    # DOFs (velocity limit -> 0 inside the QP, coordinated: the arms still solve
    # around a pinned base) whenever the operator's HEAD is not translating
    # horizontally, and releases them once it moves. Gating on HORIZONTAL head
    # speed (not vertical) means a pure squat/bend does not unfreeze the base
    # (avoids the bend-induced base drift), while a real step does.
    #
    # Two thresholds give hysteresis so the base does not chatter between
    # frozen/free near a single cutoff: freeze when the (EMA-smoothed) head speed
    # drops below base_freeze_speed, unfreeze when it rises above
    # base_unfreeze_speed (unfreeze > freeze). base_speed_alpha smooths the
    # per-tick head-speed estimate (EMA in (0,1], smaller = smoother/laggier).
    # base_deadzone is the master switch. Applied in sim_teleop via
    # WholeBodyIK.set_base_frozen(); needs enforce_limits=True (the velocity
    # limit is the freeze mechanism). Tune the speeds up if a slow real step
    # fails to release the base, down if wrist motion alone releases it.
    base_deadzone: bool = True
    base_freeze_speed: float = 0.05      # m/s: below -> freeze the base
    base_unfreeze_speed: float = 0.12    # m/s: above -> release the base
    base_speed_alpha: float = 0.3        # EMA smoothing of the head-speed estimate

    # --- Lean-spine DEAD-ZONE (Phase 1b: generalize the base dead-zone) ------ #
    # The Phase-1 base dead-zone (freeze the chassis while the head is not
    # translating horizontally) fixed the in-place base jitter (json1-3). The
    # SAME mode-switching mechanism, gated on a DIFFERENT signal, fixes the
    # in-place KNEE/lean jitter on those same clips: freeze the sagittal lean
    # spine (torso_joint_1/2/3 = ankle/knee/hip pitch) while the operator is not
    # squatting/bending, release it when they are. The gate is head VERTICAL
    # speed (squat detection), orthogonal to the base's horizontal gate (walk
    # detection). Measured separability is ~100x: stationary fine-control clips
    # sit at head-vert p90 ~0.01 m/s while a squat/pickup clip hits p90 ~1.25;
    # walking stays low (~0.10) so it does not false-trigger. Hysteresis (freeze
    # below lean_freeze_speed, release above lean_unfreeze_speed) prevents
    # chattering. Applied in sim_teleop via WholeBodyIK.set_lean_frozen(); needs
    # enforce_limits=True (the velocity limit is the freeze mechanism).
    lean_freeze_speed: float = 0.08      # m/s: head vert speed below -> freeze lean
    lean_unfreeze_speed: float = 0.20    # m/s: head vert speed above -> release lean

    # --- Phase 4: Continuous damping scheduling (chassis yaw × head yaw rate) - #
    # Generalization of the dead-zone mode-switch into CONTINUOUS gain scheduling
    # (Gemini's recommendation). Active ONLY when the base is unfrozen (speed >
    # base_unfreeze_speed) -- the Phase-1 dead-zone handles low-speed/stationary
    # (where AVP noise is worst), and the scheduler handles dynamic intent (fast
    # turning). The chassis-yaw damping continuously LOWERS from the static value
    # (damping_cost_chassis_yaw=8) toward a floor (yaw_schedule_floor=2) as the
    # COMBINED yaw rate rises (hands-head yaw rate + head yaw rate), making the
    # base cheaper to recruit for a turn instead of contorting arms/waist. The
    # combined signal captures both "turn waist" (hands sweep, head static) and
    # "turn body" (head + hands both turn). Damping cost itself is EMA-smoothed
    # (alpha) to avoid QP objective jumps ("shift shock" /换挡冲击). Default OFF
    # until data validates it on clip7/8/11.
    enable_yaw_scheduling: bool = False
    yaw_schedule_floor: float = 2.0       # chassis-yaw damping floor at high yaw rate
    yaw_schedule_rate_low: float = 0.5    # combined yaw rate (rad/s) -> static damping
    yaw_schedule_rate_high: float = 3.0   # combined yaw rate (rad/s) -> floor damping
    yaw_schedule_alpha: float = 0.1       # EMA smoothing of damping cost itself

    # --- Phase 4b: TRANSLATION (xy) continuous scheduling ------------------- #
    # Same idea as the yaw scheduler but on the base translation, gated on head
    # HORIZONTAL speed. Completes the "P/N/D gear" picture for the translation
    # axis: P = frozen (< base_unfreeze_speed), N = unfrozen at static damping
    # (base_unfreeze_speed .. trans_schedule_speed_low), D = walking fast enough
    # that damping ramps down (> trans_schedule_speed_low) so the base carries
    # the arms more eagerly. Deliberately GENTLE (the dead-zone + N gear already
    # work well): damping only eases from damping_cost_chassis_linear=12 toward
    # trans_schedule_floor=8 (67%), unlike the aggressive yaw ramp. Default OFF;
    # kept only if it doesn't worsen hand tracking on the walk clips. Reuses
    # base_speed_alpha for the damping EMA (shift-shock smoothing).
    enable_trans_scheduling: bool = False
    trans_schedule_floor: float = 8.0     # chassis-xy damping floor at high walk speed
    trans_schedule_speed_low: float = 0.20   # head horiz speed (m/s) -> static damping (N->D)
    trans_schedule_speed_high: float = 1.2   # head horiz speed (m/s) -> floor damping

    # --- Phase 4b: independent base-yaw dead-zone (decouple turn from walk) --- #
    # The Phase-1 base dead-zone freezes the WHOLE base (x/y/yaw) on head
    # horizontal speed. That wrongly locks the yaw too when the operator turns
    # IN PLACE (low translation, high turn intent -> base stayed frozen -> the
    # turn failed, clip11). Phase-4b gives the base YAW its OWN dead-zone gated on
    # the COMBINED yaw rate (hands-head + head), orthogonal to the xy dead-zone
    # (head horizontal speed). So an in-place turn releases yaw while xy stays
    # frozen. Only active when enable_yaw_scheduling is on; otherwise yaw mirrors
    # the xy freeze (byte-identical to the pre-4b behaviour). Hysteresis band
    # prevents chatter. (These are the same thresholds the scheduler's rate_low
    # would imply, but a slightly lower freeze point + a release band decouple
    # cleanly from the continuous-damping ramp above.)
    base_yaw_freeze_rate: float = 0.15    # combined yaw rate below -> freeze base yaw
    base_yaw_unfreeze_rate: float = 0.40  # above -> unfreeze base yaw (hysteresis)

    # --- Base FOLLOWS HEAD (Phase 2: drive the base from head motion) -------- #
    # Without this the base moves only REACTIVELY -- it is recruited when a hand
    # FrameTask error beats the chassis damping, so on a big walk/turn the arms
    # twist to reach the target instead of the base driving there (root cause A:
    # json7-11). base_track_cost turns on a BaseTrackingTask that pulls the
    # chassis x/y/yaw toward a per-tick reference which sim_teleop builds from the
    # operator's head HORIZONTAL displacement (walk) + head YAW (turn) since
    # calibration. The base then carries the arms along and the hands stay
    # natural. Composes with the dead-zone: while the base is frozen (head not
    # translating) the reference is held at the current base, so the base only
    # follows once the operator actually walks -- and an in-place torso twist
    # (no head translation, json7/8) keeps the base frozen and is handled by the
    # waist/arms, exactly as intended.
    #
    # base_track_cost: task weight. Above damping_cost_chassis (25, a *velocity*
    # cost) so the base actually tracks its *position* reference, but below the
    # hand FrameTasks (arm_position_cost=10) so grasp precision still wins a
    # genuine conflict. 0 disables (task not built -> byte-identical to Phase 1).
    # base_follow_scale: metres of base motion per metre of head motion (1.0 =
    # 1:1 walking; <1 shrinks the workspace, >1 amplifies). base_lean_drop: if
    # the head drops more than this (m) below its calibration height the
    # horizontal reference is HELD (a forward BEND drops+translates the head but
    # is not a step; only near-level head motion advances the base -- avoids the
    # bend-induced base drift). Yaw is not lean-gated (turning keeps head height).
    # DEFAULT 0 (OFF) after clip evaluation (2026-07-14). Driving the base from
    # HEAD displacement was measured to make hand tracking WORSE on the walk/turn
    # clips (clip9 96->254 mm, clip10 144->510 mm, clip11 305->607 mm): the
    # REACTIVE base (Phase-1 dead-zone + chassis damping) already moves the base
    # to serve the hand FrameTasks, and a head-derived base target disagrees with
    # what the arms need, so base_track_cost just competes with the hands over
    # the shared chassis DOF and drags the base off the hand-optimal spot. The
    # BaseTrackingTask mechanism is kept (verified, base-frozen composes with it)
    # but the head->base *policy* is wrong for translation; the real gap is a
    # robust TURN/heading signal (head-forward yaw is unstable when looking down)
    # -- deferred to a future Phase 2b. Set > 0 only to experiment with an
    # externally-supplied base target (see set_base_target).
    base_track_cost: float = 0.0
    base_follow_scale: float = 1.0
    base_lean_drop: float = 0.08         # m: head drop above which xy-follow pauses

    # --- Waist-yaw FOLLOWS the turn signal (Phase 2b: fix in-place twist) ---- #
    # On an in-place torso twist (json7/8) the operator's head yaws but does not
    # translate, so the dead-zone keeps the base frozen and the arms twist to
    # reach the swept hand targets. This drives the WAIST (torso_joint_4) to turn
    # with the operator instead, via a WaistYawTask that tracks the head's
    # INTERAURAL (left-right) axis yaw since calibration. That axis was measured
    # to stay horizontal even when looking down (mean|z|=0.03 on the deep-squat
    # clip), so atan2 never degenerates -- unlike the gaze axis (Phase-2's bug).
    #
    # turn_follow_scale: the head interaural yaw = neck yaw + torso twist + body
    # heading mixed, and it OVER-estimates true torso twist (turning the head to
    # LOOK also counts), plus the neck FrameTask already tracks head orientation
    # -- so scale < 1 attenuates to avoid over-rotating the waist. Main tuning
    # knob; tune up if the waist under-follows a real twist, down if a head glance
    # rotates the waist. waist_soft_limit: clamp the target below the hard joint
    # limit (+/-1.53 rad) so the waist keeps margin. MUTUALLY EXCLUSIVE with the
    # EgoPoser prior: when --egoposer is on, its NeuralPostureTask owns the waist
    # (cleaner SMPL body-twist signal) and this task is NOT built. 0 disables.
    # DEFAULT 0 (OFF) after clip evaluation (2026-07-15). IK-only replay (no sim
    # divergence, sub-cm tracking) showed driving the waist from head interaural
    # yaw does NOT reduce arm contortion on the in-place-twist clips and slightly
    # hurts: clip7 arm_dev 10.62->11.06, clip8 hand_err 7.6->13.9 mm. Root cause
    # (same as the Phase-2 translation finding): the REACTIVE hands already drive
    # the waist to its hard limit (+/-1.53) on a big twist, so a head-yaw target
    # has no headroom and only COMPETES with the hand FrameTasks. Freeing base
    # yaw instead barely helps (chassis-yaw damping 25 makes the QP prefer
    # contorting the arms). The interaural signal itself is sound (pitch-robust);
    # the head->waist *policy* is the wrong lever. The WaistYawTask mechanism is
    # kept + tested (selfcheck) for a future externally-supplied waist target.
    # Set > 0 to experiment. NOTE: the clip7/8 sim blow-ups were mj_step servo
    # divergence, NOT the IK (which tracks fine) -- a separate stability issue.
    waist_yaw_follow_cost: float = 0.0
    turn_follow_scale: float = 0.8       # head-yaw -> waist-yaw gain (<1: attenuate)
    waist_soft_limit: float = 1.2        # rad: clamp waist target (hard limit 1.53)

    # --- Hand-in-front SOFT GUARD (Phase 3: stop arms crossing behind back) -- #
    # A one-sided penalty (see HandFrontTask) that nudges a hand forward ONLY
    # once it goes behind a margin plane anchored at the pelvis (hip frame) and
    # facing the robot forward. Inert when the hands are in front (the common
    # case), so unlike the head-driven follow tasks it does not compete with the
    # hand FrameTasks in normal use. Targets json9-11 (arms twist/cross behind
    # the back on walk/turn). hand_front_cost: task weight -- keep BELOW
    # arm_position_cost (10) so a genuine reach-behind still wins; it is a
    # null-space bias ("turn to face" over "reach behind"), not a hard wall. 0
    # disables (task not built). hand_front_margin: signed forward offset (m) of
    # the guard plane from the hip; NEGATIVE puts it behind the hip so the
    # natural rest pose (hands measured ~3-4 cm behind the hip on the fine-control
    # clip) is not penalised -- only a gross reach-behind (measured to -1.0 m on
    # the walk/turn clips) is. Base-invariant (hand + pelvis both ride the base).
    # DEFAULT 0 (OFF) after clip evaluation (2026-07-15). IK-only replay showed
    # the guard does NOT fix the json10/11 crossing (cross% 36→36, even with a
    # free base) -- crossing is a LATERAL failure and this guard is a
    # forward/backward penalty (wrong axis) -- and on the reach-behind clips it
    # only marginally cuts behind-depth while HURTING hand tracking (clip7 24→35
    # mm, clip9 349→440 mm), the same competition seen in Phase 2/2b. Root cause
    # (now triply confirmed across Phase 2/2b/3): the hand FrameTargets are
    # anchored to FIXED WORLD points (relative to calibration); when the operator
    # walks/turns and the base doesn't follow, those world-fixed targets land
    # behind/crossed relative to the robot -- the TARGET encodes the failure, so
    # no guard on the robot config can fix it. The real fix is architectural:
    # BODY-RELATIVE hand targets (anchored to a moving torso/pelvis frame), not a
    # symptom guard. Mechanism kept + tested (one-sided, inert in front) for a
    # future body-relative retarget. Set > 0 to experiment.
    hand_front_cost: float = 0.0
    hand_front_margin: float = -0.15     # m: guard plane this far behind the hip

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
    #
    # Raised 0.1 -> 1.0 (matches the torso/neck tier): a small in-reach hand motion
    # was spreading into the lean spine (the trunk visibly leaned along with a tiny
    # wrist wiggle) because the lean shared the arm's rock-bottom cost. At 1.0 the
    # lean is only recruited for a SUSTAINED height change (its persistent frame
    # error outweighs the cost), not a transient reach -- a middle ground, NOT the
    # old 40 that blocked squats (verified: a deep squat still folds ~1.35 rad and
    # tracks < 5 mm at 1.0). TRADEOFF: an intentional squat is slightly less eager
    # (needs a clearer/held low target); lower this toward damping_cost_arm if you
    # want squats to trigger more readily, raise it for a stiffer trunk.
    damping_cost_lean: float = 1.0       # cost on |v| for torso_joint_1, 2, 3

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

    # --- EgoPoser neural trunk-posture prior (OPTIONAL, OFF by default) ------ #
    # Cost of the NeuralPostureTask, which tracks a human trunk posture
    # hallucinated by EgoPoser from the head + hand poses (see the
    # avp_teleop_upper_body.egoposer subpackage). It biases TWO trunk DOFs
    # toward the prior: the trunk PITCH -- tracked on the HIP joint alone
    # (torso_joint_3, see NEURAL_PITCH_JOINT), NOT the full lean sum -- and the
    # WAIST YAW (torso_joint_4). Kept DELIBERATELY LOW -- well below the balance
    # (chest_over_ankle_cost=50) and end-effector (arm=10, head=3) costs -- so
    # the prior only shapes the trunk's null space toward a more natural /
    # anthropomorphic pose and is mathematically overridden whenever it would
    # fight balance or hand precision. This realises the "deep learning for
    # human-likeness, QP math for safety" closed loop: even an aggressive or
    # transiently unstable hallucinated pose cannot tip the robot, because
    # ChestOverAnkleTask outweighs it by ~60x.
    #
    # WHY THE HIP JOINT, NOT THE LEAN SUM: the sum theta_1+theta_2+theta_3 is the
    # trunk pitch relative to the GROUND -- the very quantity ChestOverAnkleTask
    # governs -- so tracking it makes the low-cost prior fight the cost-50 balance
    # task for the same DOF and get attenuated to near-nothing (little human
    # posture survives). torso_joint_3 (the hip) is instead the "upper trunk over
    # lower limb" hinge; tracking the operator's chest-over-pelvis flexion there
    # is frame-consistent AND leaves torso_joint_1/2 free for balance to
    # counter-rotate (ankle/knee sit back to keep the chest over the ankle). The
    # prior and balance then act on near-orthogonal DOFs and both are satisfied:
    # the robot hinges forward at the hip like a person while staying balanced.
    #
    # 0 DISABLES the task entirely (default) -> the solver is byte-for-byte the
    # non-neural pipeline, with no torch dependency touched. Raise toward (but
    # keep below) trunk_upright_cost's neighbours for a stronger human bias;
    # start around 0.8 and tune on hardware. When this is > 0, sim_teleop DISABLES
    # trunk_upright_cost (its sum->0 pull would force torso_joint_1/2 to cancel
    # the hip bias back toward a ground-vertical trunk); balance is left entirely
    # to ChestOverAnkleTask. Requires neural_pitch_joint_names (defaults to the
    # hip via NEURAL_PITCH_JOINT) + neural_waist_joint_name at build time and a
    # per-tick WholeBodyIK.set_neural_target(pitch, yaw).
    neural_posture_cost: float = 0.0     # cost on (trunk pitch, waist yaw) prior error
    # Boosted posture costs when the chassis enters D gear (trans or yaw), to
    # enforce trunk/chassis alignment during locomotion. Trade tracking precision
    # for healthy posture while moving (acceptable: hands need not hit 100% while
    # walking, only when stationary). 0 = no boost (static costs stay). Multipliers:
    posture_boost_d_gear: float = 2.5    # PostureTask (home-pose) *= this in D gear
    neural_boost_d_gear: float = 2.5     # neural_posture_cost *= this in D gear (deprecated waist_boost name kept for compat)
    waist_boost_d_gear: float = 2.5      # DEPRECATED alias for neural_boost_d_gear
    waist_zero_d_gear_cost: float = 0.0  # NEW: dedicated waist→0° task, 0→this in D gear (independent of posture_boost)

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

            arms                     ``damping_cost_arm``      (moves first)
            waist yaw                ``damping_cost_waist``
            neck + lean              ``damping_cost`` / ``damping_cost_lean``
            chassis linear / yaw     ``damping_cost_chassis_{linear,yaw}`` (last resort)

        The arm is the CHEAPEST DOF so a small in-reach hand motion is realised by
        the arm alone; the waist/neck/lean are several times dearer so they stay
        put unless the arm cannot do the job (this fixes "the whole body follows a
        tiny wrist wiggle"). The lean spine (torso_joint_1/2/3) shares the
        torso/neck tier (``damping_cost_lean``); its ANTI-TIP behaviour is enforced
        by the balance tasks (chest-over-ankle / trunk-upright), NOT by this
        damping, but a middling cost keeps a transient reach from leaning the trunk
        while still allowing a SUSTAINED squat. The chassis keeps the highest cost
        so the base stays a last-resort reach DOF.
        """
        return self._assemble(
            self.damping_cost_chassis_linear, self.damping_cost_chassis_yaw,
            self.damping_cost_lean, self.damping_cost_waist,
            self.damping_cost, self.damping_cost_arm,
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

    # Outlier rejection (metres per 60 Hz tick): the largest jump the RAW
    # translation target may make in one frame before it is clamped, applied
    # ahead of the EMA (see PoseFilter). AVP occasionally emits a corrupt pose
    # or resumes after a dropout with the hand already moved far; a single big
    # jump through the near-singular arm Jacobian makes the robot twitch. 0.08 m
    # /tick ~= 4.8 m/s, well above any real hand speed, so only true outliers
    # are clamped and genuine motion just ramps over a few ticks. Set 0 to
    # disable (e.g. to A/B the effect).
    max_translation_jump: float = 0.08


def _default_egoposer_weights() -> str:
    """Absolute path to the bundled EgoPoser checkpoint, or "" if absent.

    Looks for ``egoposer/model_zoo/egoposer.pth`` next to this config module so
    ``--egoposer`` works out of the box once the weights are downloaded there,
    regardless of the current working directory.
    """
    import os

    path = os.path.join(os.path.dirname(__file__), "egoposer", "model_zoo",
                        "egoposer.pth")
    return path if os.path.isfile(path) else ""


@dataclass
class EgoPoserConfig:
    """EgoPoser neural trunk-posture prior (see the ``egoposer`` subpackage).

    OFF by default (``enabled=False``): the teleop then runs exactly as before
    with no torch dependency. When enabled, EgoPoser hallucinates the operator's
    trunk posture from the head + hand poses and the result is fed to the QP as
    a low-cost :class:`NeuralPostureTask` (see
    :attr:`WholeBodyIKConfig.neural_posture_cost`) that biases trunk pitch +
    waist yaw toward a natural human pose while balance / precision dominate.

    ``weights_path`` points at the released EgoPoser checkpoint (a ``.pth`` with
    a ``"params"`` state_dict); download it from the project's Google Drive (see
    :attr:`EgoPoserEstimator.WEIGHTS_DRIVE_URL`) or via
    ``EgoPoserEstimator.download_weights``. If it is missing or torch is not
    installed the estimator reports itself unavailable and the prior silently
    stays off, so teleop never breaks.

    ``pitch_gain`` / ``yaw_gain`` scale (and can sign-flip) the SMPL-spine ->
    robot-trunk mapping; ``max_pitch`` / ``max_yaw`` clamp the target for safety;
    ``alpha`` EMA-smooths the prior across ticks (1.0 = off). ``window_size`` and
    ``fps`` must match the trained model (80 @ 60 Hz for the released weights).
    """

    enabled: bool = False
    # Default to the checkpoint bundled under egoposer/model_zoo/ (resolved
    # absolute at construction, see default_config); empty string -> no default.
    weights_path: str = field(default_factory=lambda: _default_egoposer_weights())
    window_size: int = 80
    fps: float = 60.0
    spatial_normalization: bool = True
    device: str = "cpu"
    pitch_gain: float = 1.0
    yaw_gain: float = 1.0
    max_pitch: float = 1.2
    max_yaw: float = 1.0
    alpha: float = 0.5          # EMA smoothing of the (pitch, yaw) prior in (0, 1]


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
    # EgoPoser neural trunk-posture prior (opt-in; see EgoPoserConfig).
    egoposer: EgoPoserConfig = field(default_factory=EgoPoserConfig)

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
