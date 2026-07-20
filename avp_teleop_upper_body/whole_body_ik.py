"""Merged whole-body inverse kinematics (Pinocchio + Pink).

One differential-IK problem drives the entire body -- a 3-DOF mobile base
(chassis x, y, yaw), 4-DOF torso, 2-DOF neck, and two 7-DOF arms (23 DOF total)
-- by posing weighted tasks in a single Pink configuration:

    * FrameTask(head camera)  -> AVP head 6-DoF target
    * FrameTask(left tool)    -> left-hand target
    * FrameTask(right tool)   -> right-hand target
    * ChestOverAnkleTask      -> PRIMARY balance / anti-tip: keep the upper-body
                                 CoM (head-joint/hip-joint midpoint = chest) over
                                 the ankle joint in the ground plane
    * TrunkUprightTask        -> soft secondary: bias the sagittal lean spine's
                                 angle SUM (= trunk pitch) toward ~0 to tidy the
                                 lean redundancy (low cost; the two free spine
                                 DOFs stay free to fold into a squat)
    * NeuralPostureTask       -> optional EgoPoser prior (OFF by default): bias
                                 the trunk pitch (lean sum) + waist yaw toward a
                                 human posture hallucinated from head/hand poses;
                                 LOW cost, so balance + hand precision override it
    * ComTask(horizontal)     -> optional legacy mass-based balance (OFF by
                                 default; superseded by the two tasks above, which
                                 are base-invariant and do not creep the base)
    * PostureTask(home)       -> resolves the large null space (upright torso,
                                 natural elbows, base recentred at the origin)
    * DampingTask             -> soft penalty on |v|        (bleeds off velocity
                                 AND encodes whole-body movement priority)
    * LowAccelerationTask     -> soft penalty on |v - v_prev| (limits jerk)

Because all three end-effector tasks share one configuration, the arms and the
base **automatically compensate** for one another: when the head target leans
the torso forward, the QP simultaneously re-solves the arms so the hands stay
on their world targets; when a target is beyond the arms' reach, the base
translates/yaws to extend it. This is the merged-solver pattern of
VisionProTeleop's ``11_diffik_aloha.py`` (two arm FrameTasks + posture),
extended with a head task, a mobile base, and realised in Pink instead of mink.

Whole-body MOVEMENT PRIORITY (arms + waist first, then the mobile base, and the
sagittal lean spine -- torso_joint_1/2/3 -- only as a last resort) is encoded
through a **per-DOF DampingTask cost vector**: each tier gets a larger velocity
cost than the one above it, so the QP reaches with the arms/waist whenever it
can, drives the base next, and leans the trunk only when nothing else reaches
the target (the base frame Jacobian is ~unit, so without a heavy cost the base
would actually be the *cheapest* way to move the hands). The lean spine is the
most expensive because leaning it shifts the CoM off the wheels and destabilises
the robot; a :class:`ComTask` reinforces this by penalising horizontal CoM drift
directly while leaving the vertical CoM free (so squatting for a real height
change still works). See ``damping_costs`` / ``com_cost`` in the config.

Physical rate limits are enforced *inside* the QP via :mod:`pink.limits`:
a :class:`VelocityLimit` and an :class:`AccelerationLimit` (plus the usual
:class:`ConfigurationLimit`) bound joint velocity and acceleration as hard
inequality constraints. One ``solve_ik`` is run per control tick, so those
bounds are the true per-tick velocity / acceleration of the robot. This makes
the old "solve hard, then ``np.clip`` the net step" rate limiter obsolete: the
solver now trades rate limits against task tracking inside the optimisation
rather than truncating the answer afterwards. The teleop MJCF declares no
velocity limit, so the caps come from :class:`WholeBodyIKConfig`. The chassis
x/y DOFs are prismatic, so their caps are metric (m/s, m/s^2) while every other
DOF is angular -- pink.limits apply each cap per tangent DOF regardless of unit.
The base joints carry no position limit (unbounded travel); the posture task is
what pulls the base back to the origin when the arms no longer need it.

On top of those *hard* caps, two low-cost *soft* tasks shape the motion within
the caps for smoothness: a :class:`DampingTask` penalizes joint velocity and a
:class:`LowAccelerationTask` penalizes frame-to-frame acceleration. Their upper-
body costs sit far below the tracking costs, so they only smooth the redundant
slack and never visibly fight the end-effector targets. The low-acceleration
task is stateful (it remembers the previous tick's velocity); ``solve`` feeds it
the integrated velocity each tick, and :meth:`reset` clears it on
(re)calibration.

Frame consistency (identical guarantee to :mod:`avp_teleop.retarget.arm_ik_pink`):
the model is built from the *flattened* teleop MJCF, so every task frame
coincides with MuJoCo's to < 1e-4 m and solved joint angles are written to the
simulator with no coordinate transform. All targets are in the MuJoCo world
frame, exactly as produced by :mod:`avp_teleop.retarget.frames`.

Composite base joint: Pinocchio's MJCF parser merges the three chassis joints
into one 3-DOF ``JointModelComposite`` (DOF order x, y, yaw). So the caller
passes the *reduced-model* joint names (composite for the base) while the flat
23-DOF command vector this class returns is in the per-DOF order of
``BODY_JOINTS`` -- the two coincide because the composite's internal DOF order
matches the MuJoCo chassis qpos order. This class expands every kept joint into
its tangent DOFs, so a multi-DOF joint like the base needs no special-casing
downstream.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

import mujoco
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, DampingTask, LowAccelerationTask, ComTask
from pink.tasks.task import Task
from pink.limits import ConfigurationLimit, VelocityLimit, AccelerationLimit
from pink.exceptions import NoSolutionFound


# A per-end target: world position (3,) and optional world rotation (3, 3).
Target = Tuple[np.ndarray, Optional[np.ndarray]]


class TrunkUprightTask(Task):
    r"""Softly bias the sagittal lean spine upright by pulling its angle SUM to 0.

    The S1's three lean joints (torso_joint_1/2/3) are stacked, parallel,
    sagittal-pitch revolute joints -- a planar 3-link chain in the sagittal
    plane. Because their axes are parallel, the trunk's absolute forward PITCH
    is simply the SUM of the three joint angles:

        trunk_pitch  =  theta_1 + theta_2 + theta_3

    (verified on astribot_s1_teleop.xml: joints 1/2/3 all rotate about world x
    at z = 0.217 / 0.597 / 0.987 m -- ankle / knee / hip). So a config whose
    lean angles sum to ~0 stands the trunk upright *at any squat depth*
    (e.g. the standing pose [0,0,0] and the deep-squat [0.56, -1.15, 0.65] both
    sum to ~0), whereas a config that sums large has tipped the trunk forward
    (the [0.911, -0.47, 0.51] failure sums to ~0.95 rad = ~54 deg forward).

    This task is a single scalar error ``e = sum(theta_lean)`` with a CONSTANT
    Jacobian (a row of ones on the three lean tangent DOFs), and it is a pure
    function of the *lean joint angles only* -- base-invariant, so it cannot
    drive the mobile base or cause base creep.

    ROLE (revised after physical testing): this is now a **soft secondary
    regulariser**, not the primary balance constraint. Real-robot testing showed
    that keeping the trunk-pitch sum only *approximately* zero is enough for this
    DOF; the hard balance job -- keeping the robot from tipping -- is done by
    :class:`ChestOverAnkleTask`, which is the sharper, physically correct
    condition (the ground projection of the upper-body CoM must stay over the
    ankle). Two lean configs can share ``sum ~= 0`` yet put the chest in quite
    different places (e.g. [1,-2,1] sums to 0 but is a deep, folded shape), so the
    sum alone under-constrains balance. This task is therefore kept at a LOW cost
    to gently resolve the remaining lean redundancy toward a natural upright
    posture, while ChestOverAnkleTask enforces the actual balance.
    """

    def __init__(self, lean_v_index, lean_q_index, cost, gain=1.0, lm_damping=0.0):
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.lean_v_index = list(lean_v_index)
        self.lean_q_index = list(lean_q_index)

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        return np.array([sum(configuration.q[i] for i in self.lean_q_index)])

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        J = np.zeros((1, configuration.model.nv))
        J[0, self.lean_v_index] = 1.0
        return J

    def __repr__(self) -> str:
        return f"TrunkUprightTask(cost={self.cost})"


class BaseTrackingTask(Task):
    r"""Drive the mobile base (chassis x / y / yaw) toward a commanded target.

    Phase-2 "base follows head": instead of letting the QP recruit the base only
    reactively (when a hand FrameTask error beats the chassis damping -- which on
    a big walk/turn makes the arms twist to reach instead of the base driving
    there), this task pulls the 3 chassis DOFs toward a per-tick reference
    ``(x, y, yaw)`` derived in sim_teleop from the operator's head horizontal
    displacement (walk intent) and head yaw (turn intent). The base then carries
    the arms along, so the hands stay natural.

    Error is the 3-vector ``q[chassis] - target`` (the yaw row wrapped to
    ``(-pi, pi]`` so a target across the +/-pi seam -- e.g. a 360 deg turn --
    does not spin the wrong way), with a CONSTANT identity Jacobian on the three
    chassis tangent DOFs. It references ONLY the chassis joint angles, so it
    touches no other DOF directly (the arms re-solve around wherever the base
    goes via their own FrameTasks).

    Composes with the Phase-1 dead-zone: when the base is frozen
    (``set_base_frozen(True)`` zeroes the chassis velocityLimit) the QP holds
    chassis Delta_q = 0, so this task's error is absorbed by that hard velocity
    constraint inside the (blocked) chassis subspace and cannot leak into the
    arms. sim_teleop additionally holds the target at the current base while
    frozen, so the error is ~0 anyway. Cost is set above the chassis DampingTask
    (so it actually moves the base) but below the hand FrameTasks (so grasp
    precision still wins a genuine conflict); 0 disables (task not built).
    """

    def __init__(self, chassis_v_index, chassis_q_index, cost,
                 gain=1.0, lm_damping=0.0):
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.chassis_v_index = list(chassis_v_index)   # [x, y, yaw] tangent idx
        self.chassis_q_index = list(chassis_q_index)   # [x, y, yaw] qpos idx
        self.target = np.zeros(3)                      # x (m), y (m), yaw (rad)

    def set_target(self, x: float, y: float, yaw: float) -> None:
        self.target = np.array([float(x), float(y), float(yaw)])

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        q = configuration.q
        cur = np.array([q[i] for i in self.chassis_q_index])
        e = cur - self.target
        # Wrap the yaw error into (-pi, pi] so the base turns the short way.
        e[2] = (e[2] + np.pi) % (2.0 * np.pi) - np.pi
        return e

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        J = np.zeros((3, configuration.model.nv))
        for row, iv in enumerate(self.chassis_v_index):
            J[row, iv] = 1.0
        return J

    def __repr__(self) -> str:
        return (f"BaseTrackingTask(cost={self.cost}, "
                f"target={np.round(self.target, 3)})")


class WaistYawTask(Task):
    r"""Drive the waist yaw (torso_joint_4) toward a commanded angle.

    Phase-2b "turn follows head": on an in-place torso twist the operator's head
    yaws but does not translate, so the Phase-1 dead-zone keeps the base frozen
    and the arms would otherwise twist to reach the swept hand targets (json7/8).
    This task turns the WAIST with the operator instead -- it pulls torso_joint_4
    toward a per-tick target that sim_teleop computes from the head's interaural
    (left-right) axis yaw since calibration (pitch-robust: that axis stays
    horizontal even when looking down, unlike the gaze axis).

    Single-DOF: error ``q[waist] - target`` (yaw-wrapped to (-pi, pi]) with a
    constant Jacobian (one 1.0 on the waist tangent DOF) -- same shape as the
    waist row of :class:`NeuralPostureTask`. The waist has a hard joint limit
    (+/-1.53 rad); sim_teleop clamps the target to a soft limit below that.

    MUTUALLY EXCLUSIVE with NeuralPostureTask on the waist: both would target
    torso_joint_4. WholeBodyIK builds this ONLY when neural_posture_cost <= 0
    (EgoPoser off); when EgoPoser is on its NeuralPostureTask owns the waist (it
    has the cleaner SMPL body-twist signal). Cost 0 disables (task not built).
    """

    def __init__(self, waist_v_index, waist_q_index, cost,
                 gain=1.0, lm_damping=0.0):
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.waist_v_index = int(waist_v_index)
        self.waist_q_index = int(waist_q_index)
        self.yaw_target = 0.0

    def set_target(self, yaw: float) -> None:
        self.yaw_target = float(yaw)

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        e = configuration.q[self.waist_q_index] - self.yaw_target
        e = (e + np.pi) % (2.0 * np.pi) - np.pi
        return np.array([e])

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        J = np.zeros((1, configuration.model.nv))
        J[0, self.waist_v_index] = 1.0
        return J

    def __repr__(self) -> str:
        return f"WaistYawTask(cost={self.cost}, target={self.yaw_target:.3f})"


class NeuralPostureTask(Task):
    r"""Track a neural (EgoPoser) trunk-posture prior at low cost.

    This is the QP-side of the EgoPoser kinematic prior (see
    :mod:`avp_teleop_upper_body.egoposer`). Where :class:`TrunkUprightTask`
    biases the trunk toward *upright* (lean-angle sum -> 0), this task biases it
    toward the posture a human operator would actually adopt for the current
    head/hand configuration, as hallucinated by EgoPoser and retargeted to the
    robot's two realisable trunk DOFs:

        * TRUNK PITCH -- the forward lean, tracked on the ``pitch`` joint set
          supplied by the caller (see ``pitch_q_index``). The default wiring
          uses ONLY the HIP joint (torso_joint_3), so the pitch target is the
          human's chest-relative-to-pelvis flexion applied at the one hinge that
          is anatomically "upper-body over lower-body" -- a frame-consistent
          map (human spine vs pelvis  <->  robot upper trunk vs thigh). This is
          deliberately NOT the full lean sum theta_1+theta_2+theta_3 (= trunk
          pitch relative to the GROUND): tracking the sum makes the prior fight
          :class:`ChestOverAnkleTask` for the very same "trunk-over-ground"
          quantity, so balance (cost 50) simply attenuates the low-cost prior
          (~0.8) and little human posture survives. Tracking the hip joint alone
          leaves theta_1/theta_2 FREE for the balance task to counter-rotate
          (ankle/knee sit back to keep the chest over the ankle), so the prior
          and balance act on near-orthogonal directions and BOTH can be
          satisfied: the robot hinges forward at the hip like a person while
          staying balanced. (Because the S1 has no pitch DOF above the hip, this
          "hip hinge" is the only way it can realise a forward lean at all.)
        * WAIST YAW   -- axial twist, realised by torso_joint_4.

    The error is the 2-vector

        e = [ sum(theta_pitch) - pitch_target ,  theta_waist - yaw_target ]

    where ``theta_pitch`` is the (usually single-joint) pitch set. The Jacobian
    is CONSTANT (a row of ones on the pitch tangent DOFs, and a single one on the
    waist tangent DOF). Like the other trunk tasks it is a pure function of the
    joint angles, hence **base-invariant** -- it cannot drive or creep the
    mobile base.

    It is deliberately run at a LOW cost, well below the balance
    (:class:`ChestOverAnkleTask`) and end-effector tracking costs, so it only
    shapes the trunk's null-space toward a more human posture and is overridden
    whenever it would fight balance or hand precision. The target is refreshed
    each control tick via :meth:`WholeBodyIK.set_neural_target`; before any
    target is set (or when the estimator yields nothing) it defaults to the
    upright posture (0, 0), which is safe.

    NOTE on TrunkUprightTask: with the hip-only pitch map the two tasks no
    longer fight over the same quantity (upright pulls the SUM to 0; this pulls
    the HIP to the target). Still, to let the hip bias survive, WholeBodyIK's
    caller (sim_teleop) DISABLES the trunk-upright pitch regulariser while the
    neural prior is active -- otherwise upright would force theta_1+theta_2 to
    cancel the hip angle back toward a ground-vertical trunk.
    """

    def __init__(self, pitch_v_index, pitch_q_index, waist_v_index, waist_q_index,
                 cost, gain=1.0, lm_damping=0.0):
        # cost may be a scalar (applied to both rows) -> Pink expands it.
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        # pitch_*_index: the joint DOF(s) that realise the tracked forward lean.
        # A single index (hip only) is the default; a multi-index set sums to a
        # "trunk pitch relative to ground" as the legacy behaviour did.
        self.pitch_v_index = list(pitch_v_index)
        self.pitch_q_index = list(pitch_q_index)
        self.waist_v_index = int(waist_v_index)
        self.waist_q_index = int(waist_q_index)
        self.pitch_target = 0.0
        self.yaw_target = 0.0

    def set_target(self, pitch: float, yaw: float) -> None:
        self.pitch_target = float(pitch)
        self.yaw_target = float(yaw)

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        pitch = sum(configuration.q[i] for i in self.pitch_q_index)
        waist = configuration.q[self.waist_q_index]
        return np.array([pitch - self.pitch_target,
                         waist - self.yaw_target])

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        J = np.zeros((2, configuration.model.nv))
        J[0, self.pitch_v_index] = 1.0    # pitch row: the pitch DOF(s) (hip only)
        J[1, self.waist_v_index] = 1.0    # yaw row: the waist DOF
        return J

    def __repr__(self) -> str:
        return (f"NeuralPostureTask(cost={self.cost}, "
                f"pitch={self.pitch_target:.3f}, yaw={self.yaw_target:.3f})")


class ChestOverAnkleTask(Task):
    r"""Balance task: keep the upper-body CoM projection over the ankle.

    This is the primary anti-tip constraint (confirmed on the real robot). We
    approximate the upper-body centre of mass by the **midpoint of the head joint
    and the hip joint** -- i.e. roughly the chest -- and require its GROUND
    (horizontal xy) projection to stay above the ground projection of the ANKLE
    joint (torso_joint_1, the base of the sagittal lean spine):

        chest = 0.5 * (p_head_joint + p_hip_joint)
        error = (chest - p_ankle_joint)[:2]      # a 2-vector in the ground plane

    When the trunk leans forward, the chest projection races out ahead of the
    ankle (measured: the balanced squat [0.56,-1.15,0.65] gives ~0.1 cm offset,
    while the tipped [0.911,-0.47,0.51] gives ~66 cm), so penalising this offset
    directly prevents the tip-forward failure. Unlike a mass-based
    :class:`ComTask`, it references only the head/hip/ankle *joint frames* -- no
    inertial data -- and, because all three frames ride together with the mobile
    base, a pure base translation moves chest and ankle by the same amount and
    leaves the error unchanged: it is **base-invariant** and cannot cause the base
    creep the ComTask did. It also does not fight a genuine squat: lowering the
    body while the trunk stays vertical keeps the chest above the ankle, so the
    offset stays ~0 at any height.

    The default reference names the three JOINT frames as parsed from the teleop
    MJCF: ``astribot_head_joint_1`` (head), ``astribot_torso_joint_3`` (hip) and
    ``astribot_torso_joint_1`` (ankle).
    """

    def __init__(self, head_frame_id, hip_frame_id, ankle_frame_id, cost,
                 gain=1.0, lm_damping=0.0):
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.head_frame_id = head_frame_id
        self.hip_frame_id = hip_frame_id
        self.ankle_frame_id = ankle_frame_id

    def _p(self, configuration: pink.Configuration, fid: int) -> np.ndarray:
        return configuration.data.oMf[fid].translation

    def _Jw(self, configuration: pink.Configuration, fid: int) -> np.ndarray:
        # LOCAL_WORLD_ALIGNED -> the linear block is the world-frame translational
        # Jacobian of the frame origin (frame placements/Jacobians were refreshed
        # by Configuration.update()).
        return pin.getFrameJacobian(
            configuration.model, configuration.data, fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        chest = 0.5 * (self._p(configuration, self.head_frame_id)
                       + self._p(configuration, self.hip_frame_id))
        ankle = self._p(configuration, self.ankle_frame_id)
        return (chest - ankle)[:2]

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        J = (0.5 * (self._Jw(configuration, self.head_frame_id)
                    + self._Jw(configuration, self.hip_frame_id))
             - self._Jw(configuration, self.ankle_frame_id))
        return J[:2]

    def __repr__(self) -> str:
        return f"ChestOverAnkleTask(cost={self.cost})"


class HandFrontTask(Task):
    r"""Soft ONE-SIDED guard: keep each hand IN FRONT of the pelvis.

    On big walk/turn motions the arms can end up reaching / crossing BEHIND the
    robot (json9-11: "arms twist crossed behind the back"). This task nudges a
    hand forward, but ONLY once it goes behind a margin plane anchored at the
    pelvis and facing the robot's forward -- so it is completely INERT whenever
    the hands are in front (the common case), unlike a two-sided FrameTask or the
    head-driven follow tasks (which impose a target every tick and compete with
    the hand FrameTasks). It is a null-space bias, deliberately kept BELOW the
    hand FrameTask cost, so a genuine reach-behind still wins -- it only prefers
    "turn to face the target" over "reach behind" when both track equally.

    Geometry (per hand, base-invariant -- both operands ride with the base):
        forward = waist_link local +X in world (robot heading; world -Y at
                  neutral), taken as a constant plane normal for this tick
        s = (p_hand - p_pelvis) . forward         # signed forward offset (m)
        error_i = min(0, s - margin)              # 0 when in front of the margin
    A NEGATIVE margin places the plane slightly BEHIND the hip, so the natural
    rest pose (hands a few cm behind the hip joint) is not penalised; only a
    gross reach-behind is. The Jacobian row is ``forward . (J_hand - J_pelvis)``
    when active, else 0 (the task drops out of the QP for that hand).
    """

    def __init__(self, left_id, right_id, pelvis_id, waist_id, cost,
                 margin=-0.15, gain=1.0, lm_damping=0.0):
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.left_id = left_id
        self.right_id = right_id
        self.pelvis_id = pelvis_id
        self.waist_id = waist_id
        self.margin = float(margin)

    def _p(self, configuration, fid):
        return configuration.data.oMf[fid].translation

    def _Jw(self, configuration, fid):
        return pin.getFrameJacobian(
            configuration.model, configuration.data, fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]

    def _forward(self, configuration) -> np.ndarray:
        # Robot forward = waist link local +X in world (see _prior_anchor_pose).
        return configuration.data.oMf[self.waist_id].rotation[:, 0].copy()

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        fwd = self._forward(configuration)
        pelvis = self._p(configuration, self.pelvis_id)
        e = np.zeros(2)
        for row, fid in enumerate((self.left_id, self.right_id)):
            s = float(np.dot(self._p(configuration, fid) - pelvis, fwd))
            e[row] = min(0.0, s - self.margin)     # one-sided: 0 when in front
        return e

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        fwd = self._forward(configuration)
        pelvis = self._p(configuration, self.pelvis_id)
        Jp = self._Jw(configuration, self.pelvis_id)
        J = np.zeros((2, configuration.model.nv))
        for row, fid in enumerate((self.left_id, self.right_id)):
            s = float(np.dot(self._p(configuration, fid) - pelvis, fwd))
            if s - self.margin < 0.0:              # active only when behind
                J[row] = fwd @ (self._Jw(configuration, fid) - Jp)
        return J

    def __repr__(self) -> str:
        return f"HandFrontTask(cost={self.cost}, margin={self.margin})"


def _flatten_mjcf(mjcf_path: str) -> str:
    """Resolve all <include>s into a single self-contained XML on disk.

    Pinocchio's MJCF parser does not follow MuJoCo ``<include>`` directives;
    loading in MuJoCo and re-saving yields a flat file Pinocchio can read and
    guarantees byte-identical kinematics with the simulator.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    fd, flat_path = tempfile.mkstemp(prefix="astribot_upperbody_", suffix=".xml")
    os.close(fd)
    mujoco.mj_saveLastXML(flat_path, model)
    return flat_path


class WholeBodyIK:
    """Differential IK for the 23-DOF body (mobile base + torso + neck + arms)."""

    def __init__(
        self,
        mjcf_path: str,
        body_joint_names: Sequence[str],
        head_frame_name: str,
        left_tool_name: str,
        right_tool_name: str,
        home: np.ndarray,
        *,
        dof_names: Optional[Sequence[str]] = None,
        arm_position_cost: float = 10.0,
        arm_orientation_cost: float = 1.0,
        head_position_cost: float = 3.0,
        head_orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        lm_damping: float = 1e-3,
        damping_cost: np.ndarray | float = 1e-1,
        low_accel_cost: float = 1e-1,
        com_cost: float = 0.0,
        com_cost_vertical: float = 0.0,
        com_lm_damping: float = 1e-3,
        base_frame_name: Optional[str] = None,
        trunk_upright_cost: float = 0.0,
        trunk_lean_joint_names: Optional[Sequence[str]] = None,
        chest_over_ankle_cost: float = 0.0,
        chest_head_frame_name: Optional[str] = None,
        chest_hip_frame_name: Optional[str] = None,
        chest_ankle_frame_name: Optional[str] = None,
        neural_posture_cost: float = 0.0,
        neural_waist_joint_name: Optional[str] = None,
        neural_pitch_joint_names: Optional[Sequence[str]] = None,
        max_velocity: np.ndarray | float = 3.0,
        max_acceleration: np.ndarray | float = 100.0,
        config_limit_gain: float = 0.5,
        enforce_limits: bool = True,
        control_dt: float = 1.0 / 60.0,
        solver: str = "quadprog",
        max_joint_step: np.ndarray | float | None = None,
        base_joint_name: Optional[str] = None,
        base_track_cost: float = 0.0,
        waist_yaw_follow_cost: float = 0.0,
        waist_zero_d_gear_cost: float = 0.0,
        waist_joint_name: Optional[str] = None,
        hand_front_cost: float = 0.0,
        hand_front_margin: float = -0.15,
        hand_front_pelvis_frame_name: Optional[str] = None,
        hand_front_waist_frame_name: Optional[str] = None,
    ):
        self.head_frame_name = head_frame_name
        self.left_tool_name = left_tool_name
        self.right_tool_name = right_tool_name
        self.control_dt = float(control_dt)
        self.solver = solver
        self.enforce_limits = bool(enforce_limits)
        self.home = np.asarray(home, dtype=np.float64).copy()

        # --- build a Pinocchio model from the *flattened* MJCF ---
        flat_path = _flatten_mjcf(mjcf_path)
        try:
            full = pin.buildModelFromMJCF(flat_path)
        finally:
            try:
                os.remove(flat_path)
            except OSError:
                pass

        body_joint_names = list(body_joint_names)
        missing = [n for n in body_joint_names if not full.existJointName(n)]
        if missing:
            raise ValueError(f"Body joints not found in MJCF model: {missing}")
        for fr in (head_frame_name, left_tool_name, right_tool_name):
            if not full.existFrame(fr):
                raise ValueError(f"Frame '{fr}' not in MJCF model.")

        # Lock every joint except the body joints (fingers, and anything else
        # not listed) at neutral -- matching the simulator's posture for those.
        # body_joint_names are *reduced-model* joint names: the mobile base is
        # one composite joint here (3 DOFs), the rest are single-DOF joints.
        lock_ids = [
            jid
            for jid in range(1, full.njoints)  # 0 is the universe joint
            if full.names[jid] not in body_joint_names
        ]
        self.model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self.data = self.model.createData()

        # Map body joints -> reduced-model qpos / tangent indices, EXPANDING each
        # joint into all of its DOFs. A single-DOF joint contributes one index; a
        # multi-DOF joint (the composite mobile base, 3 DOFs) contributes a run.
        # The flattened order (kept-joint order, each expanded in DOF order) is
        # the canonical body-vector order used by solve()/home/limits/costs.
        self._q_index: List[int] = []
        self._v_index: List[int] = []
        for name in body_joint_names:
            jid = self.model.getJointId(name)
            iq, nq_j = self.model.idx_qs[jid], self.model.nqs[jid]
            iv, nv_j = self.model.idx_vs[jid], self.model.nvs[jid]
            self._q_index.extend(range(iq, iq + nq_j))
            self._v_index.extend(range(iv, iv + nv_j))
        self.n = len(self._q_index)
        if len(self._v_index) != self.n:
            # Would only happen for joints whose nq != nv (ball/free); we have
            # none, and the per-DOF body vectors assume a 1:1 q<->v layout.
            raise ValueError(
                f"Body joints have mismatched position/velocity DOFs "
                f"({self.n} vs {len(self._v_index)}); unsupported joint type."
            )
        # Optional sanity check: the caller's per-DOF command names (e.g.
        # BODY_JOINTS, with the base split into x/y/yaw) must line up 1:1 with
        # the expanded DOFs. Catches a composite/ordering mismatch early.
        if dof_names is not None and len(dof_names) != self.n:
            raise ValueError(
                f"dof_names has {len(dof_names)} entries but the model expands "
                f"to {self.n} body DOFs."
            )

        self.lower = np.array([self.model.lowerPositionLimit[i] for i in self._q_index])
        self.upper = np.array([self.model.upperPositionLimit[i] for i in self._q_index])

        # --- hard velocity / acceleration limits (QP inequalities) ----------- #
        # The MJCF carries no velocity limit, so inject ours into the model
        # (VelocityLimit reads model.velocityLimit) and pass a_max to
        # AccelerationLimit. Both are restricted to the body joints; every other
        # (locked) joint is gone from the reduced model. Caps are per tangent DOF
        # (chassis x/y in m/s, everything else in rad/s), placed by _v_index so
        # they land on the right DOFs regardless of joint ordering.
        self.max_velocity = self._as_body_vector(max_velocity, "max_velocity")
        self.max_acceleration = self._as_body_vector(max_acceleration, "max_acceleration")
        v_tan = np.full(self.model.nv, np.inf)
        a_tan = np.full(self.model.nv, np.inf)
        for iv, vmax, amax in zip(self._v_index, self.max_velocity, self.max_acceleration):
            v_tan[iv] = vmax
            a_tan[iv] = amax
        self.model.velocityLimit = v_tan
        # Kept for the Phase-0 output guard in solve(): a full-tangent per-DOF
        # velocity cap (inf on the locked DOFs) to clamp a degenerate QP result
        # before it is integrated (see solve()).
        self._v_tan = v_tan.copy()

        # --- mobile-base live freeze (Phase 1 base dead-zone) ---------------- #
        # Resolve the tangent DOFs of the mobile base so set_base_frozen() can
        # pin them inside the QP by zeroing their velocity limit for a tick.
        # base_joint_name is the reduced-model composite base joint
        # (CHASSIS_IK_JOINT); if omitted, the freeze API is inert. Pink's
        # VelocityLimit reads model.velocityLimit live each solve, so toggling
        # these entries takes effect on the very next solve() without rebuilding
        # anything. We also mirror the change into _v_tan so the Phase-0 output
        # clamp stays consistent (frozen DOFs clamp to 0).
        self._base_v_index: List[int] = []
        self._base_q_index: List[int] = []
        self._base_v_cap: np.ndarray = np.zeros(0)
        # Phase-4b: xy (translation) and yaw (turn) freeze independently, driven
        # by orthogonal intent signals (head horizontal speed vs combined yaw
        # rate). Two separate state bits; set_base_frozen() sets both at once
        # (backward-compatible convenience).
        self._base_xy_frozen = False
        self._base_yaw_frozen = False
        if base_joint_name is not None:
            if not self.model.existJointName(base_joint_name):
                raise ValueError(f"Base joint '{base_joint_name}' not in model.")
            bjid = self.model.getJointId(base_joint_name)
            biv, bnv = self.model.idx_vs[bjid], self.model.nvs[bjid]
            biq, bnq = self.model.idx_qs[bjid], self.model.nqs[bjid]
            self._base_v_index = list(range(biv, biv + bnv))
            self._base_q_index = list(range(biq, biq + bnq))
            self._base_v_cap = self.model.velocityLimit[self._base_v_index].copy()

        # Resolve the tangent DOFs of the trunk lean spine (torso_joint_1/2/3:
        # ankle/knee/hip pitch) so set_lean_frozen() can pin them -- Phase-1b
        # dead-zone, gated on head vertical speed (squat detection) in sim_teleop.
        # Same pattern as the base freeze above. Uses the already-passed
        # trunk_lean_joint_names (also used by TrunkUprightTask).
        self._lean_v_index: List[int] = []
        self._lean_v_cap: np.ndarray = np.zeros(0)
        self._lean_frozen = False
        if trunk_lean_joint_names:
            lean_v = []
            for nm in trunk_lean_joint_names:
                if not self.model.existJointName(nm):
                    raise ValueError(f"Lean joint '{nm}' not in model.")
                jid = self.model.getJointId(nm)
                iv, nv = self.model.idx_vs[jid], self.model.nvs[jid]
                lean_v.extend(range(iv, iv + nv))
            self._lean_v_index = lean_v
            self._lean_v_cap = self.model.velocityLimit[self._lean_v_index].copy()

        self.config_limit = ConfigurationLimit(self.model, config_limit_gain=config_limit_gain)
        self.velocity_limit = VelocityLimit(self.model)
        self.acceleration_limit = AccelerationLimit(self.model, a_tan)
        self._limits = [self.config_limit, self.velocity_limit, self.acceleration_limit]

        # Legacy manual per-tick step clamp. Retired by default (the QP velocity
        # limit does this physically); kept as an escape hatch / for the
        # enforce_limits=False fallback path.
        self._max_joint_step = (
            None if max_joint_step is None
            else self._as_body_vector(max_joint_step, "max_joint_step")
        )

        # --- tasks (one configuration, solved jointly) ---
        self.head_task = FrameTask(
            head_frame_name,
            position_cost=head_position_cost,
            orientation_cost=head_orientation_cost,
            lm_damping=lm_damping,
        )
        self.left_task = FrameTask(
            left_tool_name,
            position_cost=arm_position_cost,
            orientation_cost=arm_orientation_cost,
            lm_damping=lm_damping,
        )
        self.right_task = FrameTask(
            right_tool_name,
            position_cost=arm_position_cost,
            orientation_cost=arm_orientation_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = PostureTask(cost=posture_cost)
        self.posture_task.set_target(self._to_model_q(self.home))

        # Soft smoothing / movement-priority task. DampingTask penalizes |v|;
        # its cost may be a per-DOF vector (this is how whole-body priority is
        # encoded -- big cost on the base, small on the upper body). Included
        # only when some cost is > 0, so it can be A/B-disabled. Pink needs the
        # cost as a full-tangent vector (or a Python float), so we scatter the
        # per-body-DOF costs into a length-nv array by _v_index; unmentioned
        # (locked) DOFs stay 0.
        self.damping_task = self._make_damping_task(damping_cost)
        # LowAccelerationTask (stateful): solve() feeds it the integrated
        # velocity each tick and reset() clears it (see below).
        self.low_accel_task = (
            LowAccelerationTask(cost=low_accel_cost) if low_accel_cost > 0 else None
        )

        # Balance task: keep the whole-robot CoM over the wheel base. We penalise
        # only the horizontal (x, y) CoM offset -- the coordinate that governs
        # tip-over -- and leave the vertical cost at ``com_cost_vertical`` (0 by
        # default) so the robot may still raise/lower its CoM (squat) freely. The
        # ComTask error is in the WORLD frame, so a fixed world target would fight
        # the base whenever it translates to extend reach; instead solve() re-aims
        # the target at the current base position each tick, i.e. "stay balanced
        # over wherever the base is" -- leaning is penalised, driving the base is
        # not. base_frame_name selects the frame the CoM is kept above (the
        # chassis body); its horizontal position is the balance reference.
        self.com_task = None
        self._com_cost_vec = np.array([com_cost, com_cost, com_cost_vertical],
                                      dtype=np.float64)
        self.base_frame_name = base_frame_name
        if np.any(self._com_cost_vec > 0):
            if base_frame_name is None:
                raise ValueError(
                    "com_cost > 0 requires base_frame_name (the frame to keep "
                    "the centre of mass horizontally above)."
                )
            if not self.model.existFrame(base_frame_name):
                raise ValueError(f"Base frame '{base_frame_name}' not in model.")
            # cost must be a vector for anisotropic (x, y free-z) weighting.
            self.com_task = ComTask(cost=self._com_cost_vec.copy(),
                                    lm_damping=com_lm_damping)

        # Balance / anti-tip task: keep the sagittal lean spine vertical by
        # penalising the SUM of the lean joint angles (trunk pitch) toward zero.
        # This is the primary balance mechanism (see TrunkUprightTask): unlike the
        # mass-based ComTask it is a pure function of the lean joint angles, so it
        # is base-invariant and cannot creep the mobile base. Included only when
        # trunk_upright_cost > 0 and the lean joints are named.
        # Secondary regulariser: softly bias the sagittal lean spine toward
        # "upright" by pulling the SUM of the lean angles (trunk pitch) toward 0.
        # This only needs to be APPROXIMATELY zero (per real-robot testing), so it
        # runs at a low cost and merely tidies the lean redundancy; the ACTUAL
        # balance / anti-tip job is done by the chest-over-ankle task below. Like
        # that task it is base-invariant (a pure function of the lean angles).
        # Included only when trunk_upright_cost > 0 and the lean joints are named.
        self.trunk_task = None
        if trunk_upright_cost > 0:
            if not trunk_lean_joint_names:
                raise ValueError(
                    "trunk_upright_cost > 0 requires trunk_lean_joint_names "
                    "(the sagittal lean joints whose angle sum is trunk pitch)."
                )
            lean_v, lean_q = [], []
            for nm in trunk_lean_joint_names:
                if not self.model.existJointName(nm):
                    raise ValueError(f"Trunk lean joint '{nm}' not in model.")
                jid = self.model.getJointId(nm)
                lean_v.append(int(self.model.idx_vs[jid]))
                lean_q.append(int(self.model.idx_qs[jid]))
            self.trunk_task = TrunkUprightTask(lean_v, lean_q, cost=trunk_upright_cost)

        # PRIMARY balance / anti-tip task: keep the upper-body CoM (approximated by
        # the head-joint / hip-joint midpoint = the chest) over the ankle joint in
        # the ground plane. This is the physically correct anti-tip condition
        # (confirmed on the real robot) and, being a function of joint frame
        # positions only, is base-invariant so it cannot creep the mobile base.
        # Included only when chest_over_ankle_cost > 0 and the frames are named.
        self.chest_task = None
        if chest_over_ankle_cost > 0:
            missing = [n for n in (chest_head_frame_name, chest_hip_frame_name,
                                   chest_ankle_frame_name) if not n]
            if missing:
                raise ValueError(
                    "chest_over_ankle_cost > 0 requires chest_head_frame_name, "
                    "chest_hip_frame_name and chest_ankle_frame_name."
                )
            fids = {}
            for role, nm in (("head", chest_head_frame_name),
                             ("hip", chest_hip_frame_name),
                             ("ankle", chest_ankle_frame_name)):
                if not self.model.existFrame(nm):
                    raise ValueError(f"Chest-over-ankle {role} frame '{nm}' "
                                     f"not in model.")
                fids[role] = self.model.getFrameId(nm)
            self.chest_task = ChestOverAnkleTask(
                fids["head"], fids["hip"], fids["ankle"],
                cost=chest_over_ankle_cost)

        # EgoPoser trunk-posture prior (soft, LOW cost): track a hallucinated
        # human trunk pitch and waist yaw (torso_joint_4). Base-invariant (pure
        # joint-angle function). The PITCH is tracked on ``neural_pitch_joint_names``
        # -- by default the HIP joint (torso_joint_3) alone, so the target is the
        # human's chest-over-pelvis flexion applied at the one "upper-vs-lower
        # body" hinge, leaving the ankle/knee free for the balance task to
        # counter-rotate (see NeuralPostureTask). Falls back to
        # ``trunk_lean_joint_names`` (the legacy full lean sum) when the pitch set
        # is not given. Included only when neural_posture_cost > 0 and the pitch
        # joints + the waist joint are named. The per-tick target is set via
        # set_neural_target(); it defaults to (0, 0) = upright, so before the
        # estimator produces anything the task is a harmless upright regulariser.
        self.neural_task = None
        if neural_posture_cost > 0:
            pitch_joint_names = (neural_pitch_joint_names
                                 if neural_pitch_joint_names
                                 else trunk_lean_joint_names)
            if not pitch_joint_names:
                raise ValueError(
                    "neural_posture_cost > 0 requires neural_pitch_joint_names "
                    "(or trunk_lean_joint_names as a fallback) -- the joint(s) "
                    "whose angle sum is the tracked trunk pitch."
                )
            if not neural_waist_joint_name:
                raise ValueError(
                    "neural_posture_cost > 0 requires neural_waist_joint_name "
                    "(the waist-yaw joint, e.g. torso_joint_4)."
                )
            pitch_v, pitch_q = [], []
            for nm in pitch_joint_names:
                if not self.model.existJointName(nm):
                    raise ValueError(f"Neural-prior pitch joint '{nm}' not in model.")
                jid = self.model.getJointId(nm)
                pitch_v.append(int(self.model.idx_vs[jid]))
                pitch_q.append(int(self.model.idx_qs[jid]))
            if not self.model.existJointName(neural_waist_joint_name):
                raise ValueError(
                    f"Neural-prior waist joint '{neural_waist_joint_name}' "
                    f"not in model.")
            wjid = self.model.getJointId(neural_waist_joint_name)
            self.neural_task = NeuralPostureTask(
                pitch_v, pitch_q,
                int(self.model.idx_vs[wjid]), int(self.model.idx_qs[wjid]),
                cost=neural_posture_cost)

        # Phase-2 base-tracking task: drive chassis x/y/yaw toward a per-tick
        # target set from the operator's head displacement/yaw (see
        # set_base_target). Built only when base_track_cost > 0 and the base
        # joint DOFs were resolved (base_joint_name given); otherwise inert, so
        # the pipeline is byte-identical to the pre-Phase-2 behaviour.
        self.base_task = None
        if base_track_cost > 0:
            if not self._base_v_index:
                raise ValueError(
                    "base_track_cost > 0 requires base_joint_name (the composite "
                    "chassis joint whose x/y/yaw DOFs the task drives).")
            self.base_task = BaseTrackingTask(
                self._base_v_index, self._base_q_index, cost=base_track_cost)

        # Phase-2b waist-yaw follow: drive torso_joint_4 toward a per-tick target
        # (set from the head interaural yaw in sim_teleop). Built ONLY when
        # waist_yaw_follow_cost > 0 AND the neural prior is off -- the
        # NeuralPostureTask already owns the waist when EgoPoser is enabled, so
        # the two must not both target torso_joint_4 (mutual exclusion).
        self.waist_task = None
        if waist_yaw_follow_cost > 0 and neural_posture_cost <= 0:
            if not waist_joint_name:
                raise ValueError(
                    "waist_yaw_follow_cost > 0 requires waist_joint_name "
                    "(the waist-yaw joint, e.g. torso_joint_4).")
            if not self.model.existJointName(waist_joint_name):
                raise ValueError(f"Waist joint '{waist_joint_name}' not in model.")
            wjid = self.model.getJointId(waist_joint_name)
            self.waist_task = WaistYawTask(
                int(self.model.idx_vs[wjid]), int(self.model.idx_qs[wjid]),
                cost=waist_yaw_follow_cost)

        # D-gear waist-zero task: pull torso_joint_4 toward 0° when the chassis
        # enters D gear (trans or yaw), enforcing trunk/chassis alignment during
        # locomotion. Built as a separate task (independent of waist_task and
        # NeuralPostureTask) so its cost can be toggled 0 <-> waist_zero_d_gear_cost
        # at runtime without affecting other waist constraints. Target is FIXED at
        # 0.0 rad (set once at construction). Cost 0 disables (task not built).
        self.waist_zero_task = None
        if waist_zero_d_gear_cost > 0:
            if not waist_joint_name:
                raise ValueError(
                    "waist_zero_d_gear_cost > 0 requires waist_joint_name "
                    "(the waist-yaw joint, e.g. torso_joint_4).")
            if not self.model.existJointName(waist_joint_name):
                raise ValueError(f"Waist joint '{waist_joint_name}' not in model.")
            wjid = self.model.getJointId(waist_joint_name)
            self.waist_zero_task = WaistYawTask(
                int(self.model.idx_vs[wjid]), int(self.model.idx_qs[wjid]),
                cost=0.0)  # starts at 0, sim_teleop will boost in D gear
            self.waist_zero_task.set_target(0.0)  # fixed target

        # Phase-3 hand-in-front soft guard: one-sided penalty nudging a hand
        # forward once it goes behind a pelvis-anchored margin plane (see
        # HandFrontTask). Uses the two tool frames + the hip (pelvis) frame + the
        # waist link (for the forward heading). Built only when hand_front_cost>0.
        self.hand_front_task = None
        if hand_front_cost > 0:
            pf = hand_front_pelvis_frame_name
            wf = hand_front_waist_frame_name
            if not pf or not wf:
                raise ValueError(
                    "hand_front_cost > 0 requires hand_front_pelvis_frame_name "
                    "and hand_front_waist_frame_name.")
            for role, nm in (("pelvis", pf), ("waist", wf),
                             ("left tool", left_tool_name),
                             ("right tool", right_tool_name)):
                if not self.model.existFrame(nm):
                    raise ValueError(f"Hand-front {role} frame '{nm}' not in model.")
            self.hand_front_task = HandFrontTask(
                self.model.getFrameId(left_tool_name),
                self.model.getFrameId(right_tool_name),
                self.model.getFrameId(pf), self.model.getFrameId(wf),
                cost=hand_front_cost, margin=hand_front_margin)

        self._tasks = [self.head_task, self.left_task, self.right_task,
                       self.posture_task]
        for task in (self.com_task, self.chest_task, self.trunk_task,
                     self.neural_task, self.base_task, self.waist_task,
                     self.waist_zero_task, self.hand_front_task,
                     self.damping_task, self.low_accel_task):
            if task is not None:
                self._tasks.append(task)
        self._frames = {
            self.head_task: head_frame_name,
            self.left_task: left_tool_name,
            self.right_task: right_tool_name,
        }

    # -- helpers -------------------------------------------------------------
    def _as_body_vector(self, value, name: str) -> np.ndarray:
        """Broadcast a scalar / length-n value to a (n,) per-body-joint vector."""
        vec = np.asarray(value, dtype=np.float64)
        if vec.ndim == 0:
            vec = np.full(self.n, float(vec))
        if vec.shape != (self.n,):
            raise ValueError(f"{name} must be scalar or length {self.n}, got {vec.shape}")
        return vec

    def _make_damping_task(self, damping_cost) -> Optional[DampingTask]:
        """Build the DampingTask from a scalar or per-body-DOF cost vector.

        Pink's DampingTask weight is ``diag(cost)`` over the full tangent space,
        so a vector cost must be length ``model.nv`` in tangent order (not body
        order). A scalar is passed through as a Python ``float`` (Pink treats a
        float as a uniform cost; a numpy scalar or int would hit the vector
        branch and misbehave). Returns ``None`` when nothing is penalized.
        """
        vec = np.asarray(damping_cost, dtype=np.float64)
        if vec.ndim == 0:
            cost = float(vec)
            return DampingTask(cost=cost) if cost > 0 else None
        # Per-DOF: scatter the body-ordered costs onto tangent DOFs by _v_index.
        body_cost = self._as_body_vector(vec, "damping_cost")
        if not np.any(body_cost > 0):
            return None
        cost_tan = np.zeros(self.model.nv, dtype=np.float64)
        cost_tan[self._v_index] = body_cost
        return DampingTask(cost=cost_tan)

    def reset(self) -> None:
        """Forget the previous-step velocity used by the rate-aware components.

        Call on (re)calibration: the target jumps to a fresh anchor, so the
        robot should ramp up from rest rather than inherit stale velocity. Both
        the acceleration *limit* (hard constraint) and the low-acceleration
        *task* (soft cost) remember the last tick's velocity, so both are
        cleared here -- otherwise they would penalise / brake against motion
        across the re-anchor discontinuity.
        """
        self.acceleration_limit.Delta_q_prev = np.zeros(
            len(self.acceleration_limit.indices)
        )
        if self.low_accel_task is not None:
            self.low_accel_task.Delta_q_prev = None  # None == "from rest"

    def _freeze_base_dofs(self, dof_indices, frozen: bool) -> None:
        """Zero (frozen) or restore the velocity limit of the given base tangent
        DOFs. Mirrors into _v_tan so the Phase-0 output clamp stays consistent.
        The freeze is coordinated inside the QP (Pink's VelocityLimit reads
        model.velocityLimit live), so the arms/torso re-solve around the pinned
        DOFs. No-op without base_joint_name (nothing to freeze) or when
        enforce_limits is False (velocity limit bypassed)."""
        if not self._base_v_index:
            return
        for iv in dof_indices:
            k = self._base_v_index.index(iv)
            cap = 0.0 if frozen else float(self._base_v_cap[k])
            self.model.velocityLimit[iv] = cap
            self._v_tan[iv] = cap

    def set_base_xy_frozen(self, frozen: bool) -> None:
        """Pin (or release) the base TRANSLATION (chassis x, y) only. Phase-1
        dead-zone lever, gated on head horizontal speed. Leaves yaw untouched
        (Phase-4b: xy and yaw freeze independently)."""
        if not self._base_v_index or frozen == self._base_xy_frozen:
            return
        self._freeze_base_dofs(self._base_v_index[0:2], frozen)
        self._base_xy_frozen = bool(frozen)

    def set_base_yaw_frozen(self, frozen: bool) -> None:
        """Pin (or release) the base YAW (chassis turn) only. Gated on the
        combined yaw-rate signal (Phase-4b) when yaw scheduling is on, else
        mirrors the xy freeze. Independent of xy so an in-place turn (low
        translation, high yaw rate) can turn the base without walking."""
        if not self._base_v_index or frozen == self._base_yaw_frozen:
            return
        self._freeze_base_dofs([self._base_v_index[2]], frozen)
        self._base_yaw_frozen = bool(frozen)

    def set_base_frozen(self, frozen: bool) -> None:
        """Pin (or release) the WHOLE mobile base (x, y, yaw) at once.

        Backward-compatible convenience wrapping set_base_xy_frozen +
        set_base_yaw_frozen. Used by (re)calibration and the self-checks. When
        yaw scheduling is off, sim_teleop drives yaw to mirror xy so this remains
        the effective behaviour; when on, xy and yaw are gated independently.
        """
        self.set_base_xy_frozen(frozen)
        self.set_base_yaw_frozen(frozen)

    @property
    def base_xy_frozen(self) -> bool:
        """Whether the base translation (x, y) is currently pinned."""
        return self._base_xy_frozen

    @property
    def base_yaw_frozen(self) -> bool:
        """Whether the base yaw (turn) is currently pinned."""
        return self._base_yaw_frozen

    @property
    def base_frozen(self) -> bool:
        """Whether the WHOLE base is pinned (both xy and yaw). Kept for the
        status line / self-checks; prefer base_xy_frozen / base_yaw_frozen."""
        return self._base_xy_frozen and self._base_yaw_frozen

    def set_lean_frozen(self, frozen: bool) -> None:
        """Pin (or release) the trunk lean spine for subsequent solves.

        Phase-1b dead-zone (generalizing the successful Phase-1 base dead-zone):
        when the operator is not squatting or bending (head vertical speed below
        threshold), the lean angles (torso_joint_1/2/3: ankle/knee/hip pitch)
        freeze at their current values so the knees/ankles stop jittering during
        stationary fine manipulation. When the operator squats or bends forward
        (head vertical speed rises above threshold), the spine unfreezes to track
        the height/posture change. Mirrors set_base_frozen but gates on head
        VERTICAL speed (squat detection) instead of horizontal (walk detection).

        Idempotent and O(1); takes effect on the next solve() because Pink's
        VelocityLimit reads model.velocityLimit live. No-op if the solver was
        built without trunk_lean_joint_names (nothing to freeze), or when
        enforce_limits is False (bypass mode cannot freeze).
        """
        if not self._lean_v_index or frozen == self._lean_frozen:
            return
        cap = np.zeros(len(self._lean_v_index)) if frozen else self._lean_v_cap
        self.model.velocityLimit[self._lean_v_index] = cap
        # Mirror into the Phase-0 output clamp so frozen DOFs also clamp to 0.
        for k, iv in enumerate(self._lean_v_index):
            self._v_tan[iv] = cap[k]
        self._lean_frozen = bool(frozen)

    @property
    def lean_frozen(self) -> bool:
        """Whether the trunk lean spine is currently pinned (see set_lean_frozen)."""
        return self._lean_frozen

    def set_chassis_yaw_damping(self, cost: float) -> None:
        """Dynamically set the chassis yaw damping cost (Phase-4 continuous
        scheduling). Only effective when the base is UNFROZEN -- when frozen the
        velocityLimit=0 dominates and this damping value is moot. The cost is
        written live into damping_task.cost[chassis_yaw_v_index] and takes effect
        on the next solve(). Caller is responsible for smoothing the cost itself
        (EMA) to avoid QP objective discontinuities ("shift shock"). No-op if the
        solver was built without base_joint_name or damping_task."""
        if not self._base_v_index or self.damping_task is None:
            return
        self.damping_task.cost[self._base_v_index[2]] = float(cost)

    @property
    def chassis_yaw_damping(self) -> float:
        """Current chassis yaw damping cost (read-only query for telemetry)."""
        if not self._base_v_index or self.damping_task is None:
            return 0.0
        return float(self.damping_task.cost[self._base_v_index[2]])

    def set_chassis_xy_damping(self, cost: float) -> None:
        """Dynamically set the chassis TRANSLATION (x, y) damping cost (Phase-4b
        continuous scheduling, mirrors set_chassis_yaw_damping). Only effective
        when the base xy is UNFROZEN. Writes both x and y damping entries live;
        caller smooths the cost (EMA) to avoid shift shock. No-op without
        base_joint_name / damping_task."""
        if not self._base_v_index or self.damping_task is None:
            return
        self.damping_task.cost[self._base_v_index[0]] = float(cost)
        self.damping_task.cost[self._base_v_index[1]] = float(cost)

    @property
    def chassis_xy_damping(self) -> float:
        """Current chassis xy (translation) damping cost (telemetry; x entry)."""
        if not self._base_v_index or self.damping_task is None:
            return 0.0
        return float(self.damping_task.cost[self._base_v_index[0]])

    def set_base_target(self, x: float, y: float, yaw: float) -> None:
        """Set the chassis (x, y, yaw) reference for the Phase-2 base task.

        No-op if the base-tracking task was not built (base_track_cost == 0).
        sim_teleop calls this each tick with a target derived from the head
        displacement/yaw; while the base is frozen it passes the current base
        pose so the task error stays ~0.
        """
        if self.base_task is not None:
            self.base_task.set_target(x, y, yaw)

    def set_waist_yaw_target(self, yaw: float) -> None:
        """Set the waist-yaw (torso_joint_4) reference for the Phase-2b task.

        No-op if the waist-yaw task was not built (waist_yaw_follow_cost == 0, or
        EgoPoser owns the waist). sim_teleop calls this each tick with the head
        interaural yaw since calibration, clamped to a soft limit.
        """
        if self.waist_task is not None:
            self.waist_task.set_target(yaw)

    def set_waist_yaw_cost(self, cost: float) -> None:
        """Adjust the waist-yaw task weight at runtime (no-op if task absent).

        Used by sim_teleop to strengthen waist regularization when the base enters
        D gear (trans or yaw), enforcing trunk/chassis alignment during locomotion.
        """
        if self.waist_task is not None:
            self.waist_task.cost = float(cost)

    def set_neural_posture_cost(self, cost: float) -> None:
        """Adjust the EgoPoser neural-posture task weight at runtime (no-op if absent).

        Used by sim_teleop to strengthen trunk regularization when the base enters
        D gear, reinforcing healthy posture during locomotion (waist yaw included in
        the neural prior, so waist/chassis alignment is tightened together with pitch).
        """
        if self.neural_task is not None:
            self.neural_task.cost = float(cost)

    def set_posture_cost(self, cost: float) -> None:
        """Adjust the PostureTask weight at runtime (always present).

        The PostureTask pulls all joints toward the home pose. Use this to
        strengthen whole-body regularization in D gear, ensuring healthy posture
        during locomotion — notably torso_joint_4 (waist yaw) is pulled toward
        home (≈0°), keeping the trunk aligned with the chassis heading while moving.
        """
        self.posture_task.cost = float(cost)

    def set_waist_zero_cost(self, cost: float) -> None:
        """Adjust the D-gear waist-zero task weight at runtime (no-op if absent).

        The waist_zero_task (if built via waist_zero_d_gear_cost > 0) pulls
        torso_joint_4 toward 0° with a fixed target. Set cost > 0 in D gear to
        enforce trunk/chassis alignment during locomotion, 0 in P/N to disable.
        """
        if self.waist_zero_task is not None:
            self.waist_zero_task.cost = float(cost)

    def _to_model_q(self, q_body: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        for idx, qi in zip(self._q_index, np.asarray(q_body, dtype=np.float64)):
            q[idx] = qi
        return q

    def _from_model_q(self, q_model: np.ndarray) -> np.ndarray:
        return np.array([q_model[i] for i in self._q_index])

    def _set_target(self, task: FrameTask, target: Target, cfg0: pink.Configuration):
        p, R = target
        if R is None:
            # orientation_cost may be 0 (ignored), but FrameTask still needs a
            # valid rotation -- use the current frame rotation to avoid a jump.
            R = cfg0.get_transform_frame_to_world(self._frames[task]).rotation
        task.set_target(pin.SE3(np.asarray(R, dtype=np.float64),
                                np.asarray(p, dtype=np.float64)))

    def _set_com_target(self, cfg: pink.Configuration) -> None:
        """Aim the balance task at the CoM directly above the current base.

        The horizontal (x, y) target is the base frame's current world position,
        so the penalty is on the CoM *leaning away from the base* rather than on
        an absolute world spot -- translating the base to extend reach carries the
        balance target along and is not penalised, only leaning the trunk is. The
        vertical target is the actual current CoM height, so the (default-zero)
        vertical cost never pulls height and squatting stays free even if the
        vertical cost is later raised. Only the horizontal error then drives the
        task through its (com_cost, com_cost, com_cost_vertical) weight.
        """
        base_xy = cfg.get_transform_frame_to_world(self.base_frame_name).translation
        com = pin.centerOfMass(self.model, self.data, cfg.q)
        self.com_task.set_target(np.array([base_xy[0], base_xy[1], com[2]]))

    # -- public API ----------------------------------------------------------
    def set_neural_target(self, pitch: float, yaw: float) -> None:
        """Update the EgoPoser trunk prior (trunk pitch, waist yaw), in radians.

        No-op if the neural posture task is not active (neural_posture_cost 0).
        ``pitch`` is the target SUM of the three sagittal lean joints and ``yaw``
        the target waist-yaw (torso_joint_4) angle. Call once per control tick
        with the estimator's latest output; leaving it unset keeps the last
        value (default (0, 0) = upright).
        """
        if self.neural_task is not None:
            self.neural_task.set_target(pitch, yaw)

    def frame_pose(self, q_body: np.ndarray, frame_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """(R, p) of a frame at the given body config, in the model world frame.

        Useful for tests / calibration cross-checks (matches MuJoCo FK)."""
        cfg = pink.Configuration(self.model, self.data, self._to_model_q(q_body))
        T = cfg.get_transform_frame_to_world(frame_name)
        return T.rotation.copy(), T.translation.copy()

    def solve(
        self,
        q_init: np.ndarray,
        head_target: Target,
        left_target: Target,
        right_target: Target,
    ) -> np.ndarray:
        """Return the 23 body joint angles tracking all three end-effectors.

        Runs a single differential-IK QP for this control tick. With the
        velocity / acceleration limits active, that one step is already bounded
        to a physical per-tick motion, so no manual step clamp is needed.

        Each ``*_target`` is ``(world_position, world_rotation_or_None)``. A
        ``None`` rotation means "don't care" (its task should have
        orientation_cost 0); the current frame rotation is substituted.
        """
        q_init = np.clip(np.asarray(q_init, dtype=np.float64).copy(),
                         self.lower, self.upper)
        q_model = self._to_model_q(q_init)

        cfg = pink.Configuration(self.model, self.data, q_model)
        self._set_target(self.head_task, head_target, cfg)
        self._set_target(self.left_task, left_target, cfg)
        self._set_target(self.right_task, right_target, cfg)
        if self.com_task is not None:
            self._set_com_target(cfg)

        dt = self.control_dt
        v = self._solve_velocity(cfg, dt)

        # --- Phase-0 output guard (safety net) ------------------------------ #
        # A degenerate / near-singular QP can return a non-finite or physically
        # impossible velocity WITHOUT raising NoSolutionFound (quadprog on an
        # ill-conditioned Hessian). Integrating that propagates NaN/huge joint
        # values into MuJoCo, which then reports "Nan, Inf or huge value in
        # QACC" and the sim explodes. Two cheap checks stop it here:
        #   1. non-finite velocity -> hold still this tick (zero velocity);
        #   2. otherwise clamp |v| to the per-DOF velocity cap (the QP should
        #      already respect it; this catches the limit-dropped fallback path
        #      and any numerical overshoot).
        if not np.isfinite(v).all():
            # Non-finite is ALWAYS rejected (NaN protection is unconditional).
            v = np.zeros(self.model.nv)
        elif self.enforce_limits:
            # Velocity clamp only when limits are enforced -- enforce_limits=False
            # is a documented "truly unconstrained" mode (used for A/B tests), so
            # we must not silently re-impose the cap there.
            v = np.clip(v, -self._v_tan, self._v_tan)

        q_model = cfg.integrate(v, dt)
        # Remember this step's velocity so the next tick can bound / penalise the
        # change. The hard acceleration limit only matters when limits are on;
        # the soft low-acceleration task is in the QP whenever it exists.
        if self.enforce_limits:
            self.acceleration_limit.set_last_integration(v, dt)
        if self.low_accel_task is not None:
            self.low_accel_task.set_last_integration(v, dt)

        q = self._from_model_q(q_model)
        # Final belt-and-suspenders: never return a non-finite configuration --
        # hold the previous one so nothing downstream (command_arm -> MuJoCo)
        # ever sees NaN/Inf.
        if not np.isfinite(q).all():
            return q_init
        if self._max_joint_step is not None:        # legacy clamp, off by default
            q = q_init + np.clip(q - q_init,
                                 -self._max_joint_step, self._max_joint_step)
        return np.clip(q, self.lower, self.upper)

    def _solve_velocity(self, cfg: pink.Configuration, dt: float) -> np.ndarray:
        """One QP solve with the configured limits, with a graceful fallback.

        Velocity and (Flacco-style braking) acceleration limits can, near a
        joint stop, define an empty feasible set; rather than crash the teleop
        loop we drop the acceleration limit for that tick, then (worst case)
        hold position.
        """
        if not self.enforce_limits:
            # Truly unconstrained: pass limits=[] explicitly. solve_ik's default
            # (limits=None) would re-apply ConfigurationLimit + VelocityLimit,
            # and we *injected* model.velocityLimit above, so omitting this would
            # silently keep the velocity cap on even with enforce_limits=False.
            return solve_ik(cfg, self._tasks, dt, solver=self.solver,
                            limits=[], safety_break=False)
        try:
            return solve_ik(cfg, self._tasks, dt, solver=self.solver,
                            limits=self._limits, safety_break=False)
        except NoSolutionFound:
            try:
                return solve_ik(cfg, self._tasks, dt, solver=self.solver,
                                limits=[self.config_limit, self.velocity_limit],
                                safety_break=False)
            except NoSolutionFound:
                return np.zeros(self.model.nv)
