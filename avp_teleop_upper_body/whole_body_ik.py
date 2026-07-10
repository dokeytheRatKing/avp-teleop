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
        max_velocity: np.ndarray | float = 3.0,
        max_acceleration: np.ndarray | float = 100.0,
        config_limit_gain: float = 0.5,
        enforce_limits: bool = True,
        control_dt: float = 1.0 / 60.0,
        solver: str = "quadprog",
        max_joint_step: np.ndarray | float | None = None,
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

        self._tasks = [self.head_task, self.left_task, self.right_task,
                       self.posture_task]
        for task in (self.com_task, self.chest_task, self.trunk_task,
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
        q_model = cfg.integrate(v, dt)
        # Remember this step's velocity so the next tick can bound / penalise the
        # change. The hard acceleration limit only matters when limits are on;
        # the soft low-acceleration task is in the QP whenever it exists.
        if self.enforce_limits:
            self.acceleration_limit.set_last_integration(v, dt)
        if self.low_accel_task is not None:
            self.low_accel_task.set_last_integration(v, dt)

        q = self._from_model_q(q_model)
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
