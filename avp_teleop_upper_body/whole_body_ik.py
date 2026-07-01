"""Merged whole-upper-body inverse kinematics (Pinocchio + Pink).

One differential-IK problem drives the entire upper body -- 4-DOF torso,
2-DOF neck, and two 7-DOF arms (20 DOF total) -- by posing four weighted tasks
in a single Pink configuration:

    * FrameTask(head camera)  -> AVP head 6-DoF target
    * FrameTask(left tool)    -> left-hand target
    * FrameTask(right tool)   -> right-hand target
    * PostureTask(home)       -> resolves the large null space (upright torso,
                                 natural elbows)
    * DampingTask             -> soft penalty on |v|        (bleeds off velocity)
    * LowAccelerationTask     -> soft penalty on |v - v_prev| (limits jerk)

Because all three end-effector tasks share one configuration, the arms
**automatically compensate** for torso motion: when the head target leans the
torso forward, the QP simultaneously re-solves the arms so the hands stay on
their world targets. This is the merged-solver pattern of VisionProTeleop's
``11_diffik_aloha.py`` (two arm FrameTasks + posture), extended with a head
task and realised in Pink instead of mink.

Physical rate limits are enforced *inside* the QP via :mod:`pink.limits`:
a :class:`VelocityLimit` and an :class:`AccelerationLimit` (plus the usual
:class:`ConfigurationLimit`) bound joint velocity and acceleration as hard
inequality constraints. One ``solve_ik`` is run per control tick, so those
bounds are the true per-tick velocity / acceleration of the robot. This makes
the old "solve hard, then ``np.clip`` the net step" rate limiter obsolete: the
solver now trades rate limits against task tracking inside the optimisation
rather than truncating the answer afterwards. The teleop MJCF declares no
velocity limit, so the caps come from :class:`WholeBodyIKConfig`.

On top of those *hard* caps, two low-cost *soft* tasks shape the motion within
the caps for smoothness: a :class:`DampingTask` penalizes joint velocity and a
:class:`LowAccelerationTask` penalizes frame-to-frame acceleration. Their costs
sit far below the tracking costs, so they only smooth the redundant slack and
never visibly fight the end-effector targets. The low-acceleration task is
stateful (it remembers the previous tick's velocity); ``solve`` feeds it the
integrated velocity each tick, and :meth:`reset` clears it on (re)calibration.

Frame consistency (identical guarantee to :mod:`avp_teleop.retarget.arm_ik_pink`):
the model is built from the *flattened* teleop MJCF, so every task frame
coincides with MuJoCo's to < 1e-4 m and solved joint angles are written to the
simulator with no coordinate transform. All targets are in the MuJoCo world
frame, exactly as produced by :mod:`avp_teleop.retarget.frames`.
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
from pink.tasks import FrameTask, PostureTask, DampingTask, LowAccelerationTask
from pink.limits import ConfigurationLimit, VelocityLimit, AccelerationLimit
from pink.exceptions import NoSolutionFound


# A per-end target: world position (3,) and optional world rotation (3, 3).
Target = Tuple[np.ndarray, Optional[np.ndarray]]


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
    """Differential IK for the 20-DOF upper body (torso + neck + both arms)."""

    def __init__(
        self,
        mjcf_path: str,
        body_joint_names: Sequence[str],
        head_frame_name: str,
        left_tool_name: str,
        right_tool_name: str,
        home: np.ndarray,
        *,
        arm_position_cost: float = 10.0,
        arm_orientation_cost: float = 1.0,
        head_position_cost: float = 3.0,
        head_orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        lm_damping: float = 1e-3,
        damping_cost: float = 1e-1,
        low_accel_cost: float = 1e-1,
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

        # Lock every joint except the 20 body DOFs (fingers, chassis, anything
        # else) at neutral -- matching the simulator's posture for those joints.
        lock_ids = [
            jid
            for jid in range(1, full.njoints)  # 0 is the universe joint
            if full.names[jid] not in body_joint_names
        ]
        self.model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self.data = self.model.createData()

        # Map our body_joint_names order -> reduced-model qpos indices.
        self._q_index: List[int] = [
            self.model.idx_qs[self.model.getJointId(n)] for n in body_joint_names
        ]
        # ... and tangent (velocity) indices, for the per-joint rate limits.
        self._v_index: List[int] = [
            self.model.idx_vs[self.model.getJointId(n)] for n in body_joint_names
        ]
        self.n = len(body_joint_names)

        self.lower = np.array([self.model.lowerPositionLimit[i] for i in self._q_index])
        self.upper = np.array([self.model.upperPositionLimit[i] for i in self._q_index])

        # --- hard velocity / acceleration limits (QP inequalities) ----------- #
        # The MJCF carries no velocity limit, so inject ours into the model
        # (VelocityLimit reads model.velocityLimit) and pass a_max to
        # AccelerationLimit. Both are restricted to the 20 body joints; every
        # other (locked) joint is gone from the reduced model.
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

        # Soft smoothing tasks (low-cost objectives, not hard constraints). Each
        # is included only when its cost > 0 so it can be A/B-disabled. The
        # low-acceleration task is stateful: solve() feeds it the integrated
        # velocity each tick and reset() clears it (see below).
        self.damping_task = DampingTask(cost=damping_cost) if damping_cost > 0 else None
        self.low_accel_task = (
            LowAccelerationTask(cost=low_accel_cost) if low_accel_cost > 0 else None
        )

        self._tasks = [self.head_task, self.left_task, self.right_task,
                       self.posture_task]
        for task in (self.damping_task, self.low_accel_task):
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
        """Return the 20 body joint angles tracking all three end-effectors.

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
