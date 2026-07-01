"""Pinocchio + Pink inverse kinematics for one Astribot arm.

Why this exists
---------------
The hand-rolled DLS solver in ``arm_ik.py`` has no control over the 7-DOF
arm's null space: it drifts into twisted configurations, converges loosely,
and behaves poorly near singularities. This backend replaces it with Pink's
differential IK, posing two weighted tasks:

    * a ``FrameTask`` on the tool frame  -> end-effector tracking (precision)
    * a low-weight ``PostureTask`` to ``home`` -> resolves the null space so
      the elbow/wrist stay in a natural posture instead of wandering.

Pink's Levenberg-Marquardt damping (``lm_damping``) handles singularities
smoothly, and the QP enforces the URDF joint limits as hard constraints.

Frame consistency (the important part)
--------------------------------------
The model is built **from the same MJCF the simulator loads**, not from the
standalone arm URDF. The standalone ``astribot_arm_right.urdf`` has *different*
internal joint frames than the MJCF (verified: tool poses disagree by cm-scale
under a best-fit rigid transform), so solving in the URDF and copying joint
angles to MuJoCo would mis-place the tool. Building Pinocchio from the
flattened MJCF makes Pink's tool frame coincide with MuJoCo's to < 1e-4 m at
every configuration, so the solved angles are written straight to MuJoCo with
**no coordinate transform**. Targets are expressed in the MuJoCo world frame,
exactly as produced by ``retarget/frames.py``.

The same ``solve(q_init, target_p, target_R, base_qpos)`` signature as ``ArmIK``
is kept, so ``sim_teleop`` can switch backends with no other change.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Sequence

import numpy as np

import mujoco
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask


def _flatten_mjcf(mjcf_path: str) -> str:
    """Resolve all <include>s into a single self-contained XML on disk.

    Pinocchio's MJCF parser does not follow MuJoCo ``<include>`` directives,
    and the teleop model is assembled entirely from includes. Loading the
    model in MuJoCo and re-saving it produces a flat file Pinocchio can read,
    and guarantees Pink and the simulator see byte-identical kinematics.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    fd, flat_path = tempfile.mkstemp(prefix="astribot_pink_", suffix=".xml")
    os.close(fd)
    mujoco.mj_saveLastXML(flat_path, model)
    return flat_path


class PinkArmIK:
    """Differential IK for a single 7-DOF Astribot arm via Pinocchio + Pink."""

    def __init__(
        self,
        mjcf_path: str,
        arm_joint_names: Sequence[str],
        tool_frame_name: str,
        home: np.ndarray,
        *,
        position_cost: float = 10.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        lm_damping: float = 1e-3,
        max_joint_step: float = 0.15,
        control_dt: float = 1.0 / 60.0,
        solver_iters: int = 8,
        solver: str = "quadprog",
    ):
        self.tool_frame_name = tool_frame_name
        self.max_joint_step = float(max_joint_step)
        self.control_dt = float(control_dt)
        self.solver_iters = int(solver_iters)
        self.solver = solver
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

        arm_joint_names = list(arm_joint_names)
        missing = [n for n in arm_joint_names if not full.existJointName(n)]
        if missing:
            raise ValueError(f"Arm joints not found in MJCF model: {missing}")
        if not full.existFrame(tool_frame_name):
            raise ValueError(f"Tool frame '{tool_frame_name}' not in MJCF model.")

        # Lock every joint except the arm's: the other DOFs (torso, head,
        # chassis, other arm, fingers) are held at neutral, which matches the
        # simulator's posture for these joints during teleop.
        lock_ids = [
            jid
            for jid in range(1, full.njoints)  # 0 is the universe joint
            if full.names[jid] not in arm_joint_names
        ]
        self.model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self.data = self.model.createData()
        self.tool_frame_id = self.model.getFrameId(tool_frame_name)

        # Map our arm_joint_names order -> reduced-model qpos indices, so the
        # caller's joint vector ordering is honoured regardless of model order.
        self._q_index: List[int] = [
            self.model.idx_qs[self.model.getJointId(n)] for n in arm_joint_names
        ]
        self.n = len(arm_joint_names)

        # Joint limits from the reduced model (identical to MuJoCo's ranges).
        self.lower = np.array(
            [self.model.lowerPositionLimit[i] for i in self._q_index]
        )
        self.upper = np.array(
            [self.model.upperPositionLimit[i] for i in self._q_index]
        )

        # --- tasks ---
        self.frame_task = FrameTask(
            tool_frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = PostureTask(cost=posture_cost)
        self.posture_task.set_target(self._to_model_q(self.home))

    # -- helpers -------------------------------------------------------------
    def _to_model_q(self, q_arm: np.ndarray) -> np.ndarray:
        """Scatter a 7-vector (arm_joint_names order) into a full model q."""
        q = pin.neutral(self.model)
        for idx, qi in zip(self._q_index, np.asarray(q_arm, dtype=np.float64)):
            q[idx] = qi
        return q

    def _from_model_q(self, q_model: np.ndarray) -> np.ndarray:
        """Gather the arm joints (arm_joint_names order) from a full model q."""
        return np.array([q_model[i] for i in self._q_index])

    # -- public API (mirrors ArmIK.solve) -----------------------------------
    def solve(
        self,
        q_init: np.ndarray,
        target_p: np.ndarray,
        target_R: Optional[np.ndarray],
        base_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return arm joint angles tracking the target tool pose.

        Parameters match ``ArmIK.solve``. ``target_p`` / ``target_R`` are in the
        MuJoCo world frame; ``base_qpos`` is accepted for signature parity but
        unused (non-arm joints are frozen at neutral in the reduced model).
        """
        q_init = np.clip(
            np.asarray(q_init, dtype=np.float64).copy(), self.lower, self.upper
        )

        # FrameTask needs a full SE3 target. With orientation_cost == 0 the
        # rotation part is ignored, but must still be a valid rotation.
        if target_R is None:
            cfg0 = pink.Configuration(self.model, self.data, self._to_model_q(q_init))
            R = cfg0.get_transform_frame_to_world(self.tool_frame_name).rotation
        else:
            R = np.asarray(target_R, dtype=np.float64)
        self.frame_task.set_target(pin.SE3(R, np.asarray(target_p, dtype=np.float64)))

        # A few differential steps per control tick, warm-started from q_init.
        q_model = self._to_model_q(q_init)
        for _ in range(self.solver_iters):
            cfg = pink.Configuration(self.model, self.data, q_model)
            v = solve_ik(
                cfg,
                [self.frame_task, self.posture_task],
                self.control_dt,
                solver=self.solver,
            )
            q_model = cfg.integrate(v, self.control_dt)

        q = np.clip(self._from_model_q(q_model), self.lower, self.upper)

        # Per-tick step clamp for smoothness, matching the DLS backend.
        step = np.clip(q - q_init, -self.max_joint_step, self.max_joint_step)
        return q_init + step
