"""Damped least squares (DLS) inverse kinematics for one Astribot arm.

Given a desired tool pose (position, optionally orientation) this solves for the
7 arm joint angles using MuJoCo's analytic body Jacobian. It runs on a private
scratch `MjData` so the live simulation state is never touched.

Why DLS: it is singularity-robust (no matrix inversion blow-ups near the
workspace boundary), needs no external IK library, and a handful of iterations
per control tick is plenty for teleoperation.
"""

from __future__ import annotations

from typing import List, Sequence

import mujoco
import numpy as np

from avp_teleop.retarget.frames import rotation_error


class ArmIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        arm_joint_names: Sequence[str],
        tool_body_name: str,
        damping: float = 0.08,
        max_iters: int = 12,
        pos_tol: float = 2e-3,
        max_joint_step: float = 0.15,
    ):
        self.model = model
        self._scratch = mujoco.MjData(model)
        self.damping = damping
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.max_joint_step = max_joint_step

        self.tool_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, tool_body_name
        )
        if self.tool_body_id < 0:
            raise ValueError(f"Tool body '{tool_body_name}' not found in model.")

        # qpos / dof addresses for each arm joint (all hinges => 1 dof each).
        self.joint_ids: List[int] = []
        self.qpos_adr: List[int] = []
        self.dof_adr: List[int] = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Arm joint '{name}' not found in model.")
            self.joint_ids.append(jid)
            self.qpos_adr.append(int(model.jnt_qposadr[jid]))
            self.dof_adr.append(int(model.jnt_dofadr[jid]))

        self.lower = np.array([model.jnt_range[j][0] for j in self.joint_ids])
        self.upper = np.array([model.jnt_range[j][1] for j in self.joint_ids])
        self.n = len(self.joint_ids)

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _fk(self, q: np.ndarray, base_qpos: np.ndarray):
        """Forward kinematics on the scratch data for arm config q.

        `base_qpos` provides the full-body posture (torso, other arm, chassis)
        so the arm Jacobian/pose are computed in the correct context.
        """
        self._scratch.qpos[:] = base_qpos
        for adr, qi in zip(self.qpos_adr, q):
            self._scratch.qpos[adr] = qi
        mujoco.mj_kinematics(self.model, self._scratch)
        mujoco.mj_comPos(self.model, self._scratch)

    def _tool_pose(self):
        p = self._scratch.xpos[self.tool_body_id].copy()
        R = self._scratch.xmat[self.tool_body_id].reshape(3, 3).copy()
        return p, R

    def solve(
        self,
        q_init: np.ndarray,
        target_p: np.ndarray,
        target_R: np.ndarray | None,
        base_qpos: np.ndarray,
    ) -> np.ndarray:
        """Return arm joint angles tracking the target pose.

        Parameters
        ----------
        q_init : (7,) seed configuration (usually last solution).
        target_p : (3,) desired tool position in world frame.
        target_R : (3,3) desired tool orientation, or None for position-only.
        base_qpos : (nq,) full-body posture to hold fixed for non-arm joints.
        """
        q = np.clip(np.asarray(q_init, dtype=np.float64).copy(), self.lower, self.upper)
        use_ori = target_R is not None
        dim = 6 if use_ori else 3
        lam2 = self.damping ** 2

        for _ in range(self.max_iters):
            self._fk(q, base_qpos)
            p, R = self._tool_pose()

            err = np.zeros(dim)
            err[:3] = target_p - p
            if use_ori:
                err[3:] = rotation_error(R, target_R)

            if np.linalg.norm(err[:3]) < self.pos_tol and (
                not use_ori or np.linalg.norm(err[3:]) < 1e-2
            ):
                break

            mujoco.mj_jacBody(
                self.model, self._scratch, self._jacp, self._jacr, self.tool_body_id
            )
            cols = self.dof_adr
            if use_ori:
                J = np.vstack([self._jacp[:, cols], self._jacr[:, cols]])  # (6, n)
            else:
                J = self._jacp[:, cols]  # (3, n)

            # DLS: dq = J^T (J J^T + lam^2 I)^-1 err
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(dim), err)

            # Clamp per-joint step for smoothness/stability.
            np.clip(dq, -self.max_joint_step, self.max_joint_step, out=dq)
            q = np.clip(q + dq, self.lower, self.upper)

        return q
