"""Coordinate conventions and relative-pose calibration.

The AVP reports the wrist as a 4x4 world pose in a Z-up frame (the streamer
applies a Y-up -> Z-up transform internally). We do NOT try to align the AVP
world origin with the robot base. Instead we use *relative* (incremental)
teleoperation:

    * At calibration time we record the operator's wrist pose `wrist0` and the
      robot tool's current pose `tool0`.
    * Each frame we compute the operator's motion since calibration and apply
      a scaled version of it on top of `tool0`.

This means the operator can be anywhere, in any pose, when they press "start";
only motion *relative* to that instant matters. It also makes re-centering
trivial (just re-calibrate).

Position delta is scaled by `position_scale`. Orientation can be tracked or
ignored via `track_orientation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto SO(3) via SVD (nearest rotation)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


@dataclass
class WristCalibration:
    """Captured reference frames at the moment teleop is (re)started."""

    wrist0_R: np.ndarray   # (3, 3) operator wrist rotation at t0
    wrist0_p: np.ndarray   # (3,)   operator wrist position at t0
    tool0_R: np.ndarray    # (3, 3) robot tool rotation at t0
    tool0_p: np.ndarray    # (3,)   robot tool position at t0

    @classmethod
    def capture(cls, wrist_pose: np.ndarray, tool_pose: np.ndarray) -> "WristCalibration":
        return cls(
            wrist0_R=_orthonormalize(wrist_pose[:3, :3].copy()),
            wrist0_p=wrist_pose[:3, 3].copy(),
            tool0_R=_orthonormalize(tool_pose[:3, :3].copy()),
            tool0_p=tool_pose[:3, 3].copy(),
        )


def wrist_to_tool_target(
    wrist_pose: np.ndarray,
    calib: WristCalibration,
    position_scale: float,
    track_orientation: bool,
    align_R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a live wrist pose to a desired robot tool target.

    Parameters
    ----------
    align_R : optional 3x3 rotation mapping AVP-world vectors into robot-world
        vectors. This corrects axis-direction mismatches (e.g. the operator
        facing the robot flips left/right and forward/back). Defaults to
        identity. Prefer this over a negative `position_scale`, which would
        also invert the vertical axis.

    Returns
    -------
    (target_R, target_p) : 3x3 rotation and 3-vector position for the tool body,
    expressed in the robot world (the same frame MuJoCo reports body poses in).
    """
    if align_R is None:
        align_R = np.eye(3)

    wrist_R = _orthonormalize(wrist_pose[:3, :3])
    wrist_p = wrist_pose[:3, 3]

    # --- position: scaled translation of the wrist since calibration ---
    # The delta is measured in AVP world; rotate it into robot world before use.
    dp = wrist_p - calib.wrist0_p
    dp_robot = align_R @ dp
    target_p = calib.tool0_p + position_scale * dp_robot

    if not track_orientation:
        return calib.tool0_R.copy(), target_p

    # --- orientation: apply the wrist's relative rotation onto the tool ---
    # R_rel maps wrist0 -> wrist in AVP world; conjugate by align_R to express
    # the same relative rotation in robot world, then compose with tool start.
    R_rel = wrist_R @ calib.wrist0_R.T
    R_rel_robot = align_R @ R_rel @ align_R.T
    target_R = R_rel_robot @ calib.tool0_R
    return _orthonormalize(target_R), target_p


def rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Angular error as a 3-vector (axis * angle) in the world frame.

    Uses the skew-symmetric part of R_target @ R_current^T. Suitable as the
    orientation residual for damped least squares IK.
    """
    R_err = R_target @ R_current.T
    # Vee of the skew part; first-order valid for small angles, robust enough
    # for iterative IK with clamping.
    w = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) * 0.5
    # Scale by angle/sin(angle) for larger errors.
    cos_theta = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    s = np.linalg.norm(w)
    if s > 1e-9 and theta > 1e-6:
        w = w / s * theta
    return w
