"""Pose smoothing for AVP targets: EMA on translation, SLERP on rotation.

Raw AVP head / wrist poses are noisy (tracking jitter, dropped frames). Feeding
the latest frame straight into the IK makes the robot twitch. This applies a
first-order exponential filter to each *target pose* before it reaches the
solver:

    * translation -- a standard 3D vector EMA;
    * rotation    -- geodesic interpolation on SO(3) (a.k.a. quaternion SLERP),
      computed with pinocchio's ``log3`` / ``exp3`` so we never linearly blend
      rotation matrices or Euler angles (which would skew / denormalise the
      rotation and gimbal-lock).

Why SLERP and not "EMA the matrix"? Rotations live on a curved manifold (SO(3)),
not a vector space. ``alpha*R_new + (1-alpha)*R_prev`` is not a rotation in
general -- it shrinks/shears. SLERP instead walks a fraction ``alpha`` along the
*shortest rotation arc* between the two orientations, which always yields a valid
rotation and is the orientation analogue of the translation EMA. It is exactly
the rotation half of ``pinocchio.interpolate`` on an ``SE3``; we keep position
and orientation on *separate* coefficients so each can be tuned independently
(``alpha_translation`` vs ``alpha_rotation``), which a single SE(3) ``alpha``
could not do.

Convention (both channels, alpha in (0, 1]):

    filtered <- alpha * measurement + (1 - alpha) * filtered_prev   (translation)
    filtered <- slerp(filtered_prev, measurement, alpha)            (rotation)

    alpha = 1.0  -> pass-through (no smoothing, == old behaviour)
    smaller      -> smoother but laggier (averages more past frames)

The filter is stateful and per-stream: use one :class:`PoseFilter` per
end-effector (head / left / right) and :meth:`reset` it on (re)calibration so it
never smooths across the target discontinuity that a re-anchor introduces.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pinocchio as pin


def slerp_rotation(R_prev: np.ndarray, R_new: np.ndarray, alpha: float) -> np.ndarray:
    """SLERP from ``R_prev`` toward ``R_new`` by fraction ``alpha`` on SO(3).

    Walks ``alpha`` of the way along the shortest rotation arc::

        R_prev @ exp3(alpha * log3(R_prev^T @ R_new))

    ``alpha=0`` returns ``R_prev``, ``alpha=1`` returns ``R_new``; the result is
    a proper rotation for any value in between. This is the rotation part of
    ``pinocchio.interpolate`` / quaternion SLERP.
    """
    R_rel = R_prev.T @ R_new            # relative rotation, prev -> new
    return R_prev @ pin.exp3(alpha * pin.log3(R_rel))


class PoseFilter:
    """First-order EMA (translation) + SLERP (rotation) filter for one stream."""

    def __init__(
        self,
        alpha_translation: float = 1.0,
        alpha_rotation: float = 1.0,
        max_translation_jump: float = 0.0,
    ) -> None:
        self.alpha_translation = float(alpha_translation)
        self.alpha_rotation = float(alpha_rotation)
        # Outlier rejection: cap the per-tick jump of the RAW translation target
        # (metres) before it enters the EMA. AVP occasionally emits a corrupt
        # pose, or a dropout is followed by a frame taken after the hand has
        # moved far -- either way a single large jump, fed through the
        # near-singular arm Jacobian, makes the robot twitch violently. Clamping
        # the jump limits how far one frame can pull the target; a genuinely fast
        # motion just ramps over a few ticks. ``0.0`` disables the clamp (the
        # rotation channel is already limited by SLERP's fractional step).
        self.max_translation_jump = float(max_translation_jump)
        self._p: Optional[np.ndarray] = None   # filtered translation (3,)
        self._R: Optional[np.ndarray] = None   # filtered rotation (3, 3)

    def reset(self) -> None:
        """Drop the state so the next sample is taken verbatim.

        Call on (re)calibration: the target jumps to a fresh anchor and we must
        not smooth across that discontinuity (it would crawl from the old pose).
        """
        self._p = None
        self._R = None

    def filter(
        self,
        p: np.ndarray,
        R: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return the smoothed ``(p, R)`` for this sample.

        The first sample after construction / :meth:`reset` initialises the state
        and is returned unchanged (no start-up lag). ``R`` may be ``None`` when
        orientation is not tracked, in which case the returned rotation is
        ``None`` and only translation is filtered.
        """
        p = np.asarray(p, dtype=np.float64)
        if self._p is None:
            self._p = p.copy()
        else:
            # Outlier rejection first: if the raw target jumped more than
            # max_translation_jump from the last filtered output, pull it back
            # onto a sphere of that radius (limit the jump direction/magnitude)
            # before smoothing. This caps a single corrupt/post-dropout frame;
            # sustained motion converges over the next ticks.
            if self.max_translation_jump > 0.0:
                delta = p - self._p
                dist = float(np.linalg.norm(delta))
                if dist > self.max_translation_jump:
                    p = self._p + delta * (self.max_translation_jump / dist)
            a = self.alpha_translation
            self._p = a * p + (1.0 - a) * self._p

        if R is None:
            return self._p.copy(), None

        R = np.asarray(R, dtype=np.float64)
        if self._R is None:
            self._R = R.copy()
        else:
            self._R = slerp_rotation(self._R, R, self.alpha_rotation)
        return self._p.copy(), self._R.copy()
