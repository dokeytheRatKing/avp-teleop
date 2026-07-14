"""Build EgoPoser's sparse input window from AVP head + hand poses.

EgoPoser consumes a sliding window of per-frame 54-dim feature vectors (the
network's ``forward`` appends 6 temporal-delta dims -> 60 internally). The exact
per-frame layout, reproduced from the upstream ``prepare_data.py``
(``hmd_position_global_full_gt_list``), is three sensors -- head, left wrist,
right wrist -- concatenated block-wise:

    [ 0:18]  global 6D rotation           head(0:6)  L(6:12)  R(12:18)
    [18:36]  global 6D rotation velocity  head(18:24) L(24:30) R(30:36)
    [36:45]  global position (x, y, z)    head(36:39) L(39:42) R(42:45)
    [45:54]  position delta (frame-2-frame) head(45:48) L(48:51) R(51:54)

where "global rotation velocity" is ``matrot2sixd(R_{t-1}^{-1} R_t)`` and the
"position delta" is ``p_t - p_{t-1}``. Positions are in metres; rotations are
the world-frame orientation of each body. This matches EgoPoser's AMASS-derived
training distribution (Z-up, metres, joints 15/20/21 = head / L-wrist /
R-wrist).

The caller passes the *live* AVP source poses (the same 4x4 world poses the
teleop already reads), optionally pre-aligned into the robot world with
``AVP_TO_ROBOT_R`` for a consistent basis. Because we only ever consume the
network's **local** spine / waist rotations downstream (pelvis-relative; see
:mod:`avp_teleop_upper_body.egoposer.estimator`), the absolute world yaw of the
input does not affect the extracted trunk prior -- only the relative geometry of
head and hands does, which is preserved by any rigid world alignment.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .rotations import matrot2sixd

_N_SENSORS = 3          # head, left, right
_FEATURE_DIM = 54       # 3 * (6 rot + 6 rotvel + 3 pos + 3 posdelta)


class FeatureWindow:
    """Rolling window that turns per-tick head/L/R poses into a (W, 54) array."""

    def __init__(self, window_size: int = 80, align_R: Optional[np.ndarray] = None):
        if window_size < 2:
            raise ValueError("window_size must be >= 2 for velocity features.")
        self.window_size = int(window_size)
        self.align_R = (np.eye(3) if align_R is None
                        else np.asarray(align_R, dtype=np.float64))
        self._frames: "deque[np.ndarray]" = deque(maxlen=self.window_size)
        self._prev_R = None   # (3, 3, 3): previous world rotations of head/L/R
        self._prev_p = None   # (3, 3):    previous world positions of head/L/R

    def reset(self) -> None:
        """Clear the window (call on (re)calibration)."""
        self._frames.clear()
        self._prev_R = None
        self._prev_p = None

    def is_ready(self) -> bool:
        """True once enough frames have accumulated to fill the window."""
        return len(self._frames) >= self.window_size

    def push(self, head_T: np.ndarray, left_T: np.ndarray, right_T: np.ndarray) -> None:
        """Add one frame from three 4x4 world poses (head, left, right)."""
        R = np.stack([self._align(T[:3, :3]) for T in (head_T, left_T, right_T)])
        p = np.stack([self.align_R @ np.asarray(T[:3, 3], dtype=np.float64)
                      for T in (head_T, left_T, right_T)])

        if self._prev_R is None:
            rot_vel = np.tile(matrot2sixd(np.eye(3)), (_N_SENSORS, 1))  # identity
            pos_delta = np.zeros((_N_SENSORS, 3))
        else:
            rel = np.matmul(np.transpose(self._prev_R, (0, 2, 1)), R)   # R_prev^T R
            rot_vel = matrot2sixd(rel)
            pos_delta = p - self._prev_p

        feat = np.concatenate([
            matrot2sixd(R).reshape(-1),   # 0:18  global 6D rotation
            rot_vel.reshape(-1),          # 18:36 global 6D rotation velocity
            p.reshape(-1),                # 36:45 global position
            pos_delta.reshape(-1),        # 45:54 position delta
        ]).astype(np.float64)
        assert feat.shape == (_FEATURE_DIM,)

        self._frames.append(feat)
        self._prev_R = R
        self._prev_p = p

    def as_batch(self) -> np.ndarray:
        """Return the window as a (1, window_size, 54) float32 array.

        During warm-up (fewer than ``window_size`` frames pushed) the earliest
        frame is repeated to pad the front, so inference can start immediately
        after calibration rather than stalling for ~1.3 s at 60 Hz.
        """
        if not self._frames:
            raise RuntimeError("FeatureWindow is empty; push a frame first.")
        frames = list(self._frames)
        if len(frames) < self.window_size:
            frames = [frames[0]] * (self.window_size - len(frames)) + frames
        return np.stack(frames)[None, ...].astype(np.float32)

    def _align(self, R: np.ndarray) -> np.ndarray:
        """Rotate a world rotation matrix into the aligned basis."""
        return self.align_R @ np.asarray(R, dtype=np.float64)
