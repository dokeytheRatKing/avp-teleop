"""Finger retargeting: 21 human keypoints -> robot finger joint targets.

For each finger we estimate a single "curl" scalar in [0, 1] from the human
hand geometry, then distribute it across that finger's robot joints. This is the
geometric (non-neural) counterpart to GeoRT, chosen because the BrainCo hand is
effectively a curl-only hand (one flexion DOF per finger that matters for
teleop) and because it needs no training and no torch.

Curl estimate
-------------
A finger's "reach" is the straight-line tip-to-MCP distance divided by the
finger's fully-extended length (sum of segment lengths). reach ~= 1 when the
finger is straight, and drops as it curls. We map reach -> curl with a
configurable [open, closed] band and clamp to [0, 1].

This is translation-invariant (works directly on wrist-local keypoints) and
robust to hand-size differences because it is normalized by the operator's own
finger length, measured per-frame.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from avp_teleop.config import FingerSpec, RetargetConfig


def _finger_reach(keypoints: np.ndarray, chain: List[int]) -> float:
    """reach = |tip - mcp| / (sum of segment lengths). ~1 straight, ->small curled."""
    pts = keypoints[chain]  # (4, 3): mcp, pip, dip, tip
    seg_len = 0.0
    for i in range(len(pts) - 1):
        seg_len += float(np.linalg.norm(pts[i + 1] - pts[i]))
    if seg_len < 1e-6:
        return 1.0
    span = float(np.linalg.norm(pts[-1] - pts[0]))
    return span / seg_len


class HandRetargeter:
    def __init__(self, finger_specs: List[FingerSpec], cfg: RetargetConfig):
        self.specs = finger_specs
        self.cfg = cfg
        self._smoothed: Dict[str, float] = {}

    @staticmethod
    def _reach_to_curl(reach: float, open_reach: float, closed_reach: float) -> float:
        # reach == open_reach -> curl 0 ; reach == closed_reach -> curl 1.
        denom = (open_reach - closed_reach)
        if abs(denom) < 1e-6:
            return 0.0
        curl = (open_reach - reach) / denom
        return float(np.clip(curl, 0.0, 1.0))

    def finger_curls(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Return {finger_name: curl in [0,1]} with exponential smoothing."""
        out: Dict[str, float] = {}
        a = self.cfg.command_smoothing
        for spec in self.specs:
            reach = _finger_reach(keypoints, spec.human_chain)
            curl = self._reach_to_curl(
                reach, self.cfg.finger_open_reach, self.cfg.finger_closed_reach
            )
            prev = self._smoothed.get(spec.name, curl)
            curl = a * prev + (1.0 - a) * curl
            self._smoothed[spec.name] = curl
            out[spec.name] = curl
        return out

    def joint_targets(
        self,
        keypoints: np.ndarray,
        joint_ranges: Dict[str, tuple],
    ) -> Dict[str, float]:
        """Map finger curls onto per-joint target angles.

        Parameters
        ----------
        joint_ranges : {joint_name: (low, high)} from the MuJoCo model.

        Returns
        -------
        {joint_name: target_angle_rad}
        """
        curls = self.finger_curls(keypoints)
        targets: Dict[str, float] = {}
        for spec in self.specs:
            curl = curls[spec.name]
            for joint_name, weight in spec.joints:
                low, high = joint_ranges.get(joint_name, (0.0, 1.0))
                # Curl spans `weight` of the joint's positive range from `low`.
                targets[joint_name] = low + curl * weight * (high - low)
        return targets
