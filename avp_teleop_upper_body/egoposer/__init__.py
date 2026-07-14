"""EgoPoser kinematic prior for the whole-body teleop.

A deep-learning *prior* that hallucinates the operator's trunk posture from the
head + two hand poses the teleop already streams, and feeds it into the Pink QP
as a low-cost dynamic reference (:class:`~avp_teleop_upper_body.whole_body_ik.NeuralPostureTask`).
Balance and hand precision always dominate, so the prior improves naturalness /
anthropomorphism without ever compromising the physically-verified safety tasks.

Public entry point: :class:`EgoPoserEstimator` (streaming inference + SMPL-spine
-> robot-trunk retargeting). ``torch`` is imported lazily inside the estimator,
so importing this package is cheap and torch-free until the prior is enabled.

Based on EgoPoser (Jiang et al., ECCV 2024); the network is reimplemented, not
copied (the upstream repo carries no license file):
https://github.com/eth-siplab/EgoPoser
"""

from __future__ import annotations

from .estimator import (
    EgoPoserEstimator,
    TrunkPrior,
    _SMPL_PARENTS,
    _SMPL_MAPPED_JOINTS,
)
from .feature_builder import FeatureWindow

__all__ = [
    "EgoPoserEstimator", "TrunkPrior", "FeatureWindow",
    "_SMPL_PARENTS", "_SMPL_MAPPED_JOINTS",
]
