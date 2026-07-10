"""Save / load a whole-body posture (chassis + torso + neck + both arms).

A pose is stored as JSON keyed by *joint name* (not position), so it stays
correct even if ``BODY_JOINTS`` order ever changes and is human-readable /
hand-editable. Because it is name-keyed, an older 20-joint (pre-chassis) file
still loads fine -- any joint missing from the file (e.g. the 3 base DOFs)
defaults to its ``BODY_HOME`` value (the base at the origin). Files live in
:data:`POSE_DIR` (``avp_teleop_upper_body/poses``) and a bare name
(``"reach_forward"``) resolves to ``poses/reach_forward.json``.

Used by both the interactive editor (:mod:`avp_teleop_upper_body.pose_editor`,
which writes them) and the teleop loop (:mod:`avp_teleop_upper_body.sim_teleop`,
which reads one via ``--init-pose`` to seed the initial / rest posture).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

from avp_teleop_upper_body.config import BODY_JOINTS, BODY_HOME

__all__ = [
    "POSE_DIR",
    "resolve_path",
    "list_poses",
    "save_pose",
    "load_pose",
    "body_vector",
]

POSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poses")


def resolve_path(name_or_path: str) -> str:
    """Turn a bare pose name into ``poses/<name>.json``; pass paths through."""
    if os.path.sep in name_or_path or name_or_path.endswith(".json"):
        return os.path.abspath(name_or_path)
    return os.path.join(POSE_DIR, f"{name_or_path}.json")


def list_poses() -> List[str]:
    """Names of saved poses in :data:`POSE_DIR` (without the ``.json``)."""
    if not os.path.isdir(POSE_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(POSE_DIR) if f.endswith(".json")
    )


def save_pose(
    name_or_path: str,
    joint_angles: Dict[str, float],
    *,
    note: str = "",
) -> str:
    """Write ``{joint: angle}`` to a JSON pose file and return its path.

    Only the joints in ``joint_angles`` are written; reading back fills any
    missing body joint from :data:`BODY_HOME`.
    """
    path = resolve_path(name_or_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "avp_upper_body_pose_v1",
        "note": note,
        # Stored joint-name -> angle (rad). Order kept for readability only.
        "joints": {n: float(joint_angles[n]) for n in joint_angles},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def load_pose(name_or_path: str) -> Dict[str, float]:
    """Read a pose file and return its ``{joint: angle}`` mapping."""
    path = resolve_path(name_or_path)
    with open(path, "r") as fh:
        payload = json.load(fh)
    joints = payload.get("joints", payload)  # tolerate a bare {joint: angle} map
    return {str(k): float(v) for k, v in joints.items()}


def body_vector(
    pose: Dict[str, float],
    body_joints: Sequence[str] = BODY_JOINTS,
    *,
    fallback: Sequence[float] = BODY_HOME,
) -> np.ndarray:
    """Project a ``{joint: angle}`` map onto the fixed ``body_joints`` order.

    Joints absent from ``pose`` fall back to ``BODY_HOME`` so an older / partial
    file still loads. Unknown joints in ``pose`` are ignored (with a warning).
    """
    fb = {n: float(v) for n, v in zip(body_joints, fallback)}
    extra = [n for n in pose if n not in fb]
    if extra:
        print(f"[pose_io] ignoring unknown joints: {extra}")
    missing = [n for n in body_joints if n not in pose]
    if missing:
        print(f"[pose_io] missing joints (using home): {missing}")
    return np.array([float(pose.get(n, fb[n])) for n in body_joints],
                    dtype=np.float64)
