"""Record / replay trajectories for the upper-body teleop.

Two INDEPENDENT, JSON-serialised trajectory kinds live here, decoupled on
purpose (you can record either one without the other, and replay either):

1. **AVP input trajectory** (:data:`AVP_TRAJ_DIR` = ``avp_trajectory/``),
   schema ``avp_input_trajectory``. The *raw* AVP stream exactly as it left the
   headset -- per tick, the head 4x4 pose and each hand's wrist 4x4 pose, 21x3
   MANO keypoints and pinch. This is the input side of the pipeline: feeding it
   back through :mod:`sim_teleop` reproduces the full retargeting + IK + render
   WITHOUT the headset (``--replay-avp``). It contains NO robot state.

2. **Retarget trajectory** (:data:`RETARGET_TRAJ_DIR` =
   ``retargetting_trajectory/``), schema ``retarget_trajectory``. A *superset*:
   the CLI args + config snapshot that produced the run, plus per tick the raw
   AVP input, the solved robot-world end-effector targets, the 23 commanded body
   joint angles, the finger commands, and (when active) the neural prior + the
   EgoPoser skeleton. It carries everything needed to *replay with zero
   computation* (:mod:`replay_retarget` just sets the stored joints and redraws
   the stored overlays -- no IK, no estimator) and everything needed to analyse
   a run offline.

Both use a bare-name convention like :mod:`pose_io`: ``"clip1"`` resolves to
``<dir>/clip1.json``; an explicit path or ``.json`` name is passed through.

The AVP frames round-trip through the exact transport dataclasses the live
subscriber yields (:class:`avp_teleop.transport.HandFrame` +
:class:`avp_teleop_upper_body.transport.HeadFrame`), so a replayed source is
byte-for-byte substitutable for the UDP subscriber in the teleop loop.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from avp_teleop.transport import HandFrame
from avp_teleop_upper_body.transport import HeadFrame

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
AVP_TRAJ_DIR = os.path.join(_PKG_DIR, "avp_trajectory")
RETARGET_TRAJ_DIR = os.path.join(_PKG_DIR, "retargetting_trajectory")

AVP_SCHEMA = "avp_input_trajectory"
RETARGET_SCHEMA = "retarget_trajectory"
SCHEMA_VERSION = 1

__all__ = [
    "AVP_TRAJ_DIR", "RETARGET_TRAJ_DIR",
    "AVP_SCHEMA", "RETARGET_SCHEMA", "SCHEMA_VERSION",
    "resolve_path", "list_trajectories", "trim_frames",
    "AvpTrajectoryRecorder", "load_avp_trajectory", "FileAvpSource",
    "RetargetTrajectoryRecorder", "load_retarget_trajectory", "frame_target",
]


# --------------------------------------------------------------------------- #
# path helpers (mirror pose_io's bare-name convention)
# --------------------------------------------------------------------------- #
def resolve_path(name_or_path: str, base_dir: str) -> str:
    """Bare name -> ``<base_dir>/<name>.json``; a path / ``.json`` passes through."""
    if os.path.sep in name_or_path or name_or_path.endswith(".json"):
        return os.path.abspath(name_or_path)
    return os.path.join(base_dir, f"{name_or_path}.json")


def list_trajectories(base_dir: str) -> List[str]:
    """Names (without ``.json``) of the trajectory files in ``base_dir``."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(base_dir) if f.endswith(".json"))


def _mat(x) -> Optional[List]:
    return None if x is None else np.asarray(x, dtype=np.float64).tolist()


def _arr(x) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# AVP frame <-> plain-dict (used by BOTH schemas so the raw input round-trips)
# --------------------------------------------------------------------------- #
def _hand_to_dict(h: Optional[HandFrame]) -> Optional[dict]:
    """Serialise a HandFrame (or None) to a JSON-safe dict."""
    if h is None:
        return None
    return {
        "valid": bool(h.valid), "seq": int(h.seq), "stamp": float(h.stamp),
        "pinch": float(h.pinch), "wrist": _mat(h.wrist),
        "keypoints": _mat(h.keypoints),
    }


def _hand_from_dict(d: Optional[dict], side: str) -> Optional[HandFrame]:
    if d is None:
        return None
    return HandFrame(
        side=side, valid=bool(d["valid"]), seq=int(d.get("seq", 0)),
        stamp=float(d.get("stamp", 0.0)), pinch=float(d.get("pinch", 0.0)),
        wrist=np.asarray(d["wrist"], dtype=np.float32),
        keypoints=np.asarray(d["keypoints"], dtype=np.float32),
    )


def _head_to_dict(h: Optional[HeadFrame]) -> Optional[dict]:
    if h is None:
        return None
    return {"valid": bool(h.valid), "seq": int(h.seq), "stamp": float(h.stamp),
            "head": _mat(h.head)}


def _head_from_dict(d: Optional[dict]) -> Optional[HeadFrame]:
    if d is None:
        return None
    return HeadFrame(valid=bool(d["valid"]), seq=int(d.get("seq", 0)),
                     stamp=float(d.get("stamp", 0.0)),
                     head=np.asarray(d["head"], dtype=np.float32))


def _avp_frame_dict(hands: Dict[str, HandFrame],
                    head: Optional[HeadFrame]) -> dict:
    """One tick of raw AVP input: {left, right, head} as JSON-safe dicts."""
    return {
        "left": _hand_to_dict(hands.get("left")),
        "right": _hand_to_dict(hands.get("right")),
        "head": _head_to_dict(head),
    }


def _avp_frame_from_dict(d: dict) -> Tuple[Dict[str, HandFrame], Optional[HeadFrame]]:
    hands: Dict[str, HandFrame] = {}
    for side in ("left", "right"):
        h = _hand_from_dict(d.get(side), side)
        if h is not None:
            hands[side] = h
    return hands, _head_from_dict(d.get("head"))


def trim_frames(frames: List[dict], trim_seconds: float) -> List[dict]:
    """Drop the first & last ``trim_seconds`` of a frame list, re-basing ``t``.

    Frames carry a ``t`` (seconds since the first ``record``). We keep frames in
    ``[t0 + trim, t_last - trim]`` and shift their ``t`` so the survivor starts
    at 0 (clean replay timing). Used to discard the keyboard-fumble at both ends
    of a hand-triggered recording. A ``trim_seconds`` of 0 -- or a clip too short
    to survive the cut -- returns the list unchanged (the caller warns).
    """
    if trim_seconds <= 0 or len(frames) < 2:
        return frames
    t_last = frames[-1].get("t", 0.0)
    lo, hi = trim_seconds, t_last - trim_seconds
    if hi <= lo:                       # would remove everything -> keep as-is
        return frames
    kept = [f for f in frames if lo <= f.get("t", 0.0) <= hi]
    if not kept:
        return frames
    t0 = kept[0].get("t", 0.0)
    for f in kept:
        f["t"] = round(f.get("t", 0.0) - t0, 6)
    return kept


# --------------------------------------------------------------------------- #
# 1. AVP INPUT trajectory: record / load / replay-source
# --------------------------------------------------------------------------- #
class AvpTrajectoryRecorder:
    """Accumulate raw AVP ticks and write an ``avp_input_trajectory`` JSON.

    Call :meth:`record` once per control tick with the exact ``(hands, head)``
    the subscriber returned (invalid / missing ends are stored faithfully so
    the timing and dropouts replay identically). :meth:`save` writes the file.
    """

    def __init__(self, nominal_dt: float, *, note: str = "") -> None:
        self.nominal_dt = float(nominal_dt)
        self.note = note
        self._frames: List[dict] = []
        self._t0: Optional[float] = None

    def record(self, hands: Dict[str, HandFrame],
               head: Optional[HeadFrame]) -> None:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        frame = _avp_frame_dict(hands, head)
        frame["t"] = round(now - self._t0, 6)
        self._frames.append(frame)

    def __len__(self) -> int:
        return len(self._frames)

    def save(self, name_or_path: str, *, trim_seconds: float = 0.0) -> str:
        path = resolve_path(name_or_path, AVP_TRAJ_DIR)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frames = trim_frames(self._frames, trim_seconds)
        payload = {
            "schema": AVP_SCHEMA, "version": SCHEMA_VERSION,
            "metadata": {"created": time.time(), "note": self.note,
                         "nominal_dt": self.nominal_dt,
                         "n_frames": len(frames),
                         "trim_seconds": float(trim_seconds)},
            "frames": frames,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path


def load_avp_trajectory(name_or_path: str) -> dict:
    """Read an ``avp_input_trajectory`` file; returns the parsed payload."""
    path = resolve_path(name_or_path, AVP_TRAJ_DIR)
    with open(path, "r") as fh:
        payload = json.load(fh)
    if payload.get("schema") != AVP_SCHEMA:
        raise ValueError(f"{path}: not an {AVP_SCHEMA} file "
                         f"(schema={payload.get('schema')!r}).")
    return payload


class FileAvpSource:
    """A drop-in replacement for the UDP subscriber, backed by a recorded file.

    Exposes the same ``poll() -> (hands, head)`` contract as
    :class:`UpperBodySubscriber`, advancing one recorded tick per call so the
    downstream teleop loop is byte-for-byte unchanged. When ``loop`` is False
    (default) it returns empty frames once exhausted and flags :attr:`done`.
    """

    def __init__(self, name_or_path: str, *, loop: bool = False) -> None:
        payload = load_avp_trajectory(name_or_path)
        self._frames = payload["frames"]
        self.nominal_dt = float(payload["metadata"].get("nominal_dt", 1.0 / 60.0))
        self.n_frames = len(self._frames)
        self.loop = bool(loop)
        self._i = 0
        self.done = self.n_frames == 0

    def poll(self) -> Tuple[Dict[str, HandFrame], Optional[HeadFrame]]:
        if self.n_frames == 0:
            return {}, None
        if self._i >= self.n_frames:
            if self.loop:
                self._i = 0
            else:
                self.done = True
                return {}, None
        frame = self._frames[self._i]
        self._i += 1
        if self._i >= self.n_frames and not self.loop:
            self.done = True
        return _avp_frame_from_dict(frame)

    @property
    def progress(self) -> Tuple[int, int]:
        return self._i, self.n_frames

    def close(self) -> None:      # parity with the subscriber's interface
        pass


# --------------------------------------------------------------------------- #
# 2. RETARGET trajectory: record / load
# --------------------------------------------------------------------------- #
class RetargetTrajectoryRecorder:
    """Accumulate FULL retarget ticks and write a ``retarget_trajectory`` JSON.

    ``metadata`` captures the run's provenance (CLI args, resolved model path,
    body-joint order, filter/track flags). Each :meth:`record` stores the raw
    AVP input, the solved end-effector targets (+ viz orientation), the 23
    commanded body joint angles, per-side finger commands, and the neural prior
    / skeleton when present -- a superset sufficient for pure replay + analysis.
    """

    def __init__(self, *, argv: Sequence[str], model_path: str,
                 body_joints: Sequence[str], nominal_dt: float,
                 track_orientation: bool, head_track_orientation: bool,
                 extra_meta: Optional[dict] = None) -> None:
        self.metadata = {
            "created": time.time(),
            "argv": list(argv),
            "model_path": os.path.abspath(model_path),
            "body_joints": list(body_joints),
            "nominal_dt": float(nominal_dt),
            "track_orientation": bool(track_orientation),
            "head_track_orientation": bool(head_track_orientation),
        }
        if extra_meta:
            self.metadata.update(extra_meta)
        self._frames: List[dict] = []
        self._t0: Optional[float] = None

    def record(self, *, hands: Dict[str, HandFrame], head: Optional[HeadFrame],
               targets: Dict[str, Optional[Tuple[np.ndarray, Optional[np.ndarray]]]],
               viz_rot: Dict[str, Optional[np.ndarray]],
               q_body: np.ndarray,
               fingers: Optional[Dict[str, Dict[str, float]]] = None,
               neural: Optional[Tuple[float, float]] = None,
               skeleton: Optional[np.ndarray] = None) -> None:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        tgt_out = {}
        for end, tgt in targets.items():
            if tgt is None:
                tgt_out[end] = None
            else:
                p, R = tgt
                tgt_out[end] = {"p": _mat(p), "R": _mat(R)}
        frame = {
            "t": round(now - self._t0, 6),
            "avp": _avp_frame_dict(hands, head),
            "targets": tgt_out,
            "viz_rot": {e: _mat(v) for e, v in viz_rot.items()},
            "q_body": _mat(q_body),
            "fingers": ({s: {k: float(v) for k, v in ft.items()}
                         for s, ft in fingers.items()} if fingers else None),
            "neural": (None if neural is None
                       else {"pitch": float(neural[0]), "yaw": float(neural[1])}),
            "skeleton": _mat(skeleton),
        }
        self._frames.append(frame)

    def __len__(self) -> int:
        return len(self._frames)

    def save(self, name_or_path: str, *, trim_seconds: float = 0.0) -> str:
        path = resolve_path(name_or_path, RETARGET_TRAJ_DIR)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frames = trim_frames(self._frames, trim_seconds)
        self.metadata["n_frames"] = len(frames)
        self.metadata["trim_seconds"] = float(trim_seconds)
        payload = {
            "schema": RETARGET_SCHEMA, "version": SCHEMA_VERSION,
            "metadata": self.metadata, "frames": frames,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path


def load_retarget_trajectory(name_or_path: str) -> dict:
    """Read a ``retarget_trajectory`` file; returns the parsed payload."""
    path = resolve_path(name_or_path, RETARGET_TRAJ_DIR)
    with open(path, "r") as fh:
        payload = json.load(fh)
    if payload.get("schema") != RETARGET_SCHEMA:
        raise ValueError(f"{path}: not a {RETARGET_SCHEMA} file "
                         f"(schema={payload.get('schema')!r}).")
    return payload


def frame_target(frame: dict, end: str) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Pull ``(p, R)`` for one end from a retarget frame (R may be None)."""
    tgt = frame.get("targets", {}).get(end)
    if tgt is None:
        return None
    return _arr(tgt.get("p")), _arr(tgt.get("R"))

