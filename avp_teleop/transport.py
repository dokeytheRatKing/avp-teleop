"""Zero-dependency UDP transport with a fixed message schema.

This is the seam where a future ROS layer plugs in. The publisher and the
subscriber only agree on `HandFrame` (the payload) and on UDP datagrams; either
side can be swapped for a ROS publisher/subscriber without touching the
retargeting or simulation code.

Message layout (little-endian, one datagram per frame):

    magic   : 4 bytes  = b"AVPH"
    version : uint8    = 1
    side    : uint8    = 0 (left) | 1 (right)
    valid   : uint8    = 0 | 1   (tracking valid this frame)
    _pad    : uint8
    seq     : uint32   monotonically increasing frame counter
    stamp   : float64  publisher wall-clock (time.time())
    pinch   : float32  thumb-index pinch distance (metres)
    wrist   : 16 x float32  row-major 4x4 world pose of the wrist
    kpts    : 21 x 3 x float32  MANO-order keypoints in wrist-local frame

The keypoints are already reduced to the GeoRT 21-joint order by the publisher.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

_MAGIC = b"AVPH"
_VERSION = 1

# header: 4s B B B B I d f  -> magic, ver, side, valid, pad, seq, stamp, pinch
_HEADER = struct.Struct("<4sBBBBId f")
_WRIST = struct.Struct("<16f")
_KPTS = struct.Struct("<63f")  # 21 * 3
_PACKET_SIZE = _HEADER.size + _WRIST.size + _KPTS.size

_SIDE_TO_INT = {"left": 0, "right": 1}
_INT_TO_SIDE = {0: "left", 1: "right"}


@dataclass
class HandFrame:
    """One frame of retargeting input."""

    side: str
    valid: bool
    seq: int
    stamp: float
    pinch: float
    wrist: np.ndarray   # (4, 4) world pose
    keypoints: np.ndarray  # (21, 3) wrist-local positions, MANO order

    def to_bytes(self) -> bytes:
        wrist = np.ascontiguousarray(self.wrist, dtype=np.float32).reshape(16)
        kpts = np.ascontiguousarray(self.keypoints, dtype=np.float32).reshape(63)
        header = _HEADER.pack(
            _MAGIC,
            _VERSION,
            _SIDE_TO_INT.get(self.side, 1),
            1 if self.valid else 0,
            0,
            self.seq & 0xFFFFFFFF,
            float(self.stamp),
            float(self.pinch),
        )
        return header + _WRIST.pack(*wrist) + _KPTS.pack(*kpts)

    @classmethod
    def from_bytes(cls, buf: bytes) -> "HandFrame":
        if len(buf) != _PACKET_SIZE:
            raise ValueError(
                f"Bad packet size {len(buf)} (expected {_PACKET_SIZE})."
            )
        magic, ver, side, valid, _pad, seq, stamp, pinch = _HEADER.unpack_from(buf, 0)
        if magic != _MAGIC:
            raise ValueError("Bad magic; not an AVPH packet.")
        if ver != _VERSION:
            raise ValueError(f"Unsupported version {ver}.")
        off = _HEADER.size
        wrist = np.array(_WRIST.unpack_from(buf, off), dtype=np.float32).reshape(4, 4)
        off += _WRIST.size
        kpts = np.array(_KPTS.unpack_from(buf, off), dtype=np.float32).reshape(21, 3)
        return cls(
            side=_INT_TO_SIDE.get(side, "right"),
            valid=bool(valid),
            seq=int(seq),
            stamp=float(stamp),
            pinch=float(pinch),
            wrist=wrist,
            keypoints=kpts,
        )


class HandFramePublisher:
    """Sends `HandFrame`s as UDP datagrams to a fixed host:port."""

    def __init__(self, host: str, port: int):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def publish(self, frame: HandFrame) -> None:
        frame.seq = self._seq
        frame.stamp = time.time()
        self._sock.sendto(frame.to_bytes(), self._addr)
        self._seq += 1

    def close(self) -> None:
        self._sock.close()


class HandFrameSubscriber:
    """Receives `HandFrame`s. Always returns the most recent datagram, dropping
    any backlog so the consumer never lags behind the operator.

    The publisher may interleave datagrams for more than one hand (each carries
    its own ``side`` byte). ``latest_by_side`` keeps the newest frame *per side*
    so a dual-arm consumer can drive both hands; ``latest`` returns the single
    newest frame regardless of side (single-arm back-compat)."""

    def __init__(self, host: str, port: int, timeout_s: float = 0.5):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._timeout_s = timeout_s
        self._sock.settimeout(timeout_s)

    def _drain(self) -> list[HandFrame]:
        """Receive every queued datagram (newest last). [] on timeout."""
        frames: list[HandFrame] = []
        # First receive blocks up to timeout; subsequent ones are non-blocking.
        try:
            buf = self._sock.recv(_PACKET_SIZE)
        except socket.timeout:
            return frames
        try:
            frames.append(HandFrame.from_bytes(buf))
        except ValueError:
            pass

        self._sock.setblocking(False)
        try:
            while True:
                try:
                    buf = self._sock.recv(_PACKET_SIZE)
                except (BlockingIOError, socket.error):
                    break
                try:
                    frames.append(HandFrame.from_bytes(buf))
                except ValueError:
                    continue
        finally:
            self._sock.settimeout(self._timeout_s)
        return frames

    def latest(self) -> HandFrame | None:
        """Drain the socket and return the newest valid frame, or None on timeout."""
        frames = self._drain()
        return frames[-1] if frames else None

    def latest_by_side(self) -> dict[str, HandFrame]:
        """Drain the socket and return the newest valid frame for each side seen.

        Returns a possibly-empty dict keyed by ``"left"`` / ``"right"``. Because
        datagrams arrive in send order, later frames overwrite earlier ones for
        the same side, leaving the freshest pose per hand.
        """
        out: dict[str, HandFrame] = {}
        for f in self._drain():
            out[f.side] = f
        return out

    def close(self) -> None:
        self._sock.close()
