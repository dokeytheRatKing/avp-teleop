"""Transport additions for upper-body teleop: a head 6-DoF channel.

The dual-arm package already defines ``HandFrame`` (magic ``b"AVPH"``) and its
UDP publisher/subscriber. Here we add a parallel ``HeadFrame`` (magic
``b"AVPE"``) carrying the AVP head pose, and a single subscriber that listens on
one port and dispatches datagrams by their 4-byte magic. The existing
``avp_teleop.transport`` is imported and left untouched, so the dual-arm path
keeps working byte-for-byte.

Head message layout (little-endian, one datagram per frame):

    magic   : 4 bytes  = b"AVPE"
    version : uint8    = 1
    valid   : uint8    = 0 | 1
    _rsvd   : uint16   (padding / reserved)
    seq     : uint32   monotonically increasing frame counter
    stamp   : float64  publisher wall-clock (time.time())
    head    : 16 x float32  row-major 4x4 world pose of the head (camera) frame
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

# Reuse the hand message + its wire constants unchanged.
from avp_teleop.transport import (
    HandFrame,
    _MAGIC as _HAND_MAGIC,
    _PACKET_SIZE as _HAND_PACKET_SIZE,
)

_HEAD_MAGIC = b"AVPE"
_VERSION = 1

# magic(4s) ver(B) valid(B) rsvd(H) seq(I) stamp(d) head(16f)
_HEAD = struct.Struct("<4sBBHId16f")
_HEAD_PACKET_SIZE = _HEAD.size

# Receive buffer big enough for any message type we may interleave on the port.
_RECV_BUF = max(_HAND_PACKET_SIZE, _HEAD_PACKET_SIZE) + 64


@dataclass
class HeadFrame:
    """One frame of head-pose input (AVP headset / robot head camera analogue)."""

    valid: bool
    seq: int
    stamp: float
    head: np.ndarray  # (4, 4) world pose

    def to_bytes(self) -> bytes:
        head = np.ascontiguousarray(self.head, dtype=np.float32).reshape(16)
        return _HEAD.pack(
            _HEAD_MAGIC,
            _VERSION,
            1 if self.valid else 0,
            0,
            self.seq & 0xFFFFFFFF,
            float(self.stamp),
            *head,
        )

    @classmethod
    def from_bytes(cls, buf: bytes) -> "HeadFrame":
        if len(buf) != _HEAD_PACKET_SIZE:
            raise ValueError(
                f"Bad head packet size {len(buf)} (expected {_HEAD_PACKET_SIZE})."
            )
        unpacked = _HEAD.unpack_from(buf, 0)
        magic, ver, valid, _rsvd, seq, stamp = unpacked[:6]
        if magic != _HEAD_MAGIC:
            raise ValueError("Bad magic; not an AVPE packet.")
        if ver != _VERSION:
            raise ValueError(f"Unsupported head version {ver}.")
        head = np.array(unpacked[6:], dtype=np.float32).reshape(4, 4)
        return cls(valid=bool(valid), seq=int(seq), stamp=float(stamp), head=head)


class HeadFramePublisher:
    """Sends ``HeadFrame``s as UDP datagrams to a fixed host:port."""

    def __init__(self, host: str, port: int):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def publish(self, frame: HeadFrame) -> None:
        frame.seq = self._seq
        frame.stamp = time.time()
        self._sock.sendto(frame.to_bytes(), self._addr)
        self._seq += 1

    def close(self) -> None:
        self._sock.close()


class UpperBodySubscriber:
    """Receives interleaved hand + head datagrams on one port.

    ``poll()`` drains the whole socket backlog once and returns the freshest
    frame *per channel* seen in that drain: a dict of the newest ``HandFrame``
    per side, plus the newest ``HeadFrame`` (or ``None``). Dispatch is by the
    leading 4-byte magic, so the message types can share a single port.
    """

    def __init__(self, host: str, port: int, timeout_s: float = 0.5):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._timeout_s = timeout_s
        self._sock.settimeout(timeout_s)

    def _decode(self, buf: bytes):
        """Return a HandFrame / HeadFrame for a datagram, or None if unknown."""
        if len(buf) < 4:
            return None
        magic = bytes(buf[:4])
        try:
            if magic == _HAND_MAGIC:
                return HandFrame.from_bytes(buf)
            if magic == _HEAD_MAGIC:
                return HeadFrame.from_bytes(buf)
        except ValueError:
            return None
        return None

    def poll(self):
        """Drain the socket; return ({side: HandFrame}, HeadFrame|None).

        Blocks up to ``timeout_s`` for the first datagram, then consumes any
        backlog non-blocking so the consumer always acts on the newest data.
        """
        hands: dict[str, HandFrame] = {}
        head: "HeadFrame | None" = None

        try:
            buf = self._sock.recv(_RECV_BUF)
        except socket.timeout:
            return hands, head

        def _absorb(b: bytes):
            nonlocal head
            obj = self._decode(b)
            if isinstance(obj, HandFrame):
                hands[obj.side] = obj
            elif isinstance(obj, HeadFrame):
                head = obj

        _absorb(buf)

        self._sock.setblocking(False)
        try:
            while True:
                try:
                    buf = self._sock.recv(_RECV_BUF)
                except (BlockingIOError, socket.error):
                    break
                _absorb(buf)
        finally:
            self._sock.settimeout(self._timeout_s)
        return hands, head

    def close(self) -> None:
        self._sock.close()
