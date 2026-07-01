"""AVP publisher for upper-body teleop: head pose + both hands over UDP.

Extends the dual-arm publisher by also streaming the AVP head 6-DoF pose
(``streamer.latest["head"]``, a 4x4 world pose) alongside the two hand frames.
Each tick sends up to three datagrams to the same host:port -- left hand, right
hand, head -- which the upper-body subscriber demultiplexes by magic.

Run inside the ``AVP`` conda env, wearing the Vision Pro:

    python -m avp_teleop_upper_body.avp_publisher --avp-ip 10.200.177.142
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# Reuse the dual-arm hand extraction + hand publisher verbatim.
from avp_teleop.avp_publisher import extract_frame
from avp_teleop.transport import HandFramePublisher

from avp_teleop_upper_body.config import default_config
from avp_teleop_upper_body.transport import HeadFrame, HeadFramePublisher


def extract_head(data) -> HeadFrame | None:
    """Build a HeadFrame from one raw AVP ``streamer.latest`` dict.

    Returns None if the head pose is absent or degenerate.
    """
    if data is None or "head" not in data:
        return None
    head = np.asarray(data["head"], dtype=np.float32)
    if head.ndim == 3:  # (1, 4, 4) -> (4, 4)
        head = head[0]
    if head.shape != (4, 4) or not np.isfinite(head).all():
        return None
    return HeadFrame(valid=True, seq=0, stamp=0.0, head=head)


def main() -> None:
    cfg = default_config()
    parser = argparse.ArgumentParser(
        description="Publish AVP head + both hands over UDP for upper-body teleop."
    )
    parser.add_argument("--avp-ip", default=cfg.avp.connection_id,
                        help="Vision Pro IP address or 6-char room code.")
    parser.add_argument("--host", default=cfg.network.host)
    parser.add_argument("--port", type=int, default=cfg.network.port)
    parser.add_argument("--rate", type=float, default=cfg.avp.publish_rate_hz)
    parser.add_argument("--no-head", action="store_true",
                        help="Stream only the two hands (no head pose).")
    args = parser.parse_args()

    from avp_stream import VisionProStreamer

    print(f"[AVP] Connecting to {args.avp_ip} ...")
    streamer = VisionProStreamer(ip=args.avp_ip, record=False)
    hand_pub = HandFramePublisher(args.host, args.port)
    head_pub = HeadFramePublisher(args.host, args.port)
    channels = "head + left + right" if not args.no_head else "left + right"
    print(f"[AVP] Publishing {channels} to udp://{args.host}:{args.port} "
          f"at ~{args.rate:.0f} Hz. Ctrl+C to stop.")

    period = 1.0 / max(args.rate, 1.0)
    n_sent, n_drop, last_log = 0, 0, time.time()
    try:
        while True:
            latest = streamer.latest
            any_sent = False

            for side in ("left", "right"):
                frame = extract_frame(latest, side)
                if frame is not None:
                    hand_pub.publish(frame)
                    any_sent = True

            if not args.no_head:
                head = extract_head(latest)
                if head is not None:
                    head_pub.publish(head)
                    any_sent = True

            if any_sent:
                n_sent += 1
            else:
                n_drop += 1

            now = time.time()
            if now - last_log >= 2.0:
                status = "OK" if n_sent > 0 else "NO DATA"
                print(f"[AVP] {status}: sent={n_sent} dropped={n_drop} "
                      f"(last 2s)", end="\r", flush=True)
                n_sent, n_drop, last_log = 0, 0, now
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[AVP] Stopped.")
    finally:
        hand_pub.close()
        head_pub.close()
        close = getattr(streamer, "cleanup", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
