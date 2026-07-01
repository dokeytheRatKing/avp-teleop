"""AVP publisher: read the Apple Vision Pro stream and broadcast hand frames.

This is the evolution of the original `hand_pose_streaming.py`: instead of
printing keypoints, it extracts the wrist pose and the 21 MANO-order finger
keypoints and publishes them over UDP for the simulation (or, later, a ROS
node) to consume.

Run inside the `AVP` conda env, wearing the Vision Pro:

    python -m avp_teleop.avp_publisher
    python -m avp_teleop.avp_publisher --avp-ip 10.200.177.142 --side right
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from avp_teleop.config import default_config
from avp_teleop.transport import HandFrame, HandFramePublisher


# ARKit 25-joint -> GeoRT 21-joint (MANO/SMPL-X) order: drop the 4 metacarpals
# (indices 5, 10, 15, 20). Identical to GeoRT's WEBXR_TO_SMPLX.
WEBXR_TO_SMPLX = [
    0,
    1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24,
]


def extract_frame(data, side: str) -> HandFrame | None:
    """Build a HandFrame from one raw AVP `streamer.latest` dict.

    Returns None if this side is not currently tracked / data is degenerate.
    """
    if data is None:
        return None

    finger_key = f"{side}_fingers"
    wrist_key = f"{side}_wrist"
    pinch_key = f"{side}_pinch_distance"
    if finger_key not in data or wrist_key not in data:
        return None

    fingers = np.asarray(data[finger_key], dtype=np.float32)  # (25, 4, 4) wrist-local
    wrist = np.asarray(data[wrist_key], dtype=np.float32)      # (1, 4, 4) world
    if fingers.shape[1:] != (4, 4) or fingers.shape[0] < 25:
        return None
    if wrist.ndim == 3:
        wrist = wrist[0]
    if wrist.shape != (4, 4):
        return None

    # Wrist-local finger positions -> reduce to 21 MANO keypoints.
    local_xyz = fingers[:, :3, 3]
    keypoints = local_xyz[WEBXR_TO_SMPLX]

    if not (np.isfinite(keypoints).all() and np.isfinite(wrist).all()):
        return None

    pinch = float(data.get(pinch_key, 0.0))
    return HandFrame(
        side=side,
        valid=True,
        seq=0,
        stamp=0.0,
        pinch=pinch,
        wrist=wrist,
        keypoints=keypoints.astype(np.float32),
    )


def main() -> None:
    cfg = default_config()
    parser = argparse.ArgumentParser(description="Publish AVP hand frames over UDP.")
    parser.add_argument("--avp-ip", default=cfg.avp.connection_id,
                        help="Vision Pro IP address or 6-char room code.")
    parser.add_argument("--side", default=cfg.avp.side,
                        choices=["left", "right", "both"],
                        help="Which hand(s) to publish. 'both' streams left and "
                             "right (one datagram each) for dual-arm teleop.")
    parser.add_argument("--host", default=cfg.network.host)
    parser.add_argument("--port", type=int, default=cfg.network.port)
    parser.add_argument("--rate", type=float, default=cfg.avp.publish_rate_hz)
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]

    from avp_stream import VisionProStreamer

    print(f"[AVP] Connecting to {args.avp_ip} ...")
    streamer = VisionProStreamer(ip=args.avp_ip, record=False)
    pub = HandFramePublisher(args.host, args.port)
    print(f"[AVP] Publishing {'+'.join(sides)} hand(s) to "
          f"udp://{args.host}:{args.port} at ~{args.rate:.0f} Hz. Ctrl+C to stop.")

    period = 1.0 / max(args.rate, 1.0)
    n_sent, n_drop, last_log = 0, 0, time.time()
    try:
        while True:
            latest = streamer.latest
            any_sent = False
            for side in sides:
                frame = extract_frame(latest, side)
                if frame is not None:
                    pub.publish(frame)
                    any_sent = True
            if any_sent:
                n_sent += 1
            else:
                n_drop += 1

            now = time.time()
            if now - last_log >= 2.0:
                status = "OK" if n_sent > 0 else "NO HAND"
                print(f"[AVP] {status}: sent={n_sent} dropped={n_drop} "
                      f"(last 2s)", end="\r", flush=True)
                n_sent, n_drop, last_log = 0, 0, now
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[AVP] Stopped.")
    finally:
        pub.close()
        close = getattr(streamer, "cleanup", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
