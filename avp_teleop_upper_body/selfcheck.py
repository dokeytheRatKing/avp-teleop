"""Offline self-checks for the upper-body teleop pipeline (no AVP hardware).

Run inside the AVP conda env:

    python -m avp_teleop_upper_body.selfcheck

Covers:
  1. transport: HeadFrame round-trip + mixed head/left/right demux over UDP
  2. pose filter: EMA translation + SLERP rotation (pass-through, half-step, reset)
  3. teleop MJCF loads; 20 body actuators + both hands' fingers resolve
  4. merged whole-body IK places head + both tools on reachable targets
  5. auto-compensation: a head target moves the torso while the hands hold still
  6. hard limits: QP velocity/acceleration caps bound the per-tick motion
  7. soft smoothing: DampingTask + LowAccelerationTask cut peak speed/accel
  8. finger curl is monotonic from a synthetic "open" to "fist" pose
  9. end-to-end tick: synthetic head + two hands -> finite ctrl + stable step
"""

from __future__ import annotations

import sys
import time

import numpy as np
import mujoco
import pinocchio as pin

from avp_teleop.robot_interface import SimRobot
from avp_teleop.transport import HandFrame, HandFramePublisher
from avp_teleop.retarget.hand_retarget import HandRetargeter
from avp_teleop.retarget.frames import WristCalibration, wrist_to_tool_target
from avp_teleop.selfcheck import _synthetic_hand

from avp_teleop_upper_body.config import (
    default_config, MJCF_PATH, BODY_JOINTS, BODY_HOME, HEAD_FRAME_BODY,
    TOOL_BODY, all_finger_joints, finger_specs,
)
from avp_teleop_upper_body.transport import (
    HeadFrame, HeadFramePublisher, UpperBodySubscriber,
)
from avp_teleop_upper_body.whole_body_ik import WholeBodyIK
from avp_teleop_upper_body.pose_filter import PoseFilter


def _ok(name): print(f"  [PASS] {name}")
def _fail(name, msg): print(f"  [FAIL] {name}: {msg}")


def _body_qpos_adr(model):
    return [int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in BODY_JOINTS]


def _fk_frames(model, data, body_adr, q_body, names):
    """World (R, p) of each named frame with body joints at q_body, rest at 0."""
    data.qpos[:] = 0.0
    for adr, qi in zip(body_adr, q_body):
        data.qpos[adr] = qi
    mujoco.mj_forward(model, data)
    out = {}
    for n in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        out[n] = (data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy())
    return out


def check_transport() -> bool:
    # HeadFrame byte round-trip.
    head = np.eye(4, dtype=np.float32)
    head[:3, 3] = [0.1, 0.0, 1.4]
    hf = HeadFrame(valid=True, seq=3, stamp=9.0, head=head)
    g = HeadFrame.from_bytes(hf.to_bytes())
    if not (g.valid and g.seq == 3 and np.allclose(g.head, head, atol=1e-6)):
        _fail("transport round-trip", "HeadFrame decode mismatch")
        return False

    # Mixed demux over a real loopback socket.
    host, port = "127.0.0.1", 9899
    sub = UpperBodySubscriber(host, port, timeout_s=0.5)
    hand_pub = HandFramePublisher(host, port)
    head_pub = HeadFramePublisher(host, port)
    wrist = np.eye(4, dtype=np.float32); wrist[:3, 3] = [0.2, -0.1, 0.3]
    kp = np.zeros((21, 3), dtype=np.float32)
    for side in ("left", "right"):
        hand_pub.publish(HandFrame(side, True, 0, 0.0, 0.0, wrist, kp))
    head_pub.publish(HeadFrame(True, 0, 0.0, head))
    time.sleep(0.05)
    hands, got_head = sub.poll()
    sub.close(); hand_pub.close(); head_pub.close()

    if set(hands) == {"left", "right"} and got_head is not None \
            and np.allclose(got_head.head, head, atol=1e-6):
        _ok("transport: HeadFrame round-trip + head/left/right demux")
        return True
    _fail("transport", f"demux got hands={set(hands)}, head={got_head is not None}")
    return False


def check_pose_filter() -> bool:
    """EMA(translation) + SLERP(rotation): pass-through, half-step, valid SO(3)."""
    Rz1 = pin.exp3(np.array([0.0, 0.0, 1.0]))   # 1 rad about +z

    # alpha=1.0 -> pass-through (matches the old unsmoothed behaviour).
    f1 = PoseFilter(1.0, 1.0)
    f1.filter(np.zeros(3), np.eye(3))           # first sample initialises
    p, R = f1.filter(np.array([1.0, 0.0, 0.0]), Rz1)
    if not (np.allclose(p, [1.0, 0.0, 0.0]) and np.allclose(R, Rz1, atol=1e-9)):
        _fail("pose filter", "alpha=1 is not pass-through")
        return False

    # alpha=0.5 -> first sample verbatim, second moves halfway.
    f = PoseFilter(0.5, 0.5)
    p0, _ = f.filter(np.zeros(3), np.eye(3))
    if not np.allclose(p0, np.zeros(3)):
        _fail("pose filter", "first sample not taken verbatim")
        return False
    p, R = f.filter(np.array([1.0, 0.0, 0.0]), Rz1)
    half = pin.exp3(np.array([0.0, 0.0, 0.5]))  # SLERP(I, Rz(1), 0.5) = Rz(0.5)
    ortho = np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    if not (np.allclose(p, [0.5, 0.0, 0.0]) and np.allclose(R, half, atol=1e-9) and ortho):
        _fail("pose filter", f"half-step wrong: p={p}, R valid={ortho}")
        return False

    # reset() drops state so the next sample is taken verbatim again.
    f.reset()
    p, _ = f.filter(np.array([9.0, 9.0, 9.0]), np.eye(3))
    if not np.allclose(p, [9.0, 9.0, 9.0]):
        _fail("pose filter", "reset did not reinitialise")
        return False

    # untracked rotation (R=None) -> translation only, rotation stays None.
    p, R = PoseFilter(0.5, 0.5).filter(np.array([2.0, 0.0, 0.0]), None)
    if R is not None:
        _fail("pose filter", "rotation returned when not tracked")
        return False
    _ok("pose filter: EMA translation + SLERP rotation (pass-through/half/reset)")
    return True


def check_model(cfg):
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    robot = SimRobot(model, data, BODY_JOINTS, all_finger_joints(), HEAD_FRAME_BODY)
    n_body = len(robot._arm_act)
    n_finger = len(robot._finger_act)
    if n_body != len(BODY_JOINTS):
        _fail("model", f"{n_body}/{len(BODY_JOINTS)} body joints have actuators")
        return None
    if n_finger != len(all_finger_joints()):
        _fail("model", f"{n_finger}/{len(all_finger_joints())} finger actuators")
        return None
    _ok(f"model loads; {n_body} body + {n_finger} finger actuators")
    return model, data


def _build_ik(cfg, *, head_ori, arm_ori, damping=None, low_accel=None, enforce=None):
    return WholeBodyIK(
        MJCF_PATH, BODY_JOINTS, HEAD_FRAME_BODY,
        TOOL_BODY["left"], TOOL_BODY["right"], np.array(BODY_HOME),
        arm_position_cost=cfg.ik.arm_position_cost, arm_orientation_cost=arm_ori,
        head_position_cost=cfg.ik.head_position_cost, head_orientation_cost=head_ori,
        posture_cost=cfg.ik.posture_cost, lm_damping=cfg.ik.lm_damping,
        damping_cost=cfg.ik.damping_cost if damping is None else damping,
        low_accel_cost=cfg.ik.low_accel_cost if low_accel is None else low_accel,
        max_velocity=cfg.ik.max_velocity(), max_acceleration=cfg.ik.max_acceleration(),
        config_limit_gain=cfg.ik.config_limit_gain, control_dt=cfg.ik.control_dt,
        enforce_limits=cfg.ik.enforce_limits if enforce is None else enforce,
        solver=cfg.ik.solver,
    )


def check_merged_ik(cfg, model, data) -> bool:
    """All three frames must land on a jointly-reachable target."""
    try:
        ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    except Exception as e:
        _fail("merged IK", f"build failed: {e}")
        return False

    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)
    perturb = np.zeros(len(home))
    perturb[:4] = [0.15, -0.10, 0.10, 0.10]     # torso
    perturb[4:6] = [0.20, 0.15]                  # neck
    perturb[6:13] = [0.15, 0.10, -0.10, 0.10, 0.0, 0.0, 0.0]   # left arm
    perturb[13:20] = [0.15, 0.10, -0.10, 0.10, 0.0, 0.0, 0.0]  # right arm
    q_true = np.clip(home + perturb, ik.lower, ik.upper)

    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]
    tgt = _fk_frames(model, data, body_adr, q_true, names)
    head_t = (tgt[names[0]][1], tgt[names[0]][0])
    left_t = (tgt[names[1]][1], tgt[names[1]][0])
    right_t = (tgt[names[2]][1], tgt[names[2]][0])

    q = home.copy()
    for _ in range(80):
        q = ik.solve(q, head_t, left_t, right_t)

    sol = _fk_frames(model, data, body_adr, q, names)
    errs = {n: float(np.linalg.norm(sol[n][1] - tgt[n][1])) for n in names}
    in_lim = bool(np.all(q >= ik.lower - 1e-6) and np.all(q <= ik.upper + 1e-6))
    worst = max(errs.values())
    if worst < 3e-3 and in_lim:
        _ok(f"merged IK tracks head+L+R (max pos err {worst*1000:.2f} mm)")
        return True
    _fail("merged IK", f"errs(mm)={ {n: round(e*1000,2) for n,e in errs.items()} }, "
          f"within_limits={in_lim}")
    return False


def check_auto_compensation(cfg, model, data) -> bool:
    """A whole-body pose dominated by torso motion: the merged solver must drive
    the torso to place the head, and re-solve the arms so the hands still track.

    The three targets come from one real perturbed config (so they ARE jointly
    reachable). The torso is bent a lot and the arm joints barely move, so the
    hand tool targets shift almost entirely because their BASES move with the
    torso -- tracking them therefore proves the arms compensate for torso motion.
    """
    ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)

    # Big torso bend + small neck; arms left at home (so hand motion is purely
    # the torso dragging the arm bases). Jointly reachable by construction.
    q_true = home.copy()
    q_true[:4] = np.clip(home[:4] + np.array([0.30, -0.20, 0.20, 0.20]),
                         ik.lower[:4], ik.upper[:4])
    q_true[4:6] = np.clip(home[4:6] + np.array([0.10, 0.10]),
                          ik.lower[4:6], ik.upper[4:6])

    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]
    tgt = _fk_frames(model, data, body_adr, q_true, names)
    head_t = (tgt[names[0]][1], tgt[names[0]][0])
    left_t = (tgt[names[1]][1], tgt[names[1]][0])
    right_t = (tgt[names[2]][1], tgt[names[2]][0])

    q = home.copy()
    for _ in range(120):
        q = ik.solve(q, head_t, left_t, right_t)

    torso_moved = float(np.abs(q[:4] - home[:4]).max())
    sol = _fk_frames(model, data, body_adr, q, names)
    errs = {n: float(np.linalg.norm(sol[n][1] - tgt[n][1])) for n in names}
    worst = max(errs.values())

    # 5 mm bar: at a large lean the posture regularizer trades a sub-5mm offset
    # for natural posture; the tight (<3mm) precision bar is check_merged_ik.
    if torso_moved > 0.05 and worst < 5e-3:
        _ok(f"auto-compensation (torso moved {torso_moved:.2f} rad; "
            f"head+hands tracked, max err {worst*1000:.2f} mm)")
        return True
    _fail("auto-compensation",
          f"torso moved {torso_moved:.3f} rad, max frame err {worst*1000:.1f} mm")
    return False


def check_limits(cfg, model, data) -> bool:
    """Hard velocity / acceleration limits actually bound the per-tick motion.

    Drives the solver toward a far (jointly reachable) target so it wants to
    move fast, then asserts: (a) no joint ever exceeds its velocity cap, (b) the
    first tick (from rest) is bounded by a_max*dt -- proof the acceleration
    limit ramps it up rather than jumping straight to the velocity cap, (c) the
    velocity cap actually engages later, and (d) reset() clears the accel state.
    """
    ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)

    perturb = np.zeros(len(home))
    perturb[:4] = [0.25, -0.20, 0.20, 0.20]                  # torso
    perturb[4:6] = [0.20, 0.15]                              # neck
    perturb[6:13] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]   # left arm
    perturb[13:20] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]  # right arm
    q_true = np.clip(home + perturb, ik.lower, ik.upper)

    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]
    tgt = _fk_frames(model, data, body_adr, q_true, names)
    head_t = (tgt[names[0]][1], tgt[names[0]][0])
    left_t = (tgt[names[1]][1], tgt[names[1]][0])
    right_t = (tgt[names[2]][1], tgt[names[2]][0])

    v_cap = ik.max_velocity          # per-joint rad/s, BODY_JOINTS order
    a_cap = ik.max_acceleration      # per-joint rad/s^2
    dt = ik.control_dt
    ik.reset()
    q = home.copy()
    tick_vel = []                    # per-tick per-joint speed (rad/s)
    for _ in range(60):
        q_prev = q.copy()
        q = ik.solve(q, head_t, left_t, right_t)
        tick_vel.append(np.abs(q - q_prev) / dt)
    tick_vel = np.array(tick_vel)    # (T, n)

    # (a) velocity cap is never exceeded by any joint (tiny numerical tol).
    worst_v = tick_vel.max(axis=0)
    if np.any(worst_v > v_cap + 1e-6):
        bad = int(np.argmax(worst_v - v_cap))
        _fail("limits", f"velocity cap exceeded: joint {bad} "
              f"{worst_v[bad]:.3f} > {v_cap[bad]:.3f} rad/s")
        return False

    # (b) first tick from rest is bounded by a_max*dt (acceleration ramp).
    first = tick_vel[0]
    if np.any(first > a_cap * dt + 1e-6):
        bad = int(np.argmax(first - a_cap * dt))
        _fail("limits", f"first-tick speed {first[bad]:.3f} exceeds a_max*dt "
              f"{a_cap[bad]*dt:.3f} (acceleration limit not applied)")
        return False
    if not (first.max() < v_cap.max() - 1e-3):
        _fail("limits", "no ramp: first-tick speed already at the velocity cap")
        return False

    # (c) the velocity limit actually engages (some joint reaches near its cap).
    if not np.any(worst_v > 0.9 * v_cap):
        _fail("limits", f"velocity limit never engaged (max {worst_v.max():.3f} "
              f"vs cap {v_cap.max():.3f}); target too close?")
        return False

    # (d) reset() clears the acceleration limit's velocity memory.
    ik.acceleration_limit.set_last_integration(np.ones(model.nv), dt)
    ik.reset()
    if np.any(ik.acceleration_limit.Delta_q_prev != 0.0):
        _fail("limits", "reset did not clear acceleration state")
        return False

    _ok(f"hard limits (v<=cap; accel ramps from rest "
        f"{first.max():.2f} -> {worst_v.max():.2f} rad/s, cap {v_cap.max():.1f})")
    return True


def check_smoothing(cfg, model, data) -> bool:
    """Soft DampingTask + LowAccelerationTask are wired into the QP correctly.

    The *default* smoothing costs sit far below the tracking costs by design, so
    on a big slew their effect on the primary motion is intentionally tiny --
    their job is to damp null-space velocity and limit jerk, not to slow the
    hands. So this check proves the mechanism rather than a bulk metric at the
    gentle default:
      (a) default costs do not break tracking (still converges),
      (b) DampingTask has the right sign: an exaggerated damping cost measurably
          lowers the peak joint speed (hard limits off, so the soft cost is what
          shapes the motion),
      (c) LowAccelerationTask has the right sign: an exaggerated low-accel cost
          measurably lowers the peak per-tick acceleration,
      (d) the low-accel task receives its velocity each tick and reset() clears
          it (so it ramps from rest after a re-anchor).
    """
    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)
    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]

    perturb = np.zeros(len(home))
    perturb[:4] = [0.25, -0.20, 0.20, 0.20]                     # torso
    perturb[4:6] = [0.20, 0.15]                                 # neck
    perturb[6:13] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]    # left arm
    perturb[13:20] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]   # right arm

    def _run(ik, n=160):
        q_true = np.clip(home + perturb, ik.lower, ik.upper)
        tgt = _fk_frames(model, data, body_adr, q_true, names)
        head_t = (tgt[names[0]][1], None)
        left_t = (tgt[names[1]][1], None)
        right_t = (tgt[names[2]][1], None)
        dt = ik.control_dt
        ik.reset()
        q = home.copy()
        speeds, accels, v_prev = [], [], np.zeros(len(home))
        for _ in range(n):
            q_prev = q.copy()
            q = ik.solve(q, head_t, left_t, right_t)
            v = (q - q_prev) / dt
            speeds.append(np.abs(v).max())
            accels.append(np.abs(v - v_prev).max() / dt)
            v_prev = v
        sol = _fk_frames(model, data, body_adr, q, names)
        err = max(float(np.linalg.norm(sol[nm][1] - tgt[nm][1])) for nm in names)
        return np.array(speeds), np.array(accels), err

    # (a) default (gentle) costs with the full stack still track the target.
    _, _, err_default = _run(_build_ik(cfg, head_ori=1.0, arm_ori=1.0))
    if not (err_default < 5e-3):
        _fail("smoothing", f"default smoothing broke tracking: {err_default*1000:.1f} mm")
        return False

    # Exaggerated costs vs no smoothing, hard limits OFF so the soft cost is the
    # only thing shaping the motion (proves each task is wired with the right
    # sign). Limits off => peaks are large; the soft cost must shrink them.
    BIG = 10.0
    sp_plain, ac_plain, _ = _run(_build_ik(cfg, head_ori=1.0, arm_ori=1.0,
                                           damping=0.0, low_accel=0.0, enforce=False))
    sp_damp, _, _ = _run(_build_ik(cfg, head_ori=1.0, arm_ori=1.0,
                                   damping=BIG, low_accel=0.0, enforce=False))
    _, ac_lowa, _ = _run(_build_ik(cfg, head_ori=1.0, arm_ori=1.0,
                                   damping=0.0, low_accel=BIG, enforce=False))

    # (b) damping lowers peak joint speed (generous 0.8x margin; ~0.27x in fact).
    if not (sp_damp.max() < 0.8 * sp_plain.max()):
        _fail("smoothing", f"DampingTask did not slow motion: peak speed "
              f"{sp_damp.max():.2f} vs plain {sp_plain.max():.2f} rad/s")
        return False
    # (c) low-acceleration task lowers peak per-tick acceleration.
    if not (ac_lowa.max() < 0.8 * ac_plain.max()):
        _fail("smoothing", f"LowAccelerationTask did not cut accel: peak "
              f"{ac_lowa.max():.0f} vs plain {ac_plain.max():.0f} rad/s^2")
        return False

    # (d) the low-accel task gets its velocity memory each tick + reset clears it.
    ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    if ik.low_accel_task is not None:
        _run(ik, n=2)
        if ik.low_accel_task.Delta_q_prev is None:
            _fail("smoothing", "low-acceleration task never received a velocity")
            return False
        ik.reset()
        if ik.low_accel_task.Delta_q_prev is not None:
            _fail("smoothing", "reset did not clear low-acceleration state")
            return False

    _ok(f"soft smoothing wired (damping: peak v {sp_plain.max():.1f}->{sp_damp.max():.1f} "
        f"rad/s; low-accel: peak a {ac_plain.max():.0f}->{ac_lowa.max():.0f} rad/s^2; "
        f"default tracks {err_default*1000:.2f} mm)")
    return True


def check_fingers(cfg) -> bool:
    cfg.retarget.command_smoothing = 0.0
    hand = HandRetargeter(finger_specs("left"), cfg.retarget)
    open_curls = hand.finger_curls(_synthetic_hand(0.0))
    hand._smoothed.clear()
    fist_curls = hand.finger_curls(_synthetic_hand(1.0))
    avg_open = float(np.mean(list(open_curls.values())))
    avg_fist = float(np.mean(list(fist_curls.values())))
    if avg_fist > avg_open + 0.2:
        _ok(f"finger curl monotonic (open~{avg_open:.2f} < fist~{avg_fist:.2f})")
        return True
    _fail("finger curl", f"open {avg_open:.2f} vs fist {avg_fist:.2f}")
    return False


def check_end_to_end(cfg, model, data) -> bool:
    """One full tick: synthetic head + two hands -> command -> step, finite."""
    robot = SimRobot(model, data, BODY_JOINTS, all_finger_joints(), HEAD_FRAME_BODY)
    for adr, qi in zip(robot._arm_qpos_adr, BODY_HOME):
        data.qpos[adr] = qi
    mujoco.mj_forward(model, data)
    ranges = robot.joint_ranges()

    arm_ori = cfg.ik.arm_orientation_cost if cfg.track_orientation else 0.0
    head_ori = cfg.ik.head_orientation_cost if cfg.head_track_orientation else 0.0
    ik = _build_ik(cfg, head_ori=head_ori, arm_ori=arm_ori)
    retarget = {s: HandRetargeter(finger_specs(s), cfg.retarget) for s in ("left", "right")}

    # Calibrate against current robot frames; then nudge each source pose.
    end_frames = {"head": HEAD_FRAME_BODY, "left": TOOL_BODY["left"], "right": TOOL_BODY["right"]}
    src = {"head": np.eye(4), "left": np.eye(4), "right": np.eye(4)}
    calib, targets = {}, {}
    for end, body in end_frames.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        R = data.xmat[bid].reshape(3, 3).copy(); p = data.xpos[bid].copy()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        calib[end] = WristCalibration.capture(src[end], T)
    # Move sources a little.
    for end in src:
        src[end] = src[end].copy(); src[end][:3, 3] += np.array([0.04, 0.0, 0.03])

    for end in end_frames:
        track = cfg.head_track_orientation if end == "head" else cfg.track_orientation
        scale = cfg.head_position_scale if end == "head" else cfg.position_scale
        tR, tp = wrist_to_tool_target(src[end], calib[end], scale, track, cfg.align_R)
        targets[end] = (tp, tR if track else None)

    q = ik.solve(np.array(BODY_HOME), targets["head"], targets["left"], targets["right"])
    robot.command_arm(q)
    for s in ("left", "right"):
        robot.command_fingers(retarget[s].joint_targets(_synthetic_hand(0.6), ranges))
    mujoco.mj_step(model, data)

    if np.all(np.isfinite(data.ctrl)) and np.all(np.isfinite(data.qpos)):
        _ok("end-to-end tick (finite ctrl + stable step)")
        return True
    _fail("end-to-end tick", "non-finite ctrl/qpos after step")
    return False


def main() -> int:
    cfg = default_config()
    print("Running upper-body teleop self-checks...\n")
    results = [check_transport(), check_pose_filter()]

    loaded = check_model(cfg)
    if loaded is None:
        print("\nModel check failed; skipping IK/finger checks.")
        return 1
    model, data = loaded

    results.append(check_merged_ik(cfg, model, data))
    results.append(check_auto_compensation(cfg, model, data))
    results.append(check_limits(cfg, model, data))
    results.append(check_smoothing(cfg, model, data))
    results.append(check_fingers(cfg))
    results.append(check_end_to_end(cfg, model, data))

    n_pass = sum(results) + 1   # +1 for the model check
    n_total = len(results) + 1
    print(f"\n{n_pass}/{n_total} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
