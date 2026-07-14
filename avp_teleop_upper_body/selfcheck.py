"""Offline self-checks for the upper-body teleop pipeline (no AVP hardware).

Run inside the AVP conda env:

    python -m avp_teleop_upper_body.selfcheck

Covers:
  1. transport: HeadFrame round-trip + mixed head/left/right demux over UDP
  2. pose filter: EMA translation + SLERP rotation (pass-through, half-step, reset)
  3. teleop MJCF loads; 23 body actuators (3 chassis + torso/neck/arms) + fingers
  4. merged whole-body IK places head + both tools on reachable targets
  5. auto-compensation: a head target moves the torso while the hands hold still
  6. whole-body base: the mobile base is recruited only for out-of-reach targets
     (priority) and the composite base DOFs map 1:1 to the MuJoCo chassis
  7. hard limits: QP velocity/acceleration caps bound the per-tick motion
  8. soft smoothing: DampingTask + LowAccelerationTask cut peak speed/accel
  9. finger curl is monotonic from a synthetic "open" to "fist" pose
 10. end-to-end tick: synthetic head + two hands -> finite ctrl + stable step
 11. EgoPoser trunk prior: disabled=no-op, biases trunk pitch, balance overrides
     an aggressive prior, and the feature/rotation/inference plumbing is finite
"""

from __future__ import annotations

import os
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
    default_config, MJCF_PATH, BODY_JOINTS, IK_KEEP_JOINTS, BODY_HOME,
    HEAD_FRAME_BODY, CHASSIS_BASE_FRAME, TORSO_LEAN_JOINTS, NEURAL_WAIST_JOINT,
    NEURAL_PITCH_JOINT,
    TOOL_BODY, CHEST_HEAD_FRAME, CHEST_HIP_FRAME, CHEST_ANKLE_FRAME,
    all_finger_joints, finger_specs,
)
from avp_teleop_upper_body.transport import (
    HeadFrame, HeadFramePublisher, UpperBodySubscriber,
)
from avp_teleop_upper_body.whole_body_ik import WholeBodyIK
from avp_teleop_upper_body.pose_filter import PoseFilter


def _ok(name): print(f"  [PASS] {name}")
def _fail(name, msg): print(f"  [FAIL] {name}: {msg}")


# Body-vector layout in BODY_JOINTS order: chassis(3) + torso(4) + neck(2) +
# left arm(7) + right arm(7) = 23. These slices index the 23-DOF body vector.
_CHASSIS = slice(0, 3)
_TORSO = slice(3, 7)
_LEAN = slice(3, 6)      # torso_joint_1/2/3: the sagittal lean spine
_HIP = 5                 # torso_joint_3: the hip joint (neural-prior pitch DOF)
_WAIST = slice(6, 7)     # torso_joint_4: pure waist yaw
_NECK = slice(7, 9)
_LARM = slice(9, 16)
_RARM = slice(16, 23)


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


def _build_ik(cfg, *, head_ori, arm_ori, damping=None, low_accel=None,
              enforce=None, com_cost=None, trunk_upright_cost=None,
              chest_over_ankle_cost=None, neural_posture_cost=None):
    return WholeBodyIK(
        MJCF_PATH, IK_KEEP_JOINTS, HEAD_FRAME_BODY,
        TOOL_BODY["left"], TOOL_BODY["right"], np.array(BODY_HOME),
        dof_names=BODY_JOINTS,
        arm_position_cost=cfg.ik.arm_position_cost, arm_orientation_cost=arm_ori,
        head_position_cost=cfg.ik.head_position_cost, head_orientation_cost=head_ori,
        posture_cost=cfg.ik.posture_cost, lm_damping=cfg.ik.lm_damping,
        damping_cost=cfg.ik.damping_costs() if damping is None else damping,
        low_accel_cost=cfg.ik.low_accel_cost if low_accel is None else low_accel,
        com_cost=cfg.ik.com_cost if com_cost is None else com_cost,
        com_cost_vertical=cfg.ik.com_cost_vertical,
        com_lm_damping=cfg.ik.com_lm_damping,
        base_frame_name=CHASSIS_BASE_FRAME,
        trunk_upright_cost=(cfg.ik.trunk_upright_cost if trunk_upright_cost is None
                            else trunk_upright_cost),
        trunk_lean_joint_names=TORSO_LEAN_JOINTS,
        chest_over_ankle_cost=(cfg.ik.chest_over_ankle_cost
                               if chest_over_ankle_cost is None
                               else chest_over_ankle_cost),
        chest_head_frame_name=CHEST_HEAD_FRAME,
        chest_hip_frame_name=CHEST_HIP_FRAME,
        chest_ankle_frame_name=CHEST_ANKLE_FRAME,
        neural_posture_cost=(cfg.ik.neural_posture_cost
                             if neural_posture_cost is None
                             else neural_posture_cost),
        neural_waist_joint_name=NEURAL_WAIST_JOINT,
        neural_pitch_joint_names=[NEURAL_PITCH_JOINT],  # hip only (torso_joint_3)
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
    # torso = [lean_1, lean_2, lean_3, waist_yaw]: leave the LEAN spine at 0 and
    # put the torso motion in the free waist yaw. A non-zero lean here would make
    # the target trunk-forward / off-balance, which the chest-over-ankle balance
    # task (see check_balance) deliberately resists -- so it would trade tracking
    # for balance and (correctly) miss this tight 3 mm bar. Keeping the target
    # balanced isolates pure merged tracking. The arms carry real joint motion so
    # the target is still a genuine merged reach.
    perturb[_TORSO] = [0.0, 0.0, 0.0, 0.15]          # waist yaw only (balanced)
    perturb[_NECK] = [0.20, 0.15]                    # neck
    perturb[_LARM] = [0.15, 0.10, -0.10, 0.10, 0.0, 0.0, 0.0]   # left arm
    perturb[_RARM] = [0.15, 0.10, -0.10, 0.10, 0.0, 0.0, 0.0]   # right arm
    # chassis (_CHASSIS) left at 0: this target is reachable without the base.
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
    reachable). The WAIST YAW (torso_joint_4) is turned a lot and the arm joints
    barely move, so the hand tool targets shift almost entirely because their
    BASES swing with the waist -- tracking them therefore proves the arms
    compensate for torso motion. We drive the waist YAW (not the lean spine): yaw
    is a freely-moving DOF, whereas the lean spine is now deliberately damped to
    the lowest priority (see check_balance), so demanding a big lean here would
    (correctly) be resisted and would not test arm compensation cleanly.
    """
    ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)

    # Big waist YAW (torso_joint_4) + small neck; LEAN spine left at 0 and arms
    # left at home (so hand motion is purely the torso swinging the arm bases).
    # Jointly reachable by construction. torso = [lean_1, lean_2, lean_3, yaw].
    # Lean is kept at 0 so the target stays balanced (a leaned target would be
    # resisted by the chest-over-ankle balance task, see check_balance, muddying
    # the arm-compensation signal).
    q_true = home.copy()
    q_true[_TORSO] = np.clip(home[_TORSO] + np.array([0.0, 0.0, 0.0, 0.60]),
                             ik.lower[_TORSO], ik.upper[_TORSO])
    q_true[_NECK] = np.clip(home[_NECK] + np.array([0.10, 0.10]),
                            ik.lower[_NECK], ik.upper[_NECK])

    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]
    tgt = _fk_frames(model, data, body_adr, q_true, names)
    head_t = (tgt[names[0]][1], tgt[names[0]][0])
    left_t = (tgt[names[1]][1], tgt[names[1]][0])
    right_t = (tgt[names[2]][1], tgt[names[2]][0])

    q = home.copy()
    for _ in range(120):
        q = ik.solve(q, head_t, left_t, right_t)

    torso_moved = float(np.abs(q[_TORSO] - home[_TORSO]).max())
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


def check_whole_body(cfg, model, data) -> bool:
    """The mobile base is a full IK DOF: recruited only for out-of-reach targets
    (whole-body priority), and its composite DOFs map 1:1 to the MuJoCo chassis.

    (a) the reduced model exposes all 23 body DOFs (chassis included);
    (b) DOF round-trip: the SAME body vector run through the IK's own FK (which
        expands the composite base joint) and through MuJoCo's FK (which sets the
        3 chassis qpos directly) agree -> the composite x/y/yaw order lines up
        with BODY_JOINTS, so the 23-vector the solver returns drives the right
        chassis actuators;
    (c) priority: a NEAR target (reachable by the arms/torso) barely moves the
        base, while a FAR target (a large world translation the upper body can't
        cover) clearly translates the base -- and tracks it -- proving the base
        is a last-resort DOF, not an eager one.
    """
    ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)
    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]

    # (a) all 23 DOFs present, chassis included.
    if ik.n != len(BODY_JOINTS):
        _fail("whole body", f"IK exposes {ik.n} DOFs, expected {len(BODY_JOINTS)}")
        return False

    # (b) composite base DOF-order round-trip: a config with the base moved.
    q_probe = home.copy()
    q_probe[_CHASSIS] = [0.30, -0.20, 0.40]              # x, y, yaw
    q_probe[_LARM] = home[_LARM] + np.array([0.10, 0.10, 0.0, 0.10, 0.0, 0.0, 0.0])
    mj = _fk_frames(model, data, body_adr, q_probe, names)
    rt_err = 0.0
    for nm in names:
        _, p_ik = ik.frame_pose(q_probe, nm)             # Pinocchio FK (composite)
        rt_err = max(rt_err, float(np.linalg.norm(p_ik - mj[nm][1])))
    if rt_err > 1e-3:
        _fail("whole body", f"base DOF round-trip mismatch {rt_err*1000:.2f} mm "
              f"(composite x/y/yaw order != BODY_JOINTS?)")
        return False

    # (c) priority: apply a common world translation to ALL three targets. A
    # small shift is coverable by the arms/torso (base should stay put); a large
    # shift is not (base must translate). Position-only targets (R=None).
    base0 = _fk_frames(model, data, body_adr, home, names)   # poses at home

    def _solve_shift(d, iters=200):
        targets = {nm: (base0[nm][1] + d, None) for nm in names}
        ik.reset()
        q = home.copy()
        for _ in range(iters):
            q = ik.solve(q, targets[names[0]], targets[names[1]], targets[names[2]])
        sol = _fk_frames(model, data, body_adr, q, names)
        err = max(float(np.linalg.norm(sol[nm][1] - targets[nm][0])) for nm in names)
        base_disp = float(np.linalg.norm(q[_CHASSIS][:2] - home[_CHASSIS][:2]))  # x/y (m)
        return q, err, base_disp

    _, _, base_near = _solve_shift(np.array([0.05, 0.0, 0.0]), iters=200)   # within reach
    # A rigid 0.5 m shift is well beyond the arms' reach, so the base must carry
    # it; give it iterations to trundle over (the base is deliberately damped, so
    # it converges slowly -- in live teleop the target moves gradually and this
    # lag is small).
    _, err_far, base_far = _solve_shift(np.array([0.50, 0.0, 0.0]), iters=400)  # beyond reach

    if base_near > 0.05:
        _fail("whole body", f"base moved {base_near*100:.1f} cm for an in-reach "
              f"target (should stay ~put; chassis damping too low?)")
        return False
    if not (base_far > 0.20 and base_far > 5.0 * max(base_near, 1e-3)):
        _fail("whole body", f"base not recruited for a far target "
              f"(moved {base_far*100:.1f} cm; near {base_near*100:.1f} cm)")
        return False
    if err_far > 1.5e-2:
        _fail("whole body", f"far target not tracked with the base "
              f"({err_far*1000:.1f} mm residual)")
        return False

    _ok(f"whole-body base (round-trip {rt_err*1000:.2f} mm; base "
        f"{base_near*100:.1f}cm near vs {base_far*100:.1f}cm far, "
        f"far tracks {err_far*1000:.1f}mm)")
    return True


def check_balance(cfg, model, data) -> bool:
    """The chest stays over the ankle (primary) + the trunk stays ~upright.

    Balance is enforced primarily by the chest-over-ankle task: the upper-body
    CoM (approximated by the head-joint / hip-joint midpoint) must stay over the
    ankle joint (torso_joint_1) in the ground plane. The trunk-upright task (sum
    of lean angles -> 0) is a soft secondary regulariser. This check asserts the
    behaviours the design must trade off:

    (a) a GRATUITOUS reach (hands forward / forward-down, head fixed) keeps the
        chest over the ankle (small horizontal offset) -- no tip-forward;
    (b) a GENUINE height change (head AND hands descending together -- a squat)
        still succeeds and tracks well, with the chest STILL over the ankle,
        because the two free spine DOFs fold like an accordion. If the lean spine
        were frozen the squat would fail to track; if balance were missing the
        chest offset would blow up (the tip-forward bug: the cited failure pose
        put the chest ~66 cm ahead of the ankle);
    (c) NO-INPUT STABILITY: with fixed targets and no operator motion, the base
        must NOT creep -- the failure mode of the mass-based ComTask (regression
        guard for the base-drift bug).
    """
    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)
    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]

    if (cfg.ik.chest_over_ankle_cost <= 0 and cfg.ik.trunk_upright_cost <= 0
            and cfg.ik.com_cost <= 0):
        _ok("balance (no balance task configured: skipped)")
        return True

    ik = _build_ik(cfg, head_ori=1.0, arm_ori=1.0)
    base0 = _fk_frames(model, data, body_adr, home, names)

    def _joint_xy(q, joint_name):
        """World (x, y) of a joint anchor with body joints at q (rest 0)."""
        data.qpos[:] = 0.0
        for adr, qi in zip(body_adr, q):
            data.qpos[adr] = qi
        mujoco.mj_forward(model, data)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return data.xanchor[jid][:2].copy()

    def _chest_offset(q):
        """Horizontal distance from the chest (head/hip midpoint) to the ankle."""
        chest = 0.5 * (_joint_xy(q, CHEST_HEAD_FRAME) + _joint_xy(q, CHEST_HIP_FRAME))
        return float(np.linalg.norm(chest - _joint_xy(q, CHEST_ANKLE_FRAME)))

    def _drive(d_hands, d_head, iters=500):
        ik.reset()
        q = home.copy()
        tgt = {names[0]: (base0[names[0]][1] + d_head, base0[names[0]][0])}
        for nm in names[1:]:
            tgt[nm] = (base0[nm][1] + d_hands, base0[nm][0])
        for _ in range(iters):
            q = ik.solve(q, tgt[names[0]], tgt[names[1]], tgt[names[2]])
        sol = _fk_frames(model, data, body_adr, q, names)
        err = max(float(np.linalg.norm(sol[nm][1] - tgt[nm][0])) for nm in names)
        return q, _chest_offset(q), err

    # (a) gratuitous forward-down reach: chest should stay over the ankle.
    _, chest_reach, _ = _drive(np.array([0.20, 0.0, -0.10]),
                               np.array([0.0, 0.0, 0.0]))
    # (b) coordinated squat: head + hands descend together -> real height change.
    q_sq, chest_squat, err_squat = _drive(np.array([0.0, 0.0, -0.20]),
                                          np.array([0.0, 0.0, -0.18]))
    squat_fold = float(np.abs(q_sq[_LEAN] - home[_LEAN]).max())   # individual fold

    # (c) no-input stability: fixed home targets, many ticks -> no base creep.
    q0, chest_idle, _ = _drive(np.zeros(3), np.zeros(3), iters=1500)
    base_creep = float(np.linalg.norm(q0[_CHASSIS][:2] - home[_CHASSIS][:2]))

    if chest_reach > 0.08:
        _fail("balance", f"chest drifted {chest_reach*100:.1f} cm ahead of the "
              f"ankle on a plain reach (tip-forward; chest_over_ankle_cost low?)")
        return False
    if err_squat > 5e-3:
        _fail("balance", f"squat failed to track ({err_squat*1000:.1f} mm; lean "
              f"spine frozen? damping_cost_lean too high?)")
        return False
    if chest_squat > 0.08:
        _fail("balance", f"chest drifted {chest_squat*100:.1f} cm off the ankle "
              f"during the squat (should fold over the ankle)")
        return False
    if squat_fold < 0.15:
        _fail("balance", f"squat barely folded the spine ({squat_fold:.2f} rad "
              f"max joint move; can't change height?)")
        return False
    if base_creep > 0.02:
        _fail("balance", f"base crept {base_creep*100:.1f} cm with no input "
              f"(ComTask-style drift regression; com_cost re-enabled?)")
        return False
    if chest_idle > 0.05:
        _fail("balance", f"chest drifted {chest_idle*100:.1f} cm off the ankle "
              f"with no input (idle instability regression)")
        return False

    _ok(f"balance: reach keeps chest {chest_reach*100:.1f} cm off ankle; squat "
        f"tracks {err_squat*1000:.1f} mm folding {squat_fold:.2f} rad at "
        f"{chest_squat*100:.1f} cm; no-input creep {base_creep*100:.1f} cm")
    return True


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
    perturb[_TORSO] = [0.25, -0.20, 0.20, 0.20]                  # torso
    perturb[_NECK] = [0.20, 0.15]                                # neck
    perturb[_LARM] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]    # left arm
    perturb[_RARM] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]    # right arm
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
    # Small lean (lowest-priority damped tier) + waist yaw; the real slew is in
    # the arms. Keeps the target reachable at the gentle default costs so part
    # (a) can assert tracking (a big lean would be resisted by design).
    perturb[_TORSO] = [0.05, -0.05, 0.05, 0.20]                     # small lean + yaw
    perturb[_NECK] = [0.20, 0.15]                                   # neck
    perturb[_LARM] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]       # left arm
    perturb[_RARM] = [0.30, 0.20, -0.20, 0.25, 0.0, 0.0, 0.0]       # right arm

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


def check_egoposer_prior(cfg, model, data) -> bool:
    """The EgoPoser trunk prior is a safe, overridable soft bias.

    Asserts the four properties that make the prior "deep learning for
    human-likeness, QP math for safety":

    (a) DISABLED = NO-OP: with neural_posture_cost=0 the solver produces a
        bit-identical solution to the current pipeline (regression guard: the
        default teleop path is byte-for-byte unchanged);
    (b) BIAS: with the prior ON and a modest trunk-pitch target on a balanced
        reach, the trunk pitch (sum of the lean joints) moves toward the target
        vs. the prior-off solution;
    (c) SAFETY OVERRIDE: an AGGRESSIVE off-balance prior target does NOT tip the
        robot -- the chest stays over the ankle -- because ChestOverAnkleTask
        (cost 50) outweighs the low-cost prior (~0.8);
    (d) PLUMBING: the feature window builds a finite (1, W, 54) input and the
        6D rotation round-trips; if torch is present, a randomly-initialised
        EgoPoserEstimator runs end-to-end and returns a finite (pitch, yaw).
    """
    from avp_teleop_upper_body.egoposer import EgoPoserEstimator, FeatureWindow
    from avp_teleop_upper_body.egoposer.rotations import matrot2sixd, sixd2matrot
    from scipy.spatial.transform import Rotation

    body_adr = _body_qpos_adr(model)
    home = np.array(BODY_HOME)
    names = [HEAD_FRAME_BODY, TOOL_BODY["left"], TOOL_BODY["right"]]

    # A balanced reach target (waist yaw only, lean spine at 0), reused below.
    base0 = _fk_frames(model, data, body_adr, home, names)
    d_hands = np.array([0.15, 0.0, 0.0])
    tgt = {names[0]: (base0[names[0]][1], base0[names[0]][0])}
    for nm in names[1:]:
        tgt[nm] = (base0[nm][1] + d_hands, base0[nm][0])

    def _drive(ik, pitch=None, yaw=None, iters=300):
        ik.reset()
        q = home.copy()
        for _ in range(iters):
            if pitch is not None:
                ik.set_neural_target(pitch, yaw)
            q = ik.solve(q, tgt[names[0]], tgt[names[1]], tgt[names[2]])
        return q

    def _chest_offset(q):
        def _jxy(joint):
            data.qpos[:] = 0.0
            for adr, qi in zip(body_adr, q):
                data.qpos[adr] = qi
            mujoco.mj_forward(model, data)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            return data.xanchor[jid][:2].copy()
        chest = 0.5 * (_jxy(CHEST_HEAD_FRAME) + _jxy(CHEST_HIP_FRAME))
        return float(np.linalg.norm(chest - _jxy(CHEST_ANKLE_FRAME)))

    # (a) disabled == no-op (bit-identical to prior-off).
    ik_off = _build_ik(cfg, head_ori=1.0, arm_ori=1.0, neural_posture_cost=0.0)
    if ik_off.neural_task is not None:
        _fail("egoposer", "neural task built with cost 0 (should be absent)")
        return False
    q_off = _drive(ik_off)

    ik_off2 = _build_ik(cfg, head_ori=1.0, arm_ori=1.0, neural_posture_cost=0.0)
    q_off2 = _drive(ik_off2)
    if not np.array_equal(q_off, q_off2):
        _fail("egoposer", "disabled path is not deterministic/identical")
        return False

    # (b) modest prior biases the trunk pitch toward the target.
    # Mirror the deployed config: with the prior ON, sim_teleop DISABLES
    # trunk_upright_cost (its sum->0 pull would cancel the hip-joint bias), so
    # build ik_on the same way -- otherwise upright would fight the hip target.
    ik_on = _build_ik(cfg, head_ori=1.0, arm_ori=1.0, neural_posture_cost=0.8,
                      trunk_upright_cost=0.0)
    if ik_on.neural_task is None:
        _fail("egoposer", "neural task not built with cost > 0")
        return False
    def _hand_err(q):
        sol = _fk_frames(model, data, body_adr, q, names)
        return max(float(np.linalg.norm(sol[nm][1] - tgt[nm][0])) for nm in names)

    pitch_tgt = 0.30
    q_bias = _drive(ik_on, pitch=pitch_tgt, yaw=0.0)
    # Pitch is now tracked on the HIP joint (torso_joint_3) alone, not the lean
    # sum, so measure the hip angle.
    hip_off = float(q_off[_HIP])
    hip_bias = float(q_bias[_HIP])
    err_off = _hand_err(q_off)
    err_bias = _hand_err(q_bias)
    # The biased solution's hip angle should move toward the target (larger
    # than the prior-off hip) by a clear margin, WITHOUT wrecking hand
    # tracking (precision preserved at a realistic bias magnitude).
    if not (hip_bias > hip_off + 0.02):
        _fail("egoposer", f"prior did not bias trunk pitch at the hip "
              f"(off={hip_off:.3f}, on={hip_bias:.3f}, target={pitch_tgt})")
        return False
    if err_bias > err_off + 0.01:
        _fail("egoposer", f"modest prior hurt hand tracking too much "
              f"({err_off*1000:.1f} -> {err_bias*1000:.1f} mm)")
        return False

    # (c) SAFETY OVERRIDE: an AGGRESSIVE off-balance prior target cannot tip the
    # robot -- the chest stays over the ankle (ChestOverAnkleTask, cost 50) -- and
    # the low-cost prior (0.8) is strongly ATTENUATED, never reaching its target
    # (proof the balance/precision tasks dominate). A big forward pitch that the
    # QP *did* honour would fold the body and pull fixed-height hands out of
    # reach, so we assert the prior is suppressed rather than followed.
    aggr_tgt = 1.2
    q_aggr = _drive(ik_on, pitch=aggr_tgt, yaw=0.0)
    chest_aggr = _chest_offset(q_aggr)
    hip_aggr = float(q_aggr[_HIP])
    if chest_aggr > 0.08:
        _fail("egoposer", f"aggressive prior tipped the robot: chest "
              f"{chest_aggr*100:.1f} cm off the ankle (balance must override)")
        return False
    if hip_aggr > 0.8 * aggr_tgt:
        _fail("egoposer", f"aggressive prior not attenuated (hip {hip_aggr:.2f} "
              f">= 0.8*{aggr_tgt}; low-cost prior should be dominated)")
        return False

    # (d) plumbing: feature window + rotation round-trip (+ torch forward if any).
    R = Rotation.from_euler("XYZ", [0.3, -0.5, 0.2]).as_matrix()
    if np.abs(R - sixd2matrot(matrot2sixd(R))).max() > 1e-9:
        _fail("egoposer", "6D rotation round-trip failed")
        return False
    fw = FeatureWindow(window_size=cfg.egoposer.window_size, align_R=cfg.align_R)
    Th = np.eye(4); Th[:3, 3] = [0.0, 0.0, 1.5]
    Tl = np.eye(4); Tl[:3, 3] = [0.2, 0.1, 0.9]
    Tr = np.eye(4); Tr[:3, 3] = [-0.2, 0.1, 0.9]
    for _ in range(5):
        fw.push(Th, Tl, Tr)
    win = fw.as_batch()
    if win.shape != (1, cfg.egoposer.window_size, 54) or not np.isfinite(win).all():
        _fail("egoposer", f"feature window bad shape/finite: {win.shape}")
        return False

    torch_note = "torch absent (forward skipped)"
    est = EgoPoserEstimator(weights_path=None, window_size=cfg.egoposer.window_size,
                            align_R=cfg.align_R, allow_random=True)
    if est.available():
        prior = None
        for _ in range(cfg.egoposer.window_size + 2):
            prior = est.predict(Th, Tl, Tr, with_skeleton=True)
        if prior is None or not np.isfinite([prior.pitch, prior.yaw]).all():
            _fail("egoposer", "random-net estimator produced no finite prior")
            return False
        # skeleton FK (for --visualize-prior): 22 finite pelvis-local points
        # (full SMPL body: pelvis + the 21 pose_body joints).
        if (prior.skeleton is None or prior.skeleton.shape != (22, 3)
                or not np.isfinite(prior.skeleton).all()):
            _fail("egoposer", f"skeleton FK bad: {getattr(prior.skeleton,'shape',None)}")
            return False
        torch_note = (f"random-net forward OK (pitch={prior.pitch:.2f}, "
                      f"yaw={prior.yaw:.2f}); skeleton (22,3) finite")

        # If the released weights are bundled, prove they load into the clean
        # reimplementation (strict=True key parity) and produce a finite prior.
        wpath = cfg.egoposer.weights_path
        if wpath and os.path.isfile(wpath):
            import torch as _torch
            from avp_teleop_upper_body.egoposer.network import AvatarNet
            sd = _torch.load(wpath, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "params" in sd:
                sd = sd["params"]
            net = AvatarNet(input_dim=60, num_layer=3, embed_dim=256, nhead=8,
                            spatial_normalization=True, shape_estimation=False)
            missing, unexpected = [], []
            try:
                net.load_state_dict(sd, strict=True)
            except Exception as e:
                _fail("egoposer", f"released weights don't fit the reimpl "
                      f"(strict load failed): {e}")
                return False
            real = EgoPoserEstimator(weights_path=wpath,
                                     window_size=cfg.egoposer.window_size,
                                     align_R=cfg.align_R)
            if not real.available():
                _fail("egoposer", f"bundled weights failed to load: {real.reason}")
                return False
            rp = None
            for _ in range(cfg.egoposer.window_size + 2):
                rp = real.predict(Th, Tl, Tr)
            if rp is None or not np.isfinite([rp.pitch, rp.yaw]).all():
                _fail("egoposer", "bundled weights produced no finite prior")
                return False
            torch_note += (f"; released ckpt strict-loads (pitch={rp.pitch:.2f}, "
                           f"yaw={rp.yaw:.2f})")
    else:
        # torch missing is acceptable: the prior must degrade gracefully to off.
        if EgoPoserEstimator(weights_path=None).predict(Th, Tl, Tr) is not None:
            _fail("egoposer", "unavailable estimator did not return None")
            return False

    _ok(f"egoposer prior: disabled=no-op; hip bias {hip_off:.2f}->{hip_bias:.2f} rad "
        f"(hands {err_off*1000:.1f}->{err_bias*1000:.1f} mm); aggressive attenuated "
        f"{aggr_tgt}->{hip_aggr:.2f} rad, chest {chest_aggr*100:.1f} cm off ankle; "
        f"{torch_note}")
    return True


def check_trajectory_io() -> bool:
    """Record/replay round-trip: AVP-input frames and retarget frames survive
    a save -> load with byte-faithful values, and FileAvpSource re-emits them."""
    import tempfile
    from avp_teleop_upper_body import trajectory_io as tio

    # --- AVP input trajectory: build 3 ticks, save, reload via FileAvpSource ---
    wrist = np.eye(4, dtype=np.float32); wrist[:3, 3] = [0.2, -0.1, 0.3]
    head = np.eye(4, dtype=np.float32); head[:3, 3] = [0.0, 0.0, 1.45]
    kp = np.arange(63, dtype=np.float32).reshape(21, 3) * 0.001
    rec = tio.AvpTrajectoryRecorder(1.0 / 60.0, note="selfcheck")
    for seq in range(3):
        hands = {s: HandFrame(s, True, seq, 0.0, 0.3, wrist, kp)
                 for s in ("left", "right")}
        rec.record(hands, HeadFrame(True, seq, 0.0, head))
    # An invalid/empty tick must survive too (timing/dropout fidelity).
    rec.record({}, None)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.json")
        rec.save(path)
        src = tio.FileAvpSource(path)
        if src.n_frames != 4:
            _fail("trajectory io", f"AVP replay has {src.n_frames} frames, want 4")
            return False
        h0, hd0 = src.poll()
        if not (set(h0) == {"left", "right"} and hd0 is not None
                and np.allclose(h0["left"].wrist, wrist, atol=1e-6)
                and np.allclose(h0["left"].keypoints, kp, atol=1e-6)
                and abs(h0["left"].pinch - 0.3) < 1e-6
                and np.allclose(hd0.head, head, atol=1e-6)):
            _fail("trajectory io", "AVP frame values not preserved on round-trip")
            return False
        src.poll(); src.poll()          # ticks 1, 2
        h3, hd3 = src.poll()            # tick 3 = the empty one
        if h3 != {} or hd3 is not None:
            _fail("trajectory io", "empty AVP tick not preserved")
            return False
        if not src.done:
            _fail("trajectory io", "FileAvpSource did not flag done at the end")
            return False

    # --- trim_frames: drop first/last N seconds, re-base t to 0 --------------- #
    fs = [{"t": i * 0.5} for i in range(11)]          # 0.0 .. 5.0 s, 0.5 s step
    kept = tio.trim_frames([dict(f) for f in fs], 1.0)  # keep [1.0, 4.0], rebased
    if not (len(kept) == 7 and abs(kept[0]["t"]) < 1e-9
            and abs(kept[-1]["t"] - 3.0) < 1e-9):
        _fail("trajectory io", f"trim wrong: n={len(kept)}, "
              f"t0={kept[0]['t']}, tN={kept[-1]['t']}")
        return False
    # Over-long trim (would remove everything) leaves the clip untouched.
    if len(tio.trim_frames([dict(f) for f in fs], 10.0)) != len(fs):
        _fail("trajectory io", "over-long trim did not no-op")
        return False

    # --- Retarget trajectory: metadata + a frame with targets/joints/fingers ---
    rrec = tio.RetargetTrajectoryRecorder(
        argv=["--replay-avp", "clip"], model_path=MJCF_PATH,
        body_joints=BODY_JOINTS, nominal_dt=1.0 / 60.0,
        track_orientation=False, head_track_orientation=True)
    q = np.linspace(-0.1, 0.1, len(BODY_JOINTS))
    tp = np.array([0.4, 0.1, 1.0]); tR = np.eye(3)
    targets = {"head": (np.array([0.0, 0.0, 1.5]), None),
               "left": (tp, tR), "right": (tp, None)}
    viz_rot = {"head": np.eye(3), "left": tR, "right": None}
    rrec.record(hands={"left": HandFrame("left", True, 0, 0.0, 0.0, wrist, kp)},
                head=HeadFrame(True, 0, 0.0, head), targets=targets,
                viz_rot=viz_rot, q_body=q,
                fingers={"left": {"j1": 0.5}}, neural=(0.2, 0.1),
                skeleton=np.zeros((22, 3)))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.json")
        rrec.save(path)
        payload = tio.load_retarget_trajectory(path)
        meta, frames = payload["metadata"], payload["frames"]
        if not (meta["argv"] == ["--replay-avp", "clip"]
                and meta["body_joints"] == list(BODY_JOINTS)
                and os.path.isabs(meta["model_path"])):
            _fail("trajectory io", "retarget metadata not preserved")
            return False
        fr = frames[0]
        p_rt, R_rt = tio.frame_target(fr, "left")
        if not (len(frames) == 1
                and np.allclose(fr["q_body"], q, atol=1e-9)
                and np.allclose(p_rt, tp, atol=1e-9)
                and R_rt is not None and np.allclose(R_rt, tR, atol=1e-9)
                and tio.frame_target(fr, "head")[1] is None
                and abs(fr["fingers"]["left"]["j1"] - 0.5) < 1e-9
                and abs(fr["neural"]["pitch"] - 0.2) < 1e-9
                and np.asarray(fr["skeleton"]).shape == (22, 3)):
            _fail("trajectory io", "retarget frame values not preserved")
            return False

    _ok("trajectory io: AVP + retarget record/replay round-trip (values, "
        "empty ticks, metadata)")
    return True


def main() -> int:
    cfg = default_config()
    print("Running upper-body teleop self-checks...\n")
    results = [check_transport(), check_pose_filter(), check_trajectory_io()]

    loaded = check_model(cfg)
    if loaded is None:
        print("\nModel check failed; skipping IK/finger checks.")
        return 1
    model, data = loaded

    results.append(check_merged_ik(cfg, model, data))
    results.append(check_auto_compensation(cfg, model, data))
    results.append(check_whole_body(cfg, model, data))
    results.append(check_balance(cfg, model, data))
    results.append(check_limits(cfg, model, data))
    results.append(check_smoothing(cfg, model, data))
    results.append(check_fingers(cfg))
    results.append(check_end_to_end(cfg, model, data))
    results.append(check_egoposer_prior(cfg, model, data))

    n_pass = sum(results) + 1   # +1 for the model check
    n_total = len(results) + 1
    print(f"\n{n_pass}/{n_total} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
