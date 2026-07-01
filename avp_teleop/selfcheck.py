"""Offline self-checks for the teleop pipeline (no AVP hardware required).

Run inside the AVP conda env:

    python -m avp_teleop.selfcheck

Covers:
  1. transport encode/decode round-trip
  2. teleop MJCF loads; arm joints + finger actuators resolve via SimRobot
  3. arm IK (legacy MuJoCo DLS) converges to a reachable synthetic tool target
  4. arm IK (Pinocchio + Pink) matches MuJoCo FK for BOTH arms (left + right)
  5. dual-arm command: two SimRobots drive disjoint actuators without clobber
  6. finger curl is monotonic from a synthetic "open" to "fist" pose
  7. end-to-end retarget tick produces finite ctrl and a stable sim step
"""

from __future__ import annotations

import sys

import numpy as np
import mujoco

from avp_teleop.config import default_config, MJCF_PATH
from avp_teleop.transport import HandFrame
from avp_teleop.robot_interface import SimRobot
from avp_teleop.retarget.arm_ik import ArmIK
from avp_teleop.retarget.hand_retarget import HandRetargeter
from avp_teleop.retarget.frames import WristCalibration, wrist_to_tool_target


def _ok(name): print(f"  [PASS] {name}")
def _fail(name, msg): print(f"  [FAIL] {name}: {msg}")


def check_transport() -> bool:
    wrist = np.eye(4, dtype=np.float32)
    wrist[:3, 3] = [0.1, -0.2, 0.3]
    kpts = np.random.RandomState(0).randn(21, 3).astype(np.float32)
    f = HandFrame("right", True, 7, 123.5, 0.04, wrist, kpts)
    g = HandFrame.from_bytes(f.to_bytes())
    if (g.side == "right" and g.valid and g.seq == 7
            and abs(g.pinch - 0.04) < 1e-6
            and np.allclose(g.wrist, wrist, atol=1e-6)
            and np.allclose(g.keypoints, kpts, atol=1e-6)):
        _ok("transport round-trip"); return True
    _fail("transport round-trip", "decoded frame mismatch"); return False


def check_model_and_robot(cfg):
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    finger_joints = [jn for spec in cfg.finger_specs for (jn, _w) in spec.joints]
    robot = SimRobot(model, data, cfg.arm_joints, finger_joints, cfg.tool_body)

    n_finger_act = len(robot._finger_act)
    if n_finger_act != len(finger_joints):
        _fail("finger actuators",
              f"{n_finger_act}/{len(finger_joints)} finger joints have actuators")
        return None
    _ok(f"model loads; {len(robot._arm_act)} arm + {n_finger_act} finger actuators")
    return model, data, robot


def check_ik(cfg, model, data, robot) -> bool:
    ik = ArmIK(model, cfg.arm_joints, cfg.tool_body,
               damping=cfg.retarget.dls_damping, max_iters=40,
               pos_tol=cfg.retarget.ik_pos_tol, max_joint_step=0.2)

    home = np.array(cfg.arm_home)
    # Build a reachable target: FK of a perturbed config.
    q_true = np.clip(home + np.array([0.2, 0.15, -0.1, 0.25, 0.1, -0.05, 0.0]),
                     ik.lower, ik.upper)
    ik._fk(q_true, data.qpos.copy())
    p_target, R_target = ik._tool_pose()

    q_sol = ik.solve(home, p_target, R_target, data.qpos.copy())
    ik._fk(q_sol, data.qpos.copy())
    p_sol, R_sol = ik._tool_pose()
    pos_err = float(np.linalg.norm(p_sol - p_target))

    if pos_err < 5e-3:
        _ok(f"arm IK (mujoco DLS) converges (pos err {pos_err*1000:.2f} mm)")
        return True
    _fail("arm IK (mujoco DLS)", f"position error {pos_err*1000:.2f} mm too large")
    return False


def _arm_tool_world(model, data, arm_joints, tool_body, q_arm):
    """MuJoCo world tool pose for an arm config (non-arm joints at qpos=0)."""
    adr = [int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool_body)
    data.qpos[:] = 0.0
    for a, qi in zip(adr, q_arm):
        data.qpos[a] = qi
    mujoco.mj_forward(model, data)
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def check_ik_pink_both(cfg, model, data) -> bool:
    """Pink solve must place the MuJoCo tool on the target for BOTH arms.

    This is the load-bearing test for the Pink backend and for dual-arm: it
    confirms the Pinocchio model built from the flattened MJCF shares MuJoCo's
    frames for each side, so solved joint angles need no coordinate transform.
    The left arm is the new check added for dual-arm teleop.
    """
    try:
        from avp_teleop.retarget.arm_ik_pink import PinkArmIK
    except Exception as e:  # pinocchio/pink not installed
        _fail("arm IK (pink)", f"import failed: {e}")
        return False

    ok = True
    for side in ("left", "right"):
        arm_joints = cfg.arm_joints_for(side)
        tool_body = cfg.tool_body_for(side)
        home = np.array(cfg.arm_home_for(side))
        ik = PinkArmIK(
            MJCF_PATH, arm_joints, tool_body, home,
            position_cost=cfg.retarget.pink_position_cost,
            orientation_cost=0.0,
            posture_cost=cfg.retarget.pink_posture_cost,
            lm_damping=cfg.retarget.pink_lm_damping,
            max_joint_step=0.5, solver_iters=cfg.retarget.pink_solver_iters,
            solver=cfg.retarget.pink_solver,
        )

        # Reachable target: MuJoCo FK of a perturbed config.
        q_true = np.clip(home + np.array([0.2, 0.1, -0.15, 0.1, 0.0, 0.0, 0.0]),
                         ik.lower, ik.upper)
        p_target, _ = _arm_tool_world(model, data, arm_joints, tool_body, q_true)

        # Track it from home over a few control ticks (warm-started, as in the loop).
        q = home.copy()
        for _ in range(40):
            q = ik.solve(q, p_target, None)

        p_sol, _ = _arm_tool_world(model, data, arm_joints, tool_body, q)
        pos_err = float(np.linalg.norm(p_sol - p_target))
        drift = float(np.abs(q - home).max())
        in_lim = bool(np.all(q >= ik.lower - 1e-6) and np.all(q <= ik.upper + 1e-6))

        if pos_err < 2e-3 and in_lim:
            _ok(f"arm IK pink [{side}] matches MuJoCo FK "
                f"(pos err {pos_err*1000:.2f} mm, max drift {drift:.2f} rad)")
        else:
            _fail(f"arm IK pink [{side}]",
                  f"pos err {pos_err*1000:.2f} mm, within_limits={in_lim}")
            ok = False
    return ok


def check_dual_command(cfg, model, data) -> bool:
    """Two SimRobots sharing one model/data must drive disjoint actuators.

    Validates the dual-arm wiring: the left and right controllers each own a
    distinct set of ctrl indices, so commanding one never clobbers the other.
    """
    robots = {}
    for side in ("left", "right"):
        fj = [jn for spec in cfg.finger_specs_for(side) for (jn, _w) in spec.joints]
        robots[side] = SimRobot(model, data, cfg.arm_joints_for(side), fj,
                                cfg.tool_body_for(side))

    l_act = set(robots["left"]._arm_act) | set(robots["left"]._finger_act.values())
    r_act = set(robots["right"]._arm_act) | set(robots["right"]._finger_act.values())
    if l_act & r_act:
        _fail("dual command", "left/right actuator index sets overlap")
        return False

    # Commanding left must not perturb right's ctrl values (disjoint indices).
    data.ctrl[:] = 0.0
    robots["right"].command_arm(np.array(cfg.arm_home_for("right")))
    right_before = data.ctrl[robots["right"]._arm_act].copy()
    robots["left"].command_arm(np.array(cfg.arm_home_for("left")))
    right_after = data.ctrl[robots["right"]._arm_act]
    if not np.allclose(right_before, right_after):
        _fail("dual command", "left command altered right ctrl")
        return False

    mujoco.mj_step(model, data)
    if np.all(np.isfinite(data.ctrl)) and np.all(np.isfinite(data.qpos)):
        _ok(f"dual-arm command (disjoint actuators: "
            f"{len(l_act)} left + {len(r_act)} right, finite step)")
        return True
    _fail("dual command", "non-finite ctrl/qpos after step")
    return False


def _synthetic_hand(curl: float) -> np.ndarray:
    """21 MANO-order keypoints for a hand at a given uniform curl in [0,1].

    Fingers extend along +Z from the wrist; curling shortens the tip-to-mcp
    span by folding segments. Crude but monotonic, which is all we test.
    """
    kp = np.zeros((21, 3), dtype=np.float32)
    chains = {
        "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20],
    }
    x_off = {"thumb": -0.04, "index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
    seg = 0.03
    for fname, idxs in chains.items():
        base = np.array([x_off[fname], 0.0, 0.03], dtype=np.float32)
        kp[idxs[0]] = base
        # Each joint bends cumulatively: segment k is rotated by k*ang about X,
        # so a larger curl folds the finger and shortens the tip-to-mcp span.
        for k in range(1, 4):
            ang = curl * (np.pi / 2.0) * k
            d = np.array([0.0, -np.sin(ang) * seg, np.cos(ang) * seg], dtype=np.float32)
            kp[idxs[k]] = kp[idxs[k - 1]] + d
    return kp


def check_fingers(cfg) -> bool:
    hand = HandRetargeter(cfg.finger_specs, cfg.retarget)
    # disable smoothing for a clean monotonicity check
    cfg.retarget.command_smoothing = 0.0
    open_curls = hand.finger_curls(_synthetic_hand(0.0))
    hand._smoothed.clear()
    fist_curls = hand.finger_curls(_synthetic_hand(1.0))

    avg_open = np.mean(list(open_curls.values()))
    avg_fist = np.mean(list(fist_curls.values()))
    if avg_fist > avg_open + 0.2:
        _ok(f"finger curl monotonic (open~{avg_open:.2f} < fist~{avg_fist:.2f})")
        return True
    _fail("finger curl", f"open {avg_open:.2f} vs fist {avg_fist:.2f} not separated")
    return False


def check_end_to_end(cfg, model, data, robot) -> bool:
    """One full retarget tick from a synthetic frame; ensure ctrl is finite."""
    ik = ArmIK(model, cfg.arm_joints, cfg.tool_body,
               damping=cfg.retarget.dls_damping, max_iters=cfg.retarget.ik_iters,
               pos_tol=cfg.retarget.ik_pos_tol, max_joint_step=cfg.retarget.max_joint_step)
    hand = HandRetargeter(cfg.finger_specs, cfg.retarget)
    ranges = robot.joint_ranges()

    wrist0 = np.eye(4); wrist0[:3, 3] = [0.0, 0.0, 0.0]
    R_tool, p_tool = robot.get_tool_pose()
    tool_pose = np.eye(4); tool_pose[:3, :3] = R_tool; tool_pose[:3, 3] = p_tool
    calib = WristCalibration.capture(wrist0, tool_pose)

    wrist = np.eye(4); wrist[:3, 3] = [0.05, 0.0, 0.05]
    tR, tp = wrist_to_tool_target(wrist, calib, cfg.retarget.position_scale, True)
    q = ik.solve(robot.get_arm_qpos(), tp, tR, robot.base_qpos())
    robot.command_arm(q)
    robot.command_fingers(hand.joint_targets(_synthetic_hand(0.6), ranges))
    mujoco.mj_step(model, data)

    if np.all(np.isfinite(data.ctrl)) and np.all(np.isfinite(data.qpos)):
        _ok("end-to-end tick (finite ctrl + stable step)")
        return True
    _fail("end-to-end tick", "non-finite ctrl/qpos after step")
    return False


def main() -> int:
    cfg = default_config()
    print("Running teleop self-checks...\n")
    results = []
    results.append(check_transport())

    loaded = check_model_and_robot(cfg)
    if loaded is None:
        print("\nModel/robot check failed; skipping IK/finger checks.")
        return 1
    model, data, robot = loaded

    results.append(check_ik(cfg, model, data, robot))
    results.append(check_ik_pink_both(cfg, model, data))
    results.append(check_dual_command(cfg, model, data))
    results.append(check_fingers(cfg))
    results.append(check_end_to_end(cfg, model, data, robot))

    # +1 for the model/robot check, which passed if we reached here.
    n_pass = sum(results) + 1
    n_total = len(results) + 1
    print(f"\n{n_pass}/{n_total} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
