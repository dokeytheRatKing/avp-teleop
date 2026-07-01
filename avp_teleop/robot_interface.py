"""Robot abstraction layer.

The teleop loop talks to a robot only through `RobotInterface`. Today the
concrete implementation is `SimRobot` (MuJoCo). For real-hardware / ROS
deployment, implement `ROSRobot` (a stub is provided) with the same methods and
the rest of the pipeline is unchanged.

Contract
--------
    arm_joint_names / finger_joint_names : names this robot exposes.
    joint_ranges()      : {name: (low, high)} limits (drives the finger mapping).
    get_arm_qpos()      : current arm joint angles (IK seed).
    get_tool_pose()     : current (R, p) of the tool body (IK calibration).
    base_qpos()         : full-body posture context for IK FK (sim-specific;
                          a ROS robot may return its own representation or a
                          cached neutral posture).
    command_arm(q)      : send arm joint targets.
    command_fingers(d)  : send finger joint targets {name: angle}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

try:
    import mujoco
except ImportError:  # allow importing the ABC without mujoco (e.g. ROS-only host)
    mujoco = None


class RobotInterface(ABC):
    arm_joint_names: List[str]
    finger_joint_names: List[str]

    @abstractmethod
    def joint_ranges(self) -> Dict[str, Tuple[float, float]]: ...

    @abstractmethod
    def get_arm_qpos(self) -> np.ndarray: ...

    @abstractmethod
    def get_tool_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def base_qpos(self) -> np.ndarray: ...

    @abstractmethod
    def command_arm(self, q: np.ndarray) -> None: ...

    @abstractmethod
    def command_fingers(self, targets: Dict[str, float]) -> None: ...


class SimRobot(RobotInterface):
    """MuJoCo-backed robot. Owns the model+data and the actuator index maps."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        arm_joint_names: List[str],
        finger_joint_names: List[str],
        tool_body_name: str,
    ):
        self.model = model
        self.data = data
        self.arm_joint_names = list(arm_joint_names)
        self.finger_joint_names = list(finger_joint_names)

        self.tool_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, tool_body_name
        )

        # joint qpos addresses
        self._arm_qpos_adr = [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in self.arm_joint_names
        ]

        # actuator indices, keyed by the joint each actuator drives.
        self._act_by_joint: Dict[str, int] = {}
        for aid in range(model.nu):
            trnid = model.actuator_trnid[aid, 0]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
            if jname is not None:
                self._act_by_joint[jname] = aid

        missing_arm = [n for n in self.arm_joint_names if n not in self._act_by_joint]
        if missing_arm:
            raise ValueError(f"No actuator for arm joints: {missing_arm}")
        self._arm_act = [self._act_by_joint[n] for n in self.arm_joint_names]

        # Finger joints that actually have an actuator in this model.
        self._finger_act = {
            n: self._act_by_joint[n]
            for n in self.finger_joint_names
            if n in self._act_by_joint
        }

    def joint_ranges(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for n in self.arm_joint_names + self.finger_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                lo, hi = self.model.jnt_range[jid]
                out[n] = (float(lo), float(hi))
        return out

    def get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self._arm_qpos_adr])

    def get_tool_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        R = self.data.xmat[self.tool_body_id].reshape(3, 3).copy()
        p = self.data.xpos[self.tool_body_id].copy()
        return R, p

    def base_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def command_arm(self, q: np.ndarray) -> None:
        for aid, qi in zip(self._arm_act, q):
            self.data.ctrl[aid] = qi

    def command_fingers(self, targets: Dict[str, float]) -> None:
        for name, angle in targets.items():
            aid = self._finger_act.get(name)
            if aid is not None:
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = float(np.clip(angle, lo, hi))


class ROSRobot(RobotInterface):
    """Stub for a future ROS / real-hardware backend.

    Implement these against your ROS topics (see astribot_simulation's
    robot_ros_interface.py: publish RobotJointController on
    /<robot>/joint_space_command, subscribe joint_space_states). The teleop
    loop needs no changes once these methods are filled in.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ROSRobot is a placeholder. Implement command_arm/command_fingers "
            "against the Astribot ROS joint-space topics to deploy on hardware."
        )

    def joint_ranges(self): raise NotImplementedError
    def get_arm_qpos(self): raise NotImplementedError
    def get_tool_pose(self): raise NotImplementedError
    def base_qpos(self): raise NotImplementedError
    def command_arm(self, q): raise NotImplementedError
    def command_fingers(self, targets): raise NotImplementedError
