from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Side = Literal["left", "right"]
Gear = Literal["P", "N", "D"]

@dataclass(frozen=True)
class RobotState:
    stamp_ns: int
    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray

@dataclass(frozen=True)
class BodyTargets:
    stamp_ns: int
    head_pose: Optional[np.ndarray]
    left_wrist_pose: Optional[np.ndarray]
    right_wrist_pose: Optional[np.ndarray]
