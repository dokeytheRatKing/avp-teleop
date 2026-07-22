from typing import Protocol, Optional

class HandRetargeterPlugin(Protocol):
    def configure(self, robot_model: dict, config: dict) -> None: ...
    def reset(self) -> None: ...
    def retarget(self, observation, actuator_state, dt: float): ...

class NoOpHandRetargeter:
    """无灵巧手或无 keypoints 时安全返回 None，不阻塞身体 IK。"""
    def configure(self, robot_model: dict, config: dict) -> None:
        pass
    def reset(self) -> None:
        pass
    def retarget(self, observation, actuator_state, dt: float):
        return None
