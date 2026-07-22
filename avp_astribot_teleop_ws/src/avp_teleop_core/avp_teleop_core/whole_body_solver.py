from typing import Protocol

class BodySolver(Protocol):
    def reset(self, measured_q) -> None: ...
    def solve(self, state, targets, gear_state, dt): ...

class PinkWholeBodySolver:
    """仅保留 FrameTask + Posture + 平衡/直立 + 平滑 + limits/barriers。"""
    # 禁止重新引入 NeuralPostureTask / EgoPoser / BaseTrackingTask。
    pass
