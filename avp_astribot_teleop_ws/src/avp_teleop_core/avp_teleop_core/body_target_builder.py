class BodyTargetBuilder:
    """AVP 头/手腕 tracking → robot teleop anchor 下的身体目标。"""
    def reset_anchor(self, avp_sample, robot_state) -> None:
        raise NotImplementedError

    def build(self, avp_sample, calibration_state):
        raise NotImplementedError
