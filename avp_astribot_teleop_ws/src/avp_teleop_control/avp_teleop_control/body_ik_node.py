# 控制关键节点：
# - 使用 measured RobotStateFull
# - 本进程持有 ManualOverrideController / GearScheduler / PinkWholeBodySolver
# - 统一定时器求解，不在订阅回调中直接算 IK
# - 输出 /teleop/commands/body_ik
