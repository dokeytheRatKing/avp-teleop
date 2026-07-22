# AVP → Astribot S1 Teleoperation Workspace

面向以下主链路的 ROS2 工作区骨架：

Apple Vision Pro tracking → 身体目标重定向 → Pink/Pinocchio whole-body IK
→ 关节命令仲裁与安全守卫 → Astribot S1 `joint_space_command`

硬约束：
- 使用自研 Pink/Pinocchio IK，不走机器人自带 WBC。
- 键盘手动换档为独立输入节点；档位状态机保留在 body IK 进程内。
- 删除 EgoPoser / NeuralPostureTask。
- 删除 base_follow / BaseTrackingTask。
- 身体 IK 与手/夹爪重定向完全解耦。
- 仿真和真机只在 backend/bridge 层分叉。
- 真机训练数据优先复用官方 astribot_recorder；仿真输出同布局 HDF5。
