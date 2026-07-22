# 三阶段落地

## Phase 1：ROS2 化 + MuJoCo 闭环追平
完成输入、目标构建、body IK、键盘换档、命令仲裁、MuJoCo backend、HUD、调试轨迹录放。
验收：仿真效果、2-9 换档、5 s 自动释放、upper-body-only、overlay、轨迹录放均追平旧版。

## Phase 2：真机 + 安全 + 关键交互
完成 Astribot bridge、measured-state 闭环、首帧对齐、限速、急停、go-to-pose、replay align、hand plugin 边界。
验收：hold-current、upper-body-only、单臂/双臂、逐轴解锁移动体，无命令尖峰。

## Phase 3：沉浸视角 + 训练数据
完成 robot-view/self-view、官方 recorder 调度、仿真等价 HDF5、LeRobot/RLDS 导出。
验收：视角切换不影响控制；真机/仿真数据可由同一训练加载器读取。
