# AVP 上半身整体遥操系统 — 当前架构报告

> 面向重构与真机部署的技术现状描述。读者:协助做重构建议的 GPT / 工程师。
> 代码库路径 `avp_teleop_upper_body/`(主)+ `avp_teleop/`(被复用)+ `teleop_official/`(官方真机栈,只读参考)。
> 本报告嵌入了关键代码片段,因为读者看不到代码库本身。

---

## 0. 一句话概览

```
Apple Vision Pro ──UDP──> 重定向(retarget) ──> 合并整体 IK(1 个 Pink/Pinocchio QP) ──> MuJoCo 仿真
   头 + 左手 + 右手         wrist→tool / 头帧        whole_body_ik.py(20~23 DOF)      sim_teleop.py
                                                            ▲
                                              键盘手动换档(P/N/D)+ 自动模式切换
```

一个**合并求解器**同时解 **躯干(4)+ 脖子(2)+ 左臂(7)+ 右臂(7)= 20 DOF**(启用移动底盘时再 + 底盘 3 DOF = 23),
把 AVP 头 + 双手三个 6DoF 目标作为末端任务,在同一个 QP 里协同求解,自动补偿躯干运动对双臂基座的带动。

---

## 1. 数据流全景

```
AVP头显 → avp_publisher(UDP: 头AVPE + 双手AVPH) → UpperBodySubscriber.poll()
  → wrist_to_tool_target 增量重定向(每端) → PoseFilter(平移EMA + 旋转SLERP + 离群钳制)
  → [键盘手动换档 + 自动模式切换 + D档姿态增强]
  → WholeBodyIK.solve()(单QP解23DOF) → SimRobot.command_arm / command_fingers → data.ctrl
  → mj_step → MuJoCo viewer(+ 档位HUD + 目标/误差overlay)
```

- 所有末端目标都在 **MuJoCo 世界系**;IK 模型从**展平后的 teleop MJCF** 构建,解出的关节角写入模拟器**无坐标变换**(帧一致 < 1e-4 m)。
- 主循环 60 Hz(`sim_teleop.py:main`)。输入源可以是实时 UDP(`UpperBodySubscriber`)或录制回放(`FileAvpSource`),两者共享 `poll()` 契约,循环逻辑一致。

## 2. 模块清单

| 文件 | 职责 |
|---|---|
| `config.py`(904行) | 全部可调参数;23-DOF `BODY_JOINTS` 顺序、`IK_KEEP_JOINTS`、home、帧名;四个 dataclass:`WholeBodyIKConfig`/`PoseFilterConfig`/`EgoPoserConfig`/`UpperBodyConfig`。复用 `avp_teleop.config`。 |
| `whole_body_ik.py`(1341行) | **核心**。`WholeBodyIK` 类 + 6 个自定义 Pink Task。单个差分 IK QP 驱动整个身体。 |
| `sim_teleop.py`(1792行) | 主程序/主循环:CLI、构建 IK、标定、poll→目标→滤波→solve→写ctrl→step、档位切换、可视化、录制。 |
| `transport.py`(170行) | `HeadFrame`(magic `AVPE`)+ `HeadFramePublisher` + `UpperBodySubscriber`(单端口按 magic 分发头/双手)。复用 `avp_teleop` 的 `HandFrame`(`AVPH`)。 |
| `avp_publisher.py`(109行) | 佩戴 AVP 时每 tick 发左手/右手/头三个 UDP 报文。 |
| `pose_filter.py`(129行) | `PoseFilter`:平移 EMA + 旋转 SLERP(pinocchio `log3/exp3`)+ 平移离群钳制。 |
| `manual_override.py`(228行) | **键盘手动换档**:`ManualOverrideController`(键 2-9 定时强制档位)+ `GearLock`(dwell 去抖)。见 §6。 |
| `trajectory_io.py`(364行) | 录/放两种轨迹(原始输入 + 全量重定向)。见 §8。 |
| `replay_retarget.py`(217行) | 纯回放 retarget 轨迹(零计算,只 set qpos + 重画 overlay)。 |
| `pose_io.py` / `pose_editor.py` | 按关节名存读整身姿态 JSON / 交互式摆姿工具(供 `--init-pose`)。 |
| `selfcheck.py`(1642行) | 离线自检套件(不需头显)。 |
| `egoposer/`(子包) | **可选**神经躯干先验(重构中删除,见 §4.4)。 |

**复用自 `avp_teleop/`**:`robot_interface.py`(`SimRobot`/`ROSRobot`)、`retarget/frames.py`(`WristCalibration`/`wrist_to_tool_target`)、`retarget/hand_retarget.py`(`HandRetargeter`)、`config.py`(`ARM_JOINTS`/`TOOL_BODY`/`ARM_HOME`/`AVP_TO_ROBOT_R`/`MJCF_PATH`)、`transport.py`(`HandFrame`)。

## 3. IK 求解器(whole_body_ik.py — 核心)

### 3.1 23 个自由度,固定顺序

`BODY_JOINTS` = chassis(3) + torso(4) + neck(2) + 左臂(7) + 右臂(7):

1. `astribot_chassis_x`, `_y`(棱柱平移,米)+ `_zrot`(base 偏航,rad) —— **3-DOF 轮式移动底盘**(不是腿)
2. `astribot_torso_joint_1/2/3` = **矢状倾斜脊柱**(踝/膝/髋 三个平行 pitch 轴,躯干前倾 = 三角之和);`torso_joint_4` = 纯**腰偏航**(无 CoM 位移、无高度变化,安全)
3. `astribot_head_joint_1/2` = 颈部 2-DOF(真机上叫 `astribot_head`,pitch/yaw)
4. 左臂 7 + 右臂 7

模型侧:Pinocchio MJCF 解析器把三个底盘关节合并成一个 `JointModelComposite`;用 `pin.buildReducedModel` 锁死所有非 body 关节(手指等)。23-DOF 命令向量与模型平坦切空间 1:1。

### 3.2 Pink 任务表(默认 cost)

前 4 个始终构建,其余 cost>0 才构建:

| 任务 | 默认 cost | 作用 |
|---|---|---|
| `FrameTask(head_camera)` | pos=3.0 / ori=1.0 | 头相机跟 AVP 头 6DoF(cost 低于手,不猛拽躯干) |
| `FrameTask(left tool)` | pos=10.0 / ori=1.0 | 左手目标 |
| `FrameTask(right tool)` | pos=10.0 / ori=1.0 | 右手目标 |
| `PostureTask(home)` | 0.1 | 消解 23-DOF 零空间,base 拉回原点、肘自然 |
| `ChestOverAnkleTask` | **50.0** | **主平衡/防倾**:胸地面投影保持在踝上方;base-invariant(不致底盘漂移) |
| `TrunkUprightTask` | 0.5 | 次级软正则:躯干 pitch 拉向 0(低 cost,不阻碍下蹲) |
| `NeuralPostureTask` | **0.0(关)** | EgoPoser 先验 —— **重构删除**(§4.4) |
| `ComTask` | 0.0(关) | 遗留质量法平衡,已被 ChestOverAnkle 取代 |
| `BaseTrackingTask` | **0.0(关)** | "底盘跟头" —— **重构删除**(§4.3) |
| `WaistYawTask`(follow) | 0.0(关) | 腰偏航跟头 interaural 轴 |
| `WaistYawTask`(zero) | 0.0 | D 档专用:torso_joint_4 拉向 0° |
| `HandFrontTask` | 0.0(关) | 手过骨盆后方平面才推向前 |
| `DampingTask` | **逐 DOF 向量** | 运动优先级(见 3.3) |
| `LowAccelerationTask` | 0.1 | 惩罚 \|v−v_prev\|,限抖动(有状态) |

### 3.3 运动优先级 = 逐 DOF DampingTask cost 向量

`damping_costs()` 生成长度 23 的向量,cost 越高越"晚动":
- 臂 `0.1`(最便宜,先动) < 腰偏航 `0.5` < 颈/倾斜脊柱 `1.0` < **底盘 linear `12` / yaw `8`**(最贵,最后动)。
- 底盘 Jacobian ≈ 单位阵(移动手最"高效"),故需大 cost 才 planted。可用 `set_chassis_xy_damping`/`set_chassis_yaw_damping` 运行时调(连续调度用)。

### 3.4 硬约束(QP 内)

通过 `pink.limits` 加为硬不等式(取代旧的"先解后 np.clip"):
- `VelocityLimit`:`chassis_lin=1.2 m/s`, `chassis_yaw=1 rad/s`, `torso_neck=1.8`, `arm=3.0`。
- `AccelerationLimit`:`chassis_lin=4`, `chassis_yaw=20`, `torso_neck=60`, `arm=100`。
- `ConfigurationLimit`:`config_limit_gain=0.5`。
- 优雅降级:`NoSolutionFound` 时先丢加速度限,最坏 hold(0 速度)。Phase-0 保护:非有限速度→零速,非有限配置→hold 上一帧。

### 3.5 冻结机制(dead-zone 底座)

把 `model.velocityLimit` 对应 DOF 置 0(Pink 每 solve 实时读),QP 协调地绕开被 pin 的 DOF:
- `set_base_xy_frozen` / `set_base_yaw_frozen`(xy/yaw 独立,便于原地转身)/ `set_base_frozen`(两者)/ `set_lean_frozen`(倾斜脊柱)。O(1)、幂等、下一 solve 生效。

## 4. 平衡 + 两个待删模块

### 4.1 平衡设计(保留)

已从质量法 `ComTask` 迁移到 **base-invariant** 的 `ChestOverAnkleTask`(cost 50,主防倾)+ `TrunkUprightTask`(0.5,次级软正则)。前者保证胸地面投影在踝上方且不引起底盘蠕变。

### 4.2 帧一致性(保留)

`_flatten_mjcf` 先用 MuJoCo `mj_saveLastXML` 展平所有 `<include>`(Pinocchio 不跟随 include),再 `pin.buildModelFromMJCF`,保证每个任务帧与 MuJoCo 重合 < 1e-4 m。末端帧:头 `astribot_head_camera_base_link`、左右工具 `astribot_arm_{left,right}_tool_link`。坐标对齐 `AVP_TO_ROBOT_R = Rz(180°)`,`wrist_to_tool_target` 用相对增量(标定记 wrist0/tool0)。

### 4.3 【重构删除】BaseTrackingTask / base_follow

- **位置**:`whole_body_ik.py` `BaseTrackingTask` 类(约 163-217 行),由 `base_track_cost>0` 触发;`sim_teleop.py` 每 tick 从头水平位移+头偏航算 `(x,y,yaw)` 参考调 `ik.set_base_target()`(约 1588-1612 行)。
- **为何删**:`base_track_cost` **默认 0(关)**。config.py(460-491 行)记录:2026-07-14 clip 评测证明"用头位移驱动底盘让手部跟踪更差"(clip9 96→254mm,clip10 144→510mm,clip11 305→607mm)——反应式底盘已服务手部 FrameTask,头驱动的 base 目标与手需求冲突。用户判定此策略错误,**重构整块移除**。

### 4.4 【重构删除】NeuralPostureTask / EgoPoser

- **位置**:`whole_body_ik.py` `NeuralPostureTask` 类(约 266 行),由 `neural_posture_cost>0` 触发;`sim_teleop` 每 tick 跑 `EgoPoserEstimator.predict()` → EMA → `ik.set_neural_target(pitch,yaw)`;整个 `egoposer/` 子包(estimator/network/feature_builder/rotations/preview)。
- **为何删**:`neural_posture_cost` **默认 0(关)**、`egoposer.enabled=False`、torch 惰性导入缺失即静默降级。用户判定神经先验**效益不高**,**重构整块移除**(含 `egoposer/` 子包)。config 注释明确:关闭后 pipeline 与无此特性版本字节一致,故可安全移除不影响主链路。

## 5. 主循环(sim_teleop.py `main`,60 Hz 每 tick)

1. **poll**:`sub.poll()` → `(hands, head)`,抽出各端 valid 4x4。
2. **录制**(可选):`avp_rec.record(hands, head)` 存原始输入。
3. **标定**(按 `c` 或首次):`ik.reset()`;各端 `WristCalibration.capture`;`filters[end].reset()`;复位 dead-zone/调度器,锚定 head0/base0/interaural。
4. **目标构造**:每端 `wrist_to_tool_target(...)` 增量重定向(头用 `head_position_scale`,手用 `position_scale`)。
5. **滤波**:平移 EMA + 旋转 SLERP + 离群钳制。
6. ~~EgoPoser 先验~~(删除)。
7. **手动换档层**(§6)→ **自动 dead-zone/调度**(§7)→ **D 档姿态增强**(§7.2)。
8. **solve**:`q_body = ik.solve(q_body, targets["head"], targets["left"], targets["right"])`。
9. **写 ctrl**:`command_arm(q_body)` + 每手 `HandRetargeter.joint_targets` → `command_fingers`。
10. **录制 retarget 全量帧**(可选)。
11. **step**:`mj_step` × `n_steps_per_frame`。
12. **overlay**:目标标记/误差线/输入帧;`viewer.set_texts(...)` 画档位 HUD(每 3 帧);`viewer.sync()`。

暂停(space)只跳过 solve,sim 继续 step。

## 6. 键盘手动换档(manual_override.py)—— 重构需保留的交互节点

**键位映射**(ASCII,避开 0/1 的 MuJoCo 冲突):

| 键(keycode) | 轴 / 档 |
|---|---|
| 2/3/4 (50/51/52) | base_trans P / N / D |
| 5/6/7 (53/54/55) | base_yaw P / N / D |
| 8/9 (56/57) | body_pitch P / N |

**机制**:
- `ManualOverrideController(override_duration=5.0)`(CLI `--manual-override-duration`)。`handle_keypress(keycode)` 命中即为该轴创建 `Override(axis, gear, expires_at=now+duration)`;重复按**延长**(可重触发 one-shot)。
- `get_override(axis)` 惰性过期:超时自动清空返回 None(回自动)。
- **与自动共存**:主循环里手动层在自动逻辑**之前**。每轴先查 `get_override(axis)`:非 None 则强制冻结/解冻 + `gear_lock.register_switch` + 按档设 damping(EMA 平滑防冲击);后续所有自动块用 `if <axis>_override is None` 守卫**跳过**——手动期间自动完全让路。
- **定时释放 + 平滑交接**:到期后自动接管;因手动期间也维护 damping EMA 状态,交接无突变。
- **集成点**:`sim_teleop` 的 `key_callback`(约 1127-1145 行)先调 `manual_override.handle_keypress`,命中则 return;否则处理 `c`(重标定)、`space`(暂停)。

> **当前形态**:键盘输入耦合在 MuJoCo viewer 的 `key_callback` 里,`ManualOverrideController` 是纯 Python 状态对象(线程安全的单线程用法),被主循环轮询。**这是重构时需要重新安置的节点**——真机上 MuJoCo viewer 不在,键盘信号需要另找输入源(独立节点/线程发布档位话题,或 stdin/evdev 读取)。

## 7. 自动模式切换 / 档位系统(P/N/D 三轴)

三个正交轴:`base_trans`(底盘 xy)、`base_yaw`(底盘转)、`body_pitch`(倾斜脊柱)。P=frozen(velLimit=0);N=解冻在静态 damping;D=调度激活,damping 降到静态以下。`body_pitch` 只有 P/N。

**自动切换信号**(全用原始 AVP 头,不受重定向影响):

| 轴 | 门控信号 | 冻结/解冻阈值(迟滞) |
|---|---|---|
| base_trans | 头水平速度 EMA | 冻结 0.05 / 解冻 0.12 m/s |
| base_yaw | combined yaw rate(头 interaural + 手中点相对头) | 冻结 0.15 / 解冻 0.40 rad/s(仅 `enable_yaw_scheduling`;否则镜像 xy) |
| body_pitch | 头垂直速度 EMA(下蹲检测) | 冻结 0.08 / 解冻 0.20 m/s |

四层叠加:**死区**(低于冻结阈 hard-freeze,消抖)+ **迟滞**(解冻阈>冻结阈)+ **gear_lock dwell**(`GearLock(dwell_time=2.5s)`,每次切档锁 2.5s,只对离散 P↔N 生效)+ **连续调度**(D 档按信号 `np.interp` 把 damping 从静态线性降到 floor,再 EMA 防换挡冲击)。默认 `enable_yaw_scheduling=False`、`enable_trans_scheduling=False`。

### 7.1 `--upper-body-only` 安全模式

真机安全:锁底盘 + 下肢,只留腰偏航 + 上肢。实现(约 949-960 行):`set_base_xy_frozen(True)` + `set_base_yaw_frozen(True)` + `set_lean_frozen(True)` + `base_deadzone=False`;三个自动切换块加 `and not args.upper_body_only` 守卫。仍活跃:腰偏航、双臂、颈。

### 7.2 D 档姿态增强

进 D 档(底盘移动)时把姿态正则 cost ×2.5(`posture_boost_d_gear=2.5`、`neural_boost_d_gear=2.5`),外加 `waist_zero_d_gear_cost` 专用任务把腰拉向 0°——用移动中的姿态健康换手部精度。CLI `--posture-boost-d-gear` / `--neural-boost-d-gear` / `--waist-zero-d-gear-cost`。

> 注:`neural_boost` 依赖 NeuralPostureTask,随 §4.4 一起删;重构后姿态增强应只作用于 PostureTask + waist-zero。

## 8. 录制 / 回放(trajectory_io.py)

两种独立 JSON 工作流:
1. **AVP 输入轨迹**(`avp_trajectory/`,schema `avp_input_trajectory`):存**原始** AVP 流(头 4x4、每手 wrist + 21×3 keypoints + pinch),不含机器人状态。`FileAvpSource` 是 `UpperBodySubscriber` 的 drop-in,`--replay-avp` 无需头显重跑完整重定向+IK+渲染。
2. **重定向轨迹**(`retargetting_trajectory/`,schema `retarget_trajectory`):超集,含 CLI/config 快照 + 每 tick 原始输入 + 世界末端目标 + 23 关节角 + 手指命令。`replay_retarget.py` 零计算回放(只 set qpos),可前后 scrub。

> 这两个数据目录合计 ~170MB,已在 `.gitignore` 排除,不入库。

## 9. 真机接口抽象(avp_teleop/robot_interface.py)

`RobotInterface(ABC)` 抽象方法:`joint_ranges()`, `get_arm_qpos()`, `get_tool_pose()`, `base_qpos()`, `command_arm(q)`, `command_fingers(targets)`。

- `SimRobot`(MuJoCo):`command_arm(q)` = 把 23-DOF 逐个写 `data.ctrl`(含底盘 x/y/zrot,均位置伺服);`command_fingers` 按名写 ctrl + clip。
- `ROSRobot`:**桩**,`__init__` 直接 `raise NotImplementedError`,注释指向 `/<robot>/joint_space_command`。**填好这 6 个方法即可上真机,teleop 循环无需改动**——这是真机衔接的关键抽象边界(对接方案见附录 A.4)。

## 10. 手/夹爪重定向现状(重构:与身体解耦、留干净接口)

- **现状**:身体 IK(23-DOF,手腕位姿驱动 FrameTask)与手指重定向是**两条路径但同循环耦合**。手指走 `HandRetargeter(finger_specs(side), cfg.retarget).joint_targets(...) → body_robot.command_fingers(ft)`(sim_teleop 约 1668-1685 行),`HandRetargeter` 来自 `avp_teleop.retarget.hand_retarget`(曲率映射,占位实现)。
- **手数据缺失处理**:主循环用 `live[end] is None`(手腕缺帧)守卫——缺手腕时该端手臂目标不更新;但手指命令与身体 solve 仍在一起。
- **重构诉求(见 GPT prompt 硬约束 4)**:手/夹爪重定向应成为**独立求解模块**,输入手部关节/keypoints、输出手指关节命令,接口边界干净(供他人填不同灵巧手逻辑);**手数据缺失时身体 IK 照常驱动躯干+手臂**。当前 `command_fingers(targets: Dict[str,float])` 已是一个可用的输出边界,但输入侧(重定向算法)需要从主循环剥离成可替换模块。
- **灵巧手接口未定**:真机 BrainCo/灵巧手话题定义未在官方仓库找到(附录 A.6),目前只有 1-DOF 夹爪。

## 11. 姿态加载/编辑现状(重构:加运行时归位 + 回放对齐)

- **启动姿态**:`--init-pose <name>` → `pose_io.load_pose` 读 `poses/<name>.json`(schema `avp_upper_body_pose_v1`,按关节名存)→ 作为 `body_home`(初始构型 + PostureTask 静止目标)。仅启动时用一次(sim_teleop 880-885)。默认 `BODY_HOME`。
- **姿态编辑**:`pose_editor.py` 是**独立工具**,MuJoCo 里键盘摆 20 关节、`pose_io` 存 JSON。与主程序不共享运行时。
- **回放现状**:`replay_retarget.py` 零计算回放直接 set qpos——**没有"先归位到首帧"的过渡**,从当前姿态直接跳到录制首帧。
- **重构诉求(见 GPT prompt 硬约束 5)**:需要(a)回放前平滑归位到录制起始姿态;(b)运行时按键运动到指定保存姿态(`my_pose_1.json`);(c)重新考虑姿态编辑/保存功能的归属。核心缺一个**关节空间轨迹插值运动原语**(限速平滑到目标 q,而非瞬间 set)。


---

## 附录 A. 官方真机接口(teleop_official 调研)

**结论:自研 IK 可直接走关节空间命令路径,不经机器人 WBC。** `teleop_official/` 是 Astribot S1 的**外围+接口+消息定义**,控制大脑(WBC/IK/驱动)闭源、跑在机器人本体(x86 主控,`192.168.0.10`),仓库里没有。但官方对每个"部件"暴露了 ROS2(Iron)**关节空间命令话题**,接收关节角,这正是我们要用的接口。

### A.1 关节空间命令接口(主路径)

每个部件一个话题,消息类型统一 `astribot_msgs/msg/RobotJointController`:

```
/astribot_torso/joint_space_command          # N=4
/astribot_head/joint_space_command           # N=2  (我们的"脖子":joint_1=pitch, joint_2=yaw)
/astribot_arm_left/joint_space_command        # N=7
/astribot_arm_right/joint_space_command       # N=7
/astribot_gripper_left/joint_space_command    # N=1  (开合量)
/astribot_gripper_right/joint_space_command   # N=1
/astribot_chassis/joint_space_command         # N=3  (x, y, zrot)
```

消息定义(`teleop_official/software/astribot_msgs/share/astribot_msgs/msg/RobotJointController.msg`):

```
std_msgs/Header header
int8      mode        # 1=位置(常用), 2=速度, 3=力矩
string[]  name        # 该部件关节名列表
float64[] command     # 长度决定语义: N=纯位置; 2N=位置+速度; 3N=位置+速度+力矩
```

- **对我们的 Pink IK**:`mode=1` + `command` 长度 N(纯关节角,弧度)即可;若能同时给关节速度,用 2N 更平滑。
- **QoS = BEST_EFFORT + VOLATILE**。发布方 QoS 不匹配会**静默失败**(连不上也不报错),这是最容易踩的坑。
- **关节顺序/命名**(`astribot_simulation/config/astribot_s1/simulation_mujoco_param.yaml`):`command[i]` 对齐 `name[i]`。我们的 20-DOF(torso4 + neck2 + 双臂各7)干净映射到 torso + head + arm_left + arm_right;底盘 3-DOF 走 chassis。注意"脖子"在真机叫 `astribot_head`。

### A.2 状态反馈接口

```
/<部件>/joint_space_states        # astribot_msgs/RobotJointState {position[], velocity[], ...}
/<部件>/endpoint_current_states   # 当前末端笛卡尔位姿
```

**首帧安全**:下发前先订阅 `joint_space_states` 读当前角,首帧命令 = 当前实测角,避免猛跳。

### A.3 分肢体控制天然支持 `--upper-body-only`

接口是**按部件分开的独立话题**(不是扁平数组)。只给 torso/head/arm_left/arm_right 发、**不给 chassis 发**,腿/底盘就不动——这正好契合我们的 `--upper-body-only` 安全模式。

### A.4 现成的衔接点

`avp_teleop/robot_interface.py` 已有 `ROSRobot` 占位类(目前 `NotImplementedError`),主循环已抽象成 `RobotInterface`(`SimRobot` / `ROSRobot`)。**填好 ROSRobot 的 `command_arm` / `command_fingers` / `get_arm_qpos` 后,主循环无需改动**。可直接照抄的参考实现:`astribot_simulation/src/simu_utils/robot_ros_interface.py`(`RobotRosInterface`,行 212-392,仿真端用的就是同名话题+同一消息)。

### A.5 控制权限与启动(动电机前必须)

- **网络**:同一 DDS/ROS2 域,PTP 时间同步;ORIN `.11` / x86 `.10`。
- **控制权**:真机待机 `/astribot_mode`=0,需先申请:`/astribot/control_rights`、`/simu_real_switch_service`(srv `astribot_msgs/srv/RawRequest`,一进一出都是 JSON 字符串)。**具体 JSON 协议不在仓库里**,需抓包官方遥操流程或问官方。
- **WBC 备选路径(未选用)**:`/astribot_wbc_solver/wbc_cmd`(`WholeBodyCtrlCmd`,吃末端位姿、QoS=RELIABLE,无头部字段,求解器在机器人本体)。本项目**不走这条**。

### A.6 未在仓库中找到(需向官方/真机确认)

控制权/模式切换的 JSON 协议内容、WBC 求解器本体(x86 闭源)、**BrainCo/灵巧手话题定义**(仓库只有 1-DOF 夹爪,灵巧手驱动推测在本体/单独 SDK)、真机端各话题确切 QoS 深度。

---

## 附录 B. 重构相关的三个决策(与本报告配套的 GPT prompt 对应)

1. **自研 IK,不走机器人 WBC** —— 继续用 Pink 求解器输出关节角,经关节空间命令接口(附录 A)下发真机。
2. **保留键盘手动换档节点** —— 见 §6,重构需妥善安置这个交互节点。
3. **舍弃 EgoPoser 与 base_follow** —— 见 §4.3 / §4.4,两者已验证效果不佳,重构中整块删除。
4. **手/夹爪重定向独立解 IK、留干净接口** —— 见 §10,与身体解耦;手数据缺失时身体照常动;供他人填灵巧手逻辑。
5. **回放先归位 + 运行时按键运动到指定姿态** —— 见 §11,需关节空间轨迹插值运动原语;重新安置姿态编辑/保存功能。
