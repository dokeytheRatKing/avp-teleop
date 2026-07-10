# AVP → MuJoCo 遥操作链路 (avp_teleop)

用 Apple Vision Pro 记录手部位姿，实时驱动 MuJoCo 中的 Astribot S1 机器人做同样的动作。

```
Apple Vision Pro ──UDP──> 运动学映射(Retargeting) ──> MuJoCo 仿真
  avp_publisher.py          retarget/*.py              sim_teleop.py
```

机器人复现两部分动作：

1. **手臂跟随**：机器人工具末端跟随你手腕在空间中的移动。默认用 **Pinocchio + Pink** 的差分 IK（精度亚毫米、抗奇异、肘腕姿态自然），也保留了原来手写的阻尼最小二乘（DLS）IK 作为零依赖回退。
2. **手指开合**：你每根手指的弯曲程度映射到 BrainCo 机械手对应手指的关节。

**单手或双手皆可**：用 `--side right` / `--side left` 单臂遥操，或 `--side both` 让左右手**同时**驱动机器人左右臂 + 左右灵巧手（见下方"双臂遥操"）。

手指映射与传输只依赖 `mujoco` + `numpy` + `scipy` + `avp_stream`。默认的 Pink 手臂 IK 额外需要 `pinocchio` + `pink` + `qpsolvers`（见下方安装）；如果不想装，用 `--ik-backend mujoco` 即可回退到纯 MuJoCo 实现，**无需任何新依赖**。

---

## 安装 Pink IK 依赖（装进 AVP 环境，只需一次）

```bash
conda activate AVP
conda install -c conda-forge pinocchio pink qpsolvers quadprog -y
# 验证
python -c "import pinocchio, pink, qpsolvers; print('ok')"
```

> 这些只给**默认的手臂 IK** 用。Pink 走 Pinocchio（C++ 内核，无 JAX），没有首帧 JIT 卡顿，单次求解 ~0.8 ms，60 Hz 余量充足。装不上也不影响其余功能——加 `--ik-backend mujoco` 跑旧实现。

---

## 怎么运行

两个进程，开两个终端，都用 `AVP` 环境，工作目录都在 `/Users/apple/vscodeProject/AVP`。

### 第一步（可选但推荐）：先跑离线自检

确认模型、IK、手指映射都正常，不需要戴 AVP：

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
python -m avp_teleop.selfcheck
```

应当看到 `7/7 checks passed.`（其中包含 **左右两条臂** 的 Pink 解与 MuJoCo 正运动学一致性校验，以及一项双臂指令互不干扰校验；若你没装 Pink 依赖，Pink 那一项会失败，但其余项仍应通过）

### 第二步：启动 AVP 发布端（戴上 Vision Pro）

打开 Vision Pro 上的 Tracking Streamer App，记下屏幕显示的 IP 或六位房间码，然后：

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
python -m avp_teleop.avp_publisher --avp-ip 10.200.177.142 --side right
```

- `--avp-ip`：你的 AVP 地址（默认值在 `config.py` 里，也可用环境变量 `AVP_IP` 覆盖）。
- `--side`：跟踪哪只手，`right` / `left` / `both`。`both` 会把左右手各发一帧（同一端口，靠消息里的 side 字节区分），供双臂遥操使用。

看到 `OK: sent=... ` 持续刷新就说明手部数据在正常发送。

### 第三步：启动仿真端

另开一个终端：

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
python -m avp_teleop.sim_teleop --side right
```

会弹出 MuJoCo 窗口。**把手摆到一个舒服的初始姿态，在窗口里按 `c` 进行标定**，机器人就会开始跟随。

仿真窗口快捷键：

| 键 | 作用 |
|----|------|
| `c` | （重新）标定：把当前手的位姿锚定到机器人当前末端位姿。双臂模式下 `c` **同时**重标定左右两条臂。任何时候手"飘"了都可以重新按 `c` 归中。 |
| `空格` | 暂停 / 继续遥操（物理仿真继续跑） |

常用参数：

```bash
# 单手 / 双手：right、left，或 both（左右手同时遥操左右臂 + 双手）
python -m avp_teleop.sim_teleop --side both

# 选择手臂 IK 后端：pink（默认，精准/自然/抗奇异）或 mujoco（旧 DLS，零依赖回退）
python -m avp_teleop.sim_teleop --ik-backend pink
python -m avp_teleop.sim_teleop --ik-backend mujoco

# 只跟随手腕位置、忽略姿态（更稳，适合先调通；这是默认行为）
python -m avp_teleop.sim_teleop --no-orientation
# 位置调通后，再开启 6-DOF 姿态跟随
python -m avp_teleop.sim_teleop --orientation

# 调整动作放大倍率（默认 1.0，越大机器人移动幅度越大；务必保持正值，方向用 AVP_TO_ROBOT_R 调）
python -m avp_teleop.sim_teleop --position-scale 1.4
```

> **为什么默认换成 Pink？** 原来手写的 DLS IK 对 7-DOF 冗余臂的零空间不加约束，肘腕容易扭曲、靠近奇异点抖动、精度也只有毫米级。Pink 把"末端跟随"和"姿态正则（偏向 `ARM_HOME`）"作为加权任务一起解，并用 LM 阻尼平滑处理奇异。同一组可达目标上实测：Pink 中位误差 **0.00 mm** vs 旧 DLS **1.34 mm**，关节偏离更小（更自然）。两个后端 `solve()` 接口完全一致，主循环不变。

---

## 双臂遥操（`--side both`）

两个终端，发布端和仿真端都加 `--side both`：

```bash
# 终端 A：发布左右手
python -m avp_teleop.avp_publisher --avp-ip 10.200.177.142 --side both
# 终端 B：左右臂同时遥操（pink 为默认后端）
python -m avp_teleop.sim_teleop --side both
```

按 `c` 一次即同时标定左右臂，之后两只手各自独立跟随。位置调通后可加 `--orientation`。

---

## 录制轨迹 & 离线回放（真机验证前的保险步骤）

目标：先用 AVP 录一段重定向后的**关节空间动作序列**，存成 Astribot 真机能吃的格式；再在 MuJoCo 里一帧一帧回放这段序列（不接 AVP、不走网络），确认机器人能复现。同一份文件将来即可通过 ROS2 推给真机 S1。

### 1) 录制（在 `sim_teleop` 里顺带录）

```bash
# 终端 A：照常发布 AVP 手部帧
python -m avp_teleop.avp_publisher --avp-ip 10.200.177.142 --side both
# 终端 B：照常遥操，额外加 --record 指定输出文件
python -m avp_teleop.sim_teleop --side both --record my_clip.json
```

- 先按 `c` 标定，动作调顺后按 **`r`** 开始录制，再按 `r` 停止（可反复分段）。想一进去就录加 `--record-autostart`。
- 只在「已标定 + 未暂停 + 已 armed」时才记帧，所以起始摆位/settle 不会被录进去。
- 状态行会显示 `REC 123f`（正在录）或 `rec paused (123f)`。关闭窗口时自动写盘。

### 2) 离线回放（用录好的文件驱动 MuJoCo，不需要 AVP）

```bash
# 默认回放进遥操模型（灵巧手），实时速度
python -m avp_teleop.replay_sim my_clip.json
# 半速看细节 / 循环 / 无头验证
python -m avp_teleop.replay_sim my_clip.json --speed 0.5 --loop
python -m avp_teleop.replay_sim my_clip.json --no-render
```

回放器按**关节名**把每帧命令映射到当前模型里对应的执行器，因此与模型无关：遥操模型下 15 个手指关节生效、派生的夹爪值被跳过；换夹爪模型（`--mjcf .../astribot_s1_with_gripper.xml`）则相反。文件里模型没有的关节会被安全跳过并提示一次。

### 3) 推到真机（ROS2，尚未在硬件上验证）

```bash
# 在装了 ROS2 + astribot_msgs 的机器上（例如 ssh 到机器人）：
python3 -m avp_teleop.replay_ros2 my_clip.json --dry-run     # 先只打印不发布
python3 -m avp_teleop.replay_ros2 my_clip.json               # 发布双臂 + 夹爪
python3 -m avp_teleop.replay_ros2 my_clip.json --include-torso-head
```

按录制时序，把每个部件的命令以 `RobotJointController`（`mode=1` 位置）发到 `/<component>/joint_space_command`。**注意**：`rclpy`/`astribot_msgs` 不在 `AVP` conda 环境里，此脚本要在机器人/ROS2 主机上跑；夹爪值是从手指闭合度派生映射到 `[0,100]` 的，**符号/量程未在真机核对过**，务必先 `--dry-run`、留好急停、并让机器人先靠近首帧姿态。

### 录制文件格式

JSON，`schema="astribot_joint_trajectory"`, `version=1`：`metadata` 里有 `nominal_dt`、`sides`、`control_mode`、各部件的有序关节名；`frames` 是 `{t, commands:{部件->数值}}` 列表。每帧都是一条**完整的 S1 关节空间命令**（双臂 7×2 + 夹爪 ×2 + 躯干 4 + 头 2），外加 `hand_left`/`hand_right` 各 15 个 BrainCo 手指关节（供灵巧手模型忠实回放）。详见 [recording.py](recording.py)。

**设计要点（为什么这样做）**：

- **两个独立的单臂求解器，而不是一个 14-DOF 合并求解器。** 现有 Pink IK 在求解时把躯干/头/底盘/另一只手都**锁在中立位**，所以左右臂在运动学上**相互独立**——分开解就是精确解，没有被忽略的耦合。这也最贴合将来上真机时左右臂各自一个 ROS 命名空间的结构。
- **左右手共用同一套坐标对齐与 home。** `AVP_TO_ROBOT_R`（AVP→机器人的 180° Z 旋转）是**世界系**旋转，与是哪只手无关；`ARM_HOME` 左右对称。所以左臂无需任何额外坐标处理，相对/增量标定天然按各自的"手腕↔末端"锚点工作。
- **传输层零改动 wire 格式。** 发布端每拍对左右各发一个 datagram（消息里本就有 side 字节）；订阅端 `latest_by_side()` 排空 socket 后按 side 各留最新一帧。单臂路径完全兼容。
- **两个 `SimRobot` 共享同一 `model`/`data`。** 左右臂的执行器下标互不相交（实测各 22 个：7 臂 + 15 指），`command_arm`/`command_fingers` 各写各的，互不覆盖。
- 暂未做**双臂间避碰**（本次未要求）；如需可后续改成合并求解器并加碰撞约束（参考 VisionProTeleop 的 `11_diffik_aloha.py`）。

---

## 标定是怎么回事（重要）

这套链路用的是**相对（增量）遥操**，不需要你把 AVP 世界坐标和机器人基座对齐。

按下 `c` 的瞬间，系统记录：
- 你此刻的手腕位姿（作为零点）
- 机器人末端此刻的位姿（作为零点）

之后每一帧，系统只看你**相对于零点**移动了多少，再（按 `position_scale` 缩放后）叠加到机器人末端。所以：
- 你站在哪、手怎么摆都无所谓，按 `c` 那一刻就是新的中心。
- 如果机器人姿态漂得不舒服，把手放回舒服位置，再按一次 `c` 即可。

---

## 模块结构

```
avp_teleop/
  config.py            # 唯一配置入口：IP、左右手、关节名、缩放、增益、home 位姿
  transport.py         # 零依赖 UDP 收发 + 固定消息格式(手腕位姿 + 21关键点)
  avp_publisher.py     # 连 AVP，提取手腕位姿和手指关键点，UDP 发布
  retarget/
    frames.py          # 坐标约定 + 相对位姿标定
    arm_ik.py          # [回退] 阻尼最小二乘 IK：手腕目标位姿 → 7 个臂关节角
    arm_ik_pink.py     # [默认] Pinocchio + Pink 差分 IK（同 solve 接口，精准/自然/抗奇异）
    hand_retarget.py   # 21 关键点 → 每指弯曲度[0,1] → 手指关节目标
  robot_interface.py   # 机器人抽象层：SimRobot(MuJoCo) / ROSRobot(留桩)
  sim_teleop.py        # 主循环：订阅 → 映射 → 写 ctrl → 步进 + 渲染（支持双臂；--record 顺带录轨迹）
  recording.py         # 轨迹录制：关节空间命令 → Astribot ROS2 格式 JSON（含派生夹爪值）
  replay_sim.py        # 离线回放：录好的 JSON → 驱动 MuJoCo 模型逐帧复现（不接 AVP）
  replay_ros2.py       # 真机回放：JSON → 逐帧发 RobotJointController 到 ROS2（未在硬件验证）
  selfcheck.py         # 离线自检（无需硬件）

astribot_simulation/astribot_descriptions/mjcf/astribot_s1_mjcf/
  astribot_s1_teleop.xml                          # 遥操专用模型(复用原模型 + 手指驱动器)
  actuators/astribot_s1_actuator_finger_teleop.xml # 新增的手指位置驱动器
```

> 原有的 `astribot_s1_with_hand.xml`、`hand_pose_streaming.py`、`view_robot.py` 都**没有改动**。

### 为什么要新加一个 MJCF？

原版 `astribot_s1_with_hand.xml` 里 BrainCo 手的手指关节是**被动的**（每只手只有 1 个驱动器，即拇指掌骨 `joint_L1`）。要让手指能开合，必须给每个手指关节加位置驱动器。`astribot_s1_teleop.xml` 就是在原模型基础上 `include` 了一份新增的手指驱动器文件，原文件保持不变。

### Pink IK 用的是哪个机器人模型？（一个关键设计点）

Pink 的 Pinocchio 模型**直接从仿真用的同一个 MJCF 构建**，而不是仓库里的单臂 URDF（`astribot_arm_right.urdf`）。原因是实测发现：单臂 URDF 和 MJCF 的内部关节坐标系**并不一致**（同一组关节角下，工具末端位姿用最佳刚体变换对齐后仍差到厘米级），整臂/全身 URDF 同样对不上。若在 URDF 里求解再把关节角写进 MuJoCo，末端会系统性偏位。

做法：用 MuJoCo 把 `astribot_s1_teleop.xml` 的所有 `include` 展平成单文件，再用 `pin.buildModelFromMJCF` 读入，并 `buildReducedModel` 锁住除 7 个手臂关节外的全部关节（躯干/头/底盘/另一只手/手指都固定在中立位）。这样 Pink 的工具坐标系与 MuJoCo **逐构型吻合到 < 1e-4 m**，求解结果**零坐标变换**直接写入仿真。selfcheck 里的"Pink 解与 MuJoCo FK 一致性"那一项就是守这个不变量。

---

## 调参指南（都在 `config.py`）

| 参数 | 作用 | 调整方向 |
|------|------|---------|
| `RetargetConfig.ik_backend` | 手臂 IK 后端 `"pink"`/`"mujoco"` | 默认 `pink`；想零依赖回退或对比时设 `mujoco`（也可用 `--ik-backend`） |
| `RetargetConfig.position_scale` | 手腕位移放大倍率 | 机器人够不到 → 调大；动作太夸张 → 调小（保持正值） |
| `RetargetConfig.track_orientation` | 是否跟随手腕姿态 | 先 `False` 调通位置，再开 `True`（或用 `--orientation`） |
| `RetargetConfig.max_joint_step` | 每帧关节最大变化(rad) | 动作突变/不稳 → 调小（两后端共用） |
| **Pink 后端** | | |
| `RetargetConfig.pink_position_cost` | 末端跟随权重 | 跟随不够紧 → 调大 |
| `RetargetConfig.pink_posture_cost` | 姿态正则权重（偏向 `ARM_HOME`） | 手臂扭曲/不自然 → 调大（更"端正"但跟随略松） |
| `RetargetConfig.pink_lm_damping` | LM 阻尼（抗奇异） | 接近奇异点抖动 → 调大；跟随发钝 → 调小 |
| `RetargetConfig.pink_solver_iters` | 每拍差分迭代次数 | 跟随略滞后 → 调大；想更省算力 → 调小 |
| `RetargetConfig.pink_orientation_cost` | 姿态跟随权重（仅 `track_orientation=True` 用） | 姿态跟随不到位 → 调大 |
| **MuJoCo DLS 回退后端** | | |
| `RetargetConfig.dls_damping` | DLS 阻尼 | 末端抖动 → 调大；跟随发钝 → 调小 |
| **手指 / 通用** | | |
| `RetargetConfig.finger_open_reach` / `finger_closed_reach` | 手指"张开/握拳"对应的几何阈值 | 手指闭合不到位 → 调 `closed_reach` 大一点；张不开 → 调 `open_reach` 大一点 |
| `RetargetConfig.command_smoothing` | 手指指令平滑(0~1) | 手指抖 → 调大 |
| `ARM_HOME` | 机械臂初始/参考姿态（也是 Pink 的姿态正则目标） | 想换一个更舒服的工作姿态时改这里 |

---

## 我已经离线验证过的部分

`python -m avp_teleop.selfcheck`（在 AVP 环境实跑通过，`7/7 checks passed.`）：

- ✅ UDP 消息编解码往返一致
- ✅ 遥操 MJCF 能加载，7 个手臂 + 15 个手指驱动器全部解析成功
- ✅ 旧 MuJoCo DLS IK 对可达目标收敛到亚毫米（0.5 mm）
- ✅ **Pink IK 解写回 MuJoCo 后，左右两条臂末端都与目标一致（各 0.00 mm）**——证明 Pinocchio(展平 MJCF) 与 MuJoCo 坐标系对左右臂均一致、零变换
- ✅ **双臂指令互不干扰**：左右各一个 SimRobot 共享同一 model/data，执行器下标互不相交（各 22 个），命令一臂不影响另一臂，步进后数值有限
- ✅ 手指弯曲度从张开(0.0)到握拳(1.0)单调变化
- ✅ 端到端一拍：合成手部帧 → 写 ctrl → 步进，数值有限且稳定

另外做了双臂无头烟雾测试：用真实 UDP socket 同时发左右手帧，`latest_by_side()` 正确按 side 各取最新；左右两个控制器共享一个 model/data 各自标定+求解后，**两条臂都按各自手的位移移动**、ctrl 数值有限、仿真稳定。

早先单臂还做过一组 A/B 基准（30 个可达目标）：**Pink 中位末端误差 0.00 mm（p90 0.00 mm），旧 DLS 1.34 mm（p90 1.83 mm）**；两者关节相对 home 的偏离量相当（~0.4 rad），Pink 单次求解 ~0.8 ms（60 Hz 绰绰有余）。

### 我无法替你测的部分（需要你在本机 + Vision Pro 上验证）

- 实时 AVP 数据流（需要真实佩戴设备；双臂时确认左右手都被 Tracking Streamer 跟踪到）
- MuJoCo 图形窗口的实际渲染与交互
- 端到端的延迟与手感（双臂同时跟随的实时性）

如果跟随方向感觉"反了"或者别扭，多半是 AVP 手腕坐标轴和机器人末端坐标轴的约定差异——见下面排查。

---

## 常见问题排查

**仿真端一直显示 `NO DATA`**
发布端没在跑，或 IP/端口对不上。确认两个进程的 `--host/--port` 一致（默认 `127.0.0.1:9870`），且 `avp_publisher` 正在刷 `sent=`。

**发布端连不上 AVP**
确认 Vision Pro 和电脑在同一网络、Tracking Streamer App 已打开、`--avp-ip` 与 App 显示的地址一致。跨网络时用六位房间码代替 IP。

**机器人跟随方向是反的 / 旋转很别扭**
先用 `--no-orientation` 只跟位置，确认位置方向对不对。如果连位置方向都不对，需要在 `retarget/frames.py` 的 `wrist_to_tool_target` 里给位移增量 `dp` 加一个坐标轴对齐矩阵（AVP 是 Z-up 世界系，机器人末端有自己的朝向约定）。这一步要等你实际看到跟随效果后再针对性微调。

**手指动但方向/幅度不对**
调 `config.py` 里的 `finger_open_reach` / `finger_closed_reach`。这两个值是"你自己的手"张开和握拳时的几何比例，因人而异。

**末端抖动**
调大 `dls_damping`，或调小 `max_joint_step`，或调大 `command_smoothing`（手指）。

---

## 将来怎么接真机 / ROS

`robot_interface.py` 里已经把机器人抽象成 `RobotInterface`。现在用的是 `SimRobot`（MuJoCo），将来上真机只需实现 `ROSRobot`（已留桩），把 `command_arm` / `command_fingers` 接到 Astribot 的 ROS 关节话题上（参考 `astribot_simulation` 里的 `robot_ros_interface.py`，发布 `RobotJointController` 到 `/<robot>/joint_space_command`）。主循环和映射逻辑完全不用改。

**已经落地的真机对接第一步**：`recording.py` + `replay_ros2.py` 已经按上面这套 ROS2 关节空间格式实现了「录制→回放」。推荐的真机验证路径是**先录后放**（比实时遥操上真机安全）：用 `sim_teleop --record` 录一段，`replay_sim` 在仿真里确认没问题，再用 `replay_ros2`（在 ROS2 主机上）逐帧推给真机。见上面「录制轨迹 & 离线回放」一节。

同理，`transport.py` 的 UDP 收发这一层也可以整体替换成 ROS 的 pub/sub —— 发布端和订阅端只约定了 `HandFrame` 这一个数据结构。
