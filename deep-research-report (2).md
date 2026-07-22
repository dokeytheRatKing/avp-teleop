# 面向真机的 AVP 上半身遥操作系统重构报告

## 重构结论与总体判断

基于你给出的现有架构报告，我的总体判断是：这次重构不应该从“再做一个更大的主循环”开始，而应该从**把当前单进程控制闭环拆成“输入、求解、仲裁、执行、记录、显示”六类边界清晰的 ROS2 节点**开始。你已经有三个非常正确的前提：继续坚持自研 Pink/Pinocchio IK，通过官方 `/<部件>/joint_space_command` 直接发关节命令；保留键盘手动换档；彻底移除 EgoPoser 与 base_follow。现有系统已经证明，AVP→重定向→单 QP whole-body IK→MuJoCo 这条主链路本身是成立的，且 `RobotInterface` 抽象、`--upper-body-only` 安全模式、`ManualOverrideController`、调试回放、姿态 JSON 工具这些“骨架件”都已经具备，只是现在它们还耦合在同一个 60 Hz MuJoCo 主循环里。重构的目标不是重写算法，而是把这条链路变成**可替换后端、可验证、可录制、可在真机安全推进**的节点化架构。fileciteturn0file0

我建议你把新系统的核心原则定成四条。第一，**IK 节点永远只做“从目标和状态算出关节命令”**，不直接碰键盘输入、视频显示或数据采集。第二，**手动换档是独立输入源，但换档状态机必须在身体 IK 所在进程内生效**，这样才能保证 gear 切换、冻结、阻尼调度和求解是原子性的，不在命令下发端制造跳变。第三，**仿真与真机只在“robot backend/bridge”层分叉**，上游重定向、IK、换档、回放、姿态运动原语全部共用同一套代码。第四，**数据采集与操控闭环解耦**：真机复用官方 recorder，仿真自建一个字段布局对齐的等价 recorder。这个方向和你报告里已有的 `RobotInterface` 抽象、官方 `joint_space_command` 路径、官方 recorder 采集 `joint_space_command` 作为 action 的事实是完全对齐的。fileciteturn0file0

从 ROS2 工程实践看，这种拆法还有两个现实好处。其一，官方控制话题采用 DDS QoS，QoS 不匹配会直接导致“看起来一切正常、实际上根本没连上”的静默失败；把外层控制逻辑和最终命令桥拆开，更容易在 bridge 层做 QoS 显式配置和连接健康检查。其二，ROS2 的 callback groups、executor、lifecycle 节点机制天然支持把 bring-up 做成“先就绪、再配置、再激活”的受管状态，而不是当前脚本式一把梭。官方 ROS2 设计文档明确把 managed nodes 的目标定义为让外部监督进程在节点真正开始执行行为前确认各组件已正确实例化；ROS2 也支持用 callback groups 控制哪些回调必须串行、哪些可以并行。citeturn6search0turn3search2turn5search0turn5search2

我的最终建议可以浓缩成一句话：**把“AVP 输入 → 身体目标构造 → 身体 IK/换档 → 命令仲裁 → 机器人桥接”做成一条严格时间戳化的控制面，把“视频 → 录制 → 可视化 → 调试回放 → 姿态编辑”做成旁路数据面**。这样阶段一即可在 MuJoCo 内闭环追平当前效果，阶段二只替换 backend 即可上真机，阶段三再把沉浸显示和训练数据体系叠上去。fileciteturn0file0

## 面向真机的节点划分与通信架构

我建议把系统拆成四个 ROS2 package 和两类后端实现：

- `avp_teleop_msgs`：自定义消息、服务、action。
- `avp_teleop_core`：纯 Python 业务库，放重定向、IK、换档、姿态运动原语、回放逻辑，不依赖 ROS2。
- `avp_teleop_nodes`：ROS2 wrapper 节点。
- `avp_robot_bridge`：MuJoCo backend 与真机 backend。
- `MujocoBackend`：仿真实现。
- `AstribotRos2Backend`：真机实现。fileciteturn0file0

我建议的新节点图如下：

```text
[Operator IO]
  avp_ingress_node
    UDP(AVPE/AVPH) -> /teleop/raw/head
                   -> /teleop/raw/hand_left
                   -> /teleop/raw/hand_right

  keyboard_override_node
    keydown -> /teleop/ui/manual_override
           -> /teleop/ui/command_event

[Control Core]
  calibration_anchor_node
    raw head/hand + robot_state -> /teleop/calib/state

  body_target_builder_node
    raw head/hand + calib -> /teleop/targets/body

  hand_retarget_node
    raw hand + optional hand calib -> /teleop/commands/hand

  body_ik_node
    body targets + robot_state + manual_override -> /teleop/commands/body_ik
    locally owns:
      ManualOverrideController
      GearLock
      damping scheduler
      freeze logic
      WholeBodyIK

  motion_primitive_node
    /teleop/ui/command_event + robot_state + pose_store -> /teleop/commands/body_motion

  replay_controller_node
    replay source + robot_state -> /teleop/commands/body_replay

  command_mux_guard_node
    arb: estop > goto_pose > replay_align > replay > live_ik
    outputs:
      /teleop/commands/body_active
      /teleop/system/motion_state

[Backend]
  robot_state_aggregator_node
    sim or real topics -> /robot/state/full

  robot_command_bridge_node
    /teleop/commands/body_active -> sim ctrl or official joint_space_command
    /teleop/commands/hand -> sim finger ctrl or gripper/hand topic

  robot_view_bridge_node
    robot camera topics -> AVP display/WebRTC path

[Tools / Data]
  mujoco_visualizer_node
  trajectory_debug_recorder_node
  episode_manager_node
  sim_episode_recorder_node
```

这套架构里，最关键的不是“节点越多越好”，而是**边界必须按控制责任划分**。`body_ik_node` 是唯一有权决定冻结 base、切换 P/N/D、调 damping、调用 WholeBodyIK 的节点；`robot_command_bridge_node` 只是一个 dumb bridge，只管把统一关节命令拆成 torso/head/arm/chassis/gripper 多个 `RobotJointController`。这样，手动换档不会在最后一跳执行层引入“突然改发别的关节值”的跳变，而只是改变 IK 的解空间与调度状态。fileciteturn0file0

### 键盘手动换档节点的具体安置

这是你特别关心的点。我建议**键盘输入一定做成独立 ROS2 节点**，不要继续留在主循环或 MuJoCo viewer callback 里；但**ManualOverrideController 和 GearLock 的状态机不要移出 IK 进程**。更具体地说：

`keyboard_override_node` 只负责两件事：  
一是监听键盘事件；  
二是把它们发布成**高层事件**，而不是直接改任何机器人命令。fileciteturn0file0

我建议定义两个消息：

```text
# teleop_msgs/msg/ManualOverrideEvent.msg
std_msgs/Header header
uint32 seq
string axis          # "base_trans" | "base_yaw" | "body_pitch"
string gear          # "P" | "N" | "D"
builtin_interfaces/Duration duration
string source        # "keyboard"

# teleop_msgs/msg/UiCommandEvent.msg
std_msgs/Header header
string command       # "recalib" | "pause" | "goto_named_pose" | "save_pose" | "toggle_view" | ...
string arg
```

然后让 `body_ik_node` 内部持有你现有那套 `ManualOverrideController(override_duration=5.0)` 与 `GearLock(dwell_time=2.5s)` 逻辑：收到 `ManualOverrideEvent` 后更新本地 override 状态、延长 expiry、维护 damping EMA 与档位 dwell，而不是让 `keyboard_override_node` 自己算出“最终档位状态”。原因很简单：**平滑交接必须和 IK 求解共享同一个时钟、同一份上一帧状态、同一条 damping EMA 链**。现在你的报告里也明确写了，当前平滑交接之所以不突跳，是因为手动层在自动逻辑之前、并且手动期间也维护 damping EMA 状态；这部分必须留在 IK 节点内。fileciteturn0file0

在 QoS 上，我建议把 `/teleop/ui/manual_override` 设成**Reliable + Transient Local + depth 1**。因为这是低频、状态敏感的控制事件，不适合 BEST_EFFORT；同时新加入的 `body_ik_node` 应该能拿到最近一次 override 状态。ROS2 的 QoS 兼容性文档明确指出，可靠性和 durability 不匹配会导致不通信；而你对官方 `joint_space_command` 的发布则必须匹配对端的 BEST_EFFORT + VOLATILE。内部 UI 事件和外部机器人的 QoS 策略，应该故意分开。citeturn0search1turn5search1

### 控制频率与线程模型

频率上我建议这样定：

| 节点 | 建议频率 | 说明 |
|---|---:|---|
| `avp_ingress_node` | 源驱动 | AVP 到数据就发，保留原始时间戳 |
| `body_target_builder_node` | 60–100 Hz | 仿真先 60 Hz；真机建议 100 Hz |
| `body_ik_node` | 60 Hz 仿真 / 100 Hz 真机 | 真机尽量贴近官方 action 录制频率 |
| `hand_retarget_node` | 60–100 Hz | 独立于身体 |
| `motion_primitive_node` | 100 Hz | 与 `body_ik_node` 同频输出 |
| `robot_state_aggregator_node` | 事件触发 | 订阅状态，汇总后更新缓存 |
| `robot_command_bridge_node` | 跟随上游 | 收到 active 命令立即拆包发布 |
| `trajectory_debug_recorder_node` | 事件触发 | 调试轨迹 |
| `sim_episode_recorder_node` | 100 Hz 采样 | 训练采样时钟 |

这样安排的原因，是你的现系统在 MuJoCo 里以 60 Hz 闭环运行；而真机官方 recorder 记录 `joint_space_command` 为 100 Hz、`joint_space_states` 为 250 Hz。把真机控制频率升到 100 Hz，会让“你实际发出的 action”和“被 recorder 标注的 action 采样率”天然一致，后续做 imitation data 更整齐。fileciteturn0file0

线程上，我建议控制类节点都尽量采用“**一个定时控制回调 + 若干订阅回调缓存最新值**”的模型，而不是在订阅回调里直接求解。ROS2 文档建议用 callback groups 来控制并发：订阅回调可以放在 reentrant 组里更新缓存，控制定时器放在 mutually exclusive 组里独占运行，避免同一时刻既在改缓存又在求解。citeturn3search2turn3search0

## 仿真与真机的代码复用边界

你现有的 `RobotInterface(ABC)` 思路是对的，但在 ROS2 化后，我建议把复用边界再画得更清楚一些：**复用的是纯业务库，不是 ROS node 本身**。也就是说，ROS2 节点只是薄壳，真正复用的是下面这些核心类：

```text
avp_teleop_core/
  body_target_builder.py
  whole_body_solver.py
  gear_scheduler.py
  motion_primitives.py
  replay_engine.py
  pose_store.py
  hand_plugin_api.py
  robot_backend.py
```

其中最重要的是三套协议：

```python
class RobotBackend(Protocol):
    def read_state(self) -> RobotStateFull: ...
    def send_body_command(self, cmd: JointCommandStamped) -> None: ...
    def send_hand_command(self, cmd: HandCommandArray) -> None: ...

class BodySolver(Protocol):
    def reset(self, q_measured: np.ndarray) -> None: ...
    def solve(
        self,
        state: RobotStateFull,
        targets: BodyTargets,
        gear_state: GearState,
        dt: float
    ) -> JointCommandStamped: ...

class HandRetargeterPlugin(Protocol):
    def reset(self) -> None: ...
    def retarget(
        self,
        obs: HandObservation,
        hand_state: HandActuatorState,
        dt: float
    ) -> HandCommand | None: ...
```

这样，MuJoCo 和真机只在 `RobotBackend` 上分叉；WholeBodyIK、gear scheduler、pose primitive、replay align 都不分叉。你的报告已经说明，当前 `ROSRobot` 只是占位，但 `SimRobot` 和 `ROSRobot` 的抽象面是可以让主循环不改而替换 backend 的，这正是值得保留并扩大的设计。fileciteturn0file0

### 统一消息面与 backend 适配层

我建议上游控制面统一使用**全身统一关节命令**，下游 bridge 再拆成官方部件命令。原因有三个。第一，你的 IK 本来就是 20–23 DOF 单求解器，内部天然需要统一关节序。第二，回放对齐、goto pose、轨迹插值这些运动原语也是“全身命令源”，不应该分别生成 torso/head/arm_left/arm_right 四类命令再外部拼接。第三，统一命令流更适合录制成 training action。fileciteturn0file0

建议统一消息如下：

```text
# teleop_msgs/msg/JointCommandStamped.msg
std_msgs/Header header
string source                  # "teleop_ik" | "goto_pose" | "replay"
string[] joint_names
float64[] position
float64[] velocity             # optional, allow empty
float64[] effort               # optional, allow empty
bool is_hold
```

`robot_command_bridge_node` 收到这条消息后，按固定映射表拆成：

- `/astribot_torso/joint_space_command`
- `/astribot_head/joint_space_command`
- `/astribot_arm_left/joint_space_command`
- `/astribot_arm_right/joint_space_command`
- `/astribot_chassis/joint_space_command`
- `/astribot_gripper_left/joint_space_command`
- `/astribot_gripper_right/joint_space_command`。fileciteturn0file0

这里要特别强调：你报告里已经查清楚官方接口语义是 `RobotJointController`，`mode=1` 时 `command` 长度 N 表示纯位置，如果能提供 2N 也可以带速度；并且这些话题的 QoS 是 BEST_EFFORT + VOLATILE。ROS2 官方 QoS 文档同时说明了 reliability 和 durability 的兼容矩阵：如果发布端和订阅端在这些策略上不兼容，就会直接不通信。因此 `AstribotRos2Backend` 必须把 QoS 显式写死，不要依赖默认值。fileciteturn0file0 citeturn0search1turn5search1

### 仿真后端与真机后端的具体职责

`MujocoBackend` 应负责三件事：  
一是把 `/teleop/commands/body_active` 写入 MuJoCo `data.ctrl`；  
二是把当前 qpos、qvel、tool pose 发布成和真机一致的 `/robot/state/full`；  
三是发布 MuJoCo 渲染的 robot-view 图像供 recorder 和 AVP view bridge 使用。这样阶段一的整个 ROS2 架构就已经是闭环，而不是“只是为了以后上真机先包一层 ROS”。fileciteturn0file0

`AstribotRos2Backend` 则负责：  
一是订阅各 `/<part>/joint_space_states` 与 `/<part>/endpoint_current_states`，汇总成统一 `RobotStateFull`；  
二是把统一命令拆成各个 `joint_space_command`；  
三是做首帧对齐、命令限速、连接健康检查、必要的 control-rights 申请和释放；  
四是透传机器人头部相机流给 view 与 recorder 系统。你报告里已经指出需要先读 `joint_space_states` 做首帧安全、`--upper-body-only` 在官方多部件接口下天然成立、控制权 JSON 协议尚缺、灵巧手话题定义尚缺，这些都应该收敛在 backend 层，而不是污染 IK 和重定向层。fileciteturn0file0

## 真机安全推进路线与 IK 求解器改造

### 真机 bring-up 的推荐顺序

我建议你把真机 bring-up 做成**严格受管的五步**，每一步都可以独立验收，不通过就绝不进入下一步。这也符合 ROS2 生命周期设计中“先 configure，再 activate”的思想。citeturn6search0

第一步是**只读接入**。此时只启动 `robot_state_aggregator_node`，验证能稳定收到各 `joint_space_states` 和 `endpoint_current_states`，并把它们汇总成统一 `RobotStateFull`。验收标准不是“能看到话题”，而是“全身关节命名顺序与内部 `BODY_JOINTS` 一一对应、头部在真机里的 `astribot_head` 能正确映射为你逻辑中的 neck 2-DOF、tool pose 与 FK 误差在可接受范围内”。你报告已经指出 neck 在真机话题里叫 `astribot_head`，这一点必须在 state aggregator 固化掉。fileciteturn0file0

第二步是**零动作写入**。申请 control rights，但仍不让 teleop 接管；只由 `robot_command_bridge_node` 以低频发布“当前测得 q 的镜像命令”。如果此时机器人仍有缓慢漂移、抖动或关节排序错乱，就说明 bridge 或命名映射有问题。这个阶段解决的是最危险也最常见的错误：关节名映射错位、单位错、QoS 不通、对端把停止发布解释为保持旧目标等。fileciteturn0file0

第三步是**`--upper-body-only` + goto pose 原语**。保持底盘 xy、底盘 yaw、倾斜脊柱全部冻结，只允许 torso yaw、head、双臂动作；此时唯一允许的主动动作是“平滑运动到保存姿态”。你的报告里已经说明，当前 `--upper-body-only` 正是通过 `set_base_xy_frozen(True)`、`set_base_yaw_frozen(True)`、`set_lean_frozen(True)` 实现的，并且自动切换块都被守卫禁用。把这个模式保持为真机的默认 bring-up 模式是正确的。fileciteturn0file0

第四步是**单臂与双臂静态目标跟踪**。这时接入 AVP，但先关闭 base_trans/base_yaw/body_pitch 自动调度，全部固定在保守档位。先一侧手腕位姿跟踪，再双臂，再头任务。只有在 measured-state 闭环下，双臂和头都能稳定跟踪且无明显自激振荡后，才可以进入下一步。这里“measured-state 闭环”很关键：IK 每一帧都必须以 `joint_space_states` 实测角作为 q，而不是以“上一帧自己发出的命令”作为 q。当前仿真里这两者接近等价，真机会完全不同。fileciteturn0file0

第五步是**逐轴解锁移动体**。先腰偏航，再 body_pitch，再 base_yaw，最后 base_trans。并且每放开一轴，都先只允许手动档位 P/N，最后再打开自动调度。你现有系统的档位体系已经把这三轴设计成正交的 P/N/D 模式，这是非常适合真机逐轴 bring-up 的。fileciteturn0file0

### 首帧对齐、限速、急停的落地做法

首帧对齐不能只做一次“第一帧 = 当前角度”，还要做成一个**显式的对齐状态**。我建议 `command_mux_guard_node` 在 teleop 真正接管之前进入 `ALIGNING` 状态，由 `motion_primitive_node` 把机器人从当前姿态平滑送到“teleop solver 当前目标对应的最近可行姿态”，或者送到回放首帧/指定姿态。只有当 `max(|q_target - q_measured|)` 和 tool pose 残差同时低于阈值，系统状态才从 `ALIGNING` 过渡到 `LIVE_TELEOP`。这样做比“第一帧复制当前姿态”更稳，因为它显式处理了 teleop 激活前目标已经偏了很多的情况。fileciteturn0file0

限速我建议分成两层。第一层仍在 IK/QP 内保留你现在已有的 `VelocityLimit`、`AccelerationLimit`、`ConfigurationLimit` 以及 `LowAccelerationTask`。Pink 文档明确支持 limits、constraints 和 barriers，并且 `safety_break` 可以在当前配置越界时直接报错；这让解算器层负责“不要给出明显不合法的速度”。第二层在 `robot_command_bridge_node` 再加一个**基于实测状态的 command filter**：对 position delta、velocity、acceleration 逐关节裁剪。之所以一定要有第二层，是因为 Pink 自己也说明其 Tikhonov damping 是各向同性的，而浮动/移动基座与普通转动关节的速度空间并不齐次；你当前用的逐 DOF `DampingTask` 已经在补这个问题，但真机上仍需要一层与实测状态绑定的最终限幅器。citeturn9view0turn9view1turn9view2

急停我建议做成**双层机制**。第一层是硬件/官方 estop，不依赖 ROS2；第二层是软件 `estop`，由 `command_mux_guard_node` 最高优先级抢占，立即切到 `HOLD_CURRENT` 或 `RELEASE_CONTROL` 策略。由于你的报告已经指出 control-rights JSON 协议尚未拿到，因此“软件 estop 时是持续发当前角，还是释放 control rights 给官方待机模式”这件事目前不能武断下结论，必须等官方协议确认。现阶段最稳妥的默认策略，是**切到 measured-current hold 并停止一切上游动作源**。fileciteturn0file0

### 删掉 EgoPoser 与 base_follow 之后，IK 应怎么改

你要求删掉 `NeuralPostureTask/EgoPoser` 和 `BaseTrackingTask/base_follow`，这不仅可行，而且我认为应该顺手把 whole-body solver 清理成一个更容易维护的“最小必要任务集”。推荐保留的核心集合如下：

| 类别 | 任务 | 去留建议 |
|---|---|---|
| 主任务 | `FrameTask(head)` | 保留 |
| 主任务 | `FrameTask(left tool/right tool)` | 保留 |
| 零空间正则 | `PostureTask(home)` | 保留 |
| 平衡 | `ChestOverAnkleTask` | 保留，且提高地位 |
| 姿态软正则 | `TrunkUprightTask` | 保留 |
| 平滑 | `LowAccelerationTask` | 保留 |
| 稳定性代价 | 逐 DOF `DampingTask` | 保留，但重构参数化 |
| 特定模式增强 | D 档 `waist-zero` | 保留 |
| 神经先验 | `NeuralPostureTask` | 删除 |
| 反应式底盘跟头 | `BaseTrackingTask` | 删除 |
| 遗留质量法平衡 | `ComTask` | 不复活 |

这张表基本就是把你报告中的“当前有效任务”和“已验证无效任务”整理成了新默认。现有报告也记录了 `base_follow` 在 clip9/10/11 上明显劣化手部跟踪，这已经足够支持彻底删除，而不只是继续保留一个 cost=0 的死代码分支。fileciteturn0file0

但删掉之后，我建议新增两个加强点。第一个是**残差驱动的底盘参与逻辑**，取代“头部驱动底盘”。也就是：不要看头水平位移去决定底盘该不该跟，而是看**手任务残差、手臂 manipulability 下降、关节接近极限**这些更直接描述“手已够不到”的信号，来决定 base_trans/base_yaw 是否从 P 或 N 进入 D。这个建议不是复活 base_follow，而是把“移动平台参与求解”的触发条件从头部运动改成**任务可达性/残差驱动**。这是我基于你报告中 base_follow 的失败原因和 Pink 多任务框架做的工程推断。fileciteturn0file0 citeturn2search0turn9view0

第二个是**把软惩罚部分升级成 barrier 思想**。Pink 文档支持 barriers；相比把 `ChestOverAnkleTask` cost 一路调大，我更建议你补一个简单的 `SupportRegionBarrier` 或 `TorsoLeanBarrier`，让系统在“接近不安全俯身”时强烈抬高代价，而不是等残差已经很大才靠软代价慢慢拉回。这对真机尤其重要，因为“移动底盘 + 倾斜脊柱”组合会把胸部投影、安全边界、手到目标的 reachability 三者耦合得更强。citeturn9view0turn9view2

## 手部独立求解、姿态归位与交互层

### 身体 IK 与手部重定向的解耦接口

你已经明确要求身体与手/夹爪彻底解耦，而且实验室当前还没有灵巧手，需要确保“手部数据缺失时身体照常工作”。这意味着新架构里，**身体模块绝不能依赖手指 keypoints 的存在才启动**。身体模块只依赖 head pose 和左右 wrist/tool 目标；手部模块独立消费更细的 hand skeleton/keypoints。你报告里也已经说明了当前虽然逻辑上分成身体 IK 和 `HandRetargeter → command_fingers` 两条路径，但它们仍耦合同一主循环里，而且缺手腕时仅用 `live[end] is None` 守卫，不是真正的模块解耦。fileciteturn0file0

我建议把接口收敛成下面这一层协议：

```python
@dataclass
class HandObservation:
    stamp_ns: int
    side: Literal["left", "right"]
    wrist_pose_world: Optional[np.ndarray]      # 4x4
    keypoints_world: Optional[np.ndarray]       # (21, 3)
    pinch: Optional[float]
    valid_wrist: bool
    valid_keypoints: bool

@dataclass
class HandCommand:
    stamp_ns: int
    side: Literal["left", "right"]
    joint_names: list[str]
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None
    mode: Literal["position", "velocity", "effort"] = "position"
    quality: float = 1.0
    hold_last: bool = False
    timeout_ms: int = 150

class HandRetargeterPlugin(Protocol):
    def configure(self, robot_model: dict, cfg: dict) -> None: ...
    def reset(self) -> None: ...
    def retarget(
        self,
        obs: HandObservation,
        actuator_state: Optional[dict],
        dt: float
    ) -> HandCommand | None: ...
```

这个签名的重点不在 Python 语法，而在**契约**：  
如果 `valid_keypoints=False`，插件可以返回 `None` 或 `hold_last=True` 的命令；  
如果 `valid_wrist=True` 但没有 keypoints，身体仍正常运行，手部模块只是不更新 finger command；  
如果未来同事接入 BrainCo 或别的灵巧手，只需要实现这个 plugin，不需要去碰 WholeBodyIK、gear scheduler、motion primitive 或主控制循环。fileciteturn0file0

### 手部数据缺失时身体的降级策略

我建议把降级分成三档，而不是单一地“缺了就不更新”。

第一档，**有 wrist pose、无 keypoints**：身体 IK 正常使用 wrist 对应的 tool target；手部命令保持上一帧，超过手部独立 timeout 后 hand executor 自己转 safe hold。  
第二档，**短时无 wrist pose**，比如丢包 100–150 ms：身体对该侧 arm target 做 hold-last-target，并保持 `LowAccelerationTask` 与 posture regularization 继续工作。  
第三档，**长时无 wrist pose**：该侧手臂从 hold-last 平滑退回“当前实测关节姿态”和 `BODY_HOME` 之间的某个缓释 posture anchor，而另一侧臂、头、腰继续工作，不触发系统级错误。fileciteturn0file0

这样做的原因，是你已经明确说了“没有手部数据时身体也要正常驱动躯干和手臂，绝不能因为缺手数据就卡住或报错”。在工程上，这就意味着身体求解必须把左右手目标视为**各自可选的输入通道**，而不是“一个完整的双手输入包”。fileciteturn0file0

### 回放前归位与运行时 go-to-pose 原语

这一块我建议你新增一个独立的 `motion_primitive_node`，提供两个 action：

```text
# teleop_msgs/action/GoToNamedPose.action
string pose_name
float64 velocity_scale
float64 acceleration_scale
bool hold_after
---
bool success
string message
---
float32 progress

# teleop_msgs/action/PlayTrajectory.action
string path
bool align_first
float64 replay_speed
---
bool success
string message
---
float32 progress
```

内部原语只有一个：**从当前实测 q 到目标 q 的限速、限加速度、可中断的关节空间插值器**。`PlayTrajectory` 如果 `align_first=true`，就先把录制首帧的关节角作为目标调用同一个 go-to primitive，到位之后再切换成 replay source。这样，“回放前归位”和“运行时按键运动到指定姿态”实际上复用的是同一个内核，而不是两套逻辑。你报告里已经明确指出当前缺的正是这个“关节空间轨迹插值运动原语”，并且 replay 现状是直接 set qpos，真机绝不能这么做。fileciteturn0file0

实现上，我建议插值器用**trapezoidal velocity / S-curve jerk-limited** 二选一。若你希望第一版尽快落地，先做 trapezoidal velocity 就够；真正重要的是它必须**以实测 q 为起点，而不是以上一帧命令 q 为起点**。对于 position-servo 式的官方接口，这会明显降低“理论命令到了，实际还没到”的累积误差。fileciteturn0file0

### 姿态编辑、保存与运行时交互的归属

我不建议把 `pose_editor.py` 完全塞回主程序，但也不建议继续让“姿态保存”和“姿态执行”是割裂两套入口。更好的做法是：

- **保留 `pose_editor.py` 作为离线姿态作者工具**，因为它在 MuJoCo 里摆姿很高效。
- 新增 `pose_store_node`，统一管理 `poses/*.json`。
- 新增两个服务：  
  `SaveCurrentPose.srv(name)`  
  `ListNamedPoses.srv()`  
- 把键盘交互层里的“运动到指定姿态”和“保存当前姿态”配成一组命令，例如 `g` 打开 goto named pose，`Shift+g` 保存当前为 named pose。fileciteturn0file0

也就是说，**编辑器继续独立；运行时交互变成“同一套 pose store 的读写前端”**。这样既不破坏你现在的工作流，又把 runtime go-to-pose 和 pose save 放进了同一个交互层，用户认知也更一致。fileciteturn0file0

## AVP 视角切换与训练数据采集

### robot-view 与 self-view 的架构位置

这里最重要的原则只有一条：**视角切换属于显示层，不属于控制层**。也就是说，操作者看 robot-view 还是看 AVP 自己的 passthrough，不应该改变身体 IK 输入、不应该改变录制到训练数据中的 observation，也不应该影响 action 标签定义。你的报告已经非常清楚地指出：AVP 上行的是操作者自己的头手 tracking，只是遥操输入；训练数据真正需要的是“机器人视角图像 + 机器人本体状态”映射到“机器人动作命令”。fileciteturn0file0

因此我建议新增一个 `view_mode_node` 或更简单地，把它并入 `robot_view_bridge_node`，只发布一个很轻的状态话题：

```text
# teleop_msgs/msg/ViewMode.msg
std_msgs/Header header
string mode   # "robot_view" | "self_view"
string source # "keyboard" | "avp_gesture"
```

触发上，可以同时支持电脑键盘热键和 AVP 端手势；但从系统结构上，**切换动作最好在 AVP app 或 view bridge 里消费，不要进入 body_ik_node**。`body_ik_node` 只需要知道 teleop 是否 paused，不需要知道你现在看的是哪种视角。fileciteturn0file0

### robot-view 视频通路怎么接

你报告里已经说明，机器人头部相机画面走的是官方 `camera_driver` + WebRTC/H264 下行到 AVP 屏幕的路径；这说明对真机来说，最佳方案通常不是你在 PC 上“重新订阅压缩图像再自己做第二套视频栈”，而是**优先复用官方 robot-view 视频链路**，你只需要在系统层面管理“当前显示 robot-view 还是 self-view”的状态即可。fileciteturn0file0

对仿真则不同：MuJoCo 没有官方二进制视频链路，所以建议 `MujocoBackend` 直接发布：

- `/robot/camera/head/color/image_raw`
- `/robot/camera/head/color/camera_info`

如果你希望仿真期也模拟真机的 “robot-view bridge → AVP” 通路，可以把 `robot_view_bridge_node` 设计成双后端：  
真机端使用官方 WebRTC/H264；  
仿真端使用 ROS image topic + 本地 WebRTC sender。  
但无论哪种后端，**训练 recorder 都直接订阅原始 robot camera topic，不订阅 AVP 显示结果**。这样 view mode 只改变人看到的画面，不影响 dataset。fileciteturn0file0

### 真机数据采集应复用官方 recorder

这一点结论非常明确：**真机上不要自建 recorder 主链路，直接复用官方 `astribot_recorder` → MCAP → HDF5 管线。** 你的报告已经查明，这套官方管线会录 5 路 RGB、4 路深度、250 Hz `joint_space_states`、100 Hz `joint_space_command`，并且有 PTP 时间同步与 HDF5 episodic 输出；更关键的是，它录的 action 标签正是 `joint_space_command`，而你准备走的也正是这条关节命令接口。这意味着，你只要往官方 `joint_space_command` 话题发命令，官方 recorder 就能把你的 IK action 与机器人视角和状态一起打进训练数据。fileciteturn0file0

在存储格式上，这个选择也更合理。ROS2/rosbag2 在 Iron 之后默认推荐 MCAP，MCAP 本身就是面向多通道时间戳数据的自包含格式，支持把 message schema 一起存进去，便于长期可读和离线分析；官方管线再从 MCAP 转 HDF5，用于训练。对真机而言，这是远比你在 PC 端重新抓图、重新取时间戳可靠得多的路径。citeturn4search0turn4search1turn4search2turn4search4

我建议你在真机侧只新增一个很薄的 `episode_manager_node`，职责不是录制数据本身，而是：

1. 调用官方 `/astribot_storage/storage_service` 或对应 action 开始/停止/保存 episode；  
2. 记录本次 teleop 元数据，比如 operator、task label、是否成功、视角切换日志、当前 teleop config snapshot；  
3. 在 episode 保存后，写一个 sidecar metadata 文件，供后处理转换器合并。fileciteturn0file0

### 仿真阶段要自建“官方 HDF5 等价 recorder”

仿真没有官方 ARM 二进制 recorder，因此我建议你新增 `sim_episode_recorder_node`，目标不是发明一种新格式，而是**尽量对齐官方 HDF5 布局**，再额外提供导出到 LeRobot v3 的脚本。原因是：  
对齐官方 HDF5，才能让仿真数据与真机数据最容易混合；  
导出 LeRobot v3，则便于后续接 Hugging Face/LeRobot 训练生态。fileciteturn0file0 citeturn8view0

建议仿真的 canonical episode schema 这样定：

```text
/root
  /metadata/
    episode_id
    task_name
    success
    failure_reason
    teleop_config_yaml
    calibration
    view_mode_log
    source = "sim"
    clock = "ros_sim_time"

  /images_dict/
    head/rgb          [T_img, H, W, 3]
    left_wrist/rgb    [optional]
    right_wrist/rgb   [optional]

  /joints_dict/
    torso/states/{position, velocity}
    head/states/{position, velocity}
    arm_left/states/{position, velocity}
    arm_right/states/{position, velocity}
    chassis/states/{position, velocity}
    torso/command/{position, velocity}
    ...
  
  /command_poses_dict/
    left_tool_target
    right_tool_target
    head_target

  /timestamps/
    sample_time_ns
    state_time_ns
    image_time_ns
    command_time_ns
```

这套布局在命名上尽量贴近你报告里已经确认的官方 HDF5 分组方式：`images_dict/...`、`joints_dict/<part>/{states,command}`、`command_poses_dict`。fileciteturn0file0

### 多模态时间同步应该怎么做

对异步模态，**不要依赖“收到就拼一下”的到达时间同步**。ROS2 `message_filters.ApproximateTimeSynchronizer` 明确是按消息 header 时间戳同步，并且官方文档明确说 `allow_headerless=True` 应尽量避免，因为那会退化到用当前 ROS 时间、延迟不可预测。这个工具适合做某些在线调试视图，但不适合直接做训练 recorder 的最终真值对齐。citeturn0search4

我建议 recorder 采用**固定采样时钟 + 最近样本回填**的事件源模式。以 100 Hz 为 episode 主时钟，在每个 `t_k`：

- action：取最新且 `stamp <= t_k` 的 `JointCommandStamped`
- state：取最近的 `RobotStateFull`
- image：取最近的 robot-view 帧
- 同时把三者各自原始时间戳都写入数据集

这样做有三个好处。第一，动作时序和你真机上的 `joint_space_command` 采样率对齐。第二，图像与状态本来就异步，不会被强行“伪同步”成同一时刻。第三，如果以后要导出到 LeRobot v3，LeRobot 的表格数据和视频本身就是分离存储、靠 metadata 和时间字段恢复 episode 视图的，这和你这种“保留原始 stamps、再定义训练 sample clock”的策略是兼容的。LeRobot v3 文档也明确把数据分成 tabular Parquet、video MP4 和 metadata 三层，并要求录制后调用 `finalize()` 完成数据集闭合。citeturn8view0

因此我的建议是：  
**真机 canonical = 官方 HDF5；仿真 canonical = 官方 HDF5 对齐版；训练生态导出 = LeRobot v3，必要时再转 RLDS。**  
不要把 LeRobot 直接当你的唯一落盘格式；它更适合作为面向训练的导出层，而不是替代真机官方 recorder 的主存储。fileciteturn0file0 citeturn8view0

## 风险清单与三阶段落地顺序

### 现在最值得提前规避的设计债

从你报告里看，我认为最危险的坑有七个。

第一是**帧一致性和坐标锚点**。你现在在 MuJoCo 里通过 flatten MJCF + Pinocchio model 保证了 frame 一致性，末端任务帧与模拟器重合误差小于 `1e-4 m`；但真机没有“MuJoCo 世界系 = 机器人世界系”这回事。因此你必须把 `avp_world`、`teleop_anchor`、`robot_base_start`、`robot_base_live`、`head/tool target` 几个概念显式写进 tf tree 或内部坐标文档，不要把“当前代码里世界系恰好统一”误以为可以原样迁移到真机。`tf2` 的核心就是维护带时间缓存的 frame tree，这正适合承载这些锚点关系。fileciteturn0file0 citeturn3search3

第二是**状态反馈闭环缺失**。当前仿真里 `q_body` 既是 solver 状态又几乎等于执行结果，真机上绝不成立。如果 body IK 继续以内存中的上一帧命令角作为当前 q，而不是订阅 `joint_space_states` 的实测角，就会把控制误差和通信延迟全吞进 solver，表现为漂移、追踪滞后和档位切换怪异。fileciteturn0file0

第三是**QoS 静默失败**。这点你报告已经专门点出来了，而 ROS2 QoS 兼容性文档也证实了这种“订阅/发布策略不兼容就不通信”的机制。解决办法不是靠人工记忆，而是在 `AstribotRos2Backend` 启动时打印每个 publisher/subscriber 的实际 QoS，并在 graph discovery 后做 matched endpoint 检查。fileciteturn0file0 citeturn0search1turn5search1

第四是**时间同步**。真机官方 recorder 用了 PTP，这说明 Astribot 官方体系本身也把时间同步看成硬需求。你自建仿真 recorder 时，如果不用统一 ROS clock 或 monotonic clock，而是随手混用 wall-clock、arrival-time、Python `time.time()`，最后数据集会很难和真机混合。fileciteturn0file0

第五是**控制权协议未知**。报告已经明确写出 `/astribot/control_rights` 与 `/simu_real_switch_service` 的 JSON 协议内容目前未知，需要抓包或问官方。在拿到这套协议之前，不要把“自动申请控制权、自动释放、软件 estop 转待机”写死成无回退流程。fileciteturn0file0

第六是**灵巧手话题定义未知**。这正是为什么我建议你把 hand plugin 的边界先定义干净，但 hand executor 与机器人具体手部接口继续保持可替换。报告里也明确写了目前官方仓库只有 1-DOF gripper，灵巧手话题未确认。fileciteturn0file0

第七是**把阶段一做成“只有骨架、没有完整仿真闭环”**。你已经明确表示只有在 MuJoCo 里看到 ROS2 架构追平当前效果，才敢进入阶段二。这个风险不是技术性的，而是项目管理性的：如果阶段一没有把可视化 overlay、轨迹录放、gear HUD、仿真 backend 一起做通，后面所有真机问题都会和“新架构本身还没闭环”缠在一起，极难定位。fileciteturn0file0

### 三阶段落地方案

下面这三阶段，我按你给的优先级要求做了调整，但总体保持你的划分。关键是每个阶段都必须有**清晰产出、可验收门槛、为什么优先**。

#### 阶段一

阶段一的目标是：**完成 ROS2 化，但在 MuJoCo 内追平当前单进程系统的全部关键能力。** 包括：

- 节点拆分完成：`avp_ingress_node`、`keyboard_override_node`、`body_target_builder_node`、`body_ik_node`、`command_mux_guard_node`、`robot_state_aggregator_node`、`robot_command_bridge_node`、`mujoco_visualizer_node`、`trajectory_debug_recorder_node`、`replay_controller_node`
- `EgoPoser` 与 `base_follow` 彻底删除
- `WholeBodyIK` 清理成最小必要任务集
- 仿真 backend 生效，统一 `RobotStateFull` 与 `JointCommandStamped`
- 轨迹录制/回放 ROS2 化，保持你现有两类 JSON 调试轨迹
- MuJoCo overlay、gear HUD、误差线、回放 scrubbing 恢复fileciteturn0file0

这一阶段的验收标准必须非常硬：

1. AVP 实时遥操在 MuJoCo 中达到与当前项目**同等主观效果**；  
2. 键盘手动换档（2–9）在新架构里完全可用，5 秒超时释放、无明显跳变；  
3. `--upper-body-only` 在仿真中行为与旧版一致；  
4. 轨迹可录、可放，回放路径与旧版一致；  
5. 所有控制与录制节点都基于 ROS2 topic/action/service 工作，不再依赖“一个大 while loop 挂一堆状态对象”。fileciteturn0file0

它排第一，是因为这是你自己设定的进入真机的门槛，也是唯一能把“架构问题”和“真机问题”提前分离的阶段。新架构如果在仿真里都追不平旧版，就不应该碰真机。fileciteturn0file0

#### 阶段二

阶段二的目标是：**在不改变上游算法结构的前提下接入真机，并用最保守的安全路线完成首次受控闭环。** 包括：

- 实现 `AstribotRos2Backend`
- 补齐官方 `joint_space_command` / `joint_space_states` / `endpoint_current_states` 映射
- 显式配置 BEST_EFFORT + VOLATILE QoS
- measured-state 闭环替代命令内推
- `--upper-body-only` 真机 bring-up
- `motion_primitive_node` 上线，支持 go-to-named-pose 与 replay align
- 手/夹爪插件式接口上线，但允许 hand executor 仍用占位实现
- 软件 estop 与 source priority mux 上线fileciteturn0file0 citeturn0search1turn5search1

阶段二的验收标准建议是：

1. 不接 AVP 时，真机可稳定执行 go-to-named-pose；  
2. 只读状态 + hold-current mirror 无抖动；  
3. `--upper-body-only` 下，双臂与头任务可稳定 teleop；  
4. 运行时键盘换档在真机生效，且 gear 切换不引发关节命令尖跳；  
5. body 和 hand 在进程/模块上彻底解耦，hand 故障不会阻塞 body。fileciteturn0file0

它排第二，是因为这一步只替换 backend 与安全壳，不碰阶段一已经追平的上游控制骨架，风险最可控。fileciteturn0file0

#### 阶段三

阶段三的目标是：**把系统从“可控”提升到“可沉浸、可采数、可训练”。** 包括：

- `robot_view_bridge_node` 与 `view_mode` 机制上线
- 真机 `episode_manager_node` 对接官方 recorder
- 仿真 `sim_episode_recorder_node` 输出官方 HDF5 对齐版
- 离线 exporter：官方 HDF5 / sim HDF5 -> LeRobot v3
- episode 元数据、成功/失败标注、标定快照、任务标签统一管理fileciteturn0file0 citeturn8view0turn4search0turn4search1

这一阶段的验收标准建议是：

1. AVP 端可稳定切换 robot-view / self-view，切换不影响 teleop 闭环；  
2. 真机上能一键开始/停止 episode，并得到包含机器人相机、状态、动作的官方 HDF5；  
3. 仿真上能录出字段布局兼容的 episode；  
4. 训练侧能从真机与仿真数据中读取统一 observation/action 语义，至少完成一次混合数据读取验证。fileciteturn0file0 citeturn8view0

它排第三，是因为数据与沉浸都建立在“控制骨架稳定、真机 bridge 可信”之后。否则你会录到一堆无法解释的“坏数据”，后面训练和回放都会被污染。fileciteturn0file0

### 目前仍需你补充确认的信息

为了把阶段二和阶段三完全落地，还缺三类外部信息，而这些在你的报告里也已经明确标注为未确认项。

其一，**官方 control rights / mode switch 的 JSON 协议**。没有这个，软件 estop、自动接管/释放、开机 bring-up 只能先做成保守版本。  
其二，**灵巧手或 BrainCo 的真实 ROS2 话题/SDK 定义**。没有这个，你同事虽然可以按照 hand plugin API 开发，但最后一跳 executor 仍需要适配。  
其三，**真机相机视频回 AVP 的官方集成接口边界**：你现在已经知道有 `camera_driver + WebRTC/H264` 路径，但如果要把视角切换真正做进 AVP 应用，还需要知道切换是 AVP 端 UI 控制，还是 PC/机器人侧有显式接口。fileciteturn0file0

如果只看今天这份报告能支持到什么程度，我的结论是：**阶段一可以完全开始，而且应该立即开始；阶段二可以把 backend、go-to-pose、upper-body-only bring-up 的大部分先实现起来；阶段三里“真机复用官方 recorder”这个大方向已经足够明确，但控制权 JSON 和灵巧手接口仍需补齐。** fileciteturn0file0