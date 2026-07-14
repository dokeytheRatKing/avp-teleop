# AVP 全身遥操作 —— Retargeting 与 IK 技术报告

> 对象：`avp_teleop_upper_body/`（当前活跃的整机重定向包）
> 核心求解器：`WholeBodyIK`（`whole_body_ik.py`），基于 **Pinocchio + Pink** 的差分 IK
> 机器人：Astribot S1（3-DOF 移动底盘 + 4-DOF 躯干 + 2-DOF 颈部 + 双 7-DOF 手臂 = **23 DOF**）

本报告分三部分：
1. **解 IK 的原理**（数学与建模思想）
2. **工程实现**（代码结构、数据流、关键坑）
3. **目前加了哪些约束 / 任务**（逐条列举 + 权重 + 目的）

---

## 一、解 IK 的原理

### 1.1 整体思路：一个合并的差分 IK QP，而不是级联

这套系统**不是**"先解躯干、再各自解手臂"的级联结构，而是把 **底盘(3) + 躯干(4) + 颈部(2) + 左臂(7) + 右臂(7) = 23 个自由度放进同一个 `pink.Configuration`，在同一个 QP 里联合求解**。

为什么必须合并求解？关键在于运动学树的拓扑：

```
base
 └─ torso_joint_1..4 ──> astribot_torso_link_4  （双臂 + 头的共享根节点）
                              ├─ head_joint_1,2 ──> 头相机
                              ├─ left  arm (7)
                              └─ right arm (7)
```

`astribot_torso_link_4` 是**双臂和头共同的父节点**，而 `torso_joint_1..4` 又长在底盘上。因此只要躯干/底盘一动，两条手臂的基座就同时被拖动——"每条肢体独立求解"的假设直接破产。合并求解让 QP 天然获得**自动补偿（auto-compensation）**能力：

- 头部目标把躯干前倾时，QP 会同时重解双臂，使手仍停在世界目标上；
- 目标超出手臂工作空间时，QP 会平移/旋转底盘去延伸可达范围。

这是 VisionProTeleop `11_diffik_aloha.py`（双臂 FrameTask + posture）的模式，在此扩展了**头部任务 + 移动底盘 + 平衡任务**，并用 Pink（而非 mink）实现。

### 1.2 差分 IK 的数学形式

每个控制 tick（60 Hz）求解一个二次规划（QP），求解**关节速度** `v`：

```
min_v   Σ_k  w_k · ‖ J_k v − (gain/dt) · e_k ‖²        （各任务加权最小二乘）
s.t.    v_min ≤ v ≤ v_max                              （速度限位，硬约束）
        a_min ≤ (v − v_prev)/dt ≤ a_max                （加速度限位，硬约束）
        关节位置限位（Kanoun 2012 steer-away）           （构型限位，硬约束）
```

其中每个任务 `k` 提供：
- 误差 `e_k`（当前帧位姿与目标位姿之差）
- 雅可比 `J_k = ∂e_k/∂q`

解出 `v` 后用 `cfg.integrate(v, dt)` 在李群上积分得到新构型 `q`。由于**硬速度/加速度限位已在 QP 内部**，一个 tick 一次求解得到的步长本身就是物理可行的——不再需要老式的"先猛解、再 `np.clip` 截断步长"。求解器是在优化内部把限速和跟踪精度做权衡，而不是事后砍掉答案。

### 1.3 冗余度与任务优先级

23 个 DOF 去跟踪至多 18 维的末端任务（3 个末端 × 6D），有巨大的零空间。系统用两种手段处理冗余：

1. **PostureTask** 把零空间拉向 home 位姿（躯干直立、自然肘、底盘回原点）。
2. **per-DOF 的 DampingTask 代价向量**编码"全身运动优先级"——速度代价越大的 DOF 越"不愿动"，只有高优先级 DOF 够不着时才被征用。

优先级顺序（从"最先动"到"最后动"）：

```
手臂 + 颈部 + 腰部偏航 + 前倾脊柱   damping_cost = 0.1      （上半身优先干活）
移动底盘 (x, y, yaw)               damping_cost_chassis = 20（延伸可达时才动）
```

> **关键坑**：底盘的 frame 雅可比 ≈ 单位阵，即"移动底盘"在数学上是移动手最"便宜"的方式。若不给底盘一个**很高**的阻尼代价，QP 会一言不合就滑动整个底盘而不去伸手。这就是 `damping_cost_chassis=20 ≫ damping_cost=0.1` 的原因。

### 1.4 帧一致性保证

Pinocchio 模型由**展平后的 teleop MJCF** 构建（见 `_flatten_mjcf`）。所有任务 frame 与 MuJoCo 的 frame 重合到 < 1e-4 m，因此解出的关节角**无需任何坐标变换**即可直接写进仿真器。所有目标都在 MuJoCo 世界系下表达。

---

## 二、工程实现

### 2.1 数据流

```
Apple Vision Pro
   │  (UDP, 一个端口两种报文, 按 4 字节 magic 分流)
   │    AVPH -> 手腕 6D 位姿      AVPE -> 头部 6D 位姿
   ▼
UpperBodySubscriber.poll()          # transport.py: 排空 socket, 取每通道最新帧
   │
   ▼
wrist_to_tool_target()              # frames.py: 相对/增量重定向 (见 2.3)
   │   + PoseFilter (EMA + SLERP)   # pose_filter.py: 平滑抖动
   ▼
targets = {head, left, right}       # 世界系 (p, R)
   │
   ▼
WholeBodyIK.solve(q, head, left, right)   # 一次 QP -> 23 DOF
   │
   ├─ HandRetargeter.joint_targets(...)   # 手指曲屈重定向 (30 DOF)
   ▼
SimRobot.command_arm(q_body) / command_fingers(ft)  -> data.ctrl
   ▼
mj_step -> viewer
```

### 2.2 关键文件

| 文件 | 职责 |
|---|---|
| [whole_body_ik.py](whole_body_ik.py) | 合并 23-DOF 差分 IK 求解器 + 两个自定义平衡任务 |
| [config.py](config.py) | 关节顺序、home、任务权重、限速、平衡帧名 |
| [transport.py](transport.py) | 头部 `AVPE` UDP 通道 + 手/头合流订阅器 |
| [sim_teleop.py](sim_teleop.py) | 主遥操作循环：轮询→重定向→求解→驱动仿真 |
| `avp_teleop/retarget/frames.py` | AVP 手腕→机器人工具的相对位姿重定向 |
| `pose_filter.py` | 目标位姿的 EMA/SLERP 平滑 |

### 2.3 重定向（Retargeting）：相对增量映射

系统**不**尝试对齐 AVP 世界原点与机器人基座，而是采用**相对（增量）遥操作**（`wrist_to_tool_target`）：

- 标定时（按 `c`）记录操作者手腕位姿 `wrist0` 与机器人工具当前位姿 `tool0`。
- 每帧计算手腕相对标定时刻的运动，缩放后叠加到 `tool0` 上：

```
位置:   target_p = tool0_p + position_scale · (align_R · (wrist_p − wrist0_p))
姿态:   R_rel = wrist_R · wrist0_Rᵀ
        target_R = (align_R · R_rel · align_Rᵀ) · tool0_R
```

- `align_R`（`AVP_TO_ROBOT_R`）把 AVP 世界系向量旋到机器人世界系，修正"人面对机器人时左右/前后翻转"。**用旋转矩阵而非负 scale**，避免连带翻转竖直轴。
- `position_scale`（默认 1.0）放大手部平移；`track_orientation` 决定是否跟踪姿态。

好处：操作者按下标定键时可处于任意位姿，只有相对运动有意义；重新居中只需重新标定。

### 2.4 复合关节坑（load-bearing 的工程细节）

Pinocchio 的 MJCF 解析器会把底盘 3 个连续关节**合并成一个** `JointModelComposite`（名为 `Composite_astribot_chassis_x`，nq=nv=3，DOF 顺序 x,y,yaw = MuJoCo qpos 顺序）。于是 `config.py` 里存在**两份名单**：

- `BODY_JOINTS`（23 个 MuJoCo 单-DOF 名，底盘拆成 x/y/yaw）——命令/home/权重顺序；
- `IK_KEEP_JOINTS`（21 个 reduced-model 名，底盘=1 个复合关节）——喂给 Pinocchio。

`WholeBodyIK` 接收 `IK_KEEP_JOINTS` + `dof_names=BODY_JOINTS`，通过**把每个关节展开到它的 DOF 区间**来建立 `_q_index`/`_v_index`（多-DOF 关节无需特殊处理）。底盘位置限位是 ±inf（无限行程），由 posture task 把它拉回原点。

### 2.5 限位与平滑双层结构

- **硬约束**（QP 内部不等式，`pink.limits`）：`VelocityLimit` + `AccelerationLimit` + `ConfigurationLimit`。速度上限按 tangent DOF 逐个施加，单位无关：底盘 x/y = 0.6 m/s，底盘 yaw = 1.2 rad/s，躯干/颈 = 1.8 rad/s，臂 = 3.0 rad/s。
- **软任务**（低代价 QP 目标，在硬限位内塑形运动）：`DampingTask`（罚 |v|）+ `LowAccelerationTask`（罚 |v − v_prev|，有状态，逐 tick 喂入积分速度，标定时 `reset`）。代价远低于跟踪代价（arm=10, head=3），只平滑冗余量，不与末端目标对抗。

`_solve_velocity` 有优雅降级：靠近关节极限时可行域可能为空，则丢弃加速度限位重解一次，最坏情况保持不动（返回零速度）。

---

## 三、目前加了哪些约束 / 任务

求解器里的任务列表（`self._tasks`）按加入顺序如下。带 † 的是本项目自定义任务。

| # | 任务 | 类型 | 代价（默认） | 目的 |
|---|---|---|---|---|
| 1 | 头相机 FrameTask | 末端跟踪 | pos **3.0** / ori **1.0** | 头相机跟踪 AVP 头部 6D 位姿；代价适中，避免头猛拽躯干 |
| 2 | 左手工具 FrameTask | 末端跟踪 | pos **10.0** / ori **1.0** | 左工具帧跟踪左手目标 |
| 3 | 右手工具 FrameTask | 末端跟踪 | pos **10.0** / ori **1.0** | 右工具帧跟踪右手目标 |
| 4 | PostureTask(home) | 零空间正则 | **0.1** | 把 23-DOF 冗余拉向 `BODY_HOME`（直立躯干、自然肘、底盘回原点） |
| 5 † | **ChestOverAnkleTask** | **主平衡** | **50.0** | 防前倾：胸部（头/髋关节中点）地面投影保持在踝关节上方 |
| 6 † | TrunkUprightTask | 软次级平衡 | **0.5** | 前倾脊柱三关节角度和（≈躯干俯仰）软拉向 0，整理姿态冗余 |
| 7 | ComTask（水平） | 遗留质量平衡 | **0.0（关闭）** | 老式基于质量的 CoM 平衡，已被 5/6 取代，默认关闭 |
| 8 | DampingTask | 软平滑 + 优先级 | 逐-DOF：上身 **0.1** / 底盘 **20** | 罚 |v|：既平滑上身，又编码"底盘最后动"的全身优先级 |
| 9 | LowAccelerationTask | 软平滑 | **0.1** | 罚帧间加速度，降 jerk |
| 10 † | **NeuralPostureTask** | 软先验（可选） | **0.0（关闭）→ 0.8** | EgoPoser 幻想的人类躯干姿态：把躯干俯仰（前倾脊柱角度和）+ 腰部偏航软拉向先验值；代价极低，平衡/精度永远压过它。见 §3.4 |

此外还有**硬约束**（不在任务列表，而是 QP 不等式）：
- **VelocityLimit**（速度限位）
- **AccelerationLimit**（加速度限位，Flacco-style 刹车）
- **ConfigurationLimit**（关节位置限位，steer-away gain 0.5）

### 3.1 平衡设计的演化（重点）

平衡约束经过 2026-07-08→09 的迭代，是本项目最微妙的部分：

**老设计**：把平衡交给基于质量的 `ComTask` → 两个 bug：
1. **躯干前倾**——没有任何约束限制躯干俯仰，会倾到 `[0.911, −0.47, 0.51]`（和≈0.95 rad≈54°）；
2. **底盘漂移**——ComTask 每 tick 把目标重新对准当前底盘，再靠**平移底盘**去抵消手臂质量的 CoM 残差（底盘雅可比≈单位阵，最"便宜"），导致无输入时底盘自己爬行约 +0.33 m。

**现设计**（两个自定义任务，都**与底盘无关**→无质量数据、不会引起底盘漂移）：

- **`ChestOverAnkleTask`（主平衡，代价 50）**：误差是 2 维向量 `(chest_xy − ankle_xy)`，其中 `chest = 0.5·(p_头关节 + p_髋关节)`，雅可比取 `pin.getFrameJacobian(..., LOCAL_WORLD_ALIGNED)[:3]` 的 xy 行。帧名：`CHEST_HEAD_FRAME='astribot_head_joint_1'`、`CHEST_HIP_FRAME='astribot_torso_joint_3'`、`CHEST_ANKLE_FRAME='astribot_torso_joint_1'`。
  - **判别力**：好的下蹲 `[0.56,−1.15,0.65]` 测得胸-踝水平偏移 0.1 cm，而前倾姿 `[0.911,−0.47,0.51]` 为 65.7 cm——直接罚这个偏移就能锐利地阻止前倾。
  - 它与"角度和=0"**不冗余**：`[1,−2,1]` 这种和为 0 的姿态仍有 0.8 cm 偏移。
  - **经真机测试后，被用户定为主平衡任务。**

- **`TrunkUprightTask`（软次级，代价 0.5）**：标量误差 `e = Σθ_lean`（前倾脊柱三关节角度和），常数雅可比（3 个前倾 tangent DOF 上为 1 的行）。因为三个关节轴平行，躯干前倾俯仰 = θ₁+θ₂+θ₃。只需**大致为 0**即可，仅整理前倾冗余趋向自然直立。

### 3.2 前倾脊柱与"手风琴折叠"

`torso_joint_1/2/3` 不是单个"腰"，而是这台轮式机器人的髋/膝等价物——**三个平行的矢状面俯仰铰链**，堆叠在 z=0.217/0.597/0.987 m（踝/膝/髋）。`torso_joint_4` 是纯腰部偏航（不改变 CoM/高度，安全）。

**关键点**：因为轴平行，真正的下蹲需要三关节做**大幅的单独运动**（如膝≈−1.15 rad）而其**和保持 0**（"手风琴折叠"，降低身体但躯干保持竖直）。所以：
- `damping_cost_lean = 0.1`（曾是 40）——高阻尼会**阻断**下蹲需要的折叠（实测：40 时深蹲跟踪误差 36–58 mm，0.1 时 < 3 mm）；
- 防前倾的职责从"重罚每个前倾关节"移到了 `chest_over_ankle_cost` + `trunk_upright_cost`，两者都不惩罚折叠。

### 3.3 一个正确的 trade-off（非 bug）

因为 `ChestOverAnkleTask` 用到 `head_joint_1`，当**目标位姿本身不平衡**时，它会与头相机 FrameTracking 竞争——这是**正确的**（平衡 vs 指向的权衡）。所以 selfcheck 测试必须用**平衡的目标**（前倾脊柱=0，运动放在腰部偏航），否则头部跟踪会正确地退化约 6 mm。

### 3.4 EgoPoser 神经躯干先验（可选，默认关闭）

**动机**：QP 只跟踪头 + 双手三个末端，躯干/腰只由 `PostureTask`（拉向 home 直立）和平衡任务约束——所以机器人躯干姿态是"够用但不够拟人"的。[EgoPoser](https://github.com/eth-siplab/EgoPoser)（Jiang et al., ECCV 2024）能**仅凭头 + 双手位姿**幻想出人体全身姿态，我们提取其躯干先验作为 QP 的动态参考姿态，提升遥操作的自然度/拟人性。

**网络与 I/O**（`avp_teleop_upper_body/egoposer/`，`torch` 惰性导入，仅启用时才加载）：
- `AvatarNet`：3 层 `TransformerEncoder` + SlowFast 融合，embed 256，8 头。**从零重写**（上游仓库无 LICENSE），但 `state_dict` 键与上游 **48/48 完全一致**，官方权重可 `strict=True` 直接加载。
- 输入：80 帧 @60Hz 滑动窗口，每帧 54 维 = 头/左/右三传感器的 [全局 6D 旋转 + 6D 旋转速度 + 3D 位置 + 3D 位置增量]（`forward` 内再追加 6 维时序增量 → 60）。`FeatureWindow`（`feature_builder.py`）维护环形缓冲，暖机期用首帧补齐。坐标系 Z-up、米，AMASS/SMPL 约定，关节 15/20/21 = 头/左腕/右腕。
- 输出：`root_orient`（6D 骨盆）+ `pose_body`（126 = 21 个 SMPL 关节 × 6D **局部**旋转）。
- 单次推理 CPU **~3 ms（p95 3.3 ms）**，远在 60Hz（16.7 ms）预算内。

**重定向（SMPL 脊柱 → 机器人躯干）**（`estimator.py::_retarget`）：
- 躯干先验**只需前向推理，不需要 SMPL body model / betas / `human_body_prior`**——因为它只用 `pose_body` 的**局部**关节旋转（body model 只用于算关节 3D 位置，我们不需要）。这也天然继承了 EgoPoser 的"全局运动解耦"：先验与操作者全局朝向/位置无关。
- 取 spine1/2/3（SMPL 关节 3/6/9）局部旋转，复合成胸-relative-骨盆的旋转 `R_chest = R_s1·R_s2·R_s3`，分解内旋欧拉角：**flexion（绕 X）→ 机器人躯干俯仰**（映到前倾脊柱角度和 θ₁+θ₂+θ₃），**twist（绕 Y）→ 腰部偏航**（torso_joint_4）。
- SMPL 三关节分布式屈曲 ≠ 机器人三平行铰链，故用 `pitch_gain`/`yaw_gain`（含符号）缩放 + `max_pitch`/`max_yaw` 钳位，逐 tick EMA 平滑。

**注入 QP（`NeuralPostureTask`，`whole_body_ik.py`）**：
- 误差为 2 维：`[Σθ_lean − pitch_target, θ_waist − yaw_target]`，常数雅可比，**与底盘无关**（不会引起底盘漂移）。
- 代价**刻意压低（0.8）**，远低于平衡（50）和末端跟踪（10/3）——这就是"**用深度学习提升拟人性，用 QP 数学兜底安全**"的闭环：即便网络偶尔输出激进/失稳姿态，`ChestOverAnkleTask` 也以约 60× 的权重把它压住。
- 目标 `(0, 0)` 恰为"直立"，与 `TrunkUprightTask` 兼容；`neural_posture_cost=0` 时任务**根本不构建**，默认遥操作路径与之前**逐字节一致**、不触碰 torch。

**用法**：
```bash
# 一次性下载权重到 egoposer/model_zoo/（或手动从 Google Drive 取，见 WEIGHTS_DRIVE_URL）
python -c "from avp_teleop_upper_body.egoposer import EgoPoserEstimator; \
           EgoPoserEstimator.download_weights('avp_teleop_upper_body/egoposer/model_zoo')"
# 启用先验（权重在 model_zoo/egoposer.pth 时 config 自动设为默认路径，故 --egoposer 单独即可）
python -m avp_teleop_upper_body.sim_teleop --egoposer
# 可视化先验：MuJoCo 里画出 EgoPoser 幻想的 SMPL 骨架（青色 capsule 脊柱 + 橙色关节球），
# 竖直锚定在机器人髋部(torso_joint_3)随腰偏航一起转；默认灰色简化渲染 + 机器人半透明
python -m avp_teleop_upper_body.sim_teleop --visualize-prior
python -m avp_teleop_upper_body.sim_teleop --visualize-prior --body-alpha 0.2  # 更透
python -m avp_teleop_upper_body.sim_teleop --rich-render                       # 恢复完整贴图/阴影
```
官方 5 个权重（`egoposer/handtracking/30fps/*_large`）均以 `strict=True` 加载进重写网络（键 48/48 一致，self-check 已验证）。`--visualize-prior` 的骨架由 `pose_body` 局部旋转经名义 SMPL 骨长做正运动学（`estimator._skeleton`）得到，**仅供可视化**、不参与重定向数学。**锚定要点**：SMPL 骨架是世界系构建的（+Z 向上、前倾朝机器人 −Y 面），故必须锚在**竖直、随腰偏航对齐**的框架里、锚点取**髋关节 `torso_joint_3`**（`torso_joint_1/2/3` = 踝/膝/髋，髋≈人体骨盆）；早先误用踝关节 + 躯干 link 的 body 姿态（其 xmat 旋转了 90°，局部 +Z 指向世界 +X）导致脊柱**水平往前方喷出**的 bug。渲染侧默认剥离贴图/阴影/天空盒并压平成灰（省开销），`--body-alpha` 调机器人透明度，`--rich-render` 关闭简化渲染。
若 torch 或权重缺失，估计器自报 unavailable，先验静默关闭，遥操作不受影响。

---

## 四、验证状态

离线 `selfcheck` **12/12 通过**。`check_balance` 从三方面断言胸-踝偏移（主平衡）：
- (a) 普通伸手时胸部离踝 < 8 cm（≈0）；
- (b) 真下蹲折叠约 1.38 rad 时跟踪 < 5 mm 且胸部仍≈0 离踝；
- (c) **无输入回归守卫**：1500 tick 固定目标 → 底盘漂移 0.0 cm（守护 base-drift bug）。

`check_egoposer_prior` 断言 EgoPoser 先验的四条性质：
- (a) **关闭=无操作**：`neural_posture_cost=0` 时解与非神经管线**逐字节一致**（回归守卫）；
- (b) **偏置**：适度先验把躯干俯仰拉向目标（−0.01→0.20 rad）而手部跟踪仍 <2 cm；
- (c) **安全压制**：激进先验（pitch 1.2）被**衰减到不足一半**（→0.56 rad）且胸部仍 0.0 cm 在踝上方（平衡压过先验）；
- (d) **管线**：特征窗口 (1,80,54) 有限、6D 旋转往返、torch 在场时随机权重网络端到端跑通。

运行：`conda activate AVP` → `python -m avp_teleop_upper_body.sim_teleop`（另开 `avp_publisher --avp-ip <IP>`），按 `c` 标定头 + 双手，`space` 暂停。启用 EgoPoser 先验加 `--egoposer`（见 §3.4）。
