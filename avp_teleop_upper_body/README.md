# AVP → MuJoCo 上半身整体遥操 (avp_teleop_upper_body)

在双臂遥操的基础上，**加入 Apple Vision Pro 头部 6DoF 数据**，用一个**合并求解器**同时驱动机器人的
**底盘 + 躯干 + 脖子 + 左右双臂**，让整个身体随操作者协同移动（真正的**全身重定向**）。

```
Apple Vision Pro ──UDP──> 合并整体 IK (1 个 23-DOF Pink 求解器) ──> MuJoCo 仿真
  头 + 左手 + 右手            whole_body_ik.py                       sim_teleop.py
```

机器人复现的动作：

1. **头部跟随**：头戴显示器的位姿驱动机器人头部相机帧（脖子 2-DOF + 躯干 4-DOF 一起解）。
2. **双臂跟随**：左右工具末端跟随你左右手腕——并且**手臂会自动补偿躯干的运动**（见下）。
3. **底盘协同**：当目标超出手臂+躯干的可达范围时，**移动底盘**（平移 x/y + 偏航）帮忙够到——但**优先动手臂和腰、其次才动底盘**（按关节权重，见下）。
4. **手指开合**：每根手指的弯曲度映射到对应的 BrainCo 机械手关节。

> **关于"下肢 / 髋膝关节"**：这台 Astribot S1 **没有独立的髋/膝/踝关节**，它的"下肢"由两部分构成，二者共同扮演腿的角色：
> - **3-DOF 轮式移动底盘**（`chassis_x`/`_y` 平移 + `chassis_zrot` 偏航）——负责**水平移动**；
> - **`torso_joint_1/2/3` 前倾脊柱**（经运动学扫描确认：这三个关节是**矢状面的前倾/下蹲机构**，相当于髋/膝——`+0.3 rad` 前倾会把整机重心水平移出 ~14 / 7 / 2 cm，是**过度补偿导致前倾/后仰失稳**的根源；`torso_joint_4` 才是纯粹的**腰部偏航**，不影响重心/高度，可自由动）。
>
> 所以用户要求的三级优先级"**手臂+腰 › 底盘 › 下肢（髋/膝）**"在本机上**完整成立**，落为：**{手臂 + 腰偏航 + 脖子} › {底盘} › {前倾脊柱 torso_joint_1/2/3}**。前倾脊柱被放到**最低优先级**（阻尼最大），只有需要真正改变**身体高度**（如下蹲够低处）时才大幅动它——因为无腿机器人只能靠这条脊柱升降身体。见下方"全身重定向与关节优先级"。

> 这是 [`avp_teleop`](../avp_teleop/) 双臂版的**全身扩展**。双臂版（躯干锁定、两臂各自独立求解）**完全不动**，仍可单独使用；本包是一个**独立的新包**。

---

## 为什么是"合并求解器"（核心设计）

机器人 `astribot_torso_link_4` 是**双臂和头的共同根**，而 `torso_joint_1..4` 又都长在**底盘**上：

```
chassis(x,y,yaw) → torso_joint_1..4 → torso_link_4 → { head_joint_1,2 ; 左臂×7 ; 右臂×7 }
```

一旦**躯干/底盘参与运动**，它们就会带动两条手臂的基座——双臂版"躯干锁中立、两臂独立"的前提不再成立。
因此本包把 **底盘(3) + torso(4) + neck(2) + 左臂(7) + 右臂(7) = 23 个自由度**放进**同一个** Pinocchio + Pink
配置里，一个 QP 同时求解这些加权任务：

| 任务                    | 帧                                 | 作用                                                                                                      |
| ----------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `FrameTask`           | `astribot_head_camera_base_link` | 跟踪 AVP 头部 6DoF 目标（主要驱动 torso+neck）                                                            |
| `FrameTask`           | `astribot_arm_left_tool_link`    | 跟踪左手目标                                                                                              |
| `FrameTask`           | `astribot_arm_right_tool_link`   | 跟踪右手目标                                                                                              |
| `PostureTask`         | →`BODY_HOME`                    | 解 23-DOF 冗余：躯干偏向直立、肘腕自然、**底盘回到原点**（稳定性关键）                              |
| `ComTask`（水平）     | 整机重心 → 底盘上方             | **平衡约束**：只罚重心的**水平**偏移（防前倾/后仰失稳），**竖直方向权重=0**（下蹲升降高度不受限）      |
| `DampingTask`         | 全关节（**逐关节权重**）     | 软惩罚`‖v‖`（关节速度）→ 抑制零空间速度 + **编码全身三级运动优先级**（前倾脊柱 > 底盘 > 上半身，见下） |
| `LowAccelerationTask` | 全关节                             | 软惩罚`‖v−v_prev‖`（帧间加速度）→ 降低急动(jerk)、更顺滑                                            |

因为三个末端任务**共享同一配置**：当头部目标让躯干前倾时，两个手臂任务会**在同一次求解里**把手臂解到
"手仍在各自世界目标"的构型；当目标够不到时，**底盘平移/转向**来延伸可达范围——这就是**自动补偿 + 全身协同**，
无需任何手写坐标修正。这正是 VisionProTeleop `examples/11_diffik_aloha.py` 的合并双臂模式（两臂 FrameTask +
PostureTask），本包在它之上加了一个头部任务 + 一个移动底盘，并用 Pink（而非 mink）实现，与项目既有 IP 栈一致。

帧一致性沿用既有做法：模型由**展平后的同一 MJCF** 构建，三个末端帧与 MuJoCo 逐构型吻合到 <1e-4 m，
解出的关节角**零坐标变换**直接写入仿真（详见 [whole_body_ik.py](whole_body_ik.py) 顶部说明）。

> **底盘的复合关节**：Pinocchio 的 MJCF 解析器会把底盘 3 个关节**合并**成**一个** 3-DOF 复合关节
> `Composite_astribot_chassis_x`（内部 DOF 顺序 x, y, yaw，与 MuJoCo 一致）。所以求解器内部用**约简模型的关节名**
> （底盘=复合关节，即 `IK_KEEP_JOINTS`），而对外返回的 23 维指令向量按 `BODY_JOINTS`（底盘拆成 x/y/yaw 三项）排列——
> 两者逐 DOF 对齐（自检里 round-trip 实测 0.00 mm）。多 DOF 关节被自动展开成各自的切空间 DOF，下游无需特判。

---

## 依赖

**无新增依赖**。沿用 AVP 环境里双臂版已装好的 `pinocchio` + `pink` + `qpsolvers`（外加 `mujoco`/`numpy`/
`scipy`/`avp_stream`）。若尚未安装 Pink：

```bash
conda activate AVP
conda install -c conda-forge pinocchio pink qpsolvers quadprog -y
python -c "import pinocchio, pink, qpsolvers; print('ok')"
```

---

## 怎么运行

三步，都用 `AVP` 环境，工作目录 `/Users/apple/vscodeProject/AVP`。

### 第一步（推荐）：离线自检（无需 AVP 硬件）

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
python -m avp_teleop_upper_body.selfcheck
```

应当看到 `11/11 checks passed.`（含**合并求解一致性**、**自动补偿**、**全身底盘优先级**、**重心平衡/前倾脊柱优先级**、**位姿滤波**、**速度/加速度硬约束**与**软平滑任务**等关键校验）。

### 第二步：启动发布端（戴上 Vision Pro）

打开 Tracking Streamer App，记下 IP / 房间码，然后：

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
python -m avp_teleop_upper_body.avp_publisher --avp-ip 10.200.177.142
```

每拍发送 **头 + 左手 + 右手** 三个 datagram（同端口，靠 magic 区分）。`--no-head` 可只发双手。
看到 `OK: sent=...` 持续刷新即正常。

### 第三步：启动仿真端

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
python -m avp_teleop_upper_body.sim_teleop
```

弹出 MuJoCo 窗口后，**摆好一个舒服的初始姿态，在窗口里按 `c` 标定**（一次同时锚定 头+左手+右手），
机器人就开始整体跟随。

仿真窗口快捷键：

| 键       | 作用                                                                                       |
| -------- | ------------------------------------------------------------------------------------------ |
| `c`    | （重新）标定：把当前 头/左手/右手 三者锚定到机器人对应帧的当前位姿。任何时候飘了都可重按。 |
| `空格` | 暂停 / 继续遥操（物理仿真继续跑）                                                          |

常用参数：

```bash
# 手臂默认只跟位置；位置调通后再开手腕姿态(6-DOF 手臂)
python -m avp_teleop_upper_body.sim_teleop --orientation

# 头部默认跟 6DoF(含注视朝向)；只想跟头的位置、不跟朝向：
python -m avp_teleop_upper_body.sim_teleop --head-no-orientation

# 缩放(保持正值，方向用 AVP_TO_ROBOT_R 调，不要用负 scale)
python -m avp_teleop_upper_body.sim_teleop --position-scale 1.2 --head-position-scale 0.8

# 位姿平滑(抗 AVP 抖动)：alpha∈(0,1]，越小越平滑但越滞后，1.0=关闭
python -m avp_teleop_upper_body.sim_teleop --alpha-translation 0.3 --alpha-rotation 0.4
python -m avp_teleop_upper_body.sim_teleop --no-filter        # 完全关闭滤波

# 用自己编辑保存的姿态作为初始/静止姿态(见下一节"姿态编辑器")
python -m avp_teleop_upper_body.sim_teleop --init-pose my_pose

# 启用 EgoPoser 神经躯干先验(可选，需 torch + 权重；见"EgoPoser 神经躯干先验")
# 权重放在 egoposer/model_zoo/egoposer.pth 时，--egoposer 即可自动找到，无需再指定路径
python -m avp_teleop_upper_body.sim_teleop --egoposer
# 或显式指定权重
python -m avp_teleop_upper_body.sim_teleop --egoposer \
       --egoposer-weights avp_teleop_upper_body/egoposer/model_zoo/egoposer.pth
# 可视化先验：在 MuJoCo 里用线框画出 EgoPoser 幻想的 SMPL 骨盆/脊柱/头部骨架(隐含 --egoposer)
# 默认灰色简化渲染 + 机器人半透明(alpha 0.35)，便于观察骨架
python -m avp_teleop_upper_body.sim_teleop --visualize-prior
# 调机器人透明度 / 保留完整贴图阴影
python -m avp_teleop_upper_body.sim_teleop --visualize-prior --body-alpha 0.2
python -m avp_teleop_upper_body.sim_teleop --rich-render   # 关闭简化渲染，恢复原始外观
```

---

## EgoPoser 神经躯干先验（可选，默认关闭）

QP 只跟踪头 + 双手，躯干姿态"够用但不够拟人"。[EgoPoser](https://github.com/eth-siplab/EgoPoser)（ECCV 2024）仅凭头 + 双手位姿就能幻想出人体全身姿态；我们提取其**躯干俯仰 + 腰部偏航**先验，作为一个**低代价**软任务 `NeuralPostureTask`（`neural_posture_cost=0.8`）注入 QP。

**核心：用深度学习提升拟人性，用 QP 数学兜底安全。** 先验代价（0.8）远低于平衡（`ChestOverAnkleTask`=50）和末端跟踪（10/3），所以即便网络输出激进姿态也会被约 60× 权重的平衡任务压住——self-check 实测：目标 pitch=1.2 被衰减到 0.56 rad，胸部始终在踝上方。`neural_posture_cost=0`（默认）时任务**根本不构建**，遥操作路径与之前逐字节一致、不触碰 torch。

- 子包 `avp_teleop_upper_body/egoposer/`：`network.py`（从零重写 `AvatarNet`，`state_dict` 键与官方 48/48 一致，权重可直接加载）、`feature_builder.py`（80 帧 @60Hz 窗口 → 54 维输入）、`estimator.py`（推理 + SMPL 脊柱→机器人躯干重定向）、`rotations.py`（纯 numpy 6D 旋转）。
- 只需前向推理，**不需要 SMPL body model / betas**（只用 `pose_body` 的局部旋转）。CPU 推理 ~3 ms，60Hz 绰绰有余。
- 安装：`pip install "torch>=2.2,<2.6"`（CPU 版即可）。权重从官方 Google Drive 下载并放到 `egoposer/model_zoo/`：
  ```bash
  python -c "from avp_teleop_upper_body.egoposer import EgoPoserEstimator; \
             EgoPoserEstimator.download_weights('avp_teleop_upper_body/egoposer/model_zoo')"
  ```
  权重位于 `egoposer/model_zoo/egoposer.pth` 时，`config.py` 会**自动**把它设为默认 `weights_path`，故 `--egoposer` 单独即可用，无需再传 `--egoposer-weights`。5 个官方权重（`egoposer/handtracking/30fps/*_large`）都能以 `strict=True` 直接加载进重写的网络（self-check 已验证）。
- **`--visualize-prior`**：在 MuJoCo 渲染循环里，用**青色线框（capsule）+ 橙色关节球**画出 EgoPoser 幻想的 SMPL 矢状骨架（骨盆→spine1/2/3→颈→头），**竖直锚定在机器人髋部**（`astribot_torso_joint_3`；注意 `torso_joint_1/2/3` = 踝/膝/髋，髋才对应人体骨盆），并随机器人腰部偏航一起转，便于你**直观确认**先验是否合理。骨架由 `pose_body` 局部旋转经名义骨长做正运动学得到（仅用于可视化，不参与重定向数学）。隐含 `--egoposer`。
- **简化渲染（默认开启）**：为降低渲染开销并便于观察，默认把场景**压平成灰色**（剥离贴图、关阴影、关天空盒/反射/雾）。`--visualize-prior` 时机器人本体默认**半透明**（`--body-alpha 0.35`）好让骨架透出来；用 `--body-alpha` 调透明度，用 `--rich-render` 恢复完整贴图/阴影外观。
- torch 或权重缺失时估计器自报 unavailable，先验静默关闭，遥操作不受影响。
- 原理详解见 [retargetting.md](../progress%20report/retargetting.md) §3.4。

---

## 位姿滤波（抗 AVP 抖动）

AVP 的头/腕位姿有跟踪抖动、偶发丢帧，直接喂给求解器会让机器人**发抖**。所以在目标位姿进求解器**之前**，
对头/左/右**各自**做一阶指数滤波（[pose_filter.py](pose_filter.py)）：

- **平移**：标准 3D 向量 EMA —— `filtered = α·新 + (1-α)·旧`。
- **旋转**：SO(3) 上的**测地线插值 = 四元数 SLERP** —— `filtered = slerp(旧, 新, α)`，用
  `pinocchio.log3 / exp3` 实现。**不**对旋转矩阵或欧拉角做线性加权（那样结果不再是合法旋转，会被拉歪、
  甚至万向锁）。这相当于 `pinocchio.interpolate` 在 SE(3) 上的旋转分量。

| 系数                  | 作用            | 取值                                              |
| --------------------- | --------------- | ------------------------------------------------- |
| `alpha_translation` | 平移 EMA 系数   | `(0,1]`，默认 0.5；越小越平滑、越滞后；1.0=关闭 |
| `alpha_rotation`    | 旋转 SLERP 系数 | `(0,1]`，默认 0.5；同上                         |

平移与旋转**各自独立**的系数（这正是不用单一 SE(3) `alpha` 的原因——可分别调位置/姿态的平滑度）。
配置在 [config.py](config.py) 的 `PoseFilterConfig`，也可用上面的 `--alpha-translation/--alpha-rotation/--no-filter`
临时覆盖。每次按 `c` 重标定都会**重置滤波器**，不会在重锚的跳变上做平滑。

---

## 速度 / 加速度硬约束（QP 内物理限幅）

位姿滤波平滑的是**输入目标**；这里限的是**输出关节运动**本身。求解器用
[pink.limits](whole_body_ik.py) 把**速度上限**和**加速度上限**作为**硬不等式约束**放进 QP：

- `VelocityLimit`：`|Δq| ≤ dt·v_max`，即每个关节的速度不超过 `v_max`。
- `AccelerationLimit`：`|v − v_prev| ≤ dt·a_max`（带 Flacco "刹车距离"项，保证不会冲过关节限位），
  即速度变化率不超过 `a_max`。
- `ConfigurationLimit`：关节位置限位（Pink 默认就有，这里一并显式传入）。

**每个控制拍只跑一次 `solve_ik`**，所以上面三条就是机器人**真实的每拍速度/加速度上限**。这取代了
旧的"先解到底、再 `np.clip` 截断步长"做法——限幅现在在**优化内部**完成，求解器会在"限速"和"跟踪精度"
之间自行权衡，而不是事后把解砍掉（砍掉会破坏多任务的协同）。

> teleop MJCF **没有**声明速度限位，所以 `v_max` 来自 [config.py](config.py) 的 `WholeBodyIKConfig`
> （torso/neck 默认 1.8 rad/s、手臂 3.0 rad/s；加速度 60 / 100 rad/s²）。手臂的 `v_max=3.0` 恰好等于旧的
> `arm_max_step=0.05 rad/拍 @60Hz`，所以默认行为与之前的限速一致，但更平滑、且物理正确。
>
> **底盘**是移动底座，x/y 是**平移**（单位米），所以它们的上限是 **m/s、m/s²**（默认 0.6 m/s、4 m/s²），
> 而偏航 `zrot` 是**旋转**（1.2 rad/s、20 rad/s²）。`pink.limits` 逐切空间 DOF 施加上限，不假设"全是旋转关节"，
> 所以公制与角度上限可以在同一条约束里共存。

每次按 `c` 重标定会调用 `ik.reset()`，**从静止重新加速**，不把旧速度带过重锚跳变。极少数情况下
（贴着关节限位时速度盒与加速度刹车盒可能无交集）会自动**降级**（丢掉该拍的加速度约束，必要时保持不动），
不会让遥操主循环崩溃。想整体关掉硬约束做对比：把 `WholeBodyIKConfig.enforce_limits` 设为 `False`
（退化为无约束求解 + 旧式 clip）。

---

## 软平滑任务（QP 内低权重平滑）

硬约束是**不可逾越的物理上限**；软平滑任务则在**上限以内**进一步塑形运动，让它"平顺"而不是"贴着上限猛冲"。求解器额外挂了两个 [pink.tasks](whole_body_ik.py) 的低权重任务：

- `DampingTask`：惩罚 `‖v‖`（关节速度）。像"阻尼/摩擦"一样把多余速度泄掉——尤其是 23-DOF 链跟踪 ≤18 维任务时多出来的**零空间速度**（避免内部自由度乱漂），并在无目标驱动时让关节收敛到静止。**此外它的逐关节权重还编码全身运动优先级**（底盘权重远高于上半身，见"全身重定向与底盘优先级"节）。
- `LowAccelerationTask`：惩罚 `‖v − v_prev‖`（帧间加速度）。它**有状态**：每拍解完把该拍速度喂给它（`set_last_integration`），下一拍即惩罚速度变化 → 降低急动(jerk)、抗目标抖动。Pink 文档明确建议它**与 `DampingTask` 搭配**（单用它不耗散能量、会自激振荡），所以两个一起加。

> 两个 cost 都**远小于**末端跟踪 cost（手臂 10 / 头 3），默认各 `0.1`（≈100× 低于手臂跟踪）。**这是有意的**：默认权重对"手要去哪"的主运动几乎无影响（自检实测默认权重对大幅 slew 的峰值速度影响 <0.1%），它们的真正作用是**零空间阻尼 + 降 jerk**。想要更明显的平滑就**调大** `damping_cost`/`low_accel_cost`，代价是略增滞后/稳态偏移；设为 `0` 即关掉对应任务做 A/B 对比。

与硬约束的关系：硬 `AccelerationLimit` 是**加速度硬顶**，软 `LowAccelerationTask` 是**加速度软惩罚**，二者互补（一个封顶、一个在顶以内塑形）。两者都"记得上一拍速度"，所以按 `c` 重标定时 `ik.reset()` **同时**清掉它们的状态，从静止重新加减速。

---

## 全身重定向与关节优先级（逐关节权重 + 重心平衡）

本包把**移动底盘**和**整条躯干**都解锁进同一个求解器，成为真正的全身 IK。用户要求"**优先动手臂和腰，其次动底盘，最后动下肢（髋/膝）**"——这台 S1 的"下肢"由**底盘**（水平移动）+ **前倾脊柱 `torso_joint_1/2/3`**（升降/前倾，相当于髋膝）共同承担，于是完整落为**三级优先级**。实现机制正是用户要的"**给不同关节分配不同权重**"：

- **优先级 = `DampingTask` 的逐关节速度权重**。cost 可以是一个**逐 DOF 向量**（Pink 内部按 `diag(cost²)` 进 QP 的 Hessian），cost 越大 → 该关节速度越"贵" → QP 越不愿动它 → **越晚被调用**。三级（由先到后 / 由便宜到贵）：
  - **手臂 + 腰偏航(`torso_joint_4`) + 脖子**：**小** cost（`damping_cost=0.1`）→ 先动、且兼作平滑阻尼；
  - **底盘 x/y/yaw**：**中** cost（`damping_cost_chassis=20`）→ 上半身够不到时才平移/转向延伸；
  - **前倾脊柱 `torso_joint_1/2/3`**：**最大** cost（`damping_cost_lean=40`）→ **最后动**。因为前倾会把重心移出轮子基座（失稳/前倾后仰），所以要让 QP 先用手臂、再转腰、再挪底盘，实在够不到、且是**持续**的目标才前倾。
- **为什么底盘/前倾要高权重？** 底盘平移对末端的雅可比≈单位（挪底盘 1 cm，手就走 1 cm），是"移动手"最省力的方式；前倾脊柱同理还更省。若不重罚，QP 会**优先挪底盘/前倾而不是伸手**（这正是原来"过度补偿前倾失稳"的原因）。加大阻尼后：正常伸手底盘/躯干几乎不动，只有目标明显够不到才依次动它们。
- **`ComTask` 重心平衡（第二道防线）**：一个重心任务把整机**水平**重心钉在底盘正上方，直接惩罚"重心偏出底盘"的前倾/后仰；**竖直方向权重设为 0**，所以机器人**仍可自由升降重心（下蹲）**。它随底盘一起平移（底盘走到哪，平衡目标跟到哪），所以**只罚前倾、不罚走底盘**。

**关键权衡**（自检 `check_balance` 实测）：普通前伸/前下伸手，躯干只前倾 ~2.6°、重心离底盘仅 ~4.5 cm（**不再过度补偿前倾**）；而**真正的下蹲**（头和手一起下降）仍会前倾 ~15.6° 去改变身体高度——因为无腿机器人只能靠这条脊柱升降。**注意 `damping_cost_lean` 必须有限**，否则脊柱被冻住就再也蹲不下去了。

调 `damping_cost_chassis` / `damping_cost_lean` / `com_cost`（[config.py](config.py) 的 `WholeBodyIKConfig`）：

| 参数                             | 默认 | 效果                                               |
| ------------------------------ | ---- | -------------------------------------------------- |
| `damping_cost`（上半身）        | 0.1  | 手臂/腰偏航/脖子最先动。                            |
| `damping_cost_chassis`（底盘）  | 20   | 其次动；调大=底盘更懒，调小=更跟手，=`damping_cost` 即平权。 |
| `damping_cost_lean`（前倾脊柱） | 40   | 最后动；调大=躯干更"直挺"、更难蹲，调小=更易前倾/下蹲。 |
| `com_cost`（水平重心）          | 3.0  | 平衡强度；调大=更"端正稳定"（前倾更受限），0=关闭平衡任务。 |
| `com_cost_vertical`（竖直重心） | 0.0  | 保持 0 → 下蹲升降不受限；>0 会开始约束身体高度。   |

> 底盘位置**无限位**（可无限行走）；求解器靠 `PostureTask`（home=原点）在底盘/躯干不再被需要时把它们**慢慢拉回原点**，避免长期漂移。
> 三级优先级是**数据驱动**的（见 `WholeBodyIKConfig.damping_costs()` / `_assemble()`）：将来换成有独立髋/膝的**有腿**机型，只需把这些关节归入对应的权重档即可，代码结构不变。

---

## 调整 / 保存初始姿态（姿态编辑器）

`pose_editor.py` 让你在 MuJoCo 窗口里**手动摆好身体关节**（躯干 4 + 脖子 2 + 左右臂各 7；底盘 3 项固定在原点、不参与编辑），
**实时看到效果**，满意后**保存**；保存的姿态可被 `sim_teleop --init-pose` 选为遥操的**初始/静止姿态**。

> 机器人是固定底座、每个 body 关节都有位置执行器，所以编辑器**纯运动学**摆位（只 `mj_forward`，不跑物理）——
> 你摆成什么样、保存的就是什么样，零漂移、不会倒。

### 启动编辑器

```bash
conda activate AVP
cd /Users/apple/vscodeProject/AVP
# 从 home 开始，保存到 poses/my_pose.json
python -m avp_teleop_upper_body.pose_editor --outname my_pose
# 打开 my_pose 继续微调，原地存回 my_pose
python -m avp_teleop_upper_body.pose_editor --inname my_pose
# 打开 my_pose，另存为 my_pose1（不覆盖原文件）
python -m avp_teleop_upper_body.pose_editor --inname my_pose --outname my_pose1
```

> `--inname` 指定**打开/起始**的姿态(不给则从 home 开始);`--outname` 指定**保存**目标(不给则
> 与 `--inname` 相同=原地编辑,两者都不给则存到 `custom_init`)。旧名 `--from`/`--name` 仍作别名可用。

> macOS 上启动 MuJoCo 窗口若提示需要 `mjpython`，把命令里的 `python` 换成 `mjpython` 即可
> （`sim_teleop` 同理，两者都用同一个被动 viewer）。

### 编辑器按键（焦点在 MuJoCo 窗口上）

| 键               | 作用                                                                            |
| ---------------- | ------------------------------------------------------------------------------- |
| `↑` / `↓`  | 选择上一个 / 下一个关节（在 body 关节间循环）                             |
| `←` / `→`  | 把当前关节角**减小 / 增大**一个步长（自动限位钳制）                       |
| `[` / `]`    | 步长**减半 / 加倍**（默认 0.05 rad）                                      |
| `0`            | 当前关节复位到 home                                                             |
| `M`            | 把左臂**镜像**到右臂(左右对称,关节 1/3/5/7 取反、2/4/6 相同),只调左手即可 |
| `R`            | **全部**关节复位到 home                                                   |
| `S`            | **保存**当前姿态到 `--outname` 指定的文件                               |
| `P`            | 在终端打印当前完整姿态                                                          |
| `H` / `空格` | 打印帮助 / 打印当前选中关节与角度                                               |

每次调整都会在终端打印形如 `>> [12] L-arm astribot_arm_left_joint_7 = +0.340 rad ...` 的状态行。
**按 `S` 保存后再关窗口**——关窗口本身不会保存。姿态文件存到 `avp_teleop_upper_body/poses/<name>.json`
（按关节名记录、可手改）。

### 用保存的姿态作为初始姿态

```bash
python -m avp_teleop_upper_body.sim_teleop --init-pose my_pose
```

`--init-pose` 同时把该姿态用作：(1) 机器人**起始构型**；(2) 合并 IK 的 **PostureTask 静止目标**（即零空间
正则的"home"）。因此机器人从该姿态启动，且在冗余自由度上自然偏回该姿态——这正是"以保存姿态为初始姿态"。
不带 `--init-pose` 时仍用内置 `BODY_HOME`。

---

## 标定（与双臂版一致：相对/增量遥操）

按下 `c` 的瞬间，对 头/左手/右手 **各自**记录：你此刻的源位姿（AVP 头/腕）作为零点 + 机器人对应帧
（头相机 / 左右 tool）此刻位姿作为零点。之后每帧只看你**相对零点**移动了多少，叠加到机器人。所以站位、
姿态都无所谓，按 `c` 那一刻即新中心；飘了就重按 `c`。三者用同一套 `AVP_TO_ROBOT_R`（世界系 180° 绕 Z）。

---

## 录制 / 回放（不必每次都戴头套）

戴一次 Vision Pro 把输入录下来，之后就能反复离线跑重定向、调参、看渲染，不用再戴头套。有**两条相互解耦**的轨迹：

| 轨迹 | 目录 | 存什么 | 用途 |
| --- | --- | --- | --- |
| **AVP 原始输入** | `avp_trajectory/` | 每拍的头 4×4 + 双手腕 4×4 + 21×3 关键点 + pinch | 回放后**重新跑完整重定向+IK+渲染** |
| **重定向轨迹** | `retargetting_trajectory/` | 上者的超集：CLI 参数 + 原始输入 + 目标 + 23 关节角 + 手指 + 先验 + 骨架 | **纯回放（零计算）** + 离线分析 |

### 怎么开始 / 怎么结束录制（重要）

- **开始录制**：**不需要按任何键**。只要命令行带了 `--record-avp` 或 `--record-retarget`，程序加载完毕后**立即自动开始录制**，逐拍捕获（包括无数据/无效拍，保真时序）。
- **结束并保存**：**关闭 MuJoCo 窗口**，**或**在终端按 **`Ctrl+C`** —— 两者都会**停止并保存**。（再按一次 `Ctrl+C` 会强制退出、不保存。）
- **首尾裁剪**：保存时默认**丢掉首尾各 2 秒**（`--record-trim`，设 0 保留全部），用来切掉你手忙脚乱按键启动/结束的那段。剩余帧时间戳会重新归零。
- `space` 暂停的是**遥操作**；录制"原始 AVP 输入"在暂停时**仍继续**（录的是输入流，与机器人是否动无关）。

> 提示：录制时默认会在 MuJoCo 里画出**世界坐标系** + **头/左手/右手的原始位姿**（RGB 三色轴 = XYZ，黄/紫/青 = 头/左/右），方便你实时核对输入是否正常。`--no-input-frames` 关闭，`--show-input-frames` 可在非录制时也开。

### 四种工作流

```bash
# A. 戴 AVP，录原始输入（自动切首尾各 2 秒；结束用关窗口或 Ctrl+C）
python -m avp_teleop_upper_body.sim_teleop --record-avp clip1

# B. 用录好的输入跑重定向+渲染（不戴头套！首帧自动标定）
python -m avp_teleop_upper_body.sim_teleop --replay-avp clip1

# C. 录完整重定向轨迹（可戴 AVP，也可喂旧输入 clip1 再生成一条可分析轨迹）
python -m avp_teleop_upper_body.sim_teleop --replay-avp clip1 --record-retarget run1

# D. 纯回放（零计算，只重放动作 + 全部叠加层；可暂停 / 拖进度）
python -m avp_teleop_upper_body.replay_retarget run1
```

A/B/C 可自由组合。想保留完整录制不裁剪：加 `--record-trim 0`。

### 纯回放播放器按键（焦点在 MuJoCo 窗口）

| 键 | 作用 |
| --- | --- |
| `space` | 暂停 / 继续 |
| `.` / `,` | 暂停时逐帧前进 / 后退 |
| `]` / `[` | 快进 / 快退 1 秒 |
| `0` | 跳到开头 |
| `l` | 切换循环播放 |

回放是完整位姿快照逐帧重放，反向拖动零成本；终端打印文本进度条。

---

## 模块结构

```
avp_teleop_upper_body/
  config.py          # 底盘/torso(含前倾脊柱)/neck 关节名、HEAD_FRAME_BODY、CHASSIS_BASE_FRAME、23-DOF BODY_JOINTS 顺序 + BODY_HOME、合并 IK 权重 + 三级关节优先级 + 重心平衡
  transport.py       # HeadFrame(magic AVPE) + HeadFramePublisher + UpperBodySubscriber(单端口按 magic 分发)
  whole_body_ik.py   # WholeBodyIK：合并 Pink 求解器(头/左/右 FrameTask + ComTask 重心平衡 + PostureTask) + 速度/加速度硬约束 + 软平滑任务(Damping/LowAcceleration) + 逐关节三级优先级(前倾脊柱>底盘>上半身)
  avp_publisher.py   # 连 AVP，每拍发 头 + 左手 + 右手
  sim_teleop.py      # 合并求解器主循环：poll → 三目标 → solve(23-DOF) → 写 ctrl → 步进 + 渲染
  pose_filter.py     # 位姿滤波：平移 EMA + 旋转 SLERP(pinocchio)，sim_teleop 进求解前对三目标平滑
  pose_editor.py     # 交互式姿态编辑器：键盘摆 torso/neck/双臂(底盘钉在原点)、实时渲染、保存初始姿态
  pose_io.py         # 姿态存取(JSON，按关节名)；sim_teleop --init-pose / pose_editor 共用
  poses/             # 保存的姿态(<name>.json)
  trajectory_io.py   # 录制/回放：AVP 原始输入 + 完整重定向轨迹 两套 JSON schema；FileAvpSource(替换 UDP 订阅器) + 首尾裁剪 trim_frames
  replay_retarget.py # 纯回放播放器(零计算)：读重定向轨迹，逐帧灌关节+手指+复现叠加层，可暂停/拖进度
  avp_trajectory/    # 录下的 AVP 原始输入(<name>.json)
  retargetting_trajectory/  # 录下的完整重定向轨迹(<name>.json)
  selfcheck.py       # 离线自检(13 项)
  egoposer/          # EgoPoser 神经躯干先验(可选)：network/feature_builder/estimator/rotations
```

**复用 [`avp_teleop`](../avp_teleop/)（import，不复制不改动）**：`HandFrame`/`HandFramePublisher`（transport）、
`WristCalibration`/`wrist_to_tool_target`（frames）、`HandRetargeter`（hand_retarget）、`SimRobot`
（robot_interface）、`MJCF_PATH`/`ARM_JOINTS`/`TOOL_BODY`/`ARM_HOME`/`AVP_TO_ROBOT_R`/`_finger_specs`（config）。

> 同一个 `SimRobot` 实例驱动全部 23 个 body 关节（`arm_joint_names` = `BODY_JOINTS`，含底盘 3 个位置执行器）+ 双手全部手指；
> 头相机帧作为它的 `tool_body`（供头部标定读位姿）。底盘 x/y/yaw 与 torso(4)+neck(2) 的 position 执行器在 stock
> `astribot_s1_actuator_for_hand.xml` 里**本就存在**，故 **teleop MJCF 不需改动**。

---

## 调参指南（都在 [config.py](config.py) 的 `WholeBodyIKConfig`）

| 参数                                | 作用                                                | 调整方向                                            |
| ----------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| `arm_position_cost`               | 手臂末端跟随权重                                    | 默认 10（高于头部，保证抓取精度优先）               |
| `head_position_cost`              | 头部位置跟随权重                                    | 默认 3（中等，避免头部猛拽躯干）；想头更跟手→调大  |
| `head_orientation_cost`           | 头部朝向(注视)权重                                  | 仅`head_track_orientation=True` 用                |
| `arm_orientation_cost`            | 手腕姿态权重                                        | 仅`--orientation` 用                              |
| `posture_cost`                    | 偏向`BODY_HOME` 的零空间正则                      | 躯干乱晃/不自然 → 调大（更"端正"但跟随略松）       |
| `lm_damping`                      | LM 阻尼(抗奇异)                                     | 近奇异抖动 → 调大；跟随发钝 → 调小                |
| `torso_neck_max_velocity`         | torso/neck 速度上限(rad/s)                          | 躯干猛甩 → 调小（默认 1.8，比手臂更小）            |
| `arm_max_velocity`                | 手臂速度上限(rad/s)                                 | 默认 3.0（≈旧的 0.05 rad/拍 @60Hz）                |
| `chassis_max_linear_velocity`     | 底盘平移速度上限(**m/s**)                     | 默认 0.6；底盘冲太快 → 调小                        |
| `chassis_max_yaw_velocity`        | 底盘偏航速度上限(rad/s)                             | 默认 1.2                                            |
| `torso_neck_max_acceleration`     | torso/neck 加速度上限(rad/s²)                      | 起步/变向发冲 → 调小（默认 60，更平滑但更钝）      |
| `arm_max_acceleration`            | 手臂加速度上限(rad/s²)                             | 默认 100；调小=更平滑的加减速                       |
| `chassis_max_linear_acceleration` | 底盘平移加速度上限(**m/s²**)                 | 默认 4.0                                            |
| `chassis_max_yaw_acceleration`    | 底盘偏航加速度上限(rad/s²)                         | 默认 20.0                                           |
| `config_limit_gain`               | 远离关节限位的转向增益(0~1)                         | 默认 0.5                                            |
| `damping_cost`                    | **上半身**关节速度软惩罚`‖v‖`(零空间阻尼) | 默认 0.1；想更顺滑/抗漂→调大（略增滞后），0=关     |
| `damping_cost_chassis`            | **底盘**速度软惩罚 = 优先级第 2 档           | 默认 20（底盘"其次"动）；调大=底盘更懒，调小=更跟手 |
| `damping_cost_lean`               | **前倾脊柱**(torso_joint_1/2/3)速度软惩罚 = 优先级第 3 档（最后动） | 默认 40（"最后"动、防前倾失稳）；调大=躯干更直挺更难蹲，调小=更易前倾/下蹲 |
| `com_cost`                        | **水平重心**平衡软惩罚（防前倾/后仰失稳）    | 默认 3.0；调大=更端正稳定，0=关闭平衡                |
| `com_cost_vertical`               | **竖直重心**惩罚（身体高度）                 | 默认 0.0=下蹲升降自由；>0 才约束高度                 |
| `low_accel_cost`                  | 帧间加速度软惩罚`‖v−v_prev‖`(降 jerk)          | 默认 0.1；抖动明显→调大，0=关                      |
| `neural_posture_cost`             | **EgoPoser 躯干先验**软惩罚（躯干俯仰+腰偏航）| 默认 0（关闭）；`--egoposer` 时默认 0.8，调大=更拟人但更抢平衡余量。见"EgoPoser 神经躯干先验" |
| `head_position_scale`             | 头部位移放大                                        | 默认 1.0；上半身够不到 → 调大（保持正值）          |

---

## 我已经离线验证过的部分

`python -m avp_teleop_upper_body.selfcheck`（在 AVP 环境实跑通过，`12/12 checks passed.`）：

- ✅ 传输：`HeadFrame` 编解码往返 + 同端口 头/左/右 三类消息按 magic 正确分发
- ✅ **位姿滤波**：alpha=1.0 透传、alpha=0.5 平移取中点 + 旋转 SLERP 取半角且仍是合法旋转、
  `reset` 后下一帧原样、未跟踪朝向时只滤平移
- ✅ 遥操 MJCF 加载，**23 个 body 执行器（含底盘 3）+ 30 个手指执行器**全部解析
- ✅ **合并求解一致性（load-bearing）**：对一个可达构型的 头/左/右 三个目标，从 home 迭代求解后，
  MuJoCo 下三个末端位置误差均 **< 1 mm** 且在限位内——证明 Pinocchio(展平 MJCF) 与 MuJoCo
  对三个帧均逐构型一致、零变换
- ✅ **自动补偿（关键不变量）**：给一个让躯干大幅**转腰**的整体目标，求解后**躯干确实移动**（~0.45 rad）
  **且 头 + 双手三个末端仍贴合各自目标**（误差 < 5 mm）——证明手臂在同一次求解里补偿了躯干运动
- ✅ **全身底盘优先级**：(a) 复合底盘关节的 x/y/yaw 顺序与 `BODY_JOINTS` **round-trip 一致（0.00 mm）**——
  证明求解器返回的 23 维向量驱动的是正确的底盘执行器；(b) **可达目标底盘几乎不动**（~3.7 cm）、
  **够不到的远目标底盘明显平移过去且跟踪到位**（~46 cm / <10 mm）——证明底盘是"最后手段"而非"抢着动"
- ✅ **重心平衡 / 前倾脊柱优先级**：普通前伸/前下伸手，躯干只前倾 ~2.6°、重心离底盘仅 ~4.5 cm（**不再过度补偿前倾失稳**）；
  而**真正的下蹲**（头+手一起下降）仍前倾 ~15.6° 去改变身体高度——证明前倾脊柱被正确放到最低优先级、**又没被冻死**（仍能蹲）
- ✅ **速度/加速度硬约束**：把求解器拉向一个较远目标后，**没有任何关节超过其速度上限**，**首拍从静止起步
  受 `a_max·dt` 限制**（加速度限幅在起作用，而非直接跳到速度上限），且速度上限确实被触及、`reset()` 清空加速度状态
- ✅ **软平滑任务**：默认(温和)权重下整链仍精确跟踪(<5 mm)；放大 `damping_cost` 后峰值关节速度显著下降、
  放大 `low_accel_cost` 后峰值帧间加速度显著下降（证明两个软任务接入 QP 且作用方向正确）、`reset()` 清空低加速度状态
- ✅ 手指开合从张开(0.0)到握拳(1.0)单调
- ✅ 端到端一拍：合成 头+双手 帧 → 写 ctrl → 步进，数值有限且稳定
- ✅ **EgoPoser 神经先验**：(a) 关闭(cost=0)时解与非神经管线**逐字节一致**；(b) 适度先验把躯干俯仰拉向目标(−0.01→0.20 rad)而手部跟踪仍 <2 cm；(c) **激进先验(pitch 1.2)被衰减到 0.56 rad 且胸部仍 0 cm 在踝上方**（平衡以 ~60× 权重压过先验——"深度学习提拟人、QP 数学兜底安全"闭环）；(d) 特征窗口 (1,80,54) 有限、6D 旋转往返、torch 在场时随机权重网络端到端跑通

另做了**整条链路无头烟雾测试**（真实 UDP 同发 头+左+右 帧，跑满主循环逻辑）：随头部前倾 + 双手移动，
**躯干(0.23 rad)、脖子、左右臂都按各自源移动**，头相机随头部目标位移 68 mm，ctrl 有限、仿真稳定。

### 我无法替你测的部分（需你戴 Vision Pro 验证）

- 实时 AVP 数据流（确认头 + 左右手都被 Tracking Streamer 跟踪到）
- MuJoCo 图形窗口的实际渲染与交互
- 端到端延迟与手感（上半身整体协同的实时性）

---

## 常见问题排查

**躯干猛甩 / 上半身不稳**
头部任务过强或步长过大。调小 `head_position_cost`、调小 `torso_neck_max_velocity`/`torso_neck_max_acceleration`、调大 `posture_cost`。

> 注意：若**只移动头部位置却不动其朝向**（人不会这样，但合成/异常数据可能），位置与朝向两个头部子任务会
> 互相打架而把躯干拽过头。真实 AVP 数据里头的位置与朝向是一致变化的，不会触发；实在不放心可
> `--head-no-orientation` 只跟头的位置。

**仿真端一直 `NO DATA`**
发布端没跑或 `--host/--port` 不一致（默认 `127.0.0.1:9870`）。确认 `avp_publisher` 在刷 `sent=`。

**跟随方向反了 / 旋转别扭**
先 `--no-orientation`（默认）只调位置方向。坐标轴约定问题改 `avp_teleop` 的 `AVP_TO_ROBOT_R`（世界系旋转），
**不要**用负 `position_scale`。

**手指方向/幅度不对**：调 `avp_teleop/config.py` 的 `finger_open_reach` / `finger_closed_reach`。

---

## 将来怎么接真机 / ROS

与双臂版一致：`avp_teleop/robot_interface.py` 已把机器人抽象成 `RobotInterface`，真机实现 `ROSRobot` 即可。
合并求解器输出一个 **23 维全身关节向量（底盘 3 + torso 4 + neck 2 + 双臂 14）**，天然对应真机的**全身控制器**
（真实人形/移动机器人通常也是一个 whole-body 控制器，而非按肢体拆分）：把上半身 20 维接 Astribot 的关节话题、
底盘 3 维（x/y 速度 + 偏航）接底盘运动话题即可，主循环与映射逻辑不变。底盘优先级完全由 `damping_cost_chassis`
一个权重决定，与真机无关。
