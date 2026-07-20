# AVP 档位切换逻辑详解

## 目录
1. [三轴档位系统架构](#三轴档位系统架构)
2. [base_trans 档位切换](#base_trans-档位切换)
3. [base_yaw 档位切换](#base_yaw-档位切换)
4. [body_pitch 档位切换](#body_pitch-档位切换)
5. [控制开关与参数](#控制开关与参数)
6. [档位状态流转图](#档位状态流转图)

---

## 三轴档位系统架构

AVP 使用三个独立档位轴控制机器人运动自由度的招募策略：

| 轴名称 | 控制 DOF | 档位类型 | 独立性 |
|--------|----------|----------|--------|
| `base_trans` | 底盘平移 xy | P / N / D | 独立 |
| `base_yaw` | 底盘转向 θ | P / N / D | 独立或跟随 trans |
| `body_pitch` | 躯干俯仰 | P / N | 独立 |

**核心原则**：
- **P (Park)** = 冻结（velocity_limit=0），该 DOF 完全不参与 IK
- **N (Neutral)** = 释放但高阻尼，QP 可以招募但代价高
- **D (Drive)** = 释放且低阻尼，QP 优先招募该 DOF

---

## base_trans 档位切换

### 检测信号
**头部水平速度** (head horizontal speed, m/s)：
```python
# 代码位置: sim_teleop.py ~1353-1358
head_xy = np.asarray(live["head"])[:2, 3]  # AVP 头部 xy 位置
inst_speed = ||head_xy - prev_xy|| / dt
speed_ema += alpha * (inst_speed - speed_ema)  # EMA 平滑
```

### 档位切换规则

#### P ↔ N 切换 (冻结/释放)
```
P → N (解冻):  speed_ema > base_unfreeze_speed (0.12 m/s)
N → P (冻结):  speed_ema < base_freeze_speed (0.05 m/s)

滞环带: 0.05 ~ 0.12 m/s (防抖动)
```

**代码位置**: `sim_teleop.py:1362-1365`
```python
if ik.base_xy_frozen and base_dz["speed"] > cfg.ik.base_unfreeze_speed:
    ik.set_base_xy_frozen(False)  # P → N
elif not ik.base_xy_frozen and base_dz["speed"] < cfg.ik.base_freeze_speed:
    ik.set_base_xy_frozen(True)   # N → P
```

#### N → D 切换 (阻尼调度)
```
条件: enable_trans_scheduling = True (默认)
      AND base_xy_frozen = False

速度范围 → 阻尼映射:
  [0.0,  0.20] m/s  →  12.0 (静态, N 档)
  [0.20, 1.20] m/s  →  12.0 → 8.0 线性插值 (D 档)
  [1.20, ∞  ] m/s  →  8.0 (地板, D 档满速)
```

**代码位置**: `sim_teleop.py:1374-1381`
```python
if cfg.ik.enable_trans_scheduling and not ik.base_xy_frozen:
    target_xy_damp = np.interp(
        base_dz["speed"],
        [cfg.ik.trans_schedule_speed_low, cfg.ik.trans_schedule_speed_high],
        [cfg.ik.damping_cost_chassis_linear, cfg.ik.trans_schedule_floor])
    base_dz["trans_damp_ema"] += alpha * (target_xy_damp - base_dz["trans_damp_ema"])
    ik.set_chassis_xy_damping(base_dz["trans_damp_ema"])
```

### 控制参数

| 参数名 | 默认值 | 作用 | CLI 参数 |
|--------|--------|------|----------|
| `base_freeze_speed` | 0.05 m/s | N→P 阈值 | `--base-freeze-speed` |
| `base_unfreeze_speed` | 0.12 m/s | P→N 阈值 | `--base-unfreeze-speed` |
| `enable_trans_scheduling` | `True` | 开启 D 档调度 | `--enable-trans-scheduling` |
| `trans_schedule_speed_low` | 0.20 m/s | N/D 边界 | `--trans-schedule-speed-low` |
| `trans_schedule_speed_high` | 1.20 m/s | D 档满速 | `--trans-schedule-speed-high` |
| `damping_cost_chassis_linear` | 12.0 | N 档阻尼 | (固定) |
| `trans_schedule_floor` | 8.0 | D 档地板 | `--trans-schedule-floor` |
| `base_speed_alpha` | 0.1 | EMA 平滑系数 | (固定) |

### 关闭开关
```bash
# 关闭 base_trans 死区（整个轴始终解冻在 N 档）
--no-base-deadzone

# 关闭 D 档调度（只有 P/N，阻尼固定 12.0）
# (enable_trans_scheduling 需改代码，无 CLI 反向开关)
```

---

## base_yaw 档位切换

**base_yaw 有两种工作模式**，由 `enable_yaw_scheduling` 控制：

### 模式 1: 独立转向意图检测 (enable_yaw_scheduling = True)

#### 检测信号
**组合转向速率** (combined yaw rate, rad/s)：
```python
# 代码位置: sim_teleop.py ~1428-1456
# 1. 头部 yaw 速率 (interaural axis, 俯仰鲁棒)
head_yaw = _head_interaural_yaw(live["head"], align_R_mat)
head_yaw_rate = |unwrap(head_yaw - prev)| / dt

# 2. 手-头连线 yaw 速率
hands_mid = 0.5 * (left_xy + right_xy)
vec = hands_mid - head_xy
hands_yaw = atan2(vec.y, vec.x)
hands_yaw_rate = |unwrap(hands_yaw - prev)| / dt  # 需距离 > 0.25m

# 3. 组合信号
combined_rate = head_yaw_rate + hands_yaw_rate
rate_ema += alpha * (combined_rate - rate_ema)  # EMA 平滑
```

**物理意义**：
- `head_yaw_rate` 高 → 操作者转头/转身
- `hands_yaw_rate` 高 → 双手绕身体扫动（如 clip7/8 原地扭腰）
- 组合捕获两种转向意图："转身"和"扭腰"

#### 档位切换规则

##### P ↔ N 切换
```
P → N (解冻):  rate_ema > base_yaw_unfreeze_rate (0.40 rad/s)
N → P (冻结):  rate_ema < base_yaw_freeze_rate (0.15 rad/s)

滞环带: 0.15 ~ 0.40 rad/s
```

**代码位置**: `sim_teleop.py:1464-1467`
```python
if ik.base_yaw_frozen and r > cfg.ik.base_yaw_unfreeze_rate:
    ik.set_base_yaw_frozen(False)  # P → N
elif not ik.base_yaw_frozen and r < cfg.ik.base_yaw_freeze_rate:
    ik.set_base_yaw_frozen(True)   # N → P
```

##### N → D 切换 (阻尼调度)
```
速率范围 → 阻尼映射:
  [0.0, 0.5] rad/s  →  8.0 (静态, N 档)
  [0.5, 3.0] rad/s  →  8.0 → 2.0 线性插值 (D 档)
  [3.0, ∞ ] rad/s   →  2.0 (地板, D 档满速)

释放率 = (8.0 - 2.0) / 8.0 = 75% (激进调度)
```

**代码位置**: `sim_teleop.py:1470-1478`
```python
if not ik.base_yaw_frozen:
    target_damp = np.interp(
        r,
        [cfg.ik.yaw_schedule_rate_low, cfg.ik.yaw_schedule_rate_high],
        [cfg.ik.damping_cost_chassis_yaw, cfg.ik.yaw_schedule_floor])
    base_dz["yaw_sched"]["damp_ema"] += a * (target_damp - base_dz["yaw_sched"]["damp_ema"])
    ik.set_chassis_yaw_damping(base_dz["yaw_sched"]["damp_ema"])
```

---

### 模式 2: 镜像 base_trans (enable_yaw_scheduling = False, **默认**)

**代码位置**: `sim_teleop.py:1483-1488`
```python
elif yaw_override is None and cfg.ik.base_deadzone:
    # Scheduling OFF: base yaw mirrors the xy freeze
    if ik.base_yaw_frozen != ik.base_xy_frozen:
        ik.set_base_yaw_frozen(ik.base_xy_frozen)
```

**行为**：
- `base_trans` 在 P → `base_yaw` 在 P
- `base_trans` 在 N/D → `base_yaw` 在 N（**无 D 档，阻尼固定 8.0**）

**历史原因** (config.py:419-426)：
> 默认关闭，因为早期实验中头部转向信号在蹲下时不稳定，且腰部跟随任务
> (WaistYawTask) 当时没有足够的优先级空间。保留代码用于未来外部腰部目标。

---

### 控制参数

| 参数名 | 默认值 | 作用 | CLI 参数 |
|--------|--------|------|----------|
| `enable_yaw_scheduling` | `False` | **主开关**: True=独立转向检测, False=镜像trans | `--enable-yaw-scheduling` |
| `base_yaw_freeze_rate` | 0.15 rad/s | N→P 阈值 (仅模式1) | `--base-yaw-freeze-rate` |
| `base_yaw_unfreeze_rate` | 0.40 rad/s | P→N 阈值 (仅模式1) | `--base-yaw-unfreeze-rate` |
| `yaw_schedule_rate_low` | 0.5 rad/s | N/D 边界 (仅模式1) | `--yaw-schedule-rate-low` |
| `yaw_schedule_rate_high` | 3.0 rad/s | D 档满速 (仅模式1) | `--yaw-schedule-rate-high` |
| `damping_cost_chassis_yaw` | 8.0 | N 档阻尼 | (固定) |
| `yaw_schedule_floor` | 2.0 | D 档地板 (仅模式1) | `--yaw-schedule-floor` |
| `yaw_schedule_alpha` | 0.2 | EMA 平滑系数 (仅模式1) | `--yaw-schedule-alpha` |

### 关闭开关
```bash
# 关闭 base_yaw 独立死区，切换到镜像模式
# (默认就是关闭的，无需操作)

# 开启独立转向检测（模式1）
--enable-yaw-scheduling
```

---

## body_pitch 档位切换

### 检测信号
**头部垂直速度** (head vertical speed, m/s)：
```python
# 代码位置: sim_teleop.py ~1395-1405
head_z = float(np.asarray(live["head"])[2, 3])  # AVP 头部 Z
inst_vspeed = |head_z - prev_z| / dt
vspeed_ema += alpha * (inst_vspeed - vspeed_ema)  # EMA 平滑
```

### 档位切换规则

**只有 P/N 二态**（无 D 档，躯干俯仰没有阻尼调度）：
```
P → N (解冻):  vspeed_ema > lean_unfreeze_speed (0.20 m/s)
N → P (冻结):  vspeed_ema < lean_freeze_speed (0.08 m/s)

滞环带: 0.08 ~ 0.20 m/s
```

**代码位置**: `sim_teleop.py:1402-1405`
```python
if ik.lean_frozen and base_dz["vspeed"] > cfg.ik.lean_unfreeze_speed:
    ik.set_lean_frozen(False)  # P → N
elif not ik.lean_frozen and base_dz["vspeed"] < cfg.ik.lean_freeze_speed:
    ik.set_lean_frozen(True)   # N → P
```

**物理意义**：
- P 档 (冻结) → 防止静止时膝盖/脚踝抖动 (clip1-3)
- N 档 (释放) → 允许下蹲/弯腰 (垂直速度高)

### 控制参数

| 参数名 | 默认值 | 作用 | CLI 参数 |
|--------|--------|------|----------|
| `lean_freeze_speed` | 0.08 m/s | N→P 阈值 | `--lean-freeze-speed` |
| `lean_unfreeze_speed` | 0.20 m/s | P→N 阈值 | `--lean-unfreeze-speed` |
| `base_speed_alpha` | 0.1 | EMA 平滑 (复用) | (固定) |

### 关闭开关
```bash
# body_pitch 死区无独立开关，跟随 base_deadzone
--no-base-deadzone  # 关闭后 lean 始终在 N 档
```

---

## 控制开关与参数

### 全局主开关

| 开关 | 默认 | 作用范围 | CLI |
|------|------|----------|-----|
| `base_deadzone` | `True` | 控制所有三轴的 P/N 切换 | `--no-base-deadzone` 关闭 |

**关闭效果** (`--no-base-deadzone`):
- `base_trans` 始终解冻 (N 档)
- `base_yaw` 始终解冻 (N 档)
- `body_pitch` 始终解冻 (N 档)
- D 档调度仍然生效（如果对应 `enable_*_scheduling` 开启）

---

### 独立调度开关

| 开关 | 默认 | 控制 | CLI |
|------|------|------|-----|
| `enable_trans_scheduling` | `True` | base_trans 的 N→D 阻尼调度 | `--enable-trans-scheduling` |
| `enable_yaw_scheduling` | `False` | base_yaw 的独立转向检测 + N→D 阻尼调度 | `--enable-yaw-scheduling` |

**层级关系**：
```
base_deadzone (P/N gate)
    ├─ enable_trans_scheduling (N→D schedule)
    └─ enable_yaw_scheduling (独立 P/N gate + N→D schedule)
        └─ False 时 → yaw 镜像 trans 的 P/N 状态
```

---

### 推荐配置组合

#### 配置 A: 默认 (保守, 当前代码)
```bash
# 无需参数，直接运行
base_deadzone = True
enable_trans_scheduling = True
enable_yaw_scheduling = False
```
**行为**: trans 独立检测, yaw 跟随 trans, pitch 独立检测

---

#### 配置 B: 全独立 (激进)
```bash
--enable-yaw-scheduling
```
**行为**: 三轴完全独立，原地转身时 yaw 可单独解冻

---

#### 配置 C: 无死区 (最自由)
```bash
--no-base-deadzone
```
**行为**: 所有轴始终解冻，适合调试或遥控场景

---

#### 配置 D: 只要 P/N，不要 D (简化)
```bash
# 需要修改代码关闭 enable_trans_scheduling
# 或设置超高的 schedule_speed_low 让 D 档永不触发
--trans-schedule-speed-low 999
```
**行为**: trans 只有 P/N (阻尼固定 12.0), yaw 跟随

---

## 档位状态流转图

### base_trans 状态机
```
                    speed < 0.05
         ┌──────────────────────────────┐
         │                              │
         ▼                              │
    ┌────────┐  speed > 0.12       ┌───┴────┐
    │   P    │ ─────────────────▶  │   N    │
    │ 冻结    │                     │静态阻尼 │
    └────────┘  ◀─────────────────  └───┬────┘
                    speed < 0.05        │
                                        │ speed > 0.20
                                        │ (scheduling ON)
                                        ▼
                                   ┌────────┐
                                   │   D    │
                                   │低阻尼  │
                                   │调度中  │
                                   └────────┘
                           [连续调度, 无离散边界]
```

### base_yaw 状态机 (模式1: enable_yaw_scheduling=True)
```
                    rate < 0.15
         ┌──────────────────────────────┐
         │                              │
         ▼                              │
    ┌────────┐  rate > 0.40        ┌───┴────┐
    │   P    │ ─────────────────▶  │   N    │
    │ 冻结    │                     │静态阻尼 │
    └────────┘  ◀─────────────────  └───┬────┘
                    rate < 0.15         │
                                        │ rate > 0.5
                                        ▼
                                   ┌────────┐
                                   │   D    │
                                   │低阻尼  │
                                   │调度中  │
                                   └────────┘
```

### base_yaw 状态机 (模式2: enable_yaw_scheduling=False, **默认**)
```
    ┌────────────────────────────────┐
    │  镜像 base_trans 的 P/N 状态    │
    │  (无 D 档, 阻尼固定 8.0)       │
    └────────────────────────────────┘
```

### body_pitch 状态机
```
                    vspeed < 0.08
         ┌──────────────────────────────┐
         │                              │
         ▼                              │
    ┌────────┐  vspeed > 0.20      ┌───┴────┐
    │   P    │ ─────────────────▶  │   N    │
    │ 冻结    │                     │ 释放   │
    └────────┘  ◀─────────────────  └────────┘
                    vspeed < 0.08

    (无 D 档, 躯干俯仰无阻尼调度)
```

---

## 手动档位覆盖 (Manual Override)

**键盘映射** (keys 2-9, 持续 2 秒后释放回自动):
```
2 3 4 → base_trans  P / N / D
5 6 7 → base_yaw    P / N / D
8 9   → body_pitch  P / N
```

**优先级**: 手动档位 > 自动检测逻辑

**代码位置**: `sim_teleop.py:1287-1343` (manual override layer)

**平滑切换**: 手动档位设定时会重置 EMA 状态，防止释放时的换挡冲击。

---

## 总结速查表

| 配置项 | base_trans | base_yaw (默认) | base_yaw (--enable-yaw-scheduling) | body_pitch |
|--------|-----------|----------------|-----------------------------------|-----------|
| **检测信号** | 头部水平速度 | 镜像 trans | 组合转向速率 | 头部垂直速度 |
| **P→N 阈值** | 0.12 m/s | (跟随) | 0.40 rad/s | 0.20 m/s |
| **N→P 阈值** | 0.05 m/s | (跟随) | 0.15 rad/s | 0.08 m/s |
| **N→D 起点** | 0.20 m/s | 无 D | 0.5 rad/s | 无 D |
| **D 档满速** | 1.2 m/s | — | 3.0 rad/s | — |
| **阻尼释放率** | 33% (12→8) | — | 75% (8→2) | — |
| **主开关** | `base_deadzone` | `base_deadzone` | `enable_yaw_scheduling` | `base_deadzone` |
| **调度开关** | `enable_trans_scheduling` | — | `enable_yaw_scheduling` | — |

---

**文档版本**: 2025-01-20  
**代码版本**: `sim_teleop.py` (含 D 档姿态增强 + 手动档位覆盖)
