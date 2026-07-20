# AVP 档位系统完整参数总结

## 三轴档位系统概览

| 轴 | P (驻车) | N (空档) | D (驱动) |
|---|---|---|---|
| **base_trans** (底盘平移) | 冻结 | 静态阻尼 | 调度阻尼↓ |
| **base_yaw** (底盘旋转) | 冻结 | 静态阻尼 | 调度阻尼↓ |
| **body_pitch** (躯干俯仰) | 冻结 | 释放 | — |

---

## 1. base_trans (底盘平移 xy)

### 速度阈值 (头部水平速度)
```
P 档 (冻结):  < 0.05 m/s   (freeze_speed)
  ↓ 滞环
N 档 (静态):  0.05 ~ 0.12 m/s
  ↓ 0.12 m/s (unfreeze_speed) 解冻
N→D 过渡:     0.12 ~ 0.20 m/s
  ↓ 0.20 m/s (schedule_speed_low)
D 档 (调度):  0.20 ~ 1.2 m/s
  ↓ 1.2 m/s (schedule_speed_high) 到达地板
D 档满速:     > 1.2 m/s
```

### 阻尼成本 (damping cost)
```
P 档:  velocity_limit = 0 (硬冻结，无阻尼)
N 档:  12.0  (damping_cost_chassis_linear, 静态)
D 档:  12.0 → 8.0  线性插值 (speed 0.20 → 1.2)
       地板 = 8.0 (trans_schedule_floor)
       释放率 = (12-8)/12 = 33%
```

### 默认状态
- `enable_trans_scheduling = True` (调度开启)
- 进入 D 档条件: 解冻 + 速度 > 0.20 m/s + 阻尼 < 11.9

---

## 2. base_yaw (底盘转向 θ)

### 速度阈值 (组合转向速率 = 头部yaw速率 + 手-头yaw速率)
```
P 档 (冻结):  跟随 base_trans (镜像xy冻结状态)
N 档 (静态):  < 0.5 rad/s (schedule_rate_low)
D 档 (调度):  0.5 ~ 3.0 rad/s
              ↓ 3.0 rad/s (schedule_rate_high) 到达地板
D 档满速:     > 3.0 rad/s
```

### 阻尼成本
```
P 档:  velocity_limit = 0 (硬冻结)
N 档:  8.0  (damping_cost_chassis_yaw, 静态)
D 档:  8.0 → 2.0  线性插值 (rate 0.5 → 3.0)
       地板 = 2.0 (yaw_schedule_floor)
       释放率 = (8-2)/8 = 75%  (激进调度)
```

### 默认状态
- `enable_yaw_scheduling = True` (调度开启)
- 进入 D 档条件: 解冻 + 转速 > 0.5 rad/s + 阻尼 < 7.9

---

## 3. body_pitch (躯干俯仰 torso_joint_1/2/3)

### 速度阈值 (头部垂直速度 Z)
```
P 档 (冻结):  < 0.08 m/s  (lean_freeze_speed, 静止不蹲)
  ↓ 滞环
N 档 (释放):  > 0.20 m/s  (lean_unfreeze_speed, 下蹲)
```

### 无阻尼调度
body_pitch 没有 D 档概念，只有 P/N 二态：
- P = 冻结 (velocity_limit=0) → 防止膝盖/脚踝抖动
- N = 释放 → 允许下蹲/弯腰

---

## 4. 姿态正则化权重 (Posture Regularization)

### 基础任务权重 (N 档 / 静态)
```
End-effector tracking (最高优先级):
  arm_position_cost       = 10.0   (手位置追踪)
  arm_orientation_cost    = 1.0    (手方向追踪, 仅当启用)
  head_position_cost      = 3.0    (头位置追踪, 中等权重)

Posture regularization (中等优先级):
  neural_posture_cost     = 5.0    (EgoPoser躯干姿态先验, 默认0=关)
  waist_yaw_follow_cost   = 2.0    (腰部跟随头部转向, 默认0=关)

Damping (最低优先级, 速度惩罚):
  damping_cost_arm        = 0.05   (手臂关节)
  damping_cost_waist      = 0.4    (腰部yaw)
  damping_cost_neck       = 0.4    (颈部)
  damping_cost_lean       = 0.4    (躯干俯仰)
  damping_cost_chassis_linear = 12.0  (底盘xy, 高→不轻易动)
  damping_cost_chassis_yaw    = 8.0   (底盘yaw)
```

### D 档姿态增强 (NEW)
当 **base_trans 或 base_yaw 进入 D 档** 时：
```
neural_posture_cost  ×= 2.5   (5.0 → 12.5, 如果启用)
waist_yaw_follow_cost ×= 2.5  (2.0 → 5.0, 如果启用)
```

**效果**: 
- 腰部朝向更强地与底盘对齐 (不扭曲滞后)
- 躯干姿态约束更强 (健康站姿)
- **代价**: 手部追踪精度适度降低 (可接受，行走时不需 100% 精度)

**配置参数**:
```
posture_boost_d_gear = 2.5   (默认, CLI: --posture-boost-d-gear)
waist_boost_d_gear   = 2.5   (默认, CLI: --waist-boost-d-gear)
```

---

## 5. 档位切换阈值总结

### P → N (解冻)
```
base_trans:  速度 > 0.12 m/s
base_yaw:    跟随 trans (或独立: 转速 > 0.5 rad/s)
body_pitch:  垂直速度 > 0.20 m/s
```

### N → P (冻结)
```
base_trans:  速度 < 0.05 m/s
base_yaw:    跟随 trans
body_pitch:  垂直速度 < 0.08 m/s
```

### N → D (低阻尼调度)
```
base_trans:  解冻 + 速度 > 0.20 m/s → 阻尼开始下降
base_yaw:    解冻 + 转速 > 0.5 rad/s → 阻尼开始下降
body_pitch:  无 D 档
```

### D 档内部 (阻尼连续调度)
```
base_trans:  速度 [0.20, 1.2] m/s → 阻尼 [12.0, 8.0] 线性
base_yaw:    转速 [0.5, 3.0] rad/s → 阻尼 [8.0, 2.0] 线性
```

---

## 6. 权重层级 (QP 优先级)

```
Tier 1 (最高):  End-effector tracking (10.0 / 3.0 / 1.0)
Tier 2 (中等):  Posture regularization (5.0 / 2.0, D档×2.5)
Tier 3 (最低):  Damping (0.05 ~ 12.0, D档调度↓)
```

D 档时的实际效果:
- **Tier 2 增强** (姿态约束 ×2.5) → 躯干更稳定
- **Tier 3 降低** (底盘阻尼 ↓33%~75%) → 底盘更容易被招募
- **Tier 1 不变** → 但受 Tier 2 增强影响，手部精度会适度降低

---

## 速查表 (典型场景)

| 场景 | base_trans | base_yaw | body_pitch | 姿态权重 |
|---|---|---|---|---|
| 静止精细操作 | P (< 0.05 m/s) | P | P (< 0.08 m/s) | 静态 |
| 原地转身 | N (0.05~0.12) | D (> 0.5 rad/s) | P | **增强×2.5** |
| 慢走 | D (0.20~0.5 m/s) | N | P | **增强×2.5** |
| 快走 | D (> 0.5 m/s) | D | P | **增强×2.5** |
| 下蹲 | P | P | N (> 0.20 m/s) | 静态 |
| 边走边蹲 | D | D | N | **增强×2.5** |

**关键洞察**: 只要 trans 或 yaw **任一进入 D 档**，姿态约束就增强 → 保证行进时健康姿势。
