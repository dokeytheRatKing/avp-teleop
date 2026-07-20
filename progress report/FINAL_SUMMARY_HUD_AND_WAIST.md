# 完成总结：MuJoCo 档位 HUD + 腰部对齐双机制

## 1. MuJoCo 窗口档位 HUD（右上角 overlay）

### 问题诊断
- **用户反馈**：overlay 各行长度不一致，数字列不对齐
- **根本原因**：MuJoCo 的 overlay 字体是**比例字体**（proportional font）
  - 空格 = 6px，数字 = 12px，点 = 6px，'%' = 18px
  - 无法用空格填充实现数字右对齐（`%5.1f` 等格式化无效）

### 解决方案：像素级对齐填充
利用 6px 作为基准单位（digit = 2×space），实现精确的像素级右对齐：

```python
def _gear_pad_damp(damp) -> str:
    """Right-align damping (X.X) to fixed pixel width via 2-space/digit padding."""
    if damp is None:
        return " " * (2 * _GEAR_DAMP_INT_DIGITS + 3)  # blank cell, same width
    s = f"{damp:.1f}"
    int_len = len(s.split(".")[0])
    return "  " * max(0, _GEAR_DAMP_INT_DIGITS - int_len) + s
```

- 每缺少一个整数位 → 填充 **2 个空格**（= 12px = 1 digit）
- `_GEAR_DAMP_INT_DIGITS = 3`：支持最大 999.9
- `_GEAR_PCT_DIGITS = 3`：支持最大 100%（实际 clamp 到 [0,100]）

### 最终布局
```
GEAR
trans    111.7      7%  D  AUTO
yaw          0.1  100%  D  AUTO
pitch                   N  AUTO
```

**验证结果**：
- 所有行恒为 **220px** 等宽
- 数字列（damp + pct）= 120px 固定（无论 1/2/3 位数）
- 档位字母跟随数字块，自然对齐（D/N=15px P=14px，1px 误差被末尾 mode 列吸收）
- mode 列（AUTO / MANU x.xs）变化时，前面的数字列完全不动

### CLI 参数
- **`--no-gear-hud`**：关闭 HUD（默认开启）
- 终端状态行不受影响（依然每 2 秒刷新）

---

## 2. 腰部对齐双机制（D 档姿态正则化）

### 用户反馈
> "事实证明：`--waist-yaw-follow` 是一个不好的设计。我建议放弃 waist_yaw_follow_cost 作为任务进行解 IK，保留 `--waist-boost-d-gear` 用来放大姿态正则化的权重，或者在 D 档的时候加一个约束要求 torso_joint_4（腰偏航）的角度接近 0。"

### 实现：两个独立机制

#### 机制 1：PostureTask Boost（整体正则化）
- **`--posture-boost-d-gear 2.5`**（默认）
- D 档时：`PostureTask.cost *= 2.5`
- 拉**所有关节**向 home 姿态，包括 `torso_joint_4` (waist yaw) → 0°
- 适合希望整体姿态在移动时更稳定的场景

#### 机制 2：Dedicated Waist-Zero Task（专门腰部归零）
- **`--waist-zero-d-gear-cost 0.0`**（默认禁用，用户设置 > 0 激活）
- D 档时：`waist_zero_task.cost = waist_zero_d_gear_cost`（从 0 跳到该值）
- **仅**拉 `torso_joint_4` → 0°（固定目标，永不改变）
- 独立于 PostureTask，可单独调参

### 使用场景

| 场景 | `posture_boost` | `waist_zero_cost` | 效果 |
|------|----------------|-------------------|------|
| 默认（中等对齐） | 2.5 | 0 | 整体正则化，腰部随 PostureTask 向 0° |
| 强腰部对齐 | 2.5 | 5.0 | 双重约束：PostureTask + 专门腰部任务 |
| 仅腰部对齐 | 1.0 | 3.0 | 手臂/躯干自由，只约束腰部 |

### P/N 档行为
- 两机制均**关闭**：腰部自由旋转，手部跟踪优先
- PostureTask 回到静态权重，waist_zero_task.cost = 0

### 代码改动
- `whole_body_ik.py`：
  - 新增 `waist_zero_d_gear_cost` 参数
  - 构建 `self.waist_zero_task`（WaistYawTask，target=0 固定）
  - 新增 `set_posture_cost()` 和 `set_waist_zero_cost()` 方法
- `config.py`：
  - 新增 `waist_zero_d_gear_cost: float = 0.0`
  - 重命名语义：`neural_boost_d_gear`（EgoPoser），`waist_boost_d_gear` 变为废弃别名
- `sim_teleop.py`：
  - D-gear 检测块同时调整两机制
  - 新增 `--waist-zero-d-gear-cost` CLI 参数

### 向后兼容
- `--waist-boost-d-gear` 仍可用（现为 `--neural-boost-d-gear` 别名）
- 旧脚本无需修改

---

## 验证结果

### MuJoCo HUD
```bash
python -m py_compile avp_teleop_upper_body/sim_teleop.py  # ✓ COMPILE OK
# 所有行 220px 等宽，数字列 120px 固定
# ✓ 1/2/3 位数变化时列位置不动
# ✓ 手动/自动模式切换时列位置不动
```

### 腰部对齐
```bash
python -m py_compile avp_teleop_upper_body/*.py  # ✓ COMPILE OK
# ✓ waist_zero_d_gear_cost 配置存在
# ✓ --waist-zero-d-gear-cost CLI 参数存在
# ✓ set_waist_zero_cost() 方法可调用
```

---

## 文档
- `progress report/WAIST_ALIGNMENT_TWO_MECHANISMS.md`：腰部双机制完整设计文档
- 记忆已更新（待同步到 MEMORY.md）
- 代码注释完整：所有新参数、方法、逻辑块都有文档字符串

## 下一步
用户应在真实 clip 上调试 `--waist-zero-d-gear-cost`：
1. 从 0 开始（当前默认），观察 D 档时腰部是否扭曲
2. 若腰部滞后/偏离底盘朝向，逐步增加 1.0 → 2.0 → 3.0 ...
3. 找到腰部对齐与手部精度的最佳平衡点
4. 独立于 `--posture-boost-d-gear` 调参，两个旋钮互不干扰
