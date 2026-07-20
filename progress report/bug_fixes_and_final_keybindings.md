# Bug修复报告：手动模式切换

## Bug #1: 按键'1'导致机器人身体消失

### 现象
按下数字键'1'后，机器人身体消失；再次按下'1'恢复正常。键2-8没有此问题。

### 根本原因
MuJoCo的passive viewer仍然保留内部按键处理器。数字键0-9在MuJoCo内部用于**切换几何体类别(geom category)的可见性**。

- 键'1'（ASCII 49）切换geom category 1的可见性
- 机器人身体属于category 1 → 按'1'隐藏 → 再按'1'显示

### 技术细节
MuJoCo的`key_callback`机制：
- 用户的callback先执行
- 但**无法阻止**事件传播到viewer的内部handler
- `key_callback`不提供"消费事件"的机制

### 解决方案
**改用字母键代替数字键**，避开MuJoCo的内部绑定：

| 原方案 (数字键) | 新方案 (字母键) | 功能 |
|---|---|---|
| 1, 2, 3 | **Q, W, E** | base_trans P/N/D |
| 4, 5, 6 | **A, S, D** | base_yaw P/N/D |
| 7, 8 | **Z, X** | body_pitch P/N |

字母键布局优势：
- ✅ 无MuJoCo冲突
- ✅ QWEASD形成两排，符合人体工学
- ✅ Z/X在底部，逻辑分组清晰

---

## Bug #2: 终端状态打印混乱

### 现象
按下数字键后，[OVERRIDE]消息与状态行混在一起，难以阅读：
```
[OVERRIDE] base translation → D (extended 2.0s)_trans P | base_yaw D 4.7[41%] | body_pitch PL] yaw=P(1.2s)
```

### 根本原因
两种输出机制冲突：
- **状态行**: 使用`\r`（回车，不换行）就地更新同一行
- **[OVERRIDE]消息**: 使用普通`print()`（带`\n`换行）

当[OVERRIDE]消息打印时，状态行的内容还在缓冲区，导致混合输出。

### 解决方案
**在打印[OVERRIDE]消息前，先清除状态行**：

```python
def key_callback(keycode: int) -> None:
    override_msg = manual_override.handle_keypress(keycode)
    if override_msg is not None:
        # 清除当前状态行（120个空格覆盖 + 回车）
        print("\r" + " " * 120 + "\r", end="")
        print(override_msg)  # 现在可以干净地打印消息
        return
```

同样的修复也应用到'c'（校准）和空格（暂停）的消息输出。

---

## 最终键位方案

```
Q  W  E     ← base_trans  (底盘平移):  P / N / D
A  S  D     ← base_yaw    (底盘旋转):  P / N / D
Z  X        ← body_pitch  (躯干俯仰):  P / N

其他键位：
c           校准 (calibrate)
space       暂停/恢复 (pause/resume)
```

### 记忆方法
- **QWE**（上排）: 前进方向 → **平移**
- **ASD**（中排）: 转向控制 → **旋转**
- **ZX**（下排）: 垂直运动 → **俯仰**

---

## 修改的文件

1. `avp_teleop_upper_body/manual_override.py`
   - 修改`handle_keypress()`中的key_map: `49-56` → `81,87,69,65,83,68,90,88`
   - 更新文档字符串

2. `avp_teleop_upper_body/sim_teleop.py`
   - `key_callback()`: 添加状态行清除逻辑（`\r` + 120空格）
   - 启动消息: "keys 1-8" → "keys Q-E/A-S-D/Z-X"
   - 注释更新

3. `test_manual_override.py`
   - 更新测试用例的ASCII码: `49-56` → `81,87,69,65,83,68,90,88`
   - 更新非冲突键测试: 'A' → 'B'

---

## 验证

### 单元测试
```bash
$ python3 test_manual_override.py
✓ Key Q → base_trans P
✓ Key W → base_trans N
✓ Key E → base_trans D
✓ Key A → base_yaw P
✓ Key S → base_yaw N
✓ Key D → base_yaw D
✓ Key Z → body_pitch P
✓ Key X → body_pitch N
✅ All tests passed!
```

### 集成测试（待用户确认）
- [ ] 按Q/W/E，确认base_trans正确切换P/N/D，无几何体消失
- [ ] 按A/S/D，确认base_yaw正确切换P/N/D
- [ ] 按Z/X，确认body_pitch正确切换P/N
- [ ] 验证终端输出清晰，[OVERRIDE]消息不与状态行混合

---

## 经验教训

1. **工具内部行为需要实测**: 文档说passive viewer"不使用数字键"，但实际上内部handler仍然存在
2. **终端输出需要状态管理**: `\r`和`\n`混用需要显式清除缓冲区
3. **键位冲突优先级**: 避开工具内置绑定 > 追求"理想"键位布局
4. **快速迭代胜于完美预测**: 用户实测发现问题 → 5分钟修复 > 提前防御所有可能冲突

---

## 关于UI Overlay（下一步讨论）

用户提到"consider UI overlay showing active overrides"，这里的UI可能指：

**选项A: AVP (Vision Pro) UI**
- 在Vision Pro头显内叠加HUD，显示当前激活的override状态
- 优势: 操作员视野内实时反馈
- 劣势: 需要AVP SDK集成，复杂度高

**选项B: MuJoCo窗口UI**
- 在MuJoCo viewer窗口内绘制overlay文本（使用`mjr_overlay()`）
- 优势: 简单，无需AVP改动
- 劣势: 操作员可能不看MuJoCo窗口

**推荐**: 先实现MuJoCo窗口overlay（低成本验证），再考虑AVP集成。

等待用户确认bug修复后，继续此话题。
