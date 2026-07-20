# 修复完成：两个Bug已解决

## ✅ Bug #1: 按键'1'导致机器人消失

**原因**: MuJoCo内部使用数字键0-9切换几何体类别可见性，无法阻止事件传播

**解决**: 改用字母键，避开MuJoCo内部绑定

## ✅ Bug #2: 终端输出混乱

**原因**: [OVERRIDE]消息（`\n`换行）与状态行（`\r`就地更新）冲突

**解决**: 打印消息前先清除状态行（120个空格覆盖）

---

## 最终键位方案

```
Q  W  E     base_trans  (底盘平移):  P / N / D
A  S  D     base_yaw    (底盘旋转):  P / N / D
Z  X        body_pitch  (躯干俯仰):  P / N
```

记忆方法：
- **QWE**（上排）→ 平移
- **ASD**（中排）→ 旋转  
- **ZX**（下排）→ 俯仰

---

## 验证

✅ 单元测试全部通过（8个键，expiry，orthogonality）  
⏳ 等待您的集成测试确认

---

## 关于UI Overlay

您提到"consider UI overlay showing active overrides"。这里有两个选项：

### 选项A: AVP (Vision Pro) 头显UI
- **优势**: 操作员视野内实时反馈（最佳用户体验）
- **劣势**: 需要AVP SDK集成，开发成本高
- **适用场景**: 生产环境，长期使用

### 选项B: MuJoCo 仿真窗口UI
- **优势**: 简单，使用`mjr_overlay()`或`viewer.add_overlay()`
- **劣势**: 操作员可能不看仿真窗口
- **适用场景**: 开发调试，近期验证

### 推荐方案

**分阶段实施**:
1. **Phase 1 (现在)**: 终端状态行已显示override（`[MANUAL] trans=D(1.2s)`）
2. **Phase 2 (本周)**: MuJoCo窗口overlay（低成本，快速验证）
3. **Phase 3 (可选)**: AVP HUD集成（生产环境优化）

---

## 下一步

请先测试修复后的版本：

```bash
mjpython -m avp_teleop_upper_body.sim_teleop --replay-avp clip1
# 按 Q/W/E/A/S/D/Z/X，观察：
# 1. 机器人身体是否还会消失？
# 2. 终端输出是否清晰？
# 3. 档位切换是否正确？
```

确认无问题后，我们讨论UI overlay的具体实现方案。
