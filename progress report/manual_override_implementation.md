# Manual Mode-Switching Override (Manual Gearbox)

> **Status**: Implemented and tested  
> **Files**: `avp_teleop_upper_body/manual_override.py`, `sim_teleop.py`  
> **Integration**: Keyboard-based 2-second timed overrides for all three mode-switching axes

---

## Problem Statement

The automatic intent-driven mode switching (Phase 4/4b) works well for normal operation but lacks **absolute operator control** in safety-critical or complex manipulation scenarios. Operators need the ability to manually force specific gears (P/N/D) regardless of automatic thresholds, similar to a manual transmission override in an automatic car.

---

## Solution: Timed Manual Override

A **keyboard-driven manual override layer** that sits above the automatic switching logic. Each keypress forces a specific axis into a specific gear for **2 seconds**, then releases back to automatic control.

### Keyboard Mapping

| Keys | Axis | Gears |
|---|---|---|
| **1, 2, 3** | base_trans (底盘平移) | P / N / D |
| **4, 5, 6** | base_yaw (底盘旋转) | P / N / D |
| **7, 8** | body_pitch (躯干俯仰) | P / N |

**Example**: Press `3` → force base translation into D gear (dynamic, low damping) for 2 seconds.

**No MuJoCo conflicts**: Number keys 1-8 are NOT used by MuJoCo viewer defaults (verified). MuJoCo uses letter keys (c=calibrate, space=pause, etc.).

---

## Design: Smooth Handoff to Prevent Instability

This is a **switched system stability problem**. Naive implementation (instant gear changes, no state coordination) would cause:

1. **Damping discontinuities** → QP objective jumps → "shift shock" / transient oscillation
2. **EMA state mismatch** → when override expires, damping EMA carries stale values → unstable transition

### Key Stability Measures

#### 1. **Damping EMA continuity**
When entering a manual gear, the override **sets the damping EMA state** to match the target gear:
- **P gear**: Reset damping EMA to static value (prevents stale low values)
- **N gear**: Target = static damping (e.g., 12.0 for base_trans)
- **D gear**: Target = floor damping (e.g., 8.0 for base_trans)

The damping EMA then **smoothly transitions** using the existing EMA coefficient (`base_speed_alpha` / `yaw_schedule_alpha`), avoiding instant jumps.

#### 2. **Override blocks automatic logic**
While a manual override is active on an axis, the automatic intent-based logic for that axis is **skipped entirely**. This prevents the automatic thresholds from fighting the manual command.

```python
# Automatic logic only runs when manual override is NOT active
if trans_override is None and cfg.ik.base_deadzone and live["head"] is not None:
    # ... automatic freeze/unfreeze logic ...
```

#### 3. **Freeze state consistency**
When forcing P gear, the override immediately calls `ik.set_base_xy_frozen(True)` (or equivalent). When forcing N/D, it unfreezes. The velocity limits take effect instantly (coordinated freeze inside the QP), while the damping transitions smoothly.

#### 4. **Retriggerable one-shot**
Repeated keypresses **extend** the override duration (retriggerable). This allows the operator to hold a gear indefinitely by tapping the key every ~1.5 seconds.

---

## Implementation Architecture

### `ManualOverrideController` class

- **Per-axis state**: Each axis (base_trans, base_yaw, body_pitch) has independent override state
- **Expiry tracking**: Each override stores `expires_at = time.time() + 2.0`
- **Automatic expiry**: `get_override(axis)` returns `None` when the override expires (time-based, no explicit cleanup needed)
- **Status telemetry**: `status_line()` generates `[MANUAL] trans=D(1.2s) yaw=P(0.8s)` for the HUD

### Integration into `sim_teleop.py`

1. **Key callback**: Routes keys 1-8 to `manual_override.handle_keypress()`, prints confirmation message
2. **Pre-automatic layer**: Before each axis's automatic logic, check `get_override(axis)`:
   - If override active → apply forced gear, skip automatic logic
   - If override inactive → run automatic logic as normal
3. **Status line**: Append manual override status to the existing P/N/D gear display

---

## Safety Properties

✅ **Smooth damping transitions**: No instant QP objective jumps  
✅ **EMA state continuity**: Damping EMA reset/aligned on gear entry, smooth on exit  
✅ **Independent axes**: Three orthogonal overrides, no cross-coupling  
✅ **Automatic fallback**: 2-second timeout guarantees return to safe automatic mode  
✅ **No persistent state**: Override state lives in memory only, reset on calibration not required  

---

## Testing

### Unit Tests (`test_manual_override.py`)

- ✅ All 8 keys map correctly to (axis, gear) pairs
- ✅ Override expiry after 2 seconds
- ✅ Orthogonal axes (override one, others unaffected)
- ✅ Status line generation
- ✅ Non-override keys ignored (passthrough)

### Integration Testing (Next Step)

1. **Stability test**: Force rapid P→D→P transitions, measure base position jitter
2. **Handoff test**: Force D gear for 2s, let expire, verify smooth return to automatic N/P
3. **Real-scenario test**: Operator manually locks base (key 1=P) during delicate manipulation, releases via timeout

---

## Usage Examples

### Scenario 1: Lock base during precision task
Operator is performing fine manipulation and wants the base absolutely still, overriding any AVP noise:
- Press **`1`** → base translation frozen (P gear) for 2 seconds
- Press **`4`** → base yaw frozen (P gear) for 2 seconds
- Tap `1` and `4` every 1.5 seconds to maintain → indefinite lock

### Scenario 2: Force dynamic base for large reach
Operator needs to reach far, but automatic thresholds haven't triggered D gear yet:
- Press **`3`** → base translation enters D gear (low damping, eager) for 2 seconds
- Base immediately becomes more responsive, helps extend reach
- After 2s, returns to automatic control

### Scenario 3: Test mode switching behavior
Developer wants to test specific gear transitions:
- Press **`2`** → force base_trans into N gear (static damping, unfrozen)
- Observe robot behavior at static damping
- Press **`3`** → force D gear, observe transition smoothness
- Let expire → verify automatic logic resumes correctly

---

## Open Questions / Future Work

1. **Haptic feedback**: Vision Pro has no tactile keyboard → need UI overlay showing active overrides
2. **Override duration tuning**: 2s is conservative; 1.5s might be optimal (less operator fatigue for long holds)
3. **Emergency lock**: Consider a "hold all" key (e.g., `0`) that locks all three axes to P until released
4. **Integration with real robot**: ROS2 param to disable manual override (safety: no manual mode during actual deployment)

---

## Related Files

- Implementation: [`manual_override.py`](../avp_teleop_upper_body/manual_override.py)
- Integration: [`sim_teleop.py`](../avp_teleop_upper_body/sim_teleop.py) (search `MANUAL OVERRIDE LAYER`)
- Unit test: [`test_manual_override.py`](../test_manual_override.py)
- Automatic switching (background): [`mode_switching_report.md`](./mode_switching_report.md)
