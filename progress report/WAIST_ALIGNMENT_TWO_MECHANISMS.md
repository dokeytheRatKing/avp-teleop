# Waist Alignment in D Gear: Two Independent Mechanisms

## Summary

Implemented **two independent posture regularization mechanisms** for D-gear locomotion, addressing the user's feedback that `--waist-yaw-follow` (head-driven waist tracking) was a failed design.

### Mechanism 1: PostureTask Boost (existing, repurposed)
- **`--posture-boost-d-gear 2.5`** (default)
- In D gear: `PostureTask.cost *= posture_boost_d_gear`
- Pulls **ALL joints** toward `home` pose, including `torso_joint_4` (waist yaw) → home ≈ 0°
- Whole-body regularization: arms, torso, waist all biased toward upright neutral

### Mechanism 2: Dedicated Waist-Zero Task (NEW)
- **`--waist-zero-d-gear-cost 0.0`** (default disabled, user sets > 0 to activate)
- In D gear: `waist_zero_task.cost = waist_zero_d_gear_cost` (0 → this value)
- Pulls **ONLY `torso_joint_4`** toward a fixed target of **0°**
- Independent of PostureTask: can be tuned separately for stronger waist alignment
- Implemented as a `WaistYawTask` instance with `target=0.0` (never changed)

### P/N Gear Behavior
- Both mechanisms **OFF**: `PostureTask` returns to static cost, `waist_zero_task.cost = 0`
- Waist is free to rotate for better hand tracking when stationary

## Code Changes

### `whole_body_ik.py`
1. Added parameter `waist_zero_d_gear_cost: float = 0.0` to `WholeBodyIK.__init__`
2. Built `self.waist_zero_task` (a `WaistYawTask` with fixed target 0.0) when cost > 0
3. Added to `self._tasks` list
4. Added `set_posture_cost(cost)` method (adjusts PostureTask at runtime)
5. Added `set_waist_zero_cost(cost)` method (adjusts waist_zero_task at runtime, no-op if absent)

### `config.py`
1. Added `waist_zero_d_gear_cost: float = 0.0`
2. Renamed `posture_boost_d_gear` comment to clarify it boosts PostureTask (not neural)
3. Added `neural_boost_d_gear` (replaces old `waist_boost_d_gear` semantic)
4. Kept `waist_boost_d_gear` as deprecated alias for backward compat

### `sim_teleop.py`
1. Added CLI flag `--waist-zero-d-gear-cost` (default 0.0)
2. Added CLI flag `--neural-boost-d-gear` (default 2.5, for EgoPoser)
3. Deprecated `--waist-boost-d-gear` (now alias for `--neural-boost-d-gear`)
4. Updated D-gear boost block (lines ~1550-1585):
   ```python
   if in_d_gear:
       ik.set_posture_cost(cfg.ik.posture_cost * cfg.ik.posture_boost_d_gear)
       ik.set_waist_zero_cost(cfg.ik.waist_zero_d_gear_cost)
       if ik.neural_task is not None:
           ik.set_neural_posture_cost(cfg.ik.neural_posture_cost * cfg.ik.neural_boost_d_gear)
   else:
       ik.set_posture_cost(cfg.ik.posture_cost)
       ik.set_waist_zero_cost(0.0)
       if ik.neural_task is not None:
           ik.set_neural_posture_cost(cfg.ik.neural_posture_cost)
   ```

## Rationale (User's Insight)

**Problem**: `--waist-yaw-follow` tried to make the waist track the head's interaural yaw. This was a **bad design** because:
- The waist has no headroom (already constrained by hand tracking + balance tasks)
- Competing with high-priority FrameTasks → poor tracking or arm contortion
- config.py:520-526 already documented this failure

**Solution**: Abandon head-driven waist tracking. Instead:
1. Use **PostureTask boost** for general whole-body regularization (including waist → 0°)
2. Add **dedicated waist-zero task** for users who want *extra* waist alignment strength in D gear, independently tunable

## Usage

### Scenario A: Moderate waist alignment (default)
```bash
--posture-boost-d-gear 2.5   # PostureTask pulls all joints toward home in D gear
# waist_zero_d_gear_cost stays 0 (disabled)
```
Waist is biased toward 0° via the boosted PostureTask, but other joints also regularized.

### Scenario B: Strong waist alignment
```bash
--posture-boost-d-gear 2.5
--waist-zero-d-gear-cost 5.0  # Add dedicated waist→0° task with cost 5.0 in D gear
```
Waist gets BOTH the PostureTask pull AND the dedicated 5.0-cost zero task → stronger alignment.

### Scenario C: Only waist alignment, no whole-body boost
```bash
--posture-boost-d-gear 1.0    # No PostureTask boost (1.0 = no multiplier)
--waist-zero-d-gear-cost 3.0  # Only waist gets the D-gear constraint
```
Arms/torso stay free, only waist is constrained in D gear.

## Testing

```bash
python -m py_compile avp_teleop_upper_body/*.py  # ✓ COMPILE OK
python -c "from avp_teleop_upper_body.config import WholeBodyIKConfig; ..."
# ✓ waist_zero_d_gear_cost: 0.0
# ✓ CLI flag --waist-zero-d-gear-cost exists
```

## Backward Compatibility

- `--waist-boost-d-gear` still works (now an alias for `--neural-boost-d-gear`)
- Old scripts using `--posture-boost-d-gear` continue to work (same semantics, clearer name)
- `--waist-yaw-follow-cost` unchanged (still available for external waist targets, but not recommended for head-driven use per config.py comments)

## Next Steps

User should tune `--waist-zero-d-gear-cost` on real clips:
- Start with 0 (current default) → observe waist twist in D gear
- If waist lags/twists away from chassis heading, increment to 1.0, 2.0, 3.0, ...
- Find the sweet spot where waist aligns without sacrificing too much hand precision
- Independent of `--posture-boost-d-gear`, so the two can be tuned separately
