"""Manual keyboard-based mode override for the three-axis mode switching system.

Provides a 5-second timed override mechanism that forces each axis (base_trans,
base_yaw, body_pitch) into a specific gear (P/N/D) regardless of the automatic
intent-based thresholds. Designed to integrate smoothly with the existing automatic
switching without causing damping discontinuities or transient instability.

Keyboard mapping (using number keys 2-9, avoiding 0/1 which have MuJoCo conflicts):
    2, 3, 4  ->  base_trans  P / N / D
    5, 6, 7  ->  base_yaw    P / N / D
    8, 9     ->  body_pitch  P / N

Each press locks the specified axis into the specified gear for 5 seconds, then
releases back to automatic control. Multiple presses extend the override duration.
"""

from __future__ import annotations

import time
from typing import Optional, Literal, Dict
from dataclasses import dataclass

Gear = Literal["P", "N", "D"]
Axis = Literal["base_trans", "base_yaw", "body_pitch"]


@dataclass
class Override:
    """Active manual override state for one axis."""
    axis: Axis
    gear: Gear
    expires_at: float  # time.time() when this override expires


class ManualOverrideController:
    """Manages timed manual overrides for the three mode-switching axes.

    Thread-safe for single-threaded use (typical in a render loop). Each override
    lasts 5 seconds from the last keypress on that axis; repeated presses extend
    the duration (retriggerable one-shot).
    """

    def __init__(self, override_duration: float = 5.0):
        """
        Args:
            override_duration: How long (seconds) each manual override lasts.
        """
        self.duration = override_duration
        # Current override per axis (None = automatic control active)
        self._overrides: Dict[Axis, Optional[Override]] = {
            "base_trans": None,
            "base_yaw": None,
            "body_pitch": None,
        }
        # Track when each axis last switched gears (for telemetry/debugging)
        self._last_gear: Dict[Axis, Optional[Gear]] = {
            "base_trans": None,
            "base_yaw": None,
            "body_pitch": None,
        }

    def handle_keypress(self, keycode: int) -> Optional[str]:
        """Process a keyboard event and trigger override if it's a valid key.

        Args:
            keycode: ASCII keycode from the MuJoCo viewer key_callback.

        Returns:
            Human-readable status message if a valid override key was pressed,
            None otherwise (pass through to other handlers).
        """
        # Map ASCII keycodes to (axis, gear)
        # Using number keys 2-9 (avoiding 0=camera, 1=geom toggle)
        key_map = {
            50: ("base_trans", "P"),   # '2'
            51: ("base_trans", "N"),   # '3'
            52: ("base_trans", "D"),   # '4'
            53: ("base_yaw", "P"),     # '5'
            54: ("base_yaw", "N"),     # '6'
            55: ("base_yaw", "D"),     # '7'
            56: ("body_pitch", "P"),   # '8'
            57: ("body_pitch", "N"),   # '9'
        }

        if keycode not in key_map:
            return None  # Not an override key

        axis, gear = key_map[keycode]
        now = time.time()

        # Create or extend the override
        self._overrides[axis] = Override(
            axis=axis,
            gear=gear,
            expires_at=now + self.duration
        )

        # Track gear change
        prev = self._last_gear[axis]
        self._last_gear[axis] = gear

        axis_name = {
            "base_trans": "base translation",
            "base_yaw": "base yaw",
            "body_pitch": "body pitch"
        }[axis]

        if prev == gear:
            return f"[OVERRIDE] {axis_name} → {gear} (extended {self.duration:.1f}s)"
        else:
            return f"[OVERRIDE] {axis_name} {prev or '?'} → {gear} ({self.duration:.1f}s)"

    def get_override(self, axis: Axis) -> Optional[Gear]:
        """Get the current manual override gear for an axis, if active.

        Automatically expires stale overrides. Returns None when automatic
        control should resume.

        Args:
            axis: Which axis to query.

        Returns:
            The forced gear (P/N/D), or None if automatic control is active.
        """
        override = self._overrides[axis]
        if override is None:
            return None

        now = time.time()
        if now >= override.expires_at:
            # Override expired, release back to automatic
            self._overrides[axis] = None
            return None

        return override.gear

    def is_active(self, axis: Axis) -> bool:
        """Check if manual override is currently active for an axis."""
        return self.get_override(axis) is not None

    def time_remaining(self, axis: Axis) -> float:
        """Get seconds remaining on the current override, or 0 if inactive."""
        override = self._overrides[axis]
        if override is None:
            return 0.0
        now = time.time()
        return max(0.0, override.expires_at - now)

    def cancel(self, axis: Axis) -> None:
        """Immediately cancel any active override on an axis."""
        self._overrides[axis] = None

    def cancel_all(self) -> None:
        """Immediately cancel all active overrides."""
        for axis in self._overrides:
            self._overrides[axis] = None

    def status_line(self) -> str:
        """Generate a one-line status showing all active overrides.

        Returns:
            Empty string if no overrides active, otherwise something like:
            "[MANUAL] trans=D(1.2s) yaw=P(0.8s)"
        """
        active = []
        for axis in ["base_trans", "base_yaw", "body_pitch"]:
            gear = self.get_override(axis)
            if gear is not None:
                short = {"base_trans": "trans", "base_yaw": "yaw",
                         "body_pitch": "pitch"}[axis]
                remaining = self.time_remaining(axis)
                active.append(f"{short}={gear}({remaining:.1f}s)")

        if not active:
            return ""
        return "[MANUAL] " + " ".join(active)


class GearLock:
    """Dwell-time lock preventing an axis from switching gears too rapidly.

    Independent of (and complementary to) :class:`ManualOverrideController`: while
    that forces a gear on demand, this debounces the AUTOMATIC intent-based
    transitions. After ANY gear switch on an axis (P<->N freeze/unfreeze, or a
    manual override), the axis is locked in its new gear for ``dwell_time`` seconds
    before the automatic logic may switch it again. This suppresses gear chatter
    at threshold boundaries (e.g. head speed hovering around base_freeze_speed),
    which would otherwise cause repeated freeze/unfreeze and transient QP jitter.

    The N<->D damping ramp is continuous (EMA-smoothed) and self-debouncing, so
    only the DISCRETE freeze/unfreeze (P<->N) transitions are gated here.

    ``dwell_time = 0`` disables the lock (every switch is immediately permitted),
    keeping byte-identical behaviour to the pre-lock pipeline.
    """

    def __init__(self, dwell_time: float = 2.5):
        """
        Args:
            dwell_time: Seconds an axis must stay in its current gear after a
                switch before an automatic switch is allowed. 0 disables.
        """
        self.dwell = float(dwell_time)
        # time.time() until which each axis is locked in its current gear.
        self._locked_until: Dict[Axis, float] = {
            "base_trans": 0.0,
            "base_yaw": 0.0,
            "body_pitch": 0.0,
        }

    def can_switch(self, axis: Axis) -> bool:
        """Whether the automatic logic may switch this axis right now.

        True once the dwell period since the last switch has elapsed (always
        True when dwell == 0)."""
        if self.dwell <= 0.0:
            return True
        return time.time() >= self._locked_until[axis]

    def register_switch(self, axis: Axis) -> None:
        """Record that ``axis`` just switched gears; start its dwell lock."""
        self._locked_until[axis] = time.time() + self.dwell

    def time_remaining(self, axis: Axis) -> float:
        """Seconds left on the dwell lock for an axis (0 if unlocked)."""
        if self.dwell <= 0.0:
            return 0.0
        return max(0.0, self._locked_until[axis] - time.time())
