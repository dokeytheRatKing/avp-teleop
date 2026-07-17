#!/usr/bin/env bash
# Replay every recorded AVP clip through the upper-body teleop, one at a time,
# with the MuJoCo viewer so you can watch the retargeting result.
#
#   avp_trajectory/clip1.json .. clip11.json  --replay-avp-->  render
#
# Sequential: each clip opens its own viewer window; CLOSE the window (or press
# Ctrl+C) to advance to the next clip. Uses the program's DEFAULT parameters.
#
# Usage (from the repo root, inside the AVP conda env):
#   bash avp_teleop_upper_body/replay_all.sh              # clips 1..11
#   bash avp_teleop_upper_body/replay_all.sh 3 7          # only clips 3..7
#   RECORD=1 bash avp_teleop_upper_body/replay_all.sh     # also save retarget traces
#
# Toggles (edit or pass as env vars):
#   RECORD=1   -> also record the retarget trajectory to
#                 retargetting_trajectory/<clip>.json for later pure replay.
#   PLAIN=1    -> replay with the program's bare defaults (drop the EXTRA
#                 options below: no orientation / egoposer / init-pose).
set -euo pipefail

# macOS passive viewer needs mjpython; fall back to python elsewhere.
PY="$(command -v mjpython || command -v python)"
MODULE="avp_teleop_upper_body.sim_teleop"

FIRST="${1:-1}"
LAST="${2:-11}"

# Default teleop options applied to every clip. Edit this line to change what
# the batch replays with. Currently: track hand orientation, run the EgoPoser
# neural trunk prior, and use the saved my_pose1 as the initial/rest posture.
EXTRA=(--orientation --init-pose my_pose1 --body-alpha 0.2 --alpha-translation 0.3 --alpha-rotation 0.3  --damping-waist 0.5 --damping-chassis-yaw 8 --damping-chassis-linear 12)
if [[ "${PLAIN:-0}" == "1" ]]; then
  EXTRA=()                     # PLAIN=1 -> program defaults (no extra options)
fi

echo "[replay_all] using: $PY -m $MODULE"
echo "[replay_all] clips $FIRST..$LAST | RECORD=${RECORD:-0} PLAIN=${PLAIN:-0}"
echo "[replay_all] options: ${EXTRA[*]:-<program defaults>}"
echo "[replay_all] CLOSE each viewer window (or Ctrl+C) to advance to the next clip."
echo

for i in $(seq "$FIRST" "$LAST"); do
  clip="clip${i}"
  args=(--replay-avp "$clip" ${EXTRA[@]+"${EXTRA[@]}"})
  if [[ "${RECORD:-0}" == "1" ]]; then
    args+=(--record-retarget "$clip")   # -> retargetting_trajectory/clipN.json
  fi
  echo "==================================================================="
  echo "[replay_all] ($i/$LAST) $clip"
  echo "[replay_all] $PY -m $MODULE ${args[*]}"
  echo "==================================================================="
  # Don't let one clip's non-zero exit (e.g. Ctrl+C) abort the whole batch.
  "$PY" -m "$MODULE" "${args[@]}" || echo "[replay_all] $clip exited non-zero; continuing."
  echo
done

echo "[replay_all] Done ($FIRST..$LAST)."
