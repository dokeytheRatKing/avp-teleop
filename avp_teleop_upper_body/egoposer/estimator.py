"""EgoPoser estimator: AVP head/hand poses -> a robot trunk-posture prior.

This wraps the reimplemented :class:`AvatarNet` and turns its SMPL output into
the two numbers the robot's trunk can actually realise: a forward-lean **pitch**
and an axial **yaw**. It is the only place ``torch`` is touched, and only when
the neural prior is enabled -- everything is imported lazily so the default
teleop path (and the offline self-checks) never require torch.

What we extract, and why it needs no SMPL body model
----------------------------------------------------
The network outputs ``pose_body`` = 21 SMPL joint rotations in the **6D local**
form (each joint relative to its parent, in SMPL's native Y-up frame). The
operator's trunk posture is fully contained in the sagittal spine chain
spine1/spine2/spine3 (SMPL joints 3/6/9) and, if wanted, neck (joint 12). We
compose the three spine locals into the chest-relative-to-pelvis rotation and
read off two intrinsic-Euler angles:

    R_chest = R_spine1 @ R_spine2 @ R_spine3           (pelvis-frame)
    (flexion, twist, lateral) = euler(R_chest, "XYZ")  # SMPL Y-up:
        flexion  ~ rotation about X  -> robot TRUNK PITCH (forward lean)
        twist    ~ rotation about Y  -> robot WAIST  YAW  (torso_joint_4)

Because these are *local* joint rotations, the result is invariant to the
operator's global heading and position -- exactly EgoPoser's "global motion
decomposition" property -- so no body model, betas, root translation, or world
alignment is needed to obtain the trunk prior. (The SMPL body model is only
required to turn the pose into 3D joint *positions*, which we never use.)

The mapping SMPL-spine -> robot-trunk is not one-to-one (a human distributes
flexion over three anatomical joints; the S1 realises trunk pitch as the SUM of
three parallel sagittal hinges, per the whole-body IK notes), so the two raw
angles are scaled by tunable ``pitch_gain`` / ``yaw_gain`` (sign included) and
clamped to a safe magnitude. Downstream, the value is injected as a *low-cost*
soft QP objective (:class:`NeuralPostureTask`), so balance (ChestOverAnkle) and
hand precision always dominate an over-eager prior.

Upstream reference (reimplemented, not copied; repo has no license file):
https://github.com/eth-siplab/EgoPoser
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .feature_builder import FeatureWindow
from .rotations import sixd2matrot

# SMPL local-rotation indices within pose_body (row = smpl_joint - 1, since
# pose_body excludes the pelvis/root at joint 0).
_SPINE1_ROW = 2    # SMPL joint 3
_SPINE2_ROW = 5    # SMPL joint 6
_SPINE3_ROW = 8    # SMPL joint 9
_NECK_ROW = 11     # SMPL joint 12
_HEAD_ROW = 14     # SMPL joint 15

# Nominal SMPL bone offsets for the sagittal axis chain
# pelvis -> spine1 -> spine2 -> spine3 -> neck -> head, expressed in each PARENT
# joint's local frame (metres, SMPL rest pose, Z-up-ish upward stack). These are
# approximate mean-shape values used ONLY to draw a visualisation skeleton (see
# TrunkPrior.skeleton / EgoPoserEstimator.predict); they are NOT used by the
# retargeting math, which needs only the local rotations. Each entry is the
# offset from the named joint to its child along the chain.
_CHAIN_ROWS = [_SPINE1_ROW, _SPINE2_ROW, _SPINE3_ROW, _NECK_ROW, _HEAD_ROW]
_CHAIN_OFFSETS = np.array([
    [0.0, 0.0, 0.124],   # pelvis  -> spine1
    [0.0, 0.0, 0.137],   # spine1  -> spine2
    [0.0, 0.0, 0.056],   # spine2  -> spine3
    [0.0, 0.0, 0.211],   # spine3  -> neck
    [0.0, 0.0, 0.099],   # neck    -> head
], dtype=np.float64)

# --- FULL SMPL body skeleton (22 joints: pelvis + the 21 in pose_body) -------
# Kinematic tree + approximate mean-shape rest-pose bone offsets, for drawing the
# ENTIRE hallucinated body (not just the sagittal spine) in the viewer under
# --visualize-prior. VIS-ONLY: like _CHAIN_OFFSETS these are nominal mean-shape
# values (no per-operator betas -- the model does not output usable bone lengths),
# so the skeleton is a qualitative aid, not a metric reconstruction.
#
# Joint index = SMPL joint id (0..21). ``pose_body`` row = id - 1 (it excludes the
# pelvis/root at id 0). Axis convention matches _CHAIN_OFFSETS and the anchor frame
# used by sim_teleop: +Z up, -Y the robot/skeleton FORWARD (the anchor maps its own
# -Y forward onto the robot heading; the spine flexion also tips toward -Y), +X the
# skeleton LEFT. So anything anatomically "forward" (toes, etc.) is a NEGATIVE Y.
_SMPL_PARENTS = np.array([
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14, 16, 17, 18, 19,
], dtype=int)
# Offset from each joint's PARENT to the joint, in the parent's local frame (m).
# joint 0 (pelvis) is the root -> zero. Forward components are -Y (see above).
_SMPL_OFFSETS = np.array([
    [0.000,  0.000,  0.000],   # 0  pelvis (root)
    [0.058,  0.000, -0.082],   # 1  L hip
    [-0.058, 0.000, -0.082],   # 2  R hip
    [0.000,  0.000,  0.124],   # 3  spine1
    [0.000,  0.000, -0.383],   # 4  L knee
    [0.000,  0.000, -0.383],   # 5  R knee
    [0.000,  0.000,  0.137],   # 6  spine2
    [0.000, -0.080, -0.402],   # 7  L ankle  (-Y = slightly forward)
    [0.000, -0.080, -0.402],   # 8  R ankle
    [0.000,  0.000,  0.056],   # 9  spine3
    [0.000, -0.120, -0.052],   # 10 L foot   (-Y = toes forward)
    [0.000, -0.120, -0.052],   # 11 R foot
    [0.000,  0.000,  0.211],   # 12 neck
    [0.082,  0.000,  0.113],   # 13 L collar
    [-0.082, 0.000,  0.113],   # 14 R collar
    [0.000,  0.000,  0.099],   # 15 head
    [0.113,  0.000,  0.024],   # 16 L shoulder
    [-0.113, 0.000,  0.024],   # 17 R shoulder
    [0.258,  0.000,  0.000],   # 18 L elbow
    [-0.258, 0.000,  0.000],   # 19 R elbow
    [0.247,  0.000,  0.000],   # 20 L wrist
    [-0.247, 0.000,  0.000],   # 21 R wrist
], dtype=np.float64)
# Joints/bones that actually feed the robot retargeting (the sagittal spine that
# composes R_chest): pelvis + spine1/2/3. Everything else (legs, feet, arms,
# collars, neck, head) is drawn but UNUSED by the map -> lighter colour in the
# viewer. A bone is "mapped" iff its CHILD joint is in this set.
_SMPL_MAPPED_JOINTS = frozenset({0, 3, 6, 9})

# Change of basis: SMPL's NATIVE frame -> the viewer/anchor frame used above.
# The network's local joint rotations (pose_body) live in SMPL's native frame,
# which is **Y-up** (the retarget reads axial TWIST off the Y-euler, i.e. Y is
# the spine's long/vertical axis), X = the subject's LEFT, Z = FORWARD. The
# skeleton offsets / anchor here use **Z-up**, X = left, -Y = forward. So the FK
# must run in SMPL's frame (offsets AND rotations consistent) and only the final
# joint POSITIONS are rotated into the viewer frame by this matrix:
#     SMPL +X (left)    -> +X (left)
#     SMPL +Y (up)      -> +Z (up)
#     SMPL +Z (forward) -> -Y (forward)
# Applying the SMPL rotations directly to Z-up offsets (as an earlier version
# did) swaps the Y/Z axes for every joint: the spine survives (its flexion axis X
# is shared) but the ARMS get rotated about the wrong axis and shoot up ~90 deg.
_SMPL_TO_VIEW = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


@dataclass
class TrunkPrior:
    """A robot-frame trunk-posture prior extracted from EgoPoser output."""

    pitch: float   # target trunk forward-lean = target sum of the 3 lean joints
    yaw: float     # target waist yaw (torso_joint_4)
    # Optional visualisation skeleton: (22, 3) points of the hallucinated FULL
    # SMPL body (pelvis + the 21 pose_body joints: legs, spine, arms, neck, head)
    # in the SMPL pelvis-local frame (metres, pelvis at origin), or None when not
    # requested. For --visualize-prior. See _SMPL_PARENTS / _SMPL_MAPPED_JOINTS
    # for the tree and which joints feed the robot retargeting.
    skeleton: Optional[np.ndarray] = None


class EgoPoserEstimator:
    """Streaming EgoPoser inference + SMPL-spine -> robot-trunk retargeting.

    Construction never fails hard: if torch is unavailable or the weights are
    missing/unloadable, :meth:`available` returns ``False`` and :meth:`predict`
    returns ``None`` so the caller simply skips the neural prior. Pass
    ``allow_random=True`` (used by the offline self-checks) to build a
    randomly-initialised net when no weights are present, exercising the full
    plumbing without a checkpoint.
    """

    #: Google Drive folder holding the released checkpoints (see project README).
    WEIGHTS_DRIVE_URL = (
        "https://drive.google.com/drive/folders/"
        "1b0tc4T8z6vasy7AksYlfw--6dWfzH8wT"
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        *,
        window_size: int = 80,
        embed_dim: int = 256,
        num_layer: int = 3,
        nhead: int = 8,
        spatial_normalization: bool = True,
        pitch_gain: float = 1.0,
        yaw_gain: float = 1.0,
        max_pitch: float = 1.2,
        max_yaw: float = 1.0,
        align_R: Optional[np.ndarray] = None,
        device: str = "cpu",
        allow_random: bool = False,
    ):
        self.window_size = int(window_size)
        self.pitch_gain = float(pitch_gain)
        self.yaw_gain = float(yaw_gain)
        self.max_pitch = float(max_pitch)
        self.max_yaw = float(max_yaw)
        self.device = device
        self._net = None
        self._torch = None
        self._reason = ""   # human-readable why-unavailable, for logging

        self._window = FeatureWindow(window_size=self.window_size, align_R=align_R)

        try:
            import torch  # lazy: only when the prior is enabled
            from .network import AvatarNet
        except Exception as e:  # torch not installed in this env
            self._reason = f"torch unavailable ({type(e).__name__}: {e})"
            return
        self._torch = torch

        net = AvatarNet(
            input_dim=60,   # 54 raw + 6 temporal-delta appended in forward()
            num_layer=num_layer,
            embed_dim=embed_dim,
            nhead=nhead,
            spatial_normalization=spatial_normalization,
            shape_estimation=False,
        )

        loaded = False
        if weights_path and os.path.isfile(weights_path):
            try:
                # The released checkpoints are plain tensor state_dicts, so the
                # safe weights_only loader works; fall back for older torch.
                try:
                    state = torch.load(weights_path, map_location=device,
                                       weights_only=True)
                except TypeError:
                    state = torch.load(weights_path, map_location=device)
                if isinstance(state, dict) and "params" in state:
                    state = state["params"]
                net.load_state_dict(state, strict=False)
                loaded = True
            except Exception as e:
                self._reason = f"failed to load weights '{weights_path}' ({e})"
                return
        elif weights_path:
            self._reason = f"weights not found at '{weights_path}'"
            if not allow_random:
                return
        else:
            self._reason = "no weights_path configured"
            if not allow_random:
                return

        self._weights_loaded = loaded
        net.to(device)
        net.eval()
        self._net = net

    # -- status --------------------------------------------------------------
    def available(self) -> bool:
        """True if inference can run (torch present and a net was built)."""
        return self._net is not None

    @property
    def reason(self) -> str:
        """Why the estimator is unavailable (empty string if it is available)."""
        return "" if self.available() else self._reason

    def reset(self) -> None:
        """Clear the rolling window (call on (re)calibration)."""
        self._window.reset()

    # -- inference -----------------------------------------------------------
    def predict(
        self, head_T: np.ndarray, left_T: np.ndarray, right_T: np.ndarray,
        with_skeleton: bool = False,
    ) -> Optional[TrunkPrior]:
        """Push one frame and return the current trunk prior, or ``None``.

        ``None`` is returned when the estimator is unavailable (torch/weights
        missing) -- the caller then leaves the neural task untouched for this
        tick, so teleop degrades gracefully to the non-neural behaviour.

        When ``with_skeleton`` is True the returned prior also carries a (6, 3)
        ``skeleton`` of the hallucinated sagittal chain (for --visualize-prior);
        it is left ``None`` otherwise to save the small FK cost.
        """
        if self._net is None:
            return None
        self._window.push(head_T, left_T, right_T)

        torch = self._torch
        batch = self._window.as_batch()          # (1, W, 54) float32
        with torch.no_grad():
            x = {
                "sparse_input": torch.from_numpy(batch).to(self.device),
                "fov_l": None,
                "fov_r": None,
            }
            out = self._net(x)
            pose_body = out["pose_body"].detach().cpu().numpy().reshape(-1)

        return self._retarget(pose_body, with_skeleton=with_skeleton)

    # -- retargeting ---------------------------------------------------------
    def _retarget(self, pose_body: np.ndarray,
                  with_skeleton: bool = False) -> TrunkPrior:
        """SMPL local spine rotations -> (trunk pitch, waist yaw), scaled/clamped."""
        rows = pose_body.reshape(21, 6)
        R_s1 = sixd2matrot(rows[_SPINE1_ROW])
        R_s2 = sixd2matrot(rows[_SPINE2_ROW])
        R_s3 = sixd2matrot(rows[_SPINE3_ROW])
        R_chest = R_s1 @ R_s2 @ R_s3

        from scipy.spatial.transform import Rotation

        flexion, twist, _lateral = Rotation.from_matrix(R_chest).as_euler("XYZ")

        pitch = float(np.clip(self.pitch_gain * flexion, -self.max_pitch, self.max_pitch))
        yaw = float(np.clip(self.yaw_gain * twist, -self.max_yaw, self.max_yaw))
        skel = self._skeleton(rows) if with_skeleton else None
        return TrunkPrior(pitch=pitch, yaw=yaw, skeleton=skel)

    def _skeleton(self, rows: np.ndarray) -> np.ndarray:
        """Forward-kinematics the FULL SMPL body -> (22, 3) pelvis-local points.

        Walks the whole SMPL kinematic tree (pelvis + the 21 joints in
        ``pose_body``: legs, spine, arms, neck, head) using each joint's predicted
        local rotation and the nominal mean-shape rest offsets (_SMPL_OFFSETS /
        _SMPL_PARENTS). Used only for the --visualize-prior wireframe, so mean-shape
        offsets are fine (this is a *qualitative* confirmation aid, not a metric
        reconstruction -- the model outputs no usable bone lengths). Returns 22
        points (joint id order) with the pelvis at the origin, in the viewer frame.

        FRAME: the network's local rotations are in SMPL's native **Y-up** frame,
        so the FK is done entirely in that frame -- both the offsets and the
        rotations. ``_SMPL_OFFSETS`` are authored in the viewer's Z-up frame for
        readability, so each is mapped back into SMPL's frame (``_SMPL_TO_VIEW.T``)
        before use; the resulting joint POSITIONS are then rotated into the viewer
        frame (``_SMPL_TO_VIEW``) at the end. Doing FK with SMPL rotations but
        Z-up offsets (an earlier bug) swapped the Y/Z axes and threw the arms up
        ~90 deg (the spine survived because its flexion axis X is shared).

        Standard SMPL FK: pos[child] = pos[parent] + R_global[parent] @ offset,
        then R_global[child] = R_global[parent] @ R_local[child]. The pelvis
        (root) global rotation is identity; each other joint's local rotation is
        pose_body row (id - 1).
        """
        n = len(_SMPL_PARENTS)
        pts = np.zeros((n, 3))           # in SMPL native (Y-up) frame
        R_global = [np.eye(3)] * n
        smpl_from_view = _SMPL_TO_VIEW.T   # map the Z-up authored offsets -> SMPL
        for jid in range(n):
            par = _SMPL_PARENTS[jid]
            if par < 0:                      # pelvis / root
                pts[jid] = np.zeros(3)
                R_global[jid] = np.eye(3)
            else:
                offset = smpl_from_view @ _SMPL_OFFSETS[jid]
                pts[jid] = pts[par] + R_global[par] @ offset
                # pose_body has no entry for the root, so row = id - 1.
                R_global[jid] = R_global[par] @ sixd2matrot(rows[jid - 1])
        # Rotate the finished skeleton from SMPL's Y-up frame into the viewer frame.
        return pts @ _SMPL_TO_VIEW.T

    # -- utilities -----------------------------------------------------------
    @staticmethod
    def download_weights(dest_dir: str) -> str:
        """Fetch the released EgoPoser checkpoint folder via ``gdown``.

        Returns the destination directory. Requires network access and the
        ``gdown`` package (``pip install gdown``); raises with a clear message
        otherwise. Intended for one-off setup, not the teleop hot path.
        """
        try:
            import gdown
        except Exception as e:  # pragma: no cover - setup-time only
            raise RuntimeError(
                "gdown is required to download the EgoPoser weights: "
                "`pip install gdown`. Alternatively download them manually from "
                f"{EgoPoserEstimator.WEIGHTS_DRIVE_URL}"
            ) from e
        os.makedirs(dest_dir, exist_ok=True)
        gdown.download_folder(
            EgoPoserEstimator.WEIGHTS_DRIVE_URL, output=dest_dir, quiet=False
        )
        return dest_dir
