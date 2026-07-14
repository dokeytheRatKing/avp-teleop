"""EgoPoser network (clean reimplementation of ``AvatarNet``).

This is a from-scratch reimplementation of the EgoPoser network described in
"EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and
Intermittent Observations Everywhere" (Jiang et al., ECCV 2024). The upstream
repository (https://github.com/eth-siplab/EgoPoser) ships no license file, so
we do NOT copy its code; instead we reconstruct the published architecture with
**identical module attribute names and layer types** so that the authors'
released checkpoint (a ``state_dict`` under the ``"params"`` key) loads directly
via ``load_state_dict``.

Architecture (from the paper + released config ``options/test_egoposer.yaml``):

    input_dim = 60   (54 raw sensor dims + 6 temporal-delta dims appended in
                      forward(); the caller/feature builder supplies 54)
    embed_dim = 256, nhead = 8, num_layer = 3, spatial_normalization = True

    linear_embedding      : Linear(input_dim -> embed_dim)      (slow + fast share)
    linear_embedding_fast : Linear(input_dim -> embed_dim//2)   (unused at test;
                            kept only so the checkpoint's keys match 1:1)
    transformer_encoder   : TransformerEncoder(num_layer x
                            TransformerEncoderLayer(embed_dim, nhead))
    global_orientation_decoder : MLP(embed_dim -> embed_dim -> 6)
    joint_rotation_decoder     : MLP(embed_dim -> embed_dim -> 126)   (21 x 6D)
    shape_decoder              : MLP(embed_dim -> embed_dim -> 16)    (optional)

The SlowFast fusion, spatial normalisation and temporal-delta augmentation are
reproduced exactly as in the paper's description so that a window of shape
``(batch, window, 54)`` produces ``root_orient`` (batch, 6) and ``pose_body``
(batch, 126) for the *last* frame of each window -- the streaming-inference
contract EgoPoser uses.

``torch`` is imported at module load, so this module is only ever imported
lazily (from :mod:`avp_teleop_upper_body.egoposer.estimator`) when the neural
prior is actually enabled -- the default teleop path never touches torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AvatarNet(nn.Module):
    """EgoPoser regressor: sparse head/hand window -> SMPL local rotations.

    Parameters mirror the released config so ``**network_arguments`` from the
    upstream YAML can be splatted in unchanged.
    """

    def __init__(
        self,
        input_dim: int = 60,
        num_layer: int = 3,
        embed_dim: int = 256,
        nhead: int = 8,
        spatial_normalization: bool = True,
        shape_estimation: bool = False,
    ):
        super().__init__()

        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        # Present in the checkpoint but unused at inference (the released model
        # embeds both slow and fast streams with ``linear_embedding``); kept so
        # state_dict keys line up 1:1.
        self.linear_embedding_fast = nn.Linear(input_dim, embed_dim // 2)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layer
        )

        self.global_orientation_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 6),
        )
        self.joint_rotation_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 126),
        )
        self.shape_decoder = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 16),
            )
            if shape_estimation
            else None
        )

        self.spatial_normalization = spatial_normalization

    def forward(self, x: dict) -> dict:
        """Run the streaming forward pass.

        ``x['sparse_input']`` is ``(batch, window, 54)``. ``fov_l`` / ``fov_r``
        (bool visibility masks) are accepted for API parity but the deployed
        model runs with full hand visibility, so we simply ignore missing masks
        (equivalent to the upstream ``full_hand_visibility=True`` path).
        """
        input_tensor = x["sparse_input"]

        if self.spatial_normalization:
            # Make the horizontal (x, y) positions of head/L/R relative to the
            # head, keeping the global vertical component. Indices per the
            # upstream feature layout: head pos = 36:38, L = 39:41, R = 42:44.
            head_h = input_tensor[..., 36:38].clone().detach()
            input_tensor = input_tensor.clone()
            input_tensor[..., 36:38] = input_tensor[..., 36:38] - head_h
            input_tensor[..., 39:41] = input_tensor[..., 39:41] - head_h
            input_tensor[..., 42:44] = input_tensor[..., 42:44] - head_h

        # Temporal normalisation: append each horizontal position's delta from
        # the FIRST frame in the window (adds 6 dims -> 60).
        d0 = input_tensor[..., 36:38] - input_tensor[..., [0], 36:38]
        d1 = input_tensor[..., 39:41] - input_tensor[..., [0], 39:41]
        d2 = input_tensor[..., 42:44] - input_tensor[..., [0], 42:44]
        input_tensor = torch.cat([input_tensor, d0, d1, d2], dim=-1)

        # SlowFast: a fast stream (recent half) + a slow stream (every 2nd
        # frame), both embedded and summed.
        x_fast = input_tensor[:, -input_tensor.shape[1] // 2:, ...]
        x_slow = input_tensor[:, ::2, ...]
        x_fast = self.linear_embedding(x_fast)
        x_slow = self.linear_embedding(x_slow)

        # The two streams generally differ in length; the released model sums
        # them after truncating to the shorter one (both cover the window).
        n = min(x_fast.shape[1], x_slow.shape[1])
        feat = x_fast[:, -n:, :] + x_slow[:, -n:, :]

        feat = feat.permute(1, 0, 2)          # (seq, batch, embed)
        feat = self.transformer_encoder(feat)
        feat = feat.permute(1, 0, 2)          # (batch, seq, embed)
        feat = feat[:, -1]                    # last token -> current-frame pose

        return {
            "root_orient": self.global_orientation_decoder(feat),
            "pose_body": self.joint_rotation_decoder(feat),
            "betas": self.shape_decoder(feat) if self.shape_decoder is not None else None,
        }
