"""Rotation representation helpers for the EgoPoser prior (NumPy only).

EgoPoser (and its predecessor AvatarPoser) represent joint rotations in the
**6D continuous** form of Zhou et al. (CVPR 2019): the first two columns of the
3x3 rotation matrix, stacked into a 6-vector. This is the representation used
for both the network *input* (global rotations + rotation velocities of the
head and hands) and the network *output* (``root_orient`` 6D + ``pose_body``
21x6D local joint rotations).

We reimplement the exact conventions of the upstream ``utils/utils_transform.py``
(``matrot2sixd`` / ``sixd2matrot``) in pure NumPy so the whole prior runs in the
``AVP`` conda env without pulling in ``human_body_prior`` / ``pytorch3d``:

    matrot2sixd(R) = [R[:,0], R[:,1]]                  (exact, used for INPUT)
    sixd2matrot(d) : v1=d[:3], v2=d[3:6]               (used for OUTPUT decode)

On decode we additionally Gram-Schmidt orthonormalise the two basis vectors
(the standard Zhou reconstruction). The upstream ``sixd2matrot`` skips the
normalisation because a *trained* network already emits near-orthonormal 6D;
orthonormalising is strictly safer and agrees to machine precision on valid
output, so it cannot change behaviour on real weights while guarding against a
randomly-initialised net during offline self-checks.

Upstream reference (no license file in the repo; reimplemented, not copied):
https://github.com/eth-siplab/EgoPoser -- utils/utils_transform.py
"""

from __future__ import annotations

import numpy as np


def matrot2sixd(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 6) continuous 6D (first two cols)."""
    R = np.asarray(R, dtype=np.float64)
    # First two columns: R[..., :, 0] and R[..., :, 1].
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


def sixd2matrot(d6: np.ndarray) -> np.ndarray:
    """(..., 6) continuous 6D -> (..., 3, 3) rotation (Gram-Schmidt decode)."""
    d6 = np.asarray(d6, dtype=np.float64)
    v1 = d6[..., 0:3]
    v2 = d6[..., 3:6]
    b1 = _normalize(v1)
    # Remove the b1 component from v2, then normalise -> orthonormal b2.
    proj = np.sum(b1 * v2, axis=-1, keepdims=True) * b1
    b2 = _normalize(v2 - proj)
    b3 = np.cross(b1, b2)
    # Columns [b1 | b2 | b3].
    return np.stack([b1, b2, b3], axis=-1)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) axis-angle (rotation vector), via scipy."""
    from scipy.spatial.transform import Rotation

    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    rv = Rotation.from_matrix(_orthonormalize(flat)).as_rotvec()
    return rv.reshape(R.shape[:-2] + (3,))


def compose(*mats: np.ndarray) -> np.ndarray:
    """Right-multiply a chain of (3, 3) rotations: compose(A, B, C) = A @ B @ C."""
    out = np.asarray(mats[0], dtype=np.float64)
    for m in mats[1:]:
        out = out @ np.asarray(m, dtype=np.float64)
    return out


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation to each (3, 3) via SVD (handles reflections)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    dets = np.linalg.det(Rn)
    flip = dets < 0
    if np.any(flip):
        U[flip, :, -1] *= -1
        Rn = U @ Vt
    return Rn
