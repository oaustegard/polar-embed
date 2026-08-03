"""Random orthogonal rotation via Haar-distributed matrices.

Uses an explicit Householder QR with a fixed reflector convention so the
output is bit-reproducible across BLAS builds and matches the Mojo port's
encode path byte-for-byte (issue #40). The previous implementation
delegated to ``np.linalg.qr``, which calls LAPACK ``dgeqrf``; LAPACK QR
is not bit-deterministic across MKL/OpenBLAS builds or threading modes,
which made `--seed`-based reproducibility impossible end-to-end.
"""

import math

import numpy as np


def _householder_qr(A: np.ndarray) -> np.ndarray:
    """In-place Householder QR; returns Q. After the call A holds R.

    Reflector convention (must match ``remex/mojo/src/rotation.mojo``):
      - ``alpha = -sign(A[k,k]) * ||A[k:, k]||`` with ``sign(0) = +1``
      - ``v = A[k:, k] - alpha * e_1``, normalized
      - Apply ``H = I - 2 v v^T`` to ``A[k:, k:]`` from the left and to
        ``Q[:, k:]`` from the right
    """
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)
    for k in range(n - 1):
        col = A[k:, k]
        col_norm = float(np.sqrt(np.dot(col, col)))
        if col_norm == 0.0:
            continue
        sign_akk = -1.0 if A[k, k] < 0.0 else 1.0
        alpha = -sign_akk * col_norm

        v = col.copy()
        v[0] -= alpha
        v_norm = float(np.sqrt(np.dot(v, v)))
        if v_norm == 0.0:
            continue
        v /= v_norm

        # H = I - 2 v v^T applied to A[k:, k:] from the left
        # A[k:, k:] -= 2 * outer(v, v.T @ A[k:, k:])
        sub = A[k:, k:]
        sub -= 2.0 * np.outer(v, v @ sub)

        # Same H applied to Q[:, k:] from the right
        # Q[:, k:] -= 2 * outer(Q[:, k:] @ v, v)
        Qsub = Q[:, k:]
        Qsub -= 2.0 * np.outer(Qsub @ v, v)
    return Q


def haar_rotation(d: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Haar-distributed random orthogonal matrix.

    Pipeline (matches the Mojo port for byte-identical encode parity):
      1. Sample G ~ N(0, 1) of shape (d, d) in float64 via NumPy's
         default RNG (PCG64 + SeedSequence + Ziggurat).
      2. Run explicit Householder QR with the fixed reflector convention
         documented in ``_householder_qr``.
      3. Apply Mezzadri sign correction so Q is uniformly distributed
         on O(d) (the Haar measure).
      4. Cast Q to float32.

    Args:
        d: Matrix dimension.
        seed: Random seed for reproducibility.

    Returns:
        Q: (d, d) float32 orthogonal matrix, Q @ Q.T ~ I.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))  # float64 by default
    Q = _householder_qr(A)

    # Mezzadri sign correction: ensure diag(R) > 0 so Q is Haar-distributed.
    diag = np.diagonal(A).copy()
    sign_flip = diag < 0.0
    if sign_flip.any():
        Q[:, sign_flip] = -Q[:, sign_flip]

    return Q.astype(np.float32)


def _fwht_inplace(y: np.ndarray) -> None:
    """Normalized fast Walsh-Hadamard transform along the last axis, in place.

    ``y`` must be (..., B) with B a power of two.  Normalized by 1/sqrt(B) by
    the caller, which makes the transform orthogonal *and* symmetric — it is
    its own inverse.
    """
    B = y.shape[-1]
    h = 1
    while h < B:
        y2 = y.reshape(-1, B // (2 * h), 2, h)
        a = y2[:, :, 0, :].copy()
        b = y2[:, :, 1, :]
        y2[:, :, 0, :] = a + b
        y2[:, :, 1, :] = a - b
        h *= 2


def _largest_pow2_divisor(d: int) -> int:
    b = 1
    while d % (b * 2) == 0:
        b *= 2
    return b


def rht_rotation(d: int, seed: int = 42) -> np.ndarray:
    """Randomized Hadamard rotation, materialized as a dense (d, d) matrix.

    Same contract as ``haar_rotation``: a deterministic-from-seed float32
    orthogonal matrix.  It is *not* Haar-distributed — it is a randomized
    Hadamard transform, which is the standard incoherence-processing rotation
    in the QuIP#/HIGGS lineage and is what the coordinates-become-Gaussian
    argument actually needs.  Measured indistinguishable from Haar on
    retrieval recall (-0.0001 +/- 0.0013 pooled over 3 corpora x 6 bit widths
    x 5 seeds; oaustegard/experiments#11).

    Why it exists: ``haar_rotation`` runs an explicit Householder QR, which is
    O(d^3) and measured at O(d^3.08) here — 1.8 s at d=768, 11.4 s at d=1536,
    150 s at d=3072.  remex is embedding-agnostic and d is a free parameter,
    so that is a real cost at mainstream embedding sizes.  Building the RHT
    costs O(d^2 log d).

    Construction, for ANY d rather than powers of two only: rounds of
    (permute -> sign flip -> block-diagonal FWHT) with block size the largest
    power of two dividing d, enough rounds to mix every coordinate.  Padding d
    up to a power of two would change the codec's dimension, so it is not an
    option here.  Materialized by applying the transform to the identity, one
    batched pass.

    NOTE: the Mojo port regenerates the rotation from the seed
    (``mojo/src/quantizer.mojo``), and only knows the Haar construction.  A
    Quantizer built with this rotation therefore cannot be mirrored by the
    Mojo binary; ``pq_format.save_params`` refuses it rather than writing
    parameters that would silently disagree.

    Args:
        d: Matrix dimension.
        seed: Random seed. Same seed gives the same matrix.

    Returns:
        Q: (d, d) float32 orthogonal matrix, Q @ Q.T ~ I.
    """
    B = _largest_pow2_divisor(d)
    if B < 2:
        raise ValueError(
            f"d={d} is odd; the randomized Hadamard construction needs an "
            f"even dimension. Use rotation='haar'."
        )
    rng = np.random.default_rng(seed)
    rounds = 1 if B == d else max(2, math.ceil(math.log(d) / math.log(B)))

    # Apply the transform to the identity, one batched pass over all d rows.
    Y = np.eye(d, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(B))
    for _ in range(rounds):
        perm = rng.permutation(d)
        sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=d)
        Y = Y[:, perm] * sign
        Y = np.ascontiguousarray(Y.reshape(d, d // B, B))
        _fwht_inplace(Y)
        Y = Y.reshape(d, d) * scale
    return Y
