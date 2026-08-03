"""Orthogonal rotations: Haar via Householder QR, and randomized Hadamard.

`haar_rotation*` mirrors `remex.rotation.haar_rotation`:
  1. Sample A ~ N(0, 1) of shape (d, d)
  2. QR decompose
  3. Sign-correct: Q[:, j] *= sign(R[j, j])

The resulting Q is uniformly distributed on O(d) (Mezzadri 2007).

`rht_rotation` mirrors `remex.rotation.rht_rotation` — a randomized
Hadamard transform materialized as a dense (d, d) matrix. Not Haar, but
it delivers the incoherence the Lloyd-Max codebook actually depends on,
and it costs O(d^2 log d) to build instead of the QR's O(d^3).
"""

from std.math import sqrt
from std.memory import alloc, UnsafePointer
from src.rng import Xoshiro256pp
from src.rng_numpy import NumpyNormalRNG, PCG64
from src.matrix import Matrix, MatrixF64


def _norm2(p: UnsafePointer[Float64, MutExternalOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += p[i] * p[i]
    return sqrt(s)


def _householder_qr(mut A: MatrixF64, mut Q: MatrixF64):
    """In-place QR. After call: A holds R (upper triangular), Q holds Q."""
    var n = A.rows
    # Initialize Q = I
    for i in range(n):
        for j in range(n):
            Q.set(i, j, Float64(1.0) if i == j else Float64(0.0))

    for k in range(n - 1):
        # Compute alpha = -sign(A[k,k]) * ||A[k:, k]||
        var col_norm_sq: Float64 = 0.0
        for i in range(k, n):
            var v = A.get(i, k)
            col_norm_sq += v * v
        var col_norm = sqrt(col_norm_sq)
        if col_norm == 0.0:
            continue
        var sign_akk: Float64 = -1.0 if A.get(k, k) < 0.0 else 1.0
        var alpha = -sign_akk * col_norm

        # v = A[k:, k] - alpha * e1
        # store v in a temp column-vector
        var v_len = n - k
        var vp = alloc[Float64](v_len)
        for i in range(v_len):
            vp[i] = A.get(k + i, k)
        vp[0] -= alpha

        var v_norm = _norm2(vp, v_len)
        if v_norm == 0.0:
            vp.free()
            continue
        for i in range(v_len):
            vp[i] /= v_norm

        # Apply H = I - 2 v v^T to A[k:, k:] from the left:
        # for each column j >= k: A[k:, j] -= 2 v (v^T A[k:, j])
        for j in range(k, n):
            var dot: Float64 = 0.0
            for i in range(v_len):
                dot += vp[i] * A.get(k + i, j)
            var two_dot = 2.0 * dot
            for i in range(v_len):
                A.set(k + i, j, A.get(k + i, j) - two_dot * vp[i])

        # Apply H to Q from the right: Q[:, k:] -= 2 (Q[:, k:] v) v^T
        for i in range(n):
            var dot: Float64 = 0.0
            for j in range(v_len):
                dot += Q.get(i, k + j) * vp[j]
            var two_dot = 2.0 * dot
            for j in range(v_len):
                Q.set(i, k + j, Q.get(i, k + j) - two_dot * vp[j])

        vp.free()


def haar_rotation(d: Int, seed: UInt64) -> Matrix:
    """Haar-distributed (d, d) orthogonal matrix using xoshiro256++ + Marsaglia.

    NOT bit-identical to Python `Quantizer(seed=S).R` — produces a valid
    Haar sample but from a different Gaussian stream. Use
    `haar_rotation_numpy` for byte parity with Python.
    """
    var rng = Xoshiro256pp(seed)
    var A = MatrixF64(d, d)
    for i in range(d):
        for j in range(d):
            A.set(i, j, rng.next_normal())

    var Q = MatrixF64(d, d)
    _householder_qr(A, Q)

    # Sign-correct: Q[:, j] *= sign(R[j, j])
    for j in range(d):
        var diag = A.get(j, j)
        if diag < 0.0:
            for i in range(d):
                Q.set(i, j, -Q.get(i, j))

    return Q.to_float32()


def haar_rotation_numpy(d: Int, seed: UInt64) -> Matrix:
    """Haar-distributed (d, d) orthogonal matrix matching Python `Quantizer(seed=S)`.

    Generates G via the NumPy-compatible RNG (PCG64 + SeedSequence + Ziggurat),
    then runs the same Householder QR + Mezzadri sign correction as
    `haar_rotation`. End-to-end this matches Python's `remex.haar_rotation(d, seed)`
    bit-for-bit at float32 (modulo rare libm tail-rejection rounding).
    """
    var rng = NumpyNormalRNG(seed)
    var A = MatrixF64(d, d)
    # NumPy fills row-major: A[i, j] is the (i*d + j)-th draw.
    for i in range(d):
        for j in range(d):
            A.set(i, j, rng.next_normal())

    var Q = MatrixF64(d, d)
    _householder_qr(A, Q)

    for j in range(d):
        var diag = A.get(j, j)
        if diag < 0.0:
            for i in range(d):
                Q.set(i, j, -Q.get(i, j))

    return Q.to_float32()


def largest_pow2_divisor(d: Int) -> Int:
    """Largest power of two dividing `d` — the FWHT block size."""
    var b = 1
    while d % (b * 2) == 0:
        b *= 2
    return b


def rht_rounds(d: Int, B: Int) -> Int:
    """Rounds of (permute -> sign flip -> FWHT): smallest k with B**k >= d.

    Python spells this `max(2, ceil(log(d) / log(B)))`. The two agree for
    every even d (checked exhaustively to 20000): `B**k == d` forces
    `B == d`, which takes the k = 1 branch, so in the other branch the log
    quotient never lands on an integer and no rounding boundary is in
    play. The integer form is used here because it cannot drift with libm.
    """
    if B == d:
        return 1
    var k = 1
    var p = B
    while p < d:
        p *= B
        k += 1
    return k


def rht_rotation(d: Int, seed: UInt64) raises -> Matrix:
    """Randomized Hadamard rotation, byte-identical to Python `rht_rotation`.

    Applies `rounds` of (permute -> sign flip -> block-diagonal FWHT) to
    the identity, with block size the largest power of two dividing `d`.
    Every draw comes off the same NumPy PCG64 stream Python uses, in the
    same order: `Generator.permutation` consumes `random_interval`
    (masked rejection) and `Generator.choice` consumes
    `bounded_lemire_uint32`. All arithmetic is float32, matching NumPy's
    float32 `Y`.
    """
    var B = largest_pow2_divisor(d)
    if B < 2:
        raise Error(
            String("d=") + String(d) + String(
                " is odd; the randomized Hadamard construction needs an "
                "even dimension. Use the Haar rotation."
            )
        )
    var rounds = rht_rounds(d, B)
    var n_blocks = d // B
    var rng = PCG64(seed)

    var Y = Matrix(d, d)
    for i in range(d):
        Y.set(i, i, Float32(1.0))
    # float64 reciprocal-sqrt with a single rounding to float32 — matches
    # NumPy's `np.float32(1.0 / math.sqrt(B))`. Rounding sqrt to float32
    # first instead would be a real difference in general (it disagrees for
    # 1349 of the integers in [2, 5000]) but not here, because B is always a
    # power of two and the two forms agree on every one of those up to 2^30.
    # No test can tell them apart; keep this spelling anyway, so the line
    # reads the same as the Python it has to match.
    var scale = Float32(Float64(1.0) / sqrt(Float64(B)))

    var perm = alloc[Int32](d)
    var sign = alloc[Float32](d)
    var tmp = alloc[Float32](d)

    for _round in range(rounds):
        # rng.permutation(d): arange, then Fisher-Yates over random_interval.
        for i in range(d):
            perm[i] = Int32(i)
        var i = d - 1
        while i > 0:
            var j = Int(rng.random_interval(UInt64(i)))
            var swap = perm[i]
            perm[i] = perm[j]
            perm[j] = swap
            i -= 1
        # rng.choice([-1.0, 1.0], size=d) is rng.integers(0, 2, size=d) remapped.
        for k in range(d):
            if rng.bounded_lemire_u32(UInt32(1)) == UInt32(0):
                sign[k] = Float32(-1.0)
            else:
                sign[k] = Float32(1.0)

        for row in range(d):
            var base = row * d
            # Y = Y[:, perm] * sign
            for j in range(d):
                tmp[j] = Y.data[base + Int(perm[j])] * sign[j]
            for j in range(d):
                Y.data[base + j] = tmp[j]
            # Unnormalized FWHT over each contiguous run of B.
            for blk in range(n_blocks):
                var off = base + blk * B
                var h = 1
                while h < B:
                    var m = 0
                    while m < B:
                        for t in range(h):
                            var a = Y.data[off + m + t]
                            var b = Y.data[off + m + h + t]
                            Y.data[off + m + t] = a + b
                            Y.data[off + m + h + t] = a - b
                        m += 2 * h
                    h *= 2
            for j in range(d):
                Y.data[base + j] = Y.data[base + j] * scale

    perm.free()
    sign.free()
    tmp.free()
    return Y^


def matvec(M: Matrix, x: UnsafePointer[Float32, MutExternalOrigin],
           mut out_buf: UnsafePointer[Float32, MutExternalOrigin]):
    """y = M @ x, where M is (rows, cols) and x is length cols."""
    for i in range(M.rows):
        var s: Float32 = Float32(0.0)
        var base = i * M.cols
        for j in range(M.cols):
            s += M.data[base + j] * x[j]
        out_buf[i] = s


def matvec_T(M: Matrix, x: UnsafePointer[Float32, MutExternalOrigin],
             mut out_buf: UnsafePointer[Float32, MutExternalOrigin]):
    """y = M.T @ x. (i.e. dot product of x with each column of M.)"""
    for j in range(M.cols):
        out_buf[j] = Float32(0.0)
    for i in range(M.rows):
        var base = i * M.cols
        var xi = x[i]
        for j in range(M.cols):
            out_buf[j] += M.data[base + j] * xi
