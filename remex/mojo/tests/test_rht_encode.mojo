"""Encode parity for the RHT rotation, through both routes into a Quantizer.

`from_rht_seed` loads no `.params` file: the Quantizer is rebuilt in Mojo
from (d, bits, seed) alone, so a byte-identical `.pq` proves the whole
seed -> PCG64 -> permutation -> sign -> FWHT -> codebook -> encode chain
agrees with Python. `load_params` is the other route, reading R off disk,
and has to land on the same bytes.

    python3 remex/mojo/tests/build_rht_fixture.py
    mojo run -I . tests/test_rht_encode.mojo

What this cannot catch, measured rather than guessed. It runs one case,
n=64 d=384 bits=4, and two known-bads that `test_rht` rejects survive it:
a hardcoded round count (d=384 already takes 2 rounds, so nothing moves)
and an inverted sign mapping (at an even round count the inversion
cancels and R is unchanged). Neither is a defect this file can see. The
rotation gate is `test_rht`; do not treat a green run here as covering R.
Also untested here: bit widths other than 4 — at 8 bits the codebook's
erf drift makes ~0.1-0.2% of indices differ, as it does for Haar (see
README), so this would go red for a reason that is not the rotation.
"""

from std.testing import assert_equal, assert_true
from std.memory import alloc, UnsafePointer
from src.codebook import Codebook
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.params_format import load_params
from src.pq_format import load_pq, PqVectors
from src.quantizer import Quantizer, encode_batch
from src.packing import pack


def _check_encode(q: Quantizer, X_buf: UnsafePointer[Float32, MutExternalOrigin],
                  n: Int, d: Int, bits: Int, expected: PqVectors,
                  label: String) raises:
    var indices = alloc[UInt8](n * d)
    var norms = alloc[Float32](n)
    encode_batch(q, X_buf, n, indices, norms)

    var packed = alloc[UInt8](expected.packed_bytes)
    pack(indices, n * d, bits, packed)

    var max_norm_diff: Float32 = Float32(0.0)
    for i in range(n):
        var diff = norms[i] - expected.norms[i]
        if diff < Float32(0.0):
            diff = -diff
        if diff > max_norm_diff:
            max_norm_diff = diff

    var n_diff: Int = 0
    for i in range(expected.packed_bytes):
        if packed[i] != expected.packed_indices[i]:
            n_diff += 1

    print("  ", label, "— max norm diff:", max_norm_diff,
          " packed-byte differences:", n_diff, "of", expected.packed_bytes)
    assert_true(max_norm_diff < Float32(1e-5))
    assert_equal(n_diff, 0)

    indices.free()
    norms.free()
    packed.free()


def main() raises:
    var X = load_npy_2d_f32(String("/tmp/_rht_X.npy"))
    var expected = load_pq(String("/tmp/_rht_ref.pq"))
    var d = X.cols
    var n = X.rows
    var bits = expected.bits
    var seed = UInt64(7)        # must match build_rht_fixture.ENC_SEED

    # Copy X into a fresh buffer to dodge borrow weirdness on Npy2D.data
    # crossing a function boundary.
    var X_buf = alloc[Float32](n * d)
    for i in range(n):
        for j in range(d):
            X_buf[i * d + j] = X.get(i, j)

    var q_seed = Quantizer.from_rht_seed(d, bits, seed)
    _check_encode(q_seed, X_buf, n, d, bits, expected,
                  String("--seed --rotation rht"))

    var R = Matrix(d, d)
    var cb = Codebook(bits)
    load_params(String("/tmp/_rht.params"), R, cb)
    var q_params = Quantizer(R^, cb^, d, bits, seed)
    _check_encode(q_params, X_buf, n, d, bits, expected, String("--params"))

    X_buf.free()
    print("[test_rht_encode] parity ok — both --seed --rotation rht and "
          "--params match Python")
