"""Randomized Hadamard rotation: byte parity with Python, plus orthogonality.

Fixtures come from `tests/build_rht_fixture.py`:

    python3 remex/mojo/tests/build_rht_fixture.py
    mojo run -I . tests/test_rht.mojo

The anchor is NumPy's `remex.rotation.rht_rotation` — an independent
implementation this code did not produce. Byte equality, not tolerance:
the whole claim is that the two constructions agree exactly.

Known-bads this has been shown to reject (each applied to
`src/rotation.mojo` / `src/rng_numpy.mojo`, gate confirmed red):
permutation drawing Lemire instead of `random_interval`; `next_u32`
dropping the `has_uint32` buffer; signs drawn before the permutation;
a fixed round count; the FWHT butterfly halves swapped; the permutation
applied as its inverse; the sign mapping inverted.

What it cannot catch:
  - A shared misreading of the construction. The anchor is remex's own
    Python, so if `rht_rotation` there is wrong about what an RHT is,
    both sides are wrong together and this stays green. `test_orthogonal`
    and `test_incoherence` are the independent checks on that, and they
    are properties, not byte comparisons.
  - Anything at d values outside {36, 96, 128, 384, 768} — notably d where
    the FWHT block is 2, and d above 768.
  - The `polarquant` CLI wiring. This tests the function, not `--rotation`
    argument parsing.
"""

from std.math import sqrt
from std.testing import assert_true
from std.memory import alloc
from src.matrix import Matrix
from src.npy import load_npy_2d_f32
from src.rotation import (
    largest_pow2_divisor, rht_rotation, rht_rounds, matvec,
)


def _abs(x: Float32) -> Float32:
    return -x if x < Float32(0.0) else x


def _assert_bit_identical(R: Matrix, path: String, label: String) raises:
    var want = load_npy_2d_f32(path)
    assert_true(want.rows == R.rows and want.cols == R.cols)
    var mismatches = 0
    var max_diff: Float32 = Float32(0.0)
    for i in range(R.rows * R.cols):
        if R.data[i] != want.data[i]:
            mismatches += 1
            var diff = _abs(R.data[i] - want.data[i])
            if diff > max_diff:
                max_diff = diff
    print("  ", label, "mismatched elements:", mismatches,
          " max |diff| =", max_diff)
    assert_true(mismatches == 0)


def test_block_size_and_rounds() raises:
    assert_true(largest_pow2_divisor(128) == 128)
    assert_true(largest_pow2_divisor(384) == 128)
    assert_true(largest_pow2_divisor(768) == 256)
    assert_true(largest_pow2_divisor(100) == 4)
    # B == d is a single round; otherwise the smallest k with B**k >= d.
    assert_true(rht_rounds(128, 128) == 1)
    assert_true(rht_rounds(384, 128) == 2)
    assert_true(rht_rounds(768, 256) == 2)
    assert_true(rht_rounds(36, 4) == 3)
    assert_true(rht_rounds(100, 4) == 4)
    print("test_block_size_and_rounds: ok")


def test_byte_parity_with_python() raises:
    """Every element must match Python's rht_rotation exactly, not nearly."""
    var R128 = rht_rotation(128, UInt64(42))
    _assert_bit_identical(R128, String("/tmp/_rht_R_128_42.npy"), String("d=128 seed=42"))
    var R384 = rht_rotation(384, UInt64(7))
    _assert_bit_identical(R384, String("/tmp/_rht_R_384_7.npy"), String("d=384 seed=7"))
    var R768 = rht_rotation(768, UInt64(42))
    _assert_bit_identical(R768, String("/tmp/_rht_R_768_42.npy"), String("d=768 seed=42"))
    # d=36 is B=4 over 3 rounds. An odd round count is the only shape in
    # which a globally inverted sign draw survives to the output — at an
    # even count it cancels, and every other case here is 1 or 2 rounds.
    var R36 = rht_rotation(36, UInt64(5))
    _assert_bit_identical(R36, String("/tmp/_rht_R_36_5.npy"), String("d=36 seed=5"))
    print("test_byte_parity_with_python: ok")


def test_orthogonal() raises:
    var d = 96          # B = 32 < d, so this exercises the multi-round path
    var R = rht_rotation(d, UInt64(3))
    var max_err: Float32 = Float32(0.0)
    for i in range(d):
        for j in range(d):
            var s: Float32 = Float32(0.0)
            for k in range(d):
                s += R.get(i, k) * R.get(j, k)
            var target: Float32 = Float32(1.0) if i == j else Float32(0.0)
            var err = _abs(s - target)
            if err > max_err:
                max_err = err
    print("   orthogonality err (d=96) =", max_err)
    assert_true(max_err < Float32(1e-5))
    print("test_orthogonal: ok")


def test_incoherence() raises:
    """A coordinate spike must come out spread — the property the Lloyd-Max
    codebook depends on. Mirrors tests/test_rotation_rht.py."""
    var d = 128
    var R = rht_rotation(d, UInt64(5))
    var x = alloc[Float32](d)
    var y = alloc[Float32](d)
    var floor = Float32(1.0 - 1e-6) / Float32(sqrt(Float64(d)))
    var worst: Float32 = Float32(0.0)
    for spike in range(16):
        for i in range(d):
            x[i] = Float32(0.0)
        x[spike] = Float32(1.0)
        matvec(R, x, y)
        var peak: Float32 = Float32(0.0)
        for i in range(d):
            var a = _abs(y[i])
            if a > peak:
                peak = a
        assert_true(peak >= floor)
        if peak > worst:
            worst = peak
    print("   worst spike peak (d=128) =", worst, " floor =", floor)
    # d = 128 is a power of two: one round, one block spanning the row, so a
    # spike maps to exactly +-1/sqrt(d) everywhere — the attainable floor.
    assert_true(worst < Float32(2.0) * floor)
    x.free()
    y.free()
    print("test_incoherence: ok")


def test_odd_d_rejected() raises:
    var raised = False
    try:
        var R = rht_rotation(63, UInt64(1))
        _ = R.rows
    except:
        raised = True
    assert_true(raised)
    print("test_odd_d_rejected: ok")


def main() raises:
    test_block_size_and_rounds()
    test_byte_parity_with_python()
    test_orthogonal()
    test_incoherence()
    test_odd_d_rejected()
    print("[test_rht] all passed")
