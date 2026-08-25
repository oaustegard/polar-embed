"""Every bit width the library declares supported must pack and unpack losslessly.

The domain is `SUPPORTED_BITS` itself. Before it existed the same set was
spelled five times as the negative guard `bits in (5, 6, 7)` and twice more as
the literal `[1, 2, 3, 4, 8]` in the tests, so adding a width meant remembering
seven places. Looping the tuple means a width added without a packing path
fails here by name instead of failing in whichever caller reaches it first.
"""

import numpy as np
import pytest

from remex.packing import SUPPORTED_BITS, pack, packed_nbytes, unpack


def test_the_supported_widths_are_not_empty():
    """Domain floor: an empty tuple must not pass this file vacuously."""
    assert len(SUPPORTED_BITS) >= 5


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_every_supported_width_round_trips(bits):
    rng = np.random.default_rng(bits)
    values = rng.integers(0, 1 << bits, size=997, dtype=np.uint8)

    packed = pack(values, bits)
    assert packed.dtype == np.uint8
    assert len(packed) == packed_nbytes(997, 1, bits)
    np.testing.assert_array_equal(unpack(packed, bits, 997), values)


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_every_supported_width_is_accepted_by_every_guard(bits):
    """The guards are the reason the tuple is exported; each must admit it."""
    from remex.core import Quantizer

    assert Quantizer(d=16, bits=bits).bits == bits
    assert packed_nbytes(4, 16, bits) > 0


@pytest.mark.parametrize("bits", [0, 5, 6, 7, 9])
def test_every_unsupported_width_is_refused(bits):
    with pytest.raises(ValueError, match="not supported"):
        pack(np.zeros(8, dtype=np.uint8), bits)


# totality: ratchet — every one of these widths has written indexes to disk;
# dropping one silently orphans every file packed at that width
def test_no_shipped_bit_width_is_ever_removed():
    """invariant: a bit width that has shipped never leaves SUPPORTED_BITS.

    The complement of the enumeration above. `test_every_supported_width_round_trips`
    loops whatever the tuple now holds, so removing a width leaves it green over
    the four that remain.

    refuted: removed 3 from SUPPORTED_BITS and lowered the floor to >= 4, the
    way a real removal would -> the enumeration, the guard test and the refusal
    test all stayed green (14 passed) while this went red naming 3. Dropping 3
    WITHOUT touching the floor is co-detected, because the floor is a
    cardinality check; it cannot see a member substituted for another.
    """
    for shipped in (1, 2, 3, 4, 8):
        assert shipped in SUPPORTED_BITS, (
            f"{shipped}-bit left SUPPORTED_BITS; every index packed at that "
            f"width becomes unreadable"
        )
