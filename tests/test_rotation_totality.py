"""Every rotation the library declares must survive the on-disk formats.

The domain is `ROTATION_CODES` itself, not a list someone typed. A rotation
added to that dict without a persistence path decodes under the wrong basis,
and the codes come back 50%-different rather than slightly off — so the whole
registry is the domain, and a hand-written subset would hide exactly the
member nobody remembered to add.
"""

import numpy as np
import pytest

from remex import CompressedVectors, load_pq, save_pq
from remex.core import Quantizer
from remex.rotation import ROTATION_CODES, rotation_from_code

ROTATIONS = sorted(ROTATION_CODES)


def test_the_registry_is_not_empty():
    """Domain floor: a registry that collapses to zero must not pass vacuously."""
    assert len(ROTATIONS) >= 3


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_every_declared_rotation_round_trips(tmp_path, rotation):
    X = np.random.default_rng(0).standard_normal((24, 64)).astype(np.float32)
    cv = Quantizer(d=64, bits=4, seed=1, rotation=rotation).encode(X)
    assert cv.rotation == rotation

    pq = tmp_path / f"{rotation}.pq"
    save_pq(pq, cv)
    assert load_pq(pq).rotation == rotation

    npz = tmp_path / f"{rotation}.npz"
    cv.save(npz)
    assert CompressedVectors.load(npz).rotation == rotation

    assert rotation_from_code(ROTATION_CODES[rotation]) == rotation


def test_constructible_and_persistable_rotations_agree():
    """The two independent spellings of the rotation domain must stay equal.

    `Quantizer.ROTATIONS` maps a name to the function that BUILDS it;
    `ROTATION_CODES` maps a name to the byte that PERSISTS it. Nothing in the
    type system ties them together, so a rotation added to one and not the
    other either cannot be built or cannot be read back.
    """
    from remex.core import Quantizer

    assert len(ROTATION_CODES) >= 3, "domain floor"
    assert set(Quantizer.ROTATIONS) == set(ROTATION_CODES)
    for name in ROTATION_CODES:
        assert name in Quantizer.ROTATIONS
        assert rotation_from_code(ROTATION_CODES[name]) == name
