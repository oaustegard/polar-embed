# remex core

The library: quantizer, packing, rotation, on-disk formats and search.

## invariants

- every declared rotation survives a round-trip through the on-disk formats
- every supported bit width packs and unpacks losslessly
- the constructible rotations and the persistable rotations are the same set

## refutations

- every declared rotation survives a round-trip through the on-disk formats:
  added `"hadamard2": 3` to `ROTATION_CODES` with no construction behind it ->
  the whole pre-existing suite stayed green (267 passed) while this claim went
  red by name on `test_every_declared_rotation_round_trips[hadamard2]`.
- the constructible rotations and the persistable rotations are the same set:
  added `"hadamard2": identity_rotation` to `Quantizer.ROTATIONS` only -> the
  pre-existing suite stayed green (267 passed) while this claim went red on
  `test_constructible_and_persistable_rotations_agree`.
- every supported bit width packs and unpacks losslessly: added `5` to
  `SUPPORTED_BITS` with no packing branch behind it -> this claim went red on
  `test_every_supported_width_round_trips[5]`. Co-detected: the pre-existing
  `test_bits_validation` also fails, because it asserts 5 is refused. Recorded
  as the weaker refutation it is.

## works when

- rotation.py exists at this node
- packing.py exists at this node
- boundary "every declared rotation survives a round-trip through the on-disk formats" at rotation_from_code via test "test_every_declared_rotation_round_trips"
- boundary "every supported bit width packs and unpacks losslessly" at pack via test "test_every_supported_width_round_trips"
- parity "the constructible rotations and the persistable rotations are the same set" over ROTATION_CODES between ROTATIONS and rotation_from_code via test "test_constructible_and_persistable_rotations_agree"

## why

Two domains in this package are enumerated in more than one place, with nothing
tying the spellings together.

`ROTATION_CODES` maps a rotation name to the byte that persists it;
`Quantizer.ROTATIONS` maps the same names to the functions that build them. A
name in one and not the other either cannot be built or cannot be read back, and
a rotation decoded under the wrong basis comes back 50%-different rather than
slightly off — the failure is total and silent. The parity claim is what keeps
the two dicts equal; before it, the equality held only because both lists were
short enough to remember.

`SUPPORTED_BITS` was extracted for the same reason. The set was previously
spelled five times as the negative guard `bits in (5, 6, 7)` across `packing.py`,
`core.py` and `pq_format.py`, and twice more as `[1, 2, 3, 4, 8]` in the tests.
Adding a width meant remembering seven sites; now the guards and the round-trip
oracle read one tuple.

Both oracles loop the live registry rather than a copied list, so a member added
to a registry without the path behind it fails by name here instead of failing
in whichever caller reaches it first.
