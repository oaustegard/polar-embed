"""Fixture builder for tests/test_rht.mojo.

Writes, for each (d, seed) case, the Python `rht_rotation(d, seed)` matrix
plus an encode-parity bundle, so the Mojo side can assert byte equality
rather than an approximate match.

    python3 remex/mojo/tests/build_rht_fixture.py [outdir]

Default outdir is /tmp. Run from the repo root with remex importable.
"""

import sys

import numpy as np

from remex import Quantizer, save_pq, save_params
from remex.rotation import rht_rotation

# (d, seed), chosen to span the shapes the construction can take:
#   d=128  B=128, 1 round  — power of two, one block spanning the row
#   d=384  B=128, 2 rounds — mainstream embedding size, 3 blocks per row
#   d=768  B=256, 2 rounds — mainstream embedding size, 3 blocks per row
#   d=36   B=4,   3 rounds — odd round count > 1, small block, 9 per row
# The d=36 case is not decorative: with an even round count a globally
# inverted sign draw cancels itself out, so nothing else here can tell a
# sign-mapping bug from a correct one.
CASES = [(128, 42), (384, 7), (768, 42), (36, 5)]

# Encode-parity case, kept small so the Mojo test stays quick.
ENC_N, ENC_D, ENC_BITS, ENC_SEED = 64, 384, 4, 7


def main(outdir="/tmp"):
    for d, seed in CASES:
        R = rht_rotation(d, seed)
        np.save(f"{outdir}/_rht_R_{d}_{seed}.npy", R)
        print(f"wrote {outdir}/_rht_R_{d}_{seed}.npy  shape={R.shape}")

    rng = np.random.default_rng(0)
    X = rng.standard_normal((ENC_N, ENC_D)).astype(np.float32)
    np.save(f"{outdir}/_rht_X.npy", X)
    q = Quantizer(d=ENC_D, bits=ENC_BITS, seed=ENC_SEED, rotation="rht")
    save_pq(f"{outdir}/_rht_ref.pq", q.encode(X))
    # Same Quantizer dumped as params, so the Mojo test can check both routes
    # into it: rebuilt from the seed, and read straight off disk.
    save_params(f"{outdir}/_rht.params", q)
    print(f"wrote {outdir}/_rht_X.npy, {outdir}/_rht_ref.pq and "
          f"{outdir}/_rht.params "
          f"(n={ENC_N}, d={ENC_D}, bits={ENC_BITS}, seed={ENC_SEED})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp")
