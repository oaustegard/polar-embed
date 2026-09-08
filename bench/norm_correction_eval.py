"""What the reconstruction-length correction (``renorm=True``) is worth.

remex stores each vector's exact norm and a quantized unit direction. The
decoded direction is not unit length -- at 2-bit it measures about 0.89-0.97
-- so ``norms * u_hat`` reconstructs a vector whose length is off by a
per-vector factor of roughly 1%. ``renorm=True`` divides it out. The length
is read off the codes, so nothing extra is stored.

A ~1% score bias only reorders neighbours that sit within 1% of each other,
so the gain is set by how tightly packed the corpus is. This script measures
both ends of that: the synthetic cluster-spread sweep from the README, and
SPECTER2 if the cache is present.

Usage:
    python bench/norm_correction_eval.py            # synthetic only
    bash bench/fetch_specter2_cache.sh              # then:
    python bench/norm_correction_eval.py --specter2
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from remex import Quantizer  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".specter2_cache")
BITS = (1, 2, 3, 4, 8)


def exact_topk(corpus, queries, k):
    return np.argsort(-(queries @ corpus.T), axis=1)[:, :k]


def recall_at_k(pred, truth, k):
    hits = sum(len(set(p[:k].tolist()) & set(t[:k].tolist()))
               for p, t in zip(pred, truth))
    return hits / (len(pred) * k)


def neighbourhood_gap(corpus, queries, k=10):
    """Median relative score gap between rank k and rank 5k.

    The scale the ~1% length error has to cross to reorder anything.
    """
    srt = -np.sort(-(queries @ corpus.T), axis=1)
    return float(np.median((srt[:, k - 1] - srt[:, 5 * k - 1])
                           / np.abs(srt[:, k - 1])))


def arm(corpus, queries, truth, bits, renorm, seed=42):
    pq = Quantizer(d=corpus.shape[1], bits=bits, seed=seed, renorm=renorm)
    compressed = pq.encode(corpus)
    recon = pq.decode(compressed)
    pred = exact_topk(recon, queries, 100)
    return (recall_at_k(pred, truth, 10),
            recall_at_k(pred, truth, 100),
            pq.mse(corpus))


def report(name, corpus, queries, extra=""):
    truth = exact_topk(corpus, queries, 100)
    gap = neighbourhood_gap(corpus, queries)
    print(f"\n=== {name} ===")
    print(f"  n={len(corpus)} d={corpus.shape[1]} queries={len(queries)}"
          f"  rank-10-to-50 score gap={gap:.4f} {extra}")
    print(f"  {'bits':>4} {'R@10 off':>9} {'R@10 on':>8} {'delta':>7}"
          f" | {'R@100 off':>10} {'R@100 on':>9} {'delta':>7}"
          f" | {'MSE off':>10} {'MSE on':>10}")
    for bits in BITS:
        a10, a100, amse = arm(corpus, queries, truth, bits, False)
        b10, b100, bmse = arm(corpus, queries, truth, bits, True)
        print(f"  {bits:>4} {a10:>9.3f} {b10:>8.3f} {b10 - a10:>+7.3f}"
              f" | {a100:>10.3f} {b100:>9.3f} {b100 - a100:>+7.3f}"
              f" | {amse:>10.5f} {bmse:>10.5f}")


def synthetic_sweep(n=10_000, d=384, n_queries=200):
    """The README's cluster-spread axis, with the correction off and on.

    ``bench.benchmark.make_clustered_embeddings`` pins spread at 0.3; the
    same construction is inlined here so spread can be swept.
    """
    print("\n########## Synthetic: cluster spread ##########")
    for spread in (0.01, 0.05, 0.10, 0.30, 1.00):
        rng = np.random.default_rng(7)
        centres = rng.standard_normal((20, d)).astype(np.float32)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        labels = rng.integers(0, 20, size=n + n_queries)
        X = centres[labels] + spread * rng.standard_normal(
            (n + n_queries, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        report(f"clustered, spread={spread}", X[n_queries:], X[:n_queries])


def specter2_sweep(n_queries=500, seed=99):
    for part in ("broad", "narrow"):
        path = os.path.join(CACHE_DIR, f"specter2_nlp_{part}.npy")
        if not os.path.exists(path):
            print(f"\n[skip] {path} missing — run bench/fetch_specter2_cache.sh")
            continue
        X = np.load(path)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        report(f"SPECTER2 {part}", X[perm[n_queries:]].astype(np.float32),
               X[perm[:n_queries]].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specter2", action="store_true",
                    help="also run the cached SPECTER2 partitions")
    args = ap.parse_args()
    synthetic_sweep()
    if args.specter2:
        print("\n########## Real embeddings: SPECTER2 ##########")
        specter2_sweep()


if __name__ == "__main__":
    main()
