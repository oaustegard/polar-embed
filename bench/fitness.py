"""One-number fitness for supervisor-driven optimization (claude-workspace#233).

Prints recall@10 for one quantizer configuration on the synthetic clustered
corpus from bench/benchmark.py — a single scalar on stdout, everything else
on stderr. Deterministic per (config, data seeds).

    python bench/fitness.py --bits 4 --rotation rht
    0.8485
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

from benchmark import exact_knn, make_clustered_embeddings, recall_at_k  # noqa: E402
from remex import Quantizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--rotation", default="rht", choices=["rht", "haar"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--two-stage", action="store_true", help="rescore top candidates")
    ap.add_argument("--candidates", type=int, default=200, help="two-stage pool size")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    corpus = make_clustered_embeddings(args.n, args.d)
    queries = make_clustered_embeddings(args.queries, args.d, seed=7)
    truth = exact_knn(corpus, queries, args.k)

    q = Quantizer(d=args.d, bits=args.bits, rotation=args.rotation, seed=args.seed)
    enc = q.encode(corpus)
    if args.two_stage:
        idx = np.stack(
            [
                q.search_twostage(enc, qv, k=args.k, candidates=args.candidates)[0]
                for qv in queries
            ]
        )
    else:
        idx = np.stack([q.search(enc, qv, k=args.k)[0] for qv in queries])
    r = recall_at_k(idx, truth, args.k)

    print(f"{r:.4f}")
    print(
        f"bits={args.bits} rotation={args.rotation} seed={args.seed} "
        f"two_stage={args.two_stage} n={args.n} d={args.d} "
        f"wall={time.time() - t0:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
