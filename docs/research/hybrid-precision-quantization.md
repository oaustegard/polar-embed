# Investigation: mixed-precision quantization via residual-error tail selection

**Status:** complete &middot; **Issue:** [#34](https://github.com/oaustegard/remex/issues/34) &middot; **Verdict:** _(pending &mdash; fill after Step 2)_

## TL;DR

_(to be written after Step 2)_

## Background

OjaKV (arXiv:[2509.21623](https://arxiv.org/abs/2509.21623), Zhu/Yang et al. 2025) reports that the dominant gain in their KV-cache scheme comes not from the online Oja updates they brand the paper around, but from a simple "hybrid storage" policy: keep tokens with high reconstruction error under the current basis at full fidelity, compress the rest. On RULER 0.6&times; their ablation shows this hybrid policy contributes **+18 points** while the Oja updates contribute only **+3**.

OjaKV's core technique is dimension reduction, not bit quantization, so most of the paper doesn't port to remex. But the hybrid-fidelity pattern is axis-agnostic: if a small fraction of vectors carry most of the quantization error, paying a little more memory to keep them at higher precision should trade cheaply for recall.

This investigation tests whether the pattern holds for TurboQuant-style scalar quantization as used in remex.

## Hypothesis

Under 2-bit remex (random orthogonal rotation + Lloyd-Max N(0, 1/d) codebook), the per-vector relative reconstruction error is **heavy-tailed**. Specifically, the top 10% of vectors by residual carry >40% of the total squared-error mass. If this holds, promoting those vectors to higher precision (4-bit or 8-bit) should yield a favorable recall/memory trade-off versus uniform quantization at matched bits/vector.

## Method

Implemented in `bench/hybrid_precision_eval.py`.

**Step 1** &mdash; residual-error distribution. For each vector `x[i]` in a corpus, quantize at 2-bit, decode, and compute `err[i] = ||x[i] - x_hat[i]||_2 / ||x[i]||_2`. Examine the distribution shape (skewness, kurtosis) and cumulative error mass in the top-p% of vectors (tail-mass / Lorenz curve).

**Gate.** If `mass_top_10pct` < 0.40, the tail is too flat for hybrid storage to help; stop.

**Step 2** &mdash; recall benchmark (conditional on gate). Build a hybrid index:

1. Compute per-vector 2-bit residual on the corpus.
2. Top `k_pct` of vectors by residual &rarr; quantize at `high_bits` (4 or 8).
3. Remaining vectors &rarr; quantize at 2-bit.
4. ADC-score both tiers with the *same* Haar rotation; merge into a single score vector.

Baselines: uniform 2-bit, uniform 3-bit, uniform 4-bit, uniform 8-bit. Recall@{10, 100} computed against exact inner-product ground truth on 200 held-out queries.

**Verdict** uses linear interpolation on the uniform curve at the hybrid's average bits/vector. Hybrid &ldquo;ships&rdquo; if it beats the interpolated uniform R@10 by >2 points at matched bits/vector.

## Corpora

| Corpus | n | d | Source |
|--------|---|---|--------|
| Synthetic Gaussian | 10 000 | 768 | Unit-normalized standard normal &mdash; the ideal TurboQuant input distribution. |
| SPECTER2 NLP-broad | 4 017 | 768 | Paper titles + abstracts from Semantic Scholar for &ldquo;natural language processing&rdquo;, encoded with `allenai/specter2_base`. |

The SPECTER2 corpus is a superset of the [existing SPECTER2 case study](../specter2-case-study.md), which already established that this corpus has &sigma; &asymp; 0.38 &times; the expected post-rotation Gaussian &mdash; an interesting test because Lloyd-Max is misspecified, so quantization error might behave differently than on the synthetic baseline.

## Step 1 results

### Synthetic Gaussian (d=768, n=10 000)

```
Relative L2 error  (||x - x̂|| / ||x||):
  min / median / mean / max : 0.3064 / 0.3421 / 0.3422 / 0.3827
  std                       : 0.0094
  p95 / p99                 : 0.3580 / 0.3657
  skewness / excess kurt.   : 0.147 / 0.142

Squared-error tail mass:
  top  1%  →  0.012   (1.16x uniform)
  top  5%  →  0.056   (1.12x uniform)
  top 10%  →  0.110   (1.10x uniform)
  top 20%  →  0.216   (1.08x uniform)
```

The residual distribution is **essentially uniform** over vectors. Top 10% carries 11% of total error mass, only 1.10&times; what a flat distribution would produce. Skewness (0.15) and excess kurtosis (0.14) are near-zero. This is the expected behavior under TurboQuant&rsquo;s design assumptions: a Haar rotation isotropizes the error, and Lloyd-Max bins are optimal for the matching N(0, 1/d) marginal, so no vector is much worse off than the mean.

**Gate on synthetic: FAILS.** The tail is flat.

### SPECTER2 NLP-broad (d=768, n=4 017)

_(to be filled after encoding completes)_

## Gate decision

_(to be filled)_

## Step 2 results

_(to be filled)_

## Verdict

_(to be filled)_

## What to take away

_(to be filled)_

## Reproducing

```bash
pip install -e ".[dev,bench]" transformers torch matplotlib

# Fast path — synthetic only, always works
python bench/hybrid_precision_eval.py --synthetic-only --plots

# With SPECTER2 (first run downloads the model + fetches abstracts)
python bench/specter2_eval.py --cached --skip-recall   # primes corpus cache
python bench/hybrid_precision_eval.py --specter2 --plots

# Full sweep matching the issue spec
python bench/hybrid_precision_eval.py --specter2 --sweep --plots
```

Output artifacts:

- `bench/plots/*.png` &mdash; residual histograms and Lorenz-style tail-mass curves.
- `bench/hybrid_results/results_*.json` &mdash; structured Step 1 + Step 2 data for downstream analysis.
