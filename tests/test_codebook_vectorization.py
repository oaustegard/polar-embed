"""Regression tests for the vectorized Lloyd-Max codebook update.

The per-level scipy loop was replaced by a whole-array update. That is only
valid because `bounds` is materialised BEFORE any centroid moves, making the
step simultaneous (Jacobi) rather than sequential (Gauss-Seidel) — the two
give different answers, and nothing in the suite pinned which one this is.
These tests pin it, plus the published anchor the codebook must reproduce.
"""

import numpy as np
from scipy.stats import norm

from remex.codebook import lloyd_max_codebook

#: Max, "Quantizing for minimum distortion", IRE Trans. IT-6 (1960), table 1 —
#: MSE of the optimal fixed-rate scalar quantizer for a unit-variance Gaussian.
#: An external anchor: no part of this repo produced these numbers.
MAX_1960_MSE = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497, 5: 0.002499}


def _mse_against_gaussian(centroids, sigma):
    """Exact distortion of a level set against N(0, sigma^2), in closed form."""
    lv = np.sort(np.asarray(centroids, dtype=np.float64))
    edges = np.concatenate([[-np.inf], (lv[:-1] + lv[1:]) / 2.0, [np.inf]])
    rv = norm(0, sigma)
    cdf, pdf = rv.cdf(edges), rv.pdf(edges)
    p = np.diff(cdf)
    ex = sigma**2 * (pdf[:-1] - pdf[1:])            # E[X 1_cell]
    a, b = edges[:-1], edges[1:]
    fa = np.where(np.isfinite(a), a * rv.pdf(np.where(np.isfinite(a), a, 0)), 0.0)
    fb = np.where(np.isfinite(b), b * rv.pdf(np.where(np.isfinite(b), b, 0)), 0.0)
    exx = sigma**2 * (p - sigma**2 * (fb - fa) / sigma**2)  # E[X^2 1_cell]
    return float(np.sum(exx - 2 * lv * ex + lv**2 * p))


def test_reproduces_max_1960_table():
    """The codebook must match a published table, not just its own last run."""
    for bits, published in MAX_1960_MSE.items():
        _, centroids = lloyd_max_codebook(d=1, bits=bits)   # sigma = 1
        mse = _mse_against_gaussian(centroids, sigma=1.0)
        assert abs(mse - published) / published < 5e-3, (
            f"bits={bits}: {mse:.6f} vs Max (1960) {published}")


def test_update_is_simultaneous_not_sequential():
    """Pins the property the vectorization depends on.

    A sequential update — recomputing each boundary from already-moved
    centroids — converges somewhere else. If someone 'optimises' the loop back
    into that form, this fails.
    """
    d, bits, n_iter = 768, 4, 300
    sigma = 1.0 / np.sqrt(d)
    rv = norm(0, sigma)
    c = np.linspace(-3 * sigma, 3 * sigma, 2**bits)
    for _ in range(n_iter):                       # deliberately Gauss-Seidel
        for j in range(len(c)):
            lo = -np.inf if j == 0 else (c[j - 1] + c[j]) / 2
            hi = np.inf if j == len(c) - 1 else (c[j] + c[j + 1]) / 2
            prob = rv.cdf(hi) - rv.cdf(lo)
            if prob > 1e-15:
                c[j] = sigma**2 * (rv.pdf(lo) - rv.pdf(hi)) / prob
    _, centroids = lloyd_max_codebook(d=d, bits=bits, n_iter=n_iter)
    assert not np.allclose(centroids, c.astype(np.float32), atol=1e-9), (
        "simultaneous and sequential updates agree — the test cannot "
        "distinguish them, so it is not pinning anything")


def test_boundaries_are_level_midpoints_and_sorted():
    """Nearest-neighbour condition: a decision boundary is a level midpoint."""
    for d, bits in ((768, 8), (1024, 4), (256, 2)):
        boundaries, centroids = lloyd_max_codebook(d=d, bits=bits)
        assert np.all(np.diff(centroids) > 0)
        assert np.allclose(boundaries,
                           (centroids[:-1] + centroids[1:]) / 2.0, atol=1e-7)


def test_scales_with_sigma():
    """Levels for N(0, sigma) are sigma x the levels for N(0, 1)."""
    _, unit = lloyd_max_codebook(d=1, bits=4)
    for d in (256, 768, 1024):
        _, scaled = lloyd_max_codebook(d=d, bits=4)
        np.testing.assert_allclose(scaled, unit / np.sqrt(d), rtol=2e-4)
