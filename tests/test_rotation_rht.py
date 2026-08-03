"""Tests for the randomized-Hadamard rotation option.

The claim this rests on is that an RHT is a legitimate substitute for the Haar
rotation *for what the codec actually needs* — an orthogonal map that leaves
coordinates incoherent, so the Lloyd-Max codebook (fitted to N(0, 1/d)) is
quantizing the distribution it was built for. These check that property
directly rather than trusting the recall result from a different codebase.
"""

import math

import numpy as np
import pytest

from remex.core import Quantizer
from remex.pq_format import save_params
from remex.rotation import haar_rotation, rht_rotation

DIMS = [64, 128, 384, 768, 1024]


@pytest.mark.parametrize("d", DIMS)
def test_orthogonal(d):
    Q = rht_rotation(d, seed=7)
    assert Q.dtype == np.float32 and Q.shape == (d, d)
    err = np.max(np.abs(Q.astype(np.float64) @ Q.astype(np.float64).T - np.eye(d)))
    assert err < 1e-5, f"max|QQ^T - I| = {err:.2e}"


@pytest.mark.parametrize("d", DIMS)
def test_deterministic_from_seed(d):
    assert np.array_equal(rht_rotation(d, 42), rht_rotation(d, 42))
    assert not np.array_equal(rht_rotation(d, 42), rht_rotation(d, 43))


@pytest.mark.parametrize("d", [128, 384, 768])
def test_incoherence_matches_haar(d):
    """The property the codebook depends on: a coordinate spike must come out
    spread. Compared against an INDEPENDENT reference — the max|coord| of a
    uniformly random unit vector, drawn directly — not against Haar alone,
    because two equally-broken rotations would agree with each other.
    """
    rng = np.random.default_rng(3)
    E = np.eye(d, dtype=np.float32)[:64]           # 64 coordinate spikes
    U = rng.standard_normal((4096, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ideal = float(np.mean(np.max(np.abs(U), axis=1)))

    # The lower edge is ATTAINABLE and inclusive, with an ulp of slack. When
    # the Hadamard block spans the whole vector (d a power of two, one round)
    # a spike maps to exactly +-1/sqrt(d) in every coordinate -- the
    # information-theoretic floor, since the coordinates square to 1. Reaching
    # it is optimal, not a failure, and float32 rounding lands a hair below the
    # float64 constant. A strict comparison here fails on a perfect rotation.
    floor = (1.0 - 1e-6) / math.sqrt(d)
    for name, R in (("haar", haar_rotation(d, 5)), ("rht", rht_rotation(d, 5))):
        mu = float(np.mean(np.max(np.abs(E @ R.T), axis=1)))
        assert floor <= mu < 2.0 * ideal, (
            f"{name} d={d}: spike incoherence {mu:.6f} outside "
            f"[{floor:.6f}, {2 * ideal:.4f})"
        )


def test_identity_would_fail_the_incoherence_check():
    """The check above must be able to reject something. An identity 'rotation'
    is the worst case and has to land outside the bracket."""
    d = 384
    rng = np.random.default_rng(3)
    U = rng.standard_normal((4096, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ideal = float(np.mean(np.max(np.abs(U), axis=1)))
    mu_identity = 1.0                               # a spike stays a spike
    floor = (1.0 - 1e-6) / math.sqrt(d)
    assert not (floor <= mu_identity < 2.0 * ideal)


@pytest.mark.parametrize("d", [128, 768])
def test_coordinates_are_gaussian_after_rotation(d):
    """The codebook is fitted to N(0, 1/d); check the rotation delivers it."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((2000, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rot = (X @ rht_rotation(d, 5).T).ravel()
    assert abs(float(np.mean(rot))) < 5e-3
    assert abs(float(np.std(rot)) - 1.0 / math.sqrt(d)) / (1.0 / math.sqrt(d)) < 0.05


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_round_trip_fidelity_matches_haar(bits):
    """End to end: RHT must not cost reconstruction quality."""
    d, rng = 384, np.random.default_rng(0)
    X = rng.standard_normal((300, d)).astype(np.float32)
    cos = {}
    for rot in ("haar", "rht"):
        q = Quantizer(d=d, bits=bits, seed=42, rotation=rot)
        Xh = q.decode(q.encode(X))
        num = np.sum(X * Xh, axis=1)
        den = np.linalg.norm(X, axis=1) * np.linalg.norm(Xh, axis=1)
        cos[rot] = float(np.mean(num / den))
    assert abs(cos["haar"] - cos["rht"]) < 2e-3, cos


def test_rotation_is_part_of_the_encoding():
    """Codes from one rotation must not be decoded under the other. This is a
    guard against treating `rotation` as a free-floating perf knob."""
    d, rng = 128, np.random.default_rng(1)
    X = rng.standard_normal((64, d)).astype(np.float32)
    qh = Quantizer(d=d, bits=8, seed=42, rotation="haar")
    qr = Quantizer(d=d, bits=8, seed=42, rotation="rht")
    good = qh.decode(qh.encode(X))
    crossed = qr.decode(qh.encode(X))
    cos = lambda A: float(np.mean(np.sum(X * A, 1) / (
        np.linalg.norm(X, axis=1) * np.linalg.norm(A, axis=1))))
    assert cos(good) > 0.99 and cos(crossed) < 0.5, (cos(good), cos(crossed))


@pytest.mark.parametrize("rotation", ["haar", "rht"])
def test_save_params_accepts_every_mojo_rotation(tmp_path, rotation):
    """Mojo rebuilds both constructions from the seed, so params dump for
    both. The R written is the Quantizer's own, whichever rotation built it."""
    q = Quantizer(d=64, bits=4, seed=1, rotation=rotation)
    path = tmp_path / f"{rotation}.pr"
    save_params(path, q)

    raw = path.read_bytes()
    d, n_levels = 64, 1 << 4
    assert len(raw) == 16 + d * d * 4 + (n_levels - 1) * 4 + n_levels * 4
    R = np.frombuffer(raw, np.float32, count=d * d, offset=16).reshape(d, d)
    assert np.array_equal(R, q.R)


def test_save_params_refuses_a_rotation_mojo_lacks(tmp_path):
    """The guard still has to bite for anything the Mojo port has no
    construction for — it just no longer bites on 'rht'."""
    q = Quantizer(d=64, bits=4, seed=1)
    q.rotation = "hadamard-but-fancier"
    with pytest.raises(ValueError, match="save_params supports rotation in"):
        save_params(tmp_path / "bad.pr", q)


def test_odd_dimension_rejected():
    with pytest.raises(ValueError, match="even dimension"):
        rht_rotation(255, seed=1)


def test_unknown_rotation_rejected():
    with pytest.raises(ValueError, match="rotation must be one of"):
        Quantizer(d=64, bits=4, rotation="hadamard")
