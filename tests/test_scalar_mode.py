"""Scalar mode: ``Quantizer(normalize=False)`` (issue #77).

Direct Lloyd-Max quantization of coordinates, with no unit-norm
factorization and no stored norms. The motivating use case is codes as
exact-match hash keys for a meet-in-the-middle join over enumerated
algebraic values, where the default pipeline actively hurts:

- unit-norm factorization maps every constant-direction family onto ONE
  direction code, which is a mega-bucket rather than a key;
- the float32 cast on the normalizing path turns large values into
  ``inf`` and their codes into whatever ``searchsorted(nan)`` gives.

Both properties are asserted here directly, not implied.
"""
from __future__ import annotations

import numpy as np
import pytest

from remex import (
    CompressedVectors, IVFCoarseIndex, PackedVectors, Quantizer,
    load_pq, save_params, save_pq,
)
from remex.codebook import lloyd_max_codebook


def _corpus(n=200, d=16, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) * scale).astype(np.float64)


class TestConstruction:
    def test_scalar_mode_stores_no_norms(self):
        pq = Quantizer(d=16, bits=4, normalize=False)
        comp = pq.encode(_corpus())
        assert comp.norms is None
        assert comp.has_norms is False
        assert comp.indices.shape == (200, 16)

    def test_normalize_true_is_the_default_and_unchanged(self):
        assert Quantizer(d=16, bits=4).normalize is True
        comp = Quantizer(d=16, bits=4).encode(_corpus())
        assert comp.norms is not None
        assert comp.has_norms is True

    def test_scale_defaults_to_unit_and_is_rejected_when_normalizing(self):
        assert Quantizer(d=16, bits=4, normalize=False).scale == pytest.approx(1.0)
        assert Quantizer(d=64, bits=4).scale == pytest.approx(1.0 / 8.0)
        with pytest.raises(ValueError, match="normalize=False"):
            Quantizer(d=16, bits=4, scale=2.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_nonpositive_scale_rejected(self, bad):
        with pytest.raises(ValueError, match="finite and positive"):
            Quantizer(d=16, bits=4, normalize=False, scale=bad)

    def test_codebook_is_the_unit_one_rescaled(self):
        """Scalar mode changes only the coordinate sigma the cells are cut for."""
        d, bits, scale = 16, 4, 2.5
        pq = Quantizer(d=d, bits=bits, normalize=False, scale=scale)
        expect_b, expect_c = lloyd_max_codebook(d, bits, sigma=scale)
        np.testing.assert_array_equal(pq.boundaries, expect_b)
        np.testing.assert_array_equal(pq.centroids, expect_c)
        # ...and that is the unit codebook stretched by scale * sqrt(d).
        unit_b, _ = lloyd_max_codebook(d, bits)
        np.testing.assert_allclose(
            expect_b, unit_b * scale * np.sqrt(d), rtol=1e-5, atol=1e-6
        )


class TestMotivatingProperties:
    """The two behaviours the issue says block the hash-key use case."""

    def test_constant_direction_family_collapses_only_when_normalizing(self):
        """`c * ones(d)` is one direction and many values.

        Normalized, the whole family shares a single direction code — the
        mega-bucket that turned a 3-minute join into a collision storm.
        Scalar mode keys them apart.
        """
        d = 16
        family = np.array(
            [[c] * d for c in (0.1, 0.4, 0.8, 1.2, 1.7, 2.3)], dtype=np.float64
        )

        normalized = Quantizer(d=d, bits=4).encode(family).indices
        assert len({row.tobytes() for row in normalized}) == 1

        scalar = Quantizer(d=d, bits=4, normalize=False).encode(family).indices
        assert len({row.tobytes() for row in scalar}) == len(family)

    def test_scalar_mode_survives_values_that_overflow_float32(self):
        """Float32 max is ~3.4e38; the normalizing path casts to it."""
        d = 4
        X = np.array([[1e300, -1e300, 0.0, 1.0]], dtype=np.float64)

        with np.errstate(over="ignore", invalid="ignore"):
            normalized = Quantizer(d=d, bits=4).encode(X)
        assert not np.isfinite(normalized.norms).all()

        pq = Quantizer(d=d, bits=4, normalize=False, rotation="none")
        codes = pq.encode(X).indices
        n_levels = 2 ** pq.bits
        assert (codes < n_levels).all()
        # Saturation at the extreme cells, with the sign preserved.
        assert codes[0, 0] == n_levels - 1
        assert codes[0, 1] == 0
        # ...and it is a code, not a crash: the same input encodes the same
        # way every time.
        np.testing.assert_array_equal(codes, pq.encode(X).indices)

    def test_ordinary_magnitudes_are_not_saturated(self):
        """Saturation is the tail behaviour, not the common case."""
        pq = Quantizer(d=16, bits=4, normalize=False, rotation="none")
        codes = pq.encode(_corpus(n=500, d=16)).indices
        n_levels = 2 ** pq.bits
        interior = (codes > 0) & (codes < n_levels - 1)
        assert interior.mean() > 0.9


class TestDeterminism:
    """The property that makes a code usable as an exact-match hash key."""

    def test_same_params_same_codes(self):
        X = _corpus()
        a = Quantizer(d=16, bits=4, seed=42, normalize=False).encode(X)
        b = Quantizer(d=16, bits=4, seed=42, normalize=False).encode(X)
        np.testing.assert_array_equal(a.indices, b.indices)

    def test_equal_inputs_collide_unequal_inputs_mostly_do_not(self):
        pq = Quantizer(d=16, bits=8, normalize=False, rotation="none")
        X = _corpus(n=300, d=16)
        keys = pq.encode(np.vstack([X, X])).indices
        first, second = keys[:300], keys[300:]
        np.testing.assert_array_equal(first, second)
        assert len({row.tobytes() for row in first}) == 300

    def test_scale_and_normalize_change_the_codes(self):
        """Both are determinants of the encoding, like seed and rotation."""
        X = _corpus()
        base = Quantizer(d=16, bits=4, normalize=False).encode(X).indices
        other = (
            Quantizer(d=16, bits=4, normalize=False, scale=3.0)
            .encode(X).indices
        )
        assert not np.array_equal(base, other)
        assert not np.array_equal(
            base, Quantizer(d=16, bits=4).encode(X).indices
        )


class TestIdentityRotation:
    def test_encode_is_a_bare_searchsorted(self):
        pq = Quantizer(d=16, bits=4, normalize=False, rotation="none")
        X = _corpus()
        expect = np.searchsorted(pq.boundaries, X).astype(np.uint8)
        np.testing.assert_array_equal(pq.encode(X).indices, expect)

    def test_R_is_the_identity(self):
        pq = Quantizer(d=8, bits=4, rotation="none")
        np.testing.assert_array_equal(pq.R, np.eye(8, dtype=np.float32))

    def test_coordinate_j_depends_only_on_input_j(self):
        pq = Quantizer(d=8, bits=4, normalize=False, rotation="none")
        X = _corpus(n=1, d=8, seed=3)
        perturbed = X.copy()
        perturbed[0, 2] += 10.0
        a, b = pq.encode(X).indices, pq.encode(perturbed).indices
        np.testing.assert_array_equal(np.flatnonzero(a[0] != b[0]), [2])

    def test_identity_rotation_still_round_trips_when_normalizing(self):
        pq = Quantizer(d=16, bits=8, rotation="none")
        X = _corpus(n=50, d=16).astype(np.float32)
        comp = pq.encode(X)
        assert comp.rotation == "none"
        np.testing.assert_allclose(pq.decode(comp), X, atol=0.05)


class TestReconstruction:
    def test_decode_recovers_coordinates(self):
        pq = Quantizer(d=16, bits=8, normalize=False, rotation="none")
        X = _corpus(n=100, d=16)
        err = np.abs(pq.decode(pq.encode(X)) - X)
        # The outermost cells are open-ended, so a value past ~3 sigma is
        # reconstructed at its cell's centroid and can miss by more than the
        # interior cell width. Bound the bulk tightly and the tail loosely.
        assert np.percentile(err, 99) < 0.03
        assert err.max() < 0.2

    def test_mse_improves_with_bits(self):
        X = _corpus(n=200, d=16)
        errs = [
            Quantizer(d=16, bits=b, normalize=False).mse(X)
            for b in (1, 2, 3, 4, 8)
        ]
        assert errs == sorted(errs, reverse=True)

    def test_matching_scale_beats_mismatched_scale(self):
        """`scale` is the knob the caller uses to declare their conditioning."""
        X = _corpus(n=200, d=16, scale=5.0)
        matched = Quantizer(d=16, bits=4, normalize=False, scale=5.0).mse(X)
        wrong = Quantizer(d=16, bits=4, normalize=False, scale=0.2).mse(X)
        assert matched < wrong / 10

    def test_encode_accepts_a_single_vector(self):
        pq = Quantizer(d=16, bits=4, normalize=False)
        comp = pq.encode(_corpus(n=1, d=16)[0])
        assert comp.n == 1
        assert comp.norms is None

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="Expected d=16"):
            Quantizer(d=16, bits=4, normalize=False).encode(_corpus(d=8))


class TestMatryoshka:
    """Nested precision — the bit the caller-side int16 substitute gave up."""

    def test_right_shift_is_a_valid_coarser_code(self):
        pq8 = Quantizer(d=16, bits=8, normalize=False, rotation="none")
        pq1 = Quantizer(d=16, bits=1, normalize=False, rotation="none")
        X = _corpus(n=100, d=16)
        np.testing.assert_array_equal(
            pq8.encode(X).indices >> 7, pq1.encode(X).indices
        )

    def test_coarser_precision_collides_more(self):
        """Nested precision is the collision/recall dial, per query."""
        pq = Quantizer(d=4, bits=8, normalize=False, rotation="none")
        codes = pq.encode(_corpus(n=2000, d=4)).indices
        distinct = [
            len({row.tobytes() for row in (codes >> (8 - p))})
            for p in (8, 4, 2, 1)
        ]
        assert distinct == sorted(distinct, reverse=True)
        assert distinct[0] > distinct[-1]
        assert distinct[-1] <= 2 ** 4  # 1 bit x 4 coords = 16 buckets, at most

    def test_decode_at_precision(self):
        pq = Quantizer(d=16, bits=8, normalize=False, rotation="none")
        X = _corpus(n=100, d=16)
        comp = pq.encode(X)
        coarse = pq.decode(comp, precision=2)
        fine = pq.decode(comp, precision=8)
        assert _rmse(coarse, X) > _rmse(fine, X)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestSearch:
    """Scoring in scalar mode is the raw quantized inner product."""

    @staticmethod
    def _fixture():
        pq = Quantizer(d=16, bits=8, normalize=False)
        X = _corpus(n=300, d=16, seed=5)
        return pq, X, pq.encode(X)

    def test_search_finds_the_query_itself(self):
        pq, X, comp = self._fixture()
        for row in (0, 7, 42):
            idx, _ = pq.search(comp, X[row].astype(np.float32), k=5)
            assert idx[0] == row

    def test_search_scores_are_the_plain_inner_product(self):
        pq, X, comp = self._fixture()
        q = X[3].astype(np.float32)
        idx, scores = pq.search(comp, q, k=10)
        expect = pq.decode(comp)[idx] @ q
        np.testing.assert_allclose(scores, expect, rtol=1e-4)

    def test_adc_matches_cached_search(self):
        pq, X, comp = self._fixture()
        q = X[11].astype(np.float32)
        a_idx, a_sc = pq.search(comp, q, k=10)
        b_idx, b_sc = pq.search_adc(comp, q, k=10)
        np.testing.assert_array_equal(a_idx, b_idx)
        np.testing.assert_allclose(a_sc, b_sc, rtol=1e-5)

    def test_twostage_matches_full_scan_when_candidates_is_everything(self):
        pq, X, comp = self._fixture()
        q = X[19].astype(np.float32)
        a_idx, _ = pq.search(comp, q, k=10)
        b_idx, _ = pq.search_twostage(comp, q, k=10, candidates=comp.n)
        np.testing.assert_array_equal(a_idx, b_idx)

    def test_search_batch_matches_per_query_search(self):
        pq, X, comp = self._fixture()
        queries = X[:4].astype(np.float32)
        all_idx, all_sc = pq.search_batch(comp, queries, k=6)
        for i, q in enumerate(queries):
            idx, sc = pq.search(comp, q, k=6)
            np.testing.assert_array_equal(all_idx[i], idx)
            np.testing.assert_allclose(all_sc[i], sc, rtol=1e-5)

    def test_ivf_flat_scan_matches_adc(self):
        pq, X, comp = self._fixture()
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="lsh", seed=0)
        q = X[23].astype(np.float32)
        a_idx, a_sc = pq.search_adc(comp, q, k=10)
        b_idx, b_sc = ivf.search_coarse(q, k=10, nprobe=ivf.n_cells)
        np.testing.assert_array_equal(a_idx, b_idx)
        np.testing.assert_allclose(a_sc, b_sc, rtol=1e-5)

    def test_gpu_numpy_backend_matches_search(self):
        from remex.gpu import GPUSearcher

        pq, X, comp = self._fixture()
        searcher = GPUSearcher(pq, comp, backend="numpy")
        q = X[31].astype(np.float32)
        a_idx, a_sc = pq.search(comp, q, k=10)
        b_idx, b_sc = searcher.search(q, k=10)
        np.testing.assert_array_equal(a_idx, b_idx)
        np.testing.assert_allclose(a_sc, b_sc, rtol=1e-5)

    def test_gpu_does_not_pay_for_a_norms_column(self):
        from remex.gpu import GPUSearcher

        pq, X, comp = self._fixture()
        normalizing = Quantizer(d=16, bits=8)
        scalar_bytes = GPUSearcher(
            pq, comp, backend="numpy"
        ).resident_bytes_gpu
        normalized_bytes = GPUSearcher(
            normalizing, normalizing.encode(X.astype(np.float32)),
            backend="numpy",
        ).resident_bytes_gpu
        assert normalized_bytes - scalar_bytes == comp.n * 4


class TestMemory:
    def test_norms_column_is_not_paid_for(self):
        X = _corpus(n=1000, d=16)
        scalar = Quantizer(d=16, bits=4, normalize=False).encode(X)
        normalized = Quantizer(d=16, bits=4).encode(X)
        assert normalized.nbytes - scalar.nbytes == 1000 * 4
        assert scalar.nbytes_unpacked == scalar.indices.nbytes
        assert scalar.resident_bytes == scalar.indices.nbytes
        assert scalar.compression_ratio > normalized.compression_ratio

    def test_packed_vectors_carry_the_absence_through(self):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        packed = PackedVectors.from_compressed(comp)
        assert packed.norms is None
        assert packed.has_norms is False
        assert packed.nbytes == packed._packed.nbytes
        assert packed.resident_bytes == packed._packed.nbytes

        np.testing.assert_array_equal(
            packed.to_compressed().indices, comp.indices
        )
        assert packed.to_compressed().norms is None
        assert packed.subset(np.array([0, 3, 5])).norms is None
        assert packed.at_precision(2).norms is None

    def test_subset_keeps_scalar_mode(self):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        sub = comp.subset(np.array([1, 4, 9]))
        assert sub.norms is None
        assert sub.n == 3

    def test_packed_adc_matches_unpacked(self):
        pq = Quantizer(d=16, bits=4, normalize=False)
        X = _corpus(n=200, d=16)
        comp = pq.encode(X)
        packed = PackedVectors.from_compressed(comp)
        q = X[2].astype(np.float32)
        a_idx, a_sc = pq.search_adc(comp, q, k=10)
        b_idx, b_sc = pq.search_adc(packed, q, k=10)
        np.testing.assert_array_equal(a_idx, b_idx)
        np.testing.assert_allclose(a_sc, b_sc, rtol=1e-5)

    def test_from_rows_accepts_no_norms(self):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        packed = PackedVectors.from_compressed(comp)
        rows = [bytes(r) for r in packed._packed]
        rebuilt = PackedVectors.from_rows(rows, None, 16, 4)
        assert rebuilt.norms is None
        np.testing.assert_array_equal(
            rebuilt.to_compressed().indices, comp.indices
        )


class TestSerialization:
    """Absence of the norms column is what marks a file as scalar mode."""

    def test_npz_round_trip(self, tmp_path):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        path = str(tmp_path / "scalar.npz")
        comp.save(path)
        assert "norms" not in np.load(path)
        loaded = CompressedVectors.load(path)
        assert loaded.norms is None
        np.testing.assert_array_equal(loaded.indices, comp.indices)

    def test_packed_npz_round_trip(self, tmp_path):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        packed = PackedVectors.from_compressed(comp)
        path = str(tmp_path / "scalar_packed.npz")
        packed.save(path)
        loaded = PackedVectors.load(path)
        assert loaded.norms is None
        np.testing.assert_array_equal(
            loaded.to_compressed().indices, comp.indices
        )

    def test_pq_round_trip(self, tmp_path):
        comp = Quantizer(
            d=16, bits=4, normalize=False, rotation="none"
        ).encode(_corpus())
        path = tmp_path / "scalar.pq"
        save_pq(path, comp)
        loaded = load_pq(path)
        assert loaded.norms is None
        assert loaded.rotation == "none"
        np.testing.assert_array_equal(loaded.indices, comp.indices)

    def test_pq_scalar_file_is_shorter_by_exactly_the_norms(self, tmp_path):
        X = _corpus()
        scalar = Quantizer(d=16, bits=4, normalize=False).encode(X)
        normalized = Quantizer(d=16, bits=4).encode(X)
        a, b = tmp_path / "s.pq", tmp_path / "n.pq"
        save_pq(a, scalar)
        save_pq(b, normalized)
        assert b.stat().st_size - a.stat().st_size == scalar.n * 4

    def test_pq_flag_byte_defaults_off_for_normalized_files(self, tmp_path):
        path = tmp_path / "n.pq"
        save_pq(path, Quantizer(d=16, bits=4).encode(_corpus()))
        assert path.read_bytes()[18] == 0

    def test_pq_rejects_unknown_flag_bits(self, tmp_path):
        path = tmp_path / "future.pq"
        save_pq(path, Quantizer(d=16, bits=4).encode(_corpus()))
        raw = bytearray(path.read_bytes())
        raw[18] = 0x80
        path.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="unknown .pq flag bits"):
            load_pq(path)

    def test_save_params_refuses_scalar_quantizers(self, tmp_path):
        pq = Quantizer(d=16, bits=4, normalize=False)
        with pytest.raises(ValueError, match="scalar-mode"):
            save_params(tmp_path / "p.params", pq)

    def test_save_params_refuses_the_identity_rotation(self, tmp_path):
        with pytest.raises(ValueError, match="save_params supports rotation"):
            save_params(tmp_path / "p.params", Quantizer(d=16, rotation="none"))

    def test_arrow_round_trip(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.feather as feather

        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        path = str(tmp_path / "scalar.arrow")
        comp.save_arrow(path, seed=42)
        assert "norms" not in feather.read_table(path).schema.names

        loaded = CompressedVectors.load_arrow(path)
        assert loaded.norms is None
        np.testing.assert_array_equal(loaded.indices, comp.indices)


class TestModeGuard:
    """Crossing the two modes rescales every coordinate, so it must raise."""

    def test_normalizing_quantizer_refuses_scalar_codes(self):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        with pytest.raises(ValueError, match="mode mismatch"):
            Quantizer(d=16, bits=4).decode(comp)

    def test_scalar_quantizer_refuses_normalized_codes(self):
        comp = Quantizer(d=16, bits=4).encode(_corpus())
        with pytest.raises(ValueError, match="mode mismatch"):
            Quantizer(d=16, bits=4, normalize=False).decode(comp)

    def test_guard_fires_on_the_search_paths_too(self):
        comp = Quantizer(d=16, bits=4, normalize=False).encode(_corpus())
        pq = Quantizer(d=16, bits=4)
        q = np.zeros(16, dtype=np.float32)
        for call in (
            lambda: pq.search(comp, q, k=3),
            lambda: pq.search_adc(comp, q, k=3),
            lambda: pq.search_twostage(comp, q, k=3),
            lambda: pq.search_batch(comp, q[None], k=3),
        ):
            with pytest.raises(ValueError, match="mode mismatch"):
                call()

    def test_rotation_guard_still_fires_within_scalar_mode(self):
        comp = Quantizer(
            d=16, bits=4, normalize=False, rotation="none"
        ).encode(_corpus())
        with pytest.raises(ValueError, match="rotation mismatch"):
            Quantizer(d=16, bits=4, normalize=False).decode(comp)
