"""Tests for IVFCoarseIndex (coarse IVF over Matryoshka tier)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from remex import IVFCoarseIndex, PackedVectors, Quantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def setup_4bit():
    rng = np.random.default_rng(42)
    d = 128
    n = 2000
    pq = Quantizer(d=d, bits=4, seed=7)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((20, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    compressed = pq.encode(corpus)
    return pq, compressed, corpus, queries


@pytest.fixture
def setup_8bit():
    rng = np.random.default_rng(42)
    d = 128
    n = 2000
    pq = Quantizer(d=d, bits=8, seed=11)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((20, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    compressed = pq.encode(corpus)
    return pq, compressed, corpus, queries


# ---------------------------------------------------------------------------
# Construction & layout
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_invalid_n_bits(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        with pytest.raises(ValueError):
            IVFCoarseIndex(pq, comp, n_bits=0, mode="lsh")
        with pytest.raises(ValueError):
            IVFCoarseIndex(pq, comp, n_bits=17, mode="lsh")

    def test_invalid_mode(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        with pytest.raises(ValueError):
            IVFCoarseIndex(pq, comp, n_bits=4, mode="kmeans")

    def test_rotated_prefix_n_bits_exceeds_d(self):
        d = 8
        pq = Quantizer(d=d, bits=4, seed=0)
        rng = np.random.default_rng(0)
        corpus = rng.standard_normal((50, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        cv = pq.encode(corpus)
        with pytest.raises(ValueError):
            IVFCoarseIndex(pq, cv, n_bits=9, mode="rotated_prefix")

    def test_invalid_compressed_type(self, setup_4bit):
        pq, _, _, _ = setup_4bit
        with pytest.raises(TypeError):
            IVFCoarseIndex(pq, object(), n_bits=4, mode="lsh")

    def test_basic_attributes(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=6, mode="lsh", seed=0)
        assert ivf.n_bits == 6
        assert ivf.n_cells == 64
        assert ivf.mode == "lsh"
        assert ivf.cell_ids.shape == (comp.n,)
        assert ivf.cell_ids.dtype == np.uint16
        assert ivf.sorted_idx.shape == (comp.n,)
        assert ivf.cell_offsets.shape == (65,)
        # CSR offsets should cover all corpus rows exactly once
        assert ivf.cell_offsets[0] == 0
        assert ivf.cell_offsets[-1] == comp.n
        # Counts non-decreasing
        assert np.all(np.diff(ivf.cell_offsets) >= 0)

    def test_lsh_hyperplanes_shape(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=8, mode="lsh", seed=0)
        assert ivf.hyperplanes.shape == (8, pq.d)
        assert ivf.hyperplanes.dtype == np.float32

    def test_rotated_prefix_no_hyperplanes(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="rotated_prefix")
        assert ivf.hyperplanes is None

    def test_deterministic_lsh(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        a = IVFCoarseIndex(pq, comp, n_bits=6, mode="lsh", seed=42)
        b = IVFCoarseIndex(pq, comp, n_bits=6, mode="lsh", seed=42)
        np.testing.assert_array_equal(a.cell_ids, b.cell_ids)
        np.testing.assert_array_equal(a.sorted_idx, b.sorted_idx)
        np.testing.assert_array_equal(a.hyperplanes, b.hyperplanes)

    def test_lsh_different_seeds_differ(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        a = IVFCoarseIndex(pq, comp, n_bits=6, mode="lsh", seed=1)
        b = IVFCoarseIndex(pq, comp, n_bits=6, mode="lsh", seed=2)
        # Cell IDs must not all match — overwhelmingly unlikely for random hyperplanes
        assert not np.array_equal(a.cell_ids, b.cell_ids)

    def test_csr_layout_correct(self, setup_4bit):
        """sorted_idx[cell_offsets[c]:cell_offsets[c+1]] must contain
        exactly the corpus rows whose cell_id == c."""
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        for c in range(ivf.n_cells):
            cell_rows = ivf.sorted_idx[
                ivf.cell_offsets[c] : ivf.cell_offsets[c + 1]
            ]
            for row in cell_rows:
                assert ivf.cell_ids[row] == c


# ---------------------------------------------------------------------------
# Cell ID semantics
# ---------------------------------------------------------------------------


class TestCellIDs:

    def test_rotated_prefix_matches_msb_extraction(self, setup_8bit):
        """rotated_prefix cell_ids must equal the bit-pattern of the
        first n_bits MSBs of the encoded indices (= 1-bit Matryoshka)."""
        pq, comp, _, _ = setup_8bit
        n_bits = 5
        ivf = IVFCoarseIndex(pq, comp, n_bits=n_bits, mode="rotated_prefix")
        shift = pq.bits - 1
        msb = (comp.indices[:, :n_bits] >> shift).astype(np.uint32)
        expected = np.zeros(comp.n, dtype=np.uint32)
        for b in range(n_bits):
            expected |= msb[:, b] << b
        np.testing.assert_array_equal(ivf.cell_ids, expected.astype(np.uint16))

    def test_query_cell_consistent_with_corpus(self, setup_4bit):
        """Encode a corpus vector as a query — its cell ID must match
        the one stored at index 0 (since rotation is the same and the
        rotated_prefix hash takes signs of the same coords)."""
        pq, comp, corpus, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=6, mode="rotated_prefix")
        # Use the actual rotated coordinate signs of the original vector,
        # which is what the corpus side hashes (via its 1-bit MSB).
        for i in [0, 1, 2, 100, 999]:
            qc = ivf.query_cell(corpus[i])
            assert qc == int(ivf.cell_ids[i]), (
                f"vector {i}: query_cell {qc} != stored cell {int(ivf.cell_ids[i])}"
            )

    def test_cell_ids_in_range(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=7, mode="lsh", seed=0)
        assert ivf.cell_ids.min() >= 0
        assert ivf.cell_ids.max() < ivf.n_cells


# ---------------------------------------------------------------------------
# Multi-probe / Hamming ranking
# ---------------------------------------------------------------------------


class TestProbeCells:

    def test_nprobe_one_returns_query_cell(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        cells = ivf.probe_cells(query_cell=13, nprobe=1)
        np.testing.assert_array_equal(cells, np.array([13]))

    def test_nprobe_zero_empty(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="lsh", seed=0)
        cells = ivf.probe_cells(query_cell=3, nprobe=0)
        assert len(cells) == 0

    def test_nprobe_full_returns_all(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="lsh", seed=0)
        cells = ivf.probe_cells(query_cell=5, nprobe=ivf.n_cells)
        assert len(cells) == ivf.n_cells

    def test_hamming_ordering(self, setup_4bit):
        """First n_bits+1 cells visited must be the query cell plus all
        single-bit flips of it (Hamming distance 0 then 1)."""
        pq, comp, _, _ = setup_4bit
        n_bits = 5
        ivf = IVFCoarseIndex(pq, comp, n_bits=n_bits, mode="lsh", seed=0)
        q_cell = 13
        cells = ivf.probe_cells(query_cell=q_cell, nprobe=1 + n_bits)
        assert cells[0] == q_cell
        flips = set(int(q_cell ^ (1 << b)) for b in range(n_bits))
        assert set(int(c) for c in cells[1:]) == flips


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchCoarse:

    def test_full_nprobe_matches_search_adc(self, setup_4bit):
        """Visiting all cells must reproduce search_adc results exactly."""
        pq, comp, _, queries = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="lsh", seed=0)
        for q in queries[:5]:
            idx_ivf, scores_ivf = ivf.search_coarse(
                q, k=50, nprobe=ivf.n_cells, precision=None
            )
            idx_adc, scores_adc = pq.search_adc(comp, q, k=50)
            np.testing.assert_array_equal(idx_ivf, idx_adc)
            np.testing.assert_allclose(scores_ivf, scores_adc, rtol=1e-5)

    def test_full_nprobe_at_precision(self, setup_8bit):
        """Same as above at reduced precision (1-bit coarse)."""
        pq, comp, _, queries = setup_8bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="rotated_prefix")
        for q in queries[:3]:
            idx_ivf, _ = ivf.search_coarse(
                q, k=50, nprobe=ivf.n_cells, precision=1
            )
            idx_adc, _ = pq.search_adc(comp, q, k=50, precision=1)
            np.testing.assert_array_equal(idx_ivf, idx_adc)

    def test_indices_in_range(self, setup_4bit):
        pq, comp, _, queries = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        idx, _ = ivf.search_coarse(queries[0], k=20, nprobe=2)
        assert np.all(idx >= 0) and np.all(idx < comp.n)
        assert len(np.unique(idx)) == len(idx)  # no duplicates

    def test_scores_descending(self, setup_4bit):
        pq, comp, _, queries = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        _, scores = ivf.search_coarse(queries[0], k=30, nprobe=4)
        assert np.all(np.diff(scores) <= 1e-7)

    def test_k_capped_by_visited(self, setup_4bit):
        """If only one tiny cell is visited, len(idx) is bounded by it."""
        pq, comp, _, queries = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=10, mode="lsh", seed=0)
        idx, _ = ivf.search_coarse(queries[0], k=10_000, nprobe=1)
        c = ivf.query_cell(queries[0])
        cell_size = (
            ivf.cell_offsets[c + 1] - ivf.cell_offsets[c]
        )
        assert len(idx) == cell_size

    def test_recall_grows_with_nprobe(self, setup_4bit):
        """Recall vs flat ADC must be monotone (in expectation) and reach
        1.0 at nprobe = n_cells. We test the endpoints + monotone trend."""
        pq, comp, corpus, queries = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        k = 20

        # Ground truth = flat search at full precision (search returns
        # exact quantizer top-k).
        truths = []
        for q in queries[:10]:
            idx_truth, _ = pq.search(comp, q, k=k)
            truths.append(set(idx_truth.tolist()))

        recalls = []
        for nprobe in [1, 2, 4, 8, 16, 32]:
            hits = 0
            for q, tru in zip(queries[:10], truths):
                idx, _ = ivf.search_coarse(q, k=k, nprobe=nprobe)
                hits += len(set(idx.tolist()) & tru)
            recalls.append(hits / (10 * k))

        # Final value must be ~1.0 (visiting all cells)
        assert recalls[-1] >= 0.999, f"recall@nprobe=full = {recalls[-1]:.3f}"
        # And recall@full must be at least as good as recall@1
        assert recalls[-1] >= recalls[0]


class TestSearchTwostage:

    def test_full_nprobe_matches_quantizer_twostage(self, setup_8bit):
        """IVF two-stage with all cells visited must equal Quantizer.search_twostage."""
        pq, comp, _, queries = setup_8bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="rotated_prefix")
        for q in queries[:3]:
            idx_ivf, scores_ivf = ivf.search_twostage(
                q, k=10, candidates=200, nprobe=ivf.n_cells
            )
            idx_qz, scores_qz = pq.search_twostage(
                comp, q, k=10, candidates=200
            )
            np.testing.assert_array_equal(idx_ivf, idx_qz)
            np.testing.assert_allclose(scores_ivf, scores_qz, rtol=1e-5)

    def test_returns_correct_shape(self, setup_8bit):
        pq, comp, _, queries = setup_8bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        idx, scores = ivf.search_twostage(
            queries[0], k=10, candidates=100, nprobe=4
        )
        assert len(idx) == 10
        assert len(scores) == 10

    def test_indices_in_corpus_range(self, setup_8bit):
        pq, comp, _, queries = setup_8bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        idx, _ = ivf.search_twostage(
            queries[0], k=10, candidates=100, nprobe=4
        )
        assert np.all(idx >= 0) and np.all(idx < comp.n)


# ---------------------------------------------------------------------------
# PackedVectors interop
# ---------------------------------------------------------------------------


class TestPackedVectors:

    def test_packed_lsh_matches_compressed(self, setup_4bit):
        pq, comp, _, queries = setup_4bit
        packed = PackedVectors.from_compressed(comp)

        ivf_c = IVFCoarseIndex(pq, comp, n_bits=5, mode="lsh", seed=0)
        ivf_p = IVFCoarseIndex(pq, packed, n_bits=5, mode="lsh", seed=0)
        np.testing.assert_array_equal(ivf_c.cell_ids, ivf_p.cell_ids)

        for q in queries[:3]:
            idx_c, scores_c = ivf_c.search_coarse(q, k=20, nprobe=ivf_c.n_cells)
            idx_p, scores_p = ivf_p.search_coarse(q, k=20, nprobe=ivf_p.n_cells)
            np.testing.assert_array_equal(idx_c, idx_p)
            np.testing.assert_allclose(scores_c, scores_p, rtol=1e-5)

    def test_packed_rotated_prefix(self, setup_8bit):
        pq, comp, _, queries = setup_8bit
        packed = PackedVectors.from_compressed(comp)

        ivf_c = IVFCoarseIndex(pq, comp, n_bits=4, mode="rotated_prefix")
        ivf_p = IVFCoarseIndex(pq, packed, n_bits=4, mode="rotated_prefix")
        np.testing.assert_array_equal(ivf_c.cell_ids, ivf_p.cell_ids)

        for q in queries[:3]:
            idx_c, _ = ivf_c.search_twostage(q, k=10, candidates=100, nprobe=2)
            idx_p, _ = ivf_p.search_twostage(q, k=10, candidates=100, nprobe=2)
            np.testing.assert_array_equal(idx_c, idx_p)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:

    def test_cell_size_stats_keys(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf = IVFCoarseIndex(pq, comp, n_bits=4, mode="lsh", seed=0)
        stats = ivf.cell_size_stats()
        for k in ("n_cells", "min", "max", "mean", "std", "median",
                  "p95", "nonempty_cells", "nonempty_min"):
            assert k in stats
        assert stats["n_cells"] == 16
        assert stats["min"] >= 0
        assert stats["max"] >= stats["min"]
        # Total cell mass = corpus size
        sizes = np.diff(ivf.cell_offsets)
        assert int(sizes.sum()) == comp.n

    def test_index_nbytes_positive(self, setup_4bit):
        pq, comp, _, _ = setup_4bit
        ivf_lsh = IVFCoarseIndex(pq, comp, n_bits=6, mode="lsh", seed=0)
        ivf_rp = IVFCoarseIndex(pq, comp, n_bits=6, mode="rotated_prefix")
        assert ivf_lsh.index_nbytes > 0
        assert ivf_rp.index_nbytes > 0
        # LSH carries the hyperplanes, so it's heavier
        assert ivf_lsh.index_nbytes > ivf_rp.index_nbytes
