"""Reconstruction-length correction.

remex stores each vector's exact norm and a quantized unit direction. The
decoded direction is not unit length -- at 2-bit its length runs about
0.89-0.97 -- so ``norms * u_hat`` reconstructs a vector whose length is off
by a per-vector factor of roughly 1%. Dividing by that length costs no stored
bytes, because it is computable from the codes, and is what ``renorm=True``
(the default) does.

A 1% score bias only reorders neighbours that sit within 1% of each other, so
the size of the gain is set by how tightly packed the corpus is, not by how
large the bias is. The bias itself measures the same on isotropic Gaussian
data as on real embeddings. The clustered fixture below is the repo's own
distribution-sensitivity axis (see the README's cluster-spread table) and at
sigma=0.02 it reproduces the effect measured on SPECTER2.
"""
import numpy as np
import pytest

from remex import Quantizer


def clustered_corpus(n=1200, d=128, n_clusters=30, sigma=0.02, seed=0):
    """Tight clusters -- neighbours separated by less than the length bias.

    ``sigma`` is the cluster spread from the README's sensitivity table. At
    0.02 the median score gap between rank 10 and rank 50 is ~0.05, well
    inside the ~1% reconstruction-length error, so the correction moves
    recall. A plain ``rng.standard_normal`` corpus has a gap of ~0.17 and
    shows nothing.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_clusters, d)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    assign = rng.integers(0, n_clusters, n)
    X = centres[assign] + sigma * rng.standard_normal((n, d)).astype(np.float32)
    return X.astype(np.float32)


def recall_at_k(pred, truth, k):
    hits = sum(len(set(p[:k].tolist()) & set(t[:k].tolist()))
               for p, t in zip(pred, truth))
    return hits / (len(pred) * k)


def exact_topk(corpus, queries, k):
    return np.argsort(-(queries @ corpus.T), axis=1)[:, :k]


class TestDecodedLength:
    def test_renorm_decode_matches_stored_norm(self):
        X = clustered_corpus()
        pq = Quantizer(d=X.shape[1], bits=2, seed=42)
        c = pq.encode(X)
        recon = pq.decode(c)
        assert np.allclose(np.linalg.norm(recon, axis=1), c.norms, rtol=1e-4)

    def test_without_renorm_decoded_length_drifts(self):
        X = clustered_corpus()
        pq = Quantizer(d=X.shape[1], bits=2, seed=42, renorm=False)
        c = pq.encode(X)
        recon = pq.decode(c)
        ratio = np.linalg.norm(recon, axis=1) / c.norms
        assert ratio.std() > 1e-3, "fixture does not exercise the correction"
        assert not np.allclose(ratio, 1.0, rtol=1e-3)

    def test_one_bit_is_a_no_op(self):
        """At 1 bit every coordinate has the same magnitude, so the decoded
        direction length is a constant and the correction cannot reorder."""
        X = clustered_corpus()
        q = X[0]
        on = Quantizer(d=X.shape[1], bits=1, seed=42)
        off = Quantizer(d=X.shape[1], bits=1, seed=42, renorm=False)
        i_on, _ = on.search(on.encode(X), q, k=25)
        i_off, _ = off.search(off.encode(X), q, k=25)
        assert np.array_equal(i_on, i_off)


class TestRecall:
    @pytest.mark.parametrize("bits,margin", [(2, 0.15), (3, 0.15), (4, 0.10)])
    def test_renorm_improves_recall_on_tight_clusters(self, bits, margin):
        X = clustered_corpus(n=1500, d=128, seed=3)
        queries, corpus = X[:50], X[50:]
        truth = exact_topk(corpus, queries, 10)

        def r10(renorm):
            pq = Quantizer(d=corpus.shape[1], bits=bits, seed=42, renorm=renorm)
            recon = pq.decode(pq.encode(corpus))
            return recall_at_k(exact_topk(recon, queries, 10), truth, 10)

        assert r10(True) >= r10(False) + margin


class TestPathAgreement:
    @pytest.mark.parametrize("bits", [2, 4])
    def test_adc_matches_cached_search(self, bits):
        X = clustered_corpus(n=800, d=64, seed=1)
        pq = Quantizer(d=X.shape[1], bits=bits, seed=42)
        c = pq.encode(X)
        q = X[7]
        i_cached, s_cached = pq.search(c, q, k=10)
        i_adc, s_adc = pq.search_adc(c, q, k=10)
        assert np.array_equal(i_cached, i_adc)
        assert np.allclose(s_cached, s_adc, rtol=1e-4, atol=1e-5)

    def test_packed_matches_compressed(self):
        from remex import PackedVectors
        X = clustered_corpus(n=800, d=64, seed=2)
        pq = Quantizer(d=X.shape[1], bits=3, seed=42)
        c = pq.encode(X)
        p = PackedVectors.from_compressed(c)
        q = X[3]
        i_c, s_c = pq.search_adc(c, q, k=10)
        i_p, s_p = pq.search_adc(p, q, k=10)
        assert np.array_equal(i_c, i_p)
        assert np.allclose(s_c, s_p, rtol=1e-4, atol=1e-5)

    def test_batch_matches_single(self):
        X = clustered_corpus(n=600, d=64, seed=4)
        pq = Quantizer(d=X.shape[1], bits=4, seed=42)
        c = pq.encode(X)
        qs = X[:3]
        idx_b, _ = pq.search_batch(c, qs, k=5)
        for row, q in zip(idx_b, qs):
            idx_s, _ = pq.search(c, q, k=5)
            assert np.array_equal(row, idx_s)


class TestPrecisionAndModes:
    def test_correction_recomputed_per_precision(self):
        """A Matryoshka decode at precision p must use p's direction length,
        not the full-width one."""
        X = clustered_corpus(n=400, d=64, seed=5)
        pq = Quantizer(d=X.shape[1], bits=8, seed=42)
        c = pq.encode(X)
        for precision in (2, 4, 8):
            recon = pq.decode(c, precision=precision)
            assert np.allclose(
                np.linalg.norm(recon, axis=1), c.norms, rtol=1e-4
            ), f"precision={precision}"

    def test_scalar_mode_untouched(self):
        """Scalar mode has no norms, so there is nothing to correct."""
        rng = np.random.default_rng(6)
        X = rng.standard_normal((200, 32)).astype(np.float64)
        on = Quantizer(d=32, bits=4, seed=42, normalize=False,
                       rotation="none", scale=1.0)
        off = Quantizer(d=32, bits=4, seed=42, normalize=False,
                        rotation="none", scale=1.0, renorm=False)
        assert np.array_equal(on.decode(on.encode(X)), off.decode(off.encode(X)))

    def test_renorm_is_not_part_of_the_codes(self):
        """The flag changes interpretation, not encoding, so a container
        written by one setting decodes under the other."""
        X = clustered_corpus(n=300, d=64, seed=7)
        on = Quantizer(d=64, bits=4, seed=42)
        off = Quantizer(d=64, bits=4, seed=42, renorm=False)
        assert np.array_equal(on.encode(X).indices, off.encode(X).indices)
