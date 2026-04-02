"""Core polar-embed encoder/decoder with Matryoshka bit precision."""

import numpy as np
from typing import Optional, Tuple
from polar_embed.codebook import lloyd_max_codebook, nested_codebooks
from polar_embed.rotation import haar_rotation


class CompressedVectors:
    """Container for quantized vector data."""

    __slots__ = ("indices", "norms", "d", "bits", "n")

    def __init__(self, indices: np.ndarray, norms: np.ndarray, d: int, bits: int):
        self.indices = indices
        self.norms = norms
        self.d = d
        self.bits = bits
        self.n = indices.shape[0]

    def subset(self, idx: np.ndarray) -> "CompressedVectors":
        """Return a CompressedVectors containing only the given row indices."""
        return CompressedVectors(
            self.indices[idx], self.norms[idx], self.d, self.bits
        )

    @property
    def nbytes(self) -> int:
        """Actual memory footprint in bytes."""
        return self.indices.nbytes + self.norms.nbytes

    @property
    def compression_ratio(self) -> float:
        """Ratio vs float32 storage."""
        return (self.n * self.d * 4) / self.nbytes

    def save(self, path: str):
        """Save to compressed .npz file."""
        np.savez_compressed(
            path,
            indices=self.indices,
            norms=self.norms,
            d=np.int32(self.d),
            bits=np.int32(self.bits),
        )

    @classmethod
    def load(cls, path: str) -> "CompressedVectors":
        """Load from .npz file."""
        data = np.load(path)
        return cls(data["indices"], data["norms"], int(data["d"]), int(data["bits"]))


class PolarQuantizer:
    """
    Data-oblivious vector quantizer with Matryoshka bit precision.

    Encodes vectors by:
    1. Normalizing to unit sphere (storing norms separately)
    2. Applying a random orthogonal rotation (makes coordinates ~N(0, 1/d))
    3. Scalar-quantizing each coordinate with a Lloyd-Max codebook

    Supports **nested bit precision**: encode once at full bit-width,
    then search at any lower precision by right-shifting indices.
    The top k bits of an n-bit code are a valid k-bit code, with
    centroid tables precomputed for each level. Nesting penalty is
    typically <1.5% recall vs independently optimized codebooks.

    This enables two-stage retrieval: coarse search at low bits for
    candidates, then rerank at full precision.

    Args:
        d: Vector dimension.
        bits: Bits per coordinate (1-8). 3-4 is the sweet spot.
        seed: Random seed for rotation matrix.
    """

    def __init__(self, d: int, bits: int = 4, seed: int = 42):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be 1-8, got {bits}")

        self.d = d
        self.bits = bits
        self.seed = seed

        self.R = haar_rotation(d, seed)
        self.boundaries, self.centroids = lloyd_max_codebook(d, bits)

        # Precompute nested centroid tables for all bit levels <= bits
        self._nested = nested_codebooks(d, bits)

    def encode(self, X: np.ndarray) -> CompressedVectors:
        """
        Quantize a batch of vectors.

        Args:
            X: (n, d) float array. Need not be unit-normalized.

        Returns:
            CompressedVectors container with indices and norms.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got {X.shape[1]}")

        norms = np.linalg.norm(X, axis=1)
        X_unit = X / np.maximum(norms, 1e-8)[:, None]

        X_rot = X_unit @ self.R.T
        indices = np.searchsorted(self.boundaries, X_rot).astype(np.uint8)

        return CompressedVectors(indices, norms.astype(np.float32), self.d, self.bits)

    def decode(
        self, compressed: CompressedVectors, precision: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct vectors from compressed representation.

        Args:
            compressed: CompressedVectors from encode().
            precision: Bit precision for reconstruction (1 to self.bits).
                       None = full precision.

        Returns:
            (n, d) float32 array of approximate vectors.
        """
        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)

        X_hat_rot = centroids[indices]
        X_hat_unit = X_hat_rot @ self.R  # R is orthogonal -> R^T inverts R.T
        return X_hat_unit * compressed.norms[:, None]

    def search(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors by approximate inner product.

        Operates in rotated space to avoid full dequantization.

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Number of results.
            precision: Bit precision for search (1 to self.bits).
                       Lower = faster/coarser, higher = more accurate.
                       None = full precision (self.bits).

        Returns:
            (indices, scores): top-k corpus indices and approximate scores.
        """
        query = np.asarray(query, dtype=np.float32)
        q_rot = self.R @ query

        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)

        X_hat_rot = centroids[indices]
        scores = (X_hat_rot @ q_rot) * compressed.norms

        if k >= compressed.n:
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_idx, scores[topk_idx]

    def search_twostage(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        candidates: int = 500,
        coarse_precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage retrieval: coarse search then full-precision rerank.

        Stage 1: Search at coarse_precision for top `candidates`.
        Stage 2: Rerank candidates at full precision for top `k`.

        This leverages the Matryoshka bit nesting — the same encoded
        data is searched at two different precision levels.

        Args:
            compressed: Encoded corpus.
            query: (d,) query vector.
            k: Final number of results.
            candidates: Number of coarse candidates (stage 1).
            coarse_precision: Bit precision for coarse pass.
                              Default: max(1, self.bits - 2).

        Returns:
            (indices, scores): top-k corpus indices and full-precision scores.
            Indices are into the original corpus (not the candidate set).
        """
        if coarse_precision is None:
            coarse_precision = max(1, self.bits - 2)

        # Stage 1: coarse pass
        coarse_k = min(candidates, compressed.n)
        coarse_idx, _ = self.search(compressed, query, k=coarse_k,
                                    precision=coarse_precision)

        # Stage 2: rerank at full precision
        subset = compressed.subset(coarse_idx)
        rerank_idx, rerank_scores = self.search(subset, query, k=k)

        # Map back to original corpus indices
        original_idx = coarse_idx[rerank_idx]
        return original_idx, rerank_scores

    def mse(self, X: np.ndarray, precision: Optional[int] = None) -> float:
        """Compute mean per-vector reconstruction MSE (L2 squared)."""
        compressed = self.encode(X)
        X_hat = self.decode(compressed, precision=precision)
        return float(np.mean(np.sum((np.asarray(X, np.float32) - X_hat) ** 2, axis=1)))

    def _resolve_centroids(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Get centroid table for the requested precision level."""
        if precision is None:
            return self.centroids
        if precision < 1 or precision > self.bits:
            raise ValueError(
                f"precision must be 1-{self.bits}, got {precision}"
            )
        return self._nested[precision]

    def _resolve_indices(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Right-shift indices to the requested precision level."""
        if precision is None or precision == self.bits:
            return compressed.indices
        shift = self.bits - precision
        return compressed.indices >> shift
