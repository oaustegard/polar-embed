# CLAUDE.md

## Project overview

**remex** (formerly polar-embed) is a Python library for retrieval-validated embedding compression. It implements random orthogonal rotation + Lloyd-Max scalar quantization (from TurboQuant, Zandieh et al. ICLR 2026) to compress embedding vectors 2-16x with measured recall, optimized for nearest-neighbor retrieval in RAG systems.

Key differentiator: **data-oblivious** — no training required. The quantizer is fully determined by `(dimension, bits, seed, rotation, normalize, scale)`.

## Architecture

```
remex/
├── __init__.py       # Public API, version
├── core.py           # Quantizer, CompressedVectors (main classes)
├── codebook.py       # Lloyd-Max codebooks + Matryoshka nested tables
├── ivf.py            # IVFCoarseIndex — coarse-tier IVF, data-oblivious
├── packing.py        # Bit-packing for sub-byte storage (1-8 bit)
├── rotation.py       # Haar (QR), randomized Hadamard, and the identity
└── gpu.py            # Optional GPU backend (CuPy/PyTorch/NumPy)

tests/
├── test_polar_embed.py   # Core: rotation, codebook, quantizer, retrieval, packing
├── test_matryoshka.py    # Nested codebooks, precision parameter, two-stage search, subset
├── test_adc_gpu.py       # ADC search, memory accounting, GPUSearcher (numpy fallback)
├── test_ivf.py           # IVFCoarseIndex: cell ID, multi-probe, recall, packed interop
├── test_packed_vectors.py # PackedVectors creation, unpacking, ADC, serialization
├── test_coverage_gaps.py  # Edge cases, save/load all bits, subset search
└── test_scalar_mode.py   # normalize=False: hash-key properties, rotation="none", mode guard

bench/
├── benchmark.py          # Self-contained benchmark (no external deps)
├── real_embedding_eval.py  # Real embeddings benchmark (needs sentence-transformers, faiss)
└── RESULTS.md            # Benchmark results with distribution sensitivity analysis
```

### Data flow

```
float32 embeddings
    → normalize (store norms separately)
    → rotate (R @ x, random orthogonal matrix)
    → quantize (searchsorted into Lloyd-Max boundaries → uint8 indices)
    → CompressedVectors (indices + norms)

Search:
    → rotate query (R @ q)
    → score via matmul (cached dequant) or ADC (lookup table over indices)
    → top-k selection
```

**Scalar mode** (`Quantizer(normalize=False)`) drops the first step and the
norms with it — codes are Lloyd-Max indices of the raw coordinates, cut for
a caller-declared `scale` instead of the unit sphere's `1/sqrt(d)`, in
float64 rather than float32. It exists for use cases where the codes *are*
the product (exact-match hash keys, joins, bucketing) rather than an
approximation of a direction; `rotation="none"` pairs with it to make a code
a bare `searchsorted`. `norms is None` is what every container and
serializer uses to record the mode, and mixing modes raises.

### Three search strategies

| Method | Memory | Speed | When to use |
|--------|--------|-------|-------------|
| `search()` | High (caches n*d*4 float32) | Fast (matmul) | Repeated queries, RAM available |
| `search_adc()` | Low (uint8 indices only) | Slower (table lookup) | Memory-constrained, serverless |
| `search_twostage()` | Low (ADC coarse + small fine) | Medium | Best recall/memory trade-off |

### Sublinear coarse-tier scan: `IVFCoarseIndex`

Optional inverted-file index over the coarse Matryoshka tier. Visits
only `nprobe` of `2**n_bits` cells per query, replacing the
bandwidth-bound flat coarse scan in `search_twostage` for very large
corpora (≥ tens of millions of vectors). Two **data-oblivious** hash
modes (no k-means, no fitting):

- `mode='lsh'` — random-hyperplane SimHash, deterministic from
  `(d, n_bits, seed)`.
- `mode='rotated_prefix'` — sign of the first `n_bits` post-rotation
  coordinates; free given the existing rotation matrix.

Multi-probe is by Hamming distance from the query's hash. Setting
`nprobe = 2**n_bits` recovers a flat scan exactly (verified by
test_ivf.py). The latency-recall Pareto and cross-FoS bridge edge
preservation are benchmarked in `bench/specter2_eval.py`.

## Development

```bash
pip install -e ".[dev]"    # numpy, scipy, pytest, pytest-cov
pytest                      # 126 tests, ~6 min
pytest tests/test_adc_gpu.py -v  # just ADC/GPU tests, ~30s
```

### Running benchmarks

```bash
python bench/benchmark.py               # synthetic data, no extra deps
pip install -e ".[bench]"               # for real embedding benchmarks
python bench/real_embedding_eval.py     # needs sentence-transformers + faiss-cpu

# SPECTER2 (allenai/specter2_base, d=768) — encoding the transformer takes
# ~50 min on CPU per 10k papers. Skip the encode by pulling a precomputed
# cache from GH release:
bash bench/fetch_specter2_cache.sh      # ~45 MB, restores .specter2_cache/
python bench/specter2_eval.py --cached  # then run the bench against the cache
```

## Code conventions

- **NumPy-only core**: No PyTorch/CuPy dependency in `remex/core.py`. GPU support is opt-in via `remex/gpu.py`.
- **No training**: Fully data-oblivious. The quantizer is determined by `(d, bits, seed, rotation, normalize, scale)` alone. Scalar mode keeps this: `scale` is a value the caller *declares*, never one remex measures off the data.
- **The rotation is part of the encoding**: every persisted container (`.pq` byte 17, `.npz` `rotation` key,
  Arrow `b"rotation"` metadata) records which rotation wrote it, and an absent record means `"haar"` — the
  frozen historical value, never the live default. That rule is what makes the default safe to change; see
  `bench/gates/rotation_identity_gate.py`.
- **So is the mode**: a scalar-mode container (`normalize=False`) has `norms is None`, and every serializer
  records that by *omitting* the norms column (`.npz`/Arrow) or setting the no-norms flag (`.pq` byte 18,
  bit 0). Decoding or searching across the two modes raises, like a rotation mismatch.
- **Honest compression**: `nbytes` property uses bit-packed sizes, not uint8. Benchmark tables report packed compression ratios.
- **Deterministic**: Same `(d, bits, seed)` must produce identical results across runs. `rotation="none"` strengthens this to bit-identical across BLAS builds, because no matmul enters the encoding.
- **Test thresholds**: Recall tests use conservative bounds (e.g. 2-bit R@10 >= 0.3, not exact values) because recall depends on random data.

## Key design decisions

1. **Norms stored separately as float32** — preserves inner-product ranking up to quantization error. This is why 8-bit gives R@10=0.98+ despite "only" 4x compression.

2. **Matryoshka via right-shift** — An n-bit index's top k bits are a valid k-bit code. This enables two-stage search from a single encoding. The nesting penalty depends on which level you extract:
   - **1-bit**: 0% penalty (the MSB of an n-bit Lloyd-Max code *is* the sign bit, which is exactly the standalone 1-bit code). `tests/test_matryoshka.py::TestPrecisionOneBit::test_matryoshka_1bit_equals_standalone_1bit` enforces this bit-for-bit equality.
   - **4-bit**: ~1.2% recall penalty vs an independently-optimized 4-bit codebook.
   - **2-bit**: ~10% recall penalty (worst level — the inner Lloyd-Max boundaries don't align with sign-based partitioning).

3. **ADC for memory efficiency** — The lookup table `(d, 2^bits)` is tiny (~6KB for 2-bit d=384). Chunked scoring keeps temporary allocation at ~6MB regardless of corpus size.

4. **GPU is a wrapper, not a fork** — `GPUSearcher` wraps `Quantizer + CompressedVectors` rather than replacing them. The core stays pure NumPy.

## Testing

- All tests in `tests/` directory, run with `pytest`
- Tests use `np.random.default_rng(seed)` for reproducibility
- ADC tests verify exact match with cached search (same top-k, same scores to rtol=1e-5)
- GPU tests run against numpy fallback backend (no GPU required in CI)
- Matryoshka tests cover nesting correctness, precision bounds, and two-stage recall

## Common tasks

**Adding a new search method**: Add to `Quantizer` in `core.py`, add corresponding method in `GPUSearcher` in `gpu.py`, add tests in `test_adc_gpu.py`.

**Changing codebook generation**: Modify `codebook.py`. Run `test_polar_embed.py::TestCodebook` and `test_matryoshka.py::TestNestedCodebooks` — they verify symmetry, monotonicity, and nesting properties.

**Changing bit-packing**: Modify `packing.py`. The `TestPacking` class in `test_polar_embed.py` runs roundtrip tests for all 1-8 bit widths.
