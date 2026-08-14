# Changelog

## Unreleased

### Added

- **Scalar mode — `Quantizer(d, bits, normalize=False, scale=1.0)`**
  ([#77](https://github.com/oaustegard/remex/issues/77)). Quantizes the
  (optionally rotated) coordinates directly with the Lloyd-Max codebook:
  no unit-sphere factorization, no norms computed or stored, codes are the
  whole output. For use cases where the codes *are* the product — exact-match
  hash keys for a join, bucketing, dedup — the default pipeline works against
  you twice: unit-norm factorization maps every constant-direction family onto
  a single direction code, and the float32 cast that buys `.pq` parity with
  the Mojo port turns large values into `inf`. Scalar mode drops the first and
  works in float64, saturating at the outermost cell instead. Lloyd-Max cell
  shaping, Matryoshka nesting and determinism are unchanged.

- **`rotation="none"`** — the identity, on-disk code 2. Intended for scalar
  mode, where it makes a code a bare `searchsorted` of the input value: no
  matmul, so identical inputs give identical codes across BLAS builds and
  thread counts, and coordinate *j* of the code depends only on coordinate
  *j* of the input.

- **`sigma` on `lloyd_max_codebook` / `nested_codebooks`**, plus
  `coordinate_sigma(d, sigma)` — the coordinate spread the cells are cut for.
  `None` remains the unit-sphere `1/sqrt(d)`, bit-for-bit as before.

- **`has_norms`** on `CompressedVectors` / `PackedVectors`.

### Changed

- `norms` is now optional on `CompressedVectors` and `PackedVectors`, and its
  absence is how every serializer records scalar mode: no `norms` entry in
  `.npz`, no `norms` column in Arrow, and a no-norms flag at `.pq` byte 18
  bit 0 (previously reserved-and-zero, so existing files are unaffected).
  A scalar-mode `.pq` is *shorter* than an old reader's arithmetic expects, so
  such a reader — the Mojo port included — fails it as truncated rather than
  reading indices as norms. `save_params` rejects a scalar quantizer for the
  same reason.

- Decoding or searching codes with a quantizer in the other mode now raises
  `mode mismatch`, alongside the existing `rotation mismatch` check. The two
  codebooks differ by a factor of `scale * sqrt(d)`, so crossing them would
  otherwise rescale every reconstructed coordinate.

The default path is untouched: same codes, same files, same
`(d, bits, seed, rotation)` determinism.

## v0.6.0 — 2026-08-04

Four months since v0.5.1 (2026-04-06), reconstructed from the 29 pull requests
merged in between. Entries below are grouped by the work they belong to rather
than listed per-PR; each links the PRs that carry the detail.

Three things dominate the release: a Mojo port of the quantizer that reaches
parity with the Python implementation, GPU kernels on Mojo 1.0 / Apple Metal,
and — landing last — construction that is two orders of magnitude faster plus a
rotation identity that is finally recorded on disk.

### Added

- **Mojo port of `Quantizer`** — encode and ADC search
  ([#36](https://github.com/oaustegard/remex/pull/36)), `decode()` and
  `PackedVectors` ([#44](https://github.com/oaustegard/remex/pull/44)),
  Matryoshka nested codebooks and `search_twostage`
  ([#43](https://github.com/oaustegard/remex/pull/43)), and `IVFCoarseIndex`
  at parity with Python ([#63](https://github.com/oaustegard/remex/pull/63)).
  A NumPy-bit-identical `--seed` and a deterministic Haar in Python
  ([#46](https://github.com/oaustegard/remex/pull/46)) are what make "parity"
  checkable rather than asserted.

- **GPU / Metal execution.** Scaffolding for `--device gpu`
  ([#45](https://github.com/oaustegard/remex/pull/45)), then encode and ADC
  kernels on Mojo 1.0 and Apple Metal
  ([#65](https://github.com/oaustegard/remex/pull/65)), a persistent
  `GPUCorpus` for search with coalesced `Rᵀ` reads
  ([#66](https://github.com/oaustegard/remex/pull/66)), and `--device auto`
  for per-stage device routing
  ([#68](https://github.com/oaustegard/remex/pull/68)).

- **`IVFCoarseIndex`** — data-oblivious IVF over the Matryoshka coarse tier
  ([#58](https://github.com/oaustegard/remex/pull/58)), with `precision=1`
  extraction covered in tests, bench and docs
  ([#56](https://github.com/oaustegard/remex/pull/56)).

- **Opt-in randomized Hadamard rotation** (`rotation="rht"`), O(d² log d)
  instead of Haar's O(d³)
  ([#71](https://github.com/oaustegard/remex/pull/71)), implemented in Mojo
  byte-identically to Python
  ([#73](https://github.com/oaustegard/remex/pull/73)). Haar remains the
  default.

- **CI** ([#74](https://github.com/oaustegard/remex/pull/74)) — including tests
  for the documentation's own claims, which is what makes the rotation gate
  below mean anything.

- **Benchmark caches and results** — SPECTER2 embeddings fetcher
  ([#57](https://github.com/oaustegard/remex/pull/57)), n=10k 1-bit Matryoshka
  results ([#59](https://github.com/oaustegard/remex/pull/59)), narrowed cache
  with real IVF/bridge numbers
  ([#62](https://github.com/oaustegard/remex/pull/62)), a
  `gemini-embedding-001` cache
  ([#64](https://github.com/oaustegard/remex/pull/64)), SPECTER2 distribution
  analysis ([#32](https://github.com/oaustegard/remex/pull/32)), and Mojo
  bench results in `mojo/bench/RESULTS.md`
  ([#48](https://github.com/oaustegard/remex/pull/48),
  [#54](https://github.com/oaustegard/remex/pull/54)).

- **Research notes** — mixed-precision quantization via residual-error tail
  selection ([#35](https://github.com/oaustegard/remex/pull/35)), Matryoshka ×
  scalar quantization for byte-optimal retrieval
  ([#70](https://github.com/oaustegard/remex/pull/70)), and canonical
  references with ADC defined
  ([#33](https://github.com/oaustegard/remex/pull/33)).

### Packaging

- **The Mojo port is not in the PyPI distribution.** `.mojo` sources are not
  installable by pip and Mojo is not a declarable dependency, so
  `remex.mojo*` is now excluded from `packages.find`. Measured on the 0.6.0
  build: before the exclude the wheel carried **zero** `.mojo` files but did
  ship `remex/mojo/bench/compare.py` and two test-fixture builders — a package
  path present in the wheel with its actual content absent. After, both wheel
  and sdist contain zero `mojo` entries and eight `remex/*.py` modules. Run
  the Mojo and GPU paths from a checkout.

- **`__version__` is resolved from installed metadata.** It was the hardcoded
  literal `"0.5.0"` — stale even against the v0.5.1 already on PyPI — so a
  0.6.0 wheel reported 0.5.0 on import. Now `importlib.metadata.version`, with
  `"0.0.0+unknown"` for a bare checkout. Same fix remax made in its v0.1.0.

### Changed

- **The rotation is recorded in every container, and enforced on read**
  ([#74](https://github.com/oaustegard/remex/pull/74)). Containers recorded
  `(d, bits, n)` and sometimes `seed`, never the rotation, so a reader fell
  back on whatever `Quantizer`'s default happened to be at read time. Flipping
  that default would have decoded every stored index in a frame its codes were
  never written in — nothing raises, `search()` still returns k neighbours, and
  they are the wrong k. #73 made this reachable rather than theoretical by
  letting rht-encoded indexes exist. **An absent record resolves to `"haar"`,
  never to the live default**, so indexes written before this release keep
  decoding correctly.

- **`Quantizer.__init__` is 100–200× faster**
  ([#71](https://github.com/oaustegard/remex/pull/71)). `lloyd_max_codebook`
  called scipy's *scalar* `norm.cdf` / `norm.pdf` 307,200 times; the per-level
  loop vectorizes exactly, because `bounds` is materialised before any centroid
  moves, so every cell reads the same boundaries. Bit-identical output.

  | d | before | after | |
  |---|---|---|---|
  | 768 | 30.6 s | 0.22 s | 139× |
  | 1536 | ~45 s | 0.39 s | ~115× |
  | 3072 | ~358 s | 1.79 s | ~200× |

  *(with `rotation="rht"`; at the Haar default, d=3072 is 208 s → still
  dominated by the QR.)*

- **Mojo hot paths row-blocked at NB=8** — the rotation matvec
  ([#47](https://github.com/oaustegard/remex/pull/47)) and coarse-stage ADC
  scoring ([#52](https://github.com/oaustegard/remex/pull/52)) — plus a
  heap-based coarse top-k
  ([#51](https://github.com/oaustegard/remex/pull/51)) and a SIMD-vectorized
  encode hot path ([#37](https://github.com/oaustegard/remex/pull/37)).

### Reverted

- **The encode `Rᵀ` transpose** ([#67](https://github.com/oaustegard/remex/pull/67)),
  which cost 1.8× on Apple Metal. The `GPUCorpus` search work from the same
  line was kept.

## v0.5.1 — 2026-04-06

See the [release](https://github.com/oaustegard/remex/releases/tag/v0.5.1) and
the pull requests up to that tag. This file starts at v0.6.0; earlier history
was not reconstructed.
