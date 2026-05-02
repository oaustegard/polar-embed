#!/usr/bin/env bash
# Fetch precomputed SPECTER2 embeddings into bench/.specter2_cache/ so that
# `python bench/specter2_eval.py --cached` works without rerunning the
# ~50-minute CPU encode of the SPECTER2 transformer model.
#
# Cache contents (~90 MB total):
#   specter2_nlp_broad.npy          — (10000, 768) float32 embeddings, broad NLP
#   specter2_nlp_broad_texts.json   — original "title [SEP] abstract" strings
#   specter2_nlp_narrow.npy         — (10000, 768) float32 embeddings, narrow
#                                     subfield (transformer attention mechanism)
#   specter2_nlp_narrow_texts.json  — original texts for narrow partition
#
# Both partitions are required because bench/specter2_eval.py:main() runs
# the IVF cross-FoS bridge sweep with `bridge_corpus=corpus_narrow`. Without
# the narrow cache, `--cached --only-ivf` falls through to a ~50-minute
# encode for the narrow partition (issue #60).
#
# Source: oaustegard/claude-container-layers releases (free GH-hosted asset).
# Re-run safe; overwrites with `--clobber`.
set -euo pipefail

REPO=${SPECTER2_CACHE_REPO:-oaustegard/claude-container-layers}
BROAD_TAG=${SPECTER2_BROAD_TAG:-specter2-nlp-broad-10k}
NARROW_TAG=${SPECTER2_NARROW_TAG:-specter2-nlp-narrow-10k}
CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.specter2_cache"

mkdir -p "$CACHE_DIR"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found. Install: https://cli.github.com/" >&2
  exit 1
fi

fetch_assets() {
  local tag="$1"; shift
  for asset in "$@"; do
    out="$CACHE_DIR/$asset"
    echo "fetching $asset → $out"
    gh release download "$tag" --repo "$REPO" --pattern "$asset" \
      --output "$out" --clobber
  done
}

fetch_assets "$BROAD_TAG"  specter2_nlp_broad.npy  specter2_nlp_broad_texts.json
fetch_assets "$NARROW_TAG" specter2_nlp_narrow.npy specter2_nlp_narrow_texts.json

echo "done. run: python bench/specter2_eval.py --cached"
