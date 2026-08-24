# remex

Retrieval-validated embedding compression: quantize float32 embedding matrices to
1-8 bit codes and search them without decompressing.

## works when

- pyproject.toml exists at root
- remex/rotation.py exists at root

## why

The root establishes how the package is built and which module owns the on-disk
format contract. Implementation claims belong to the component specs below it.
