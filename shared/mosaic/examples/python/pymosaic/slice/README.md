# slice — Python port of `examples/cpp/slice/`

Demonstrates mosaic's CuTe-style underscore slicing using the Python
`Layout` API, where the cpp's `_` placeholder is spelled `:` (full
slice).

| cpp form                                | Python form                |
| --------------------------------------- | -------------------------- |
| `L(_, k)`                               | `L[:, k]`                  |
| `L(i, _)`                               | `L[i, :]`                  |
| `slice(make_coord(_, k), L)`            | (same as `L[:, k]`)        |
| `dice(make_coord(_, k), L)`             | `L.dice((slice(None), k))` / `dice(coord, L)` |
| `L(_, make_coord(1, 1))` (hierarchical) | `L[:, (1, 1)]`             |

## Run

```bash
# From the repo root
PYTHONPATH=python/pymosaic python examples/python/pymosaic/slice/slice.py        # text
PYTHONPATH=python/pymosaic python examples/python/pymosaic/slice/visualize.py    # PNG into figures/
```

First run JIT-compiles a `.so` per unique `(shape, stride)`; subsequent
runs hit the on-disk cache.

## Visualization

`visualize.py` writes `figures/slice_demo.png` when you run it.
That image is not committed.

Left: the full 4x8 column-major layout, each cell labelled with its
offset, column 3 outlined in red. Right: the kept-mode slice
``L[:, 3]`` (the "slice-only" offsets, starting from 0) vs the
caller's recovered absolute offsets ``L[:, 3] + base`` — the
intended use site.

## Notes vs. the cpp version

* `dice()` is exposed as `L.dice(coord)` (method) and `dice(coord, L)`
  (free function, mosaic arg order). It keeps the *fixed* modes — the
  complement of slicing.
* Hierarchical slicing works: `L[:, (1, 1)]` fixes a nested mode with a
  nested coord, and `L[(0, :), k]` keeps the inner `:` leaf of a nested
  mode (kept inner leaves flatten up, matching mosaic's `slice`).
* Only the full-axis slice `:` is supported; range slicing (`L[0:2]`)
  raises `NotImplementedError`.
