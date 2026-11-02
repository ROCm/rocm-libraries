# basic_tiling — Python port of `examples/cpp/basic_tiling/`

Demonstrates a 2-level workgroup tiling built with `blocked_product`.
An 8x8 grid of workgroups is remapped so that consecutive IDs walk
within the same 2x2 tile before moving to the next tile.

The cpp version runs an actual HIP kernel that records the schedule
from the GPU; this port stays on the host because the schedule itself
is a pure function of the layout — no per-thread work is needed to
demonstrate it.

## Run

```bash
# From the repo root
PYTHONPATH=python/pymosaic python examples/python/pymosaic/basic_tiling/check_tiling.py   # text output
PYTHONPATH=python/pymosaic python examples/python/pymosaic/basic_tiling/visualize.py      # PNG into figures/
```

## Visualization

`visualize.py` writes `figures/tiling_schedule.png` when you run it.
That image is not committed.

Left panel is the default row-major schedule; right panel is the
`blocked_product`-derived one. Consecutive workgroup IDs in the right
panel cluster in 2x2 tiles, and the tiles themselves are visited in
column-major order across the grid.

## What you'll see (text)

Two 8x8 grids printed:

1. `map[transform(wg_id)] = wg_id` — at each tile position, which
   original wgid ends up here.
2. `map[wg_id] = transform(wg_id)` — for each consecutive wgid, where
   it lands in the grid.

With `inner = row_major(2, 2)` and `outer = col_major(4, 4)`, the
mapping permutes the 64 workgroup IDs into a tile-locality-preserving
order, and the layout is asserted bijective onto `[0, 64)`.

## Notes

`blocked_product` is not yet exposed through the Python JIT
(`Layout.blocked_product` and a free-function form are planned —
see the docstring at `python/mosaic/layout.py:9-12`). This example
inlines a 10-line pure-Python port that mirrors the C++ template at
`include/mosaic/algebra.hpp:462-499`. Once the Python binding lands,
swapping the inline helper for the official call should be a no-op.
