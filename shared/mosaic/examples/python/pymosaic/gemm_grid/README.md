# gemm_grid — host-side GEMM *schedule* visualisation

This example uses mosaic's 3-level hierarchical workgroup schedule plus an
XCD-chiplet remap. There is no GPU kernel here. The **layout reasoning** —
which workgroup ID executes which `(pid_m, pid_n)` tile — is pure Python so
you can see what each strategy actually does.

## Run

```bash
# From the repo root
PYTHONPATH=python/pymosaic python examples/python/pymosaic/gemm_grid/grid_mapping.py   # text
PYTHONPATH=python/pymosaic python examples/python/pymosaic/gemm_grid/visualize.py      # PNG into figures/
```

## Visualization

`visualize.py` writes `figures/gemm_grid_schedule.png` when you run it.
That image is not committed.

Three panels on a 16x16 tile grid:

1. **Default row-major** — wgid increases left-to-right, top-to-bottom.
2. **Mosaic schedule** — `chiplet_transform_chunk` + 3-level
   `blocked_product`. The 8x4 L2 tiles and 2x4 L3 super-tiles are
   visible as block patterns.
3. **XCD assignment** — categorical colour per XCD (0..7). Each XCD
   owns a contiguous vertical pair of tile columns, so all workgroups
   touching the same A-row stay on the same chiplet's L2.

## What you'll see

For a 16x16 tile grid (the smallest grid that exercises the full L3
nesting) with `NUM_XCDS=8`:

1. The constructed 3-level layout's `(shape, stride)` after two
   nested `blocked_product` calls.
2. A side-by-side visualisation of the default row-major schedule and
   the mosaic schedule — the L2 tiles (8x4 contiguous wgids) and L3
   super-tiles (2x4 L2 tiles) are visible as solid blocks in the
   mosaic schedule.
3. An assertion that both schedules are bijections over the tile
   grid (every tile is executed by exactly one workgroup).
4. The XCD assignment for the first 32 workgroups under the chiplet
   remap — consecutive groups of 32 wgids land on the same XCD,
   keeping their working sets in the same chiplet's L2 cache.

## What this port does NOT do

* No actual GEMM — there's no kernel and no GPU work.
* `blocked_product`, `chiplet_transform_chunk`, and `gemm_grid_transform`
  are inlined as pure-Python helpers that mirror
  `include/mosaic/algebra.hpp:462-499` and `include/mosaic/algorithms.hpp:46-169`.
  Once the JIT exposes them, swap to the official APIs.
