# mfma — Python port of `examples/cpp/mfma/`

Encodes AMD MFMA layouts as CuTe-style thread-value (TV) layouts using
mosaic's hierarchical `(shape, stride)` tuples, then verifies the
encoding against AMD's matrix instruction calculator.

A TV layout maps `(tid, vid) -> matrix-coord`, where `tid` is the
lane in the wavefront (0..63 for wave64) and `vid` is the per-lane
VGPR id. The matrix-coord is a col-major flat index into the operand
tile (`flat = j * rows + i`).

This port pins down the TV layouts for `v_mfma_f32_32x32x8_f16`:

| Operand | Layout                                              | Matrix size       |
| ------- | --------------------------------------------------- | ----------------- |
| A       | `((32, 2), 4) : ((1, 128), 32)`                     | 32x8, col-major   |
| B       | `((32, 2), 4) : ((1, 128), 32)`                     | 32x8, col-major   |
| D       | `((32, 2), (4, 4)) : ((32, 4), (1, 8))`             | 32x32, col-major  |

## Run

```bash
# From the repo root
PYTHONPATH=python/pymosaic python examples/python/pymosaic/mfma/mfma_layout.py     # text + ground-truth check
PYTHONPATH=python/pymosaic python examples/python/pymosaic/mfma/visualize.py       # PNG into figures/
```

## Visualization

`visualize.py` writes `figures/mfma_tv_layouts.png` when you run it.
That image is not committed.

Three operands of `v_mfma_f32_32x32x8_f16` rendered as "which lane
owns which cell" heatmaps. A and B are 32x8 (M x K and N x K
respectively); D is the 32x32 accumulator. Cell colour = lane id
(0..63) using a continuous palette so adjacent lanes are visually
distinguishable. The chosen lane's cells in D are annotated with
their per-lane VGPR id (vid 0..15) — you can see how a single lane
holds 16 D-tile cells scattered across the matrix.

Try `python visualize.py --annotate-lane 17` to see a different lane.

The script will:

1. Print the three TV layouts.
2. Walk a worked example — where does `D_tv[lane=5, vid=2]` land?
3. Assert every operand cell is covered exactly once by the
   wavefront's 64 lanes x `vals_per_lane` VGPRs (bijectivity check).
4. If you place AMD's `--matrix-layout --csv` outputs under
   `examples/cpp/mfma/matrix_calculator_layouts/CDNA3_v_mfma_f32_32x32x8_f16/`,
   verify every `(lane, vid) -> operand cell` mapping matches.
   Those dumps are not committed; without them this step is skipped.

## Note on B's axis swap

B's TV layout canonicalises to `(rows=N, cols=K)`, but AMD's
calculator labels B as `B[k][n]` (k-major). The CSV parser
ground-truths B with `swap=True` to compensate. See the
`verify_against_calc` helper for details.
