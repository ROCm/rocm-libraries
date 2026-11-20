# MFMA Thread-Value Layouts

The matrix cores on AMD Instinct GPUs use **MFMA** (matrix
fused-multiply-add) instructions to perform one or more matrix
multiplications per wavefront with very high throughput. Each
instruction executes at the wave level, with all 64 lanes
co-operating: every operand element of every block lives in a specific
VGPR of a specific lane, and the matrix core expects them in exactly
that arrangement.

The mapping from `(lane, vgpr)` → matrix coordinate is
instruction-specific and not particularly intuitive. Following CuTe, we
encode it as a **thread-value (TV) layout**:

```
TV : (tid, vid) → matrix-coord
```

In mosaic this is just a single `Layout(shape, stride)` with
hierarchical sub-tuples — the *outer* shape splits into a `tid`-mode
and a `vid`-mode, each of which can itself be hierarchical to capture
the lane- and VGPR-level decomposition. There is no separate
"register layout" or "matrix layout" — one TV layout *is* the
description of the operand distribution.

## Worked example: `v_mfma_f32_32x32x8_f16`

The TV layouts for the three operands are:

| Operand | Shape | Layout `Layout(shape, stride)` | Matrix size |
|---|---|---|---|
| A | M×K = 32×8 | `((32, 2), 4) : ((1, 128), 32)` | 32×8, col-major flat |
| B | N×K = 32×8 | `((32, 2), 4) : ((1, 128), 32)` | 32×8, col-major flat |
| D | M×N = 32×32 | `((32, 2), (4, 4)) : ((32, 4), (1, 8))` | 32×32, col-major flat |

Reading the D layout: the `tid` mode `(32, 2):(32, 4)` says lanes 0–31
each own a column-block of 32 cells (stride 32 along the col-major
flat), and lanes are split into 2 col-blocks (the second 32 lanes are
4 columns over). The `vid` mode `(4, 4):(1, 8)` says the 16 VGPRs per
lane are 4 row-groups of 4 contiguous rows each, with row-groups 8
rows apart.

The result of `D_tv(tid, vid)` is a column-major flat index into the
32×32 D tile (`flat = j * rows + i`); decode with `mosaic::idx2crd` to
recover `(row, col)`.

In code, the layouts are constructed directly:

```cpp
auto D_tv = mosaic::Layout(
    mosaic::make_shape (mosaic::make_shape(32u, 2u), mosaic::make_shape(4u, 4u)),
    mosaic::make_stride(mosaic::make_stride(32u, 4u), mosaic::make_stride(1u, 8u)));
```

See [mfma_layout.cpp](mfma_layout.cpp) for the full demo, including
A and B, a worked `(lane, vid) → (row, col)` walk, a bijectivity
check that every operand cell is touched exactly once by the wave,
and a ground-truth check against AMD's matrix calculator (see below).

## Ground-truth verification

The example can also check the TV layouts against the
[AMD matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
for every `(lane, vid)` entry. Place the calculator's
`--matrix-layout --csv` outputs for `v_mfma_f32_32x32x8_f16` under
`matrix_calculator_layouts/CDNA3_v_mfma_f32_32x32x8_f16/` and the demo
parses each file, checking that `L_tv(lane, vid)` produces the same
operand cell `X[i][j]` (modulo B's k-major → N-major canonicalization).
Those dumps are not committed. Without them the demo skips the check
and still runs the bijectivity walk.

## Calling convention

`L(tid, vid)` accepts plain scalar lane and value indices
(`L(5, 2)`). Internally mosaic decomposes each scalar against its
sub-shape via `idx2crd`, then takes the inner product with the
corresponding sub-stride
(see [include/mosaic/stride.hpp:79-107](../../../include/mosaic/stride.hpp)).
That means there is no ergonomic penalty for the hierarchical TV
form — callers do not need to construct nested coord tuples.
