# Conv Backward Data (dgrad) — Implicit-GEMM Instance

Computes the input gradient of a 2-D convolution:

```
dX[n, hi, wi, c] = sum_{y, x, k} dY[n, ho, wo, k] * W[k, y, x, c]
```

## GEMM Orientation

| Dim | Expression | Operand |
|-----|-----------|---------|
| M | `N * Hi * Wi` | rows of dX |
| N_dg | `C` | cols of dX |
| K_dg | `Y * X * K` | reduction |

- **A** = `dY` (NHWK) — output gradient
- **B** = `W` (KYXC) — weight tensor
- **D** = `dX` (NHWC) — input gradient (output of this kernel)

## Tilde Decomposition

For stride > 1 the backward convolution decomposes into
`y_tilde × x_tilde` independent sub-GEMMs, where:

```
y_tilde = sH / gcd(sH, dH)
x_tilde = sW / gcd(sW, dW)
```

For stride=1, dilation=1: `y_tilde = x_tilde = 1` → single sub-GEMM.

### Why tilde decomposition?

When stride > 1 not all `(hi, y)` pairs produce a valid output row `ho`:

```
ho = (hi + pH - y * dH) / sH   (must be an integer and in [0, Ho))
```

The tilde decomposition partitions the filter positions `y` into `y_tilde`
groups such that within each group the integrality constraint is always
satisfied. Each group becomes one independent sub-GEMM.

## Pipeline Variants

| Pipeline | Description |
|----------|-------------|
| `mem` | Single-buffer LDS, synchronous loads, no scheduler hints. Default. |
| `wavelet` | Load/math wave specialization for **gfx1250/WMMA only**. Extra `num_load_waves` waves handle all DRAM→LDS transfers while the `warp_m × warp_n` math waves run WMMA exclusively. Requires gfx1250's separate VMEM and WMMA issue slots to achieve true hardware concurrency. Incompatible with `async_dma=True`, `split_k > 1` and `lds_k_outer=True`. Single-buffer LDS shared by both roles; synchronization via a `barrier_0 / barrier_A / barrier_B` protocol. |

The MFMA/CDNA pipelines (`compv3`, `compv4`) are not supported for dgrad; `is_valid_spec` rejects them.

## Kernel Architecture

All convolutions — stride=1 and strided — use a **single unified tiled kernel**:

- The host packs per-sub-GEMM constants into `sub_gemm_buf` (a flat `i32` array).
- Each CTA binary-searches the buffer to find its sub-GEMM and loads record fields.
- The K-loop uses runtime descriptor closures that compute `dY` and `W` offsets
  from the record's coefficients.
- **Epilogue dispatch** based on `needs_atomic`:
  - `False` (1 sub-GEMM, split_k=1): direct `buffer_store` into `dX`.
  - `True` (stride > 1 or split_k > 1): `global_atomic_fadd` into `dX`
    (caller must zero-initialise `dX` before launch).

### Sub-GEMM record layout (22 × i32)

| Field | Index | Description |
|-------|-------|-------------|
| `block_start` | 0 | first flat tile index for this sub-GEMM |
| `num_m_tiles` | 1 | M-tile count |
| `num_n_tiles` | 2 | N-tile count |
| `gemm_m` | 3 | `N * HTildeSlice * WTildeSlice` |
| `gemm_k` | 4 | `YDotSlice * XDotSlice * K` |
| `h_tilde_slice` | 5 | HTildeSlice |
| `w_tilde_slice` | 6 | WTildeSlice |
| `h_tilde_slice_begin` | 7 | HTildeSliceBegin |
| `w_tilde_slice_begin` | 8 | WTildeSliceBegin |
| `y_dot_slice` | 9 | YDotSlice |
| `x_dot_slice` | 10 | XDotSlice |
| `a_embed_h_coeff` | 11 | `ho = htl + h_begin + ydot * coeff_h` |
| `a_embed_w_coeff` | 12 | `wo = wtl + w_begin + xdot * coeff_w` |
| `b_y_stride` | 13 | `y = ydot * b_y_stride + b_y_offset` |
| `b_y_offset` | 14 | |
| `b_x_stride` | 15 | `x = xdot * b_x_stride + b_x_offset` |
| `b_x_offset` | 16 | |
| `d_h_stride` | 17 | `hi = htl * d_h_stride + d_h_offset` |
| `d_h_offset` | 18 | |
| `d_w_stride` | 19 | `wi = wtl * d_w_stride + d_w_offset` |
| `d_w_offset` | 20 | |
| `gemm_k_padded` | 21 | padded K for split-K |

## Kernel ABI

```
(dY, W, dX, dY_bytes, W_bytes, dX_bytes, sub_gemm_buf, num_sub_gemms)
```

All kernels — stride=1 and strided — share this 8-param ABI. For stride=1
`sub_gemm_buf` holds exactly one record and the binary search trivially
returns index 0.

## Grid Layout

```
grid = (flat_tiles, 1, split_k)
```

where `flat_tiles = sub_gemms[-1].block_end` (sum of all sub-GEMMs' tile counts).

## Split-K

When `split_k > 1` the K reduction is partitioned across `split_k` Z-grid CTAs.
The caller must zero-initialise `dX`. Supported dtypes:

- `fp32` — scalar `global_atomic_add` (f32 fadd)
- `bf16` — packed `global_atomic_fadd_v2bf16` (`<2 x bfloat>`)
- `fp16` — packed `global_atomic_fadd_v2f16` (`<2 x half>`)

## Vector Loads

Both A and B tiles are loaded via `CoalescedTileLoader` with dtype-aware vector
widths. The widths are derived from `DgradConvSpec.default_vector_sizes(C, K, dtype)`
which returns `(vec_a, vec_b, vec_c)`.

### A (dY, NHWK)

`k_out` is the innermost index of the GEMM-K decomposition, which maps
contiguously onto the last dim of `dY` (dim K). Vector width is therefore
constrained by `K % load_vec_a == 0`. Split-K forces `load_vec_a = 1` because
the per-CTA K-slice boundary may not be K-aligned.

The loader uses `vector_axis="col"` (the standard column-axis path).

### B (W, KYXC)

The GEMM row axis is `N_dg = C` (input channels), which is the **stride-1** axis
of `W` in `KYXC` layout. Vector loads therefore go along the free (row) axis and
the loader transposes the tile into row-major LDS layout on store — exactly the
same mechanism used by wgrad for its B operand (`X`, `NHWC`).

That transpose-on-store is what `lds_k_outer` removes; see
[LDS Tile Layout](#lds-tile-layout-lds_k_outer) below. Under `lds_k_outer=True`
this tile is stored K-outer and the scatter disappears, but the width selection
below is unchanged.

Constraint: `C % load_vec_b == 0`. The loader uses `vector_axis="row"`.

Width selection (Python / C++):

1. Compute `max_from_C` — largest power-of-two dividing `C` up to 8 (fp16/bf16)
   or 4 (fp32) from `default_vector_sizes`.
2. Call `CoalescedTileLoader.choose_vec(tile_rows=block_n, tile_cols=block_k, ...,
   max_vec=max_from_C, vector_axis="row")`.
3. If `spec.vector_size_b` is set explicitly, use that; else use the chosen value
   if `> 1`, otherwise fall back to 1 with `vector_axis="col"`.

### D (dX, NHWC)

The epilogue writes `dX` whose last dim is also `C`. Store vector width follows
`C % store_vec == 0`, derived from `default_vector_sizes` (third element).

## LDS Tile Layout (`lds_k_outer`)

`lds_k_outer=True` stores the **B tile only** K-outer (`LDS[k][n]`, row stride
`block_n + _KOUTER_PAD` with `_KOUTER_PAD = 8`) and recovers the MFMA/WMMA
operand layout with transpose reads, instead of transposing on store.

**Why B only.** This is a deliberate asymmetry with wgrad, which flips both
operands:

- **B (`W`, KYXC)** has the GEMM free axis `c` stride-1, forcing
  `vector_axis="row"` and a per-element `ds_write_b16` scatter whose inter-lane
  dword delta is a multiple of the 32-dword bank period. This is the cost worth
  removing.
- **A (`dY`, NHWK)** already has a stride-1 reduction axis (`k_out` innermost),
  so its loader is already `vector_axis="col"` with one wide `smem_store_vN`, and
  its M-outer fragment read is already conflict-free via `lds_k_pad`. Flipping A
  would put the global vector along `m = (n, hi, wi)` — stride `K` in NHWK — and
  destroy coalescing for zero write-side gain.

**Instruction effect on the B side.** The store collapses to one wide
`smem_store_vN` (`b128` for a 16-bit 8-wide vector), dropping `load_vec − 1`
address adds and `load_vec` `vec_extract`s per chunk. The read pays `n / 4`
`ds_read_b64_tr_b16` (wave64) or `n / 8` `ds_load_tr16_b128` (wave32) for a
per-lane fragment length `n`, where the M-outer path issued a single
`smem_load_vN`: **+1** read per fragment on the `n = 8` atoms (`32x32x16`,
`16x16x32`), exactly **zero** on the `n = 4` atoms (`16x16x16`, `32x32x8`).
Global `buffer_load`s are unchanged.

**Gating.** Two regimes, `_LDS_K_OUTER_ARCH_WAVE = {"gfx950": 64,
"gfx1250": 32}`, each arch pinned to its wave size so a mismatched spec is
rejected rather than emitting a lane formula the hardware does not implement.
gfx950 wave64 admits `warp_tile_n ∈ (16, 32)`; gfx1250 wave32 admits only the
`16x16x32` atom (whose 16-element fragment is two `ds_load_tr16_b128` reads).
Plus 16-bit B. Rejected with `async_dma=True` (the tilde builder has no direct
global→LDS path) and with `pipeline="wavelet"` (`build_wavelet_loaders` pins the
B tile to `(block_n, block_k)` and takes the unswapped descriptor, so it would
write M-outer into a K-outer allocation).

**Not a knob.** The spec field defaults `False`; the value is deduced by
`DgradConvSpec.default_lds_k_outer(arch, dtype_b, warp_tile_n, cpg, wave_size,
pipeline)`, which both library dispatch (`library/dispatch/grouped_convolution.py`,
`_dgrad_lds_k_outer`) and the sweep driver call. The predicate is
asymmetric with wgrad's — B-side dtype and warp tile only, never the A-side
counterparts — and additionally keys on `cpg`: the saving is proportional to the
B load width, which collapses to 1 on an odd channel run, where `axis_b` is
already `"col"` and there is no scatter to remove.

## Key Files

| File | Purpose |
|------|---------|
| `conv_implicit_gemm_dgrad.py` | Python builder (this instance) |
| `../../benchmark/benchmark_implicit_gemm_conv.py` | `--direction dgrad` sweep |
| `../../benchmark/conv_reference.py` | `dgrad_reference()` via `torch.nn.grad.conv2d_input` |
| `../../dispatch/families/conv_dgrad.py` | Dispatcher family |
| `../../../cpp/instances/common/conv_implicit_gemm_dgrad.cpp` | C++ port (byte-identical) |
| `../../../cpp/include/rocke/instance_conv_implicit_gemm_dgrad.h` | C99 header |
| `../../tests/instances/parity/conv_implicit_gemm_dgrad_emit.{c,py}` | C-vs-Python parity emitters |

## Differences from Wgrad

| Aspect | Wgrad | Dgrad |
|--------|-------|-------|
| Number of GEMMs | 1 | `y_tilde × x_tilde` |
| GEMM-M | `K` | `N * HTildeSlice * WTildeSlice` |
| GEMM-N | `Y*X*C` | `C` |
| GEMM-K | `N*Ho*Wo` | `YDotSlice * XDotSlice * K` |
| A operand | `dY` (NHWK) | `dY` (NHWK) |
| B operand | `X` (NHWC) — reuses fwd A desc | `W` (KYXC) |
| Output | `dW` (KYXC) | `dX` (NHWC) |
| Tilde decomposition | Not needed | Required for stride > 1 |
| Output accumulation | Atomic only when split_k > 1 | Atomic when num_sub_gemms > 1 OR split_k > 1 |
| `lds_k_outer` scope | Flips **both** A and B | Flips **B only** (A already has a stride-1 reduction axis) |
| `lds_k_outer` + `async_dma` | Required together | Mutually exclusive |
