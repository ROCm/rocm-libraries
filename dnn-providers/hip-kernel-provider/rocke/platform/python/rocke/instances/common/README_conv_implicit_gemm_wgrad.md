# Implicit-GEMM Backward-Weight Convolution (Wgrad)

## Algorithm

The wgrad kernel computes the weight gradient of a 2-D (or 3-D) convolution:

```
dW[k, y, x, c] = sum_{n, ho, wo} dY[n, ho, wo, k] * X[n, hi, wi, c]
  where  hi = ho*sH - pH + y*dH,  wi = wo*sW - pW + x*dW
```

This is cast as an implicit-GEMM with the following dimension mapping:

| GEMM dim | Wgrad meaning                   | Size        |
|----------|---------------------------------|-------------|
| M        | Output channels (weight rows)   | K           |
| N        | Filter spatial × input channel  | Y×X×C       |
| K (red.) | Output spatial positions        | N×Ho×Wo     |

### Operands

| Role | Tensor | Layout | GEMM shape         |
|------|--------|--------|--------------------|
| A    | dY (output gradient)    | NHWK  | (K_wg, M)ᵀ = (N·Ho·Wo, K) |
| B    | X  (input activations)  | NHWC  | (K_wg, N_wg) = (N·Ho·Wo, Y·X·C) |
| D    | dW (weight gradient)    | KYXC  | (M, N_wg) = (K, Y·X·C) |

### Descriptor reuse

The B descriptor for X reuses `make_a_descriptor` from `_conv_implicit_gemm_common`. The convolution address map for X is identical to the forward pass A operand — `k_wg` plays the role of the forward `m` (output spatial position) and `n_wg` plays the role of the forward `k` (filter+channel index). This avoids duplicating the convolution affine-embed and boundary-pad transform chain.

### Epilogues

Four epilogue paths are supported, selected automatically based on `spec.epilogue` and `spec.split_k`:

| Path | Condition | Output |
|------|-----------|--------|
| Direct store | `epilogue="default"`, `split_k=1` | Per-lane scalar write to dW via the KYXC descriptor |
| CShuffleEpilogue | `epilogue="cshuffle"`, `split_k=1` | LDS-staged vectorised store |
| Split-K direct atomic | `epilogue="default"`, `split_k > 1`, `dtype_d="fp32"` | `global_atomic_add` per MFMA-lane directly from accumulators |
| Split-K cshuffle atomic | `epilogue="cshuffle"`, `split_k > 1` | LDS scatter + `global_atomic_add` / `global_atomic_add_pk_bf16` / `global_atomic_add_pk_f16` from LDS |

For bf16/fp16 output `epilogue="cshuffle"` is **required** when `split_k > 1`.
The direct-atomic path emits zero-filled packed atomics at the scattered MFMA
layout; the cshuffle path produces genuinely contiguous adjacent pairs after the
LDS shuffle, which is necessary for correct `<2 x dtype>` packed atomics.

---

## Changelog

### Initial implementation

- Introduced `WgradConvSpec`, `build_implicit_gemm_conv_wgrad`, and the three
  tensor descriptors (`make_dy_descriptor`, `make_x_wgrad_descriptor`,
  `make_dw_descriptor`).
- Supports 2-D and 3-D convolutions, `mem` / `compv4` / `async_dma` pipelines,
  `default` and `cshuffle` epilogues, and chiplet-aware workgroup swizzle.
- CDNA targets (gfx940+): fp16, bf16, fp32 dtypes.
- RDNA targets (gfx1151): WMMA path with `mem` pipeline and `default` epilogue.

### Split-K support

Split-K was added to address the reduction-heavy nature of the wgrad problem.
`K_wg = N·Ho·Wo` can be orders of magnitude larger than the `M×N` tile area
(e.g. K_wg = 25,088 vs M×N = 64×576 for a typical training shape), resulting in
a grid that is too small to fully saturate the device.

**Mechanism:**

- `split_k` partitions K_wg into equal slices along the Z grid dimension.
- Each CTA accumulates its partial f32 result and atomic-adds it directly into
  `dW` — no separate reduction kernel is needed.
- K_wg is zero-padded to the next multiple of `tile_k × split_k`; out-of-range
  buffer loads return zero silently via the buffer descriptor OOB-clamp.

**Supported output dtypes:**

| dtype | Atomic instruction               | Requirement        |
|-------|----------------------------------|--------------------|
| fp32  | `global_atomic_add` (f32 fadd)   | gfx940+            |
| bf16  | `global_atomic_add_pk_bf16`      | gfx940+, even C    |
| fp16  | `global_atomic_add_pk_f16`       | gfx940+, even C    |

**Caller contract (`split_k > 1`):**

1. Zero-initialise the `dW` buffer before launch.
2. Launch with grid `(ceil(wg_N/tile_n), ceil(wg_M/tile_m), split_k)`.

The kernel ABI is identical across `split_k=1` and `split_k>1` — no extra
parameters are required.

**Auto mode (`split_k=-1`):** The split degree can be chosen automatically at
build time using `select_split_k_wgrad` (the CK formula:
`floor((waves_per_cu × num_cus) / base_grid)`, clamped to `[1, wg_K]`).

### Split-K cshuffle atomic epilogue

Added `CShuffleEpilogue.atomic_store` and `_emit_wgrad_split_k_cshuffle_epilogue`
to support split-K for bf16/fp16 output dtypes without the zero-fill artefact of
the direct-atomic path.

**Motivation:** The existing split-K epilogue (`_emit_wgrad_split_k_epilogue`)
emits one `global_atomic_add_pk_f16/bf16` per MFMA accumulator slot. Because
each slot's column index may be odd or even, the code resolves the column parity
at runtime and fills the unused half of the `<2 x dtype>` pair with zero. This
avoids touching a neighbour's data but produces two overlapping atomics for each
adjacent pair of lanes — one with `(val, 0)` and one with `(0, val)` — which
serialise on the same address and are wasteful.

The cshuffle path avoids this entirely:
1. MFMA accumulators are scattered to an LDS staging buffer in row-major order
   (identical to the non-atomic `CShuffleEpilogue.store` path).
2. After a barrier, each thread reads back an `sv`-wide chunk of consecutive
   N-position elements in one row.  Because the cshuffle guarantees row-major
   order, adjacent elements `(col, col+1)` are always in the same row and
   consecutive in N, forming a genuine `<2 x dtype>` pair — no zero-fill needed.
3. Paired `global_atomic_add_pk_bf16` / `global_atomic_add_pk_f16` are issued,
   one per pair.

**Constraints:**
- `epilogue="cshuffle"` is now **required** for bf16/fp16 split-K (enforced by
  `WgradConvSpec.validate()` and `is_valid_wgrad_spec`).
- `store_vec` (`sv`) must be even for bf16/fp16 (guaranteed by `from_grid` via
  the existing `cpg % 2 == 0` constraint).
- The caller must zero-initialise `dW` before launch (atomic-adds only).

The benchmark driver (`benchmark_implicit_gemm_conv.py`) was updated to generate
only `split_k=0` (runtime-atomic) combos by default instead of `(1, 0)`, so the
`epilogue="cshuffle"` requirement is respected without filtering cshuffle combos
out of the sweep.

### Pointwise explicit-GEMM fast path

For **pointwise convolutions** (`Y=X=1`, `sH=sW=1`, `pH=pW=0` — and for 3-D: `Z=1`, `sD=1`, `pD=0`) the wgrad kernel automatically bypasses the coordinate-transform descriptor DAG and replaces all three operand address computations with flat multiply-add arithmetic.

**Detection:** `ConvProblem.is_pointwise` returns `True`; no user-facing flag is needed.

**Address arithmetic (pointwise path):**

| Operand | Formula | Replaces |
|---------|---------|---------|
| dY (A) | `offset = k_wg_red * K + k_out` | `make_dy_descriptor` + `unmerge_magic` on K_wg |
| X  (B) | `offset = k_wg_red * C + n_wg`  | `make_x_wgrad_descriptor` + full conv DAG |
| dW (D) | `offset = k_out * C + n_wg`     | `make_dw_descriptor` + `unmerge_magic` + pads |

**Why this is faster:** For 1×1/s1/p0 the spatial affine map (embed), the filter-unmerge (unmerge_magic on y, x, c), and the boundary pads (pad on y, x) all collapse to identity. The implicit descriptor computes the same address but generates extra VALU instructions (multiplications, additions, comparisons) that the compiler cannot always eliminate. Flat arithmetic emits exactly one `mul` and one `add` per operand.

The split-K epilogue (`global_atomic_add` / `global_atomic_add_pk_*`) already used flat arithmetic (`c_m * wg_N + c_n`) and required no change.

---

## Changelog (continued)

### Free-axis vector loads for A and B

The K_wg reduction axis is not the innermost dimension of either input tensor:

- **A (dY, NHWK):** consecutive K_wg positions are separated by stride K
  (output channels); the stride-1 axis is `k_out` (= GEMM **M**, the free axis).
- **B (X, NHWC):** consecutive K_wg positions are separated by stride C (input
  channels); the stride-1 axis is the inner C of `N_wg` (the free axis).

A `buffer_load_vN` along K_wg would read N consecutive *channel* values at one
spatial position — wrong data — which is why the loads were historically scalar.
The fix rearranges the load tile so the vector runs along the **free** axis
(the last tensor dimension) instead of the reduction axis: the loader's new
`vector_axis="row"` mode (`helpers/loads.py`) issues one coalesced
`buffer_load_dwordx4` (V=8 fp16/bf16) along the stride-1 free axis, then
*transposes on store* — scattering the V elements into `[row+i, col]` of the
existing row-major `(M/N, K)` LDS tile. The MFMA consumer still reads that tile
K-contiguously, unchanged (no new LDS layout, no consumer-read change).

Enabled for the **sync CDNA-MFMA** path (`op.family == "mma"`): the width is
`vec_a | K` (A) and `vec_b | C` (B, so a vector never crosses a `(y,x)` filter
boundary), falling back to the scalar `vector_axis="col"` path (byte-identical)
when a width > 1 is not admissible. The async-DMA and WMMA paths are follow-ons
(see below). The optional `K0-M-K1` LDS layout below is an *additional*
bank-conflict/wider-`ds_read` optimization, independent of this vectorised load.

## Next steps

### Async DMA for all pipelines

`async_dma=True` works today but is gated to the software-pipelined (`unroll_k`)
path. The `mem` and `compv4` pipelines fall back to synchronous
`CoalescedTileLoader` because `raw_ptr_buffer_load_lds` writes a packed
lane-contiguous tile that is incompatible with non-zero `lds_k_pad`. To extend
async DMA to all pipelines the load path needs to either:

- Accept the packed layout and downstream adjust SMEM read indexing to match, or
- Introduce a padding-aware async path that inserts the `lds_k_pad` columns
  during the DMA itself.

### K0-M-K1 LDS layout

The current LDS layout stores tiles in `(M, K)` row-major order with a small
`lds_k_pad` column pad to break bank conflicts. A `K0-M-K1` layout (also called
the transposed or interleaved LDS layout, after the CK naming convention)
reorders the tile as `(K0, M, K1)` where `K = K0 × K1`. This means each MFMA
atom's K slice is contiguous in LDS, which eliminates the bank-conflict
cross-section that the current padding only partially mitigates and enables
wider ds_read instructions. Adding this layout requires:

1. A new `LdsLayout` variant that encodes the `(K0, M, K1)` stride formula.
2. Updated `CoalescedTileLoader` / `AsyncTileLoader` store-index calculations
   to write into the transposed shape.
3. Updated SMEM-load index expressions in `_emit_smem_load` /
   `_emit_frag_smem_load` to read from the transposed shape.
