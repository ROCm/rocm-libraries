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

Three epilogue paths are supported, selected automatically based on `spec.epilogue` and `spec.split_k`:

| Path | Condition | Output |
|------|-----------|--------|
| Direct store | `epilogue="default"`, `split_k=1` | Per-lane scalar write to dW via the KYXC descriptor |
| CShuffleEpilogue | `epilogue="cshuffle"`, `split_k=1` | LDS-staged vectorised store |
| Split-K atomic epilogue | `split_k > 1` | `global_atomic_add` / `global_atomic_add_pk_bf16` / `global_atomic_add_pk_f16` |

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

---

## Next steps

### Enable vector loads for A and B

Currently `load_vec_a` and `load_vec_b` are hard-coded to 1
(`conv_implicit_gemm_wgrad.py:822`). The root cause is that the K_wg reduction
axis is not the innermost dimension of either tensor:

- **A (dY, NHWK):** consecutive K_wg positions are separated by stride K
  (output channels).
- **B (X, NHWC):** consecutive K_wg positions are separated by stride C
  (input channels).

`buffer_load_vN` with `N > 1` would read N consecutive *channel* values at the
same spatial position instead of the intended N consecutive spatial positions.
Enabling wider loads requires either rearranging the load tile so that the fast
axis aligns with the last tensor dimension, or introducing a transposing LDS
stage so the data lands in LDS in the order the MFMA atoms expect it.

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
