# Epilogue Kernels

These kernels are compiled from MLIR sources in `~/aster/kernels/`.

---

## `row_div` — Fused row-sum + reciprocal-sqrt scale

**Source:** `kernels/row_div.mlir`

Scales every element of a bf16 matrix `C` by the reciprocal square root of a
weighted row sum drawn from a separate f32 matrix `D`.

**Kernel arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `C` | `bf16 [m, n]` | Input/output matrix (row-major) |
| `D` | `f32 [m, n_d]` | Row-sum accumulation matrix (row-major) |
| `m` | `index` | Number of rows |
| `n` | `index` | Number of columns in C |
| `n_c` | `index` | Columns of C handled by this column-block |
| `n_d` | `index` | Number of columns in D |
| `inv_d` | `f32` | Scaling factor applied before rsqrt |
| `eps` | `f32` | Epsilon added for numerical stability |

**Grid:** `(m, n_split, 1)` — `block_id_x` selects the row; `block_id_y`
selects the N-column partition of C. D is summed redundantly across column
blocks (D is typically small); splitting N increases occupancy and HBM bandwidth
utilization.

**Computation:**

```python
# Phase 1 — accumulate D row sum (inner loop, step 64; wavefront reduce)
for i in range(m):                        # block_id_x
    row_sum = sum(D[i, :])                # DPP butterfly reduction across wavefront

    # Phase 2 — compute shared scale factor
    scale = rsqrt(inv_d * row_sum + eps)

    # Phase 3 — apply scale to the assigned column block of C
    for j in range(block_id_y * n_c, (block_id_y + 1) * n_c):
        C[i, j] = bf16(f32(C[i, j]) * scale)
```

Odd-column tail (when `n_c` is odd): lane 0 handles the last unpaired bf16
element explicitly, as it cannot fill a full dword for the vectorised store path.

---

## `small_reduction` — Row rsqrt reduction

**Source:** `kernels/rsqrt_row.mlir`

Computes a per-row reciprocal square root from a f32 matrix `D` and writes one
f32 result per row. This is the reduction-only half of `row_div`, useful when
the scale factors are needed separately (e.g. to feed a subsequent kernel).

**Kernel arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `result` | `f32 [m]` | Output vector, one value per row |
| `D` | `f32 [m, n_d]` | Input matrix (row-major) |
| `m` | `index` | Number of rows |
| `n_d` | `index` | Number of columns in D |
| `inv_d` | `f32` | Scaling factor applied before rsqrt |
| `eps` | `f32` | Epsilon added for numerical stability |

**Grid:** `(m, 1, 1)` — one block per row; 64 threads per block (one wavefront).
Out-of-bounds lanes read 0 via the buffer OOB trick, so no explicit masking is
needed even when `n_d` is not a multiple of 64.

**Computation:**

```python
for i in range(m):                        # block_id_x
    # Phase 1 — sum D[i, :] in steps of 64 across the wavefront
    row_sum = sum(D[i, :])                # wavefront reduction via gpu.subgroup_reduce

    # Phase 2 — compute and store (lane 0 only to avoid redundant writes)
    result[i] = rsqrt(inv_d * row_sum + eps)
```

---

## `row_div_quant` — Fused row-div + bf16→fp8 quantization

**Source:** `kernels/row_div_quant.mlir`

Extends `row_div` with an in-kernel OCP E4M3 fp8 quantization step. Each
element of C is scaled by the row rsqrt and an additional per-tensor `scale`
factor, then converted to fp8. Both a bf16 pre-quantization output and the fp8
output are written.

**Kernel arguments:**

| Arg | Type | Description |
|-----|------|-------------|
| `C` | `bf16 [m, n]` | Input matrix (row-major, read-only) |
| `D` | `f32 [m, n_d]` | Row-sum accumulation matrix (row-major) |
| `out_fp8` | `fp8 [m, n]` | fp8 (OCP E4M3) output matrix |
| `out_bf16` | `bf16 [m, n]` | Pre-quantization bf16 output matrix |
| `scale` | `f32` | Per-tensor quantization scale |
| `m` | `index` | Number of rows |
| `n` | `index` | Total columns of C |
| `n_c` | `index` | Columns handled by this column-block |
| `n_d` | `index` | Number of columns in D |
| `inv_d` | `f32` | Scaling factor applied before rsqrt |
| `eps` | `f32` | Epsilon for numerical stability |

**Grid:** `(m, n_split, 1)`, `block_dim = 128` (2 wavefronts).

**Computation:**

```python
for i in range(m):                        # block_id_x
    # Phase 1 — D row sum, wavefront reduction
    row_sum = sum(D[i, :])

    # Phase 2 — two scale factors: bf16 uses only rsqrt, fp8 also divides by scale
    bf16_scale = rsqrt(inv_d * row_sum + eps)
    fp8_scale  = bf16_scale / scale

    # Phase 3 — load 8 bf16, apply per-output scale, quantize pairs to fp8 via v_cvt_pk_fp8_f32
    for j in range(block_id_y * n_c, (block_id_y + 1) * n_c):
        val_f32      = f32(C[i, j])
        out_bf16[i, j] = bf16(val_f32 * bf16_scale)
        out_fp8[i, j]  = fp8_e4m3(val_f32 * fp8_scale)   # packed 2-at-a-time via v_cvt_pk_fp8_f32
```

The main loop processes elements 8 at a time (vectorized bf16 load), packing
each pair of f32 values into one fp8 dword with `v_cvt_pk_fp8_f32`. Four pairs
fill a `vector<2xi32>` for the output store. The last `n_c mod 4` columns form
a partial dword and are stored byte-wise by lane 0.
