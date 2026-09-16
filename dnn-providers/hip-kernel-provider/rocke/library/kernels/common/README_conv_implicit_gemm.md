# Implicit-GEMM Forward Convolution

## Algorithm

The forward kernel computes a 2-D (or 3-D) convolution:

```
D[n, ho, wo, k] = sum_{y, x, c} A[n, ho*sH - pH + y*dH, wo*sW - pW + x*dW, c] * B[k, y, x, c]
```

This is cast as an implicit-GEMM with the following dimension mapping:

| GEMM dim | Forward meaning              | Size        |
|----------|------------------------------|-------------|
| M        | Output spatial positions     | N×Ho×Wo     |
| N        | Output channels              | K           |
| K (red.) | Filter spatial × input chan  | Y×X×C       |

### Operands

| Role | Tensor | Layout | GEMM shape              |
|------|--------|--------|-------------------------|
| A    | Input activations  | NHWC  | (M, K_gemm) = (N·Ho·Wo, Y·X·C) |
| B    | Weights            | KYXC  | (N_gemm, K_gemm) = (K, Y·X·C)  |
| D    | Output activations | NHWK  | (M, N_gemm) = (N·Ho·Wo, K)     |

### Coordinate-transform descriptor DAG

The A operand requires a non-trivial address computation because the input tensor
is indexed by `(n, hi, wi, c)` while the GEMM loop variable is `(m, k)`.
The transform DAG maps `(m, k) → NHWC offset`:

```
naive(NHWC):  (n, hi, wi, c)
  + unmerge('m' → n, ho, wo)          — decompose flat output position
  + embed((ho, y) → hi, stride=sH, dH, offset=-pH)  — affine spatial map H
  + embed((wo, x) → wi, stride=sW, dW, offset=-pW)  — affine spatial map W
  + unmerge('k' → y, x, c)            — decompose filter+channel index
  + pad('y', lo=0, hi=Y)              — boundary guard (partial K-tile)
  + pad('x', lo=0, hi=X)              — boundary guard (partial K-tile)
```

B and D use simpler descriptors:

- **B (KYXC):** `naive` + `unmerge('k_gemm' → y, x, c)` + `pad(y)` + `pad(x)`.
- **D (NHWK):** `naive` + `unmerge('m' → n, ho, wo)`.

### Pipelines

| Pipeline | Description |
|----------|-------------|
| `mem`    | Single-buffer LDS; no scheduler hints. Correctness-first baseline. |
| `compv3` | Single-buffer LDS + `sched_group_barrier` hints to overlap DS_read with MFMA. |
| `compv4` | Double-buffer LDS (ping-pong) + scheduler hints + `s_setprio`. |
| `unroll_k` | Python-level K-loop unroll with double-buffering; also triggers double-buffer allocation. |
| `async_dma` | Direct DRAM→LDS via `raw_ptr_buffer_load_lds`; paired with `SoftwarePipeline` for overlap. |
| `wavelet`  | Load/math wave specialization (CK Tile PR #8009). **gfx1250/WMMA only.** Appends `num_load_waves` dedicated load waves to the workgroup; those waves handle all DRAM→LDS transfers while the `warp_m × warp_n` math waves run WMMA exclusively. Single-buffer LDS shared by both roles; no scheduler hints. |

### Epilogues

| Path | Condition | Output |
|------|-----------|--------|
| Direct store | `epilogue="default"` | Per-lane scalar write to D via the NHWK descriptor. |
| CShuffleEpilogue | `epilogue="cshuffle"` | LDS-stage accumulators in MFMA layout, re-read coalesced for wide vector global stores. |

### Additional features

- **Grouped convolution:** `groups > 1` supported on CDNA targets. The descriptor DAG adds a group-unmerge so each group's GEMM stays within its own `(C/groups, K/groups)` slab.
- **Chiplet-aware workgroup swizzle:** `chiplet_swizzle=True` remaps the 2D block grid through `chiplet_aware_super_tile` to improve XCD-local L2 reuse.
- **Accumulator epilogue:** `ConvAccumulatorEpilogue` applies optional bias, scale, ReLU, or clamp directly on MFMA f32 fragments before the store path.

---

## Changelog

### Initial implementation

- Introduced `ImplicitGemmConvSpec` and `build_implicit_gemm_conv`.
- Descriptor DAG covering 2-D NHWC convolution with arbitrary stride, padding, dilation, and groups.
- `mem` pipeline with direct (`default`) and LDS-staged (`cshuffle`) epilogues.
- Arch-aware spec validation via `is_valid_spec` / `is_valid_spec_for_problem`.

### 3-D convolution

- Extended `ConvProblem` with `Di`, `Z`, `sD`, `pD`, `dD` fields.
- `make_a_descriptor` and `make_b_descriptor` emit the 3-D DAG when `p.is_3d` is `True`, adding `embed((do, z) → di)` and `unmerge('k' → z, y, x, c)` steps.

### compv3 / compv4 pipelines

- `compv3`: `sched_group_barrier` hints after each MFMA K-atom step to overlap DS_read + MFMA.
- `compv4`: double-buffer LDS (`A_smem` / `A_smem2`) ping-pong; `s_setprio` to push the K-loop into compute steady state.

### Async DMA

- `async_dma=True` switches the global→LDS path to `raw_ptr_buffer_load_lds` via `AsyncTileLoader`.
- LDS layout must be plain `[block_m, block_k]` (no K-pad); paired with `SoftwarePipeline` for DRAM/MFMA overlap.

### Wavelet pipeline (`wavelet`)

The wavelet pipeline implements the load/math wave specialization introduced in CK Tile PR #8009 (`gemm_pipeline_ag_bg_cr_wavelet.hpp`). It targets **gfx1250 (WMMA/RDNA wave32)** exclusively and is rejected by `is_valid_spec` on MFMA/CDNA targets.

#### Wave role split

The workgroup is enlarged by `num_load_waves` extra waves appended after the standard `warp_m × warp_n` math waves:

| Role | Thread range | Responsibility |
|------|-------------|----------------|
| Math waves | `[0, block_size)` = `warp_m × warp_n × wave_size` threads | WMMA only — never touch global memory |
| Load waves | `[block_size, launch_block_size)` = `num_load_waves × wave_size` threads | DRAM→LDS transfers only |

```
launch_block_size = block_size + num_load_waves * wave_size
                  = (warp_m × warp_n × wave_size) + (num_load_waves × wave_size)
```

With the gfx1250 defaults (`warp_m=2`, `warp_n=2`, `wave_size=32`, `num_load_waves=4`):
- `block_size` = 128 (math waves)
- `launch_block_size` = 256 (math + load waves)

For all other pipelines `launch_block_size == block_size`. The HIP launch and the kernel's `max_workgroup_size` attribute must use `launch_block_size`, not `block_size`.

#### Overlap mechanism: split fetch / store

Because gfx1250 has separate VMEM and WMMA issue queues, DRAM loads (`buffer_load_vN`) and WMMA matrix instructions can be in flight simultaneously — unlike MFMA targets where both share a single issue slot. The wavelet pipeline exploits this by splitting each tile load into two phases:

1. **`wavelet_fetch`** — issues `buffer_load_vN` into registers *before* the synchronisation barrier. The DRAM request is in flight while math waves execute WMMA.
2. **`wavelet_store`** — after `s_waitcnt(vmcnt=0)` confirms data arrival, writes the fetched registers to LDS.

This is the `fetch` / `store_fetched` split on `CoalescedTileLoader` in `helpers/loads.py`.

#### Barrier protocol (`scf_if_else` form)

The kernel emits a single `scf_if_else` block (`tid < block_size`) that drives both roles through matched barrier sequences. The number of barriers per branch must be identical:

```
MATH branch:   barrier_0
               repeat K-2 times: WMMA-atom → barrier_A → barrier_B
               tail WMMA-atom
               epilogue stub barriers

LOAD branch:   fetch[0] → store[0] → barrier_0
               repeat K-2 times: fetch[i+1] → barrier_A → store[i+1] → barrier_B
               N_epi stub barriers (no-op for load role)
```

This uses LLVM `br i1` branching rather than `s_and_saveexec_b64` exec-mask manipulation, so both branches are lowered to ordinary conditional jumps and the separate issue queues remain active for both roles.

#### LDS allocation

The wavelet pipeline uses a **single-buffer** LDS tile (no ping-pong):

- The `_double` flag that enables ping-pong for `compv4` / `async_dma` / `unroll_k` does not include `wavelet`.
- Because both branches of the `scf_if_else` reference A/B LDS simultaneously throughout the kernel body, the liveness packer cannot alias the cshuffle C tile onto the A/B region. The `_no_alias` flag is forced `True` for wavelet regardless of `cshuffle_no_alias`, making the LDS budget always additive: `total = ab + c` (not `max(ab, c)`).
- The cshuffle epilogue requires **two WAR barriers** (`_war_barriers = 2`) instead of one, because two synchronisation points are needed: math waves done (load waves may exit) and load waves exited (A/B LDS safe to overwrite).

#### Scheduler hints

The wavelet schedule (`helpers/schedule.py`) sets `emit_hints=False` and `mode="default"` — no `sched_group_barrier` hints are emitted. Hardware concurrency between VMEM and WMMA is provided by the separate issue queues, not by software scheduling.

#### Known issue: `ds_load_tr16_b128` mis-optimisation (gfx1250)

The AMDGPU LLVM backend may replace a `load <8 x half>, ptr addrspace(3)` that feeds a WMMA instruction with `ds_load_tr16_b128` (a transposed LDS read). This optimisation assumes LDS was stored in column-major layout, but the wavelet kernel stores A_smem / B_smem row-major (for coalesced DRAM→LDS writes). The result is garbage WMMA inputs even though the load-wave transfers completed correctly. Root cause and three candidate fixes are tracked in the debug notes;

### Python-level K-loop unroll (`unroll_k`)

- `unroll_k=True` Python-unrolls the K loop at build time with double-buffering, removing the `scf.for_iter` overhead entirely.

### WMMA support (gfx1251/gfx1250)

- Added RDNA wave32 path using the `wmma` MMA family via `_resolve_conv_op`.
- Supports 16×16×4, 16×16×16, and 16×16×32 WMMA atoms; `mem` pipeline and `default`/`cshuffle` epilogues.
- Dedicated `_emit_direct_epilogue_wmma` and `_emit_wgrad_direct_epilogue_wmma` using `op.c_layout()`.

### Chiplet-aware workgroup swizzle

- `chiplet_swizzle=True` remaps `(block_id_y, block_id_x)` through `chiplet_aware_super_tile` at build time so consecutive workgroups share an XCD (and an L2 slice).

### Grouped convolution

- `ImplicitGemmConvSpec.groups > 1` supported on CDNA. The descriptor adds a group-unmerge to keep each group's GEMM within its `(C/groups, K/groups)` slab.

### Per-group vector-size derivation for grouped convolutions

`default_vector_sizes` and the `is_valid_spec` gate were both called with the
**total** `C` and `K` values instead of the per-group `cpg = C/groups` and
`kpg = K/groups`. For grouped convolutions the output tensor `D` (NHWK) is
indexed within a group's slab — the relevant stride-1 dimension has extent `kpg`,
not `K`. Using `K` produced an over-wide `vec_c` that did not divide `kpg` and
wrote out-of-bounds when `K % vec_c == 0` but `kpg % vec_c != 0`.

### Accumulator epilogue

- `ConvAccumulatorEpilogue` applies optional `bias`, `scale`, `relu`, `clamp_min`/`clamp_max` directly on MFMA f32 fragments before the D-store path.

### Pointwise explicit-GEMM fast path

For **pointwise convolutions** (`Y=X=1`, `sH=sW=1`, `pH=pW=0` — and for 3-D: `Z=1`, `sD=1`, `pD=0`) the forward kernel automatically bypasses the full coordinate-transform descriptor DAG and replaces all three operand address computations with flat multiply-add arithmetic.

**Detection:** `ConvProblem.is_pointwise` returns `True`; no user-facing flag is needed. The benchmark prints a note when this path is active.

**Address arithmetic (pointwise path):**

| Operand | Formula | Replaces |
|---------|---------|---------|
| A (input)  | `offset = m * C + k`     | `unmerge(m)` + `embed(ho,y→hi)` + `embed(wo,x→wi)` + `unmerge(k→y,x,c)` + `pad(y)` + `pad(x)` |
| B (weight) | `offset = k_out * C + k_gemm` | `unmerge(k_gemm→y,x,c)` + `pad(y)` + `pad(x)` |
| D (output) | `offset = m * K + k_out` | `unmerge(m→n,ho,wo)` |

**Why this is faster:** With `Y=X=1`, `sH=sW=1`, `pH=pW=0` the embed transforms reduce to identity (`hi=ho`, `wi=wo`), the filter-unmerge produces `y=x=0` trivially, and the pad guards are always true. The implicit descriptor computes the same values but still emits the full VALU sequence (magic divisions, multiplications, bounds comparisons). Flat arithmetic emits exactly one `mul` and one `add` per operand — zero dead instructions.

---

## Next steps

### K0-M-K1 LDS layout

The current LDS layout stores tiles in `(M, K)` row-major order with a small
`lds_k_pad` column pad to break bank conflicts. A `K0-M-K1` layout reorders the
tile as `(K0, M, K1)` where `K = K0 × K1`, making each MFMA atom's K slice
contiguous in LDS. This eliminates the bank-conflict cross-section that padding
only partially mitigates and enables wider `ds_read` instructions. Adding this
requires:

1. A new `LdsLayout` variant encoding the `(K0, M, K1)` stride formula.
2. Updated `CoalescedTileLoader` / `AsyncTileLoader` store-index calculations.
3. Updated `_emit_smem_load` / `_emit_frag_smem_load` read-index expressions.

### Persistent kernel / stream-K partitioning

The current grid maps one CTA to one output tile. For small M×N problems the
grid is too small to saturate the device. Stream-K partitions the work along
the K loop across CTAs (similar to split-K but without the atomic reduction
overhead), potentially delivering better utilisation for small or irregular
shapes.

### Wider vector loads for A

The A descriptor emits one element offset per thread per load. Enabling vector
loads (`load_vec > 1`) requires that consecutive `k` values within a tile row
map to contiguous memory addresses. For general convolutions this is only true
along the C axis (the innermost dimension of NHWC); enabling it requires either
a K0/K1 split where K1 == C, or a transposing LDS stage.
