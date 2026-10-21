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

The split degree is never compiled in: every `split_k > 1` builds the same
kernel (named `..._spk`), which takes the degree as the `ks_count` / `ks`
kernargs. The benchmark driver (`benchmark_implicit_gemm_conv.py --split-k 0`)
therefore compiles one split-K kernel per configuration and times it at every
degree, instead of compiling one kernel per degree.

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
when a width > 1 is not admissible. The WMMA path is a follow-on. This
vectorised load is what makes the *store* side the remaining cost, which the
K-outer tile below removes.

### K-outer LDS tile + transpose-read operand fetch (`lds_k_outer`)

The free-axis vector load above leaves a transpose *on store*: the loader reads
`load_vec` contiguous elements along the free axis, then scatters them into
`[row+i, col]` of the row-major `(M/N, K)` tile — one narrow `ds_write_b16` per
element plus its address math (`CoalescedTileLoader._store_tile` in `"row"`
mode). Those writes are bank-degenerate by construction: adjacent lanes step the
tile by `load_vec` **rows**, so the inter-lane dword delta is
`load_vec × (block_k + lds_k_pad) / 2`, an exact multiple of the 32-dword bank
period at every swept `tile_k` (16 / 32 / 64) and either pad (0 or 8).
`lds_k_pad` cannot fix this — that pad is derived for a row step of 1, i.e. for
the *read* path.

`lds_k_outer=True` stores the tile K-outer (`LDS[k][mn]`, row stride
`block_mn + _KOUTER_PAD` with `_KOUTER_PAD = 8`, or `0` under `async_dma`) and
recovers the MFMA operand layout with transpose reads instead:

| Side | M-outer | K-outer |
|------|---------|---------|
| Store | `load_vec` × `ds_write_b16` + address math per chunk | one wide `smem_store_vN` (`b128` for a 16-bit 8-wide vector); drops `load_vec − 1` address adds and `load_vec` `vec_extract`s per chunk |
| Read | one `smem_load_vN` per fragment | `n / 4` × `ds_read_b64_tr_b16` (wave64) or `n / 8` × `ds_load_tr16_b128` (wave32), for per-lane fragment length `n` |

Net read-side delta is **+1** instruction per fragment on the `n = 8` atoms
(`32x32x16`, `16x16x32`) and exactly **zero** on the `n = 4` atoms (`16x16x16`,
`32x32x8`). Global `buffer_load`s are unchanged: `choose_vec` tests `tile_rows`
in `"row"` mode and `tile_cols` in `"col"` mode, and the flip transposes the tile
too, so both calls test the same free-axis extent — only the LDS store
instruction changes.

**Wgrad flips both operands.** A (`dY`, NHWK) and B (`X`, NHWC) are both
contiguous along the GEMM free axis and strided along the reduction axis, so both
paid the scatter. (Dgrad flips B only — see its README.)

**Gating.** Two regimes, `_LDS_K_OUTER_ARCH_WAVE = {"gfx950": 64,
"gfx1250": 32}`, each arch pinned to its wave size so a mismatched spec is
rejected rather than emitting a lane formula the hardware does not implement.
gfx950 wave64 admits `warp_tile ∈ (16, 32)` and `ds_read_b64_tr_b16` (4 elements
per lane); gfx1250 wave32 admits only the `16x16x32` atom and
`ds_load_tr16_b128` (8 per lane, so its 16-element fragment is two reads). Plus
16-bit A/B.

**Not a knob.** The spec field still defaults `False` (existing goldens are
unmoved); the value is deduced by the keyword-only
`WgradConvSpec.default_lds_k_outer(*, arch, dtype_a, dtype_b, warp_tile_m,
warp_tile_n, wave_size=64)`, which the sweep driver calls.

### Async DMA on the K-outer tile

`async_dma=True` no longer depends on `unroll_k`. `SchedulePolicy.for_pipeline`
is selected as `"async_dma" if spec.async_dma else spec.pipeline`, so the async
leg pins its own schedule (interwave, `s_setprio 1`) and ignores `spec.pipeline`
rather than being gated by it; it double-buffers on its own
(`_double = spec.async_dma or spec.unroll_k`). This is why the sweep driver pins
the pipeline to `"mem"` on that leg instead of compiling one body under several
kernel names.

The old blocker — `raw_ptr_buffer_load_lds` writes a packed lane-contiguous tile
incompatible with a non-zero `lds_k_pad` — was resolved by *accepting* the packed
layout: the K-outer tile **is** that layout. Hence `async_dma` now **requires**
`lds_k_outer=True` on wgrad (`validate()` and `is_valid_wgrad_spec`), and forces
the K-outer row pad to 0. It remains incompatible with `pipeline="basic"`.

Both loops that Python-unroll the K iteration — `pipeline="basic"` and
`async_dma` — are bounded by `_MAX_UNROLLED_K_ITERS` (128). Over the cap the spec
is rejected; raise `split_k` or `tile_k` rather than the constant.

### Derived LDS row pad (`default_lds_k_pad`)

Left unset, the M-outer tile takes a flat `lds_k_pad` of 8 for every
`tile_k >= 16`. That is not a derived number: the row stride is `tile_k + pad`,
and the stride is what decides how many of the 32 banks an operand fetch
touches. `WgradConvSpec.default_lds_k_pad` derives it instead, from two
constraints — the `ds_read_b64` fragment must stay 8-byte aligned on every row
(`pad % 4 == 0`), and the stride in dwords should share as little as possible
with the 32-dword bank period (`tile_k + pad ≡ 4 mod 8`, giving 16 banks). The
flat 8 reaches only 8 on the whole `tile_k` ladder. Among the pads satisfying
both, the smallest wins: the pad is charged per row of *both* operand tiles, so
a larger one can cost an occupancy step.

It is selection policy, not a body change — the field still defaults to `None`,
so untouched specs keep their name and golden. Callers that build specs
(dispatch, the sweep driver, the AOT cache grid) ask it what to pass, the same
arrangement as `default_lds_k_outer`.

Gated on `arch` rather than on `wave_size` alone: gfx950 is wave64 CDNA too,
but has 64 banks and a 64-dword `ds_read_b64` period, so the congruence above
solves for the wrong modulus there. Relying on gfx950 deducing the K-outer
layout to keep it out is not enough — the AOT cache grid leaves `lds_k_outer`
at its `False` default for every arch, so gfx950 does reach the M-outer path.

### LDS budget is a max, not a sum

`is_valid_wgrad_spec` charged `ab_lds + c_lds`. The cshuffle C tile is built
with `no_alias=False` (wgrad never passes the flag), so `make_lds_view` gives it
`exclusive=False` and it *aliases* onto the A/B bytes rather than adding to
them. Charging the sum rejected specs that fit — it capped `tile_k` well below
what the emitted code object actually asks for, and `tile_k` is what amortises
the K-loop address decode over each LDS row. Now `max(ab_lds, c_lds)`, matching
what the forward and dgrad validators already did; wgrad was the outlier. The
one case that would genuinely be additive, `pipeline="wavelet"`, is rejected for
wgrad earlier in the same function.

### Stream-K (`streamk`, `streamk_reduction`, `streamk_ctas`)

Split-K cuts `K_wg` into `split_k` equal slices and gives every output tile the
same number of them. That is the right shape when the tiles outnumber the CUs,
but a wgrad is usually the opposite: `wg_M x wg_N` is filter-sized while
`K_wg = N*Ho*Wo` is huge, so the tile count can be a handful and the leftover
CUs idle no matter how the slices are cut.

Stream-K (CK Tile's `StreamKTilePartitioner`) balances *MAC iterations* instead.
Output tiles that divide the CTA pool evenly run data-parallel — one CTA, the
whole K loop, no reduction at all — and only the remainder is spread over
`streamk_ctas` stream-K CTAs, each owning a contiguous, balanced range of
iterations that may straddle tile boundaries. The partition is resolved
host-side (`rocke.helpers.streamk.StreamKIterPartition`, reached through
`ConvArgs.streamk_partition`) and passed in as the `sk_*` kernargs; the device
only maps `blockIdx.x` onto its iteration range. Like the split-K degree, the
CTA pool is a launch parameter rather than a build-time constant, so one AOT
binary serves any shape and any pool.

`streamk="dp_sk"` launches `dp_tiles + sk_ctas` CTAs. `streamk="persistent"`
launches only the pool: each CTA round-robins the data-parallel tiles
(`tile = cta + t*pool`, CK's static `StreamKDispatch`, no work queue) and then
runs its stream-K range. Both walk their work with one bounded `scf.for` whose
trip count is CTA-uniform, so the barriers inside the GEMM body stay uniform and
the body is emitted once.

`streamk_reduction` picks how the CTAs sharing a tile combine partial sums:

| | Determinism | dW zeroing | Workspace |
|---|---|---|---|
| `linear` | yes | not needed | flags + one f32 tile per SK CTA |
| `tree` | yes | not needed | same |
| `atomic` | no | required | none (fp32 dW only) |
| `workspace` | no | not needed | the two-stage f32 scratch + Stage 2 reduce |

`linear` and `tree` hand partials over explicitly: a non-owner stores its tile
to its own slot, issues a release fence and sets a per-wave flag; the consumer
spins on an acquire load of that flag and then adds. Both are deterministic
because the fold order is fixed by the partition, not by arrival. `linear` has
the owner walk its later contributors in turn; `tree` fans in pairwise over
`ceil(log2)` rounds. The flag protocol is three core IR ops (`memref.fence`,
`memref.global_flag_store`, `memref.global_flag_wait_eq`) rather than raw cache
hints, so the AMDGPU memory model supplies the L2 writeback/invalidate that
makes a partial visible across XCDs.

Size the buffer with `wgrad_streamk_workspace_nbytes`; split it with
`wgrad_streamk_workspace_layout` (`sk_flags` at the base, `sk_partials` at
`base + flags_bytes`). **Only the flag region has to be zeroed before each
launch** — partials are always written before they are read.

Constraints, all enforced by `validate()` / `is_valid_wgrad_spec` in both
engines: gfx942/gfx950 (MFMA wave64) only; `split_k=1` and no `two_stage`
(stream-K owns the K range); `epilogue="default"`; no `async_dma` / `unroll_k`
(they pipeline the K loop across work items, which stream-K re-enters per tile); no `chiplet_swizzle` or
`group_merge` (both remap tiles, which stream-K owns). Conv groups are folded
into GEMM-M instead of riding `block_id_z`: `m_tiles = ceil(wg_M/tile_m) *
groups`, decoded group-fastest. Defining the tile count per group makes the fold
exact for any `wg_M`, avoiding CK's `GemmM % MPerBlock` caveat.

`streamk_ctas` sizes the host partition only; `-1` resolves per reduction
(`wgrad_streamk_default_ctas`):

* `workspace` / `atomic` never wait on another CTA, so their pool is sized for
  throughput: four CTAs per CU (`_STREAMK_CTAS_PER_CU`). Each resident CTA
  streams its own K range, so this counts independent reduction streams per CU,
  and is counted in CTAs rather than waves: a wider CTA adds no streams. One
  CTA per CU leaves a single stream per CU with nothing to hide memory latency
  behind, which is what made a CU-sized pool lose to split-K on most shapes;
  past four the CTAs mostly contend for the same caches.
* `linear` / `tree` stay at one CTA per CU, because these fixups get slower
  with a larger pool: the tile owner folds every additional contributor. The
  pool does not have to be resident at once. A CTA waits only on CTAs with
  higher indices and publishes its own partial before it waits, so with
  workgroups dispatched in ascending order only the CTAs of the one tile that
  straddles the dispatch front (its owner for `linear`, about one per round
  for `tree`) can wait on a CTA that has not started yet.

Not selected by the grouped-convolution dispatcher: `ConvGroupedRequest` has
no stream-K field and the dispatcher only builds the grid-per-tile kernels, so
stream-K is reached by building a `WgradConvSpec` with `streamk` set. To
measure it, use the implicit-GEMM sweep
(`library/benchmarks/common/benchmark_implicit_gemm_conv.py`, also driven by
`benchmark_conv_compare.py`) with `--direction wgrad --streamk {dp_sk,persistent,all}`,
optionally narrowed with `--streamk-reductions` and pinned to a pool with
`--streamk-ctas` (default `-1`, the per-reduction pool above). The JIT sweep then
adds a stream-K leg, one kernel per geometry x pipeline x mode x reduction
(`atomic` only for fp32 dW); `--compile-all --streamk ...` adds stream-K kernels
to the AOT cache (`kernel_sweep.py`, opt-in) and `--run-from-cache --streamk ...`
runs the cached ones. The cache keeps a subset: the widest vector width of the
dtype without `tree`, plus a scalar `dp_sk` `tree`/`workspace` slice on wide
tiles for large-filter depthwise layers; other narrow-channel shapes fall to the
split-K kernels. Every
launch follows the zeroing contract above, and `workspace` launches are timed
together with their Stage 2 reduce.

## Next steps

### K0-M-K1 LDS layout

Superseded for the bank-conflict case by `lds_k_outer` above, which reaches the
same goal (conflict-free LDS traffic, wide stores) with a 2-D `LDS[k][mn]` tile
and a transpose read rather than a new 3-D layout. A true `K0-M-K1` layout —
`(K0, M, K1)` with `K = K0 × K1`, after the CK naming convention — would still be
needed to make each atom's K slice contiguous for consumers that cannot use a
transpose-read intrinsic (no `ds_read_b64_tr_b16` / `ds_load_tr16_b128` on the
target, or non-16-bit operands, both of which `lds_k_outer` rejects today).
Adding it requires:

1. A new `LdsLayout` variant that encodes the `(K0, M, K1)` stride formula.
2. Updated `CoalescedTileLoader` / `AsyncTileLoader` store-index calculations
   to write into the transposed shape.
3. Updated SMEM-load index expressions in `_emit_smem_load` /
   `_emit_frag_smem_load` to read from the transposed shape.
