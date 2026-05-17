# Small Ops, Reductions, Norms, Transpose, Quantization, And MoE Infra

This page covers every kernel that lives outside the GEMM / attention /
convolution paths -- the "small-op" surface that grew through
of the CK Tile parity roadmap.

Pre-roadmap kernels:

- `instances/elementwise.py`
- `instances/reduce.py`
- `instances/layernorm2d.py`
- `instances/rmsnorm2d.py`
- `instances/transpose.py`

composition of existing primitives:
- `instances/img2col.py` -- CK Tile `04_img2col`
- `instances/pooling.py` -- CK Tile `36_pooling`
- `instances/permute_nd.py` -- CK Tile `06_permute`
- `instances/batched_transpose.py` -- CK Tile `35_batched_transpose`
- `instances/gemm_multi_d.py` -- CK Tile `19_gemm_multi_d`
- `instances/gemm_multi_abd.py` -- CK Tile `22_gemm_multi_abd`
- `instances/elementwise.py` (ext) -- silu / swish / tanh / sigmoid /
 quick_gelu unary + swiglu / geglu binary
- `instances/reduce.py` (ext) -- min / prod combiners

quantisation:
- `instances/smoothquant.py` -- CK Tile `12_smoothquant`
- `instances/moe_smoothquant.py` -- CK Tile `14_moe_smoothquant`
- `instances/add_rmsnorm2d_rdquant.py` -- CK Tile `11_add_rmsnorm2d_rdquant`
- `instances/topk_softmax.py` -- CK Tile `09_topk_softmax`

MoE infrastructure:
- `instances/moe_sorting.py` -- CK Tile `13_moe_sorting`

StreamK + flat / contraction GEMM variants:
- `instances/streamk_gemm.py` -- CK Tile `40_streamk_gemm`
- `instances/flatmm.py` -- CK Tile `18_flatmm`
- `instances/batched_contraction.py` -- CK Tile `41_batched_contraction`

FP8 / BF8 / i4 / MX quantised GEMM:
- `instances/block_scale_gemm.py` -- CK Tile `38_block_scale_gemm`
- `instances/mx_gemm.py` -- CK Tile `42_mx_gemm`

 also extends `helpers/atoms.py` with four FP8 / BF8 MFMA atoms
(``fp8_16x16x32``, ``fp8_32x32x16``, and their bf8 siblings); see
``primitives/intrinsics_and_primitives.md`` for the lane layouts.

fused MoE forward:
- `instances/fused_moe.py` -- CK Tile `15_fused_moe`
 - `build_moe_gather` -- pre-sort -> per-bucket gather
 - `build_moe_silu_mul` -- SwiGLU activation fusion
 - `build_moe_topk_weighted_reduce` -- atomic-accumulate scatter
 - `FusedMoeLauncher` -- call-graph descriptor for the
 full moe-sort -> gather ->
 per-expert GEMMs -> reduce
 pipeline (per-expert GEMMs
 compose with the existing
 ``block_scale_gemm`` /
 ``universal_gemm`` builders).

FMHA expansion -- see ``instances/attention.md`` for
the full per-variant breakdown:
- `instances/_fmha_common.py` -- shared scaffolding
- `instances/fmha_varlen.py` -- CK Tile `01_fmha` varlen
- `instances/fmha_appendkv.py` -- CK Tile `01_fmha` appendkv
- `instances/fmha_paged_prefill.py` -- CK Tile `01_fmha` pagedkv_prefill
- `instances/fmha_splitkv_decode.py` -- CK Tile `01_fmha` splitkv
- `instances/fmha_head_grouping.py` -- CK Tile `01_fmha` head_grouping
- `instances/fmha_bwd.py` -- CK Tile `01_fmha` bwd
- `instances/fmha_fwd_fp8.py` -- CK Tile `01_fmha` fp8

Helpers:

- `helpers/rotary.py` -- RoPE primitives (interleaved + half)
- `helpers/rng.py` -- Philox4x32-10 RNG for dropout
- `helpers/attention.py` (ext) -- alibi-bias-log2, alibi-bias-matrix,
 custom-mask helpers

Sage attention quant variants + sparse attention --
see ``instances/attention.md`` for the full per-variant breakdown:
- `instances/sage_attention.py` -- CK Tile ``49_sageattention``
 (fp16_bf16 / fp8_bf16 / i8_fp8_bf16 / i4_fp8_bf16 quant variants)
- `instances/sparse_attention.py` -- CK Tile ``50_sparse_attn``
 (jenga block-sparse + VSA variable-size sparse)

Helpers:

- `helpers/qk_scale.py` -- per-head / per-block Q+K scale loaders
- `helpers/codebook.py` -- i8 / i4 -> fp8 codebook dequant chains
- `helpers/sparse_iter.py` -- block-sparse bitmap + VSA LUT K-iterators

These kernels use the "small-op" helper layer more than the GEMM-shape layer.

Common helpers:

- `helpers/spec.py`
- `helpers/io.py`
- `helpers/sweep.py`
- `helpers/reduction.py`
- `helpers/tensor_view.py`
- `helpers/distribution.py`
- `helpers/quant.py` -- : f32 <-> {i8, fp8e4m3, bf8e5m2}
- `helpers/scan.py` -- : block histogram + exclusive scan
- `helpers/persistent.py` -- : persistent-grid pattern
- `helpers/streamk.py` -- : StreamK tile partitioner
- `helpers/i4_dequant.py` -- : packed-i4 unpack + dequant
- `helpers/mx_scale.py` -- : OCP MX E8M0 shared exponent
- `helpers/preshuffle.py` -- : preshuffled-B tile-major layout
- `helpers/gather_scatter.py` -- : per-token indirect address
 calculation (gather-by-bucket-row +
 scatter-by-token + ``SortedTokenIds``
 / ``SortedWeights`` load helpers)
- `helpers/rotary.py` -- : rotary position embedding
 (interleaved + LLaMA-half layouts)
- `helpers/rng.py` -- : Philox4x32-10 RNG for dropout
- `helpers/qk_scale.py` -- : per-head / per-block Q+K scale
 loaders for Sage attention
- `helpers/codebook.py` -- : i8 / i4 -> fp8 codebook dequant
 chains for the Sage int variants
- `helpers/sparse_iter.py` -- : block-sparse bitmap + VSA LUT
 K-iterators for sparse attention

## Common Pattern

Most small ops follow:

```text
1. Define a compact spec dataclass.
2. Validate dtype, op, vector width, block size, and row size via IOSpecRule.
3. Build a simple ABI with pointers and shape args (SignatureBuilder).
4. Use one CTA per row or one CTA per contiguous element tile.
5. Use vector loads where fully in bounds (load_vec_as_f32).
6. Promote to f32 for math.
7. Reduce through LDS if needed (block_lds_reduce).
8. Pack back to output dtype (pack_f32_to / store_scalar_from_f32).
9. Vector or scalar store with tail masks.
```

`IOSpecRule` (from `helpers/spec.py`) provides the shared validation surface across this family:

```text
allowed_dtypes = ("f16", "fp16", "bf16")
allowed_block_sizes = (64, 128, 256, 512, 1024)
allowed_vecs = (2, 4, 8)
n_per_block % (block_size * vec) == 0
optional max_elems_per_thread cap
```

These kernels are usually memory-bound. Vector width, coalescing, launch overhead, and dtype dispatch matter more than exotic scheduling hints.

## Elementwise

File:

```text
instances/elementwise.py
```

Main symbols:

- `ElementwiseSpec`
- `build_elementwise`
- `elementwise_grid`
- `elementwise_signature`

Supported shapes:

```text
1D contiguous tensor
N elements
```

Unary ABI:

```text
A, C, N
```

Binary ABI:

```text
A, B, C, N
```

Algorithm:

```text
1. Compute global vector base:
 base = (block_id_x * block_size + tid) * vec

2. If base + vec <= N:
 - vector-load A, and B for binary ops;
 - promote lanes to f32 where needed;
 - apply op;
 - pack to output dtype;
 - vector-store C.

3. Else tail path:
 - for i in 0..vec:
 idx = base + i
 if idx < N:
 scalar-load
 apply op
 scalar-store
```

Example ops:

Unary (+ extensions):

- copy, neg, abs, relu, exp2;
- gelu_tanh, quick_gelu (ext);
- silu / swish (aliases, ext);
- tanh, sigmoid (ext).

Binary:

- add, sub, mul, max, min;
- swiglu (`silu(a) * c`, ext);
- geglu (`gelu_tanh(a) * c`, ext).

Optimization levers:

- vector width;
- block size;
- avoiding scalar tail overhead for aligned production shapes;
- fusing multiple elementwise passes into one future fused kernel.

## Reduce2D

File:

```text
instances/reduce.py
```

Main symbols:

- `Reduce2DSpec`
- `ReduceOp`
- `build_reduce2d`
- `reduce2d_grid`
- `reduce2d_signature`

Contract:

```text
X: [M, N]
Y: [M]
```

One CTA reduces one row.

Algorithm:

```text
1. block_id_x selects row m.
2. tid sweeps columns in chunks:
 for col = tid*vec; col < N; col += block_size*vec
3. Vector-load X[m, col:col+vec] where in bounds.
4. Promote to f32.
5. Accumulate local partial:
 - sum: partial += values
 - max: partial = max(partial, values)
6. Reduce partials across block with block_lds_reduce.
7. For mean, divide sum by N.
8. Thread 0 stores Y[m].
```

 extension: the op set is now `sum / max / min / mean / prod`. The
helper `block_lds_reduce` was extended in lock-step to accept `min` and
`prod` combiners (`helpers/reduction.py`).

Correctness details:

- `max` identity is negative infinity (or lowest representable f32);
- `min` identity is positive infinity (analogous);
- `prod` identity is `1.0f`;
- `sum / mean` accumulation is f32;
- tails must not contribute garbage;
- mean denominator should match the contract exactly.

## LayerNorm2D

File:

```text
instances/layernorm2d.py
```

Main symbols:

- `LayerNorm2DSpec`
- `build_layernorm2d`
- `layernorm2d_grid`
- `layernorm2d_signature`

Contract:

```text
X: [M, N]
gamma: [N]
beta: [N]
Y: [M, N]
optional mean: [M]
optional invstd: [M]
```

One CTA handles one row.

Algorithm:

```text
Pass 1:
1. Each thread sweeps row chunks.
2. Vector-load X.
3. Promote values to f32.
4. Accumulate sum and sumsq.
5. block_lds_reduce sum.
6. block_lds_reduce sumsq.
7. Compute:
 mean = sum / N
 variance = sumsq / N - mean*mean
 inv_std = rsqrt(variance + epsilon)
8. Optionally store mean and invstd.

Pass 2:
9. Each thread sweeps row chunks again.
10. Load X, gamma, beta.
11. y = (x - mean) * inv_std * gamma + beta
12. Pack to output dtype.
13. Store Y.
```

The two-pass shape avoids caching an entire row in registers or LDS for large N.

Optimization levers:

- vector width;
- block size;
- using f32 reductions;
- reducing redundant loads where N is small enough to cache;
- writeback of mean/invstd only when requested.

## RMSNorm2D

File:

```text
instances/rmsnorm2d.py
```

Main symbols:

- `RMSNorm2DSpec`
- `build_rmsnorm2d`
- `rmsnorm2d_grid`
- `rmsnorm2d_signature`

Contract:

```text
Y = X * rsqrt(mean(X^2) + epsilon) * gamma
```

One CTA handles one row.

Algorithm:

```text
Pass 1:
1. Sweep X row.
2. Accumulate sumsq in f32.
3. block_lds_reduce sumsq.
4. inv_rms = rsqrt(sumsq / N + epsilon)
5. Optionally store inv_rms.

Pass 2:
6. Sweep X and gamma.
7. y = x * inv_rms * gamma
8. Pack and store.
```

RMSNorm removes the mean and beta path from LayerNorm, so it has less arithmetic and less metadata.

## Transpose2D

File:

```text
instances/transpose.py
```

Main symbols:

- `Transpose2DSpec`
- `build_transpose2d`
- `transpose2d_grid`
- `transpose2d_signature`

Contract:

```text
Y[n, m] = X[m, n]
```

Algorithm:

```text
1. Each CTA owns a tile_m x tile_n input tile.
2. Phase 1:
 - threads load contiguous vectors from X;
 - store into LDS tile [tile_m][tile_n + lds_pad].
3. Barrier.
4. Phase 2:
 - threads read transposed positions from LDS;
 - pack contiguous vectors for Y;
 - store coalesced output vectors.
```

LDS padding avoids bank conflicts when reading columns.

Optional grid order:

- row-major default;
- Morton order exists for experimentation but row-major is the documented preferred default unless measurements show otherwise.

## Distribution-Driven Examples

The helper docs mention examples:

- `examples/distribution_reduce_demo.py`
- `examples/distribution_2d_add_demo.py`

These show the `TileDistributionEncoding` path. Production small ops currently mostly use the simpler sweep helpers because the row-wise patterns are straightforward.

## Small-Op Performance Levers

For memory-bound ops:

- maximize coalesced vector loads/stores;
- keep tail handling cheap;
- avoid unnecessary barriers;
- minimize extra passes over memory;
- use f32 math only where needed by numerics;
- benchmark enough iterations that launch overhead is understood.

For reductions/norms:

- tune block size for row length;
- tune vector width;
- inspect LDS bank conflicts;
- ensure reduction identity values are correct;
- consider multiple rows per CTA only if occupancy or row size demands it.

For transpose:

- LDS padding;
- tile shape;
- vector load/store width;
- grid ordering and cache behavior.

## Img2Col File: `instances/img2col.py`. Matches CK Tile `04_img2col`.

Materialises the implicit-GEMM `A` operand for a convolution as a real
`[M, K]` matrix where `M = N * Ho * Wo` and `K = R * S * C`. Out-of-image
positions are zero (pad).

Implementation: reuses
`conv_implicit_gemm.make_a_descriptor(problem)` from the implicit-GEMM
conv path; each thread takes one output element, runs the descriptor
to get a `(global_offset, valid)` pair, and uses the buffer-rsrc OOB
sentinel trick to silently zero-fill the padded zone. One thread per
output element (block-size capped at 1024).

## Pooling2D File: `instances/pooling.py`. Matches CK Tile `36_pooling` (2D subset of
the upstream 3D example; the 3D extension is a single descriptor axis
addition and is a v2 follow-on).

`max / avg / sum` over NHWC inputs with optional dilation + padding. The
input address goes through a coordinate-transform descriptor (two
`embed` transforms for the conv-style `hi = ho*sH - pH + y*dH`); the
descriptor's `valid` predicate masks padded cells with the reduction's
neutral element (`-inf` for max, `0` for sum / avg). One thread per
output element.

## Permute (rank-N)

File: `instances/permute_nd.py`. Matches CK Tile `06_permute`.

`Y[i_0, ..., i_{n-1}] = X[i_{pi(0)}, ..., i_{pi(n-1)}]`. Rank up to 8
with arbitrary permutations, `f16` / `bf16` (fp8 / fp32 in v2). Shape +
permutation are compile-time in the spec; the per-thread decompose +
recompose is a handful of div/mod ops that fold cleanly at IR
construction.

## BatchedTranspose2D File: `instances/batched_transpose.py`. Matches CK Tile `35_batched_transpose`.

Same per-batch algorithm as `instances/transpose.py` (LDS-staged tile +
bank-pad), with a `block_id_z` batch dim and runtime per-batch element
strides (`batch_stride_x` / `batch_stride_y`). Square tiles in
`{16, 32, 64}`, `vec in {2, 4, 8}`, `f16` / `bf16`.

## GEMM multi-D and multi-A/B/D Files: `instances/gemm_multi_d.py`, `instances/gemm_multi_abd.py`.
Matches CK Tile `19_gemm_multi_d` and `22_gemm_multi_abd`.

Thin wrappers around `build_universal_gemm` that thread N D operands
through the existing fused `cshuffle` epilogue. Each D operand becomes
an extra pointer parameter; per-D ops are `add` / `mul` (composed via
`helpers/fuse.py`'s `ResidualAdd` / `ResidualMul`).

In v1 `multi_abd` only supports `num_a == num_b == 1` (delegates to
`multi_d` with future-proofed spec surface); the load-time
`AElementWise` / `BElementWise` combine is a v2 follow-on.

## SmoothQuant File: `instances/smoothquant.py`. Matches CK Tile `12_smoothquant`.

Per-row dynamic quantisation:

```text
y[m, n] = x[m, n] * smscale[n]
amax[m] = max_n(|y[m, n]|)
yscale[m] = max(amax, eps) / quant_max
qy[m, n] = quantise(y[m, n], inv_scale=1/yscale[m])
```

`out_dtype in {i8, fp8e4m3, bf8e5m2}` -- the only difference between
the three is the `quant_max` clamp (`127` / `448` / `57344`) and the
trailing cast (`cvt_f32_to_i8_sat` / `cvt_f32_to_fp8` /
`cvt_f32_to_bf8`). Two-pass row kernel: pass 1 streams `x`, computes
the amax over `y = x * smscale`; pass 2 re-loads `smscale` and emits
the quantised output. The f32 x-cache from pass 1 is reused so HBM is
hit exactly twice (once for `x`, once for `smscale`).

## MoeSmoothQuant File: `instances/moe_smoothquant.py`. Matches CK Tile `14_moe_smoothquant`.

Extension of SmoothQuant where `smscale` is a `(experts, N)` per-expert
table and the kernel produces `topk * tokens` output rows: each input
token is quantised once per selected expert with that expert's
`smscale`. The per-token expert id is looked up from `TopkIds` and
promoted to an SGPR via `b.to_sgpr_u32(...)` so the SmScale row stride
multiply stays scalar.

Grid is `(topk * tokens, 1, 1)`; out-row layout matches CK Tile's
`i_topk * tokens + i_token`.

## AddRmsnorm2dRdquant File: `instances/add_rmsnorm2d_rdquant.py`. Matches CK Tile
`11_add_rmsnorm2d_rdquant`.

Fused `(a + b)` -> RMSNorm -> quantise pipeline. Pass 1 computes
`x = a + b`, the per-thread `sum(x^2)` and `max(|x * gamma|)`, and
optionally writes `x` back as the residual stream output for the next
layer. Two block-LDS reductions feed
`inv_rms = 1 / sqrt(sum_sq/N + eps_rms)` and
`amax_y = inv_rms * max(|x * gamma|)`. Pass 2 re-reads `gamma`, fuses
the `x * inv_rms * gamma` multiply with the quant cast, and stores
`qy`. Same `out_dtype in {i8, fp8e4m3, bf8e5m2}` set as SmoothQuant.

## TopkSoftmax File: `instances/topk_softmax.py`. Matches CK Tile `09_topk_softmax`.

MoE router primitive: for each row of `X[M, N]`, find the K largest
values + their indices and apply softmax over those K. v1 uses a
simple tournament:

```text
for pick_k in 0..K:
 local_max + local_arg = per-thread scan of cached row slice
 global_max = block_lds_reduce(local_max, "max")
 matching threads = race-write local_arg into LDS slot 0
 sync; winner_idx = LDS[0]
 record (global_max, winner_idx) as pick_k
 mask: lanes that own winner_idx in their cache set it to -inf

softmax: vmax = picks[0]; exps[k] = exp(picks[k] - vmax)
 y[m, k] = exps[k] / sum(exps)
```

K up to 32, N up to `block_size * 64`, `dtype / out_dtype in {f16, bf16, f32}`.
The full row is cached in per-thread f32 registers (`ceil(N / block_size)`
slots per lane) so HBM is hit exactly once per row.

## MoE Sorting File: `instances/moe_sorting.py`. Matches CK Tile `13_moe_sorting`.

Three-launch pipeline that rearranges `(tokens, topk)` routing into
expert-major contiguous runs:

1. `build_moe_sort_histogram` -- one global `atomic_add` per
 `(t, k)` pair into `Hist[expert_id]`.
2. `build_moe_sort_scan` -- single-block exclusive prefix sum over
 `Hist`; outputs both `Offsets` (exclusive scan) and `Counts`
 (unchanged source copy). Uses
 `helpers.scan.block_exclusive_scan_i32` (Hillis-Steele).
3. `build_moe_sort_scatter` -- per `(t, k)` pair: claim the next free
 slot in `expert_id`'s bucket via `atomic_add(Counter[expert_id], 1)`,
 then write `(t, k, weight)` into
 `(SortedTokenIds, SortedTopkIds, SortedWeights)[bucket_idx]`.

Workspace is `4 * experts` bytes (a single i32 counter array that
phase 1 writes, phase 2 reads, phase 3 reuses as the per-expert
"next free" counter after the host re-clears it). `experts <= block_size`
in v1 (the single-block scan); multi-pass scan lifts this cap.

## infra helpers

The MoE-sort kernels rely on three reusable helpers shared with the
 StreamK path and the FMHA-backward path:

- `helpers/scan.py`
 - `lds_zero_i32(b, lds, *, tid, block_size, length)` -- cooperative
 LDS clear + sync.
 - `block_histogram_i32(b, lds, keys, *, tid, block_size, num_bins)` --
 zero + per-key `lds_atomic_add(1)` + sync.
 - `block_exclusive_scan_i32(b, lds, *, tid, block_size, length)` --
 in-place Hillis-Steele scan with a final inclusive-to-exclusive
 right-shift. Single-block; requires `length <= block_size`.
- `helpers/persistent.py`
 - `persistent_tile_for_each(b, *, counter, num_tiles, max_iters, body)`
 -- wraps the persistent-grid pattern: each CTA atomic-fetches its
 first tile id, then the loop body fetches subsequent tile ids
 until the counter exhausts `num_tiles`.
 - `persistent_tile_loop(...)` (context-manager form) -- for callers
 that need to interleave non-tile work inside the iteration.
- Underlying IR primitives:
 - `IRBuilder.global_atomic_add(ptr, idx, value, *, ordering)` -- LLVM
 `atomicrmw add / fadd` on `addrspace(1)`; emits
 `global_atomic_add` / `global_atomic_add_f32` on AMDGPU.
 - `IRBuilder.lds_atomic_add(smem, indices, value, *, ordering)` --
 LLVM `atomicrmw add / fadd` on `addrspace(3)`; emits
 `ds_add_u32` / `ds_pk_add_f32` on AMDGPU.

## Small-Op Failure Modes

- Tail path loads out-of-bounds before masking.
- Reduction identity is wrong for max or empty tails.
- Mean/variance uses integer or low-precision accumulation by accident.
- `N` is runtime but a helper assumes compile-time chunk count.
- Vector store crosses row boundary.
- Transpose LDS padding is removed and bank conflicts dominate.
- Benchmark compares one launch of tiny elementwise op against a fused library path.

Quantisation-specific failure modes :

- `eps` clamp on `amax` is missing: an all-zero row produces `+inf`
 for `1/yscale` and the `cvt_f32_to_<q>` saturates to the wrong
 sign on zero inputs.
- `quant_max` constant divergence vs CK Tile (`127.0f` for i8,
 `448.0f` for fp8e4m3, `57344.0f` for bf8e5m2). Use
 `helpers.quant.quant_max_abs(qdtype)` to stay aligned.
- Integer round mode: i8 quant uses `llvm.rint.f32` (round-to-
 nearest-even). Truncation (`fptosi`) differs at `.5` boundaries.

MoE-sort-specific failure modes :

- Workspace counter not cleared between phase 2 and phase 3 -- the
 scatter step reuses the histogram slot as a per-expert "next free"
 counter and expects it to start at zero.
- `experts > block_size` -- the single-block scan silently truncates.
 v1 rejects this at spec validation.
- An out-of-range expert id in `TopkIds` is silently dropped (the
 histogram and scatter passes both guard with `cmp_lt(eid, num_experts)`)
 rather than corrupting an unrelated counter.

## StreamK GEMM File: `instances/streamk_gemm.py`. Matches CK Tile `40_streamk_gemm`.

The capstone: a persistent kernel that uses
`helpers.persistent.persistent_tile_for_each` to pull macro tiles
``(m_tile, n_tile, k_iter)`` from a global counter via
`global_atomic_add(Counter, 1)`. Each macro tile is one ``tile_k``
slice of the GEMM K loop; the partial K-iter contributions to a
shared ``(m_tile, n_tile)`` output position land via:

* **Atomic strategy** (v1): every CTA atomic-adds its f32 partial
 into a shared workspace ``Cf32[M, N]``. Simple, requires a
 separate finalisation pass to convert f32 -> output dtype.
 Implemented end-to-end in v1.
* **Reduction strategy** (v2): cooperative reduction through a
 per-tile flag table; the last contributor performs the conversion
 in-kernel. More complex but avoids the second launch. Spec accepts
 it today, builder rejects it until the helper lands.

v1's *inner* GEMM is a scalar per-thread inner-product
(no MFMA). This is intentional -- it keeps the StreamK
*infrastructure* (partitioner + persistent + atomic) exercised
end-to-end in a reviewable kernel (~150 LOC). The MFMA upgrade
follows; the partitioner + atomic surface stay stable.

Workspace required: ``4 * M * N + 4`` bytes (Cf32 + Counter; both
must be zero-cleared by the caller before launch).

The persistent grid + StreamK partitioner pair (`helpers/persistent.py`
+ `helpers/streamk.py`) is the durable deliverable; this kernel
is the smallest end-to-end consumer that proves they compose.

## FlatMM File: `instances/flatmm.py`. Matches CK Tile `18_flatmm`.

CK Tile's "FlatMM" is a *batched* matmul with preshuffled B; it's
shipped in the upstream library as an alternative configuration to
the standard ``03_gemm`` Preshuffle pipeline. In v1 we re-use the
``build_batched_gemm`` body verbatim, tag the kernel symbol with a
``ck_dsl_flatmm`` prefix so a sweep / dispatcher can distinguish the
two configurations, and expose ``preshuffle_b`` on the spec surface
(rejected at build time today, wired with the preshuffle-B
helper).

## BatchedContraction File: `instances/batched_contraction.py`. Matches CK Tile
`41_batched_contraction`.

Generalises ``batched_gemm`` to *arbitrary* leading-batch ranks:
``(B_0, B_1, ..., B_{r-1}, M, K) x (B_0, ..., B_{r-1}, K, N) ->
(B_0, ..., B_{r-1}, M, N)``. v1 flattens the leading batches into a
single ``batch = product(batch_shape)`` axis on the host side and
delegates to ``build_batched_gemm`` with the standard
``(stride_a, stride_b, stride_c)`` arg trio. The helper
``flatten_batch_strides(batch_shape, inner_size)`` computes the
contiguous per-batch element stride for the launcher.

StreamK-specific failure modes :

- ``Counter`` / ``Cf32`` workspace not pre-cleared -- the first CTAs
 inherit garbage and corrupt every output.
- ``M / N / K`` not divisible by their tile sizes -- the v1 scalar
 inner doesn't handle partial tiles; spec validation rejects this.
- Picking too small a ``max_iters`` for the persistent loop -- some
 macro tiles never get processed. ``compute_streamk_grid_size``
 + ``StreamKPartition.num_macro_tiles`` give a safe upper bound.
- Calling the scalar v1 kernel against a perf reference -- the v1
 is a correctness oracle, not a perf target. The MFMA upgrade
 lives in ``build_streamk_gemm`` follow-on commits.

## Block-scaled GEMM File: `instances/block_scale_gemm.py`. Matches CK Tile
``38_block_scale_gemm``. The kernel computes::

 C[m, n] = sum_k_block(
 a_scale[m // gm, k // gk] * b_scale[k // gk, n // gn]
 * sum_inner_k(A[m, k] * B[k, n])
 )

where A and B carry quantised mantissa values (``fp8e4m3`` or
``bf8e5m2`` in v1; ``i4_fp8`` / ``i4_bf8`` are spec-recognised but
require the v2 MFMA body). The variant matrix from CK Tile's
example (29 distinct preconfigured kernels) collapses to one builder
parameterised by:

* ``quant_mode`` -- ``"aquant"`` / ``"bquant"`` / ``"abquant"``.
* ``mantissa_dtype`` -- ``"fp8e4m3"`` / ``"bf8e5m2"`` /
 ``"i4_fp8"`` / ``"i4_bf8"``.
* ``preshuffle_b`` -- spec field, rejected at build time in v1
 (v2 unlocks).
* ``group_size_mnk`` -- ``(group_m, group_n, group_k)`` matching
 CK Tile's ``--group_size 1x1x128`` default.

v1 ships scalar-inner kernels for the
``{aquant, bquant, abquant} x {fp8e4m3, bf8e5m2}`` matrix
(6 configurations); each writes f32 to ``C``. The MFMA path
(v2) replaces the per-element loop with one
``mfma_f32_16x16x32_fp8(...)`` call per MX block plus the block-scale
multiply post-MFMA, against the same spec surface.

## MX GEMM File: `instances/mx_gemm.py`. Matches CK Tile ``42_mx_gemm``. Uses the
OCP MX-spec shared-exponent format: each 32-element mantissa block
carries one 8-bit unbiased E8M0 scale.

Per-element math::

 A_real[i] = mantissa[i] * 2^(A_scale[i // 32] - 127)

The kernel uses :func:`ck_dsl.helpers.decode_mx_scale_e8m0` to turn
the E8M0 byte into an f32 multiplier (with NaN / zero sentinel
handling matching the AMDGPU MX MFMA hardware path) and
:func:`ck_dsl.helpers.apply_mx_scale` to apply the scale before the
accumulate.

v1 ships fp8 and bf8 mantissa with ``group_k=32`` (the MX spec
default); fp4 and fp6 mantissa variants land with the matching
unpack helpers in v2.

 failure modes:

- ``preshuffle_b=True`` requires the MFMA-based v2 body; v1 rejects
 this loudly at spec validation.
- Group sizes that don't divide K cleanly silently truncate the last
 partial block; v1 rejects ``K % group_k != 0``.
- ``i4`` mantissa requires the v2 dequant path; v1 rejects.
- For MX GEMM, ``group_k`` must be exactly 32 -- the OCP MX spec
 pins the shared-exponent block size.

## Fused MoE Forward File: `instances/fused_moe.py`. Matches CK Tile ``15_fused_moe``.

The capstone: the full fused-MoE forward pipeline. CK Tile's
reference is six logical stages stitched together by a small handful of
kernel launches; this module implements the three *MoE-specific*
kernels plus a launcher class that documents the orchestration.

End-to-end pipeline (token=T, expert=E, topk=K, hidden=H,
intermediate=I)::

 (0) router -> TopkIds[T, K], TopkWeights[T, K] (caller-provided)
 (1) moe_sorting -> Offsets[E], Counts[E], (, 3 launches)
 SortedTokenIds[T*K], SortedTopkIds[T*K],
 SortedWeights[T*K]
 (2) moe_gather -> GroupedInput[T*K, H] (, 1 launch)
 (2b) moe_smoothq -> QGroupedInput, QScale (, 1 launch)
 (3) gate gemm -> GateOut[T*K, I] ( block_scale_gemm)
 (3b) up gemm -> UpOut[T*K, I] ( block_scale_gemm)
 (4) moe_silu_mul -> Hidden[T*K, I] (, 1 launch)
 (4b) moe_smoothq -> QHidden, HScale (, 1 launch)
 (5) down gemm -> DownOut[T*K, H] ( block_scale_gemm)
 (6) moe_reduce -> Y[T, H] (atomic, f32) (, 1 launch)

 ships three kernels:

1. ``build_moe_gather`` -- for each bucket ``b in [0, T*K)``: read
 ``token_id = SortedTokenIds[b]``, then stream
 ``X[token_id, :]`` into ``GroupedInput[b, :]``. The CTA per bucket
 loads the wave-uniform ``token_id`` once and derives every lane's
 column offset via :func:`ck_dsl.helpers.gather_row_offset`.

2. ``build_moe_silu_mul`` -- SwiGLU activation fusion across the gate
 / up MLP output: ``Hidden[b, i] = silu(GateOut[b, i]) * UpOut[b,
 i]`` with ``silu(x) = x * sigmoid(x)``. Compute is in f32; the
 sigmoid uses ``exp2(-x * log2(e))`` so the AMDGPU backend lowers it
 to one ``v_exp_f32`` + ``v_rcp_f32`` per element.

3. ``build_moe_topk_weighted_reduce`` -- the final atomic-accumulate.
 For each bucket ``b``: load ``token_id`` and ``weight = SortedWeights[b]``,
 then per hidden column ``h``: ``atomic_add(Y[token_id, h], weight *
 DownOut[b, h])``. The accumulator is f32 (``Y`` must be pre-cleared);
 the caller's follow-on kernel does the f32->dtype cast if needed.

The :class:`FusedMoeLauncher` class is a documentation-grade
descriptor of the call graph. It returns
``[(phase_name, kernel_def, grid, signature), ...]`` for the MoE
stages and exposes ``expert_gemm_shape(stage, expert_count)`` so the
caller can size per-expert ``block_scale_gemm`` launches. The
per-expert GEMM dispatch loop (which depends on the runtime
``Counts[e]``) is left to the caller's runtime driver -- v1 keeps the
launcher pure-Python and import-time-safe; the production runtime
launcher (with per-expert workspace reuse) is a v2 follow-on.

 helpers:

- ``helpers/gather_scatter.py``
 - ``gather_row_offset(b, sorted_token_ids, bucket_idx, *, hidden,
 col)`` -- one SSA chain for ``sorted_token_ids[bucket_idx] *
 hidden + col``, the canonical per-token indirect address.
 - ``scatter_token_offset(b, token_id, *, hidden, col)`` -- the
 write-side counterpart, for parity / readability.
 - ``load_sorted_token_id(b, sorted_token_ids, bucket_idx)`` --
 one ``i32`` global load with the alignment tightened to 4
 bytes.
 - ``load_sorted_topk_weight(b, sorted_weights, bucket_idx)`` --
 one ``f32`` global load, alignment 4.

 failure modes:

- ``hidden % block_size != 0`` or ``intermediate % block_size != 0``
 -- v1 requires one CTA per bucket row to cover the full vector in
 one ``elems_per_thread`` loop. The launcher rejects this at spec
 validation; multi-tile dispatch is a v2 follow-on.
- Topk ids out of range -- the gather and reduce kernels both guard
 with ``cmp_ge(token_id, 0)`` and silently drop the contribution,
 matching CK Tile's behaviour for unused bucket slots (router emits
 sentinel ``-1`` ids for inactive experts).
- ``Y`` accumulator not pre-cleared -- the topk-weighted reduce uses
 atomic-add; non-zero initial contents poison every output row.
- Per-expert GEMM ``M=0`` -- when ``Counts[e] == 0`` the caller must
 *skip* the per-expert launch entirely. The launcher's
 ``expert_gemm_shape`` returns ``(0, I, H)`` faithfully so the
 caller's runtime can guard.

The MoE-sort-specific failure modes from still apply to the
``(0)``->``(1)`` step in the pipeline (workspace counter must be
zero-cleared between phases 2 and 3 of the sort).
