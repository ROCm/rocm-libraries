# Attention Instances

The attention implementation is spread across:

Pre-roadmap (production tiled MFMA, paged-KV decode):
- `instances/attention_unified.py`
- `instances/attention_tiled_2d.py`
- `instances/attention_tiled_3d.py`
- `helpers/attention.py`

 FMHA expansion (CK Tile `01_fmha` family parity):
- `instances/_fmha_common.py` -- shared FmhaCommonSpec + scalar-inner body
- `instances/fmha_varlen.py` -- 01_fmha varlen
- `instances/fmha_appendkv.py` -- 01_fmha appendkv (with optional rotary)
- `instances/fmha_paged_prefill.py` -- 01_fmha pagedkv_prefill
- `instances/fmha_splitkv_decode.py` -- 01_fmha splitkv (segment + reduce)
- `instances/fmha_head_grouping.py` -- 01_fmha head_grouping (GQA / MQA)
- `instances/fmha_bwd.py` -- 01_fmha bwd (dQ / dK / dV atomic)
- `instances/fmha_fwd_fp8.py` -- 01_fmha fp8 (per-tensor scales)

Helpers:
- `helpers/rotary.py` -- RoPE (interleaved + half layouts)
- `helpers/rng.py` -- Philox4x32-10 RNG for dropout
- `helpers/attention.py` (ext) -- alibi_bias_log2, alibi_bias_matrix, custom_mask

 Sage + sparse attention (CK Tile `49_sageattention`, `50_sparse_attn`):
- `instances/sage_attention.py` -- 4 quant variants (fp16/bf16, fp8-bf16, i8-fp8, i4-fp8)
- `instances/sparse_attention.py` -- jenga + VSA sparse fwd

Helpers:
- `helpers/qk_scale.py` -- per-head / per-block Q+K scale loaders
- `helpers/codebook.py` -- i8 / i4 -> fp8 codebook dequant chains
- `helpers/sparse_iter.py` -- block-sparse bitmap + VSA LUT K-iterators

The shipped stack supports scalar correctness kernels, tiled 2D MFMA kernels,
and tiled 3D split-KV pipelines pre-; added the v1 scalar-inner
kernels for every FMHA variant CK Tile ships in `01_fmha`, and closed
the roadmap with Sage attention (four quant variants sharing one builder) and
two sparse attention kernels (Jenga block-sparse + VSA variable-size). All v2
MFMA hoists drop in via the same per-variant spec surface (same pattern
 use for StreamK / block-scale GEMM).

## Main Concepts

`UnifiedAttentionProblem` describes the problem and selects between paths.

The runtime entry point:

```text
run_unified_attention_torch(...)
```

Path selection:

```text
auto -> prefer tiled 3D for long-context decode
 -> use tiled 2D where appropriate
 -> scalar 2D for simple correctness/reference paths
```

The runtime caches HSACO and launchers by semantic problem keys. This matters because attention benchmarks should not include recompilation or module-load costs.

## Attention Features

Current coverage (verified via `examples/attention/parity_unified_attention.py` `default` scenario set):

- causal masking (`helpers/attention.py::causal_mask`);
- sliding window (`sliding_window_mask`);
- softcap (`apply_softcap_log2`, `apply_softcap_scalar`);
- sinks;
- ALiBi (positional bias);
- QQ bias (query-query bias);
- fp16 and bf16 storage;
- head sizes 128 and 256;
- block_size 16 and 64;
- paged KV cache (transform DAG `indirect + unmerge` for `block_tables`);
- 2D scalar correctness kernel + 2D MFMA tiled kernel;
- 3D split-KV (segment + reduce pipeline).

bf16 attention uses the `mfma_f32_16x16x16_bf16` op via the `_1k` LLVM variant (with `<4 x i16>` operands; the plain `bf16.16x16x16` intrinsic does not exist on this target).

CK Tile ``01_fmha`` family parity:
- variable-length packed-batch fwd (`fmha_varlen.py`);
- KV-cache append with optional rotary apply (`fmha_appendkv.py`);
- paged-KV prefill (`fmha_paged_prefill.py`);
- two-launch split-KV decode (`fmha_splitkv_decode.py`);
- GQA / MQA (`fmha_head_grouping.py`);
- backward dQ / dK / dV with atomic fp32 accumulate (`fmha_bwd.py`);
- FP8 KV cache with per-tensor `k_scale` / `v_scale` (`fmha_fwd_fp8.py`);
- additional mask families: alibi-bias-log2, alibi-bias-matrix, custom-mask (i8);
- rotary position embedding helpers: interleaved + LLaMA-half layouts;
- Philox4x32-10 RNG for dropout (CK Tile bit-for-bit compatible).

 kernels are v1 scalar-inner correctness oracles -- the body is shared
across every variant via `_fmha_common.fmha_fwd_inner_body`. The v2 MFMA
hoists drop in via the same spec surface; the production tiled MFMA bodies
already live in `attention_tiled_2d.py` / `attention_tiled_3d.py` and will be
hoisted under the same `FmhaCommonSpec` in v2.

The FP8 path uses the existing `cvt_fp8_to_f32` / `cvt_bf8_to_f32`
intrinsics and per-tensor scales. Per-block / per-head Q+K scales arrive in
 via `helpers.qk_scale` and are wired into `instances.sage_attention`.

CK Tile ``49_sageattention``, ``50_sparse_attn``:
- Sage attention fwd with four ``quant_mode`` variants:
 - ``"fp16_bf16"`` -- baseline (pipeline validation path).
 - ``"fp8_bf16"`` -- fp8e4m3 K/V + per-block scales (real Sage).
 - ``"i8_fp8_bf16"`` -- i8 K/V; codebook re-materialises fp8 per element.
 - ``"i4_fp8_bf16"`` -- packed i4 K/V (two nibbles per byte) + 16-entry codebook.
- Block-sparse attention (``jenga_sparse_attention``) -- one-hot
 ``Mask[q_block, k_block]`` bitmap drives a per-K-block ``scf.if`` guard.
- Variable-size attention (``vsa_sparse_attention``) -- per-q LUT
 ``BlockLut[q_block, slot]`` of length ``BlockCount[q_block]``.
- Per-head / per-block QK scale loaders shared across the Sage family.
- i8 / i4 codebook dequant chains shared with the Sage int variants.
- Block-sparse / VSA K-iterators shared between the two sparse kernels.

## Helper Layer

File:

```text
helpers/attention.py
```

Important helper ideas:

- online softmax state;
- paged KV descriptor;
- query/output descriptors;
- config selection for 2D and 3D;
- warp XOR reductions for max/sum;
- softcap in log2 space;
- causal and sliding-window masks;
- dtype-specific MFMA dispatch helpers.

Attention uses both raw IR and descriptors. Addressing is descriptor-friendly; butterfly reductions and some per-warp control patterns are specialized enough to stay closer to raw IR.

## Scalar 2D Kernel

File:

```text
instances/attention_unified.py
```

Concept:

```text
one workgroup computes one output scalar or narrow output unit for a query/head/dim coordinate
```

This is not the fastest path. It is valuable because it is direct and easier to validate.

Step by step:

```text
1. Decode workgroup IDs into query token, query head, and dimension.
2. Read sequence metadata such as cu_q and KV lengths.
3. Locate the query vector Q.
4. Initialize online softmax state:
 m = -inf
 l = 0
 acc = 0
5. Loop over all valid KV tokens for this query.
6. For each KV token:
 - resolve paged KV physical block through block table;
 - load K element(s);
 - accumulate qk dot in f32;
 - apply scale;
 - apply optional softcap;
 - apply causal/sliding/window masks;
 - update online softmax max m;
 - update denominator l;
 - load V;
 - update weighted accumulator.
7. Normalize acc by l.
8. Convert to output dtype.
9. Store output.
```

Correctness-sensitive details:

- the online softmax update must rescale prior accumulator when a new max appears;
- mask order must match reference semantics;
- paged KV table lookup must be part of address computation, not a post-hoc pointer adjustment;
- output should be compared against the AITER/PyTorch reference with dtype-appropriate tolerances.

## Tiled 2D Kernel

File:

```text
instances/attention_tiled_2d.py
```

Concept:

```text
one CTA covers a Q block and one KV head
```

High-level stages:

```text
1. Stage Q into LDS.
2. Iterate over K/V tiles.
3. Compute QK with MFMA.
4. Apply masks/bias/softcap.
5. Online softmax reduce per row.
6. Compute P*V with MFMA.
7. Store output with coalesced vector stores.
```

Detailed flow:

```text
1. Workgroup IDs identify:
 - query block;
 - KV head;
 - batch/sequence context.

2. Load query tile:
 - descriptor maps query token/head/dim to Q memory;
 - threads cooperatively stage Q into LDS;
 - invalid query rows are masked.

3. For each KV tile:
 - use paged-KV descriptor to map logical token to physical block;
 - async-load or buffer-load K tile;
 - load V tile as needed;
 - wait/sync before LDS reads.

4. QK:
 - read Q and K fragments from LDS;
 - issue MFMA for q x k;
 - produce f32 score fragments.

5. Score processing:
 - apply scale;
 - apply ALiBi/QQ bias if configured;
 - apply causal and sliding-window masks;
 - apply softcap when configured;
 - set invalid positions to -inf.

6. Online softmax:
 - reduce row max across lanes/waves;
 - compute exp2(score - max);
 - reduce row sum;
 - rescale previous accumulator and denominator;
 - update state.

7. P*V:
 - probabilities are staged/arranged for MFMA;
 - V fragments are read, often with transpose-friendly LDS patterns;
 - accumulate f32 output fragments.

8. Output:
 - normalize by denominator;
 - convert to fp16/bf16;
 - cshuffle-like or vectorized store to output descriptor.
```

Tiled 2D is useful for chunked prefill, sliding-window rows, and moderate contexts. It can under-occupy for long single-token decode compared with split-KV 3D.

## Tiled 3D Split-KV Kernel

Files:

```text
instances/attention_tiled_3d.py
```

The 3D path has two stages:

```text
segment kernel
reduce kernel
```

Runtime uses `PipelineLauncher` to launch both on one stream with managed workspaces.

## 3D Segment Kernel

Grid:

```text
grid_x = total query blocks
grid_y = KV heads
grid_z = KV segments
```

Each workgroup processes one KV segment for one query block/head.

Step by step:

```text
1. Decode q block, kv head, and segment index.
2. Determine segment KV token range.
3. Stage Q.
4. Iterate only over KV tiles in this segment.
5. Compute QK with MFMA.
6. Apply masks/bias/softcap.
7. Maintain online softmax state for this segment:
 - seg_max
 - seg_expsum
 - seg_output accumulator
8. Write segment output workspace:
 - segm_output in f32
 - segm_max
 - segm_expsum
```

The segment kernel does not produce the final normalized output when multiple segments exist. It produces numerically stable partials.

## 3D Reduce Kernel

Grid:

```text
grid_x = total query tokens
grid_y = query heads
grid_z = 1
```

Step by step:

```text
1. Load all segment maxima for the query/head.
2. Compute overall maximum:
 overall_m = max(seg_m)
3. For each segment:
 - weight = exp2(seg_m - overall_m)
 - denominator contribution = seg_expsum * weight
 - output contribution = seg_output * weight
4. Sum denominator and weighted output across segments.
5. Normalize:
 out = weighted_output / denominator
6. Convert to output dtype.
7. Store final output.
```

This is the standard numerically stable split-softmax combine.

## Workspace

The 3D path uses persistent workspace specs:

```text
segm_output
segm_max
segm_expsum
```

`WorkspacePool` keeps torch tensors alive across launches and avoids allocator lifetime races.

## Paged KV Addressing

Paged KV cache address calculation is non-affine because logical KV blocks map through `block_tables`.

Descriptor shape:

```text
logical token/block
 -> indirect table lookup
 -> physical block
 -> token within block
 -> kv head
 -> dimension
 -> byte offset
```

The transform DAG uses `indirect(...)` and `unmerge(...)` so page-table lookup is part of the descriptor rather than scattered through the kernel.

## Online Softmax

The stable update is conceptually:

```text
new_m = max(old_m, score_max)
old_scale = exp2(old_m - new_m)
score_scale = exp2(score - new_m)
new_l = old_l * old_scale + sum(score_scale)
new_acc = old_acc * old_scale + sum(score_scale * V)
```

This avoids materializing the full attention matrix and keeps numerical behavior stable across long sequences.

## Attention Runtime Path

Runtime path:

```text
1. Build semantic key from problem/config.
2. Check HSACO/launcher cache.
3. If missing, build KernelDef and compile_kernel.
4. Create KernelLauncher or PipelineLauncher.
5. Allocate or reuse workspace.
6. Resolve torch stream.
7. Launch selected path.
8. Optionally time with HIP events.
```

This is why benchmark methodology matters: first-run compile and cache creation are not kernel steady-state.

## Attention Failure Modes

- Comparing 2D CK against 3D Triton or vice versa without labeling algorithm.
- Timing Triton and CK with different event/stream mechanisms.
- Missing stream synchronization with torch allocator.
- Workspace tensors freed before asynchronous kernels complete.
- Segment reduce uses unstable softmax combine.
- Causal/sliding masks applied after softmax update instead of before.
- Paged KV table stride or block size mismatch.
- Head mapping wrong for grouped-query attention.
- 3D segment count too small or too large for occupancy.
- Vectorized output store crosses invalid query/head/dim tail.
