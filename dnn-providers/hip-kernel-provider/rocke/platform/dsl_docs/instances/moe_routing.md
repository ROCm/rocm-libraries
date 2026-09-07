# MoE Routing And Compact Compute Pipeline

## Contents

- [Scope](#scope)
- [Top-k active packing](#top-k-active-packing)
- [Compact gather and quantization](#compact-gather-and-quantization)
- [Fused expert compute](#fused-expert-compute)
- [Static-grid handoff](#static-grid-handoff)
- [Verification](#verification)

## Scope

This pipeline targets small decode batches on gfx950. It turns f32 router logits
and narrow input activations into the compact metadata and block-quantized
activation layout consumed by the fused expert kernel.

The implementation is split at synchronization boundaries:

1. correction-biased top-k and compact metadata packing;
2. routed-row gather plus blockwise FP8 quantization;
3. fused gate/up, gated activation, down projection, and weighted reduction.

All stages use fixed-capacity output buffers and sentinel block ids, so the
consumer can launch without reading a dynamic block count back to the host.

## Top-k active packing

[`moe_topk_active_pack.py`](../../python/rocke/instances/common/moe_topk_active_pack.py)
implements the one-workgroup decode route:

```text
score[token, expert] = sigmoid(logit[token, expert])
selection_score = score + correction_bias
ids = topk(selection_score)
weights = normalize(gather(score, ids)) * routed_scale
```

The correction bias affects selection only; routed weights retain the unbiased
sigmoid scores. Equal selection scores choose the smaller expert id.

The same workgroup builds:

- per-expert routed-pair counts;
- exclusive offsets over `ceil(count / tile_m)` blocks;
- compact `BlockExpertIds`;
- padded `SortedTokenIds`, `SortedTopkIds`, and `SortedWeights`.

Version 1 supports the one-group routing case, `topk <= 32`, and
`tokens * topk <= block_size`. The group fields remain in the spec so a later
multi-group selector can extend the contract explicitly.

## Compact gather and quantization

[`moe_compact_gather_quant.py`](../../python/rocke/instances/common/moe_compact_gather_quant.py)
gathers source rows through `SortedTokenIds` and quantizes each
`[tile_m, 128]` slab to FP8 E4M3.

One scale is shared by all rows in that slab and copied into every row's scale
slot. This is a correctness requirement for the current MegaMoE accumulator:
one MFMA fragment spans multiple output rows, so a single post-instruction
activation scale cannot represent independently quantized token rows.

Inactive sentinel blocks produce zero activation data and a finite floor scale.

## Fused expert compute

[`moe_fused_mega_fp8.py`](../../python/rocke/instances/common/moe_fused_mega_fp8.py)
now supports:

- FP8 E4M3 activations with existing FP8 E4M3 block-scaled weights;
- FP8 E4M3 activations with packed MXFP4 E2M1 weights and unsigned E8M0 scales;
- SiLU gated multiplication;
- SITU gated multiplication with required gate beta and optional linear beta.

The initial MXFP4 path is a correctness-first implementation. Each packed E2M1
fragment is recoded exactly into FP8 registers, then a K=32 FP8 MFMA is scaled
by the activation block scale and the decoded E8M0 weight scale. Four such
groups cover each 128-wide contraction slab. This preserves the original
per-32 MX scale semantics without requiring the mixed-format instruction in the
first implementation.

SITU uses an `exp2`-based stable tanh formulation because the gfx950 COMGR path
does not accept a generic `llvm.tanh` call in this kernel:

```text
gate = beta * tanh(gate / beta) * sigmoid(gate)
up = linear_beta * tanh(up / linear_beta)  # when configured
hidden = gate * up
```

## Static-grid handoff

For a routing spec with `P = tokens * topk`:

```text
BlockExpertIds capacity = P
Sorted* capacity = P * tile_m
compact activation rows = P * tile_m
```

Only the prefix reported by `NumBlocks` is active. Every remaining block id is
`-1`; the fused expert kernel's existing active-block guard skips those work
items. This avoids a host synchronization between routing and compute.

## Verification

CPU structure, dispatch, and lowering:

```text
python -m pytest tests/instances/test_moe_topk_active_pack.py
python -m pytest tests/instances/test_moe_compact_gather_quant.py
python -m pytest tests/instances/test_moe_fused_mega_situ.py
python -m pytest tests/dispatch/dispatch_tests/moe
```

gfx950 numeric verification:

```text
python -m rocke.examples.gfx950.moe.verify_rank_reduce_and_routing
python -m rocke.examples.gfx950.moe.verify_fused_moe_situ
python -m rocke.examples.gfx950.moe.verify_moe_reduction_pipeline
```

The final command exercises the full static-grid handoff at both one and eight
tokens with 896 router experts and top-k 16.
