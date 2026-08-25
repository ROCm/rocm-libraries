#!/usr/bin/env python3
"""Emit the gfx950 attention_dense_prefill family.json.

Reproduces AttentionDenseSpec.kernel_name() so family.json can exist before the
first build (rocke_add_aot_family warns-and-skips a folder with no family.json).
The producer then cross-checks every emitted artifact.kernel_name against this
file and fails the build on any drift, so the duplication is guarded.

Plain stdlib: runnable before the rocKE pyenv exists.
"""

import json
import os
import sys

ARCH = "gfx950"
FAMILY = "attention_dense_prefill"

BLOCK_M = 256          # kernels.gfx950.attention_dense._BLOCK_M (kernel faults otherwise)
BLOCK_N = 64
NUM_WAVES = BLOCK_M // 32   # 8 wave64s
BLOCK_THREADS = NUM_WAVES * 64   # 512
NUM_PERSISTENT = 256

# Must match produce_attention_dense_co.py exactly.
# Batch is compile-time (it sizes the K/V buffer extents as a const_i32), so each batch
# size is a separate .co pinned with `equals` -- see the producer's comment.
#
# Geometry sweep taken from the ckdsl attention bench matrix
# (bench_cases_extended_attn.json, capabilities flash_gqa_prefill + flash_mha_prefill).
# Every one of these was confirmed buildable by supports_attention_dense() before being
# added -- see remote/29_bench_coverage.py. gqa_ratio and head count are NOT kernel
# limits; the previous ratio-4-only table was self-imposed.
BENCH_GEOMS = [
    (32, 8),    # Llama-3-8B, MiMo-V2      gqa 4
    (40, 8),    # Qwen3-14B                gqa 5
    (28, 4),    # Qwen2.5-7B               gqa 7
    (64, 8),    # Llama-3-70B              gqa 8
    (128, 8),   # Llama-3.1-405B           gqa 16
    (64, 4),    # Qwen3-235B-A22B          gqa 16
    (32, 32),   # Llama-2-7B               MHA
    (40, 40),   # GPT-3-13B-class          MHA
]
BENCH_SEQLENS = (512, 1024, 2048, 4096, 8192)   # all multiples of BLOCK_M=256

# Vision / diffusion and head_size-64 serving shapes. `causal` and `ragged` are not new
# kernel features -- the kernel has both already; the previous table simply never asked
# for them. `ragged` handles a seqlen that is not a multiple of BLOCK_M/block_n by
# bounds-checking the partial tile (self-attention only, so seqlen_q == seqlen_kv), which
# is what lets ViT's 197 and 257 compile at all.
#
# (batch, seqlen, num_query_heads, num_kv_heads, head_size, causal, ragged)
TRACK_A_SHAPES = [
    (2, 1024, 10, 10, 64, False, False),    # SDXL self-attention 32x32
    (2, 256, 20, 20, 64, False, False),     # SDXL self-attention 16x16
    (1, 4608, 24, 24, 128, False, False),   # Flux-MMDiT
    (8, 257, 16, 16, 64, False, True),      # ViT-L/14   (257 -> ragged)
    (16, 197, 12, 12, 64, False, True),     # ViT-B/16   (197 -> ragged)
    (1, 2048, 64, 8, 64, True, False),      # d64-serving GQA8
    (1, 4096, 64, 8, 64, True, False),
    (1, 8192, 64, 8, 64, True, False),
    (2, 1024, 64, 8, 64, True, False),
    (1, 16384, 32, 8, 128, True, False),    # long-context prefill
    (1, 32768, 32, 8, 128, True, False),
]

# (batch, seqlen, num_query_heads, num_kv_heads, head_size, causal, ragged)
SHAPES = [
    (1, 256, 4, 1, 128, True, False),      # parity-test shape (one BLOCK_M, GQA 4:1)
    (2, 256, 4, 1, 128, True, False),      # batched parity-test shape
    (1, 256, 4, 1, 64, False, False),      # parity: non-causal + packed D=64 path
    (1, 197, 4, 1, 64, False, True),       # parity: ragged tile + non-causal key-pad mask
]
# The bench sweep, batch 1 (the matrix is B=1 for every prefill case).
SHAPES += [(1, s, hq, hkv, 128, True, False) for (hq, hkv) in BENCH_GEOMS for s in BENCH_SEQLENS]
# Batch variants, kept to the Llama-3-8B/Qwen3-8B geometry we A/B end-to-end.
SHAPES += [(b, s, 32, 8, 128, True, False) for s in (2048, 4096) for b in (2, 4, 8)]
SHAPES += TRACK_A_SHAPES

# Chunked prefill: a chunk of queries scored against a longer KV cache, which is what
# vLLM and SGLang issue on every chunk. Needs causal_bottom_right -- the queries are the
# LAST S_q positions of the sequence, so top-left alignment would let query 0 see only
# key 0 and compute the wrong attention silently. Nine-tuples carry S_q and S_kv
# separately: (batch, S_q, S_kv, hq, hkv, d, causal, ragged, bottom_right)
CHUNKED_SHAPES = [
    (1, 256, 512, 4, 1, 128, True, False, True),      # parity shape: cheap CPU reference
    (1, 512, 8192, 32, 8, 128, True, False, True),    # Llama-3-8B 512-chunk vs 8K
    (1, 2048, 8192, 32, 8, 128, True, False, True),   # 2K chunk vs 8K
    (1, 2048, 32768, 32, 8, 128, True, False, True),  # 2K chunk vs 32K long cache
    (1, 256, 4096, 32, 8, 128, True, False, True),    # small chunk vs 4K
    (1, 512, 8192, 40, 8, 128, True, False, True),    # Qwen3-14B, non-power-of-2 GQA
    (1, 512, 8192, 28, 4, 128, True, False, True),    # Qwen2.5-7B, NQK=7
    (1, 2048, 16384, 64, 8, 128, True, False, True),  # Llama-3-70B 2K chunk vs 16K
    (1, 2048, 16384, 64, 4, 128, True, False, True),  # Qwen3-235B MoE attention
    (1, 2048, 32768, 40, 8, 128, True, False, True),  # Qwen3-14B long-cache chunk
]

# Chunked prefill at ARBITRARY lengths, via the ragged path. The shapes above are all
# tile-aligned, which a real chunk is not: the KV cache holds whatever it holds. Ragged
# pads the boundary tiles on-chip, so these need no host padding.
#
# The risk these cover is the partial last KV tile. Plain causal skips the key-pad mask
# because every query stops before the padding; bottom-right shifts every query's reach
# right, so the parity cases below are what confirm the last real query lands on
# S_kv - 1 exactly and no further.
RAGGED_CHUNKED_SHAPES = [
    (1, 197, 400, 4, 1, 128, True, True, True),    # parity: both lengths off-tile, 1 qblock
    (1, 300, 1234, 4, 1, 128, True, True, True),   # parity: 2 query blocks, partial second
]


def _norm(t):
    """Widen a (b, S, ...) self-attention tuple to the full 9-field form."""
    if len(t) == 7:
        b, s, hq, hkv, d, causal, ragged = t
        return (b, s, s, hq, hkv, d, causal, ragged, False)
    return t


SHAPES = [_norm(t) for t in SHAPES] + CHUNKED_SHAPES + RAGGED_CHUNKED_SHAPES
SHAPES = list(dict.fromkeys(SHAPES))   # order-preserving dedupe
DTYPES = ["bf16", "fp16"]     # spec spelling
PERSISTENT = [False, True]

# Spec dtype spelling -> catalog constraint token.
DTYPE_TOKEN = {"bf16": "bf16", "fp16": "f16"}

# The dense kernel's ABI is 5 arguments and a strict subset of what
# SdpaAdapter::buildBindings already emits, so no adapter change is needed to BIND
# it. Note scale_raw, not scale_log2: this kernel multiplies by log2(e) internally.
ARGS = [
    {"name": "Q", "type": "ptr"},
    {"name": "K", "type": "ptr"},
    {"name": "V", "type": "ptr"},
    {"name": "O", "type": "ptr"},
    {"name": "scale_raw", "type": "f32"},
]


def resolved_persist_decode(hq, hkv, seqlen_q, batch, num_persistent):
    """Mirror AttentionDenseSpec.resolved_persist_decode for 'auto'."""
    gqa = hq // hkv
    nqb = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    if gqa > 1 and gqa * nqb * batch >= 2 * num_persistent:
        return "hkv_major"
    return "qb_major"


def kernel_name(b, sq, skv, hq, hkv, d, dtype, persistent, causal=True, ragged=False,
                bottom_right=False):
    parts = ["rocke_attention_dense", f"d{d}", f"hq{hq}", f"kv{hkv}", f"bn{BLOCK_N}", dtype]
    # kpad is only in the name on the packed head_size<128 path.
    if 128 // d > 1:
        parts.append("kpad8")
    parts += [f"sq{sq}", f"sk{skv}", "causal" if causal else "full"]
    if bottom_right:
        parts.append("br")
    if ragged:
        parts.append("ragged")
    parts.append("lazyrs")
    if persistent:
        parts.append(f"persist{NUM_PERSISTENT}")
        if resolved_persist_decode(hq, hkv, sq, b, NUM_PERSISTENT) == "hkv_major":
            parts.append("hkvmaj")
    return "_".join(parts)


def entry(b, sq, skv, hq, hkv, d, dtype, persistent, causal=True, ragged=False,
          bottom_right=False):
    sym = kernel_name(b, sq, skv, hq, hkv, d, dtype, persistent, causal, ragged,
                      bottom_right)
    # kernel_name() omits batch, so batch variants of one shape share a symbol and are
    # told apart only by co_file. Catalog.cpp resolves each entry's symbol inside its
    # own .co, so this is legal; the producer writes the same name.
    co_file = f"{sym}__b{b}.co"
    if persistent:
        # A 1-D grid of long-lived CTAs that grid-strides over the work items. This is
        # a compile-time literal, not a runtime CU query, so it is expressible.
        grid = {"x": NUM_PERSISTENT, "y": 1, "z": 1}
    else:
        grid = {"x": {"ceil_div": ["S_q", BLOCK_M]}, "y": "H", "z": "B"}
    return {
        "symbol": sym,
        "co_file": co_file,
        "constraints": {
            "dtype": {"equals": DTYPE_TOKEN[dtype]},
            # Every shape axis is baked into the .co, so all of these are exact.
            "B": {"equals": b},
            "S_q": {"equals": sq},
            "S_kv": {"equals": skv},
            "H": {"equals": hq},
            "H_kv": {"equals": hkv},
            "D": {"equals": d},
            "gqa_ratio": {"equals": hq // hkv},
            # --- layout ---
            "d_contiguous": {"equals": True},
            "batch_foldable": {"equals": True},
            # The kernel has NO stride args and hardcodes packed BSHD. batch_foldable
            # alone does NOT cover this: it returns true unconditionally at B == 1, so
            # a contiguous BHSD graph would otherwise select this kernel and compute
            # garbage. bshd_packed is the guard.
            "bshd_packed": {"equals": True},
            # --- capability posture: every key stated, none left to default ---
            # Omitting a key asserts the kernel handles that case, and selection
            # skips absent keys -- so an unstated key fails OPEN. Every one is
            # pinned. Causal is per-kernel: the family ships both top-left causal
            # prefill and non-causal (vision/diffusion) variants, and a problem can
            # only ever match the one whose `causal` value it published.
            "causal": {"equals": causal},
            # Top-left vs bottom-right is a numerical difference, not a performance
            # one, so it is pinned like any other capability: a chunked-prefill
            # graph (which publishes causal_bottom_right=true) can only match a
            # kernel built with the offset, and vice versa.
            "causal_bottom_right": {"equals": bottom_right},
            "has_diagonal_band": {"equals": False},
            "has_mma_core_mode": {"equals": False},
            "has_alibi": {"equals": False},
            "has_padding_mask": {"equals": False},
            "has_attn_mask": {"equals": False},
            "has_block_mask": {"equals": False},
            "has_sink": {"equals": False},
            "has_dropout": {"equals": False},
            "paged": {"equals": False},
            "varlen": {"equals": False},
            "gen_stats": {"equals": False},
            "fp8": {"equals": False},
            "runtime_scale": {"equals": False},
        },
        "grid": grid,
        "block": [BLOCK_THREADS, 1, 1],
        "args_signature": ARGS,
        # LDS is a static addrspace(3) module allocation, not a dynamic launch
        # parameter, and the kernel takes no workspace pointer.
        "workspace_bytes": 0,
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "family.json"
    kernels = [
        entry(b, sq, skv, hq, hkv, d, dt, p, causal, ragged, br)
        for (b, sq, skv, hq, hkv, d, causal, ragged, br) in SHAPES
        # causal_bottom_right is non-persistent only (the persistent path prunes
        # from the same diagonal and needs its own offset), so skip that pairing.
        for dt in DTYPES
        for p in PERSISTENT
        if not (br and p)
    ]

    doc = {
        "family": f"{FAMILY}_{ARCH}",
        "op_kind": "sdpa",
        "arch": ARCH,
        "dtype": ["bf16", "f16"],
        "_comment": (
            "rocKE gfx950 dense flash-attention prefill (kernels.gfx950.attention_dense), "
            "CDNA4/MFMA, self-attention only (S_q == S_kv), top-left causal or "
            "non-causal. Shape is COMPILE-TIME in this kernel "
            "-- batch/seqlen/heads/head_size are baked into each .co (the KV loop trip "
            "count is a const_i32), so every shape axis is pinned with `equals` and the "
            "kernel list IS the coverage surface; there is no multiple_of range to widen. "
            "That includes BATCH: B sizes the K/V buffer resource extents as a const_i32, "
            "and an AMD buffer load returns 0 out-of-range rather than faulting, so a B=1 "
            "binary launched at grid.z=4 would silently attend over zeros. Each batch size "
            "is therefore its own .co. Since kernel_name() omits batch, those variants "
            "share a symbol and are distinguished by co_file, which the catalog supports. "
            "The ABI is 5 args (Q,K,V,O + f32 scale) and takes the RAW scale: the kernel "
            "multiplies by log2(e) itself, unlike the gfx1151 fmha family which wants a "
            "pre-multiplied scale_log2. It has no stride arguments and hardcodes packed "
            "BSHD (stride_token=H*D, stride_head=D, stride_batch=S*H*D), which is what a "
            "model projecting to [B,S,H,D] and calling .transpose(1,2) hands to SDPA; a "
            ".contiguous() BHSD tensor must be declined, hence the bshd_packed constraint. "
            "Each shape ships a persistent and a non-persistent variant so the engine's "
            "measure-and-cache selection can pick; they are numerically identical."
        ),
        "kernels": kernels,
    }

    with open(out, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")
    print(f"wrote {out}: {len(kernels)} kernels")
    for k in kernels:
        print("  ", k["symbol"])


if __name__ == "__main__":
    sys.exit(main())
