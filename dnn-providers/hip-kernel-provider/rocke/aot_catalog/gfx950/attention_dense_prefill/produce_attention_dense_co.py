# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Co-located, build-time producer for the gfx950 attention_dense_prefill family:
# emit rocKE's dense flash-attention prefill .co (HSACO) into argv[1]. Runtime never
# touches rocKE -- the per-family CMakeLists runs this at build time to drop the .co
# next to the checked-in family.json (family.json is the source of truth).
#
# Normally run by this family's CMakeLists via the rocKE build interpreter
# (${ROCKE_PYENV_PYTHON}), which editable-installs BOTH rocKE source roots:
#   platform/ -> package `rocke`      library/ -> packages `kernels`, `builders`, `dispatch`
# so `from kernels.gfx950.attention_dense import ...` works with no PYTHONPATH surgery.
# To run standalone:  <build>/rocke-pyenv/bin/python produce_attention_dense_co.py <out>
#
# WHAT THIS BUILDS: rocKE's gfx950-native dense prefill attention
# (`kernels.gfx950.attention_dense.build_attention_dense`) -- the `attention_dense`
# path measured on MI355X at +155% median over rocKE's older 2-D attention path and
# roughly at parity with aiter-Triton on its covered surface. It is CDNA4/MFMA and
# gfx950-only; `supports_attention_dense` rejects every other arch.
#
# THREE THINGS THAT MAKE THIS FAMILY DIFFERENT FROM THE gfx1151 fmha FAMILY:
#
#  1. SHAPE IS COMPILE-TIME. batch/seqlen/heads/head_size are baked into the kernel
#     (n_ktiles = S_kv // block_n becomes a const_i32 loop bound), so each .co serves
#     exactly one shape and family.json must pin S_q/S_kv/H/H_kv/D/B with `equals`,
#     never `multiple_of`. The shape table below is the whole coverage surface.
#
#  2. THE KERNEL WANTS THE RAW SCALE. gfx1151's kernel takes a pre-multiplied
#     `scale_log2`; this one does `qk_scale = scale * log2(e)` internally, so
#     family.json names `scale_raw`. Naming `scale_log2` would apply log2(e) twice.
#
#  3. THE ABI HAS NO STRIDE ARGUMENTS. The kernel hardcodes packed BSHD addressing
#     (stride_token = H*D, stride_head = D, stride_batch = S*H*D). family.json
#     therefore constrains the `bshd_packed` fact, which SdpaAdapter::decode
#     publishes for exactly this purpose -- see the family.json _comment.

import json
import os
import sys

from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense, supports_attention_dense
from rocke.helpers.compile import compile_kernel

ARCH = "gfx950"

# Fixed knobs for every entry.
#   block_n=64        -- 64 and 128 both match ~peak; 64 uses ~74 KB LDS against the
#                        gfx950 160 KB budget, 128 needs ~148 KB and pins the 256-VGPR
#                        cap. 64 is strictly more resource-efficient.
#   waves_per_eu=2    -- measured free win; 3 forces spills (-20%).
#   lds_k_group_pad=8 -- only live on the packed head_size<128 path; harmless at 128.
#   lazy_rescale=True -- always-on adaptive online-softmax rescale, parity-identical.
_BLOCK_N = 64
_WAVES_PER_EU = 2
_LDS_K_GROUP_PAD = 8
_LAZY_RESCALE = True

# Shape table. Llama-3-8B-shaped GQA prefill (Hq=32, Hkv=8, D=128) is the canonical
# rocKE dense workload and what the injection's SDPA calls look like on a real
# decoder model. seqlen_q must be a multiple of 256 (BLOCK_M) and seqlen_kv a
# multiple of block_n, both enforced by AttentionDenseSpec.__post_init__.
#
# The small (1, 256, 4, 1, 128) entry is the parity-test shape: 256 is the minimum
# legal seqlen_q (one BLOCK_M) and gqa_ratio 4 with a single KV head keeps the CPU
# reference softmax cheap while still exercising the GQA head mapping. It is a real
# selectable kernel, not a test-only artifact.
#
# BATCH IS A COMPILE-TIME AXIS, not just a grid dimension. `B` sizes the K/V buffer
# resource extents as a const_i32 (`buffer_rsrc(k, const_i32(B*Skv*Hkv*D*2))`) and the
# persistent work-item count `W = NQB*Hq*B`. A B=1 binary launched with grid.z=4 would
# read past the descriptor, and AMD buffer loads return 0 out-of-range instead of
# faulting -- silently wrong attention. So every batch size needs its own .co, and
# family.json must pin B with `equals`. (rocKE makes the same point by keying its own
# launcher cache on (kernel_name(), batch).)
#
# (batch, seqlen, num_query_heads, num_kv_heads, head_size)
#
# MUST MATCH gen_dense_family_json.py exactly. Geometry sweep from the ckdsl attention
# bench matrix (flash_gqa_prefill + flash_mha_prefill); each was confirmed buildable by
# supports_attention_dense() first (remote/29_bench_coverage.py).
_BENCH_GEOMS = [
    (32, 8),    # Llama-3-8B, MiMo-V2      gqa 4
    (40, 8),    # Qwen3-14B                gqa 5
    (28, 4),    # Qwen2.5-7B               gqa 7
    (64, 8),    # Llama-3-70B              gqa 8
    (128, 8),   # Llama-3.1-405B           gqa 16
    (64, 4),    # Qwen3-235B-A22B          gqa 16
    (32, 32),   # Llama-2-7B               MHA
    (40, 40),   # GPT-3-13B-class          MHA
]
_BENCH_SEQLENS = (512, 1024, 2048, 4096, 8192)

# Vision / diffusion and head_size-64 serving. Neither `causal=False` nor `ragged` is a
# new kernel capability -- both already exist; the table simply never asked. `ragged`
# bounds-checks a partial final tile so a seqlen that is not a multiple of
# BLOCK_M/block_n compiles (self-attention only), which is what admits ViT's 197 and 257.
#
# (batch, seqlen, num_query_heads, num_kv_heads, head_size, causal, ragged)
_TRACK_A_SHAPES = [
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
_SHAPES = [
    (1, 256, 4, 1, 128, True, False),   # parity shape: one BLOCK_M, GQA 4:1, cheap reference
    (2, 256, 4, 1, 128, True, False),   # batched parity shape -- small enough for a reference
    (1, 256, 4, 1, 64, False, False),   # parity: non-causal + packed D=64 path
    (1, 197, 4, 1, 64, False, True),    # parity: ragged tile + non-causal key-pad mask
]
_SHAPES += [(1, s, hq, hkv, 128, True, False) for (hq, hkv) in _BENCH_GEOMS for s in _BENCH_SEQLENS]
_SHAPES += [(b, s, 32, 8, 128, True, False) for s in (2048, 4096) for b in (2, 4, 8)]
_SHAPES += _TRACK_A_SHAPES

# Chunked prefill (vLLM / SGLang): a chunk of queries against a longer KV cache. Requires
# causal_bottom_right, because the queries are the LAST S_q positions of the sequence --
# top-left alignment would let query 0 see only key 0 and be silently wrong.
# (batch, S_q, S_kv, hq, hkv, d, causal, ragged, bottom_right)
_CHUNKED_SHAPES = [
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
# tile-aligned; a real chunk is not, because the KV cache holds whatever it holds. Ragged
# pads the boundary tiles on-chip, so no host padding is needed. Kept in step with
# RAGGED_CHUNKED_SHAPES in gen_dense_family_json.py.
_RAGGED_CHUNKED_SHAPES = [
    (1, 197, 400, 4, 1, 128, True, True, True),    # parity: both lengths off-tile, 1 qblock
    (1, 300, 1234, 4, 1, 128, True, True, True),   # parity: 2 query blocks, partial second
]


def _norm(t):
    """Widen a (b, S, ...) self-attention tuple to the full 9-field form."""
    if len(t) == 7:
        b, s, hq, hkv, d, causal, ragged = t
        return (b, s, s, hq, hkv, d, causal, ragged, False)
    return t


_SHAPES = [_norm(t) for t in _SHAPES] + _CHUNKED_SHAPES + _RAGGED_CHUNKED_SHAPES
_SHAPES = list(dict.fromkeys(_SHAPES))

# Spec spelling ("fp16"/"bf16") differs from the catalog dtype constraint token
# ("f16"/"bf16"); family.json carries the latter.
_DTYPES = ["bf16", "fp16"]

# Ship persistent and non-persistent for every shape and let the engine's
# measure-and-cache selection pick. Persistent measured ~1.75x faster (512 -> 853
# TFLOPS at Sq=8192), and since the shapes are pinned with `equals` anyway the extra
# candidate costs nothing in constraint terms. num_persistent=256 is exactly one
# 8-wave block per CU on a 256-CU MI355X.
_PERSISTENT = [False, True]
_NUM_PERSISTENT = 256


def _specs():
    for (b, sq, skv, hq, hkv, d, causal, ragged, bottom_right) in _SHAPES:
        for dtype in _DTYPES:
            for persistent in _PERSISTENT:
                # causal_bottom_right is non-persistent only: the persistent path
                # prunes the KV loop from the same diagonal and needs its own offset.
                if bottom_right and persistent:
                    continue
                yield AttentionDenseSpec(
                    batch=b,
                    seqlen_q=sq,
                    seqlen_kv=skv,
                    causal_bottom_right=bottom_right,
                    num_query_heads=hq,
                    num_kv_heads=hkv,
                    head_size=d,
                    causal=causal,
                    dtype=dtype,
                    sliding_window=0,
                    ragged=ragged,
                    varlen=False,
                    block_n=_BLOCK_N,
                    waves_per_eu=_WAVES_PER_EU,
                    lds_k_group_pad=_LDS_K_GROUP_PAD,
                    persistent=persistent,
                    num_persistent=_NUM_PERSISTENT,
                    lazy_rescale=_LAZY_RESCALE,
                )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # The checked-in family.json is the contract; every symbol we emit must appear in
    # it, and every symbol it declares must be emitted. A mismatch would either drop
    # the whole family at catalog load (a missing .co makes parseKernel throw, which
    # aborts the entire file) or leave a kernel that can never be selected.
    # Match on (symbol, co_file), not symbol alone: kernel_name() omits batch, so the
    # B=1 and B=4 builds of one shape export the SAME symbol and are distinguished only
    # by which .co they live in. Catalog.cpp supports that -- it keeps kernels in a
    # per-entry vector and resolves each entry's symbol inside its own co_file.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "family.json")) as f:
        declared = {(k["symbol"], k["co_file"]) for k in json.load(f)["kernels"]}

    emitted = set()
    failed = []

    for spec in _specs():
        ok, why = supports_attention_dense(spec, arch=ARCH)
        if not ok:
            failed.append(f"{spec.kernel_name()}: {why}")
            continue

        artifact = compile_kernel(build_attention_dense(spec, arch=ARCH), arch=ARCH)
        if not artifact.hsaco:
            failed.append(f"{spec.kernel_name()}: compiled .co is empty")
            continue

        # Take the symbol from the ARTIFACT, not the spec: kernel_name() omits `batch`
        # and `waves_per_eu`, and the persist-decode suffix ("hkvmaj") is derived from
        # a gqa*NQB*B >= 2*num_persistent inequality, so the emitted name is the only
        # authority.
        symbol = artifact.kernel_name
        co_file = f"{symbol}__b{spec.batch}.co"
        emitted.add((symbol, co_file))
        with open(os.path.join(out_dir, co_file), "wb") as f:
            f.write(artifact.hsaco)
        print(
            f"symbol={symbol} B={spec.batch} S={spec.seqlen_q} "
            f"Hq={spec.num_query_heads} Hkv={spec.num_kv_heads} D={spec.head_size} "
            f"dtype={spec.dtype} persistent={spec.persistent} "
            f"grid={_grid(spec)} block={spec.num_waves * 64} "
            f"bytes={len(artifact.hsaco)}"
        )

    missing = declared - emitted
    extra = emitted - declared
    if failed or missing or extra:
        for m in failed:
            print(f"ERROR build: {m}", file=sys.stderr)
        for sym, co in sorted(missing):
            print(f"ERROR family.json declares '{sym}' in '{co}' but no .co was emitted",
                  file=sys.stderr)
        for sym, co in sorted(extra):
            print(f"ERROR emitted '{sym}' as '{co}' but family.json does not declare it",
                  file=sys.stderr)
        return 1

    print(f"OK {len(emitted)} kernel(s), family.json in sync")
    return 0


def _grid(spec):
    from kernels.gfx950.attention_dense import attention_dense_grid

    return attention_dense_grid(spec)


if __name__ == "__main__":
    raise SystemExit(main())
