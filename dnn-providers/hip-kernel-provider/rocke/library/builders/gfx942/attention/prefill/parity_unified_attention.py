# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 (CDNA3 / MI300X) unified-attention 2D tiled SDPA-fwd parity + latency.

Torch-reference-only correctness + latency harness for the gfx942 narrow /
wide (flash-regime) tiled attention kernel
(``kernels.gfx942.attention_tiled_2d``). It is the gfx942 sibling of
``examples/gfx950/attention/parity_unified_attention.py``, but with **no**
Triton/AITER dependency: the oracle is a fp32 torch reference, so the example
runs on any box with torch + a gfx942 GPU.

The harness:

  1. Defines shapes inline across three scenario groups (``default`` /
     ``fmha`` / ``creative``); ``--scenario`` selects subsets by group name
     or exact shape name, defaulting to all groups.
  2. For each shape, builds the gfx942 tiled-2D kernel via
     ``UnifiedAttention2DTiledSpec`` / ``build_unified_attention_2d_tiled``
     (auto-routes to ``instances/gfx942/attention_tiled_2d.py``), gated by
     ``supports_tiled_2d``.
  3. Demonstrates the shipped knobs by constructing the spec EXPLICITLY: the
     **wide4** (WG=256, num_warps=4) flash-regime path for D128 fp16, and the
     **L4** (WG=64, K single-buffer) contrast. NOTE: wide4 is the PROVIDER's
     analytic default (``compile_service.py`` ``_flash_wide=4`` +
     ``SdpaCandidateSelector.analyticTarget``), NOT the DSL spec's default --
     a spec built with no flash knobs lands on L4. So this harness sets
     ``num_warps=4`` (+ ``use_mfma_32x32x8`` / ``use_transposed_qk_32x32`` /
     ``use_k_single_buffer=False``) explicitly to reproduce the shipped peak;
     ``HIPDNN_GFX942_FLASH_WIDE=0`` selects the L4 contrast instead.
  4. Launches it on a paged-KV layout derived from the dense SDPA problem and
     compares the output against a fp32 torch reference (causal + GQA).
  5. Reports per shape: correctness PASS/FAIL (tol ~2e-2 fp16 / ~4e-2 bf16),
     latency (us), and achieved TFLOPS.

Run (needs torch + a gfx942 GPU):

    PYTHONPATH=python .venv/bin/python \\
        python/rocke/library/builders/gfx942/attention/parity_unified_attention.py \\
        --scenario default

    # force the L4 (WG=64) fallback instead of the default wide4:
    HIPDNN_GFX942_FLASH_WIDE=0 PYTHONPATH=python .venv/bin/python \\
        python/rocke/library/builders/gfx942/attention/parity_unified_attention.py \\
        --scenario Fp16_Prefill_GQA_S2048_D128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Block size for the paged-KV mapping. The dense SDPA problems are mapped onto
# one contiguous run of paged-KV blocks per sequence; block_size=64 matches the
# gfx942 flash-regime tile (T=64) so the default wide4 / L4 geometries apply.
BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    name: str
    dtype: str  # "fp16" | "bf16"
    seqlen_q: int
    seqlen_k: int
    head_size: int
    heads: int  # Hq
    kv_heads: int  # Hkv
    batch: int  # B
    causal: bool
    group: str  # "default" | "fmha" | "creative"

    @property
    def num_queries_per_kv(self) -> int:
        return self.heads // self.kv_heads


# ---------------------------------------------------------------------------
# In-code scenario functions (adapted from gfx950's parity_unified_attention).
#
# gfx942 constraints vs. gfx950:
#   - causal-only (no sliding window / softcap / ALiBi / QQ-bias / sinks)
#   - BLOCK_SIZE = 64 (tile size T = BLOCK_SIZE)
#   - head_size ∈ {64, 128} for the shipped paths (+ select 256 decode shapes)
#   - Scenarios with non-uniform seq_lens are split into one Shape per unique
#     (seqlen_q, seqlen_k) pair (batch=1) and given a disambiguating suffix.
# ---------------------------------------------------------------------------


def _shape(
    name: str,
    dtype: str,
    seqlen_q: int,
    seqlen_k: int,
    heads: int,
    kv_heads: int,
    head_size: int,
    batch: int,
    group: str,
) -> Shape:
    return Shape(
        name=name,
        dtype=dtype,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_size=head_size,
        heads=heads,
        kv_heads=kv_heads,
        batch=batch,
        causal=True,
        group=group,
    )


def default_scenarios() -> List[Shape]:
    """Core production scenarios adapted from the gfx950 default set.

    Sliding-window / softcap / ALiBi / QQ-bias / sinks scenarios are dropped
    (the harness's torch reference is causal-only, not a kernel limitation).
    are included; they exercise the T=64 single-query path.
    Non-uniform seq_lens from the gfx950 mixed/prefill scenarios are split
    into one Shape per unique (q, k) pair.
    """
    g = "default"
    shapes: List[Shape] = []

    # decode_d128_b16 / b64 equivalents (block_size pinned to 64 on gfx942)
    for sq, sk in [(1, 1024), (1, 2048), (1, 4096), (1, 512)]:
        shapes.append(
            _shape(f"decode_d128_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 128, 1, g)
        )

    # decode_d256 (head_size=256, fp16)
    for sq, sk in [(1, 1024), (1, 2048)]:
        shapes.append(
            _shape(f"decode_d256_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 256, 1, g)
        )

    # bf16 D256 decode — Qwen3-Next-80B-A3B gated_attn GQA 16/2 (d256_decode cohort)
    for sq, sk in [(1, 1024), (1, 2048), (1, 4096), (1, 8192)]:
        shapes.append(
            _shape(f"bf16_decode_d256_q{sq}k{sk}", "bf16", sq, sk, 16, 2, 256, 1, g)
        )

    # prefill_d128 (uniform batch from gfx950 prefill_d128_b16)
    for sq, sk in [(64, 64), (128, 256), (32, 256)]:
        shapes.append(
            _shape(f"prefill_d128_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 128, 1, g)
        )

    # mixed_d128 split into individual shapes
    for sq, sk in [(1, 1328), (5, 18), (129, 463)]:
        shapes.append(
            _shape(f"mixed_d128_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 128, 1, g)
        )

    # bf16 decode
    for sq, sk in [(1, 1024), (1, 2048), (1, 4096)]:
        shapes.append(
            _shape(f"bf16_decode_d128_q{sq}k{sk}", "bf16", sq, sk, 16, 2, 128, 1, g)
        )

    # GQA non-power-of-2 NQK ratios (from gfx950 gqa_nqk* scenarios)
    for sq, sk in [(1024, 1024), (512, 512)]:
        shapes.append(
            _shape(f"gqa_nqk5_d128_q{sq}k{sk}", "bf16", sq, sk, 40, 8, 128, 1, g)
        )
        shapes.append(
            _shape(f"gqa_nqk7_d128_q{sq}k{sk}", "bf16", sq, sk, 28, 4, 128, 1, g)
        )
        shapes.append(
            _shape(f"gqa_nqk3_d128_q{sq}k{sk}", "bf16", sq, sk, 24, 8, 128, 1, g)
        )
        shapes.append(
            _shape(f"gqa_nqk8_d128_q{sq}k{sk}", "bf16", sq, sk, 32, 4, 128, 1, g)
        )

    # combo bf16 d64/b32/h64kv8 prefill (from gfx950 combo_bf16_d64_b32_gqa8_64x8)
    shapes.append(
        _shape("combo_bf16_d64_h64kv8_q512k1024", "bf16", 512, 1024, 64, 8, 64, 2, g)
    )

    return shapes


def fmha_scenarios() -> List[Shape]:
    """FMHA matrix shapes adapted from gfx950 (causal-compatible subset only).

    Drops sliding-window / softcap / ALiBi / QQ-bias variants. Non-uniform
    seq_lens are split per (q, k) pair. Batch counts are kept from the
    gfx950 ``rep(n, q, k)`` helper where the layout is uniform.
    """
    g = "fmha"
    shapes: List[Shape] = []

    # GQA_4to1_Prefill_Basic (32:8 → 16:4 to stay ≤ 16 NQK)
    shapes.append(
        _shape("fmha_gqa_4to1_prefill_2k_b1", "fp16", 2048, 2048, 16, 4, 128, 1, g)
    )
    shapes.append(
        _shape("fmha_gqa_4to1_prefill_2k_b4", "fp16", 2048, 2048, 16, 4, 128, 4, g)
    )
    shapes.append(
        _shape("fmha_gqa_4to1_prefill_2k_bf16", "bf16", 2048, 2048, 16, 4, 128, 2, g)
    )

    # GQA_16to1
    shapes.append(
        _shape("fmha_gqa_16to1_prefill_2k_b1", "fp16", 2048, 2048, 16, 1, 128, 1, g)
    )

    # MQA decode (seqlen_q = 1)
    shapes.append(
        _shape("fmha_mqa_16to1_decode_1k_b8", "fp16", 1, 1024, 16, 1, 128, 8, g)
    )
    shapes.append(
        _shape("fmha_mqa_16to1_decode_4k_b8", "bf16", 1, 4096, 16, 1, 128, 8, g)
    )

    # Tiny sequences (split per pair)
    for sq, sk in [(1, 10), (3, 99), (33, 33), (1, 33), (3, 10)]:
        shapes.append(
            _shape(f"fmha_tiny_seqs_q{sq}k{sk}", "fp16", sq, sk, 16, 1, 128, 1, g)
        )

    # Asymmetric seqlen (split per pair)
    for sq, sk in [(100, 256), (51, 99), (256, 1024)]:
        shapes.append(
            _shape(f"fmha_asym_b1_q{sq}k{sk}", "fp16", sq, sk, 16, 1, 128, 1, g)
        )
    for sq, sk in [(100, 256), (51, 99), (256, 1024)]:
        shapes.append(
            _shape(f"fmha_asym_b2_d64_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 128, 2, g)
        )

    # Cross-attention (sq << sk)
    for sq, sk in [
        (1, 1024),
        (1, 4096),
        (32, 1024),
        (32, 4096),
        (128, 1024),
        (128, 4096),
    ]:
        shapes.append(
            _shape(f"fmha_cross_attn_q{sq}k{sk}", "fp16", sq, sk, 16, 4, 128, 1, g)
        )

    # Paged decode (many sequences, short Q)
    shapes.append(_shape("fmha_paged_decode_b80", "fp16", 1, 4096, 16, 4, 128, 80, g))
    shapes.append(
        _shape("fmha_paged_decode_q4_b16", "fp16", 4, 4096, 16, 4, 128, 16, g)
    )

    # Padding / boundary stress (odd lengths)
    for sq in [259, 500, 987, 1023]:
        shapes.append(
            _shape(f"fmha_pad_boundary_q{sq}", "fp16", sq, sq, 16, 4, 128, 1, g)
        )

    # Prefill odd lengths
    for sq, sk in [(113, 203), (339, 339), (799, 799), (1023, 1024), (3131, 3131)]:
        shapes.append(
            _shape(f"fmha_prefill_odd_q{sq}k{sk}", "fp16", sq, sk, 16, 4, 128, 1, g)
        )

    # H256 high-LDS shapes
    shapes.append(_shape("fmha_h256_2k", "bf16", 2048, 2048, 16, 2, 256, 1, g))
    shapes.append(_shape("fmha_h256_4k_b2", "bf16", 4096, 4096, 8, 4, 256, 2, g))

    # Long sequence stress (8K / 16K)
    shapes.append(_shape("fmha_long_8k", "bf16", 8192, 8192, 16, 4, 128, 1, g))
    shapes.append(_shape("fmha_long_16k", "bf16", 16384, 16384, 16, 4, 128, 1, g))

    # Extreme batch (many short sequences)
    shapes.append(_shape("fmha_batch_64_s128", "fp16", 128, 128, 8, 8, 128, 64, g))
    shapes.append(_shape("fmha_batch_128_s128", "fp16", 128, 128, 8, 8, 128, 128, g))

    # CK benchmark standard
    shapes.append(_shape("fmha_bench_1k_b4", "fp16", 1024, 1024, 16, 16, 128, 4, g))
    shapes.append(_shape("fmha_bench_4k_b2_d256", "fp16", 4096, 4096, 8, 8, 256, 2, g))

    # Vision transformer shapes
    shapes.append(_shape("fmha_vit_1k_b4", "fp16", 1024, 1024, 16, 8, 128, 4, g))
    shapes.append(_shape("fmha_vit_256_b4", "fp16", 256, 256, 16, 8, 128, 4, g))

    return shapes


def creative_scenarios() -> List[Shape]:
    """Exploratory sweep adapted from gfx950 creative scenarios (causal-only).

    Long-context decode, GQA/MQA variants, head_size=256, bf16, chunked
    prefill, and batch sweeps. Sliding-window / softcap / ALiBi / QQ-bias
    shapes are omitted entirely.
    """
    g = "creative"
    shapes: List[Shape] = []

    # Long-context decode
    shapes.append(_shape("creative_decode_8k", "fp16", 1, 8192, 16, 2, 128, 1, g))
    shapes.append(_shape("creative_decode_32k", "fp16", 1, 32768, 16, 2, 128, 1, g))
    shapes.append(
        _shape("creative_decode_64k_bf16", "bf16", 1, 65536, 16, 2, 128, 1, g)
    )

    # Large decode batch (varied kv per seq → one shape each)
    for sk in [512, 1024, 2048, 4096, 768, 1536, 3072, 6144, 384, 896, 1280, 2560]:
        shapes.append(
            _shape(f"creative_decode_b16_k{sk}", "fp16", 1, sk, 16, 2, 128, 1, g)
        )

    # Tiny + large prefill
    shapes.append(_shape("creative_prefill_tiny", "fp16", 4, 4, 16, 2, 128, 1, g))
    shapes.append(_shape("creative_prefill_512", "fp16", 512, 512, 16, 2, 128, 1, g))
    shapes.append(_shape("creative_prefill_2k", "fp16", 2048, 2048, 16, 2, 128, 1, g))

    # Chunked prefill (sq < sk)
    for sq, sk in [(128, 2048), (256, 4096), (512, 8192)]:
        shapes.append(
            _shape(
                f"creative_chunk_prefill_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 128, 1, g
            )
        )

    # GQA / MQA variants
    for sq, sk in [(1, 4096), (1, 8192)]:
        shapes.append(
            _shape(f"creative_gqa_h32k4_q{sq}k{sk}", "fp16", sq, sk, 32, 4, 128, 1, g)
        )
        shapes.append(
            _shape(f"creative_mqa_h16k1_q{sq}k{sk}", "fp16", sq, sk, 16, 1, 128, 1, g)
        )
    for sq, sk in [(1, 2048), (1, 4096)]:
        shapes.append(
            _shape(f"creative_gqa_h64k8_q{sq}k{sk}", "fp16", sq, sk, 64, 8, 128, 1, g)
        )

    # D64 bf16 (from gfx950 creative_r1r4_bf16_d64_sinks -- sinks dropped)
    for sq, sk in [(640, 704), (640, 768)]:
        shapes.append(
            _shape(f"creative_d64_bf16_q{sq}k{sk}", "bf16", sq, sk, 64, 8, 64, 1, g)
        )

    # head_size=256 with bf16
    for sq, sk in [(1, 4096), (1, 8192)]:
        shapes.append(
            _shape(
                f"creative_d256_bf16_decode_q{sq}k{sk}",
                "bf16",
                sq,
                sk,
                16,
                2,
                256,
                1,
                g,
            )
        )
    for sq, sk in [(128, 1024), (256, 2048)]:
        shapes.append(
            _shape(
                f"creative_d256_prefill_q{sq}k{sk}", "fp16", sq, sk, 16, 2, 256, 1, g
            )
        )

    # bf16 chunked prefill (from gfx950 creative_bf16_b64_chunk)
    for sq, sk in [(64, 2048), (128, 4096)]:
        shapes.append(
            _shape(f"creative_bf16_chunk_q{sq}k{sk}", "bf16", sq, sk, 16, 2, 128, 1, g)
        )

    return shapes


def all_inline_scenarios() -> List[Shape]:
    return default_scenarios() + fmha_scenarios() + creative_scenarios()


def select_shapes(selectors: Optional[List[str]]) -> List[Shape]:
    """Resolve ``--scenario`` selectors to a concrete shape list.

    A selector is either a group name (``default`` / ``fmha`` / ``creative`` /
    ``all``) or an exact shape name. Multiple ``--scenario`` flags union.
    No selector → all inline scenarios.
    """
    all_shapes = all_inline_scenarios()
    if not selectors:
        return all_shapes

    wanted: List[Shape] = []
    seen: set = set()

    def _add(s: Shape) -> None:
        if s.name not in seen:
            seen.add(s.name)
            wanted.append(s)

    for sel in selectors:
        if sel in ("all", "inline"):
            for s in all_shapes:
                _add(s)
        elif sel in ("default", "fmha", "creative"):
            for s in all_shapes:
                if s.group == sel:
                    _add(s)
        else:
            for s in all_shapes:
                if s.name == sel:
                    _add(s)
    return wanted


# ---------------------------------------------------------------------------
# Reference paged attention (torch fp32, causal + GQA). Adapted from the
# gfx950 harness's ``ref_paged_attn`` (sliding-window / softcap / bias / sinks
# trimmed -- the gfx942 SDPA-fwd net is causal-only).
# ---------------------------------------------------------------------------


def ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables,
    scale: float,
):
    import torch

    num_seqs = len(query_lens)
    # block_tables on CPU as a plain python list of lists (no numpy: the
    # container torch build ships without numpy).
    block_tables_cpu = block_tables.cpu().tolist()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = int(kv_lens[i])
        q = query[start_idx : start_idx + query_len]
        q = q * scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = torch.tensor(
            block_tables_cpu[i][:num_kv_blocks],
            dtype=torch.long,
            device=key_cache.device,
        )
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        # Chunk the fp32 attn matrix over heads so peak memory stays bounded: the
        # full [h, q, k] fp32 tensor is h*Sq*Sk*4 bytes (e.g. 128*8192*8192*4 =
        # 32 GB), which OOMs the reference on large-head long-context shapes even
        # though the kernel under test is fine. A per-head-chunk loop keeps the
        # peak at chunk*Sq*Sk*4 with identical numerics.
        num_heads = q.shape[1]
        # Build the causal mask directly as bool -- a float [q,k] tensor here
        # would be 4x the bytes, working against this loop's peak-memory bound.
        mask = torch.triu(
            torch.ones(query_len, kv_len, dtype=torch.bool, device=q.device),
            diagonal=kv_len - query_len + 1,
        )
        head_bytes = query_len * kv_len * 4
        chunk = max(1, min(num_heads, int(4 * 1024**3) // max(1, head_bytes)))
        out_chunks = []
        for h0 in range(0, num_heads, chunk):
            h1 = min(num_heads, h0 + chunk)
            attn = torch.einsum("qhd,khd->hqk", q[:, h0:h1], k[:, h0:h1]).float()
            attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1).to(v.dtype)
            out_chunks.append(torch.einsum("hqk,khd->qhd", attn, v[:, h0:h1]))
            del attn
        out = torch.cat(out_chunks, dim=1)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Materialise paged inputs from a dense SDPA shape
# ---------------------------------------------------------------------------


def make_inputs(s: Shape, seed: int = 0):
    """Build the paged-KV kernel inputs for a dense [B, Hq, Sq, D] problem.

    Each batch element becomes one sequence with ``(query_len, kv_len) =
    (seqlen_q, seqlen_k)``; the per-sequence block table is a contiguous run of
    ``BLOCK_SIZE``-token cache blocks. The small uniform range keeps the softmax
    accumulation in a numerically friendly part of the 16-bit float range
    (matching the integration test's +/-0.1 fill).
    """
    import torch

    torch.manual_seed(seed)
    dtype = torch.float16 if s.dtype == "fp16" else torch.bfloat16
    query_lens = [s.seqlen_q] * s.batch
    kv_lens_list = [s.seqlen_k] * s.batch
    num_seqs = s.batch
    scale = s.head_size**-0.5

    max_blocks_per_seq = (s.seqlen_k + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max_blocks_per_seq * num_seqs

    query = torch.empty(
        sum(query_lens), s.heads, s.head_size, dtype=dtype, device="cuda"
    ).uniform_(-0.1, 0.1)
    key_cache = torch.empty(
        num_blocks, BLOCK_SIZE, s.kv_heads, s.head_size, dtype=dtype, device="cuda"
    ).uniform_(-0.1, 0.1)
    value_cache = torch.empty_like(key_cache).uniform_(-0.1, 0.1)

    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    # Contiguous, non-overlapping block tables: sequence i owns blocks
    # [i*max_blocks_per_seq, (i+1)*max_blocks_per_seq).
    block_tables = (
        torch.arange(num_blocks, dtype=torch.int32, device="cuda")
        .view(num_seqs, max_blocks_per_seq)
        .contiguous()
    )
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_q": cu_q,
        "kv_lens": kv_lens,
        "query_lens": query_lens,
        "kv_lens_list": kv_lens_list,
        "block_tables": block_tables,
        "scale": scale,
        "max_query_len": max(query_lens),
        "max_kv_len": max(kv_lens_list),
    }


# ---------------------------------------------------------------------------
# gfx942 kernel build + launch
# ---------------------------------------------------------------------------


def _flash_wide_setting() -> int:
    """Resolve the wide-tile width from ``HIPDNN_GFX942_FLASH_WIDE``.

    Mirrors the provider ``compile_service.py`` semantics: unset -> the shipped
    default (wide4, WG=256) for the qualifying D128 fp16 shape; ``0`` -> the L4
    (WG=64) fallback; ``2``/``4`` -> an explicit width for A/B testing.
    """
    env = os.environ.get("HIPDNN_GFX942_FLASH_WIDE", "").strip().lower()
    if env in ("2", "4"):
        return int(env)
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return 0
    return 4  # shipped default


def _cfv_store_enabled() -> bool:
    """Opt into the experimental gfx942 store-path conflict-free V feed."""
    env = os.environ.get("HIPDNN_GFX942_CFV_STORE", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _cfv_store_split_enabled() -> bool:
    env = os.environ.get("HIPDNN_GFX942_CFV_STORE_SPLIT", "1").strip().lower()
    return env not in ("0", "off", "disable", "disabled", "no", "false")


def _cfv_ck_vlds_enabled() -> bool:
    env = os.environ.get("HIPDNN_GFX942_CFV_CK_VLDS", "1").strip().lower()
    return env not in ("0", "off", "disable", "disabled", "no", "false")


def _cfv_enabled() -> bool:
    """Opt into the legacy gather-fill conflict-free V diagnostic path."""
    env = os.environ.get("HIPDNN_GFX942_CFV", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _k_sliced_ring_enabled() -> bool:
    """Opt into the experimental sliced K ring for the cfvst path."""
    env = os.environ.get("HIPDNN_GFX942_K_SLICED_RING", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _k_sliced_ldsseq_enabled() -> bool:
    """Opt into the CK Tile LdsSeq sliced K variant."""
    env = os.environ.get("HIPDNN_GFX942_K_LDSSEQ", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true", "ck")


def _waves_per_eu_setting():
    env = os.environ.get("HIPDNN_GFX942_WAVES_PER_EU", "").strip()
    return int(env) if env else None


def _num_warps_setting(default: int) -> int:
    env = os.environ.get("HIPDNN_GFX942_NUM_WARPS", "").strip()
    return int(env) if env else default


def _iglp_enabled() -> bool:
    env = os.environ.get("HIPDNN_GFX942_IGLP", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true", "iglp1")


def _kv_cache_policy() -> str:
    return os.environ.get("HIPDNN_GFX942_KV_CACHE_POLICY", "stream").strip().lower()


def _q_direct_enabled() -> bool:
    env = os.environ.get("HIPDNN_GFX942_Q_DIRECT", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _global_load_lds_k_enabled() -> bool:
    env = os.environ.get("HIPDNN_GFX942_GLOBAL_LOAD_LDS_K", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _q_major_grid_enabled() -> bool:
    env = os.environ.get("HIPDNN_GFX942_Q_MAJOR_GRID", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _is_flash_wide_eligible(s: Shape) -> bool:
    """The gfx942 flash-regime (32x32x8) path is D128 fp16 only."""
    return s.head_size == 128 and s.dtype == "fp16"


def _gfx942_spec_class():
    """The gfx942 ``UnifiedAttention2DTiledSpec`` (NOT the default gfx950 one).

    ``kernels.UnifiedAttention2DTiledSpec`` re-exports the gfx950 spec
    (the default arch); the gfx950 class does not declare the gfx942-only flash
    fields (``use_mfma_32x32x8`` / ``use_k_single_buffer`` / ...). The arch
    dispatch seam ``_tiled_2d_impl("gfx942")`` returns the gfx942 spec class,
    matching how the provider builds the gfx942 kernel.
    """
    from kernels.common.attention_unified import _tiled_2d_impl

    spec_cls, _build, _supports = _tiled_2d_impl("gfx942")
    return spec_cls


def _build_spec(s: Shape):
    """Construct the gfx942 ``UnifiedAttention2DTiledSpec`` for shape ``s``.

    Routes D128 fp16 to the shipped flash regime (wide4 by default, or the L4
    WG=64 fallback when ``HIPDNN_GFX942_FLASH_WIDE=0``); every other shape uses
    the narrow 16x16x16 default path (num_warps chosen from block_size, mw=32
    for D64), matching the case-study selector policy.
    """
    UnifiedAttention2DTiledSpec = _gfx942_spec_class()

    base = dict(
        head_size=s.head_size,
        block_size=BLOCK_SIZE,
        num_query_heads=s.heads,
        num_kv_heads=s.kv_heads,
        dtype=s.dtype,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=s.batch,
        tile_size=BLOCK_SIZE,
    )

    wide = _flash_wide_setting()
    wpe = _waves_per_eu_setting()
    iglp = _iglp_enabled()
    kvcp = _kv_cache_policy()
    qdir = _q_direct_enabled()
    gldlds = _global_load_lds_k_enabled()
    qgrid = _q_major_grid_enabled()
    if _is_flash_wide_eligible(s) and wide in (2, 4):
        # Flash-regime wide tile: 32x32x8 transposed-x8 (register-P^T). With
        # BLOCK_M = num_warps*32 > T=64 (wide4), K is double-buffered; for wide2
        # BLOCK_M=64 <= T so K single-buffering applies.
        num_warps = _num_warps_setting(wide)
        config = f"wide{wide}"
        cfvst = _cfv_store_enabled()
        cfv = _cfv_enabled()
        ksring = _k_sliced_ring_enabled()
        ksldsseq = _k_sliced_ldsseq_enabled()
        cfvst_split = _cfv_store_split_enabled()
        cfvst_ck_vlds = _cfv_ck_vlds_enabled()
        spec = UnifiedAttention2DTiledSpec(
            **base,
            num_warps=num_warps,
            waves_per_eu=wpe,
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_k_single_buffer=(num_warps * 32 <= BLOCK_SIZE),
            use_conflict_free_v=cfv,
            use_conflict_free_v_store=cfvst,
            use_conflict_free_v_store_split=cfvst_split,
            use_conflict_free_v_ck_vlds=cfvst_ck_vlds,
            use_k_sliced_ring=ksring,
            use_k_sliced_ldsseq=ksldsseq,
            use_iglp_opt=iglp,
            use_q_direct_global=qdir,
            kv_cache_policy=kvcp,
            use_global_load_lds_k=gldlds,
            use_q_major_grid=qgrid,
        )
        if cfv:
            return spec, f"{config}_cfv"
        if ksldsseq:
            return spec, f"{config}_cfvst_ksldsseq"
        if ksring:
            return spec, f"{config}_cfvst_ksring"
        return spec, f"{config}_cfvst" if cfvst else config

    if _is_flash_wide_eligible(s):
        # HIPDNN_GFX942_FLASH_WIDE=0 -> L4: transposed-x8 + K single-buffer at
        # WG=64 (num_warps=1, BLOCK_M=32 <= T=64).
        cfvst = _cfv_store_enabled()
        cfv = _cfv_enabled()
        cfvst_split = _cfv_store_split_enabled()
        cfvst_ck_vlds = _cfv_ck_vlds_enabled()
        spec = UnifiedAttention2DTiledSpec(
            **base,
            num_warps=1,
            waves_per_eu=wpe,
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_k_single_buffer=True,
            use_conflict_free_v=cfv,
            use_conflict_free_v_store=cfvst,
            use_conflict_free_v_store_split=cfvst_split,
            use_conflict_free_v_ck_vlds=cfvst_ck_vlds,
            use_iglp_opt=iglp,
            use_q_direct_global=qdir,
            kv_cache_policy=kvcp,
            use_global_load_lds_k=gldlds,
            use_q_major_grid=qgrid,
        )
        if cfv:
            return spec, "L4_cfv"
        return spec, "L4_cfvst" if cfvst else "L4"

    # Narrow 16x16x16 default path. D64 ships mw=32 + 1x tile; num_warps keys
    # on block_size (bs>=64 -> nw4). D128 bf16 stays nw2 narrow.
    if s.head_size == 64:
        spec = UnifiedAttention2DTiledSpec(**base, num_warps=4, block_m_per_warp=32)
        return spec, "narrow_d64"
    spec = UnifiedAttention2DTiledSpec(**base, num_warps=2)
    return spec, "narrow"


def _build_kernel(s: Shape):
    """Build (and return) the gfx942 kernel launcher + spec + config tag.

    Raises ``NotImplementedError`` if the shape is not buildable on gfx942.
    """
    from rocke import compile_kernel
    from kernels import build_unified_attention_2d_tiled, supports_tiled_2d
    from kernels.common.attention_unified import _attn_signature
    from rocke.runtime import KernelLauncher

    spec, config = _build_spec(s)

    # ``supports_tiled_2d``'s LDS gate is the generic footprint (P_lds present,
    # K double-buffered, full Acc_lds). That is accurate for the narrow path but
    # OVER-counts the transposed-x8 flash path (P^T is register-resident -> no
    # P_lds; Acc_lds is epilogue-only and backend-aliased into the loop-dead K/V
    # region) -- so it would wrongly reject the shipped wide4 D128 config. The
    # provider uses an accurate ``_lds_bytes_transposed_x8`` model instead; here
    # we mirror that by skipping the generic gate for the flash configs and
    # letting the spec's own ``__post_init__`` + comgr be the arbiters.
    if config in ("narrow", "narrow_d64"):
        ok, reason = supports_tiled_2d(
            head_size=s.head_size,
            block_size=BLOCK_SIZE,
            dtype=s.dtype,
            num_queries_per_kv=s.num_queries_per_kv,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=s.dtype,
            num_warps=spec.num_warps,
            block_m_per_warp=spec.block_m_per_warp,
            tile_size=BLOCK_SIZE,
            arch="gfx942",
        )
        if not ok:
            raise NotImplementedError(reason)

    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    artifact = compile_kernel(kernel, arch="gfx942", capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=_attn_signature(
            s.dtype, include_bt_stride=True, include_qq_bias_stride=True
        ),
        cache_key=("gfx942_attn_example", spec.kernel_name(), config),
    )
    return launcher, spec, config


def _run_rocke(s: Shape, data, launcher, spec, *, warmup: int, attempts: int):
    """Launch the gfx942 kernel and return ``(output, ms)``.

    Builds the kernarg pack via ``_attn_values`` and recomputes the launch grid
    the same way the production dispatcher does
    (``(num_kv_heads, total_num_q_blocks, 1)`` / ``block=(64*num_warps,1,1)``).
    """
    import torch
    from kernels import UnifiedAttentionProblem
    from kernels.common.attention_unified import _attn_values
    from rocke.runtime import (
        LaunchConfig,
        synchronize_and_release,
        time_launches,
    )

    q = data["query"]
    out = torch.empty_like(q)
    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=s.batch,
        num_query_heads=s.heads,
        num_kv_heads=s.kv_heads,
        head_size=s.head_size,
        block_size=BLOCK_SIZE,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype=s.dtype,
        sliding_window=0,
        softcap=0.0,
        use_sinks=False,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        num_cus=120,
    )
    hip_stream = int(torch.cuda.current_stream().cuda_stream)
    vals = _attn_values(
        problem=problem,
        q=q,
        k=data["key_cache"],
        v=data["value_cache"],
        out=out,
        cu_seqlens_q=data["cu_q"],
        seqused_k=data["kv_lens"],
        softmax_scale=data["scale"],
        block_table=data["block_tables"],
        softcap=0.0,
        sinks=None,
        bt_stride=int(data["block_tables"].stride(0)),
        include_bt_stride=True,
        alibi_slopes=None,
        qq_bias=None,
        qq_bias_stride_0=0,
        include_qq_bias_stride=True,
    )
    block_q = spec.block_q
    total_num_q_blocks = q.shape[0] // block_q + s.batch
    grid = (
        (int(total_num_q_blocks), int(s.kv_heads), 1)
        if getattr(spec, "use_q_major_grid", False)
        else (int(s.kv_heads), int(total_num_q_blocks), 1)
    )
    cfg = LaunchConfig(
        grid=grid,
        block=(64 * spec.num_warps, 1, 1),
        stream=hip_stream,
    )

    def call_once():
        launcher(vals, config=cfg)

    ms = time_launches(call_once, warmup=warmup, iters=attempts, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return out, ms


# ---------------------------------------------------------------------------
# Compare + report
# ---------------------------------------------------------------------------


def compare(reference, out) -> dict:
    a = reference.float()
    b = out.float()
    abs_diff = (a - b).abs()
    return {
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
    }


def mismatch_summary(reference, out, tol: float, limit: int) -> dict:
    """Return compact diagnostics for a failing correctness comparison."""
    import torch

    a = reference.float()
    b = out.float()
    abs_diff = (a - b).abs()
    bad = abs_diff > tol
    sign_mismatch = (torch.signbit(a) != torch.signbit(b)) & bad
    n_bad = int(bad.sum().item())
    n_sign = int(sign_mismatch.sum().item())
    worst_vals, worst_flat = torch.topk(
        abs_diff.flatten(), k=min(limit, abs_diff.numel())
    )
    samples = []
    for rank, flat in enumerate(worst_flat.cpu().tolist()):
        idx = list(torch.unravel_index(torch.tensor(flat), abs_diff.shape))
        idx_int = [int(x.item()) for x in idx]
        samples.append(
            {
                "rank": rank,
                "index": idx_int,
                "ref": float(a.flatten()[flat].item()),
                "out": float(b.flatten()[flat].item()),
                "abs": float(worst_vals[rank].item()),
            }
        )
    return {
        "bad": n_bad,
        "total": int(abs_diff.numel()),
        "sign_mismatch": n_sign,
        "samples": samples,
    }


def attention_tflops(s: Shape, ms: float) -> float:
    """FMHA-fwd TFLOPS: two GEMMs (QK^T and PV), each 2*B*Hq*Sq*Sk*D."""
    flops = 4.0 * s.batch * s.heads * s.seqlen_q * s.seqlen_k * s.head_size
    seconds = ms / 1e3
    return flops / seconds / 1e12 if seconds > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="gfx942 unified-attention parity + latency harness"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="group (default|fmha|creative|all) or exact shape name; repeatable",
    )
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="abs tolerance override (default 2e-2 fp16 / 4e-2 bf16)",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--debug-mismatch",
        type=int,
        default=0,
        help="on failure, print this many worst mismatch samples",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA/HIP device unavailable; exiting", file=sys.stderr)
        return 1
    dev_name = torch.cuda.get_device_name(0)
    print("device:", dev_name)
    wide = _flash_wide_setting()
    print(
        f"HIPDNN_GFX942_FLASH_WIDE -> {wide} "
        f"({'wide' + str(wide) if wide else 'L4 (WG=64)'} for D128 fp16)"
    )
    if _cfv_store_enabled():
        print("HIPDNN_GFX942_CFV_STORE -> enabled (experimental cfvst)")
    if _cfv_enabled():
        print("HIPDNN_GFX942_CFV -> enabled (legacy cfv diagnostic)")
    if _k_sliced_ring_enabled():
        print("HIPDNN_GFX942_K_SLICED_RING -> enabled (experimental ksring)")
    if _k_sliced_ldsseq_enabled():
        print("HIPDNN_GFX942_K_LDSSEQ -> enabled (CK Tile LdsSeq)")
    if _waves_per_eu_setting() is not None:
        print(f"HIPDNN_GFX942_WAVES_PER_EU -> {_waves_per_eu_setting()}")
    if os.environ.get("HIPDNN_GFX942_NUM_WARPS", "").strip():
        print(f"HIPDNN_GFX942_NUM_WARPS -> {os.environ['HIPDNN_GFX942_NUM_WARPS']}")
    if _iglp_enabled():
        print("HIPDNN_GFX942_IGLP -> enabled (iglp1)")
    if _kv_cache_policy() != "stream":
        print(f"HIPDNN_GFX942_KV_CACHE_POLICY -> {_kv_cache_policy()}")
    if _q_direct_enabled():
        print("HIPDNN_GFX942_Q_DIRECT -> enabled")
    if _global_load_lds_k_enabled():
        print("HIPDNN_GFX942_GLOBAL_LOAD_LDS_K -> enabled")
    if _q_major_grid_enabled():
        print("HIPDNN_GFX942_Q_MAJOR_GRID -> enabled")

    shapes = select_shapes(args.scenario)
    if not shapes:
        print(f"no shapes matched {args.scenario!r}", file=sys.stderr)
        return 2

    results = []
    n_fail = 0
    for s in shapes:
        tol = (
            args.tol if args.tol is not None else (2e-2 if s.dtype == "fp16" else 4e-2)
        )
        print(
            f"\n=== {s.name}  dtype={s.dtype} D={s.head_size} "
            f"Hq{s.heads}/Hkv{s.kv_heads} S{s.seqlen_q}x{s.seqlen_k} B{s.batch} ==="
        )
        torch.cuda.synchronize()
        try:
            from rocke.runtime import synchronize_and_release

            synchronize_and_release()
        except Exception:
            pass
        torch.cuda.empty_cache()

        try:
            launcher, spec, config = _build_kernel(s)
        except NotImplementedError as e:
            print(f"  SKIP (unsupported on gfx942): {e}")
            results.append({"name": s.name, "status": "skip", "reason": str(e)})
            continue

        data = make_inputs(s)
        with torch.inference_mode():
            ref = ref_paged_attn(
                query=data["query"],
                key_cache=data["key_cache"],
                value_cache=data["value_cache"],
                query_lens=data["query_lens"],
                kv_lens=data["kv_lens_list"],
                block_tables=data["block_tables"],
                scale=data["scale"],
            ).float()
            out, ms = _run_rocke(
                s, data, launcher, spec, warmup=args.warmup, attempts=args.attempts
            )
            torch.cuda.synchronize()
            diffs = compare(ref, out)

        ok = diffs["max_abs"] <= tol
        tag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
            if args.debug_mismatch > 0:
                dbg = mismatch_summary(ref, out, tol, args.debug_mismatch)
                print(
                    f"  mismatch: bad={dbg['bad']}/{dbg['total']} "
                    f"sign_mismatch={dbg['sign_mismatch']}"
                )
                for sample in dbg["samples"]:
                    print(
                        "    "
                        f"rank={sample['rank']} idx={sample['index']} "
                        f"ref={sample['ref']:+.6e} out={sample['out']:+.6e} "
                        f"abs={sample['abs']:.3e}"
                    )
        us = ms * 1e3
        tf = attention_tflops(s, ms)
        print(
            f"  config={config:11s} kernel={spec.kernel_name()}\n"
            f"  {tag}  max_abs={diffs['max_abs']:.3e} (tol {tol:.0e})  "
            f"{us:9.2f} us  {tf:7.1f} TFLOPS"
        )
        results.append(
            {
                "name": s.name,
                "dtype": s.dtype,
                "config": config,
                "status": tag.lower(),
                "max_abs": diffs["max_abs"],
                "median_us": us,
                "tflops": tf,
                "kernel": spec.kernel_name(),
            }
        )

    n_pass = sum(1 for r in results if r.get("status") == "pass")
    n_skip = sum(1 for r in results if r.get("status") == "skip")
    print(f"\nsummary: {n_pass} PASS / {n_skip} SKIP / {n_fail} FAIL on {dev_name}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2))
        print(f"wrote report -> {args.report}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
