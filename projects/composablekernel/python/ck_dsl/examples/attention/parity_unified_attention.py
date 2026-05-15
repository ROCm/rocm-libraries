# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity + benchmark harness comparing Triton and CK DSL unified attention.

Run with the venv's python interpreter (it has torch + triton + aiter installed):

    sudo -n /workspace/dsl_bake_off/venv/bin/python3 \\
        python/ck_dsl/examples/attention/parity_unified_attention.py [--scenario name]

The harness:
  1. Builds the standard AITER unified-attention test inputs (paged KV, GQA).
  2. Runs the Triton backend.
  3. Runs the CK DSL backend (if `supports_native_unified_attention` returns true).
  4. Compares both to the reference `ref_paged_attn` implementation.
  5. Reports latency (ms) for each backend across `attempts` repetitions.

The reference path matches AITER's own test
(`op_tests/triton_tests/attention/test_unified_attention.py`) byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[5]  # composablekernel/
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, "/workspace/aiter")

import aiter.ops.triton.attention.unified_attention as _uam  # noqa: E402
from aiter.ops.triton.attention.unified_attention import unified_attention  # noqa: E402


# ---------------------------------------------------------------------------
# Path forcing
#
# AITER's `unified_attention` internally picks between its 2D and 3D Triton
# kernels via `use_2d_kernel(...)`. The CK DSL backend has its own dispatch
# (`run_unified_attention_torch`) that prefers 3D split-KV whenever
# supported. For an honest apples-to-apples comparison we want both
# backends to run the same algorithmic path. These helpers force either
# backend to a specific (2D or 3D) kernel.
# ---------------------------------------------------------------------------


_ORIG_USE_2D = _uam.use_2d_kernel
_LAST_TRITON_PATH = {"path": None}


def _force_triton_path(path: str):
    """Force AITER's `unified_attention` to pick a specific Triton kernel.

    `path` must be one of `"2d"`, `"3d"`, or `"auto"`. `"auto"` restores the
    default heuristic but also records which kernel the heuristic actually
    picked in `_LAST_TRITON_PATH["path"]`.
    """
    if path == "2d":
        _uam.use_2d_kernel = lambda *a, **kw: _record_path("2d", True)
    elif path == "3d":
        _uam.use_2d_kernel = lambda *a, **kw: _record_path("3d", False)
    elif path == "auto":

        def _sniffer(*args, **kwargs):
            v = _ORIG_USE_2D(*args, **kwargs)
            return _record_path("2d" if v else "3d", v)

        _uam.use_2d_kernel = _sniffer
    else:
        raise ValueError(f"unknown force-path {path!r}")


def _record_path(name: str, value):
    _LAST_TRITON_PATH["path"] = name
    return value


def _ck_backend(path: str) -> str:
    """Translate a force-path string into a CK DSL `run_unified_attention_torch` backend."""
    if path == "2d":
        return "tiled"
    if path == "3d":
        return "3d"
    if path == "auto":
        return "auto"
    raise ValueError(f"unknown force-path {path!r}")


# ---------------------------------------------------------------------------
# Reference paged attention (verbatim from AITER op_tests)
# ---------------------------------------------------------------------------


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    qq_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference paged attention. Matches AITER's reference plus ALiBi/QQ-bias.

    ALiBi: `S += alibi_slope[h] * (key_pos - context_len)`.
    QQ-bias: `S += qq_bias[q_local, k_local - context_len]` over key positions
    inside the query section (else 0).
    Both biases are added *after* the standard scaling and softcap, before the
    softmax row reduction (matches the AITER Triton kernel exactly).
    """
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = int(kv_lens[i])
        context_len = kv_len - query_len
        q = query[start_idx : start_idx + query_len]
        q = q * scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if alibi_slopes is not None:
            # alibi_slopes shape: [num_query_heads]
            num_q_heads = q.shape[1]
            pos = (
                torch.arange(kv_len, device=q.device, dtype=torch.float32) - context_len
            )
            slopes = alibi_slopes.float().view(num_q_heads, 1, 1)
            attn = attn + slopes * pos.view(1, 1, kv_len)
        if qq_bias is not None:
            # qq_bias shape: [N0, N1]; first axis indexed by local query pos
            # within this sequence; second axis indexed by local key pos
            # within the query section (key_rel_pos = k_idx - context_len).
            qq_b = torch.zeros(query_len, kv_len, dtype=torch.float32, device=q.device)
            qq_size_0 = qq_bias.shape[0]
            qq_size_1 = qq_bias.shape[1]
            for qp in range(query_len):
                if qp >= qq_size_0:
                    continue
                for kp in range(kv_len):
                    krp = kp - context_len
                    if 0 <= krp < qq_size_1:
                        qq_b[qp, kp] = qq_bias[qp, krp].float()
            attn = attn + qq_b.view(1, query_len, kv_len)
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sw_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sw_mask
        attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    seq_lens: List[Tuple[int, int]]
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype: torch.dtype
    sliding_window: Optional[int] = None
    softcap: float = 0.0
    use_sinks: bool = False
    num_blocks: int = 32768
    use_alibi: bool = False
    use_qq_bias: bool = False
    qq_bias_stride_0: int = 0


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="decode_d128_b16",
            seq_lens=[(1, 1024), (1, 2048), (1, 4096), (1, 512)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="decode_d128_b64",
            seq_lens=[(1, 1024), (1, 2048), (1, 4096), (1, 512)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            dtype=torch.float16,
        ),
        Scenario(
            name="decode_d256_b16",
            seq_lens=[(1, 1024), (1, 2048)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="prefill_d128_b16",
            seq_lens=[(64, 64), (128, 256), (32, 256)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="mixed_d128_b16",
            seq_lens=[(1, 1328), (5, 18), (129, 463)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="sliding_d128_b16",
            seq_lens=[(1, 2048), (1, 4096), (1, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            sliding_window=256,
        ),
        Scenario(
            name="softcap_d128_b16",
            seq_lens=[(1, 1024), (1, 2048)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            softcap=50.0,
        ),
        Scenario(
            name="bf16_decode_d128_b64",
            seq_lens=[(1, 1024), (1, 2048), (1, 4096)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            dtype=torch.bfloat16,
        ),
        Scenario(
            name="alibi_decode_d128_b16",
            seq_lens=[(1, 1024), (1, 2048), (1, 4096)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            use_alibi=True,
        ),
        Scenario(
            name="alibi_mixed_d128_b16",
            seq_lens=[(1, 1328), (5, 18), (129, 463)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            use_alibi=True,
        ),
        Scenario(
            name="qq_bias_prefill_d128_b16",
            seq_lens=[(64, 64), (128, 256), (32, 256)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            use_qq_bias=True,
            qq_bias_stride_0=256,
        ),
    ]


def fmha_scenarios() -> List[Scenario]:
    """Shapes adapted from `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml`.

    The CK FMHA matrix is a dense Cartesian product over `(batch,
    seqlen_q, seqlen_k, nhead_q, nhead_k, hdim_q, hdim_v, dtype, mask,
    bias, dropout, lse)`. Our paged-attention API is more
    constrained: causal-only, `head_size in {128, 256}`,
    `block_size in {16, 64}`, `hdim_q == hdim_v`, no dropout/FP8,
    and `num_queries_per_kv <= 16` so the 2D tiled kernel can map
    `(qpos, qhead)` onto a 16-row MFMA `BLOCK_M`.

    The scenarios below cover every YAML *group* whose shape space
    fits those constraints, translated to `(batch * seqlen_q,
    seqlen_k)` per-sequence pairs.
    """
    fp16 = torch.float16
    bf16 = torch.bfloat16

    def rep(n: int, q: int, k: int) -> List[Tuple[int, int]]:
        return [(q, k)] * int(n)

    return [
        # --- GQA_4to1_Prefill_Basic (32:8, 2K, fp16/bf16) ---
        Scenario(
            name="fmha_gqa_4to1_prefill_2k_b1",
            seq_lens=rep(1, 2048, 2048),
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_gqa_4to1_prefill_2k_b4",
            seq_lens=rep(4, 2048, 2048),
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_gqa_4to1_prefill_2k_bf16",
            seq_lens=rep(2, 2048, 2048),
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=64,
            dtype=bf16,
        ),
        # --- GQA_16to1_Large (capped to 16:1 inside our 16-head limit) ---
        Scenario(
            name="fmha_gqa_16to1_prefill_2k_b1",
            seq_lens=rep(1, 2048, 2048),
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        # --- MQA_128to8_Decode adapted: cap 16:1 MQA ratio ---
        Scenario(
            name="fmha_mqa_16to1_decode_1k_b8",
            seq_lens=rep(8, 1, 1024),
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_mqa_16to1_decode_4k_b8",
            seq_lens=rep(8, 1, 4096),
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            dtype=bf16,
        ),
        # --- CK_Tiny_Sequences: very short sq/sk ---
        Scenario(
            name="fmha_tiny_seqs",
            seq_lens=[(1, 10), (3, 99), (33, 33), (1, 33), (3, 10)],
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        # --- CK_Asymmetric_Seqlen (varlen) ---
        # Paged attention requires `seqlen_q <= seqlen_k` per sequence so
        # `context_len = seqlen_k - seqlen_q >= 0`. The original YAML row
        # also has `q > k` combinations that NaN ref_paged_attn; we drop
        # those and keep only valid varlen mixes.
        Scenario(
            name="fmha_asym_seqlen_b1",
            seq_lens=[(100, 256), (51, 99), (256, 1024)],
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_asym_seqlen_b2_d64variant",
            seq_lens=[(100, 256), (51, 99), (256, 1024)] * 2,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            dtype=fp16,
        ),
        # --- Cross_Attention_Shapes (sq << sk) ---
        Scenario(
            name="fmha_cross_attn",
            seq_lens=[
                (1, 1024),
                (1, 4096),
                (32, 1024),
                (32, 4096),
                (128, 1024),
                (128, 4096),
            ],
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        # --- Paged_Decode_Shapes (multi-batch short Q, mid-long KV) ---
        Scenario(
            name="fmha_paged_decode_b80",
            seq_lens=rep(80, 1, 4096),
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_paged_decode_q4_b16",
            seq_lens=rep(16, 4, 4096),
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=64,
            dtype=fp16,
        ),
        # --- Padding_Boundary_Stress_Odd_Lengths ---
        Scenario(
            name="fmha_pad_boundary_odd",
            seq_lens=[(259, 259), (500, 500), (987, 987), (1023, 1023)],
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        # --- Prefill_Odd_Lengths ---
        Scenario(
            name="fmha_prefill_odd",
            seq_lens=[(113, 203), (339, 339), (799, 799), (1023, 1024), (3131, 3131)],
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        # --- MHA_H256_High_LDS_Pressure (with hdim=256) ---
        Scenario(
            name="fmha_h256_high_lds_2k",
            seq_lens=rep(1, 2048, 2048),
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            dtype=bf16,
        ),
        Scenario(
            name="fmha_h256_high_lds_4k_b2",
            seq_lens=rep(2, 4096, 4096),
            num_query_heads=8,
            num_kv_heads=4,
            head_size=256,
            block_size=16,
            dtype=bf16,
        ),
        # --- Long_Sequence_Stress (8K / 16K seqs, GQA 16:4) ---
        Scenario(
            name="fmha_long_8k",
            seq_lens=[(8192, 8192)],
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=bf16,
        ),
        Scenario(
            name="fmha_long_16k",
            seq_lens=[(16384, 16384)],
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=bf16,
        ),
        # --- Extreme_Batch_Size_Stress (many short sequences) ---
        Scenario(
            name="fmha_batch_64_s128",
            seq_lens=rep(64, 128, 128),
            num_query_heads=8,
            num_kv_heads=8,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_batch_128_s128",
            seq_lens=rep(128, 128, 128),
            num_query_heads=8,
            num_kv_heads=8,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        # --- CK_Benchmark_Standard subset (mid-context, mid-batch) ---
        Scenario(
            name="fmha_bench_1k_b4",
            seq_lens=rep(4, 1024, 1024),
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_bench_4k_b2_d256",
            seq_lens=rep(2, 4096, 4096),
            num_query_heads=8,
            num_kv_heads=8,
            head_size=256,
            block_size=16,
            dtype=fp16,
        ),
        # --- Bias_Variants_Sweep (ALiBi at modest lengths) ---
        Scenario(
            name="fmha_alibi_512_b4",
            seq_lens=rep(4, 512, 512),
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            dtype=fp16,
            use_alibi=True,
        ),
        Scenario(
            name="fmha_alibi_1k_b4",
            seq_lens=rep(4, 1024, 1024),
            num_query_heads=16,
            num_kv_heads=16,
            head_size=128,
            block_size=16,
            dtype=fp16,
            use_alibi=True,
        ),
        # --- Vision_Transformer_Shapes (h=128 subset, GQA capped) ---
        Scenario(
            name="fmha_vit_1k_b4",
            seq_lens=rep(4, 1024, 1024),
            num_query_heads=16,
            num_kv_heads=8,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
        Scenario(
            name="fmha_vit_256_b4",
            seq_lens=rep(4, 256, 256),
            num_query_heads=16,
            num_kv_heads=8,
            head_size=128,
            block_size=16,
            dtype=fp16,
        ),
    ]


def creative_scenarios() -> List[Scenario]:
    """Exploratory sweep: corners we don't hit in the default 11 scenarios.

    Covers long-context decode (up to 64K), tiny prefill, large prefill,
    chunked prefill, varied GQA factors (MQA, num_kv_heads in {1, 4, 8}),
    head_size=256 with bf16, block_size=64 + bf16, sliding-window
    extremes, softcap+sw, ALiBi+sw, QQ-bias on chunked prefill, and the
    full bias kitchen-sink (sliding + softcap + ALiBi).
    """
    return [
        # --- long-context decode (stress the 3D split-KV pipeline) ---
        Scenario(
            name="creative_decode_8k",
            seq_lens=[(1, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="creative_decode_32k",
            seq_lens=[(1, 32768)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="creative_decode_64k_bf16_b64",
            seq_lens=[(1, 65536)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            dtype=torch.bfloat16,
        ),
        # --- big decode batch (varied kv per seq) ---
        Scenario(
            name="creative_decode_batch16",
            seq_lens=[
                (1, 512),
                (1, 1024),
                (1, 2048),
                (1, 4096),
                (1, 768),
                (1, 1536),
                (1, 3072),
                (1, 6144),
                (1, 384),
                (1, 896),
                (1, 1280),
                (1, 2560),
                (1, 200),
                (1, 333),
                (1, 444),
                (1, 999),
            ],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        # --- tiny + large prefill ---
        Scenario(
            name="creative_prefill_tiny",
            seq_lens=[(4, 4)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="creative_prefill_512",
            seq_lens=[(512, 512)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="creative_prefill_2k",
            seq_lens=[(2048, 2048)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        # --- chunked prefill (q < kv, mid-context) ---
        Scenario(
            name="creative_chunk_prefill",
            seq_lens=[(128, 2048), (256, 4096), (512, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        # --- GQA / MQA / wider heads ---
        Scenario(
            name="creative_gqa_h32_k4",
            seq_lens=[(1, 4096), (1, 8192)],
            num_query_heads=32,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="creative_mqa_h16_k1",
            seq_lens=[(1, 4096), (1, 8192)],
            num_query_heads=16,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        Scenario(
            name="creative_gqa_h64_k8",
            seq_lens=[(1, 2048), (1, 4096)],
            num_query_heads=64,
            num_kv_heads=8,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
        ),
        # --- head_size=256 with bf16 ---
        Scenario(
            name="creative_d256_bf16_decode",
            seq_lens=[(1, 4096), (1, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            dtype=torch.bfloat16,
        ),
        Scenario(
            name="creative_d256_prefill",
            seq_lens=[(128, 1024), (256, 2048)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=256,
            block_size=16,
            dtype=torch.float16,
        ),
        # --- bf16 + block_size=64 chunked prefill ---
        Scenario(
            name="creative_bf16_b64_chunk",
            seq_lens=[(64, 2048), (128, 4096)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            dtype=torch.bfloat16,
        ),
        # --- sliding-window extremes ---
        Scenario(
            name="creative_sw_short",
            seq_lens=[(1, 4096), (1, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            sliding_window=64,
        ),
        Scenario(
            name="creative_sw_large",
            seq_lens=[(1, 16384), (1, 32768)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            sliding_window=4096,
        ),
        # --- ALiBi + sliding window ---
        Scenario(
            name="creative_alibi_sw",
            seq_lens=[(1, 4096), (1, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            sliding_window=512,
            use_alibi=True,
        ),
        # --- softcap + sliding ---
        Scenario(
            name="creative_softcap_sw",
            seq_lens=[(1, 4096), (1, 8192)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            sliding_window=512,
            softcap=30.0,
        ),
        # --- QQ-bias on chunked prefill ---
        Scenario(
            name="creative_qq_chunk",
            seq_lens=[(128, 512), (256, 1024)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            use_qq_bias=True,
            qq_bias_stride_0=512,
        ),
        # --- kitchen-sink: causal + sliding + softcap + ALiBi together ---
        Scenario(
            name="creative_kitchen_sink",
            seq_lens=[(64, 2048), (32, 1024)],
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            sliding_window=256,
            softcap=30.0,
            use_alibi=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Materialise inputs
# ---------------------------------------------------------------------------


def make_inputs(s: Scenario, seed: int = 0):
    torch.manual_seed(seed)
    query_lens = [x[0] for x in s.seq_lens]
    kv_lens_list = [x[1] for x in s.seq_lens]
    num_seqs = len(s.seq_lens)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    scale = s.head_size**-0.5
    query = torch.randn(
        sum(query_lens), s.num_query_heads, s.head_size, dtype=s.dtype, device="cuda"
    )
    key_cache = torch.randn(
        s.num_blocks,
        s.block_size,
        s.num_kv_heads,
        s.head_size,
        dtype=s.dtype,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    max_blocks = (max_kv_len + s.block_size - 1) // s.block_size
    block_tables = torch.randint(
        0, s.num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device="cuda"
    )
    sinks = (
        torch.randn(s.num_query_heads, dtype=torch.bfloat16, device="cuda")
        if s.use_sinks
        else None
    )
    alibi_slopes = None
    if s.use_alibi:
        # Match Triton's typical ALiBi slope distribution (small negative
        # exponents). Using moderate magnitudes keeps S finite under fp16.
        alibi_slopes = -torch.linspace(
            0.05, 0.5, s.num_query_heads, dtype=torch.float32, device="cuda"
        )
    qq_bias = None
    if s.use_qq_bias:
        # qq_bias shape: [N0, N1]; first dim indexed by local query pos within
        # this sequence, second by local key pos within the query section.
        # N1 = qq_bias_stride_0 by definition (contiguous last dim).
        n0 = max(max_query_len, s.qq_bias_stride_0)
        n1 = s.qq_bias_stride_0
        qq_bias = torch.randn(n0, n1, dtype=torch.float32, device="cuda") * 0.1
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
        "sinks": sinks,
        "alibi_slopes": alibi_slopes,
        "qq_bias": qq_bias,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _time_call_loop(call_once, warmup: int, attempts: int):
    """Standard event-timer: ``warmup`` untimed + ``attempts`` timed launches.

    Returns mean per-call ms. The caller owns the output tensor and the
    streaming context. After timing we always do a full
    ``hipDeviceSynchronize`` (via :func:`synchronize_and_release`) before
    returning: on systems where torch's current stream is the legacy null
    stream (`cuda_stream == 0`) and our raw ctypes kernels also target
    stream 0, `torch.cuda.Event.synchronize()` does not always observe
    the ctypes-issued work. Forcing a device sync at the lane boundary
    is the only reliable way to guarantee the output tensor reflects all
    of the launched kernels' writes before the caller reads them.
    """
    for _ in range(warmup):
        call_once()
    torch.cuda.synchronize()
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_start.record()
    for _ in range(attempts):
        call_once()
    t_end.record()
    t_end.synchronize()
    try:
        from ck_dsl.runtime import synchronize_and_release

        synchronize_and_release()
    except Exception:
        torch.cuda.synchronize()
    return t_start.elapsed_time(t_end) / attempts


def _run_triton(s: Scenario, data, *, path: str, warmup: int, attempts: int):
    """Run AITER's Triton `unified_attention` with the requested path forced."""
    output = torch.empty_like(data["query"])
    window_size = (s.sliding_window - 1, 0) if s.sliding_window else (-1, -1)
    _force_triton_path(path)
    try:

        def call_once():
            unified_attention(
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=output,
                cu_seqlens_q=data["cu_q"],
                seqused_k=data["kv_lens"],
                max_seqlen_q=data["max_query_len"],
                max_seqlen_k=data["max_kv_len"],
                softmax_scale=data["scale"],
                causal=True,
                window_size=window_size,
                block_table=data["block_tables"],
                softcap=s.softcap,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                alibi_slopes=data["alibi_slopes"],
                qq_bias=data["qq_bias"],
                sinks=data["sinks"],
                backend="triton",
            )

        ms = _time_call_loop(call_once, warmup, attempts)
        return output, ms
    finally:
        _force_triton_path("auto")


def _run_ck_dsl(s: Scenario, data, *, path: str, warmup: int, attempts: int):
    """Run CK DSL `run_unified_attention_torch` with the requested path forced.

    Uses the DSL's direct entry point (bypassing AITER's wrapper) so that
    the path argument is honored exactly. The CK launches are placed on
    torch's current stream (via `resolve_stream` inside the runtime) so
    the torch caching allocator sees them and cannot recycle workspace
    memory while a kernel is still pending -- the same invariant Triton
    and AITER's paged-attention shim rely on.
    """
    import torch
    from ck_dsl.instances import (
        UnifiedAttentionProblem,
        run_unified_attention_torch,
    )

    output = torch.empty_like(data["query"])
    q = data["query"]
    qq_bias = data["qq_bias"]
    qq_bias_stride_0 = int(qq_bias.stride(0)) if qq_bias is not None else 0
    dtype_str = (
        "fp16"
        if q.dtype is torch.float16
        else "bf16"
        if q.dtype is torch.bfloat16
        else str(q.dtype)
    )
    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(s.seq_lens),
        num_query_heads=s.num_query_heads,
        num_kv_heads=s.num_kv_heads,
        head_size=s.head_size,
        block_size=s.block_size,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype=dtype_str,
        sliding_window=s.sliding_window or 0,
        softcap=float(s.softcap),
        use_sinks=data["sinks"] is not None,
        use_alibi=data["alibi_slopes"] is not None,
        use_qq_bias=qq_bias is not None,
        use_fp8=False,
        num_sms=120,
    )

    # Launch on torch's current stream so the torch caching allocator
    # sees the kernel launches. Default HIP stream 0 is decoupled from
    # torch's streams, which previously produced sporadic NaNs and
    # illegal accesses after a 2D launch was followed by a 3D launch
    # because torch would recycle workspace memory while a kernel was
    # still pending on stream 0.
    torch_stream = int(torch.cuda.current_stream().cuda_stream)

    def call_once():
        run_unified_attention_torch(
            problem=problem,
            q=q,
            k=data["key_cache"],
            v=data["value_cache"],
            out=output,
            cu_seqlens_q=data["cu_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(s.softcap),
            sinks=data["sinks"],
            alibi_slopes=data["alibi_slopes"],
            qq_bias=qq_bias,
            qq_bias_stride_0=qq_bias_stride_0,
            backend=_ck_backend(path),
            stream=torch_stream,
        )

    ms = _time_call_loop(call_once, warmup, attempts)
    return output, ms


def run_unified(backend: str, s: Scenario, data, warmup: int = 3, attempts: int = 10):
    """Backwards-compatible wrapper: backend in {triton, ck_dsl}; path auto."""
    if backend == "triton":
        return _run_triton(s, data, path="auto", warmup=warmup, attempts=attempts)
    if backend == "ck_dsl":
        return _run_ck_dsl(s, data, path="auto", warmup=warmup, attempts=attempts)
    raise ValueError(f"unknown backend {backend!r}")


def run_reference(s: Scenario, data) -> torch.Tensor:
    return ref_paged_attn(
        query=data["query"],
        key_cache=data["key_cache"],
        value_cache=data["value_cache"],
        query_lens=data["query_lens"],
        kv_lens=data["kv_lens_list"],
        block_tables=data["block_tables"],
        scale=data["scale"],
        sliding_window=s.sliding_window,
        soft_cap=s.softcap if s.softcap > 0 else None,
        sinks=data["sinks"],
        alibi_slopes=data["alibi_slopes"],
        qq_bias=data["qq_bias"],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compare(reference: torch.Tensor, out: torch.Tensor) -> dict:
    # Promote both sides to fp32 explicitly. This (1) avoids fp16 overflow
    # in the subtraction when the two tensors disagree by more than the
    # fp16 range (rare but possible under aliasing bugs) and (2) lets the
    # harness keep the reference as fp32 throughout to immunize it from
    # any in-flight aliasing.
    a = reference.float()
    b = out.float()
    diff = a - b
    abs_diff = diff.abs()
    abs_ref = a.abs()
    rel = abs_diff / (abs_ref + 1e-6)
    return {
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def _isolate_benchmark_lane() -> None:
    """Strongly isolate independent benchmark lanes.

    The parity harness intentionally runs several independent kernels
    (Triton auto, CK auto, Triton 2D, CK 2D, Triton 3D, CK 3D) against
    the same inputs and reference. The comparisons allocate temporary
    tensors between lanes, while CK DSL launches are raw HIP ctypes
    calls with their own retained-args/tensor lifetime queue. Synchronize
    and release that queue before the next lane so report-generation
    allocations cannot perturb a later lane. This runs *after* event
    timing, so it does not affect the reported kernel latency.
    """
    torch.cuda.synchronize()
    try:
        from ck_dsl.runtime import synchronize_and_release

        synchronize_and_release()
    except Exception:
        pass
    torch.cuda.empty_cache()


def _safe_run(fn, *, skip_err: bool = True):
    """Run a `(out, ms)`-returning callable, capturing NotImplementedError."""
    try:
        return fn(), None
    except NotImplementedError as e:
        return None, ("skip", str(e))
    except Exception as e:  # pragma: no cover - capture and continue
        if not skip_err:
            raise
        return None, ("error", repr(e))


def _row_print(label: str, out_ms, ref_out, t_out):
    if out_ms is None:
        return
    out, ms = out_ms
    # Hard sync via the runtime so that raw HIP launches issued via ctypes
    # have actually completed. torch.cuda.synchronize() alone is not always
    # enough when torch is on the legacy null stream (handle 0) AND we
    # launched via hipModuleLaunchKernel on the same null stream; some
    # ROCm setups treat the two as distinct queues. Using
    # `Runtime.sync()` directly calls `hipDeviceSynchronize` which
    # always drains every stream on the device.
    try:
        from ck_dsl.runtime import synchronize_and_release

        synchronize_and_release()
    except Exception:
        torch.cuda.synchronize()
    diffs = compare(ref_out, out) if ref_out is not None else None
    extra = f"max_abs={diffs['max_abs']:.4g}" if diffs else ""
    print(f"  {label:14s}: {ms * 1000:9.2f} us  {extra}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, action="append")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--skip-ck", action="store_true")
    # Which apples-to-apples lanes to run. Default: all four.
    parser.add_argument(
        "--paths",
        default="auto,2d,3d",
        help="comma-separated list of paths to run: auto, 2d, 3d",
    )
    parser.add_argument(
        "--set",
        choices=("default", "creative", "fmha", "all"),
        default="default",
        help=(
            "Which scenario set to use. 'default' is the 11 production "
            "scenarios; 'creative' is an exploratory sweep covering "
            "long-context decode, GQA/MQA variants, head_size=256, "
            "bf16, sliding window extremes, and bias combinations; "
            "'fmha' is the subset of CK Tile's "
            "`tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml` that "
            "fits our paged-attention constraints "
            "(causal-only, h in {128,256}, b in {16,64}, "
            "num_queries_per_kv <= 16); 'all' is default + creative."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device unavailable; exiting", file=sys.stderr)
        return 1
    print("device:", torch.cuda.get_device_name(0))

    if args.set == "default":
        scenarios = default_scenarios()
    elif args.set == "creative":
        scenarios = creative_scenarios()
    elif args.set == "fmha":
        scenarios = fmha_scenarios()
    else:  # all
        scenarios = default_scenarios() + creative_scenarios()

    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.name in wanted]
        if not scenarios:
            print(f"unknown scenarios {args.scenario!r}", file=sys.stderr)
            return 2

    requested_paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    for p in requested_paths:
        if p not in ("auto", "2d", "3d"):
            print(f"unknown path {p!r} (expected: auto, 2d, 3d)", file=sys.stderr)
            return 2

    results = []
    for s in scenarios:
        print(
            f"\n=== scenario: {s.name}  dtype={s.dtype}  block={s.block_size}  d={s.head_size} ==="
        )
        # Strongly isolate the previous scenario before allocating inputs.
        # Without this the prior scenario's retained tensor args (held by
        # the CK DSL runtime to keep raw HIP launches' arg buffers alive
        # until the stream sync) can outlive `data` allocation, and the
        # new reference / scenario inputs land in storage the prior
        # scenario's launches are still touching. The symptom is large
        # max_abs drift on the longer-context rows even though each
        # kernel is bit-correct in isolation.
        _isolate_benchmark_lane()
        data = make_inputs(s)
        row = {
            "scenario": s.name,
            "dtype": str(s.dtype),
            "block_size": s.block_size,
            "head_size": s.head_size,
            "num_seqs": len(s.seq_lens),
            "total_q": sum(q for q, _ in s.seq_lens),
        }

        with torch.inference_mode():
            # Keep an independent fp32 clone of the reference so later
            # comparisons cannot be silently corrupted by an adjacent
            # allocation: when many large tensors are allocated in the
            # same scenario the torch caching allocator can hand back a
            # block adjacent to `ref_out`, and an OOB write from a CK
            # ctypes-launched kernel can spill onto it. Clone is fp32 so
            # subsequent compare() promotes losslessly.
            ref_out_raw = run_reference(s, data)
            ref_out = ref_out_raw.float().clone()
            _isolate_benchmark_lane()

            # Reference Triton "natural" path. This is what unified_attention()
            # picks via its own use_2d_kernel selector.
            t_auto, err = _safe_run(
                lambda: _run_triton(
                    s,
                    data,
                    path="auto",
                    warmup=args.warmup,
                    attempts=args.attempts,
                )
            )
            if t_auto:
                t_auto_out, t_auto_ms = t_auto
                row["triton_auto_ms"] = t_auto_ms
                row["triton_auto_vs_ref"] = compare(ref_out, t_auto_out)
                row["triton_natural_path"] = _LAST_TRITON_PATH["path"]
            _row_print("triton-auto", t_auto, ref_out, None)
            _isolate_benchmark_lane()

            # Lane 2/3: forced 2D and 3D on both Triton and CK DSL.
            for path in requested_paths:
                if path == "auto":
                    # Already done above. Just run CK auto for the
                    # apples-to-apples lane on CK side.
                    if not args.skip_ck:
                        _isolate_benchmark_lane()
                        ck_auto, err_ck = _safe_run(
                            lambda: _run_ck_dsl(
                                s,
                                data,
                                path="auto",
                                warmup=args.warmup,
                                attempts=args.attempts,
                            )
                        )
                        if ck_auto:
                            ck_out, ck_ms = ck_auto
                            row["ck_auto_ms"] = ck_ms
                            row["ck_auto_vs_ref"] = compare(ref_out, ck_out)
                            if t_auto:
                                row["speedup_auto"] = t_auto_ms / ck_ms
                                row["ck_auto_vs_triton"] = compare(t_auto_out, ck_out)
                        elif err_ck:
                            row["ck_auto_status"] = err_ck
                        _row_print("ck-auto", ck_auto, ref_out, None)
                        _isolate_benchmark_lane()
                    continue

                # Force-path: Triton on `path`, CK DSL on `path`.
                _isolate_benchmark_lane()
                t_p, err_t = _safe_run(
                    lambda p=path: _run_triton(
                        s,
                        data,
                        path=p,
                        warmup=args.warmup,
                        attempts=args.attempts,
                    )
                )
                _row_print(f"triton-{path}", t_p, ref_out, None)
                _isolate_benchmark_lane()

                if not args.skip_ck:
                    _isolate_benchmark_lane()
                    ck_p, err_c = _safe_run(
                        lambda p=path: _run_ck_dsl(
                            s,
                            data,
                            path=p,
                            warmup=args.warmup,
                            attempts=args.attempts,
                        )
                    )
                    _row_print(f"ck-{path}", ck_p, ref_out, None)

                    if t_p:
                        t_p_out, t_p_ms = t_p
                        row[f"triton_{path}_ms"] = t_p_ms
                        row[f"triton_{path}_vs_ref"] = compare(ref_out, t_p_out)
                    if ck_p:
                        ck_p_out, ck_p_ms = ck_p
                        row[f"ck_{path}_ms"] = ck_p_ms
                        row[f"ck_{path}_vs_ref"] = compare(ref_out, ck_p_out)
                    if t_p and ck_p:
                        row[f"speedup_{path}"] = (
                            row[f"triton_{path}_ms"] / row[f"ck_{path}_ms"]
                            if row[f"ck_{path}_ms"] > 0
                            else 0.0
                        )
                        row[f"ck_{path}_vs_triton_{path}"] = compare(t_p_out, ck_p_out)
                    if err_t and not t_p:
                        row[f"triton_{path}_status"] = err_t
                    if err_c and not ck_p:
                        row[f"ck_{path}_status"] = err_c
                    _isolate_benchmark_lane()
                else:
                    if t_p:
                        t_p_out, t_p_ms = t_p
                        row[f"triton_{path}_ms"] = t_p_ms
                        row[f"triton_{path}_vs_ref"] = compare(ref_out, t_p_out)
                    _isolate_benchmark_lane()
        results.append(row)

    # Apples-to-apples table summaries.
    print("\n--- triton-auto vs ck-auto (each backend's own selector) ---")
    print(
        f"  {'scenario':32s}  {'tri-auto':>10s} {'ck-auto':>10s} {'speedup':>9s} {'tri-path':>9s}"
    )
    for row in results:
        t_ms = row.get("triton_auto_ms")
        c_ms = row.get("ck_auto_ms")
        sp = row.get("speedup_auto", 0.0)
        tp = row.get("triton_natural_path", "?")
        if t_ms is not None and c_ms is not None:
            print(
                f"  {row['scenario']:32s}  {t_ms * 1000:9.2f}us {c_ms * 1000:9.2f}us {sp:8.2f}x {tp:>9s}"
            )
        else:
            print(f"  {row['scenario']:32s}  -")

    if "2d" in requested_paths:
        print("\n--- 2D vs 2D (force both to 2D kernel) ---")
        print(f"  {'scenario':32s}  {'tri-2d':>10s} {'ck-2d':>10s} {'speedup':>9s}")
        for row in results:
            t_ms = row.get("triton_2d_ms")
            c_ms = row.get("ck_2d_ms")
            sp = row.get("speedup_2d", 0.0)
            if t_ms is not None and c_ms is not None:
                print(
                    f"  {row['scenario']:32s}  {t_ms * 1000:9.2f}us {c_ms * 1000:9.2f}us {sp:8.2f}x"
                )
            else:
                t_str = f"{t_ms * 1000:9.2f}us" if t_ms else "      err "
                c_str = (
                    f"{c_ms * 1000:9.2f}us"
                    if c_ms
                    else (
                        " skip(ck) "
                        if row.get("ck_2d_status", (None,))[0] == "skip"
                        else "      err "
                    )
                )
                print(f"  {row['scenario']:32s}  {t_str} {c_str}")

    if "3d" in requested_paths:
        print("\n--- 3D vs 3D (force both to 3D split-KV kernel) ---")
        print(f"  {'scenario':32s}  {'tri-3d':>10s} {'ck-3d':>10s} {'speedup':>9s}")
        for row in results:
            t_ms = row.get("triton_3d_ms")
            c_ms = row.get("ck_3d_ms")
            sp = row.get("speedup_3d", 0.0)
            if t_ms is not None and c_ms is not None:
                print(
                    f"  {row['scenario']:32s}  {t_ms * 1000:9.2f}us {c_ms * 1000:9.2f}us {sp:8.2f}x"
                )
            else:
                t_str = f"{t_ms * 1000:9.2f}us" if t_ms else "      err "
                c_str = (
                    f"{c_ms * 1000:9.2f}us"
                    if c_ms
                    else (
                        " skip(ck) "
                        if row.get("ck_3d_status", (None,))[0] == "skip"
                        else "      err "
                    )
                )
                print(f"  {row['scenario']:32s}  {t_str} {c_str}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2))
        print(f"\nwrote report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
