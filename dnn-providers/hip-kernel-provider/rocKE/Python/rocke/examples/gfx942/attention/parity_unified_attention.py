# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 (CDNA3 / MI300X) unified-attention 2D tiled SDPA-fwd parity + latency.

Torch-reference-only correctness + latency harness for the gfx942 narrow /
wide (flash-regime) tiled attention kernel
(``rocke.instances.gfx942.attention_tiled_2d``). It is the gfx942 sibling of
``examples/gfx950/attention/parity_unified_attention.py``, but with **no**
Triton/AITER dependency: the oracle is a fp32 torch reference, so the example
runs on any box with torch + a gfx942 GPU.

The harness:

  1. Loads the canonical shapes we validate from the shipped ``shapes.json``
     (``--scenario`` selects subsets); these mirror the rocke-provider
     integration-test net plus the case-study perf shapes.
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

    PYTHONPATH=Python .venv/bin/python \\
        Python/rocke/examples/gfx942/attention/parity_unified_attention.py \\
        --scenario correctness

    # force the L4 (WG=64) fallback instead of the default wide4:
    HIPDNN_GFX942_FLASH_WIDE=0 PYTHONPATH=Python .venv/bin/python \\
        Python/rocke/examples/gfx942/attention/parity_unified_attention.py \\
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

ROOT = Path(__file__).resolve().parents[5]  # composablekernel/
sys.path.insert(0, str(ROOT / "Python"))

SHAPES_JSON = Path(__file__).resolve().parent / "shapes.json"

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
    group: str  # "correctness" | "perf" | "decode"

    @property
    def num_queries_per_kv(self) -> int:
        return self.heads // self.kv_heads


def load_shapes(path: Path = SHAPES_JSON) -> List[Shape]:
    """Load the shipped canonical shape set from ``shapes.json``."""
    data = json.loads(path.read_text())
    shapes: List[Shape] = []
    for group, items in data["scenarios"].items():
        for it in items:
            shapes.append(
                Shape(
                    name=it["name"],
                    dtype=it["dtype"],
                    seqlen_q=int(it["seqlen_q"]),
                    seqlen_k=int(it["seqlen_k"]),
                    head_size=int(it["head_size"]),
                    heads=int(it["heads"]),
                    kv_heads=int(it["kv_heads"]),
                    batch=int(it["batch"]),
                    causal=bool(it["causal"]),
                    group=group,
                )
            )
    return shapes


def select_shapes(shapes: List[Shape], selectors: Optional[List[str]]) -> List[Shape]:
    """Resolve ``--scenario`` selectors to a concrete shape list.

    A selector is either a group name (``correctness`` / ``perf`` / ``all``)
    or an exact shape name. Multiple ``--scenario`` flags union together.
    """
    if not selectors:
        return shapes
    wanted: List[Shape] = []
    seen = set()
    for sel in selectors:
        if sel == "all":
            picked = shapes
        elif sel in ("correctness", "perf", "decode"):
            picked = [s for s in shapes if s.group == sel]
        else:
            picked = [s for s in shapes if s.name == sel]
        for s in picked:
            if s.name not in seen:
                seen.add(s.name)
                wanted.append(s)
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
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
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

    ``rocke.instances.UnifiedAttention2DTiledSpec`` re-exports the gfx950 spec
    (the default arch); the gfx950 class does not declare the gfx942-only flash
    fields (``use_mfma_32x32x8`` / ``use_k_single_buffer`` / ...). The arch
    dispatch seam ``_tiled_2d_impl("gfx942")`` returns the gfx942 spec class,
    matching how the provider builds the gfx942 kernel.
    """
    from rocke.instances.common.attention_unified import _tiled_2d_impl

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
    from rocke.instances import build_unified_attention_2d_tiled, supports_tiled_2d
    from rocke.instances.common.attention_unified import _attn_signature
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
    from rocke.instances import UnifiedAttentionProblem
    from rocke.instances.common.attention_unified import _attn_values
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
        num_sms=120,
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
        help="group (correctness|perf|all) or an exact shape name; repeatable",
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

    shapes = select_shapes(load_shapes(), args.scenario)
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
