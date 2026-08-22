# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Authoritative prefill-2D benchmark: LIVE Triton vs rocke variant sweep.

Unlike ``benchmark_prefill2d_traces.py`` (which only times rocke and joins a
pre-profiled Triton CSV), this harness:

  * runs AITER's Triton ``unified_attention`` LIVE, forced to the 2D kernel,
    on the same stream + timer as rocke (apples-to-apples);
  * sweeps a set of rocke 2D kernel variants per shape;
  * checks correctness of every rocke variant against the Triton output;
  * reports, per shape and per bucket (sw / no-sw, bf16 / fp8), the best
    correct rocke variant and its speedup over Triton.
  * optionally benchmarks FlyDSL flash attention (ROCm/FlyDSL dualwave_swp path)
    as an additional comparison baseline.

It is the canonical workbench for closing the prefill-2D gap.

FlyDSL constraints on gfx950:
  * bf16 or fp16 (no fp8 in paged/varlen paths)
  * head_dim must be 64 or 128
  * no sliding window, no alibi, no softcap, no sinks
  * paged KV supported natively (block_table + seqlen_k, page_size=64)
  * uses ROCm/FlyDSL dualwave_swp kernel (gfx950-optimised)
  * requires ROCm/FlyDSL repo; set FLYDSL_PATH to its root (kernels/ is added
    to sys.path at the end to avoid shadowing rocke's own kernels package)

Run:

    export AITER_PATH=<path/to/aiter>
    export FLYDSL_PATH=/path/to/ROCm/FlyDSL
    PYTHONPATH="python:${AITER_PATH}" \
      python rocke/library/benchmarks/gfx950/attention/prefill/benchmark_prefill2d_live.py \
        --shapes <path/to/unified_attention_shapes.jsonl> \
        --variants prod combo fallback \
        --flydsl \
        --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from rocke.assets import shape_utils_dir

DEFAULT_SHAPE_UTILS = shape_utils_dir()


# --------------------------------------------------------------------------
# shape utils + triton
# --------------------------------------------------------------------------
def _load_shape_utils(path: Path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from _ua_shape_utils import (  # type: ignore
        attention_flops,
        dedupe_shapes,
        filter_prefill_2d,
        load_shapes,
        make_inputs,
    )

    return attention_flops, dedupe_shapes, filter_prefill_2d, load_shapes, make_inputs


_UAM = None
_UNIFIED_ATTENTION = None
_ORIG_USE_2D = None


def _import_triton():
    global _UAM, _UNIFIED_ATTENTION, _ORIG_USE_2D
    if _UNIFIED_ATTENTION is not None:
        return
    import aiter.ops.triton.unified_attention as uam  # type: ignore
    from aiter.ops.triton.unified_attention import unified_attention  # type: ignore

    _UAM = uam
    _UNIFIED_ATTENTION = unified_attention
    _ORIG_USE_2D = uam.use_2d_kernel


def _force_triton_2d():
    _import_triton()
    _UAM.use_2d_kernel = lambda *a, **kw: True


def _restore_triton():
    if _UAM is not None and _ORIG_USE_2D is not None:
        _UAM.use_2d_kernel = _ORIG_USE_2D


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


# --------------------------------------------------------------------------
# FlyDSL support check and runner (ROCm/FlyDSL dualwave_swp path for gfx950)
# --------------------------------------------------------------------------
_FLYDSL_FUNC = None
_FLYDSL_KERNELS: dict = {}  # FlyDSL's kernels.* snapshot
_ROCKE_KERNELS: dict = {}  # rocke's kernels.* snapshot


def _import_flydsl():
    global _FLYDSL_FUNC, _FLYDSL_KERNELS, _ROCKE_KERNELS
    if _FLYDSL_FUNC is not None:
        return
    import os

    flydsl_path = os.environ.get("FLYDSL_PATH")
    if not flydsl_path:
        raise RuntimeError(
            "FLYDSL_PATH env var must be set to the ROCm/FlyDSL repo root"
        )
    sys.path.insert(0, flydsl_path)
    # Stash rocke's kernels.* and load FlyDSL's instead.
    _rocke = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "kernels" or k.startswith("kernels.")
    }
    try:
        import kernels.attention.flash_attn_generic  # type: ignore  # noqa: F401
        import kernels.attention.flash_attn_gfx950  # type: ignore  # noqa: F401
        from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # type: ignore

        _FLYDSL_FUNC = flydsl_flash_attn_func
        _FLYDSL_KERNELS = {
            k: v
            for k, v in sys.modules.items()
            if k == "kernels" or k.startswith("kernels.")
        }
    finally:
        sys.path.remove(flydsl_path)
        for k in [k for k in sys.modules if k == "kernels" or k.startswith("kernels.")]:
            del sys.modules[k]
        sys.modules.update(_rocke)
        _ROCKE_KERNELS = {
            k: v
            for k, v in sys.modules.items()
            if k == "kernels" or k.startswith("kernels.")
        }


def _flydsl_swap(flydsl: bool) -> None:
    """Swap kernels.* in sys.modules between FlyDSL and rocke."""
    for k in [k for k in sys.modules if k == "kernels" or k.startswith("kernels.")]:
        del sys.modules[k]
    sys.modules.update(_FLYDSL_KERNELS if flydsl else _ROCKE_KERNELS)


_FLYDSL_PAGED_PAGE_SIZE = 64


def _flydsl_supported(shape) -> tuple[bool, str]:
    """Return (supported, reason) for whether FlyDSL can run this shape on gfx950."""
    if shape.head_size not in (64, 128):
        return False, f"head_size={shape.head_size} (need 64 or 128)"
    if "float8" in shape.k_dtype:
        return False, "fp8 KV not supported in paged/varlen path"
    if shape.softcap > 0:
        return False, f"softcap={shape.softcap} not supported"
    if shape.has_alibi:
        return False, "alibi not supported"
    if shape.has_sinks:
        return False, "sinks not supported"
    if shape.window_size[0] >= 0:
        return False, "sliding_window not supported"
    return True, ""


def _run_flydsl_live(shape, data, *, warmup, iters):
    """Time FlyDSL flash attention (ROCm/FlyDSL dualwave_swp path, THD layout).

    Uses paged KV natively when block_size == 64 (FlyDSL page_size requirement).
    Falls back to dense KV reconstruction + varlen path for other block sizes.
    """
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    _import_flydsl()
    hip_stream = _bench_stream_handle()

    q = data["query"]  # [total_q, nheads_q, head_dim]
    k_cache = data["key_cache"]  # [num_blocks, block_size, nheads_k, head_dim]
    v_cache = data["value_cache"]
    block_tables = data["block_tables"]  # [num_seqs, max_blocks_per_seq]
    kv_lens = data["kv_lens"]  # [num_seqs] int32
    cu_seqlens_q = data["cu_seqlens_q"]  # [num_seqs+1] int32

    num_seqs = shape.num_seqs
    cu_seqlens_kv = torch.zeros(num_seqs + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_kv[1:] = kv_lens.cumsum(0).to(torch.int32)
    out = torch.empty_like(q)

    if shape.block_size == _FLYDSL_PAGED_PAGE_SIZE:
        # Paged path: FlyDSL handles paging natively.
        def call_once():
            _FLYDSL_FUNC(
                q,
                k_cache,
                v_cache,
                causal=True,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=int(data["max_query_len"]),
                block_table=block_tables,
                seqlen_k=kv_lens,
                cross_seqlen=True,
                out=out,
            )

    else:
        # Varlen path: reconstruct dense KV from paged cache (one-time setup,
        # outside the timed loop) and call FlyDSL with cu_seqlens_kv only.
        block_size = shape.block_size
        nheads_k = shape.num_kv_heads
        head_dim = shape.head_size
        kv_lens_cpu = kv_lens.cpu().tolist()
        total_k = sum(kv_lens_cpu)
        k_dense = torch.empty(
            total_k, nheads_k, head_dim, dtype=q.dtype, device=q.device
        )
        v_dense = torch.empty(
            total_k, nheads_k, head_dim, dtype=q.dtype, device=q.device
        )
        offset = 0
        for seq_idx in range(num_seqs):
            seq_kv_len = kv_lens_cpu[seq_idx]
            blocks_needed = (seq_kv_len + block_size - 1) // block_size
            tokens_copied = 0
            for blk in range(blocks_needed):
                phys = int(block_tables[seq_idx, blk].item())
                n = min(block_size, seq_kv_len - tokens_copied)
                k_dense[offset : offset + n] = k_cache[phys, :n]
                v_dense[offset : offset + n] = v_cache[phys, :n]
                offset += n
                tokens_copied += n

        def call_once():
            _FLYDSL_FUNC(
                q,
                k_dense,
                v_dense,
                causal=True,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=int(data["max_query_len"]),
                max_seqlen_kv=int(data["max_kv_len"]),
                cross_seqlen=True,
                out=out,
            )

    _flydsl_swap(flydsl=True)
    try:
        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
    finally:
        _flydsl_swap(flydsl=False)
    return out, ms


def _gm(vals: list[float]) -> float:
    vals = [v for v in vals if v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


# --------------------------------------------------------------------------
# rocke variant specs
# --------------------------------------------------------------------------
def _variant_flags(name: str, *, sliding_window: int, dtype: str, is_fp8: bool) -> dict:
    """Return the UnifiedAttention2DTiledSpec transposed/opt flags for a variant.

    Variant grammar (``_`` separated tokens layered on a base):
      fallback        : plain R4 (16x16x32, no transpose opts), nw4 mw16
      fallback_nw2    : fallback with num_warps=2
      combo           : full s1mask + half-local-pv + mask-limit + fast-kv-desc
      combo_nw2       : combo with num_warps=2 (BLOCK_M=64)
      combo_t1        : combo with tile_size = 1*block_size (T=32)
      combo_t4        : combo with tile_size = 4*block_size (T=128)
      combo_earlyv    : combo + use_early_v_schedule
      r4_t32          : mfma_32x32 + transposed_qk only
    """
    base = dict(
        num_warps=4,
        block_m_per_warp=32,
        tile_mult=2,  # tile_size = tile_mult * block_size
        use_mfma_32x32=False,
        use_transposed_qk_32x32=False,
        use_transposed_scalar_state=False,
        use_transposed_mask_once=False,
        use_transposed_half_local_pv=False,
        use_mfma32_skip_legacy_qreg=False,
        use_transposed_mask_limit=False,
        use_fast_paged_kv_desc=False,
        use_early_v_schedule=False,
        waves_per_eu=2,
        use_i64_kv_addr=False,
    )
    toks = name.split("_")
    head = toks[0]
    if head == "fallback":
        base.update(num_warps=4, block_m_per_warp=16)
    elif head == "r4" and len(toks) > 1 and toks[1] == "t32":
        base.update(use_mfma_32x32=True, use_transposed_qk_32x32=True)
        toks = [head] + toks[2:]
    elif head == "combo":
        base.update(
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=(sliding_window == 0),
            use_transposed_half_local_pv=True,
            use_mfma32_skip_legacy_qreg=True,
            use_transposed_mask_limit=(sliding_window == 0),
            use_fast_paged_kv_desc=True,
            # Match the production dispatcher: ``_select_2d_waves_per_eu``
            # returns 4 for the combo family. The harness used to default the
            # combo to ``waves_per_eu=2`` (the base value below), which builds
            # an occupancy-starved kernel that does NOT match what production
            # ships -- it under-reported the combo by ~25% (0.85x vs the
            # ~1.07x the wpe=4 kernel the dispatcher actually builds gets) on
            # the long-context multi-seq cohort. The ``_we2`` / ``_we3`` /
            # ``_wenone`` modifier tokens still override for sweeps.
            waves_per_eu=4,
        )
    else:
        raise ValueError(f"unknown variant head {head!r}")
    # modifier tokens
    for t in toks[1:]:
        if t == "nw2":
            base["num_warps"] = 2
        elif t == "nw1":
            base["num_warps"] = 1
        elif t == "t1":
            base["tile_mult"] = 1
        elif t == "t4":
            base["tile_mult"] = 4
        elif t == "t8":
            base["tile_mult"] = 8
        elif t == "mw16":
            base["block_m_per_warp"] = 16
        elif t == "nomlim":
            base["use_transposed_mask_limit"] = False
        elif t == "earlyv":
            base["use_early_v_schedule"] = True
        elif t == "i64":
            base["use_i64_kv_addr"] = True
        elif t == "we3":
            base["waves_per_eu"] = 3
        elif t == "we4":
            base["waves_per_eu"] = 4
        elif t == "wenone":
            base["waves_per_eu"] = None
        elif t in ("t32",):
            pass
        else:
            raise ValueError(f"unknown variant modifier {t!r} in {name!r}")
    # mw16 cannot use mfma_32x32 transpose path
    if base["block_m_per_warp"] == 16:
        base.update(
            use_mfma_32x32=False,
            use_transposed_qk_32x32=False,
            use_transposed_scalar_state=False,
            use_transposed_mask_once=False,
            use_transposed_half_local_pv=False,
            use_mfma32_skip_legacy_qreg=False,
            use_transposed_mask_limit=False,
        )
    return base


class CkVariantBench:
    def __init__(self, *, compile_backend: Optional[str] = None, num_cus: int = 256):
        self.compile_backend = compile_backend
        self.num_cus = num_cus
        self._launchers: dict[tuple, Any] = {}

    def _problem(self, shape, sliding_window: int, is_fp8: bool):
        from kernels import UnifiedAttentionProblem

        return UnifiedAttentionProblem(
            total_q=shape.total_q,
            num_seqs=shape.num_seqs,
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size,
            block_size=shape.block_size,
            max_seqlen_q=shape.max_seqlen_q,
            max_seqlen_k=shape.max_seqlen_k,
            dtype="bf16" if shape.q_dtype == "torch.bfloat16" else "fp16",
            sliding_window=sliding_window,
            softcap=float(shape.softcap),
            use_sinks=shape.has_sinks,
            use_alibi=shape.has_alibi,
            use_qq_bias=False,
            use_fp8=is_fp8,
            num_cus=self.num_cus,
            compile_backend=self.compile_backend,
        )

    def build(self, shape, variant: str, sliding_window: int, is_fp8: bool):
        from rocke import compile_kernel
        from kernels import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )
        from kernels.common.attention_unified import (
            _attn_signature,
            _select_2d_compile_backend,
        )
        from rocke.runtime import KernelLauncher

        dtype = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
        problem = self._problem(shape, sliding_window, is_fp8)
        flags = _variant_flags(
            variant, sliding_window=sliding_window, dtype=dtype, is_fp8=is_fp8
        )
        tile_size = flags["tile_mult"] * shape.block_size
        kv_storage_dtype = "fp8e4m3" if is_fp8 else None
        ok, reason = supports_tiled_2d(
            head_size=shape.head_size,
            block_size=shape.block_size,
            dtype=dtype,
            num_queries_per_kv=problem.num_queries_per_kv,
            use_alibi=problem.use_alibi,
            use_qq_bias=False,
            use_fp8=is_fp8,
            q_dtype=problem.q_dtype,
            num_warps=flags["num_warps"],
            block_m_per_warp=flags["block_m_per_warp"],
            kv_storage_dtype=kv_storage_dtype,
            tile_size=tile_size,
            use_mfma_32x32x8=flags["use_mfma_32x32"],
            use_transposed_qk_32x32=flags["use_transposed_qk_32x32"],
        )
        if not ok:
            raise NotImplementedError(f"supports_tiled_2d: {reason}")

        spec = UnifiedAttention2DTiledSpec(
            head_size=shape.head_size,
            block_size=shape.block_size,
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            dtype=dtype,
            use_sinks=shape.has_sinks,
            sliding_window=sliding_window,
            has_softcap=shape.softcap > 0,
            use_alibi=shape.has_alibi,
            use_qq_bias=False,
            num_seqs=shape.num_seqs,
            num_warps=flags["num_warps"],
            waves_per_eu=flags["waves_per_eu"],
            tile_size=tile_size,
            block_m_per_warp=flags["block_m_per_warp"],
            kv_storage_dtype=kv_storage_dtype,
            use_fp8_mfma_qk=is_fp8,
            use_mfma_32x32=flags["use_mfma_32x32"],
            use_transposed_qk_32x32=flags["use_transposed_qk_32x32"],
            use_transposed_scalar_state=flags["use_transposed_scalar_state"],
            use_transposed_mask_once=flags["use_transposed_mask_once"],
            use_transposed_half_local_pv=flags["use_transposed_half_local_pv"],
            use_mfma32_skip_legacy_qreg=flags["use_mfma32_skip_legacy_qreg"],
            use_transposed_mask_limit=flags["use_transposed_mask_limit"],
            use_fast_paged_kv_desc=flags["use_fast_paged_kv_desc"],
            use_early_v_schedule=flags["use_early_v_schedule"],
            use_i64_kv_addr=flags["use_i64_kv_addr"],
        )
        key = (shape.signature, variant, spec.kernel_name(), self.compile_backend)
        if key not in self._launchers:
            kernel = build_unified_attention_2d_tiled(spec)
            backend = self.compile_backend or _select_2d_compile_backend(problem)
            if backend == "hipcc":
                from rocke.helpers.compile import compile_kernel_via_hipcc

                artifact = compile_kernel_via_hipcc(kernel)
            else:
                artifact = compile_kernel(kernel, capture_ir_text=False)
            self._launchers[key] = (
                KernelLauncher(
                    hsaco=artifact.hsaco,
                    kernel_name=artifact.kernel_name,
                    signature=_attn_signature(
                        dtype, include_bt_stride=True, include_qq_bias_stride=True
                    ),
                    cache_key=("prefill2d_live", key),
                ),
                spec,
                problem,
            )
        return self._launchers[key]

    def run(self, shape, data, variant, sliding_window, is_fp8, *, warmup, iters):
        import torch
        from kernels.common.attention_unified import _attn_values
        from rocke.runtime import LaunchConfig, synchronize_and_release, time_launches

        launcher, spec, problem = self.build(shape, variant, sliding_window, is_fp8)
        hip_stream = _bench_stream_handle()
        out = torch.empty_like(data["query"])
        vals = _attn_values(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_seqlens_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(shape.softcap),
            sinks=data["sinks"],
            bt_stride=int(data["block_tables"].stride(0)),
            include_bt_stride=True,
            alibi_slopes=data["alibi_slopes"],
            qq_bias=None,
            qq_bias_stride_0=0,
            include_qq_bias_stride=True,
            k_scale=1.0,
            v_scale=1.0,
        )
        cfg = LaunchConfig(
            grid=(
                int(shape.num_kv_heads),
                int(shape.total_q // spec.block_q + shape.num_seqs),
                1,
            ),
            block=(64 * spec.num_warps, 1, 1),
            stream=hip_stream,
        )

        def call_once():
            launcher(vals, config=cfg)

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return out, ms, spec.kernel_name()


def _run_triton_live(shape, data, sliding_window, is_fp8, *, warmup, iters):
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    _import_triton()
    out = torch.empty_like(data["query"])
    window_size = (sliding_window - 1, 0) if sliding_window else (-1, -1)
    descale = None
    if is_fp8:
        descale = torch.ones(1, dtype=torch.float32, device=data["query"].device)
    hip_stream = _bench_stream_handle()
    _force_triton_2d()
    try:

        def call_once():
            _UNIFIED_ATTENTION(
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=out,
                cu_seqlens_q=data["cu_seqlens_q"],
                seqused_k=data["kv_lens"],
                max_seqlen_q=data["max_query_len"],
                max_seqlen_k=data["max_kv_len"],
                softmax_scale=data["scale"],
                causal=True,
                window_size=window_size,
                block_table=data["block_tables"],
                softcap=float(shape.softcap),
                q_descale=None,  # Q is bf16 in these traces; only K/V are fp8
                k_descale=descale,
                v_descale=descale,
                alibi_slopes=data["alibi_slopes"],
                qq_bias=None,
                sinks=data["sinks"],
            )

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
    finally:
        _restore_triton()
    return out, ms


def _run_aoTriton_live(shape, data, sliding_window, is_fp8, *, warmup, iters):
    """Time AOTriton flash SDPA as a secondary baseline.

    Reconstructs dense [B, H, S, D] KV from the paged cache before timing
    (reconstruction excluded from the timed region). Skips FP8 and shapes
    where flash is ineligible (non-square causal, alibi). Returns
    (out_tensor, ms) or raises on skip/failure.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from rocke.runtime import synchronize_and_release, time_launches

    if is_fp8:
        raise NotImplementedError("AOTriton skip: FP8 not supported by flash SDPA")
    if shape.has_alibi:
        raise NotImplementedError("AOTriton skip: alibi not supported by flash SDPA")
    if sliding_window:
        raise NotImplementedError(
            "AOTriton skip: sliding_window not supported by flash SDPA"
        )
    if float(shape.softcap):
        raise NotImplementedError("AOTriton skip: softcap not supported by flash SDPA")

    is_causal = True

    # Reconstruct dense KV from paged cache (outside timed region).
    kv_lens_list = data["kv_lens"].tolist()
    num_seqs = shape.num_seqs
    num_kv_heads = shape.num_kv_heads
    head_size = shape.head_size
    nrep = shape.num_query_heads // num_kv_heads

    ks, vs = [], []
    for bi in range(num_seqs):
        klen = kv_lens_list[bi]
        bt = data["block_tables"][bi]
        ks.append(data["key_cache"][bt].reshape(-1, num_kv_heads, head_size)[:klen])
        vs.append(data["value_cache"][bt].reshape(-1, num_kv_heads, head_size)[:klen])

    max_klen = max(t.shape[0] for t in ks)
    dtype = data["query"].dtype
    device = data["query"].device

    kh = torch.zeros(
        num_seqs, num_kv_heads, max_klen, head_size, dtype=dtype, device=device
    )
    vh = torch.zeros(
        num_seqs, num_kv_heads, max_klen, head_size, dtype=dtype, device=device
    )
    for bi in range(num_seqs):
        klen = ks[bi].shape[0]
        kh[bi, :, :klen, :] = ks[bi].transpose(0, 1)
        vh[bi, :, :klen, :] = vs[bi].transpose(0, 1)
    kh = kh.repeat_interleave(nrep, dim=1).contiguous()
    vh = vh.repeat_interleave(nrep, dim=1).contiguous()

    query_lens = data["cu_seqlens_q"][1:] - data["cu_seqlens_q"][:-1]
    max_qlen = int(query_lens.max().item())
    num_query_heads = shape.num_query_heads
    qh = torch.zeros(
        num_seqs, num_query_heads, max_qlen, head_size, dtype=dtype, device=device
    )
    for bi in range(num_seqs):
        qlen = int(query_lens[bi].item())
        q_start = int(data["cu_seqlens_q"][bi].item())
        qh[bi, :, :qlen, :] = data["query"][q_start : q_start + qlen].transpose(0, 1)

    scale = data["scale"]

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            _ = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=is_causal, scale=scale
            )
        torch.cuda.synchronize()
    except RuntimeError as e:
        raise NotImplementedError(f"AOTriton skip: flash ineligible — {e}") from e

    hip_stream = _bench_stream_handle()

    def call_once():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=is_causal, scale=scale
            )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out_padded = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, is_causal=is_causal, scale=scale
        )
    torch.cuda.synchronize()
    out = torch.zeros(
        shape.total_q, num_query_heads, head_size, dtype=dtype, device=device
    )
    for bi in range(num_seqs):
        qlen = int(query_lens[bi].item())
        q_start = int(data["cu_seqlens_q"][bi].item())
        out[q_start : q_start + qlen] = out_padded[bi, :, :qlen, :].transpose(0, 1)

    return out, ms


def _compare(a, b) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).abs().max().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", type=Path, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="bf16")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["prod", "combo", "fallback"],
        help="rocke variants to sweep: prod combo fallback r4_t32 combo_sw",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="subsample every Nth shape")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # Production paged-KV caches have hundreds of thousands of blocks, so the
    # KV working set vastly exceeds L2 and attention is HBM-bandwidth-bound.
    # A small cap makes the cache artificially L2-resident, which understates
    # rocke (its async-DMA KV loads are more bandwidth-efficient than
    # Triton's, the advantage that only shows once you are HBM-bound). Default
    # to a production-representative cap. Measured: bf16 cohort 0.90x at
    # cap=8192 vs 1.11x at cap=65536.
    ap.add_argument("--cap-blocks", type=int, default=65536)
    ap.add_argument("--num-cus", type=int, default=256)
    ap.add_argument(
        "--compile-backend",
        choices=("auto", "llvm", "hipcc"),
        default="auto",
        help="Compile backend for all CK DSL lanes (prod + build() variants). "
        "'auto' (default) matches production dispatch "
        "(_select_2d_compile_backend); 'llvm'/'hipcc' force a backend for A/B. "
        "Forcing 'llvm' mismeasures cohorts whose production backend is hipcc "
        "(e.g. D256 large prefill) by ~3x.",
    )
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--shape-utils-path", type=Path, default=DEFAULT_SHAPE_UTILS)
    ap.add_argument(
        "--output-json", type=Path, default=Path("/tmp/prefill2d_live.json")
    )
    ap.add_argument(
        "--flydsl",
        action="store_true",
        help="Also benchmark FlyDSL flash attention (set FLYDSL_PATH to ROCm/FlyDSL repo root)",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    (
        attention_flops,
        dedupe_shapes,
        filter_prefill_2d,
        load_shapes,
        make_inputs,
    ) = _load_shape_utils(args.shape_utils_path)

    dtype_filter = None if args.dtype == "all" else args.dtype
    shapes = filter_prefill_2d(load_shapes(args.shapes), dtype=dtype_filter)
    shapes = dedupe_shapes(shapes)
    shapes = shapes[:: args.stride]
    if args.limit is not None:
        shapes = shapes[: args.limit]
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shapes: {len(shapes)}  variants: {args.variants}  flydsl={args.flydsl}")

    _cb = None if args.compile_backend == "auto" else args.compile_backend
    bench = CkVariantBench(num_cus=args.num_cus, compile_backend=_cb)
    results = []
    n_fly_supported = 0
    n_fly_correct = 0
    for i, shape in enumerate(shapes, 1):
        sw = shape.window_size[0] + 1 if shape.window_size[0] >= 0 else 0
        is_fp8 = "float8" in shape.k_dtype
        tag = f"[{i}/{len(shapes)}] {shape.signature}"
        try:
            data = make_inputs(shape, seed=args.seed, cap_blocks=args.cap_blocks)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  INPUT FAIL: {exc!r}")
            traceback.print_exc()
            continue

        tri_out, tri_ms = None, None
        try:
            tri_out, tri_ms = _run_triton_live(
                shape, data, sw, is_fp8, warmup=args.warmup, iters=args.iterations
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  TRITON SKIP: {exc!r}")

        # AOTriton flash baseline (best-effort; skipped for FP8 / ineligible shapes).
        aot_ms = None
        try:
            _, aot_ms = _run_aoTriton_live(
                shape, data, sw, is_fp8, warmup=args.warmup, iters=args.iterations
            )
        except NotImplementedError as exc:
            print(f"{tag}  AOTRITON SKIP: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  AOTRITON FAIL: {exc!r}")

        # ------------------------------------------------------------------
        # FlyDSL measurement (optional, --flydsl flag)
        # ------------------------------------------------------------------
        fly_ms = None
        fly_err = None
        fly_ok = None
        fly_skip_reason = None
        fly_error = None

        if args.flydsl:
            fly_ok_for_shape, fly_skip_reason = _flydsl_supported(shape)
            if not fly_ok_for_shape:
                fly_status = f"SKIP({fly_skip_reason})"
            else:
                n_fly_supported += 1
                try:
                    fly_out, fly_ms = _run_flydsl_live(
                        shape, data, warmup=args.warmup, iters=args.iterations
                    )
                    fly_err = _compare(fly_out, tri_out)
                    fly_ok = fly_err <= args.tol
                    if fly_ok:
                        n_fly_correct += 1
                    fly_status = f"{fly_ms * 1000:.1f}us({'ok' if fly_ok else 'WRONG'})"
                except Exception as exc:  # noqa: BLE001
                    fly_error = repr(exc)
                    fly_status = f"ERR({exc!r})"

        rec = {
            "signature": shape.signature,
            "sliding_window": sw,
            "is_fp8": is_fp8,
            "num_seqs": shape.num_seqs,
            "total_q": shape.total_q,
            "max_seqlen_k": shape.max_seqlen_k,
            "triton_ms": tri_ms,
            "aoTriton_ms": aot_ms,
            "flydsl_ms": fly_ms,
            "flydsl_ok": fly_ok,
            "flydsl_max_abs": fly_err,
            "flydsl_skip_reason": fly_skip_reason,
            "flydsl_error": fly_error,
            "variants": {},
        }
        if tri_ms is not None and fly_ms is not None:
            rec["speedup_flydsl_vs_triton"] = tri_ms / fly_ms if fly_ms > 0 else 0.0
        best = None
        for v in args.variants:
            try:
                if v in ("prod", "ck3d"):
                    # production dispatch via run_unified_attention_torch
                    ck_out, ck_ms, kname = _run_prod(
                        shape,
                        data,
                        sw,
                        is_fp8,
                        bench,
                        warmup=args.warmup,
                        iters=args.iterations,
                        backend=("3d" if v == "ck3d" else "auto"),
                    )
                else:
                    ck_out, ck_ms, kname = bench.run(
                        shape,
                        data,
                        v,
                        sw,
                        is_fp8,
                        warmup=args.warmup,
                        iters=args.iterations,
                    )
                err = _compare(ck_out, tri_out) if tri_out is not None else None
                ok = err <= args.tol if err is not None else True
                spd = tri_ms / ck_ms if (tri_ms and ck_ms > 0) else 0.0
                rec["variants"][v] = {
                    "ms": ck_ms,
                    "speedup": spd,
                    "max_abs": err,
                    "ok": ok,
                    "kernel": kname,
                }
                if ok and (best is None or ck_ms < best[1]):
                    best = (v, ck_ms)
            except Exception as exc:  # noqa: BLE001
                rec["variants"][v] = {"error": repr(exc)}
        rec["best_variant"] = best[0] if best else None
        best_ms = rec["variants"][best[0]]["ms"] if best else None
        rec["best_ms"] = best_ms
        rec["best_speedup_vs_triton"] = (
            tri_ms / best_ms
            if (tri_ms and best_ms is not None and best_ms > 0)
            else None
        )
        rec["best_speedup_vs_aoTriton"] = (
            aot_ms / best_ms
            if (aot_ms is not None and best_ms is not None and best_ms > 0)
            else None
        )
        rec["best_speedup_vs_flydsl"] = (
            fly_ms / best_ms
            if (fly_ms is not None and best_ms is not None and best_ms > 0)
            else None
        )
        flops = attention_flops(shape, data["query_lens"], data["kv_lens_list"])
        rec["flops"] = flops
        rec["best_tflops"] = (
            flops / (best_ms * 1e-3) / 1e12
            if (best_ms is not None and best_ms > 0)
            else None
        )
        results.append(rec)
        tri_str = f"tri={tri_ms * 1000:.1f}us" if tri_ms else "tri=N/A"
        aot_str = f"aot={aot_ms * 1000:.1f}us" if aot_ms else "aot=N/A"
        vs = "  ".join(
            f"{v}={rec['variants'][v]['ms'] * 1000:.1f}us"
            f"{'' if rec['variants'][v].get('ok') else '!'}"
            for v in args.variants
            if "ms" in rec["variants"][v]
        )
        best_str = f"{best_ms * 1000:.1f}us" if best_ms is not None else "N/A"
        tf_str = f" {rec['best_tflops']:.1f}TF" if rec.get("best_tflops") else ""
        fly_str = ""
        if args.flydsl:
            fly_spd_str = ""
            if tri_ms is not None and fly_ms is not None and fly_ms > 0:
                fly_spd_str = f"({tri_ms / fly_ms:.2f}x tri)"
            fly_best_spd_str = (
                f" fly_spd={rec['best_speedup_vs_flydsl']:.2f}x"
                if rec.get("best_speedup_vs_flydsl") is not None
                else ""
            )
            fly_str = f" | fly={fly_status}{fly_spd_str}{fly_best_spd_str}"
        print(
            f"{tag} sw={sw} fp8={int(is_fp8)} {tri_str} {aot_str} | {vs} | best={rec['best_variant']}={best_str}{tf_str}{fly_str}"
        )

    args.output_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.output_json}  ({len(results)} shapes)")

    # summary
    def bucket(r):
        return (
            "fp8" if r["is_fp8"] else "bf16",
            "sw" if r["sliding_window"] else "nosw",
        )

    buckets: dict[tuple, list] = {}
    for r in results:
        buckets.setdefault(bucket(r), []).append(r)
    print("\n=== geomean best rocke speedup vs Triton, AOTriton, FlyDSL ===")
    for b in sorted(buckets):
        rs = buckets[b]
        tri_spds = [
            r["best_speedup_vs_triton"] for r in rs if r.get("best_speedup_vs_triton")
        ]
        aot_spds = [
            r["best_speedup_vs_aoTriton"]
            for r in rs
            if r.get("best_speedup_vs_aoTriton")
        ]
        fly_spds = [
            r["best_speedup_vs_flydsl"] for r in rs if r.get("best_speedup_vs_flydsl")
        ]
        best_us = [r["best_ms"] * 1000 for r in rs if r.get("best_ms")]
        tflops = [r["best_tflops"] for r in rs if r.get("best_tflops")]
        lat_part = f"best_lat_gm={_gm(best_us):.1f}us (n={len(best_us)})"
        tf_part = f"  tflops_gm={_gm(tflops):.1f}" if tflops else ""
        tri_part = (
            f"  vs_tri={_gm(tri_spds):.3f}x  wins={sum(1 for x in tri_spds if x > 1)}/{len(tri_spds)}"
            if tri_spds
            else ""
        )
        aot_part = (
            f"  vs_aot={_gm(aot_spds):.3f}x (n={len(aot_spds)})" if aot_spds else ""
        )
        fly_part = (
            f"  vs_fly={_gm(fly_spds):.3f}x (n={len(fly_spds)})" if fly_spds else ""
        )
        print(
            f"  {b[0]:4s}/{b[1]:4s}  n={len(rs):3d}  {lat_part}{tf_part}{tri_part}{aot_part}{fly_part}"
        )
    print("\n=== per-variant geomean (correct shapes only) ===")
    for v in args.variants:
        sp = [
            r["variants"][v]["speedup"]
            for r in results
            if v in r["variants"] and r["variants"][v].get("ok")
        ]
        ncorrect = sum(
            1 for r in results if v in r["variants"] and r["variants"][v].get("ok")
        )
        nfail = sum(
            1
            for r in results
            if v in r["variants"] and r["variants"][v].get("ok") is False
        )
        print(
            f"  {v:10s}  geomean={_gm(sp):.3f}x  correct={ncorrect} incorrect={nfail}"
        )
    if args.flydsl:
        print("\n=== FlyDSL summary ===")
        fly_results = [r for r in results if r["flydsl_ms"] is not None]
        if fly_results:
            print(f"  shapes total:       {len(results)}")
            print(f"  FlyDSL supported:   {n_fly_supported}")
            print(f"  FlyDSL ran:         {len(fly_results)}")
            print(
                f"  correctness:        {n_fly_correct}/{n_fly_supported} correct (tol={args.tol})"
            )
            print()
            fly_buckets: dict[tuple, list] = {}
            for r in fly_results:
                fly_buckets.setdefault(bucket(r), []).append(r)
            for b in sorted(fly_buckets):
                rs = fly_buckets[b]
                vs_tri = [
                    r["speedup_flydsl_vs_triton"]
                    for r in rs
                    if r.get("speedup_flydsl_vs_triton")
                ]
                vs_ck = [
                    r["best_speedup_vs_flydsl"]
                    for r in rs
                    if r.get("best_speedup_vs_flydsl")
                ]
                tri_part = (
                    f"vs_tri={_gm(vs_tri):.3f}x  wins={sum(1 for x in vs_tri if x > 1)}/{len(vs_tri)}"
                    if vs_tri
                    else "vs_tri=N/A"
                )
                ck_part = (
                    f"  rocke_vs_fly={_gm(vs_ck):.3f}x (n={len(vs_ck)})"
                    if vs_ck
                    else ""
                )
                print(f"  {b[0]:4s}/{b[1]:4s}  n={len(rs):3d}  {tri_part}{ck_part}")
        else:
            print("  No FlyDSL results (all shapes skipped or errored).")
    return 0


def _run_prod(shape, data, sw, is_fp8, bench, *, warmup, iters, backend="auto"):
    """Time the production path via the platform dispatcher.

    Builds an :class:`~dispatch.attention.AttentionRequest` and calls
    :func:`~dispatch.attention.dispatch_attention` to select the registered
    kernel candidate (2d-tiled or 3d split-KV).  The resolved path drives
    ``run_unified_attention_torch`` so the benchmark exercises exactly the same
    code path as the production provider.
    """
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention
    from kernels import run_unified_attention_torch
    from kernels.common.attention_unified import _resolve_attention_arch
    from rocke.runtime import synchronize_and_release, time_launches

    arch = _resolve_attention_arch()
    dtype_str = "bf16" if shape.q_dtype == "torch.bfloat16" else "fp16"
    req = AttentionRequest(
        batch=shape.num_seqs,
        nhead_q=shape.num_query_heads,
        nhead_k=shape.num_kv_heads,
        seqlen_q=shape.max_seqlen_q,
        seqlen_k=shape.max_seqlen_k,
        hdim_q=shape.head_size,
        hdim_v=shape.head_size,
        arch=arch,
        dtype=dtype_str,
        sliding_window=sw,
        kv_block_size=shape.block_size,
        num_cus=bench.num_cus,
    )

    # Force path when caller requests a specific one (e.g. backend="3d").
    if backend != "auto":
        req = AttentionRequest(**{**req.__dict__, "algorithm": f"unified_{backend}"})

    try:
        result = dispatch_attention(req)
        dispatched_path = result.spec.path  # "2d" or "3d"
    except ValueError:
        # No registered candidate supports this shape — fall back to scalar label.
        dispatched_path = backend if backend != "auto" else "scalar"

    run_backend = "tiled" if dispatched_path == "2d" else dispatched_path

    problem = bench._problem(shape, sw, is_fp8)
    out = torch.empty_like(data["query"])
    hip_stream = _bench_stream_handle()

    # Report the tiling-level kernel name (includes num_warps / tile_size / etc.)
    # so the benchmark output is comparable to the pre-dispatcher baseline.
    if run_backend == "tiled":
        from kernels import supports_native_unified_attention_tiled
        from kernels.common.attention_unified import _tiled_spec_from_problem

        ok_t, _ = supports_native_unified_attention_tiled(problem)
        instance_name = (
            _tiled_spec_from_problem(problem).kernel_name() if ok_t else "scalar"
        )
    elif run_backend == "3d":
        from kernels import supports_native_unified_attention_3d_tiled

        ok_3d, _ = supports_native_unified_attention_3d_tiled(problem)
        instance_name = "3d" if ok_3d else "scalar"
    else:
        instance_name = "scalar"

    def call_once():
        run_unified_attention_torch(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_seqlens_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(shape.softcap),
            sinks=data["sinks"],
            alibi_slopes=data["alibi_slopes"],
            backend=run_backend,
            stream=hip_stream,
        )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return out, ms, instance_name


if __name__ == "__main__":
    sys.exit(main())
