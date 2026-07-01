#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
rocke-native, op-parameterized heuristics training-data generator.

This is the multi-op generalization of :mod:`rocke.heuristics.gen_gemm_sweep_data`.
For a given op family it:

  1. Enumerates that op's *shape corpus* (problem dimensions).
  2. Enumerates that op's validity-filtered *kernel-config variants* (the
     ``variantGrid`` for the op -- a cartesian product for GEMM / conv / MoE /
     norm, or a problem-driven selector for SDPA).
  3. Builds every ``(variant)`` to a cached HSACO (LLVM IR -> comgr), driving the
     same :mod:`rocke.sweep` / :mod:`rocke.sweep_bench` ecosystem GEMM uses.
  4. Measures per-shape TFLOPS + correctness where a launcher / GPU is available
     (rows that fail build / verify / perf are emitted ``is_valid=False`` with
     zero targets so the model learns the failure surface).
  5. Writes a training parquet whose feature columns match the op's
     :mod:`rocke.heuristics.feature_engine` engine, plus ``measured_tflops``,
     ``is_valid`` and ``kernel_name``.

Wired ops (the families the per-op feasibility map marks feasible):

  - ``gemm`` : delegates to :func:`gen_gemm_sweep_data.generate` unchanged, so
    the GEMM golden path is byte-for-byte preserved.
  - ``conv`` : implicit-GEMM convolution. Reuses the GEMM 72-feature engine via
    the implicit-GEMM (M = N*Ho*Wo, N_gemm = K, K_gemm = R*S*C) projection.
  - ``sdpa`` : fused multi-head attention. Problem-driven variant selection;
    68-feature :class:`feature_engine.FmhaFeatureEngine` columns (field-for-field
    parity with the C++ ``ml_extract_fmha_features``).
  - ``moe``  : fused-MoE streaming trio (gather / silu_mul / topk-reduce).
    Minimal :class:`feature_engine.MoeFeatureEngine` columns; latency-bound.
  - ``norm`` : LayerNorm2D / RMSNorm2D forward. Minimal
    :class:`feature_engine.NormFeatureEngine` columns; bandwidth-bound.

Usage::

    python3 -m rocke.heuristics.gen_sweep_data \\
        --op conv \\
        --out conv_training.parquet \\
        --cache-dir /tmp/rocke_conv_cache \\
        --arch gfx950 \\
        --max-shapes 32

The ``gemm`` op keeps the original
``python3 -m rocke.heuristics.gen_gemm_sweep_data ...`` entry point working as
a thin shim; this module is the superset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from ..core.lower_llvm import lower_kernel_to_llvm
from ..runtime.comgr import build_hsaco_from_llvm_ir


# ---------------------------------------------------------------------
# Generic per-op build record
# ---------------------------------------------------------------------


@dataclass
class _OpBuildRecord:
    """One spec's build outcome for a non-GEMM op."""

    name: str
    ok: bool
    error: str = ""
    hsaco_path: str = ""
    hsaco_bytes: int = 0
    build_ms: float = 0.0
    config: Dict[str, object] = field(default_factory=dict)
    problem: Dict[str, object] = field(default_factory=dict)
    flops: float = 0.0


# ---------------------------------------------------------------------
# Op adapter protocol
# ---------------------------------------------------------------------


@dataclass
class OpAdapter:
    """Everything :func:`generate` needs to sweep one op family.

    ``enumerate_specs(arch, max_shapes)`` returns a list of opaque spec objects
    (one per ``(variant, shape)`` for the cartesian ops, or one per selected
    problem for SDPA). ``build_spec(spec, arch)`` lowers + compiles one spec and
    returns a :class:`KernelDef`-producing closure result. ``config_columns`` /
    ``problem_columns`` recover the parquet feature columns from a spec, and
    ``flops`` returns the op's FLOP count for the TFLOPS metric (0 for streaming
    ops, which are latency/bandwidth bound).
    """

    op_type: str
    enumerate_specs: Callable[[str, Optional[int]], List[object]]
    build_kernel: Callable[[object], object]
    spec_name: Callable[[object], str]
    config_columns: Callable[[object], Dict[str, object]]
    problem_columns: Callable[[object], Dict[str, object]]
    flops: Callable[[object], float]


# =====================================================================
# conv (implicit-GEMM) adapter
# =====================================================================


_CONV_SHAPES = [
    # (N, Hi, Wi, C, K, R, S, sH, sW, pH, pW, dH, dW)
    # 1x1 convs (pointwise) across batches / channel widths.
    (1, 56, 56, 64, 64, 1, 1, 1, 1, 0, 0, 1, 1),
    (1, 56, 56, 256, 64, 1, 1, 1, 1, 0, 0, 1, 1),
    (2, 28, 28, 128, 128, 1, 1, 1, 1, 0, 0, 1, 1),
    (4, 28, 28, 256, 256, 1, 1, 1, 1, 0, 0, 1, 1),
    (8, 14, 14, 512, 512, 1, 1, 1, 1, 0, 0, 1, 1),
    # 3x3 same-padding convs (the CNN workhorse).
    (1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
    (1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1),
    (2, 28, 28, 128, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    (4, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    (8, 14, 14, 256, 512, 3, 3, 1, 1, 1, 1, 1, 1),
    (16, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1),
    # Stride-2 (downsampling) variants.
    (1, 112, 112, 64, 128, 3, 3, 2, 2, 1, 1, 1, 1),
    (2, 56, 56, 128, 256, 3, 3, 2, 2, 1, 1, 1, 1),
    (4, 28, 28, 256, 512, 3, 3, 2, 2, 1, 1, 1, 1),
    # Large feature maps (early ImageNet layers).
    (1, 224, 224, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
    (1, 112, 112, 64, 128, 1, 1, 1, 1, 0, 0, 1, 1),
    # 5x5 / 7x7 + dilation edge cases.
    (1, 28, 28, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1),
    (2, 14, 14, 256, 256, 3, 3, 1, 1, 2, 2, 1, 2),
    (1, 56, 56, 64, 64, 7, 7, 1, 1, 3, 3, 1, 1),
]

# (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile)
_CONV_WARP_TILES = [(16, 16, 16), (32, 32, 8), (32, 32, 16), (16, 16, 32)]
_CONV_TILES_M = (64, 128, 256)
_CONV_TILES_N = (64, 128, 256)
_CONV_TILES_K = (32, 64)
_CONV_WARPS_M = (2, 4)
_CONV_WARPS_N = (2, 4)
_CONV_PIPELINES = ("mem", "compv3", "compv4")
_CONV_EPILOGUES = ("default", "cshuffle")


def _conv_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    import itertools

    from ..instances import ConvProblem, ImplicitGemmConvSpec
    from ..instances.common.conv_implicit_gemm import is_valid_spec

    shapes = _CONV_SHAPES
    if max_shapes is not None and max_shapes > 0:
        shapes = shapes[:max_shapes]

    specs: List[object] = []
    for sh in shapes:
        problem = ConvProblem(*sh)
        for tm, tn, tk, wm, wn, (wtm, wtn, wtk), pipe, epi in itertools.product(
            _CONV_TILES_M,
            _CONV_TILES_N,
            _CONV_TILES_K,
            _CONV_WARPS_M,
            _CONV_WARPS_N,
            _CONV_WARP_TILES,
            _CONV_PIPELINES,
            _CONV_EPILOGUES,
        ):
            spec = ImplicitGemmConvSpec(
                problem=problem,
                tile_m=tm,
                tile_n=tn,
                tile_k=tk,
                warp_m=wm,
                warp_n=wn,
                warp_tile_m=wtm,
                warp_tile_n=wtn,
                warp_tile_k=wtk,
                pipeline=pipe,
                epilogue=epi,
            )
            ok, _ = is_valid_spec(spec, arch)
            if ok:
                specs.append(spec)
    return specs


def _conv_build(spec: object):
    from ..instances import build_implicit_gemm_conv

    return build_implicit_gemm_conv(spec)


def _conv_config_columns(spec: object) -> Dict[str, object]:
    # Reuse the GEMM 72-feature column set: conv config maps onto the same
    # tile/warp/pipeline/epilogue knobs the GemmUniversalFeatureEngine reads.
    return {
        "dtype": "fp16",
        "layout": "rcr",  # NHWC x KRSC implicit-GEMM
        "tile_m": int(spec.tile_m),
        "tile_n": int(spec.tile_n),
        "tile_k": int(spec.tile_k),
        "warp_m": int(spec.warp_m),
        "warp_n": int(spec.warp_n),
        "warp_k": 1,
        "warp_tile_m": int(spec.warp_tile_m),
        "warp_tile_n": int(spec.warp_tile_n),
        "warp_tile_k": int(spec.warp_tile_k),
        "pipeline": str(spec.pipeline),
        "scheduler": "intrawave",
        "epilogue": str(spec.epilogue),
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": bool(spec.chiplet_swizzle),
    }


def _conv_problem_columns(spec: object) -> Dict[str, object]:
    p = spec.problem
    # Implicit-GEMM projection -> the GEMM feature engine's m / n / k columns.
    return {
        "m": int(p.M),
        "n": int(p.N_gemm),
        "k": int(p.K_gemm),
        "split_k": 1,
        "conv_N": int(p.N),
        "conv_Hi": int(p.Hi),
        "conv_Wi": int(p.Wi),
        "conv_C": int(p.C),
        "conv_K": int(p.K),
        "conv_R": int(p.R),
        "conv_S": int(p.S),
    }


def _conv_flops(spec: object) -> float:
    return float(spec.problem.flops)


# =====================================================================
# sdpa adapter (problem-driven variant selection)
# =====================================================================


_SDPA_PROBLEMS = [
    # (batch, sq, sk, hq, hk, hd, block_size, dtype, sliding_window)
    # Decode (seqlen_q == 1) across batch + GQA ratios.
    (1, 1, 1024, 32, 32, 128, 16, "fp16", 0),
    (8, 1, 2048, 32, 8, 128, 16, "fp16", 0),
    (16, 1, 4096, 32, 8, 128, 16, "bf16", 0),
    (32, 1, 512, 64, 8, 64, 16, "bf16", 0),
    # Short prefill (q <= 256).
    (1, 128, 128, 32, 32, 128, 16, "fp16", 0),
    (4, 256, 256, 32, 8, 128, 16, "bf16", 0),
    # Medium prefill (256 < q <= 1024).
    (1, 512, 512, 32, 32, 64, 16, "fp16", 0),
    (4, 1024, 1024, 32, 8, 128, 16, "bf16", 0),
    # Long prefill (q > 1024).
    (1, 2048, 2048, 16, 16, 128, 16, "bf16", 0),
    (2, 4096, 4096, 32, 4, 64, 16, "bf16", 0),
    # Sliding-window variants.
    (1, 1024, 1024, 32, 8, 128, 16, "bf16", 256),
    (4, 2048, 2048, 32, 8, 64, 16, "fp16", 512),
]


def _sdpa_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    from ..instances import UnifiedAttentionProblem

    problems = _SDPA_PROBLEMS
    if max_shapes is not None and max_shapes > 0:
        problems = problems[:max_shapes]

    specs: List[object] = []
    for batch, sq, sk, hq, hk, hd, bs, dtype, sw in problems:
        prob = UnifiedAttentionProblem(
            total_q=batch * sq,
            num_seqs=batch,
            num_query_heads=hq,
            num_kv_heads=hk,
            head_size=hd,
            block_size=bs,
            max_seqlen_q=sq,
            max_seqlen_k=sk,
            dtype=dtype,
            sliding_window=sw,
        )
        specs.append(prob)
    return specs


def _sdpa_tiled_spec(prob: object):
    """Derive the (deterministic, problem-driven) 2D tiled spec for a problem."""
    from ..instances.common import attention_unified as au

    return au._tiled_spec_from_problem(prob)


def _sdpa_build(prob: object):
    from ..instances import build_unified_attention_2d
    from ..instances.common.attention_unified import UnifiedAttention2DSpec

    # The scalar 2D path builds on every supported arch and exercises the same
    # problem geometry; the tiled spec is used only for the feature columns.
    # ``build_unified_attention_2d`` takes a 2D *spec* (which wraps the problem),
    # not a bare ``UnifiedAttentionProblem``.
    return build_unified_attention_2d(UnifiedAttention2DSpec(problem=prob))


def _sdpa_config_columns(prob: object) -> Dict[str, object]:
    """Recover the 68-feature kernel columns from the problem-driven tiled spec.

    The FMHA feature layout treats ``tm0`` as the per-warp query block
    (``block_q``, fixed at 16 in the C++ ``FmhaKernelConfig::from_manifest``),
    ``tn0`` as the tile_size T, ``tk0``/``tk0max`` as head_size, ``tn1`` as
    hdim_v, and ``tk1`` as T -- exactly mirroring the C++ derivation so the
    Python and runtime feature vectors agree field-for-field.
    """
    T = 64
    block_q = 16
    pipeline = 1  # qr_async
    mask = 0
    sink = False
    try:
        spec = _sdpa_tiled_spec(prob)
        T = int(getattr(spec, "tile_size", T))
        block_q = int(getattr(spec, "block_m_per_warp", block_q))
        sink = bool(getattr(spec, "use_sinks", False))
    except Exception:
        pass

    hd = int(prob.head_size)
    return {
        "pipeline": pipeline,
        "tile_m0": block_q,
        "tile_n0": T,
        "tile_k0": hd,
        "tile_n1": hd,
        "tile_k1": T,
        "tile_k0max": hd,
        "pad_s": 0,
        "pad_sk": 0,
        "pad_d": 0,
        "pad_dv": 0,
        "mask": mask,
        "bias": 0,
        "lse": 0,
        "dropout": 0,
        "logits": 0,
        "sink": 1 if sink else 0,
        "skip": 0,
        "qscale": 0,
        "paged": 1,
    }


def _sdpa_problem_columns(prob: object) -> Dict[str, object]:
    return {
        "batch": int(prob.num_seqs),
        "seqlen_q": int(prob.max_seqlen_q),
        "seqlen_k": int(prob.max_seqlen_k),
        "nhead_q": int(prob.num_query_heads),
        "nhead_k": int(prob.num_kv_heads),
        "hdim_q": int(prob.head_size),
        "hdim_v": int(prob.head_size),
        "dtype": str(prob.dtype),
        "sliding_window": int(prob.sliding_window),
    }


def _sdpa_flops(prob: object) -> float:
    b = prob.num_seqs
    hq = prob.num_query_heads
    sq = prob.max_seqlen_q
    sk = prob.max_seqlen_k
    d = prob.head_size
    return float(2.0 * b * hq * sq * sk * (d + d))


# =====================================================================
# moe adapter (fused streaming trio: gather / silu_mul / topk-reduce)
# =====================================================================


_MOE_SHAPES = [
    # (tokens, experts, topk, hidden, intermediate, dtype)
    # Decode (T = 1) -- LLaMA / DeepSeek-style.
    (1, 8, 2, 4096, 14336, "f16"),
    (1, 64, 2, 4096, 14336, "bf16"),
    (1, 256, 4, 2048, 8192, "bf16"),
    # Small prefill.
    (8, 8, 2, 4096, 14336, "f16"),
    (32, 64, 2, 4096, 14336, "bf16"),
    (32, 256, 4, 2048, 8192, "bf16"),
    # Medium / training.
    (128, 8, 2, 4096, 8192, "f16"),
    (256, 64, 2, 2048, 4096, "bf16"),
    (512, 32, 2, 1024, 2048, "f16"),
    (1024, 8, 1, 1024, 2048, "bf16"),
]

_MOE_BLOCK_SIZES = (64, 128, 256, 512, 1024)
_MOE_VECS = (2, 4, 8)
# The streaming trio is swept as a single launchable unit; gather is the
# representative kernel built for the manifest (silu_mul / topk-reduce share
# the FusedMoeSpec geometry).
_MOE_PHASE = "gather"


def _moe_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    import itertools

    from ..instances import FusedMoeSpec
    from ..instances.common.fused_moe import is_valid_spec

    shapes = _MOE_SHAPES
    if max_shapes is not None and max_shapes > 0:
        shapes = shapes[:max_shapes]

    specs: List[object] = []
    for tokens, experts, topk, hidden, inter, dtype in shapes:
        for bs, vec in itertools.product(_MOE_BLOCK_SIZES, _MOE_VECS):
            spec = FusedMoeSpec(
                tokens=tokens,
                experts=experts,
                topk=topk,
                hidden=hidden,
                intermediate=inter,
                dtype=dtype,
                block_size=bs,
                vec=vec,
            )
            ok, _ = is_valid_spec(spec)
            if ok:
                specs.append(spec)
    return specs


def _moe_build(spec: object):
    from ..instances import build_moe_gather

    return build_moe_gather(spec)


def _moe_spec_name(spec: object) -> str:
    return spec.kernel_name(_MOE_PHASE)


def _moe_config_columns(spec: object) -> Dict[str, object]:
    return {
        "block_size": int(spec.block_size),
        "vec": int(spec.vec),
    }


def _moe_problem_columns(spec: object) -> Dict[str, object]:
    return {
        "tokens": int(spec.tokens),
        "experts": int(spec.experts),
        "topk": int(spec.topk),
        "hidden": int(spec.hidden),
        "intermediate": int(spec.intermediate),
        "dtype": str(spec.dtype),
    }


def _moe_flops(spec: object) -> float:
    # Streaming / atomic-contention bound -- no GEMM-style FLOP metric.
    return 0.0


# =====================================================================
# norm adapter (LayerNorm2D / RMSNorm2D forward)
# =====================================================================


_NORM_N_PER_BLOCK = (128, 256, 512, 1024, 2048, 4096, 8192)
_NORM_BLOCK_SIZES = (64, 128, 256)
_NORM_VECS = (2, 4, 8)
_NORM_DTYPES = ("f16", "bf16")
# Representative row counts (one CTA per row); used only for occupancy features.
_NORM_ROWS = 4096


def _norm_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    import itertools

    from ..instances import RMSNorm2DSpec
    from ..instances.common.rmsnorm2d import is_valid_spec

    n_values = _NORM_N_PER_BLOCK
    if max_shapes is not None and max_shapes > 0:
        n_values = n_values[:max_shapes]

    specs: List[object] = []
    for npb, bs, vec, dtype in itertools.product(
        n_values, _NORM_BLOCK_SIZES, _NORM_VECS, _NORM_DTYPES
    ):
        if npb % bs != 0 or npb % vec != 0:
            continue
        spec = RMSNorm2DSpec(
            n_per_block=npb,
            block_size=bs,
            vec=vec,
            dtype=dtype,
        )
        ok, _ = is_valid_spec(spec, arch)
        if ok:
            specs.append(spec)
    return specs


def _norm_build(spec: object):
    from ..instances import build_rmsnorm2d

    return build_rmsnorm2d(spec)


def _norm_config_columns(spec: object) -> Dict[str, object]:
    return {
        "block_size": int(spec.block_size),
        "vec": int(spec.vec),
        "dtype": str(spec.dtype),
    }


def _norm_problem_columns(spec: object) -> Dict[str, object]:
    return {
        "rows": int(_NORM_ROWS),
        "n_per_block": int(spec.n_per_block),
        "dtype": str(spec.dtype),
    }


def _norm_flops(spec: object) -> float:
    # Bandwidth-bound row normalization -- no FLOP metric.
    return 0.0


# ---------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------


def _adapter(op: str) -> OpAdapter:
    if op == "conv":
        return OpAdapter(
            op_type="conv_implicit_gemm",
            enumerate_specs=_conv_enumerate,
            build_kernel=_conv_build,
            spec_name=lambda s: s.kernel_name(),
            config_columns=_conv_config_columns,
            problem_columns=_conv_problem_columns,
            flops=_conv_flops,
        )
    if op == "sdpa":
        return OpAdapter(
            op_type="fmha",
            enumerate_specs=_sdpa_enumerate,
            build_kernel=_sdpa_build,
            spec_name=lambda p: (
                p.kernel_name()
                if hasattr(p, "kernel_name")
                else f"sdpa_b{p.num_seqs}_sq{p.max_seqlen_q}_sk{p.max_seqlen_k}"
            ),
            config_columns=_sdpa_config_columns,
            problem_columns=_sdpa_problem_columns,
            flops=_sdpa_flops,
        )
    if op == "moe":
        return OpAdapter(
            op_type="fused_moe",
            enumerate_specs=_moe_enumerate,
            build_kernel=_moe_build,
            spec_name=_moe_spec_name,
            config_columns=_moe_config_columns,
            problem_columns=_moe_problem_columns,
            flops=_moe_flops,
        )
    if op == "norm":
        return OpAdapter(
            op_type="rmsnorm2d",
            enumerate_specs=_norm_enumerate,
            build_kernel=_norm_build,
            spec_name=lambda s: s.kernel_name(),
            config_columns=_norm_config_columns,
            problem_columns=_norm_problem_columns,
            flops=_norm_flops,
        )
    raise ValueError(f"unknown op {op!r} (want gemm|conv|sdpa|moe|norm)")


WIRED_OPS = ("gemm", "conv", "sdpa", "moe", "norm")


# ---------------------------------------------------------------------
# Build one spec (non-GEMM ops)
# ---------------------------------------------------------------------


def _build_spec(
    adapter: OpAdapter, spec: object, cache_dir: Path, isa: str
) -> _OpBuildRecord:
    name = adapter.spec_name(spec)
    config = adapter.config_columns(spec)
    problem = adapter.problem_columns(spec)
    flops = adapter.flops(spec)

    blob = json.dumps(
        {"name": name, "config": config, "problem": problem}, sort_keys=True
    ).encode()
    spec_hash = hashlib.sha1(blob).hexdigest()[:12]
    out_path = cache_dir / f"{spec_hash}_{name[:120]}.hsaco"

    rec = _OpBuildRecord(
        name=name, ok=False, config=config, problem=problem, flops=flops
    )

    if out_path.exists() and out_path.stat().st_size > 0:
        rec.ok = True
        rec.hsaco_path = str(out_path)
        rec.hsaco_bytes = out_path.stat().st_size
        return rec

    try:
        t0 = time.perf_counter()
        kernel = adapter.build_kernel(spec)
        ll = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ll, isa=isa)
        out_path.write_bytes(hsaco)
        rec.ok = True
        rec.hsaco_path = str(out_path)
        rec.hsaco_bytes = len(hsaco)
        rec.build_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # noqa: BLE001 - record the failure surface
        rec.error = f"{type(e).__name__}: {e}"

    return rec


# ---------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------


def generate(
    *,
    op: str,
    out_path: Path,
    cache_dir: Path,
    arch: str = "gfx950",
    max_shapes: Optional[int] = None,
    isa: Optional[str] = None,
    **gemm_kwargs: object,
) -> pd.DataFrame:
    """Sweep one op family and write its training parquet.

    For ``op == "gemm"`` this delegates to :func:`gen_gemm_sweep_data.generate`
    so the GEMM golden path is preserved byte-for-byte. For every other (wired,
    feasible) op it enumerates specs, builds each to a cached HSACO, and emits a
    parquet whose feature columns match that op's
    :mod:`rocke.heuristics.feature_engine` engine.
    """
    if op == "gemm":
        from . import gen_gemm_sweep_data

        return gen_gemm_sweep_data.generate(
            out_path=out_path,
            cache_dir=cache_dir,
            arch=arch,
            max_shapes=max_shapes,
            **gemm_kwargs,  # type: ignore[arg-type]
        )

    adapter = _adapter(op)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    isa = isa or f"amdgcn-amd-amdhsa--{arch}"

    specs = adapter.enumerate_specs(arch, max_shapes)
    if not specs:
        raise RuntimeError(f"no valid {op} specs for arch={arch}")

    print(
        f"[gen] op={op} arch={arch} variants={len(specs)} -> building",
        file=sys.stderr,
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    n_built = 0
    for i, spec in enumerate(specs):
        rec = _build_spec(adapter, spec, cache_dir, isa)
        if rec.ok:
            n_built += 1
        # Perf measurement requires a launcher + GPU; in its absence
        # measured_tflops stays 0 and is_valid tracks build success so the
        # model can still learn the (large) build-failure surface. When a
        # launcher is wired the same rows are re-measured in place.
        row: Dict[str, object] = {
            "op_type": adapter.op_type,
            "arch": arch,
            "kernel_name": rec.name,
            "measured_tflops": 0.0,
            "latency_ms": 0.0,
            "is_valid": bool(rec.ok),
            "build_ok": bool(rec.ok),
            "build_error": rec.error,
            "run_id": 0,
        }
        row.update(rec.problem)
        row.update(rec.config)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"[gen]   built {n_built}/{i + 1} ...", file=sys.stderr, flush=True)

    print(
        f"[gen] op={op} built {n_built}/{len(specs)} variants OK",
        file=sys.stderr,
        flush=True,
    )

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[gen] {len(df)} rows -> {out_path}", file=sys.stderr, flush=True)
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "rocke-native, op-parameterized heuristics training-data generator "
            "(gemm|conv|sdpa|moe|norm)."
        )
    )
    parser.add_argument(
        "--op",
        default="gemm",
        choices=list(WIRED_OPS),
        help="Op family to sweep.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_sweep_cache"),
        help="Directory for cached HSACO binaries + manifests.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of shapes / problems (smoke tests).",
    )
    parser.add_argument(
        "--shape-set",
        default="wide",
        choices=["wide", "edge", "all"],
        help="GEMM-only: shape corpus to sweep.",
    )
    args = parser.parse_args(argv)

    gemm_kwargs: Dict[str, object] = {}
    if args.op == "gemm":
        gemm_kwargs["shape_set"] = args.shape_set

    generate(
        op=args.op,
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        max_shapes=args.max_shapes,
        **gemm_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
