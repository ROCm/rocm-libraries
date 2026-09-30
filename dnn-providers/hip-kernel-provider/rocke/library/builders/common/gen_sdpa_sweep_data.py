#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA (fused multi-head attention) heuristics training-data generator.

Library-side entry point for the SDPA sweep; the platform
:mod:`rocke.heuristics.gen_sweep_data` no longer carries the sdpa adapter.
This module owns the sdpa problem corpus, the problem-driven variant
selector, and the OpAdapter, then calls
:func:`rocke.heuristics.gen_sweep_data.generate` as a service.

Usage::

    python3 -m builders.common.gen_sdpa_sweep_data \\
        --out sdpa_training.parquet \\
        --cache-dir /tmp/rocke_sdpa_cache \\
        --arch gfx950 \\
        --max-shapes 4
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rocke.heuristics.gen_sweep_data import OpAdapter, generate


# =====================================================================
# sdpa problem corpus
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
    from kernels.common.attention_unified import UnifiedAttentionProblem

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
    from kernels.common import attention_unified as au

    return au._tiled_spec_from_problem(prob)


def _sdpa_build(prob: object):
    from kernels import build_unified_attention_2d
    from kernels.common.attention_unified import UnifiedAttention2DSpec

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
# Public adapter factory
# =====================================================================


def build_sdpa_adapter() -> OpAdapter:
    """Construct the SDPA OpAdapter for use with ``generate()``."""
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


# =====================================================================
# CLI
# =====================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SDPA (fused multi-head attention) heuristics training-data generator. "
            "Library entry point — calls rocke.heuristics.gen_sweep_data.generate() "
            "with the sdpa adapter."
        )
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_sdpa_cache"),
        help="Directory for cached HSACO binaries + manifests.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of SDPA problems (smoke tests).",
    )
    args = parser.parse_args(argv)

    generate(
        op="sdpa",
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        max_shapes=args.max_shapes,
        adapter=build_sdpa_adapter(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
