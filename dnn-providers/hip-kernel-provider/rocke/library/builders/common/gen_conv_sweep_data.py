#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convolution (implicit-GEMM) heuristics training-data generator.

Library-side entry point for the conv sweep; the platform
:mod:`rocke.heuristics.gen_sweep_data` no longer carries the conv adapter.
This module owns the conv problem corpus, the variant grid, and the
OpAdapter, then calls :func:`rocke.heuristics.gen_sweep_data.generate` as a
service. Exact analogue of :mod:`builders.common.gen_sdpa_sweep_data`.

The feature columns are unchanged: conv reuses the GEMM 72-feature engine
via the implicit-GEMM projection (M = N*Ho*Wo, N_gemm = K, K_gemm = R*S*C).

Usage::

    python3 -m builders.common.gen_conv_sweep_data \\
        --out conv_training.parquet \\
        --cache-dir /tmp/rocke_conv_cache \\
        --arch gfx950 \\
        --max-shapes 32
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rocke.heuristics.gen_sweep_data import OpAdapter, generate


# =====================================================================
# conv (implicit-GEMM) problem corpus + variant grid
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
    from kernels.common.conv_implicit_gemm import is_valid_spec

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
    from kernels import build_implicit_gemm_conv

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
# Public adapter factory
# =====================================================================


def build_conv_adapter() -> OpAdapter:
    """Construct the conv OpAdapter for use with ``generate()``."""
    return OpAdapter(
        op_type="conv_implicit_gemm",
        enumerate_specs=_conv_enumerate,
        build_kernel=_conv_build,
        spec_name=lambda s: s.kernel_name(),
        config_columns=_conv_config_columns,
        problem_columns=_conv_problem_columns,
        flops=_conv_flops,
    )


# =====================================================================
# CLI
# =====================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convolution (implicit-GEMM) heuristics training-data generator. "
            "Library entry point - calls rocke.heuristics.gen_sweep_data.generate() "
            "with the conv adapter."
        )
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_conv_cache"),
        help="Directory for cached HSACO binaries + manifests.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of conv problems (smoke tests).",
    )
    args = parser.parse_args(argv)

    generate(
        op="conv",
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        max_shapes=args.max_shapes,
        adapter=build_conv_adapter(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
