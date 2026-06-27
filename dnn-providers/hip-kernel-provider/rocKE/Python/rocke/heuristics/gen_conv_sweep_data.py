#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
rocke-native conv heuristics training-data generator.

Analogous to gen_gemm_sweep_data.py: sweeps the implicit-GEMM conv variant
space over the same shape corpus used by sample_shapes_conv.py /
augment_coverage_conv.py, builds each (config, shape) pair to a cached HSACO,
and writes a training parquet that train.py / GroupedConvFeatureEngine consume.

This is the rocKE replacement for the old ConvCandidateSweep.cpp binary. Instead
of a pre-built C++ executable it drives the rocKE Python build pipeline:

  1. Enumerate a *shape corpus* from generate_coverage_conv (G=1 shapes only —
     ImplicitGemmConvSpec does not yet support grouped / depthwise conv).
  2. Enumerate *kernel configs* as (tile_m, tile_n, tile_k, warp, pipeline,
     epilogue) tuples, filtered by is_valid_spec(arch).  Config validity is
     shape-independent so this enumeration happens once.
  3. For each (config, shape) pair: construct an ImplicitGemmConvSpec, build
     to a cached HSACO via build_implicit_gemm_conv + lower_kernel_to_llvm +
     build_hsaco_from_llvm_ir.
  4. Write a training parquet whose shape + config columns match what
     GroupedConvFeatureEngine.extract_batch() and train.py expect.

Timing: measured_tflops is left at 0.0. A conv-aware launcher pass (analogous
to sweep_bench.sweep_run for GEMM) will fill in actual throughput figures once
rocKE has a conv timing harness.  The is_valid / build_ok columns track build
success in the interim so the model can learn the build-failure surface.

gen_sweep_data.py --op conv delegates to generate() here (same relationship as
--op gemm delegates to gen_gemm_sweep_data.generate()).

Usage:
    python3 -m rocke.heuristics.gen_conv_sweep_data \\
        --out training.parquet \\
        --cache-dir /tmp/rocke_conv_cache \\
        --arch gfx950 \\
        --shape-set wide

Output parquet columns consumed by GroupedConvFeatureEngine / train.py:
    N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w  (problem)
    tile_m, tile_n, tile_k, pipeline                              (config)
    measured_tflops, latency_us, is_valid                         (targets)
    op_type, arch, kernel_name, build_ok, build_error, run_id     (metadata)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from .generate_coverage_conv import generate_wide_shapes, generate_edge_shapes


# ---------------------------------------------------------------------------
# Tile / warp config grid (mirrors gen_sweep_data._CONV_* constants)
# ---------------------------------------------------------------------------

_TILES_M    = (64, 128, 256)
_TILES_N    = (64, 128, 256)
_TILES_K    = (32, 64)
_WARPS_M    = (2, 4)
_WARPS_N    = (2, 4)
_WARP_TILES = ((16, 16, 16), (32, 32, 8), (32, 32, 16), (16, 16, 32))
_PIPELINES  = ("mem", "compv3", "compv4")
_EPILOGUES  = ("default", "cshuffle")


# ---------------------------------------------------------------------------
# Shape corpus
# ---------------------------------------------------------------------------

def _filter_g1(raw: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Drop G>1 shapes and repack to (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW).

    Input tuples are in generate_coverage_conv order:
      (N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w)
    G>1 (grouped / depthwise) shapes are dropped — ImplicitGemmConvSpec does
    not yet support grouped conv.
    """
    out = []
    for s in raw:
        N, G, C, K, Hi, Wi, Y, X, sH, sW, pH, pW = s
        if G != 1:
            continue
        out.append((N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW))
    return out


def generate_shape_corpus(shape_set: str) -> List[Tuple[int, ...]]:
    """Return G=1 conv shapes from the standard coverage corpus.

    ``shape_set`` is one of ``"wide"``, ``"edge"``, or ``"all"``.
    """
    if shape_set == "wide":
        raw = generate_wide_shapes()
    elif shape_set == "edge":
        raw = generate_edge_shapes()
    elif shape_set == "all":
        raw = sorted(set(generate_wide_shapes()) | set(generate_edge_shapes()))
    else:
        raise ValueError(f"unknown shape_set {shape_set!r} (want wide|edge|all)")
    return _filter_g1(raw)


def load_shapes_from_csvs(paths: Sequence[Path]) -> List[Tuple[int, ...]]:
    """Load and deduplicate shapes from one or more CSV files.

    Accepts any CSV produced by generate_coverage_conv.py,
    sample_shapes_conv.py, or augment_coverage_conv.py — all share the same
    13-column format:
      N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w, direction

    G>1 shapes are dropped (same rule as generate_shape_corpus).
    """
    import csv

    seen: set[Tuple[int, ...]] = set()
    out: List[Tuple[int, ...]] = []
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                N  = int(row["N"]);  G  = int(row["G"])
                C  = int(row["C"]);  K  = int(row["K"])
                Hi = int(row["Hi"]); Wi = int(row["Wi"])
                Y  = int(row["Y"]);  X  = int(row["X"])
                sH = int(row["stride_h"]); sW = int(row["stride_w"])
                pH = int(row["pad_h"]);    pW = int(row["pad_w"])
                if G != 1:
                    continue
                t = (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW)
                if t not in seen:
                    seen.add(t)
                    out.append(t)
    return out


# ---------------------------------------------------------------------------
# Variant enumeration (shape-independent)
# ---------------------------------------------------------------------------

# Each config is (tile_m, tile_n, tile_k, warp_m, warp_n,
#                 warp_tile_m, warp_tile_n, warp_tile_k, pipeline, epilogue).
Config = Tuple[int, int, int, int, int, int, int, int, str, str]


def enumerate_configs(arch: str = "gfx950") -> List[Config]:
    """Return all arch-valid (tile, warp, pipeline, epilogue) configs.

    Validity is checked against a dummy 8x8 / 64-channel problem — tile / warp
    geometry and LDS budget are shape-independent, so the set of valid configs
    is the same for every conv shape.
    """
    from ..instances import ImplicitGemmConvSpec, ConvProblem
    from ..instances.common.conv_implicit_gemm import is_valid_spec

    dummy = ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=1, X=1)

    configs: List[Config] = []
    seen: set[str] = set()
    for tm, tn, tk, wm, wn, (wtm, wtn, wtk), pipe, epi in itertools.product(
        _TILES_M, _TILES_N, _TILES_K,
        _WARPS_M, _WARPS_N,
        _WARP_TILES,
        _PIPELINES,
        _EPILOGUES,
    ):
        spec = ImplicitGemmConvSpec(
            problem=dummy,
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_m=wtm, warp_tile_n=wtn, warp_tile_k=wtk,
            pipeline=pipe,
            epilogue=epi,
        )
        ok, _ = is_valid_spec(spec, arch)
        if not ok:
            continue
        key = f"{tm}x{tn}x{tk}_w{wm}x{wn}_a{wtm}x{wtn}x{wtk}_{pipe}_{epi}"
        if key not in seen:
            seen.add(key)
            configs.append((tm, tn, tk, wm, wn, wtm, wtn, wtk, pipe, epi))

    return configs


# ---------------------------------------------------------------------------
# Build one (shape, config) pair
# ---------------------------------------------------------------------------

class ConvBuildRecord(NamedTuple):
    name: str
    ok: bool
    hsaco_path: str = ""
    hsaco_bytes: int = 0
    build_ms: float = 0.0
    error: str = ""


def _build_one(
    shape: Tuple[int, ...],
    config: Config,
    cache_dir: Path,
    isa: str,
) -> ConvBuildRecord:
    from ..instances import ImplicitGemmConvSpec, ConvProblem
    from ..instances.common.conv_implicit_gemm import build_implicit_gemm_conv
    from ..core.lower_llvm import lower_kernel_to_llvm
    from ..runtime.comgr import build_hsaco_from_llvm_ir

    N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW = shape
    tm, tn, tk, wm, wn, wtm, wtn, wtk, pipe, epi = config

    problem = ConvProblem(N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=Y, X=X,
                          sH=sH, sW=sW, pH=pH, pW=pW)
    spec = ImplicitGemmConvSpec(
        problem=problem,
        tile_m=tm, tile_n=tn, tile_k=tk,
        warp_m=wm, warp_n=wn,
        warp_tile_m=wtm, warp_tile_n=wtn, warp_tile_k=wtk,
        pipeline=pipe,
        epilogue=epi,
    )
    name = spec.kernel_name()

    # Cache key combines config + shape so kernels for different shapes that
    # happen to share a name (edge case) get distinct HSACO files.
    blob = json.dumps(
        {"name": name, "shape": list(shape)}, sort_keys=True
    ).encode()
    spec_hash = hashlib.sha1(blob).hexdigest()[:12]
    out_path = cache_dir / f"{spec_hash}_{name[:120]}.hsaco"

    if out_path.exists() and out_path.stat().st_size > 0:
        return ConvBuildRecord(
            name=name, ok=True,
            hsaco_path=str(out_path),
            hsaco_bytes=out_path.stat().st_size,
        )

    try:
        t0 = time.perf_counter()
        kernel = build_implicit_gemm_conv(spec)
        ll = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ll, isa=isa)
        out_path.write_bytes(hsaco)
        return ConvBuildRecord(
            name=name, ok=True,
            hsaco_path=str(out_path),
            hsaco_bytes=len(hsaco),
            build_ms=(time.perf_counter() - t0) * 1000.0,
        )
    except Exception as e:  # noqa: BLE001 — record the build-failure surface
        return ConvBuildRecord(name=name, ok=False, error=f"{type(e).__name__}: {e}")


def _build_batch(
    pairs: List[Tuple],
    cache_dir: Path,
    isa: str,
) -> List[ConvBuildRecord]:
    """Build a batch of (shape, config) pairs in a single subprocess worker."""
    return [_build_one(s, c, cache_dir, isa) for s, c in pairs]


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------

def generate(
    *,
    out_path: Path,
    cache_dir: Path,
    arch: str = "gfx950",
    shape_set: str = "wide",
    shape_csvs: Optional[Sequence[Path]] = None,
    max_shapes: Optional[int] = None,
    isa: Optional[str] = None,
) -> pd.DataFrame:
    """Build the (config × shape) grid and write the training parquet.

    One row per (config, shape) pair. measured_tflops is 0.0 until a
    conv-aware launcher pass fills in actual throughput figures.

    ``shape_csvs``: if provided, shapes are loaded from these CSV files
    (output of generate_coverage_conv.py / sample_shapes_conv.py /
    augment_coverage_conv.py) and ``shape_set`` is ignored.
    """
    if shape_csvs:
        shapes = load_shapes_from_csvs([Path(p) for p in shape_csvs])
    else:
        shapes = generate_shape_corpus(shape_set)
    if max_shapes is not None and max_shapes > 0:
        shapes = shapes[:max_shapes]

    configs = enumerate_configs(arch=arch)
    if not configs:
        raise RuntimeError(f"no valid conv configs for arch={arch}")

    total = len(shapes) * len(configs)
    print(
        f"[gen] arch={arch} shapes={len(shapes)} configs={len(configs)} "
        f"-> {total} (shape, config) pairs",
        file=sys.stderr, flush=True,
    )

    isa_str = isa or f"amdgcn-amd-amdhsa--{arch}"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    n_built = 0

    _BATCH = 500  # pairs per subprocess worker — limits crash blast radius

    def _run_batch_isolated(pairs: List[Tuple]) -> List[ConvBuildRecord]:
        # Each batch runs in a fresh subprocess; a comgr SIGSEGV kills only that worker.
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
            try:
                return ex.submit(_build_batch, pairs, cache_dir, isa_str).result(timeout=300)
            except concurrent.futures.process.BrokenProcessPool:
                # Worker crashed — mark the whole batch as failed.
                return [ConvBuildRecord(name="<crashed>", ok=False, error="comgr SIGSEGV")
                        for _ in pairs]

    all_pairs = list(itertools.product(shapes, configs))
    for batch_start in range(0, total, _BATCH):
        batch = all_pairs[batch_start: batch_start + _BATCH]
        results = _run_batch_isolated(batch)
        for i_rel, ((shape, config), rec) in enumerate(zip(batch, results)):
            i = batch_start + i_rel
            if rec.ok:
                n_built += 1

            N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW = shape
            tm, tn, tk, wm, wn, wtm, wtn, wtk, pipe, epi = config

            # block_size = warp_m * warp_n * wave_size (64 for all MFMA archs)
            block_size = wm * wn * 64
            rows.append({
                "op_type":      "grouped_conv",
                "dtype":        "fp16",
                "arch":         arch,
                "kernel_name":  rec.name,
                "N":        N,  "G": 1,
                "C":        C,  "K": K,
                "Hi":       Hi, "Wi": Wi,
                "Y":        Y,  "X":  X,
                "stride_h": sH, "stride_w": sW,
                "pad_h":    pH, "pad_w":    pW,
                "gemm_m_per_block": tm, "gemm_n_per_block": tn, "gemm_k_per_block": tk,
                "block_size": block_size,
                "wave_mode":  "intrawave",   # all conv pipelines are intrawave
                "has_dsb":    0,
                "has_si":     0,
                "pipeline":   pipe,
                "epilogue":   epi,
                "tflops":     0.0,
                "latency_us": 0.0,
                "is_valid":        rec.ok,
                "build_ok":        rec.ok,
                "build_error":     rec.error,
                "run_id":          0,
            })

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(
                    f"[gen]   {i + 1}/{total} pairs, {n_built} built OK",
                    file=sys.stderr, flush=True,
                )

    print(
        f"[gen] {n_built}/{total} built OK -> {out_path}",
        file=sys.stderr, flush=True,
    )

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "rocke-native conv heuristics training-data generator "
            "(builds implicit-GEMM conv kernels, emits training parquet)."
        )
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_conv_cache"),
        help="Directory for cached HSACO binaries.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")

    shape_source = parser.add_mutually_exclusive_group()
    shape_source.add_argument(
        "--shapes",
        nargs="+",
        type=Path,
        metavar="CSV",
        help=(
            "One or more shape CSVs produced by generate_coverage_conv.py, "
            "sample_shapes_conv.py, or augment_coverage_conv.py. "
            "Mutually exclusive with --shape-set."
        ),
    )
    shape_source.add_argument(
        "--shape-set",
        default="wide",
        choices=["wide", "edge", "all"],
        help="Built-in shape corpus to sweep (default: wide). Mutually exclusive with --shapes.",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of shapes (smoke tests).",
    )
    args = parser.parse_args(argv)

    generate(
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        shape_set=args.shape_set if args.shapes is None else "wide",
        shape_csvs=args.shapes,
        max_shapes=args.max_shapes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
