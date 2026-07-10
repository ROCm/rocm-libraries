#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
rocke-native GEMM heuristics training-data generator (GEMM shim).

The op-parameterized superset lives in :mod:`rocke.heuristics.gen_sweep_data`
(``--op gemm|conv|sdpa|moe|norm``); its ``gemm`` op delegates straight to
:func:`generate` here, so this module remains the authoritative GEMM golden path
and keeps the ``python3 -m rocke.heuristics.gen_gemm_sweep_data`` CLI working
unchanged.

This is the rocke replacement for the old CK-Tile ``generate_benchmark_data.py``
/ ``generate_wide_coverage.py`` flow (CMake + ninja-built ``benchmark_gemm_*``
binaries). Instead of shelling out to pre-built CK-Tile executables, it drives
the rocke sweep ecosystem end-to-end:

  1. Enumerate a *shape corpus* (M, N, K) covering inference / training / edge
     cases (folded in from the retired ``generate_wide_coverage.py`` and
     ``generate_edge_dims.py``).
  2. Enumerate *kernel-config variants* as :class:`UniversalGemmSpec` objects via
     :func:`rocke.instances.all_dispatcher_configs` (validity-filtered by
     ``is_valid_spec``).
  3. Build every variant on the fly with
     :func:`rocke.sweep.build_all_instances` (LLVM IR -> HSACO, cached).
  4. Run each ``(variant, shape)`` through
     :func:`rocke.sweep_bench.sweep_run` to record per-shape TFLOPS +
     correctness.
  5. Write a training parquet with EXACTLY the columns
     ``feature_engine.GemmUniversalFeatureEngine`` / ``train.py`` consume.

The downstream ML pipeline (``train.py`` / ``predict.py`` / ``evaluate.py`` /
``search.py`` / ``feature_engine.py``) is unchanged -- the parquet schema this
emits is identical to what the CK-Tile path produced, so existing models and
``build_training_dataset`` keep working.

Usage:
    python3 -m rocke.heuristics.gen_gemm_sweep_data \\
        --out training.parquet \\
        --cache-dir /tmp/rocke_sweep_cache \\
        --arch gfx950 \\
        --shape-set wide \\
        --max-shapes 64

The config columns are recovered directly from each
:class:`UniversalGemmSpec` (which is authoritative), not by parsing the kernel
name, so the parquet is always self-consistent with the kernels that were built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..instances import (
    UniversalGemmSpec,
    all_dispatcher_configs,
)
from .. import sweep
from .. import sweep_bench


# ---------------------------------------------------------------------
# Shape corpus (folded in from generate_wide_coverage / generate_edge_dims)
# ---------------------------------------------------------------------


def generate_wide_shapes() -> List[Tuple[int, int, int]]:
    """Comprehensive (M, N, K) corpus covering inference / training / edge cases.

    Categories: M=1 single-token inference, tiny/small/medium/large M, square
    powers-of-two, skinny/tall, deep/shallow K, prime dims, and LLM-specific
    shapes. Ported verbatim from the retired CK-Tile ``generate_wide_coverage``.
    """
    shapes = set()

    # M=1 (single token) across various N, K
    for n in [512, 1024, 1536, 2048, 3072, 4096, 4608, 7168, 8192, 11008, 14336, 28672]:
        for k in [256, 512, 1024, 1536, 2048, 2304, 4096, 7168, 8192]:
            shapes.add((1, n, k))

    # Tiny M (2-16)
    for m in [2, 4, 8, 16]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 4096, 7168]:
                shapes.add((m, n, k))

    # Small M (32-128)
    for m in [32, 48, 64, 96, 128]:
        for n in [512, 1536, 4096, 7168, 8192]:
            for k in [256, 512, 2048, 4096, 7168]:
                shapes.add((m, n, k))

    # Medium M (256-2048)
    for m in [256, 384, 512, 768, 1024, 1536, 2048]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 2048, 4096, 7168]:
                shapes.add((m, n, k))

    # Large M (4096-20480)
    for m in [4096, 8192, 12288, 16384, 20480]:
        for n in [512, 1536, 4096, 7168]:
            for k in [256, 1024, 2048, 7168]:
                shapes.add((m, n, k))

    # Square shapes (powers of 2): 32 .. 8192
    for p in range(5, 14):
        d = 2**p
        shapes.add((d, d, d))

    # Skinny M, tall N
    for m in [1, 4, 16, 64]:
        for n in [8192, 16384, 28672]:
            for k in [1024, 4096, 8192]:
                shapes.add((m, n, k))

    # Tall M, skinny N
    for m in [4096, 8192, 16384]:
        for n in [32, 64, 128, 256]:
            for k in [1024, 4096]:
                shapes.add((m, n, k))

    # Deep K (K >> M, N)
    for m in [16, 64, 256]:
        for n in [16, 64, 256]:
            for k in [4096, 8192, 16384, 32768]:
                shapes.add((m, n, k))

    # Shallow K (K << M, N)
    for m in [1024, 4096, 8192]:
        for n in [1024, 4096, 8192]:
            for k in [16, 32, 64, 128]:
                shapes.add((m, n, k))

    # Prime dimensions
    primes = [17, 31, 37, 127, 251, 509, 1021, 2039, 4093]
    for p in primes:
        shapes.add((p, p, p))
    for p in primes[:5]:
        shapes.add((p, 4096, 4096))
        shapes.add((4096, p, 4096))
        shapes.add((4096, 4096, p))

    # LLM-specific shapes (DeepSeek MoE, LLaMA-7B/70B, GPT-style attention)
    llm_shapes = [
        (1, 1536, 7168),
        (1, 4608, 7168),
        (1, 7168, 2048),
        (1, 7168, 2304),
        (1, 7168, 256),
        (1, 576, 7168),
        (1, 512, 7168),
        (1, 3072, 1536),
        (1, 4096, 4096),
        (32, 4096, 4096),
        (128, 4096, 4096),
        (1, 4096, 11008),
        (32, 4096, 11008),
        (1, 11008, 4096),
        (32, 11008, 4096),
        (1, 8192, 8192),
        (32, 8192, 8192),
        (128, 8192, 8192),
        (1, 8192, 28672),
        (32, 8192, 28672),
        (1, 28672, 8192),
        (128, 128, 64),
        (128, 128, 128),
        (256, 256, 64),
        (512, 512, 64),
        (1024, 1024, 64),
        (2048, 2048, 64),
    ]
    for s in llm_shapes:
        shapes.add(s)

    # Non-power-of-2 common sizes
    for m in [48, 96, 192, 384, 576, 768, 1152, 1536, 2304, 3072, 4608, 6144]:
        shapes.add((m, m, m))
        shapes.add((m, 4096, 4096))

    return sorted(shapes)


def generate_edge_shapes() -> List[Tuple[int, int, int]]:
    """Degenerate / edge-case shapes: N=1, K=1, M=1 and tiny dims.

    Ported verbatim from the retired CK-Tile ``generate_edge_dims``.
    """
    shapes = set()

    # N=1: vector-matrix multiply
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for k in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192]:
            shapes.add((m, 1, k))

    # K=1: rank-1 update / outer product
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        for n in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192]:
            shapes.add((m, n, 1))

    # M=1, N=1: dot product
    for k in [1, 16, 64, 256, 1024, 4096, 8192]:
        shapes.add((1, 1, k))

    # M=1, K=1: scalar-vector
    for n in [1, 16, 64, 256, 1024, 4096, 8192]:
        shapes.add((1, n, 1))

    # N=1, K=1: scalar-vector
    for m in [1, 16, 64, 256, 1024, 4096, 8192]:
        shapes.add((m, 1, 1))

    # All ones
    shapes.add((1, 1, 1))

    # Small N (2-16)
    for m in [64, 256, 1024, 4096]:
        for n in [2, 3, 4, 7, 8, 15, 16]:
            for k in [64, 256, 1024, 4096]:
                shapes.add((m, n, k))

    # Small K (2-16)
    for m in [64, 256, 1024, 4096]:
        for n in [64, 256, 1024, 4096]:
            for k in [2, 3, 4, 7, 8, 15, 16]:
                shapes.add((m, n, k))

    return sorted(shapes)


def generate_shape_corpus(shape_set: str) -> List[Tuple[int, int, int]]:
    """Return the requested shape corpus.

    ``shape_set`` is one of ``"wide"``, ``"edge"``, or ``"all"``.
    """
    if shape_set == "wide":
        return generate_wide_shapes()
    if shape_set == "edge":
        return generate_edge_shapes()
    if shape_set == "all":
        merged = set(generate_wide_shapes()) | set(generate_edge_shapes())
        return sorted(merged)
    raise ValueError(f"unknown shape_set {shape_set!r} (want wide|edge|all)")


# ---------------------------------------------------------------------
# Variant enumeration
# ---------------------------------------------------------------------


def enumerate_variants(
    *,
    arch: str = "gfx950",
    pipelines: Sequence[str] = ("compv3", "compv4"),
    epilogues: Sequence[str] = ("default", "cshuffle"),
) -> List[UniversalGemmSpec]:
    """Enumerate the validity-filtered :class:`UniversalGemmSpec` candidate grid.

    Thin wrapper over :func:`all_dispatcher_configs` (which already applies
    ``is_valid_spec``) so callers can override the pipeline / epilogue families
    while keeping the rest of the dispatcher's default grid.
    """
    return list(
        all_dispatcher_configs(
            pipeline=pipelines,  # type: ignore[arg-type]
            epilogue=epilogues,  # type: ignore[arg-type]
            arch=arch,
        )
    )


# ---------------------------------------------------------------------
# Config-column recovery (spec -> canonical parquet row fields)
# ---------------------------------------------------------------------


def _config_columns(spec: UniversalGemmSpec) -> Dict[str, object]:
    """Recover the kernel-config columns directly from the spec.

    The :class:`UniversalGemmSpec` is authoritative for every kernel-config
    field; we read them off the spec rather than round-tripping through the
    kernel name so the parquet is always self-consistent with the kernels that
    were built. Column names + casing match what
    :class:`feature_engine.GemmUniversalFeatureEngine` and ``train.py`` expect.
    """
    t = spec.tile
    tr = spec.trait
    d = spec.data
    return {
        # Metadata.
        "dtype": d.dtype_a,
        "layout": d.layout.lower(),
        # Tile / warp dims.
        "tile_m": int(t.tile_m),
        "tile_n": int(t.tile_n),
        "tile_k": int(t.tile_k),
        "warp_m": int(t.warp_m),
        "warp_n": int(t.warp_n),
        "warp_k": int(t.warp_k),
        "warp_tile_m": int(t.warp_tile_m),
        "warp_tile_n": int(t.warp_tile_n),
        "warp_tile_k": int(t.warp_tile_k),
        # Pipeline knobs (lowercase strings the *_MAP encoders expect).
        "pipeline": str(tr.pipeline),
        "scheduler": str(tr.scheduler),
        "epilogue": str(tr.epilogue),
        # Padding / persistent flags.
        "pad_m": bool(tr.pad_m),
        "pad_n": bool(tr.pad_n),
        "pad_k": bool(tr.pad_k),
        "persistent": bool(tr.persistent),
    }


def _bandwidth_gb_s(m: int, n: int, k: int, latency_ms: float, dtype: str) -> float:
    """Bytes-moved / latency, matching convert_json_to_parquet's formula."""
    bpe = {"fp8": 1, "fp16": 2, "bf16": 2, "fp32": 4}.get(dtype, 2)
    if latency_ms <= 0.0:
        return 0.0
    bytes_moved = (m * k + k * n + m * n) * bpe
    return (bytes_moved / 1e9) / (latency_ms / 1000.0)


# ---------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------


def generate(
    *,
    out_path: Path,
    cache_dir: Path,
    arch: str = "gfx950",
    shape_set: str = "wide",
    max_shapes: Optional[int] = None,
    pipelines: Sequence[str] = ("compv3", "compv4"),
    epilogues: Sequence[str] = ("default", "cshuffle"),
    parallel: Optional[int] = None,
    attempts: int = 3,
    launcher: Optional[Path] = None,
    verify_shape: Tuple[int, int, int] = (256, 256, 256),
    isa: Optional[str] = None,
) -> pd.DataFrame:
    """Build + run the variant x shape grid and write the training parquet.

    Returns the DataFrame that was written. One row per ``(variant, shape)``;
    rows that failed build, verification, or perf are emitted with
    ``is_valid=False`` and zero targets so the model can learn the failure
    surface (same convention as the CK-Tile path).
    """
    shapes = generate_shape_corpus(shape_set)
    if max_shapes is not None and max_shapes > 0:
        shapes = shapes[:max_shapes]

    specs = enumerate_variants(arch=arch, pipelines=pipelines, epilogues=epilogues)
    if not specs:
        raise RuntimeError(
            f"no valid UniversalGemmSpec variants for arch={arch} "
            f"(pipelines={pipelines}, epilogues={epilogues})"
        )

    print(
        f"[gen] arch={arch} shapes={len(shapes)} variants={len(specs)} "
        f"-> {len(shapes) * len(specs)} (variant, shape) rows",
        file=sys.stderr,
        flush=True,
    )

    isa = isa or f"amdgcn-amd-amdhsa--{arch}"

    # Stage 1: build every variant (cached).
    print(f"[gen] building {len(specs)} variants -> {cache_dir}", file=sys.stderr)
    records = sweep.build_all_instances(
        specs, cache_dir=cache_dir, isa=isa, parallel=parallel
    )
    n_built = sum(1 for r in records if r.ok)
    print(f"[gen] built {n_built}/{len(records)} variants OK", file=sys.stderr)

    # Index the spec config by kernel name so we can attach config columns to
    # each RunResult (which only carries the name back).
    config_by_name: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        config_by_name[spec.kernel_name()] = _config_columns(spec)

    # Stage 2: write the sweep manifest and run it.
    manifest_path = cache_dir / "sweep_manifest.json"
    sweep.write_sweep_manifest(records, manifest_path, shapes=shapes)
    print(f"[gen] wrote manifest {manifest_path}", file=sys.stderr)

    print(
        f"[gen] running sweep ({attempts} attempts/shape) ...",
        file=sys.stderr,
        flush=True,
    )
    results = sweep_bench.sweep_run(
        manifest_path,
        launcher,
        shapes=shapes,
        verify_shape=verify_shape,
        attempts=attempts,
        skip_uncorrect=False,
        out_csv=cache_dir / "sweep_results.csv",
    )

    rows: List[Dict[str, object]] = []
    for res in results:
        cfg = config_by_name.get(res.name)
        if cfg is None:
            # A name we can't map back (shouldn't happen) -- skip it.
            continue
        tflops = res.median_tflops if res.correct else 0.0
        is_valid = bool(res.correct and res.runs and tflops > 0.0)
        dtype = str(cfg["dtype"])
        # sweep_bench reports TFLOPS, not latency; derive a latency estimate
        # from TFLOPS so the latency/bandwidth columns are populated.
        flops = 2.0 * res.M * res.N * res.K
        latency_ms = (flops / (tflops * 1e12) * 1000.0) if tflops > 0.0 else 0.0
        row: Dict[str, object] = {
            "op_type": "gemm_universal",
            "arch": arch,
            "kernel_name": res.name,
            "m": int(res.M),
            "n": int(res.N),
            "k": int(res.K),
            "split_k": 1,
            "measured_tflops": float(tflops),
            "latency_ms": float(latency_ms),
            "bandwidth_gb_s": _bandwidth_gb_s(res.M, res.N, res.K, latency_ms, dtype),
            "is_valid": is_valid,
            "run_id": 0,
        }
        row.update(cfg)
        rows.append(row)

    df = pd.DataFrame(rows)

    n_valid = int(df["is_valid"].sum()) if len(df) else 0
    print(
        f"[gen] {len(df)} rows ({n_valid} valid) -> {out_path}",
        file=sys.stderr,
        flush=True,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "rocke-native GEMM heuristics training-data generator "
            "(drives rocke.sweep + rocke.sweep_bench, emits training parquet)."
        )
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_sweep_cache"),
        help="Directory for cached HSACO binaries + sweep manifest/results.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")
    parser.add_argument(
        "--shape-set",
        default="wide",
        choices=["wide", "edge", "all"],
        help="Shape corpus to sweep.",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of shapes (smoke tests).",
    )
    parser.add_argument(
        "--pipelines",
        default="compv3,compv4",
        help="Comma-separated pipeline families to enumerate.",
    )
    parser.add_argument(
        "--epilogues",
        default="default,cshuffle",
        help="Comma-separated epilogue families to enumerate.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Build parallelism (default os.cpu_count(); 1 = serial).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Fresh-process perf attempts per (variant, shape).",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=None,
        help="Optional C++ launcher; omit to use python -m rocke.run_manifest.",
    )
    args = parser.parse_args(argv)

    pipelines = tuple(p.strip() for p in args.pipelines.split(",") if p.strip())
    epilogues = tuple(e.strip() for e in args.epilogues.split(",") if e.strip())

    generate(
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        shape_set=args.shape_set,
        max_shapes=args.max_shapes,
        pipelines=pipelines,
        epilogues=epilogues,
        parallel=args.parallel,
        attempts=args.attempts,
        launcher=args.launcher,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
