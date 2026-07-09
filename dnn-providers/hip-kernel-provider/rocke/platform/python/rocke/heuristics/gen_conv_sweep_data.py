#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
rocke-native conv heuristics training-data generator.

Wraps the C++ rocke_conv_sweep binary which JIT-compiles and times implicit-GEMM
conv candidates with per-candidate process isolation (fork+exec).

Orchestrates:
  1. Generate a shapes CSV (from built-in coverage or user-supplied CSVs).
  2. Invoke the rocke_conv_sweep binary (JIT compile + time on GPU).
  3. Read the output CSV, attach hw profile + metadata, write training parquet.

The parquet output is compatible with GroupedConvFeatureEngine / train.py.

gen_sweep_data.py --op conv delegates to generate() here (same relationship as
--op gemm delegates to gen_gemm_sweep_data.generate()).

Usage:
    python3 -m rocke.heuristics.gen_conv_sweep_data \\
        --out sweep.parquet --arch gfx90a --shape-set wide --max-shapes 20

    # With explicit shape CSVs (from augment_coverage_conv.py):
    python3 -m rocke.heuristics.gen_conv_sweep_data \\
        --out sweep.parquet --arch gfx950 \\
        --shapes coverage.csv augmented.csv

    # With a pre-built binary:
    ROCKE_CONV_SWEEP_BIN=/path/to/rocke_conv_sweep \\
    python3 -m rocke.heuristics.gen_conv_sweep_data --out sweep.parquet --arch gfx90a

Output parquet columns consumed by GroupedConvFeatureEngine / train.py:
    N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w  (problem)
    tile_m, tile_n, tile_k, pipeline                              (config)
    tflops, latency_ms, is_valid                                  (targets)
    op_type, arch, kernel_name, build_ok, build_error, run_id     (metadata)
    hw_num_cus, hw_simds_per_cu, hw_shader_engines, ...           (hardware)
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .data_pipeline import get_hardware_profile as _get_hw_profile_rocminfo


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

_SILICON_CONSTANTS: Dict[str, Dict[str, int]] = {
    "gfx950": {
        "simds_per_cu": 4,
        "shader_engines": 32,
        "l1_cache_kb": 32,
        "l2_cache_kb": 4096,
        "l3_cache_kb": 262144,
        "num_xcd": 8,
    },
    "gfx942": {
        "simds_per_cu": 4,
        "shader_engines": 28,
        "l1_cache_kb": 32,
        "l2_cache_kb": 4096,
        "l3_cache_kb": 262144,
        "num_xcd": 8,
    },
    "gfx90a": {
        "simds_per_cu": 4,
        "shader_engines": 8,
        "l1_cache_kb": 16,
        "l2_cache_kb": 8192,
        "l3_cache_kb": 131072,
        "num_xcd": 1,
    },
}

_HIP_ATTR_CLOCK_RATE = 5
_HIP_ATTR_MAX_THREADS_PER_MP = 57
_HIP_ATTR_MULTIPROCESSOR_COUNT = 63
_HIP_ATTR_WARP_SIZE = 87
_HIP_ATTR_MAX_SHARED_MEM_PER_MP = 10002


def _hip_device_attr(lib, attr_id: int, device: int = 0) -> Optional[int]:
    """Call hipDeviceGetAttribute and return the integer value."""
    import ctypes

    val = ctypes.c_int(0)
    fn = lib.hipDeviceGetAttribute
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    rc = fn(ctypes.byref(val), attr_id, device)
    return val.value if rc == 0 else None


def _get_hw_profile_hip() -> Dict[str, object]:
    """Query GPU hardware profile via HIP ctypes (no rocminfo needed)."""
    import ctypes
    import re

    try:
        from ..runtime.hip_module import _resolve_hip

        lib = _resolve_hip()
    except Exception:
        return {}

    buf = ctypes.create_string_buffer(4096)
    rc = -1
    for sym in ("hipGetDevicePropertiesR0600", "hipGetDeviceProperties"):
        try:
            fn = getattr(lib, sym)
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
            rc = fn(buf, 0)
            if rc == 0:
                break
        except (AttributeError, OSError):
            continue
    if rc != 0:
        return {}

    m = re.search(rb"gfx[0-9a-z]+", buf.raw)
    arch = m.group(0).decode("ascii") if m else ""
    arch_base = arch.split(":")[0] if arch else ""

    profile: Dict[str, object] = {}
    if arch_base:
        profile["gfx_name"] = arch_base

    clock_khz = _hip_device_attr(lib, _HIP_ATTR_CLOCK_RATE)
    if clock_khz and clock_khz > 0:
        profile["max_clock_mhz"] = (clock_khz + 500) // 1000

    num_cus = _hip_device_attr(lib, _HIP_ATTR_MULTIPROCESSOR_COUNT)
    if num_cus and num_cus > 0:
        profile["num_cus"] = num_cus

    warp_size = _hip_device_attr(lib, _HIP_ATTR_WARP_SIZE)
    if warp_size and warp_size > 0:
        profile["wavefront_size"] = warp_size

    max_threads_mp = _hip_device_attr(lib, _HIP_ATTR_MAX_THREADS_PER_MP)
    if max_threads_mp and max_threads_mp > 0 and warp_size and warp_size > 0:
        profile["max_waves_per_cu"] = max_threads_mp // warp_size

    lds = _hip_device_attr(lib, _HIP_ATTR_MAX_SHARED_MEM_PER_MP)
    if lds and lds > 0:
        profile["lds_capacity"] = lds

    silicon = _SILICON_CONSTANTS.get(arch_base, {})
    for k, v in silicon.items():
        if k not in profile:
            profile[k] = v

    return profile


def _get_hw_profile() -> Dict[str, object]:
    """Get hardware profile: try rocminfo first, fall back to HIP ctypes."""
    profile = _get_hw_profile_rocminfo()
    if profile and "num_cus" in profile:
        return {f"hw_{k}": v for k, v in profile.items()}
    hip_profile = _get_hw_profile_hip()
    if hip_profile:
        return {f"hw_{k}": v for k, v in hip_profile.items()}
    return {}


# ---------------------------------------------------------------------------
# Shape loading
# ---------------------------------------------------------------------------


def load_shapes_from_csvs(paths: Sequence[Path]) -> List[Tuple[int, ...]]:
    """Load and deduplicate shapes from one or more CSV files.

    Accepts any CSV produced by generate_coverage_conv.py,
    shard_shapes.py, or augment_coverage_conv.py — all share the same
    column format: N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w
    """
    seen: set[Tuple[int, ...]] = set()
    out: List[Tuple[int, ...]] = []
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = (
                    int(row["N"]),
                    int(row["G"]),
                    int(row["C"]),
                    int(row["K"]),
                    int(row["Hi"]),
                    int(row["Wi"]),
                    int(row["Y"]),
                    int(row["X"]),
                    int(row["stride_h"]),
                    int(row["stride_w"]),
                    int(row["pad_h"]),
                    int(row["pad_w"]),
                )
                if t not in seen:
                    seen.add(t)
                    out.append(t)
    return out


def _write_shapes_csv(
    out_path: Path,
    shape_set: str = "wide",
    max_shapes: int = 0,
    shape_csvs: Optional[Sequence[Path]] = None,
) -> int:
    """Write a shapes CSV for the C++ binary to consume.

    If ``shape_csvs`` is provided, shapes are loaded from those files.
    Otherwise, shapes are generated from the built-in coverage corpus.
    """
    if shape_csvs:
        shape_list = load_shapes_from_csvs(shape_csvs)
    else:
        from .generate_coverage_conv import generate_wide_shapes, generate_edge_shapes

        if shape_set == "wide":
            shapes = generate_wide_shapes()
        elif shape_set == "edge":
            shapes = generate_edge_shapes()
        elif shape_set == "all":
            shapes = generate_wide_shapes() | generate_edge_shapes()
        else:
            print(f"ERROR: unknown shape-set '{shape_set}'", file=sys.stderr)
            return 0
        shape_list = sorted(shapes)

    if max_shapes and max_shapes < len(shape_list):
        shape_list = shape_list[:max_shapes]

    header = [
        "N",
        "G",
        "C",
        "K",
        "Hi",
        "Wi",
        "Y",
        "X",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in shape_list:
            w.writerow(s[:12])

    return len(shape_list)


# ---------------------------------------------------------------------------
# Sweep binary
# ---------------------------------------------------------------------------


def _find_sweep_binary() -> Optional[Path]:
    """Locate the rocke_conv_sweep binary."""
    env = os.environ.get("ROCKE_CONV_SWEEP_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    candidates = [
        Path("build/Cpp/tools/conv_sweep/rocke_conv_sweep"),
        Path("build/rocke_conv_sweep"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    return None


def run_sweep(
    out_path: Path,
    arch: str,
    shape_set: str = "wide",
    max_shapes: int = 20,
    dtype: str = "fp16",
    candidate_timeout: int = 120,
    binary_path: Optional[Path] = None,
    shape_csvs: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    """Run the C++ conv sweep and produce a training parquet."""

    sweep_bin = binary_path or _find_sweep_binary()
    if not sweep_bin or not sweep_bin.is_file():
        print(
            "ERROR: rocke_conv_sweep binary not found. Set ROCKE_CONV_SWEEP_BIN "
            "or build with -DROCKE_BUILD_CONV_SWEEP=ON.",
            file=sys.stderr,
        )
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="rocke_sweep_") as tmpdir:
        shapes_csv = Path(tmpdir) / "shapes.csv"
        raw_csv = Path(tmpdir) / "sweep_raw.csv"

        n_shapes = _write_shapes_csv(shapes_csv, shape_set, max_shapes, shape_csvs)
        print(
            f"[gen_conv] {n_shapes} shapes -> {shapes_csv}", file=sys.stderr, flush=True
        )

        cmd = [
            str(sweep_bin),
            "--shapes",
            str(shapes_csv),
            "--out",
            str(raw_csv),
            "--dtype",
            dtype,
            "--candidate-timeout",
            str(candidate_timeout),
        ]
        print(f"[gen_conv] {' '.join(cmd)}", file=sys.stderr, flush=True)
        result = subprocess.run(cmd, timeout=7200)
        # 0 = success, 1 = no data for some shapes, 2 = validator-gap triage
        if result.returncode not in (0, 1, 2):
            print(
                f"[gen_conv] ERROR: sweep binary exited {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        if not raw_csv.is_file() or raw_csv.stat().st_size == 0:
            print("[gen_conv] ERROR: no output produced", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(raw_csv)

    print(
        f"[gen_conv] {len(df)} timing rows from C++ sweep", file=sys.stderr, flush=True
    )

    if df.empty:
        print("[gen_conv] ERROR: sweep produced no timing data", file=sys.stderr)
        sys.exit(1)

    # C++ sweep outputs latency_us; training pipeline expects latency_ms.
    if "latency_us" in df.columns:
        df["latency_ms"] = df["latency_us"] / 1000.0
        df.drop(columns=["latency_us"], inplace=True)

    # Attach metadata columns for train.py compatibility
    df["op_type"] = "grouped_conv"
    df["dtype"] = dtype
    df["arch"] = arch
    df["is_valid"] = True
    df["build_ok"] = True
    df["build_error"] = ""
    df["run_id"] = 0

    df.rename(
        columns={
            "tile_m": "gemm_m_per_block",
            "tile_n": "gemm_n_per_block",
            "tile_k": "gemm_k_per_block",
        },
        inplace=True,
    )

    df["block_size"] = 256
    df["wave_mode"] = "intrawave"
    df["has_dsb"] = 0
    df["has_si"] = 0
    df["epilogue"] = "default"

    df["kernel_name"] = df.apply(
        lambda r: (
            f"conv_igemm_N{int(r['N'])}H{int(r['Hi'])}W{int(r['Wi'])}C{int(r['C'])}"
            f"_K{int(r['K'])}Y{int(r['Y'])}X{int(r['X'])}"
            f"_t{int(r['gemm_m_per_block'])}x{int(r['gemm_n_per_block'])}x{int(r['gemm_k_per_block'])}"
            f"_{r['pipeline']}"
        ),
        axis=1,
    )

    hw_profile = _get_hw_profile()
    if hw_profile:
        for k, v in hw_profile.items():
            df[k] = v
        print(
            f"[gen_conv] hw profile: {len(hw_profile)} fields",
            file=sys.stderr,
            flush=True,
        )
    else:
        print("[gen_conv] warning: hw profile unavailable", file=sys.stderr, flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    n_valid = df["is_valid"].sum()
    print(
        f"[gen_conv] wrote {len(df)} rows ({n_valid} valid) -> {out_path}",
        file=sys.stderr,
        flush=True,
    )
    return df


# ---------------------------------------------------------------------------
# generate() — compatibility interface for gen_sweep_data.py --op conv
# ---------------------------------------------------------------------------


def generate(
    *,
    out_path: Path,
    cache_dir: Path = Path("/tmp/rocke_conv_cache"),
    arch: str = "gfx950",
    shape_set: str = "wide",
    shape_csvs: Optional[Sequence[Path]] = None,
    max_shapes: Optional[int] = None,
    isa: Optional[str] = None,
    warmup_iters: int = 3,
    timed_iters: int = 20,
) -> pd.DataFrame:
    """Build + benchmark the (config x shape) grid and write the training parquet.

    ``shape_csvs``: if provided, shapes are loaded from these CSV files
    (output of generate_coverage_conv.py / shard_shapes.py /
    augment_coverage_conv.py) and ``shape_set`` is ignored.

    ``cache_dir``, ``isa``, ``warmup_iters``, ``timed_iters`` are accepted for
    backward compatibility but ignored — the C++ binary manages these internally.
    """
    return run_sweep(
        out_path=out_path,
        arch=arch,
        shape_set=shape_set,
        max_shapes=max_shapes or 0,
        dtype="fp16",
        shape_csvs=shape_csvs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "rocke-native conv heuristics training-data generator "
            "(C++ sweep binary -> CSV -> training parquet)."
        )
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_conv_cache"),
        help="(ignored, kept for backward compatibility).",
    )
    parser.add_argument(
        "--arch", type=str, default="gfx950", help="GPU architecture (default: gfx950)."
    )

    shape_source = parser.add_mutually_exclusive_group()
    shape_source.add_argument(
        "--shapes",
        nargs="+",
        type=Path,
        metavar="CSV",
        help=(
            "One or more shape CSVs produced by generate_coverage_conv.py, "
            "shard_shapes.py, or augment_coverage_conv.py. "
            "Mutually exclusive with --shape-set."
        ),
    )
    shape_source.add_argument(
        "--shape-set",
        default="wide",
        choices=["wide", "edge", "all"],
        help="Built-in shape corpus to sweep (default: wide). "
        "Mutually exclusive with --shapes.",
    )

    parser.add_argument(
        "--max-shapes", type=int, default=None, help="Limit number of shapes."
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", help="Data type (default: fp16)."
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=120,
        help="Per-candidate timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--binary", type=Path, default=None, help="Path to rocke_conv_sweep binary."
    )
    args = parser.parse_args(argv)

    run_sweep(
        out_path=args.out,
        arch=args.arch,
        shape_set=args.shape_set if args.shapes is None else "wide",
        max_shapes=args.max_shapes or 0,
        dtype=args.dtype,
        candidate_timeout=args.candidate_timeout,
        binary_path=args.binary,
        shape_csvs=args.shapes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
