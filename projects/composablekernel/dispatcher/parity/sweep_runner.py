#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
T2.3 — Sweep runner: iterate (problem, kernel) pairs; record results to Parquet.

For every combination of problem size (M, N, K) × translated dispatcher config,
the runner:

  1. Drives codegen for that single config (drive_codegen.py).
  2. Builds the single-kernel harness against the generated header (build_harness.sh).
  3. Invokes the harness with -verify=1 to get validation verdict and TFLOP/s.
  4. Appends a row to the output Parquet file immediately after each run.

On restart, already-completed rows are skipped so a crashed sweep can resume
without redoing finished work.

Usage:
    # Dry-run: print the plan without invoking codegen/build/GPU.
    python sweep_runner.py configs/single_fp16_rcr.json --dry-run

    # CPU-only (Stage 1 identifier check only per config, skip GPU stages):
    python sweep_runner.py configs/single_fp16_rcr.json --cpu-only

    # Full sweep on a GPU node:
    python sweep_runner.py configs/single_fp16_rcr.json \\
        --sizes 512x512x512,1024x1024x1024,257x257x56 \\
        --output results.parquet

    # Resume a partially-complete sweep:
    python sweep_runner.py configs/single_fp16_rcr.json \\
        --output results.parquet   # existing file is loaded; done rows skipped

    # Multiple config files (one kernel-set per config):
    python sweep_runner.py configs/single_fp16_rcr.json configs/padding_fp16_rcr.json \\
        --output results.parquet

Output Parquet schema (one row per kernel × problem):
    config_file     str   -- source JSON path
    config_index    int   -- index within the translated config list
    identifier      str   -- canonical dispatcher registry key
    kernel_name     str   -- raw TE kernel name (used for header/binary filenames)
    datatype        str   -- fp16 / bf16 / fp8 / bf8 / int8
    layout          str   -- rcr / ccr / etc.
    pipeline        str   -- compv3 / compv4 / preshufflev2
    scheduler       str   -- intrawave / interwave / auto
    tile_m          int
    tile_n          int
    tile_k          int
    split_k         int
    pad_m           bool
    pad_n           bool
    pad_k           bool
    persistent      bool
    M               int
    N               int
    K               int
    verdict         str   -- PASSED / FAILED / SKIPPED / ERROR / DRYRUN
    tflops          float -- None if not available
    error_msg       str   -- empty string on success
    stage_failed    str   -- '' / 'codegen' / 'build' / 'harness'
    ts              str   -- ISO timestamp of this row
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from identifier import encode_identifier
from te_to_dispatcher import TranslationError, translate_file

_HERE = Path(__file__).resolve().parent
_DRIVE_CODEGEN = _HERE / "drive_codegen.py"
_BUILD_HARNESS = _HERE / "build_harness.sh"

# Subprocess timeouts (seconds) – matching check_parity.py constants.
_TIMEOUT_CODEGEN = 300
_TIMEOUT_BUILD   = 600
_TIMEOUT_HARNESS = 120

_PARQUET_SCHEMA = [
    "config_file", "config_index", "identifier", "kernel_name",
    "datatype", "layout", "pipeline", "scheduler",
    "tile_m", "tile_n", "tile_k", "split_k",
    "pad_m", "pad_n", "pad_k", "persistent",
    "M", "N", "K",
    "verdict", "tflops", "error_msg", "stage_failed", "ts",
]


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def _load_done_keys(output: Path) -> set:
    """Return set of (identifier, M, N, K) tuples already recorded."""
    if not output.exists():
        return set()
    try:
        df = pd.read_parquet(output)
        return set(zip(df["identifier"], df["M"], df["N"], df["K"]))
    except Exception:
        return set()


def _append_row(output: Path, row: Dict[str, Any]) -> None:
    """Append a single result row to the Parquet file."""
    df_new = pd.DataFrame([{k: row.get(k) for k in _PARQUET_SCHEMA}])
    if output.exists():
        df_old = pd.read_parquet(output)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(output, index=False)


# --------------------------------------------------------------------------- #
# Subprocess wrappers
# --------------------------------------------------------------------------- #

def _run_codegen(config_path: Path, index: int, output_dir: Path,
                 kernel_set: str, dry_run: bool) -> Tuple[bool, str]:
    """Drive codegen for one config. Returns (success, error_msg)."""
    cmd = [
        sys.executable, str(_DRIVE_CODEGEN), str(config_path),
        "--index", str(index),
        "--output-dir", str(output_dir),
        "--kernel-set", kernel_set,
    ]
    if dry_run:
        return True, ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_CODEGEN)
    except subprocess.TimeoutExpired:
        return False, f"codegen timed out after {_TIMEOUT_CODEGEN}s"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "codegen failed").strip()
    return True, ""


def _run_build(header: Path, arch: str, dry_run: bool) -> Tuple[bool, str]:
    """Build the single-kernel harness. Returns (success, error_msg)."""
    cmd = ["bash", str(_BUILD_HARNESS), str(header), arch]
    if dry_run:
        return True, ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_BUILD)
    except subprocess.TimeoutExpired:
        return False, f"build timed out after {_TIMEOUT_BUILD}s"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "build failed").strip()
    return True, ""


def _run_harness(m: int, n: int, k: int, dry_run: bool,
                 ) -> Tuple[str, Optional[float], str]:
    """Run the harness for one problem size. Returns (verdict, tflops, error)."""
    harness = _HERE / "harness"
    if not harness.exists() and not dry_run:
        return "ERROR", None, "harness binary not found; build first"
    cmd = [str(harness), f"-m={m}", f"-n={n}", f"-k={k}", "-verify=1"]
    if dry_run:
        return "DRYRUN", None, ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_HARNESS)
    except subprocess.TimeoutExpired:
        return "FAILED", None, f"harness timed out after {_TIMEOUT_HARNESS}s"
    stdout = proc.stdout
    if proc.returncode != 0 and "SKIPPED" not in stdout and "PASSED" not in stdout:
        return "ERROR", None, (proc.stderr or stdout or "harness crashed").strip()
    # Parse verdict and tflops from harness stdout.
    verdict = "UNKNOWN"
    tflops: Optional[float] = None
    import re
    for line in stdout.splitlines():
        if "PASSED" in line:
            verdict = "PASSED"
        elif "FAILED" in line:
            verdict = "FAILED"
        elif "SKIPPED" in line:
            verdict = "SKIPPED"
        # Harness prints: time   : 0.0464 ms  (46305.2 GFLOP/s)
        # Convert GFLOP/s → TFLOP/s for the parquet column.
        m_gflops = re.search(r"\(([0-9.]+)\s*GFLOP/s\)", line)
        if m_gflops:
            try:
                tflops = float(m_gflops.group(1)) / 1000.0
            except ValueError:
                pass
        # Also accept explicit TFLOP/s lines (future-proofing).
        m_tflops = re.search(r"\(([0-9.]+)\s*TFLOP/s\)", line)
        if m_tflops:
            try:
                tflops = float(m_tflops.group(1))
            except ValueError:
                pass
    return verdict, tflops, ""


# --------------------------------------------------------------------------- #
# Sweep logic
# --------------------------------------------------------------------------- #

def _sweep_config(
    config_path: Path,
    sizes: List[Tuple[int, int, int]],
    output: Path,
    arch: str,
    cpu_only: bool,
    dry_run: bool,
    done_keys: set,
) -> int:
    """Sweep all translated configs × sizes for one config file. Returns error count."""
    try:
        configs = translate_file(config_path)
    except TranslationError as e:
        print(f"[ERROR] Translation failed for {config_path}: {e}", file=sys.stderr)
        return 1

    if not configs:
        print(f"[WARN] No valid dispatcher configs from {config_path}", file=sys.stderr)
        return 0

    print(f"\n{'='*72}")
    print(f"Config: {config_path}  ({len(configs)} kernel(s), {len(sizes)} size(s))")
    print(f"{'='*72}")

    error_count = 0

    for idx, cfg in enumerate(configs):
        ident = encode_identifier(cfg)
        te = cfg["_te"]
        alg = cfg["algorithm"]
        kernel_name = _kernel_name(cfg)
        kernel_set = f"sweep_{config_path.stem}_{idx}"

        print(f"\n  [{idx+1}/{len(configs)}] {ident}")

        # --- Codegen + build (once per kernel, not per size) ---
        output_dir = _HERE / "generated"
        header = output_dir / kernel_set / f"gemm_{kernel_name}.hpp"

        if not cpu_only and not dry_run:
            # Codegen: only if header not yet generated.
            if not header.exists():
                ok, err = _run_codegen(config_path, idx, output_dir, kernel_set, dry_run)
                if not ok:
                    print(f"    [codegen FAIL] {err}", file=sys.stderr)
                    for m, n, k in sizes:
                        key = (ident, m, n, k)
                        if key in done_keys:
                            continue
                        _append_row(output, _make_row(
                            config_path, idx, cfg, m, n, k,
                            "ERROR", None, err, "codegen",
                        ))
                    error_count += 1
                    continue

            # Build: always rebuild for this kernel before running its sizes.
            # The harness binary is overwritten per-config; skipping the build
            # would leave the previous config's binary in place.
            ok_build, err_build = _run_build(header, arch, dry_run)
            if not ok_build:
                print(f"    [build FAIL] {err_build}", file=sys.stderr)
                for m, n, k in sizes:
                    key = (ident, m, n, k)
                    if key in done_keys:
                        continue
                    _append_row(output, _make_row(
                        config_path, idx, cfg, m, n, k,
                        "ERROR", None, err_build, "build",
                    ))
                error_count += 1
                continue

        # --- Per-size harness runs ---
        for m, n, k in sizes:
            key = (ident, m, n, k)
            if key in done_keys:
                print(f"    [SKIP] {m}x{n}x{k} already recorded")
                continue

            if cpu_only:
                verdict, tflops, err = "SKIPPED", None, "cpu-only mode"
            else:
                verdict, tflops, err = _run_harness(m, n, k, dry_run)

            status_tag = verdict
            _append_row(output, _make_row(
                config_path, idx, cfg, m, n, k,
                verdict, tflops, err, "" if verdict != "ERROR" else "harness",
            ))
            print(f"    {m}x{n}x{k}  {status_tag}"
                  + (f"  {tflops:.3f} TFLOP/s" if tflops else ""))
            if verdict in ("FAILED", "ERROR"):
                error_count += 1

    return error_count


def _kernel_name(cfg: Dict[str, Any]) -> str:
    """Derive raw TE kernel name (mirrors check_parity.te_kernel_name)."""
    te = cfg["_te"]
    alg = cfg["algorithm"]
    cap = lambda b: str(bool(b)).capitalize()
    name = (
        f"{te['datatype']}_{te['layout']}_"
        f"{te['pipeline']}_{te['epilogue']}_{te['scheduler']}_"
        f"{cap(alg['pad_m'])}_{cap(alg['pad_n'])}_{cap(alg['pad_k'])}_{cap(alg['persistent'])}_"
        f"{alg['tile_m']}x{alg['tile_n']}x{alg['tile_k']}_"
        f"{alg['warp_m']}x{alg['warp_n']}x{alg['warp_k']}_"
        f"{alg['warp_tile_m']}x{alg['warp_tile_n']}x{alg['warp_tile_k']}"
    )
    if te["pipeline"] in ("preshufflev2",):
        name += "_preshuffle"
    return name


def _make_row(
    config_path: Path,
    idx: int,
    cfg: Dict[str, Any],
    m: int, n: int, k: int,
    verdict: str,
    tflops: Optional[float],
    error_msg: str,
    stage_failed: str,
) -> Dict[str, Any]:
    te = cfg["_te"]
    alg = cfg["algorithm"]
    return {
        "config_file":   str(config_path),
        "config_index":  idx,
        "identifier":    encode_identifier(cfg),
        "kernel_name":   _kernel_name(cfg),
        "datatype":      te["datatype"],
        "layout":        te["layout"],
        "pipeline":      te["pipeline"],
        "scheduler":     te["scheduler"],
        "tile_m":        alg["tile_m"],
        "tile_n":        alg["tile_n"],
        "tile_k":        alg["tile_k"],
        "split_k":       cfg["signature"]["split_k"],
        "pad_m":         alg["pad_m"],
        "pad_n":         alg["pad_n"],
        "pad_k":         alg["pad_k"],
        "persistent":    alg["persistent"],
        "M": m, "N": n, "K": k,
        "verdict":       verdict,
        "tflops":        tflops,
        "error_msg":     error_msg,
        "stage_failed":  stage_failed,
        "ts":            datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_sizes(sizes_str: str) -> List[Tuple[int, int, int]]:
    sizes = []
    for s in sizes_str.split(","):
        parts = s.strip().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid size {s!r}: expected MxNxK")
        sizes.append(tuple(int(p) for p in parts))
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", type=Path, nargs="+", help="Tile Engine config JSON file(s)")
    ap.add_argument(
        "--sizes", default="512x512x512,1024x1024x1024,257x257x56,513x511x40",
        help="Comma-separated MxNxK problem sizes (default: %(default)s)",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("sweep_results.parquet"),
        help="Output Parquet file (appended to; existing rows skipped). Default: %(default)s",
    )
    ap.add_argument("--arch", default="gfx942", help="GPU target arch for hipcc (default: %(default)s)")
    ap.add_argument(
        "--cpu-only", action="store_true",
        help="Skip GPU stages (codegen/build/harness); record SKIPPED rows for all sizes",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without invoking codegen, build, or harness",
    )
    args = ap.parse_args()

    try:
        sizes = _parse_sizes(args.sizes)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    done_keys = _load_done_keys(args.output)
    if done_keys:
        print(f"Resuming: {len(done_keys)} (identifier, M, N, K) rows already recorded.")

    total_errors = 0
    for config_path in args.configs:
        if not config_path.exists():
            print(f"error: config not found: {config_path}", file=sys.stderr)
            total_errors += 1
            continue
        total_errors += _sweep_config(
            config_path, sizes, args.output, args.arch,
            args.cpu_only, args.dry_run, done_keys,
        )

    print(f"\n{'='*72}")
    print(f"Sweep complete. Results: {args.output}")
    if args.output.exists():
        df = pd.read_parquet(args.output)
        total = len(df)
        passed = (df["verdict"] == "PASSED").sum()
        failed = (df["verdict"] == "FAILED").sum()
        errored = (df["verdict"] == "ERROR").sum()
        skipped = df["verdict"].isin(["SKIPPED", "DRYRUN"]).sum()
        print(f"  Total rows : {total}")
        print(f"  PASSED     : {passed}")
        print(f"  FAILED     : {failed}")
        print(f"  ERROR      : {errored}")
        print(f"  SKIPPED    : {skipped}")
        if total > 0:
            print(f"  Pass rate  : {passed/total*100:.1f}%")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
