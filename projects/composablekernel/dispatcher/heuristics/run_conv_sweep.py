#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Per-shard conv-fwd fp16 benchmark sweep driver.

Reads a shard CSV of 2D grouped-conv forward shapes (produced by
sample_conv_shapes.py), runs every kernel instance listed in a profiler
config JSON against each shape via ckProfiler, and writes one CSV row per
(shape, kernel) pair.

Output CSV columns:
    kernel, N, G, C, K, Hi, Wi, Y, X, stride_h, stride_w,
    pad_h, pad_w, tflops, latency_ms

Results are flushed per shape so partial output survives preemption.

Usage:
    python3 run_conv_sweep.py \\
        --profiler   /opt/rocm/bin/ckProfiler \\
        --config     $CK_HEURISTICS/../codegen/configs/grouped_conv/forward/profiler/nhwgc_fp16.json \\
        --shapes     /work/shapes/shard_00.csv \\
        --out        /work/results/shard_00.csv \\
        --warmup 5 --repeat 20
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# ckProfiler positional encoding for grouped_conv_fwd_tile
_OP       = "grouped_conv_fwd_tile"
_DTYPE    = "1"   # fp16
_LAYOUT   = "1"   # NHWGC / GKYXC / NHWGK
_IDXTYPE  = "0"   # 32-bit index
_VERIFY   = "0"
_INIT     = "2"   # decimal init (deterministic)
_LOG      = "0"
_TIME     = "1"

SHAPE_HEADER = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
                "stride_h", "stride_w", "pad_h", "pad_w", "direction"]

OUT_HEADER = ["kernel", "N", "G", "C", "K", "Hi", "Wi", "Y", "X",
              "stride_h", "stride_w", "pad_h", "pad_w", "tflops", "latency_ms"]


def load_instances(config_path: Path) -> list[dict]:
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg["instances"]


def instance_name(inst: dict) -> str:
    """Reconstruct the kernel name the profiler prints, for CSV identification."""
    dsb = "_dsb" if inst.get("double_smem_buffer") else ""
    si  = "_si"  if inst.get("split_image") else ""
    ts  = f"{inst['tile_m']}x{inst['tile_n']}x{inst['tile_k']}"
    pipeline = inst["pipeline"]
    scheduler = inst.get("scheduler", "intrawave")
    return f"grouped_conv_forward_fp16_2d_{ts}_{pipeline}_{scheduler}{dsb}{si}"


def profiler_args(profiler: str, shape: dict, warmup: int, repeat: int) -> list[str]:
    """Build the ckProfiler positional argument list for a 2D conv shape."""
    N, G, C, K   = shape["N"], shape["G"], shape["C"], shape["K"]
    Hi, Wi       = shape["Hi"], shape["Wi"]
    Y, X         = shape["Y"], shape["X"]
    sh, sw       = shape["stride_h"], shape["stride_w"]
    ph, pw       = shape["pad_h"], shape["pad_w"]
    # ckProfiler left/right padding are symmetric for standard conv
    return [
        profiler,
        _OP, _DTYPE, _LAYOUT, _IDXTYPE, _VERIFY, _INIT, _LOG, _TIME,
        "2",                    # num_dim_spatial
        str(G), str(N), str(K), str(C),
        str(Y), str(X),         # filter spatial
        str(Hi), str(Wi),       # input spatial
        str(sh), str(sw),       # strides
        "1", "1",               # dilations (always 1 for this sweep)
        str(ph), str(pw),       # left padding
        str(ph), str(pw),       # right padding (symmetric)
    ]


def compute_tflops(shape: dict, latency_ms: float) -> float:
    """2*N*K*(C/G)*Y*X*Ho*Wo MACs, dilation=1 assumed."""
    Ho = (shape["Hi"] + 2 * shape["pad_h"] - shape["Y"]) // shape["stride_h"] + 1
    Wo = (shape["Wi"] + 2 * shape["pad_w"] - shape["X"]) // shape["stride_w"] + 1
    flops = 2 * shape["N"] * shape["K"] * (shape["C"] // shape["G"]) \
              * shape["Y"] * shape["X"] * Ho * Wo
    return flops / latency_ms / 1e9


def parse_profiler_output(stdout: str, shape: dict) -> list[dict]:
    """Parse ckProfiler stdout. Each supported kernel prints:
         Perf:  <latency_ms> ms, <kernel_name>
    Returns list of result dicts, one per supported kernel.
    """
    results = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("Perf:"):
            continue
        # Format: "Perf:    0.16487 ms, grouped_conv_fwd_fp16_nhwgc_2d_..."
        try:
            rest = line[len("Perf:"):].strip()
            ms_str, kernel_name = rest.split(" ms, ", 1)
            latency_ms = float(ms_str.strip())
            kernel_name = kernel_name.strip()
        except (ValueError, IndexError):
            continue
        results.append({
            "kernel": kernel_name,
            "N": shape["N"], "G": shape["G"], "C": shape["C"], "K": shape["K"],
            "Hi": shape["Hi"], "Wi": shape["Wi"],
            "Y": shape["Y"], "X": shape["X"],
            "stride_h": shape["stride_h"], "stride_w": shape["stride_w"],
            "pad_h": shape["pad_h"], "pad_w": shape["pad_w"],
            "tflops": compute_tflops(shape, latency_ms),
            "latency_ms": latency_ms,
        })
    return results


def run_shape(profiler: str, shape: dict, instances: list[dict],
              warmup: int, repeat: int, timeout: int) -> list[dict]:
    """Run ckProfiler once for this shape (all instances in one invocation)."""
    import os
    args = profiler_args(profiler, shape, warmup, repeat)
    # Total timeout: generous per-instance budget times instance count
    total_timeout = timeout * len(instances)
    try:
        proc = subprocess.run(
            args,
            capture_output=True, text=True, timeout=total_timeout,
            env={k: v for k, v in os.environ.items()},
        )
        return parse_profiler_output(proc.stdout, shape)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {total_timeout}s", file=sys.stderr, flush=True)
        return []
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr, flush=True)
        return []


def main():
    parser = argparse.ArgumentParser(description="Per-shard conv-fwd fp16 benchmark sweep")
    parser.add_argument("--profiler", default="/opt/rocm/bin/ckProfiler",
                        help="Path to ckProfiler binary")
    parser.add_argument("--config", required=True,
                        help="Path to nhwgc_fp16.json profiler config")
    parser.add_argument("--shapes", required=True,
                        help="Input shard CSV (from sample_conv_shapes.py)")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-kernel timeout in seconds")
    args = parser.parse_args()

    profiler = Path(args.profiler)
    if not profiler.exists():
        print(f"ERROR: ckProfiler not found at {profiler}", file=sys.stderr)
        sys.exit(1)

    instances = load_instances(Path(args.config))
    print(f"Loaded {len(instances)} kernel instances from {args.config}", file=sys.stderr)

    shapes = []
    with open(args.shapes, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append({k: int(v) if k != "direction" else v for k, v in row.items()})
    print(f"Loaded {len(shapes)} shapes from {args.shapes}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUT_HEADER)
        writer.writeheader()

        for i, shape in enumerate(shapes):
            results = run_shape(
                str(profiler), shape, instances,
                args.warmup, args.repeat, args.timeout,
            )
            writer.writerows(results)
            out_f.flush()
            print(
                f"[{i+1}/{len(shapes)}] "
                f"N={shape['N']} G={shape['G']} C={shape['C']} K={shape['K']} "
                f"Hi={shape['Hi']} Wi={shape['Wi']} Y={shape['Y']} X={shape['X']} "
                f"-> {len(results)} results",
                file=sys.stderr, flush=True,
            )

    print(f"\nDone. Results written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
