#!/usr/bin/env python3
"""okl: query hipBLASLt for the optimal kernel for a given GEMM problem.

Thin wrapper around `hipblaslt-bench --algo_method heuristic --print_kernel_info`.
The heuristic dispatch already encodes the shipped tuning, so we delegate to
hipBLASLt rather than reimplementing its solution-selection logic.

Output: JSON to stdout with the chosen solution's name, index, achieved
gflops, achieved GB/s, the problem echoed back, and the raw bench command
for reproducibility. Non-zero exit on bench failure.

Usage:
    okl.py -m 4096 -n 4096 -k 4096 --transa T --transb N \\
           --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \\
           --compute-type f32_r
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_CANDIDATES = [
    "/home/alvasile/rocm-libraries/projects/hipblaslt/build/release/clients/hipblaslt-bench",
    "/opt/rocm/bin/hipblaslt-bench",
]

LIBPATH_CANDIDATES = [
    "/opt/rocm/lib/hipblaslt/library",
    "/opt/rocm-7.2.1/lib/hipblaslt/library",
    "/home/alvasile/rocm-libraries/projects/hipblaslt/build/release/Tensile/library",
]

SOL_NAME_PREFIX = "--Solution name:"
SOL_IDX_PREFIX = "--Solution index:"
HEADER_MARKER = "hipblaslt-Gflops"


def find_bench(override):
    if override:
        return override
    p = shutil.which("hipblaslt-bench")
    if p:
        return p
    for c in BENCH_CANDIDATES:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    sys.exit("error: hipblaslt-bench not found. Pass --bench /path/to/hipblaslt-bench.")


def find_libpath(override):
    if override:
        return override
    env = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
    if env:
        return env
    for c in LIBPATH_CANDIDATES:
        if Path(c).is_dir() and any(Path(c).glob("TensileLibrary_lazy_*.dat")):
            return c
    return None  # let bench error out itself


def parse_output(stdout):
    """Pull solution name/index and one timing row out of bench stdout."""
    sol_name, sol_idx = None, None
    header_fields, value_fields = None, None
    lines = stdout.splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith(SOL_NAME_PREFIX):
            sol_name = line[len(SOL_NAME_PREFIX):].strip()
        elif line.startswith(SOL_IDX_PREFIX):
            try:
                sol_idx = int(line[len(SOL_IDX_PREFIX):].strip())
            except ValueError:
                pass
        elif HEADER_MARKER in line and "," in line:
            after_colon = line.split(":", 1)[1] if line.startswith("[") and ":" in line else line
            header_fields = [f.strip() for f in after_colon.split(",")]
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                if cand and "," in cand and not cand.startswith("["):
                    value_fields = [f.strip() for f in cand.split(",")]
                    break

    timing = {}
    if header_fields and value_fields and len(header_fields) == len(value_fields):
        row = dict(zip(header_fields, value_fields))
        for src, dst in (
            ("hipblaslt-Gflops", "gflops"),
            ("hipblaslt-GB/s", "gb_per_s"),
            ("us", "microseconds"),
        ):
            if src in row:
                try:
                    timing[dst] = float(row[src])
                except ValueError:
                    timing[dst] = row[src]
    return sol_name, sol_idx, timing


def build_bench_args(a):
    args = [
        "-m", str(a.m), "-n", str(a.n), "-k", str(a.k),
        "--batch_count", str(a.batch),
        "--transA", a.transa, "--transB", a.transb,
        "--a_type", a.a_type, "--b_type", a.b_type,
        "--c_type", a.c_type, "--d_type", a.d_type,
        "--compute_type", a.compute_type,
        "--algo_method", "heuristic",
        "--requested_solution", "1",
        "--print_kernel_info",
        "--iters", str(a.iters),
        "--cold_iters", str(a.cold_iters),
    ]
    if a.bias_vector:
        args.append("--bias_vector")
    if a.bias_type:
        args += ["--bias_type", a.bias_type]
    if a.activation_type and a.activation_type != "none":
        args += ["--activation_type", a.activation_type]
    if a.alpha is not None:
        args += ["--alpha", str(a.alpha)]
    if a.beta is not None:
        args += ["--beta", str(a.beta)]
    if a.extra:
        args.extend(a.extra)
    return args


def main():
    p = argparse.ArgumentParser(
        description="Query hipBLASLt for the optimal kernel for one GEMM shape.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Anything after `--` is forwarded verbatim to hipblaslt-bench.",
    )
    p.add_argument("-m", type=int, required=True, help="M (rows of op(A) / D)")
    p.add_argument("-n", type=int, required=True, help="N (cols of op(B) / D)")
    p.add_argument("-k", type=int, required=True, help="K (inner dimension)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--transa", default="N", choices=("N", "T"))
    p.add_argument("--transb", default="N", choices=("N", "T"))
    p.add_argument("--a-type", default="f16_r")
    p.add_argument("--b-type", default="f16_r")
    p.add_argument("--c-type", default="f16_r")
    p.add_argument("--d-type", default="f16_r")
    p.add_argument("--compute-type", default="f32_r")
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--bias-vector", action="store_true")
    p.add_argument("--bias-type")
    p.add_argument("--activation-type", default="none")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--cold-iters", type=int, default=2)
    p.add_argument("--bench", help="Path to hipblaslt-bench. Default: $PATH, then known build/install locations.")
    p.add_argument("--libpath", help="HIPBLASLT_TENSILE_LIBPATH override (dir with TensileLibrary_lazy_<arch>.dat).")
    p.add_argument("--timeout", type=int, default=120, help="Seconds before killing bench.")
    p.add_argument("--keep-stdout", action="store_true", help="Include the full bench stdout in the JSON.")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Pass-through to hipblaslt-bench.")
    a = p.parse_args()

    bench = find_bench(a.bench)
    libpath = find_libpath(a.libpath)
    bench_args = build_bench_args(a)

    env = os.environ.copy()
    if libpath:
        env["HIPBLASLT_TENSILE_LIBPATH"] = libpath

    try:
        proc = subprocess.run(
            [bench, *bench_args],
            env=env,
            capture_output=True,
            text=True,
            timeout=a.timeout,
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"error: hipblaslt-bench timed out after {a.timeout}s")
    except FileNotFoundError as e:
        sys.exit(f"error: cannot execute {bench}: {e}")

    sol_name, sol_idx, timing = parse_output(proc.stdout)

    out = {
        "problem": {
            "m": a.m, "n": a.n, "k": a.k, "batch": a.batch,
            "transA": a.transa, "transB": a.transb,
            "a_type": a.a_type, "b_type": a.b_type,
            "c_type": a.c_type, "d_type": a.d_type,
            "compute_type": a.compute_type,
        },
        "solution_name": sol_name,
        "solution_index": sol_idx,
        "timing": timing,
        "libpath": libpath,
        "bench": bench,
        "bench_args": bench_args,
        "bench_returncode": proc.returncode,
    }
    if proc.returncode != 0 or sol_name is None:
        out["bench_stderr_tail"] = proc.stderr[-2000:]
        out["bench_stdout_tail"] = proc.stdout[-2000:]
    if a.keep_stdout:
        out["bench_stdout"] = proc.stdout

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if proc.returncode != 0 or sol_name is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
