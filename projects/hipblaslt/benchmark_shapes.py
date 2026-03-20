#!/usr/bin/env python3
"""
Benchmark hipblaslt-bench across a matrix of GEMM shapes and produce a CSV
with runtime, TFLOPs, and kernel source (aiter / rocroller) for each entry.
"""

import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HIPBLASLT_DIR = Path(__file__).resolve().parent
BUILD_DIR = HIPBLASLT_DIR / "build"
BENCH_BIN = BUILD_DIR / "clients" / "hipblaslt-bench"

DEFAULT_SHAPES = [
    # (M, N, K) — M,N must be multiples of 32; K must be a multiple of 256
    # Square shapes
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    # Rectangular — small M (decode-like)
    (32, 8192, 8192),
    (64, 8192, 8192),
    (128, 8192, 8192),
    (256, 8192, 8192),
    (512, 8192, 8192),
    (1024, 8192, 8192),
    (2048, 8192, 8192),
    (4096, 8192, 8192),
    # Rectangular — varied aspect ratios
    (8192, 8192, 4096),
    (8192, 4096, 8192),
    (16384, 8192, 4096),
    (4096, 16384, 8192),
    (8192, 16384, 4096),
    # LLM hidden-dim shapes (e.g. Llama-style FFN)
    (4096, 14336, 4096),
    (14336, 4096, 4096),
    (8192, 28672, 8192),
    (28672, 8192, 8192),
]


def build_bench(jobs: int = 0):
    """Rebuild hipblaslt-bench via ninja."""
    cmd = ["ninja", "hipblaslt-bench"]
    if jobs:
        cmd += [f"-j{jobs}"]
    print(f"Building: {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd, cwd=str(BUILD_DIR))


def run_one(m: int, n: int, k: int, iters: int, env: dict,
            gpu_queue: queue.Queue | None = None) -> dict:
    """Run hipblaslt-bench for a single shape and return parsed results.

    If gpu_queue is provided, acquires a GPU ID from it for the duration of
    the run, setting HIP_VISIBLE_DEVICES accordingly.
    """
    gpu_id = None
    if gpu_queue is not None:
        gpu_id = gpu_queue.get()
        env = env.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        str(BENCH_BIN),
        "--api_method", "c",
        "-m", str(m), "-n", str(n), "-k", str(k),
        "--alpha", "1", "--beta", "0",
        "--transA", "T", "--transB", "N",
        "--batch_count", "1",
        "--scaleA", "1001", "--scaleB", "1001",
        "--a_type", "f4_r", "--b_type", "f4_r",
        "--c_type", "bf16_r", "--d_type", "bf16_r",
        "--compute_type", "f32_r",
        "--rotating", "0",
        "--cold_iters", "2",
        "--iters", str(iters),
        "--swizzleA",
        "--verify",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(BUILD_DIR))
    finally:
        if gpu_queue is not None and gpu_id is not None:
            gpu_queue.put(gpu_id)

    if result.returncode != 0:
        print(f"  FAILED m={m} n={n} k={k}: {result.stderr[-500:]}", file=sys.stderr)
        return {"m": m, "n": n, "k": k, "status": "FAIL", "kernel_source": "",
                "tile": "", "tflops": "", "runtime_us": "", "kernel_name": "",
                "norm_error": "", "atol": "", "rtol": "", "correctness": ""}

    kernel_source = ""
    kernel_name = ""
    tile = ""
    for line in result.stderr.splitlines():
        match = re.search(
            r"\[KERNEL_SOURCE\].*source=(\w+)"
            r"(?:.*tile=(\S+))?"
            r"(?:.*kernel=(\S+))?", line)
        if match:
            kernel_source = match.group(1)
            tile = match.group(2) or ""
            kernel_name = match.group(3) or ""

    header_line = None
    data_line = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("["):
            # [0]: header or data — strip prefix
            stripped = re.sub(r"^\[\d+\]:", "", stripped)
        if "hipblaslt-Gflops" in stripped:
            header_line = stripped
        elif stripped and header_line and stripped[0] in "TN0123456789-":
            data_line = stripped

    if not header_line or not data_line:
        print(f"  PARSE ERROR m={m} n={n} k={k}", file=sys.stderr)
        return {"m": m, "n": n, "k": k, "status": "PARSE_ERROR", "kernel_source": kernel_source,
                "tile": tile, "tflops": "", "runtime_us": "", "kernel_name": kernel_name,
                "norm_error": "", "atol": "", "rtol": "", "correctness": ""}

    fields = [f.strip() for f in header_line.split(",")]
    values = [v.strip() for v in data_line.split(",")]
    row = dict(zip(fields, values))

    gflops_str = row.get("hipblaslt-Gflops", "")
    us_str = row.get("us", "")

    try:
        tflops = float(gflops_str) / 1e3
    except (ValueError, TypeError):
        tflops = ""

    norm_error = row.get("norm_error", "")
    atol = row.get("atol", "")
    rtol = row.get("rtol", "")

    correctness = ""
    if norm_error != "":
        try:
            correctness = "PASS" if float(norm_error) == 0 else "FAIL"
        except ValueError:
            correctness = "FAIL"
    if atol == "failed" or rtol == "failed":
        correctness = "FAIL"

    return {
        "m": m,
        "n": n,
        "k": k,
        "status": "OK",
        "kernel_source": kernel_source,
        "tile": tile,
        "kernel_name": kernel_name,
        "tflops": f"{tflops:.3f}" if isinstance(tflops, float) else "",
        "gflops": gflops_str,
        "runtime_us": us_str,
        "norm_error": norm_error,
        "atol": atol,
        "rtol": rtol,
        "correctness": correctness,
    }


def load_shapes_file(path):
    """Load tagged shapes from a CSV file with columns: tag,m,n,k"""
    tagged = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tagged.append((row["tag"], int(row["m"]), int(row["n"]), int(row["k"])))
    return tagged


def main():
    parser = argparse.ArgumentParser(description="Benchmark hipblaslt across GEMM shapes")
    parser.add_argument("--no-build", action="store_true", help="Skip ninja rebuild")
    parser.add_argument("--iters", type=int, default=3, help="Hot iterations per shape")
    parser.add_argument("-o", "--output", default="benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--shapes", nargs="*", help="Shapes as MxNxK (e.g. 4096x4096x4096)")
    parser.add_argument("--shapes-file", help="CSV file with columns: tag,m,n,k")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="Ninja parallel jobs")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to run benchmarks on in parallel (default: 1)")
    args = parser.parse_args()

    if not args.no_build:
        build_bench(args.jobs)

    if args.shapes_file:
        tagged_shapes = load_shapes_file(args.shapes_file)
    elif args.shapes:
        tagged_shapes = []
        for s in args.shapes:
            parts = s.split("x")
            if len(parts) != 3:
                print(f"Invalid shape: {s}, expected MxNxK", file=sys.stderr)
                sys.exit(1)
            tagged_shapes.append(("", int(parts[0]), int(parts[1]), int(parts[2])))
    else:
        tagged_shapes = [("", m, n, k) for m, n, k in DEFAULT_SHAPES]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/library:{BUILD_DIR}/rocroller" + \
        (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")

    has_tags = any(t for t, *_ in tagged_shapes)
    csv_fields = (["tag"] if has_tags else []) + [
        "m", "n", "k", "status", "kernel_source", "tile", "kernel_name",
        "tflops", "gflops", "runtime_us",
        "norm_error", "atol", "rtol", "correctness"]

    total = len(tagged_shapes)
    num_gpus = max(1, args.num_gpus)
    print_lock = threading.Lock()

    if num_gpus == 1:
        results = []
        for i, (tag, m, n, k) in enumerate(tagged_shapes, 1):
            tag_str = f" [{tag}]" if tag else ""
            print(f"[{i}/{total}] Benchmarking m={m} n={n} k={k}{tag_str} ...", file=sys.stderr)
            row = run_one(m, n, k, args.iters, env)
            if has_tags:
                row["tag"] = tag
            results.append(row)
            status = row["status"]
            src = row.get("kernel_source", "")
            t = row.get("tile", "")
            tflops = row.get("tflops", "")
            us = row.get("runtime_us", "")
            corr = row.get("correctness", "")
            print(f"  -> {status}  source={src}  tile={t}  {tflops} TFLOPS  {us} us  verify={corr}", file=sys.stderr)
    else:
        gpu_q: queue.Queue[int] = queue.Queue()
        for gid in range(num_gpus):
            gpu_q.put(gid)

        results = [None] * total
        completed = [0]

        def _run(idx, tag, m, n, k):
            row = run_one(m, n, k, args.iters, env, gpu_queue=gpu_q)
            if has_tags:
                row["tag"] = tag
            with print_lock:
                completed[0] += 1
                tag_str = f" [{tag}]" if tag else ""
                status = row["status"]
                src = row.get("kernel_source", "")
                t = row.get("tile", "")
                tflops = row.get("tflops", "")
                us = row.get("runtime_us", "")
                corr = row.get("correctness", "")
                print(f"[{completed[0]}/{total}] m={m} n={n} k={k}{tag_str}"
                      f"  -> {status}  source={src}  tile={t}  {tflops} TFLOPS  {us} us"
                      f"  verify={corr}",
                      file=sys.stderr)
            return idx, row

        print(f"Running {total} benchmarks across {num_gpus} GPUs ...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            futures = [
                pool.submit(_run, i, tag, m, n, k)
                for i, (tag, m, n, k) in enumerate(tagged_shapes)
            ]
            for fut in as_completed(futures):
                idx, row = fut.result()
                results[idx] = row

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r.get(field, "") for field in csv_fields})

    print(f"\nResults written to {output_path}", file=sys.stderr)

    tag_col = f"{'Tag':<20} " if has_tags else ""
    print(f"\n{tag_col}{'M':>8} {'N':>8} {'K':>8}  {'Source':<10} {'Tile':<16} {'TFLOPS':>8} {'Runtime(us)':>12}  {'Verify':<6}  Status")
    print("-" * (116 + (21 if has_tags else 0)))
    for r in results:
        tag_col = f"{r.get('tag',''):<20} " if has_tags else ""
        print(f"{tag_col}{r['m']:>8} {r['n']:>8} {r['k']:>8}  "
              f"{r.get('kernel_source',''):.<10} "
              f"{r.get('tile',''):<16} "
              f"{r.get('tflops',''):>8} "
              f"{r.get('runtime_us',''):>12}  "
              f"{r.get('correctness',''):<6}  "
              f"{r['status']}")


if __name__ == "__main__":
    main()
