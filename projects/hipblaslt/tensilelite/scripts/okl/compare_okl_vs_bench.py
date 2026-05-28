#!/usr/bin/env python3
"""Compare okl_run vs hipblaslt-bench TFLOPS for each kernel in ./packages.

For each <pkg>/kernel.conf:
  - Parse M, N, K, batch from the conf and solution_index from okl.json.
  - Run ./okl_run <pkg>/kernel.conf -- capture TFLOPS + microseconds/iter.
  - Run hipblaslt-bench in CPU-timer mode with --algo_method index -- same.
  - Run hipblaslt-bench with --use_gpu_timer -- same.
  - Compute the wrapper's advantage vs the GPU-timer bench number (the
    cleanest apples-to-apples comparison since both exclude host wall time).

Writes:
  packages/comparison.json  machine-readable
  packages/comparison.md    human-readable table + commentary

Run from this folder:
  python3 compare_okl_vs_bench.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
OUT_ROOT = HERE / "packages"
OKL_RUN  = HERE / "okl_run"

# Hardware-specific tooling -- adjust to match your install.
BENCH   = "/opt/rocm-6.4.3/bin/hipblaslt-bench"
LIBPATH = "/opt/rocm-6.4.3/lib/hipblaslt/library"

# Match okl_run's steady-state methodology.
ITERS = 500
COLD  = 500


def parse_conf(path):
    """Tiny key=value parser for kernel.conf; ignores '#' comments and
    `slot = ...` / `buffer = ...` mini-DSL lines (we only need the scalars)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        if k in ("slot", "buffer"):
            continue
        out[k] = v
    return out


def run_okl_run(conf_path):
    """Returns (gflops, us_per_iter) from one ./okl_run invocation."""
    out = subprocess.run(
        [str(OKL_RUN), str(conf_path)],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    gflops = float(re.search(r"^perf:\s+([\d.]+)", out, re.M).group(1))
    us     = float(re.search(r"^time:\s+([\d.]+)", out, re.M).group(1))
    return gflops, us


def run_bench(M, N, K, solution_index, *, use_gpu_timer):
    """Run hipblaslt-bench against a specific solution and return
    (gflops, us_per_iter) parsed from the CSV row.

    Trailing CSV fields are: ...,hipblaslt-Gflops,hipblaslt-GB/s,us
    """
    args = [
        BENCH,
        "-m", str(M), "-n", str(N), "-k", str(K),
        "--transA", "T", "--transB", "N",
        "--a_type", "bf16_r", "--b_type", "bf16_r",
        "--c_type", "bf16_r", "--d_type", "bf16_r",
        "--compute_type", "f32_r",
        "--algo_method", "index", "--solution_index", str(solution_index),
        "--iters", str(ITERS), "--cold_iters", str(COLD),
    ]
    if use_gpu_timer:
        args.append("--use_gpu_timer")
    env = {**os.environ, "HIPBLASLT_TENSILE_LIBPATH": LIBPATH}
    out = subprocess.run(args, capture_output=True, text=True,
                         timeout=600, env=env).stdout
    for raw in out.splitlines():
        s = raw.strip()
        if s and s[0] in "NT" and "," in s and "bf16_r" in s:
            fields = [x.strip() for x in s.split(",")]
            return float(fields[-3]), float(fields[-1])
    return None, None


def render_markdown(results):
    md = []
    md.append("# okl_run vs hipblaslt-bench timing comparison")
    md.append("")
    md.append(f"Same kernel by index in both runners, {ITERS} hot iters after "
              f"{COLD} cold iters, bf16 TN. CPU-timer bench is hipblaslt-bench's "
              "default; GPU-timer adds `--use_gpu_timer`.")
    md.append("")
    md.append(f"- Bench   : `{BENCH}`")
    md.append(f"- Library : `{LIBPATH}`")
    md.append("")
    md.append("| Package | Shape | sol idx | okl_run | bench (CPU) | bench (GPU) | wrapper Δ | bench extra |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.append("| | | | TFLOPS / µs | TFLOPS / µs | TFLOPS / µs | vs bench-GPU | µs/iter |")
    for r in results:
        md.append(
            f"| `{r['name']}` "
            f"| {r['M']}×{r['N']}×{r['K']} "
            f"| {r['solution_index']} "
            f"| {r['okl_run']['tflops']:.1f} / {r['okl_run']['us_per_iter']:.2f} "
            f"| {r['bench_cpu']['tflops']:.1f} / {r['bench_cpu']['us_per_iter']:.2f} "
            f"| {r['bench_gpu']['tflops']:.1f} / {r['bench_gpu']['us_per_iter']:.2f} "
            f"| {r['wrapper_vs_bench_gpu_pct']:+.1f}% "
            f"| +{r['bench_extra_us_per_iter']:.2f} |"
        )
    md.append("")
    md.append("## Reading")
    md.append("")
    md.append("- `okl_run` loads the same .co, packs the kernarg once, and "
              "launches in a tight loop via raw `hipExtModuleLaunchKernel`. "
              "No Tensile / hipBLASLt link.")
    md.append("- `hipblaslt-bench` wraps each launch in `hipblasLtMatmul`, "
              "which validates args, looks up the algo, manages workspace, "
              "and can launch additional helper kernels per call.")
    md.append("- **bench extra µs/iter** = `bench_gpu_us - okl_us`. For "
              "tiny shapes this looks like CPU-side API marshaling; for "
              "large shapes it scales with kernel size, which points at "
              "on-stream workspace/state management hipBLASLt does that "
              "the raw launch path skips.")
    md.append("- For comparing this kernel against a non-hipBLASLt "
              "implementation (cuBLAS, custom assembly), the **okl_run** "
              "number is fairer -- both sides pay only for kernel work. "
              "For predicting what a real hipBLASLt user observes, the "
              "**bench** number is.")
    md.append("")
    md.append("## Per-package kernel symbols")
    md.append("")
    for r in results:
        md.append(f"- `{r['name']}`: `{r['kernel_symbol']}`")
    md.append("")
    return "\n".join(md)


def main():
    if not OKL_RUN.exists():
        sys.exit(f"error: {OKL_RUN} not built. "
                 "Run: /opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run")

    packages = sorted(p for p in OUT_ROOT.iterdir()
                      if p.is_dir() and (p / "kernel.conf").exists())
    if not packages:
        sys.exit(f"no packages in {OUT_ROOT}. Run: python3 package_examples.py")

    results = []
    for pkg in packages:
        conf = parse_conf(pkg / "kernel.conf")
        info = json.loads((pkg / "okl.json").read_text())
        sol_idx = info["solution_index"]
        M, N, K = int(conf["m"]), int(conf["n"]), int(conf["k"])
        batch   = int(conf.get("batch", 1))

        print(f"== {pkg.name}: {M}x{N}x{K} (sol {sol_idx}) ==")
        okl_gf, okl_us = run_okl_run(pkg / "kernel.conf")
        print(f"  okl_run:        {okl_gf/1000:8.1f} TFLOPS  {okl_us:8.2f} us/iter")
        cpu_gf, cpu_us = run_bench(M, N, K, sol_idx, use_gpu_timer=False)
        print(f"  bench (CPU):    {cpu_gf/1000:8.1f} TFLOPS  {cpu_us:8.2f} us/iter")
        gpu_gf, gpu_us = run_bench(M, N, K, sol_idx, use_gpu_timer=True)
        print(f"  bench (GPU):    {gpu_gf/1000:8.1f} TFLOPS  {gpu_us:8.2f} us/iter")

        delta_pct = (okl_gf - gpu_gf) / gpu_gf * 100
        print(f"  wrapper Δ:      {delta_pct:+.1f}%   "
              f"(bench +{gpu_us - okl_us:.2f} us/iter extra)")
        print()

        results.append({
            "name":           pkg.name,
            "M": M, "N": N, "K": K, "batch": batch,
            "solution_index": sol_idx,
            "kernel_symbol":  conf["kernel_symbol"],
            "okl_run":     {"tflops": okl_gf / 1000, "us_per_iter": okl_us},
            "bench_cpu":   {"tflops": cpu_gf / 1000, "us_per_iter": cpu_us},
            "bench_gpu":   {"tflops": gpu_gf / 1000, "us_per_iter": gpu_us},
            "wrapper_vs_bench_gpu_pct": delta_pct,
            "bench_extra_us_per_iter":  gpu_us - okl_us,
        })

    json_path = OUT_ROOT / "comparison.json"
    md_path   = OUT_ROOT / "comparison.md"
    json_path.write_text(json.dumps({
        "iters":     ITERS,
        "cold_iters": COLD,
        "bench":     BENCH,
        "libpath":   LIBPATH,
        "results":   results,
    }, indent=2))
    md_path.write_text(render_markdown(results))

    print("=== Wrote:")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
