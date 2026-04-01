# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import json
import subprocess
from pathlib import Path


class FlatmmBenchmark:
    def __init__(self, build_dir: str, verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose

    def discover_kernels(self):
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            raise FileNotFoundError(f"Binary directory {bin_dir} does not exist")
        return sorted(bin_dir.glob("benchmark_flatmm_*"))

    def run_kernel(self, kernel_path: Path, params):
        cmd = [str(kernel_path)]
        for key, value in params.items():
            cmd.append(f"-{key}={value}")
        cmd.append("-json_output=true")

        if self.verbose:
            print("Running:", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"{kernel_path.name} failed: {result.stderr.strip()}")
        return json.loads(result.stdout.strip())

    def benchmark_problem(self, params):
        kernels = self.discover_kernels()
        results = []
        for kernel_path in kernels:
            payload = self.run_kernel(kernel_path, params)
            perf = payload.get("perf_result", {})
            results.append(
                {
                    "name": payload.get("name", kernel_path.stem),
                    "latency_ms": perf.get("latency(ms)", 0),
                    "tflops": perf.get("tflops(TFlops)", 0),
                    "bandwidth_gb_s": perf.get("bandwidth(GB/s)", 0),
                    "payload": payload,
                }
            )
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark all tile-engine FLATMM kernels")
    parser.add_argument("--build_dir", required=True, help="CMake build directory")
    parser.add_argument("--m", type=int, default=3840)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--split_k", type=int, default=1)
    parser.add_argument("--verify", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--flush_cache", default="true")
    parser.add_argument("--rotating_count", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    benchmark = FlatmmBenchmark(args.build_dir, verbose=args.verbose)
    params = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "split_k": args.split_k,
        "verify": args.verify,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "flush_cache": args.flush_cache,
        "rotating_count": args.rotating_count,
    }

    results = benchmark.benchmark_problem(params)
    if not results:
        print("No results")
        return

    best = max(results, key=lambda item: item["tflops"])
    print(json.dumps(best["payload"], indent=2))


if __name__ == "__main__":
    main()
