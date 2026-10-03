#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Publish a HipKittens solution, then run it with hipblaslt-bench --verify
through hipblasLtMatmul by its solution index, with JIT off, alpha 2, beta 1 and
three batches."""

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bench_smoke import require, rows  # noqa: E402

KERNEL = "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5"


def check_numerics(stdout):
    found = rows(stdout)
    require(found, "No benchmark correctness row")
    for row in found:
        error = float(row["norm_error"])
        require(math.isfinite(error) and error <= 0.1, f"BF16 norm error {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path, help="hipblaslt-jit-hipkittens-test")
    parser.add_argument("bench", type=Path, help="hipblaslt-bench")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    env = dict(os.environ, HIPBLASLT_JIT_LIBRARY_PATH=str(args.output / "jit-library"))

    # The "library" mode publishes M=1024 N=512 K=768 with alpha 1, beta 0 and
    # one batch; the entry serves every alpha, beta and batch count.
    published = subprocess.run(
        [str(args.test), "library"], env=env, text=True, capture_output=True, timeout=600
    )
    (args.output / "publish.log").write_text(published.stdout + published.stderr)
    require(published.returncode == 0, "Publishing the HipKittens solution failed")
    index = re.search(r"^INDEX (\d+)$", published.stdout, re.M).group(1)

    bench = subprocess.run(
        [str(args.bench), "-m", "1024", "-n", "512", "-k", "768",
         "--transA", "T", "--transB", "N",
         "--a_type", "bf16_r", "--b_type", "bf16_r", "--c_type", "bf16_r",
         "--d_type", "bf16_r", "--compute_type", "f32_r", "--alpha", "2", "--beta", "1",
         "--batch_count", "3",
         "--algo_method", "index", "--solution_index", index,
         "--verify", "--iters", "3", "--cold_iters", "1", "--print_kernel_info"],
        env=env, text=True, capture_output=True, timeout=600,
    )
    (args.output / "bench.log").write_text(bench.stdout + bench.stderr)
    require(bench.returncode == 0, f"hipblaslt-bench exited with {bench.returncode}")
    check_numerics(bench.stdout)
    require(
        re.search(rf"--kernel name:\s+{KERNEL}$", bench.stdout, re.M),
        f"hipblaslt-bench did not run {KERNEL}",
    )
    print(f"PASS hipblaslt-bench --verify ran {KERNEL} as index {index}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
