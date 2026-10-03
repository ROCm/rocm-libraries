#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HipKittens in JIT heuristic queries: with HIPBLASLT_JIT_BACKENDS=hipkittens,
hipblaslt-bench --verify returns it through hipblaslt_ext::Gemm with beta 0 and
through the C API with beta 1, and hipblasLtMatmul without an algorithm runs
it; with the variable unset it is not configured, and with its headers
missing one warning names it."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hipkittens_bench import KERNEL, check_numerics, require  # noqa: E402

MISSING = "JIT backend HipKittens not available"


def run(output, name, command, library, **variables):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HIPBLASLT_JIT_BACKENDS", "HIPBLASLT_JIT_HIPKITTENS_PATH")
    }
    env.update(HIPBLASLT_JIT="2", HIPBLASLT_JIT_LIBRARY_PATH=str(library), **variables)
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=900)
    (output / f"{name}.log").write_text(result.stdout + result.stderr)
    require(result.returncode == 0, f"{name} exited with {result.returncode}")
    return result


# M=1024 N=512 K=768 BF16 TN; "mix" queries through hipblaslt_ext::Gemm with
# the bench's beta, "c" through hipblasLtMatmulAlgoGetHeuristic, which assumes
# beta 1. Without a solution, hipblaslt-bench reports none and exits with 0.
def bench(args, name, api="mix", beta="0", **variables):
    library = args.output / f"{name}-library"
    result = run(
        args.output,
        name,
        [str(args.bench), "-m", "1024", "-n", "512", "-k", "768",
         "--transA", "T", "--transB", "N",
         "--a_type", "bf16_r", "--b_type", "bf16_r", "--c_type", "bf16_r",
         "--d_type", "bf16_r", "--compute_type", "f32_r", "--alpha", "1", "--beta", beta,
         "--api_method", api, "--requested_solution", "1",
         "--verify", "--iters", "3", "--cold_iters", "1", "--print_kernel_info"],
        library,
        **variables,
    )
    listed = result.stdout.split("Winner:")[0]
    kernels = re.findall(r"^\s*--kernel name:\s+(\S+)$", listed, re.M)
    return kernels, result, backends(library)


def backends(library):
    return sorted(
        json.loads(path.read_text())["backend"]["id"]
        for path in library.glob("v1/*/cache-key.json")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path, help="hipblaslt-jit-hipkittens-test")
    parser.add_argument("bench", type=Path, help="hipblaslt-bench")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    no_headers = args.output / "no-headers"
    no_headers.mkdir()

    for name, api, beta in (("mix", "mix", "0"), ("c-api", "c", "1")):
        kernels, result, keys = bench(args, name, api, beta, HIPBLASLT_JIT_BACKENDS="hipkittens")
        require(kernels == [KERNEL], f"{name} with beta {beta}, expected {KERNEL}: {kernels}")
        check_numerics(result.stdout)
        require(keys == ["hipkittens"], f"{name}: expected one HipKittens key, got {keys}")
        print(f"PASS {name} beta {beta} --verify: {KERNEL}")

    kernels, result, keys = bench(args, "unset", HIPBLASLT_JIT_HIPKITTENS_PATH=str(no_headers))
    require(KERNEL not in kernels and "hipkittens" not in keys, f"Unset, HipKittens served: {kernels}")
    require("HipKittens" not in result.stderr, "Unset, HipKittens was configured")
    print("PASS unset: HipKittens is not configured")

    kernels, result, keys = bench(
        args,
        "missing",
        HIPBLASLT_JIT_BACKENDS="hipkittens",
        HIPBLASLT_JIT_HIPKITTENS_PATH=str(no_headers),
    )
    require(KERNEL not in kernels and not keys, f"Without headers, HipKittens served: {kernels}")
    reports = [line for line in result.stderr.splitlines() if MISSING in line]
    require(
        len(reports) == 1 and "JIT configure failed" in reports[0],
        f"Expected one configure report naming HipKittens: {reports}",
    )
    print("PASS missing headers: one configure report, and no HipKittens solution")

    first = args.output / "matmul-library"
    result = run(
        args.output,
        "matmul",
        [str(args.test), "heuristic"],
        first,
        HIPBLASLT_JIT_BACKENDS="hipkittens",
    )
    require(backends(first) == ["hipkittens"], "hipblasLtMatmul did not publish a HipKittens key")
    print(result.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
