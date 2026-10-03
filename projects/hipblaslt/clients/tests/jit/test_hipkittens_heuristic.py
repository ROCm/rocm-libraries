#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HipKittens in JIT heuristic queries: hipblaslt-bench --verify, through
hipblaslt_ext::Gemm with beta 0 and through the C API with beta 1, returns it
after TensileLite when HIPBLASLT_JIT_BACKENDS names it, TensileLite alone when
it does not or when its headers are missing, and hipblasLtMatmul without an
algorithm runs it when it comes first."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from test_jit_gemm import check_numerics, require  # noqa: E402

KERNEL = "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5"
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
# the bench's beta, "c" through hipblasLtMatmulAlgoGetHeuristic, which assumes beta 1.
def bench(args, name, requested, api="mix", beta="0", library="jit-library", **variables):
    result = run(
        args.output,
        name,
        [str(args.bench), "-m", "1024", "-n", "512", "-k", "768",
         "--transA", "T", "--transB", "N",
         "--a_type", "bf16_r", "--b_type", "bf16_r", "--c_type", "bf16_r",
         "--d_type", "bf16_r", "--compute_type", "f32_r", "--alpha", "1", "--beta", beta,
         "--api_method", api, "--requested_solution", str(requested),
         "--verify", "--iters", "3", "--cold_iters", "1", "--print_kernel_info"],
        args.output / library,
        **variables,
    )
    check_numerics(result.stdout)
    listed = result.stdout.split("Winner:")[0]
    return re.findall(r"^\s*--kernel name:\s+(\S+)$", listed, re.M), result.stderr


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

    kernels, _ = bench(args, "mixed", 4, HIPBLASLT_JIT_BACKENDS="tensilelite,hipkittens")
    tensilelite = kernels[:-1]
    require(
        kernels[-1:] == [KERNEL] and tensilelite
        and all(kernel.startswith("Cijk_") for kernel in tensilelite),
        f"Expected TensileLite kernels, then {KERNEL}: {kernels}",
    )
    require(
        backends(args.output / "jit-library") == ["hipkittens", "tensilelite"],
        "Expected one JIT library entry for each backend",
    )
    print(f"PASS mix --verify: {len(tensilelite)} TensileLite solutions, then {KERNEL}")

    kernels, _ = bench(
        args, "c-api", 4, api="c", beta="1", library="c-api-library",
        HIPBLASLT_JIT_BACKENDS="tensilelite,hipkittens",
    )
    require(
        kernels[-1:] == [KERNEL] and kernels[:-1]
        and all(kernel.startswith("Cijk_") for kernel in kernels[:-1]),
        f"C API with beta 1, expected TensileLite kernels, then {KERNEL}: {kernels}",
    )
    print(f"PASS C API beta 1 --verify: {len(kernels) - 1} TensileLite solutions, then {KERNEL}")

    kernels, stderr = bench(
        args, "default", len(tensilelite), HIPBLASLT_JIT_HIPKITTENS_PATH=str(no_headers)
    )
    require(kernels == tensilelite, f"Unset, expected the TensileLite solutions: {kernels}")
    require("HipKittens" not in stderr, "Unset, HipKittens was configured")
    print("PASS unset: TensileLite alone, and HipKittens is not configured")

    kernels, stderr = bench(
        args,
        "missing",
        len(tensilelite) + 1,
        HIPBLASLT_JIT_BACKENDS="tensilelite,hipkittens",
        HIPBLASLT_JIT_HIPKITTENS_PATH=str(no_headers),
    )
    # TensileLite alone also takes the slot kept for HipKittens.
    require(
        kernels[: len(tensilelite)] == tensilelite and KERNEL not in kernels,
        f"Without headers, expected TensileLite alone: {kernels}",
    )
    reports = [line for line in stderr.splitlines() if MISSING in line]
    require(
        len(reports) == 1 and reports[0].startswith("hipblaslt warning: JIT configure failed"),
        f"Expected one configure warning naming HipKittens: {reports}",
    )
    print("PASS missing headers: one warning, and TensileLite serves alone")

    first = args.output / "hipkittens-first"
    result = run(
        args.output,
        "matmul",
        [str(args.test), "heuristic"],
        first,
        HIPBLASLT_JIT_BACKENDS="hipkittens,tensilelite",
    )
    require(backends(first) == ["hipkittens"], "TensileLite generated for a single solution")
    print(result.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
