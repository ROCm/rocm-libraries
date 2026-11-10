#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full BATCHED GEMM benchmark sweep driven through the Dispatcher bridge.

Batched counterpart of gemm_full_benchmark.py, whose measurement it shares
(``benchmark_on_devices``: device-pinned workers, failure records, interrupt
handling) along with its pair lists and its adaptive search mode. A batched
problem has a batch count besides M, N and K: its key in --pairs,
--list-pairs and the search's files is "BxMxNxK" (batch_count first, as in
the CSV), and its kernels run in the GEMM worker on ``GpuBatchedGemmRunner``
(run_one_batched_gemm_kernel.py).

Phases:
  Phase 1: Compile all batched kernels (parallel, returns .so paths -- no GPU)
  Phase 2: Load batched problems (batch_count, M, N, K)
  Phase 3: Benchmark via subprocess isolation, fanned across all visible GPUs

Tile Engine generates NO binaries here: it expands its sweep config into shared
``BatchedGemmKernelConfig`` objects and hands them to the dispatcher, which
codegens + compiles each into a .so. Each kernel runs in a disposable worker
subprocess so a GPU fault takes down only one worker.

``--search-config`` replaces the build and the full measurement with the
adaptive kernel search (``gemm_search.py``, see "Batched GEMM search" in
README.md): only the kernels it proposes are built, every measurement is
verified, and the confirmed winners are written as a --pairs file.

Examples:
    python batched_gemm_full_benchmark.py
    python batched_gemm_full_benchmark.py \
        batched_gemm/configs/default_config.json --devices 4 --csv out.csv
    # every (problem, kernel) pair a full run measures, nothing built:
    python batched_gemm_full_benchmark.py batched_gemm/configs/default_config.json \
        --arch gfx950 --dtype bf16 --layout rcr --problems problems.json \
        --list-pairs pairs.json
    # the adaptive search over the same pairs, then a plain run of its winners:
    python batched_gemm_full_benchmark.py batched_gemm/configs/default_config.json \
        --arch gfx950 --dtype bf16 --layout rcr --problems problems.json \
        --search-config batched_gemm/search/search_settings_gfx950.json \
        --search-output-dir runs/search-001
    python batched_gemm_full_benchmark.py batched_gemm/configs/default_config.json \
        --arch gfx950 --dtype bf16 --layout rcr --problems problems.json \
        --pairs runs/search-001/winners.json --verify --csv recheck.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# Sets up the dispatcher and common paths this driver's imports need too.
import gemm_full_benchmark as gfb  # noqa: E402
from gemm_full_benchmark import resolve_devices  # noqa: E402
from batched_gemm_utils import (  # noqa: E402
    BATCHED_SUPPORTED_DTYPES as SUPPORTED_DTYPES,
    BATCHED_SUPPORTED_LAYOUTS as SUPPORTED_LAYOUTS,
    BATCHED_VERIFY_TOL,
    _resolve_arch,
    setup_multiple_batched_gemm_dispatchers,
    expand_sweep,
)
from gemm_vector_fallback import VectorFallback, add_vector_fallback_arg  # noqa: E402

# The batched op keeps its sweep configs in tile_engine/ops/gemm/batched_gemm/configs.
CONFIG_DIR = _THIS_DIR / "batched_gemm" / "configs"
CI_CONFIG_NAME = "default_ci_config.json"
DEFAULT_CONFIG_NAME = "default_config.json"
EXAMPLE_PROBLEMS_NAME = "example_problems.json"
# A batched problem's fields, in the order of its key ("BxMxNxK") and the CSV.
DIMS = ("batch_count", "M", "N", "K")
WORKER = "run_one_batched_gemm_kernel.py"

# Fallback batched problem set if none is supplied.
DEFAULT_PROBLEMS = [
    {"batch_count": 8, "M": 1024, "N": 1024, "K": 1024},
    {"batch_count": 4, "M": 2048, "N": 2048, "K": 2048},
    {"batch_count": 16, "M": 512, "N": 512, "K": 512},
    {"batch_count": 2, "M": 3840, "N": 4096, "K": 2048},
]


def resolve_configs(args):
    """Resolve positional configs -> concrete list of config paths."""
    if args.configs:
        return args.configs
    return [str(CONFIG_DIR / CI_CONFIG_NAME)]


def load_problems(path):
    if path:
        with open(path) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    example = CONFIG_DIR / EXAMPLE_PROBLEMS_NAME
    if example.exists():
        with open(example) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    return DEFAULT_PROBLEMS


def main():
    parser = argparse.ArgumentParser(
        description="Batched GEMM Benchmark Sweep (via Dispatcher)"
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="TE sweep config JSON files (default: batched_gemm/configs/default_ci_config.json; "
        "on gfx1250/MI400 pass batched_gemm/configs/default_ci_config_gfx1250.json for WMMA 16x16x32)",
    )
    # Default None so the bridge utilities auto-detect the actual GPU arch via
    # rocminfo (_resolve_arch); never hardcode gfx942 -- that would build an
    # incompatible kernel on gfx90a/gfx950/gfx1250 and launch it on the visible device.
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU arch (gfx90a/gfx942/gfx950/gfx1250); default: auto-detect via rocminfo.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=SUPPORTED_DTYPES,
        help=f"Input dtype (supported: {', '.join(SUPPORTED_DTYPES)})",
    )
    parser.add_argument(
        "--layout",
        default="rcr",
        choices=SUPPORTED_LAYOUTS,
        help=f"A/B/C layout (supported: {', '.join(SUPPORTED_LAYOUTS)})",
    )
    parser.add_argument(
        "--problems", default=None, help="JSON file of batched problems"
    )
    parser.add_argument("--csv", type=str, default="batched_gemm_results.csv")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel build workers"
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="GPUs to use: int count or comma-list of ids; default auto-detect",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Kernels per subprocess (overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout", type=int, default=30, help="Per-kernel timeout (s)"
    )
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=0,
        help="Limit to first N kernels plus their vector-width variants (0=all)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check each kernel's output against an fp32 numpy reference "
        "(per-batch A @ B); a mismatch counts as a failure",
    )
    parser.add_argument(
        "--verify-tol",
        type=float,
        default=None,
        help="Relative tolerance for --verify (default per dtype: "
        + ", ".join(f"{d} {t:g}" for d, t in BATCHED_VERIFY_TOL.items())
        + ")",
    )
    add_vector_fallback_arg(parser)
    gfb.add_pair_and_search_args(parser, key="BxMxNxK")
    args = parser.parse_args()
    if args.verify_tol is None:
        args.verify_tol = BATCHED_VERIFY_TOL[args.dtype]

    # --batch-size is the step of range(0, len(built_kernels), args.batch_size);
    # a zero step raises ValueError and a negative one silently yields no batches,
    # so reject non-positive values up front with a clear message.
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer (kernels per subprocess)")
    search_settings = gfb.check_pair_and_search_args(
        parser, args, [("--csv", args.csv), ("--pairs", args.pairs)], "batched_gemm"
    )

    arch = _resolve_arch(args.arch)
    config_paths = resolve_configs(args)
    devices = resolve_devices(args.devices)

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile batched kernels")
    print(f"{'=' * 80}")
    print(f"  Configs: {', '.join(config_paths)}")

    problems = load_problems(args.problems)
    vfb = VectorFallback(
        problems, args.layout, args.dtype, "batched", args.no_vector_fallback,
        args.tune_c_vector_width,
    )

    all_configs = []
    for cfg_path in config_paths:
        all_configs.extend(
            expand_sweep(
                cfg_path,
                arch,
                dtype=args.dtype,
                layout=args.layout,
                **vfb.expand_kwargs,
            )
        )

    vfb.report_rejects()

    all_configs = vfb.limit_base_kernels(all_configs, args.max_kernels)

    print(f"  Expanded configs: {len(all_configs)}")
    context = {"arch": arch, "layout": args.layout, "variant": "batched",
               "configs": config_paths}
    if args.list_pairs:
        header = {"arch": arch, "dtype": args.dtype, **context}
        gfb.write_pair_listing(args.list_pairs, header, problems, all_configs, vfb,
                               DIMS)
        return 0

    wanted = paired = None
    if args.pairs:
        listed = gfb.read_listed_pairs(args.pairs, problems, all_configs, vfb,
                                       search_settings is not None, DIMS)
        if listed is None:
            return 1
        wanted, paired, star, needed = listed
        all_configs = [cfg for cfg in all_configs if cfg.name in needed]
    if search_settings is not None:
        return gfb.run_search_mode(
            args, search_settings, problems, all_configs, vfb,
            None if wanted is None else (wanted, paired), devices,
            build=setup_multiple_batched_gemm_dispatchers,
            worker=gfb.worker_setup(WORKER), context=context, dims=DIMS,
        )
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    lib_paths = setup_multiple_batched_gemm_dispatchers(
        all_configs, verbose=True, max_workers=args.workers
    )
    build_time = time.perf_counter() - t0

    vfb.report_builds(all_configs, lib_paths)
    built_kernels, duplicate_count = gfb.unique_built_kernels(all_configs, lib_paths)

    print(
        f"\n  Built {len(all_configs)} configs -> {len(built_kernels)} unique kernels "
        f"({duplicate_count} duplicates filtered) in {build_time:.0f}s"
    )

    if not built_kernels:
        print("  ERROR: No kernels built successfully")
        return 1

    # ========================================================================
    # Phase 2: Load problems
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 2: Load test problems")
    print(f"{'=' * 80}")

    if wanted is None:
        pairs = vfb.pairs(problems, built_kernels)
    else:
        pairs, _ = gfb.measured_listed_pairs(
            problems, wanted, paired, star, built_kernels, vfb, DIMS
        )

    # ========================================================================
    # Phase 3: Benchmark across all visible GPUs (subprocess isolation, batched)
    # ========================================================================
    worker_path, base_env = gfb.worker_setup(WORKER)
    csv_path = Path(args.csv)
    # The workers' fp8/bf8 host format follows the arch the kernels were built for.
    stats = gfb.benchmark_on_devices(
        problems, pairs, built_kernels, devices, args, worker_path, base_env,
        csv_path, {"arch": arch}, dims=DIMS,
    )

    gfb.print_summary(build_time, stats, devices, csv_path)
    return 1 if stats["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
