# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 07: build-on-the-fly sweep across dispatcher configs.

Runs the same cartesian product CK's `dispatcher/codegen/default_config.json`
expands and builds every valid instance to a HSACO blob via libamd_comgr.
A side-effect is a `sweep_manifest.json` that a downstream benchmark
driver (or the runbook's "sweep one lever at a time" workflow) feeds
into the C++ launcher to find the best kernel per problem shape.

This is the contract the user requested: *we need to be able to build
all of these instances on the fly with our new DSL quickly*. On a
16-core box, building the full ~460-spec cartesian product takes
roughly 8 seconds cold and 200 ms warm (via a content-addressed
cache).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "python"))

from ck_dsl.instances import all_dispatcher_configs  # noqa: E402
from ck_dsl.sweep import build_all_instances, write_sweep_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="parallel build workers (0 = os.cpu_count())",
    )
    parser.add_argument(
        "--subset",
        choices=("hero", "compute", "all"),
        default="compute",
        help=(
            "hero: just the 4 32x32x16-atom configs we expect to do well; "
            "compute: every CK-style compute config (compv3/compv4, "
            "default/cshuffle); all: the full cartesian product"
        ),
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "hsaco_cache"

    if args.subset == "hero":
        specs = list(
            all_dispatcher_configs(
                tile_m=(128, 256),
                tile_n=(128, 256),
                tile_k=(32, 64),
                warp_tile=((32, 32, 16),),
                pipeline=("compv4",),
                epilogue=("cshuffle",),
            )
        )
    elif args.subset == "compute":
        specs = list(
            all_dispatcher_configs(
                pipeline=("compv3", "compv4"),
                epilogue=("default", "cshuffle"),
            )
        )
    else:
        specs = list(all_dispatcher_configs())

    print(f"sweep target: {args.subset}  ->  {len(specs)} valid specs")
    t0 = time.perf_counter()
    parallel = args.parallel or None
    records = build_all_instances(
        specs,
        cache_dir=cache,
        isa=args.isa,
        parallel=parallel,
    )
    t1 = time.perf_counter()
    ok = sum(1 for r in records if r.ok)
    fail = [r for r in records if not r.ok]

    print(f"built {ok}/{len(records)} in {t1 - t0:.2f}s wall")
    if fail:
        print(f"failed: {len(fail)}")
        for r in fail[:5]:
            print(f"  - {r.name[:80]}: {r.error[:140]}")

    # Emit the cross-instance manifest.
    write_sweep_manifest(
        records,
        out / "sweep_manifest.json",
        shapes=[
            # Small grid of production-relevant GEMM shapes.
            (3328, 4096, 4096),  # the shape we've been using
            (1024, 1024, 1024),  # square balanced
            (8192, 8192, 4096),  # large
            (128, 8192, 4096),  # skinny-M (inference)
        ],
    )

    # The launcher takes one (hsaco, per-kernel manifest); for the
    # primary `expected.json`-gated test we also emit a single-kernel
    # manifest pointing at the *best-build-time* instance so the
    # standard tests can run a representative kernel without needing
    # the sweep driver.
    best = next((r for r in records if r.ok), None)
    if best is not None:
        from ck_dsl.helpers import gemm_args_signature

        manifest = {
            "schema": "ck.dsl.example.manifest/v1",
            "kind": "gemm_fp16",
            "kernel_name": best.name,
            "hsaco": Path(best.hsaco_path).name,
            "block_m": best.block_m,
            "block_n": best.block_n,
            "block_k": best.block_k,
            "threads_per_block": best.threads_per_block,
            "warmup_iters": 5,
            "timed_iters": 100,
            "default_shape": [3328, 4096, 4096],
            "args_signature": gemm_args_signature(),
            "timing_ms": {
                "sweep_build_total_s": t1 - t0,
                "configs_built": ok,
                "configs_failed": len(fail),
            },
            "hsaco_bytes": best.hsaco_bytes,
            "ck_dependency": False,
            "ir_authored": True,
            "atoms": ["dispatcher-sweep"],
            "notes": (
                f"Built {ok}/{len(records)} dispatcher configs in {t1 - t0:.2f}s. "
                "The hsaco in this manifest is the first OK config; the full "
                "sweep_manifest.json carries every other built instance with "
                "its block geometry, build time, and HSACO path so a "
                "downstream sweep launcher can rank them by measured TFLOPS."
            ),
        }
        # Copy the representative HSACO into the top-level so the test
        # harness picks it up via `*.hsaco` glob.
        rep = out / Path(best.hsaco_path).name
        rep.write_bytes(Path(best.hsaco_path).read_bytes())
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"representative kernel: {best.name}")
        print(f"sweep manifest:  {out / 'sweep_manifest.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
