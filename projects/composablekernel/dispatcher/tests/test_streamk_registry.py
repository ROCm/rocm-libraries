#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Stream-K deep-core registry test (requires a GPU + hipcc).

Guards the deep-core path that lets Stream-K ride the registry like regular GEMM:
codegen -> generated SK wrapper -> Registry -> Dispatcher::run() (workspace alloc
+ strategy-aware reset) -> generated_tile_backend_streamk -> verify vs reference.

Each reduction strategy (atomic/linear/tree) is a *distinct compiled kernel*
(SkReductionStrategy is a compile-time constexpr), so we generate all three from a
single tile config and build the 04 registry driver once per strategy, force-
including that strategy's header. For each we assert:
  * the encode_identifier() suffix matches the strategy (..._streamk[_linear|_tree])
  * the Dispatcher selects that kernel by Problem::reduction_strategy
  * the result verifies against the reference GEMM

The test SKIPs (exit 77) when no GPU or no hipcc is available, so it is safe in
CPU-only CI; it only runs the heavy build+launch where a GPU is present.

Usage:
    python3 test_streamk_registry.py
    python3 test_streamk_registry.py --arch gfx942 --m 3840 --n 4096 --k 2048
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
CK_DIR = DISPATCHER_DIR.parent
CODEGEN = DISPATCHER_DIR / "codegen" / "unified_gemm_codegen.py"
DRIVER = DISPATCHER_DIR / "examples" / "gemm" / "cpp" / "04_streamk_registry_driver.cpp"
REGISTRY_SRC = DISPATCHER_DIR / "src" / "registry.cpp"
DISPATCHER_SRC = DISPATCHER_DIR / "src" / "dispatcher.cpp"

SKIP = 77  # ctest SKIP_RETURN_CODE

# One tile config, all three reduction strategies.
TILE = "128x128x64_2x2x1_32x32x16"
TILE_CONFIG_JSON = json.dumps(
    {
        "tile_config": {
            "tile_m": [128], "tile_n": [128], "tile_k": [64],
            "warp_m": [2], "warp_n": [2], "warp_k": [1],
            "warp_tile_m": [32], "warp_tile_n": [32], "warp_tile_k": [16],
            "block_size": [256],
        },
        "trait_config": {
            "pipeline": ["compv3"], "epilogue": ["cshuffle"], "scheduler": ["intrawave"],
            "pad_m": [False], "pad_n": [False], "pad_k": [False], "persistent": [False],
        },
        "streamk_config": {"reduction_strategy": ["atomic", "linear", "tree"]},
    }
)

# strategy -> (header variant suffix, expected encode_identifier suffix)
STRATEGIES = {
    "atomic": ("streamk", "_streamk"),
    "linear": ("streamk_linear", "_streamk_linear"),
    "tree": ("streamk_tree", "_streamk_tree"),
}


def detect_arch(fallback=None):
    try:
        sys.path.insert(0, str(DISPATCHER_DIR / "python"))
        from dispatcher_common import detect_gpu_arch  # noqa: E402

        return detect_gpu_arch()
    except Exception:
        out = shutil.which("rocminfo")
        if out:
            try:
                txt = subprocess.run(
                    ["rocminfo"], capture_output=True, text=True, timeout=30
                ).stdout
                for line in txt.splitlines():
                    if "gfx" in line and "Name:" in line:
                        return line.split("gfx")[1].split()[0].join(["gfx", ""])
            except Exception:
                pass
        return fallback


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--m", type=int, default=3840)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=2048)
    args = ap.parse_args()

    hipcc = shutil.which("hipcc")
    if not hipcc:
        print("SKIP: hipcc not found")
        return SKIP

    arch = args.arch or detect_arch()
    if not arch:
        print("SKIP: no GPU / could not detect gfx arch")
        return SKIP
    print(f"Stream-K registry test on {arch} @ {args.m}x{args.n}x{args.k}")

    inc = ["-I", str(CK_DIR / "include"), "-I", str(DISPATCHER_DIR / "include")]

    with tempfile.TemporaryDirectory(prefix="sk_reg_test_") as td:
        gen = Path(td) / "gen"
        # 1) generate all three strategy headers from one tile config
        g = run(
            [
                sys.executable, str(CODEGEN),
                "--datatype", "fp16", "--layout", "rcr",
                "--gpu-target", arch, "--variants", "stream_k",
                "--tile-config-json", TILE_CONFIG_JSON,
                "--output-dir", str(gen),
            ],
            timeout=600,
        )
        if g.returncode != 0:
            print("FAIL: codegen failed\n" + g.stderr[-2000:])
            return 1

        # 2) build the core objects once (no force-include)
        reg_o, disp_o = Path(td) / "registry.o", Path(td) / "dispatcher.o"
        for src, obj in ((REGISTRY_SRC, reg_o), (DISPATCHER_SRC, disp_o)):
            c = run(
                [hipcc, "-std=c++17", f"--offload-arch={arch}", "-O3", *inc,
                 "-c", str(src), "-o", str(obj)],
                timeout=900,
            )
            if c.returncode != 0:
                print(f"FAIL: compiling {src.name}\n" + c.stderr[-2000:])
                return 1

        failures = []
        for strat, (variant, want_suffix) in STRATEGIES.items():
            header = gen / (
                f"gemm_fp16_rcr_compv3_cshuffle_intrawave_"
                f"False_False_False_False_{TILE}_{variant}.hpp"
            )
            if not header.exists():
                failures.append(f"{strat}: generated header missing ({header.name})")
                continue

            drv_o, exe = Path(td) / f"d_{variant}.o", Path(td) / f"skreg_{variant}"
            c = run(
                [hipcc, "-std=c++17", f"--offload-arch={arch}", "-O3",
                 "-DCK_TILE_SINGLE_KERNEL_INCLUDE", f'-DGFX_ARCH="{arch}"',
                 *inc, "-I", str(gen), "-include", str(header),
                 "-c", str(DRIVER), "-o", str(drv_o)],
                timeout=900,
            )
            if c.returncode != 0:
                failures.append(f"{strat}: driver compile failed\n{c.stderr[-1500:]}")
                continue
            l = run(
                [hipcc, f"--offload-arch={arch}", str(drv_o), str(disp_o),
                 str(reg_o), "-o", str(exe)],
                timeout=300,
            )
            if l.returncode != 0:
                failures.append(f"{strat}: link failed\n{l.stderr[-1500:]}")
                continue

            r = run(
                [str(exe), "--m", str(args.m), "--n", str(args.n),
                 "--k", str(args.k), "--strategy", strat, "--validate", "1"],
                timeout=300,
            )
            out = r.stdout
            ok_verify = "Verification: PASS" in out
            ok_suffix = f"identifier=fp16_rcr" in out and want_suffix in out.split(
                "identifier="
            )[1].split()[0]
            if r.returncode != 0 or not ok_verify or not ok_suffix:
                failures.append(
                    f"{strat}: rc={r.returncode} verify={ok_verify} "
                    f"suffix_ok={ok_suffix}\n{out[-800:]}{r.stderr[-400:]}"
                )
            else:
                tflops = next(
                    (ln for ln in out.splitlines() if "TFlops" in ln), ""
                ).strip()
                print(f"  PASS {strat:6s} -> {want_suffix}  | {tflops}")

        if failures:
            print("\nSTREAM-K REGISTRY TEST FAILED:")
            for f in failures:
                print(" - " + f)
            return 1

    print("All Stream-K strategies registered, dispatched, and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
