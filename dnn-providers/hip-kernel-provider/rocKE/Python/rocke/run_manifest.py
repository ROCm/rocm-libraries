# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python-native CK DSL manifest runner.

This replaces the C++ `example/ck_tile/dsl/common/launcher.cpp` path
for day-to-day DSL development. The flow:

  1. `gen.py` emits a HSACO blob + `manifest.json`.
  2. Python loads the code object with `hipModuleLoadData`.
  3. Python allocates tensors (torch CUDA tensors), passes their raw pointers
     into `hipModuleLaunchKernel`, verifies with torch reference ops, and times
     with HIP events.

No host C++ compile is involved. The C++ launcher can stay as a CMake/CK-Tile
compatibility target, but this module is the maintained runtime path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .runtime.hip_module import Runtime
from .instances.common.deep_fused_conv_pool import (
    run_deep_fused_conv_pool_fp16_manifest_problem,
)
from .instances.common.manifest_runner.conv import run_conv_manifest_problem
from .instances.common.manifest_runner.gemm import (
    run_batched_gemm_manifest_problem,
    run_gemm_iu8_manifest_problem,
    run_gemm_manifest_problem,
)
from .instances.common.manifest_runner.matmul_nbits import (
    run_matmul_nbits_manifest_problem,
)
from .instances.common.manifest_runner.simple_ops import run_simple_op_manifest_problem
from .instances.gfx1151.deep_fused_conv_pool import (
    run_deep_fused_conv_pool_i8i4_manifest_problem,
)

# Try to import torch-based launcher, fall back to direct HIP timing if unavailable
try:
    from .runtime.launcher import time_launches

    HAS_TORCH_LAUNCHER = True
except ImportError:
    HAS_TORCH_LAUNCHER = False


@dataclass
class RunSummary:
    ms: float
    tflops: float
    gbps: float
    max_abs_diff: float = 0.0
    bad_count: int = 0
    total: int = 0


def _parse_shape(s: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not s:
        return None
    parts = [int(x) for x in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"--shape expects three ints, got {s!r}")
    return parts[0], parts[1], parts[2]


def _load(manifest_path: Path, hsaco_path: Optional[Path]):
    manifest = json.loads(manifest_path.read_text())
    if hsaco_path is None:
        hsaco_path = manifest_path.parent / str(manifest["hsaco"])
    return manifest, hsaco_path.read_bytes(), hsaco_path


def _launch_timed(
    rt: Runtime, fn, grid, block, args: bytes, warmup: int, iters: int
) -> float:
    """Time `iters` repeats of `rt.launch(fn, grid, block, args)` on
    the default stream.

    Delegates to `rocke.runtime.launcher.time_launches` when torch is
    available, so the manifest runner and the in-tree Launcher abstraction
    share one bench-timing path; see that function's docstring for the
    correctness rationale (no per-call module reload, no module unload,
    args buffer lifetime tracked by Runtime._pending_args).

    Falls back to direct HIP event timing when torch is unavailable
    (torch-free environments).
    """
    if HAS_TORCH_LAUNCHER:
        return time_launches(
            lambda: rt.launch(fn, grid, block, args),
            warmup=warmup,
            iters=iters,
        )
    else:
        # Fallback: Direct HIP event timing (torch-free)
        for _ in range(warmup):
            rt.launch(fn, grid, block, args)
        rt.sync()
        e0 = rt.event()
        e1 = rt.event()
        e0.record()
        for _ in range(iters):
            rt.launch(fn, grid, block, args)
        e1.record()
        e1.synchronize()
        total_ms = e0.elapsed_to(e1)
        e0.destroy()
        e1.destroy()
        return total_ms / iters


def run_manifest(
    manifest_path: Path,
    hsaco_path: Optional[Path] = None,
    *,
    shape: Optional[Tuple[int, int, int]] = None,
    verify: bool = False,
) -> RunSummary:
    manifest, blob, _resolved = _load(manifest_path, hsaco_path)
    rt = Runtime()
    module = rt.load_module(blob)
    fn = module.get_function(str(manifest["kernel_name"]))
    kind = str(manifest["kind"])
    if kind == "gemm_fp16":
        make_args, grid, block, flop, bytes_xfer, check = run_gemm_manifest_problem(
            manifest, shape, verify
        )
    elif kind == "gemm_iu8":
        make_args, grid, block, flop, bytes_xfer, check = run_gemm_iu8_manifest_problem(
            manifest, shape, verify
        )
    elif kind == "batched_gemm_fp16":
        make_args, grid, block, flop, bytes_xfer, check = (
            run_batched_gemm_manifest_problem(manifest, shape, verify)
        )
    elif kind in ("conv_fp16", "conv_bf16", "conv_fp32"):
        make_args, grid, block, flop, bytes_xfer, check = run_conv_manifest_problem(
            manifest, shape, verify
        )
    elif kind == "matmul_nbits_fp16":
        make_args, grid, block, flop, bytes_xfer, check = (
            run_matmul_nbits_manifest_problem(manifest, shape, verify)
        )
    elif kind == "deep_fused_conv_pool_i8i4":
        make_args, grid, block, flop, bytes_xfer, check = (
            run_deep_fused_conv_pool_i8i4_manifest_problem(manifest, shape, verify)
        )
    elif kind == "deep_fused_conv_pool_fp16":
        make_args, grid, block, flop, bytes_xfer, check = (
            run_deep_fused_conv_pool_fp16_manifest_problem(manifest, shape, verify)
        )
    elif kind in (
        "elementwise_fp16",
        "reduce_fp16",
        "layernorm_fp16",
        "rmsnorm_fp16",
        "transpose_fp16",
    ):
        make_args, grid, block, flop, bytes_xfer, check = (
            run_simple_op_manifest_problem(manifest, shape, verify)
        )
    else:
        raise ValueError(f"unsupported manifest kind {kind!r}")

    args, ptrs = make_args(rt)
    warmup = int(manifest.get("warmup_iters", 5))
    iters = int(manifest.get("timed_iters", 100))
    ms = _launch_timed(rt, fn, grid, block, args, warmup, iters)
    max_abs_diff, bad_count, total = check(rt, ptrs)
    for ptr in ptrs:
        rt.free(ptr)
    module.unload()
    return RunSummary(
        ms=ms,
        tflops=flop / 1e9 / ms,
        gbps=bytes_xfer / 1e6 / ms,
        max_abs_diff=max_abs_diff,
        bad_count=bad_count,
        total=total,
    )


def main(argv: Optional[list[str]] = None) -> int:
    # Pin the newest (torch-bundled) comgr/HIP runtime before loading any HSACO,
    # so runtime/flavor selection cannot depend on import order: a stale
    # /opt/rocm otherwise loads a disjoint HIP runtime (hipError 500 / wrong LLVM
    # flavor on a torch-newer-than-system venv). No-op when torch is absent.
    from .runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()
    ap = argparse.ArgumentParser()
    ap.add_argument("hsaco")
    ap.add_argument("manifest")
    ap.add_argument("--shape", default=None)
    ap.add_argument("--verify", action="store_true")
    ns = ap.parse_args(argv)
    summary = run_manifest(
        Path(ns.manifest),
        Path(ns.hsaco),
        shape=_parse_shape(ns.shape),
        verify=ns.verify,
    )
    if ns.verify:
        print(
            f"verify max_abs_diff={summary.max_abs_diff:.8g} "
            f"bad={summary.bad_count}/{summary.total}"
        )
    print(
        f"Perf: {summary.ms:.6g} ms, {summary.tflops:.6g} TFlops, {summary.gbps:.6g} GB/s"
    )
    # Machine-readable line for tooling in the same package (e.g. the GEMM
    # sweep harness). Parse this rather than the human "Perf:" string so that
    # formatting changes above never silently drop metrics.
    print(
        "PerfJSON: "
        + json.dumps(
            {
                "ms": summary.ms,
                "tflops": summary.tflops,
                "gbps": summary.gbps,
                "max_abs_diff": summary.max_abs_diff,
                "bad_count": summary.bad_count,
                "total": summary.total,
            }
        )
    )
    if ns.verify and summary.bad_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
