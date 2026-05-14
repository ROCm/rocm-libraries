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
import ctypes
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .runtime.hip_module import Runtime
from .runtime.launcher import time_launches


@dataclass
class RunSummary:
    ms: float
    tflops: float
    gbps: float
    max_abs_diff: float = 0.0
    bad_count: int = 0
    total: int = 0


def _require_numpy():
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("ck_dsl.run_manifest requires numpy") from e
    return np


def _nbytes(a) -> int:
    return int(a.nbytes)


def _as_u8_buffer(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


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
    the default stream. Delegates to `ck_dsl.runtime.launcher.time_launches`
    so the manifest runner and the in-tree Launcher abstraction share
    one bench-timing path; see that function's docstring for the
    correctness rationale (no per-call module reload, no module unload,
    args buffer lifetime tracked by Runtime._pending_args).
    """
    return time_launches(
        lambda: rt.launch(fn, grid, block, args),
        warmup=warmup,
        iters=iters,
    )


def _gemm_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = _require_numpy()
    if shape is None:
        ds = manifest.get("default_shape", [3328, 4096, 4096])
        M, N, K = int(ds[0]), int(ds[1]), int(ds[2])
    else:
        M, N, K = shape
    rng = np.random.default_rng(0xC0FFEE)
    A = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float16)
    B = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float16)
    C = np.empty((M, N), dtype=np.float16)
    grid = (
        (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"]),
        (M + int(manifest["block_m"]) - 1) // int(manifest["block_m"]),
        1,
    )
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * M * N * K
    bytes_xfer = 2.0 * (M * K + N * K + M * N)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(B))
        C_dev = rt.alloc(_nbytes(C))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B), _nbytes(B))
        rt.memset(C_dev, 0, _nbytes(C))
        return struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K), (
            A_dev,
            B_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(_as_u8_buffer(C), ptrs[2], _nbytes(C))
        ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
        diff = np.abs(C.astype(np.float32) - ref.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > 0)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def _conv_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = _require_numpy()
    cv = [int(x) for x in manifest["conv"]]
    if len(cv) < 13:
        raise ValueError("conv manifest needs [N,H,W,C,K,R,S,sH,sW,pH,pW,dH,dW]")
    N, H, W, C, K, R, S, sH, sW, pH, pW, dH, dW = cv[:13]
    groups = int(manifest.get("groups", 1))
    cpg = int(manifest.get("cpg", C // groups))
    kpg = int(manifest.get("kpg", K // groups))
    if groups * cpg != C or groups * kpg != K:
        raise ValueError(
            f"invalid grouping groups={groups} cpg={cpg} kpg={kpg} C={C} K={K}"
        )

    rng = np.random.default_rng(1234)
    A = (rng.random((N, H, W, C), dtype=np.float32) * 0.04 - 0.02).astype(np.float16)
    B = (rng.random((K, R, S, cpg), dtype=np.float32) * 0.04 - 0.02).astype(np.float16)
    D = np.empty((N, H, W, K), dtype=np.float16)

    if "grid_explicit" in manifest:
        gx, gy, gz = [int(x) for x in manifest["grid_explicit"]]
    else:
        bm = int(manifest["block_m"])
        bn = int(manifest["block_n"])
        M = N * H * W
        gx, gy, gz = (
            (K + bn - 1) // bn,
            (M + bm - 1) // bm,
            int(manifest.get("grid_z", 1)),
        )
        if manifest.get("grid_order") == "MN":
            gx, gy = gy, gx
    grid = (gx, gy, gz)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * N * H * W * K * R * S * cpg
    bytes_xfer = 2.0 * (A.size + B.size + D.size)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(B))
        D_dev = rt.alloc(_nbytes(D))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B), _nbytes(B))
        rt.memset(D_dev, 0, _nbytes(D))
        if int(manifest.get("sig_has_bytes", 1)):
            args = struct.pack(
                "<QQQiii", A_dev, B_dev, D_dev, _nbytes(A), _nbytes(B), _nbytes(D)
            )
        else:
            args = struct.pack("<QQQ", A_dev, B_dev, D_dev)
        return args, (A_dev, B_dev, D_dev)

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, D.size
        rt.memcpy_d2h(_as_u8_buffer(D), ptrs[2], _nbytes(D))
        # Vectorized grouped conv reference, fp32 accumulation then fp16 output.
        Ap = np.pad(A, ((0, 0), (pH, pH), (pW, pW), (0, 0)), mode="constant")
        ref = np.zeros_like(D, dtype=np.float32)
        for r in range(R):
            for s in range(S):
                x = Ap[:, r : r + H * sH : sH, s : s + W * sW : sW, :]
                for g in range(groups):
                    xs = x[..., g * cpg : (g + 1) * cpg].astype(np.float32)
                    ws = B[g * kpg : (g + 1) * kpg, r, s, :].astype(np.float32)
                    ref[..., g * kpg : (g + 1) * kpg] += np.einsum(
                        "nhwc,kc->nhwk", xs, ws, optimize=True
                    )
        ref_h = ref.astype(np.float16)
        diff = np.abs(D.astype(np.float32) - ref_h.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > 1e-2)), D.size

    return make_args, grid, block, flop, bytes_xfer, check


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
        make_args, grid, block, flop, bytes_xfer, check = _gemm_problem(
            manifest, shape, verify
        )
    elif kind == "conv_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _conv_problem(
            manifest, shape, verify
        )
    else:
        raise ValueError(f"unsupported manifest kind {kind!r}")

    args, ptrs = make_args(rt)
    warmup = int(manifest.get("warmup_iters", 5))
    iters = int(manifest.get("timed_iters", 100))
    ms = _launch_timed(rt, fn, grid, block, args, warmup, iters)
    max_abs, bad, total = check(rt, ptrs)
    for ptr in ptrs:
        rt.free(ptr)
    module.unload()
    return RunSummary(
        ms=ms,
        tflops=flop / 1e9 / ms,
        gbps=bytes_xfer / 1e6 / ms,
        max_abs_diff=max_abs,
        bad_count=bad,
        total=total,
    )


def main(argv: Optional[list[str]] = None) -> int:
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
        if summary.bad_count:
            return 1
    print(
        f"Perf: {summary.ms:.6g} ms, {summary.tflops:.6g} TFlops, {summary.gbps:.6g} GB/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
