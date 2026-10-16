# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch ONE dgrad request and run exactly what dispatch ships, from the AOT cache.

Replay driver for the gfx950 dgrad dispatcher. It asks
``dispatch_conv_grouped`` for the request, prints the selected candidate and
its knobs, finds the cache binary the selection names
(:func:`benchmarks.common.grouped_conv.dgrad_dispatch_cache.dgrad_cache_identity`)
in the implicit-GEMM or the direct-conv AOT cache, and launches it with the
dispatcher's own signature (``DispatchResult.signature``), grid, block and
kernarg values (``launch_values_for``). Nothing is compiled: a selection whose
binary is not cached is an error. For a direct pick the launch is also checked
against the cache's own launch plan (``direct_kernel_sweep.plan_for``).

* ``--verify``: torch (CPU, fp32) reference, rule
  ``|dX - ref| <= 1e-2 + 1e-2*|ref|`` on the dtype-rounded reference, with dX
  pre-filled with 0xFF bytes (NaN in fp16 and bf16) so an unwritten element
  fails (zero-filled for an implicit-GEMM pick that accumulates atomically).
* ``--time``: median over ``--reps`` of ``time_launches`` means.
* ``--loop N``: issue N back-to-back launches and exit (for
  ``rocprofv3 --kernel-trace``). torch is not imported in this mode.

Verify and time in separate processes.

Usage (from ``library/``, ``PYTHONPATH=$(pwd)/../platform/python:$(pwd)``)::

    python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \\
        --N 8 --C 96 --K 96 --H 12 --W 12 --G 24 --dtype bf16 \\
        --cache-dir <igemm cache> --direct-cache-dir <direct cache> --verify
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, default in (("N", 8), ("C", 96), ("K", 96), ("H", 12), ("W", 12)):
        ap.add_argument(f"--{name}", type=int, default=default)
    ap.add_argument("--G", type=int, default=1, help="groups")
    ap.add_argument("--Y", type=int, default=3, help="square filter size")
    ap.add_argument("--pad", type=int, default=None, help="default (Y-1)/2")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=("fp16", "bf16"))
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--cache-dir", default=None, help="implicit-GEMM AOT cache")
    ap.add_argument("--direct-cache-dir", default=None, help="direct-conv AOT cache")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--loop", type=int, default=0)
    return ap.parse_args(argv)


def _rand_bits(np, shape, dtype, seed):
    """``(fp32 values, 16-bit storage)`` of uniform(-1, 1) data in ``dtype``."""
    f = np.random.default_rng(seed).uniform(-1, 1, size=shape).astype(np.float32)
    if dtype == "bf16":
        u = f.view(np.uint32)
        bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32), bits
    bits = f.astype(np.float16).view(np.uint16)
    return bits.view(np.float16).astype(np.float32), bits


def _to_float(np, bits, dtype):
    if dtype == "bf16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view(np.float16).astype(np.float32)


def _round(np, ref, dtype):
    if dtype == "bf16":
        u = ref.astype(np.float32).view(np.uint32)
        u = u + np.uint32(0x7FFF) + ((u >> 16) & 1).astype(np.uint32)
        return (u & np.uint32(0xFFFF0000)).view(np.float32)
    return ref.astype(np.float16).astype(np.float32)


def _reference(np, dy, w, a, pad):
    """dX (NHWC, fp32) of ``dy`` (NHWK) and ``w`` (KYXC) on CPU."""
    import torch

    dx = torch.nn.grad.conv2d_input(
        (a.N, a.C, a.H, a.W),
        torch.from_numpy(w).permute(0, 3, 1, 2).contiguous(),
        torch.from_numpy(dy).permute(0, 3, 1, 2).contiguous(),
        stride=a.stride,
        padding=pad,
        groups=a.G,
    )
    return dx.permute(0, 2, 3, 1).contiguous().numpy()


def main(argv=None) -> int:
    a = _parse_args(argv)
    if not a.verify:
        sys.modules["torch"] = None  # keep torch's HIP runtime out of the process
    import numpy as np
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.grouped_conv.dgrad_dispatch_cache import (
        dgrad_cache_identity,
        dgrad_cache_problem,
    )
    from benchmarks.common.kernel_cache import KernelCache
    from dispatch.grouped_convolution import (
        ConvDirectDgradSpec,
        ConvGroupedRequest,
        _problem,
        dispatch_conv_grouped,
        launch_values_for,
    )
    from kernels.common.conv_implicit_gemm_dgrad import pack_sub_gemm_buffer
    from rocke.runtime import synchronize_and_release, time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    pad = (a.Y - 1) // 2 if a.pad is None else a.pad
    req = ConvGroupedRequest(
        N=a.N,
        C=a.C,
        K=a.K,
        Hi=a.H,
        Wi=a.W,
        Y=a.Y,
        X=a.Y,
        G=a.G,
        pad_h=pad,
        pad_w=pad,
        stride_h=a.stride,
        stride_w=a.stride,
        dtype=a.dtype,
        arch=a.arch,
        direction="dgrad",
    )
    r = dispatch_conv_grouped(req)
    print("\n".join(r.explanation))
    direct = isinstance(r.spec, ConvDirectDgradSpec)
    cache_dir = a.direct_cache_dir if direct else a.cache_dir
    if cache_dir is None:
        print(
            f"pass --{'direct-' if direct else ''}cache-dir for a "
            f"{'direct' if direct else 'implicit-GEMM'} pick"
        )
        return 2
    cache = KernelCache(cache_dir, a.arch)
    ident = dgrad_cache_identity(req, r.spec)
    print(f"cache identity: {ident.label()}")
    entry = cache.get(ident)
    if entry is None:
        print("NOT CACHED: the selection names no binary of this cache")
        return 2
    problem = dgrad_cache_problem(req, r.spec)
    ok, why = cache.supports_problem(ident, problem)
    if not ok:
        print(f"NOT SUPPORTED: {why}")
        return 2
    hsaco, meta = entry
    rc = 0
    if direct:
        plan, why = dks.plan_for(
            cache, ident, cache.hsaco_path(ident), meta, problem, a.arch
        )
        if plan is None:
            print(f"NO PLAN: {why}")
            return 2
        (step,) = plan.steps
        for what, mine, theirs in (
            ("signature", list(r.signature), step.signature),
            ("grid", tuple(r.grid), tuple(step.grid)),
            ("block", tuple(r.block), tuple(step.block)),
        ):
            if mine != theirs:
                print(f"MISMATCH {what}: dispatch {mine} vs plan {theirs}")
                rc = 1

    p = _problem(req)
    dy32, dyb = _rand_bits(np, (a.N, p.Ho, p.Wo, a.K), a.dtype, 1)
    w32, wb = _rand_bits(np, (a.K, a.Y, a.Y, a.C // a.G), a.dtype, 2)
    dx_bytes = a.N * a.H * a.W * a.C * 2
    rt = Runtime()
    ptr = {
        "dY": rt.alloc(dyb.nbytes),
        "W": rt.alloc(wb.nbytes),
        "dX": rt.alloc(dx_bytes),
    }
    for role, arr in (("dY", dyb), ("W", wb)):
        rt.memcpy_h2d(
            ptr[role],
            (ctypes.c_uint8 * arr.nbytes).from_address(arr.ctypes.data),
            arr.nbytes,
        )
    extras = {}
    zero_dx = False
    if not direct:
        inst = r.spec.to_dgrad_spec(p)
        zero_dx = inst.needs_atomic
        sgs = inst.compute_sub_gemms()
        flat = np.array(
            pack_sub_gemm_buffer(sgs, inst.tile_m, inst.tile_n), dtype=np.int32
        )
        ptr["sub_gemm"] = rt.alloc(flat.nbytes)
        rt.memcpy_h2d(ptr["sub_gemm"], flat.ctypes.data, flat.nbytes)
        extras = {"sub_gemm_buf": ptr["sub_gemm"], "num_sub_gemms": len(sgs)}
    values = launch_values_for(
        req,
        r.spec,
        A_ptr=ptr["dY"],
        B_ptr=ptr["W"],
        D_ptr=ptr["dX"],
        A_bytes=dyb.nbytes,
        B_bytes=wb.nbytes,
        D_bytes=dx_bytes,
        **extras,
    )
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=meta["kernel_name"], signature=list(r.signature)
    )
    cfg = LaunchConfig(grid=tuple(r.grid), block=tuple(r.block))
    print(f"launch {meta['kernel_name']} grid={r.grid} block={r.block}")

    def run():
        if zero_dx:
            rt.memset(ptr["dX"], 0, dx_bytes)
        launcher(values, config=cfg)

    if a.verify:
        rt.memset(ptr["dX"], 0 if zero_dx else 0xFF, dx_bytes)
        launcher(values, config=cfg)
        synchronize_and_release(0)
        out = np.empty((a.N, a.H, a.W, a.C), dtype=np.uint16)
        rt.memcpy_d2h(
            (ctypes.c_uint8 * out.nbytes).from_address(out.ctypes.data),
            ptr["dX"],
            out.nbytes,
        )
        got = _to_float(np, out, a.dtype)
        ref = _round(np, _reference(np, dy32, w32, a, pad), a.dtype)
        tol = 1e-2
        bad = int((~(np.abs(got - ref) <= tol + tol * np.abs(ref))).sum())
        nan = int(np.isnan(got).sum())
        print(f"VERIFY {'PASS' if bad == 0 else 'FAIL'} bad={bad} nan={nan}/{got.size}")
        rc = rc or (0 if bad == 0 else 1)
    if a.time:
        ms = [
            time_launches(run, warmup=a.warmup, iters=a.iters, stream=0)
            for _ in range(a.reps)
        ]
        print(
            f"TIME median_ms={statistics.median(ms):.5f} min_ms={min(ms):.5f} "
            f"(reps={a.reps}, iters={a.iters})"
        )
    if a.loop:
        with no_fence():
            for _ in range(a.loop):
                run()
            rt.sync()
        print(f"LOOP done {a.loop}")
    synchronize_and_release(0)
    for d in ptr.values():
        rt.free(d)
    return rc


if __name__ == "__main__":
    sys.exit(main())
