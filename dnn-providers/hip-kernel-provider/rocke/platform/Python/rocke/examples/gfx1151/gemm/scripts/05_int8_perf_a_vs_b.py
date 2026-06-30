# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark Path A (true int8 -> f16) vs Path B (int8 storage / f16 compute) GEMM.

Both kernels are RCR int8 GEMMs on gfx1151 that take the *same* int8 A/B bytes and
output f16, with the identical ABI (``A,B | C f16 | M,N,K i32 | scale_a,scale_b f32``)
and grid (one wave per 16×16 tile), so they share host buffers and differ only in
the compute:

  * Path B (``wmma_gemm_int8``): int8 -> f16 dequant in the K-loop, then the f16
    ``wmma_f32_16x16x16_f16`` (f32 accumulate). Compute == f16. A/B read as i8.
  * Path A (``wmma_gemm_iu8_dequant``): true ``wmma_i32_16x16x16_iu8`` (int8 in,
    int32 accumulate), dequant to f16 in the epilogue. A/B read as i32-packed
    (same bytes). ~2x the tensor-core throughput ceiling.

Methodology (benchmark differentiated from correctness):
  * **Correctness gate** runs once up front (128³, both kernels); per-shape numpy
    verification is opt-in (``--verify``).
  * **Hardened timing**: ``_robust_time_us`` adaptively ramps the timed-loop
    iteration count until the measured interval clears ``--min-ms`` (so fast
    kernels never fall below GPU event-timer resolution), then reports the median
    of ``--reps`` measurements with the min..max spread. Kernels that cannot clear
    the floor even at ``--max-iters`` (launch-overhead bound) are flagged
    ``[!] launch-bound`` rather than reported as a confident rate. The shared
    ``runtime.launcher.time_launches`` is used unchanged.
  * **Perf-shape suite** is roofline-tagged by regime; reports arithmetic
    intensity (ops/byte), op rate, and % of the gfx1151 ceilings (~59 TF f16,
    ~118 TOPS int8).

Op count is ``2*M*N*K`` for both. Must run on a gfx1151 device.

    python scripts/05_int8_perf_a_vs_b.py
    python scripts/05_int8_perf_a_vs_b.py --m 2048 --n 2048 --k 2048 --reps 9 --min-ms 12
"""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # examples/gfx1151/gemm
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # python root

from rocke.helpers import compile_kernel  # noqa: E402
from rocke.instances.gfx1151.wmma_gemm_int8 import (  # noqa: E402
    WmmaGemmInt8Spec,
    build_wmma_gemm_int8,
    wmma_gemm_int8_grid,
)
from rocke.instances.gfx1151.wmma_gemm_iu8_dequant import (  # noqa: E402
    WmmaGemmIu8DequantSpec,
    build_wmma_gemm_iu8_dequant,
)
from rocke.runtime.hip_module import Runtime  # noqa: E402
from rocke.runtime.launcher import time_launches  # noqa: E402

# gfx1151 (Radeon 8060S) tensor-core ceilings: int8 WMMA is ~2x the f16 rate.
_PEAK_F16_TF = 59.0
_PEAK_I8_TOPS = 118.0

# Perf-shape suite (M, N, K, regime tag). AI (ops/byte) is computed per row.
_PERF_SHAPES = [
    (1024, 1024, 1024, "balanced"),
    (2048, 2048, 2048, "balanced-large"),
    (512, 512, 8192, "K-heavy (overhead-amortized)"),
    (256, 256, 16384, "K-heavy-narrow"),
    (4096, 4096, 512, "wide-MN (load/epilogue-bound)"),
    (8192, 512, 512, "tall-skinny (LLM-ish proj)"),
]


def _write_data(name: str, payload: dict) -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / f"{name}.json").write_text(json.dumps(payload, indent=2))


def _robust_time_us(launch, *, min_ms: float, max_iters: int, reps: int):
    """Median per-iter latency (us) with adaptively-scaled iters + repeats.

    Ramps the timed-loop iteration count until a measured block actually clears
    ``min_ms`` (a one-shot probe under-shoots for sub-microsecond kernels), then
    takes the median of ``reps`` measurements. Returns ``(median_us, min_us,
    max_us, iters, reliable)``; ``reliable`` is False when even ``max_iters``
    cannot raise the total measured time to the floor (launch-overhead bound).
    ``median_us`` is ``None`` only if the timer never returned a positive interval.
    """
    iters = 20
    ms = time_launches(launch, warmup=10, iters=iters)
    guard = 0
    while iters < max_iters and guard < 24 and (ms <= 0.0 or ms * iters < min_ms):
        if ms <= 0.0:
            iters = min(max_iters, iters * 8)
        else:
            want = (min_ms / (ms * iters)) * 1.3
            iters = min(max_iters, max(iters * 2, int(iters * want)))
        ms = time_launches(launch, warmup=max(5, iters // 10), iters=iters)
        guard += 1

    samples_us = []
    for _ in range(reps):
        m = time_launches(launch, warmup=max(5, iters // 10), iters=iters)
        if m > 0.0:
            samples_us.append(m * 1e3)
    if not samples_us:
        return None, None, None, iters, False
    samples_us.sort()
    med = samples_us[len(samples_us) // 2]
    reliable = (med * 1e-3 * iters) >= (min_ms * 0.5)
    return med, samples_us[0], samples_us[-1], iters, reliable


def _u8(np, a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))


def _correctness_gate(rt, fn_b, fn_a, *, sa, sb, tol):
    """One-shot correctness check of both kernels on a small shape. Returns ok."""
    import numpy as np

    M = N = K = 128
    rng = np.random.default_rng(0xC0FFEE)
    A = rng.integers(-8, 9, size=(M, K), dtype=np.int8)
    B = rng.integers(-8, 9, size=(N, K), dtype=np.int8)
    C = np.zeros((M, N), dtype=np.float16)
    ref = (
        ((A.astype(np.float32) * sa) @ (B.astype(np.float32) * sb).T)
        .astype(np.float16)
        .astype(np.float32)
    )

    ad, bd, cd = rt.alloc(A.nbytes), rt.alloc(B.nbytes), rt.alloc(C.nbytes)
    rt.memcpy_h2d(ad, _u8(np, A), A.nbytes)
    rt.memcpy_h2d(bd, _u8(np, B), B.nbytes)
    grid = wmma_gemm_int8_grid(M, N)
    block = (32, 1, 1)
    packed = struct.pack("<QQQiiiff", ad, bd, cd, M, N, K, sa, sb)

    ok = True
    for fn in (fn_b, fn_a):
        rt.memset(cd, 0, C.nbytes)
        rt.launch(fn, grid, block, packed)
        rt.sync()
        rt.memcpy_d2h(_u8(np, C), cd, C.nbytes)
        ok &= bool(np.allclose(C.astype(np.float32), ref, rtol=tol, atol=tol))
    for ptr in (ad, bd, cd):
        rt.free(ptr)
    return ok


def _bench_shape(rt, fn_b, fn_a, M, N, K, tag, *, sa, sb, tol, verify, timing):
    import numpy as np

    rng = np.random.default_rng(0xC0FFEE)
    A = rng.integers(-8, 9, size=(M, K), dtype=np.int8)
    B = rng.integers(-8, 9, size=(N, K), dtype=np.int8)
    C = np.zeros((M, N), dtype=np.float16)

    ad, bd, cd = rt.alloc(A.nbytes), rt.alloc(B.nbytes), rt.alloc(C.nbytes)
    rt.memcpy_h2d(ad, _u8(np, A), A.nbytes)
    rt.memcpy_h2d(bd, _u8(np, B), B.nbytes)
    grid = wmma_gemm_int8_grid(M, N)
    block = (32, 1, 1)
    packed = struct.pack("<QQQiiiff", ad, bd, cd, M, N, K, sa, sb)

    note = ""
    if verify:
        ref = (
            ((A.astype(np.float32) * sa) @ (B.astype(np.float32) * sb).T)
            .astype(np.float16)
            .astype(np.float32)
        )
        oks = []
        for fn in (fn_b, fn_a):
            rt.memset(cd, 0, C.nbytes)
            rt.launch(fn, grid, block, packed)
            rt.sync()
            rt.memcpy_d2h(_u8(np, C), cd, C.nbytes)
            oks.append(bool(np.allclose(C.astype(np.float32), ref, rtol=tol, atol=tol)))
        note = f" verify[B={'OK' if oks[0] else 'FAIL'} A={'OK' if oks[1] else 'FAIL'}]"

    med_b, lo_b, hi_b, it_b, rel_b = _robust_time_us(
        lambda: rt.launch(fn_b, grid, block, packed), **timing
    )
    med_a, lo_a, hi_a, it_a, rel_a = _robust_time_us(
        lambda: rt.launch(fn_a, grid, block, packed), **timing
    )
    for ptr in (ad, bd, cd):
        rt.free(ptr)

    ops = 2.0 * M * N * K
    ai = ops / (M * K + N * K + 2.0 * M * N)  # ops per byte (int8 in, f16 out)
    if not med_a or not med_b:
        print(f"  {M}x{N}x{K} [{tag}] AI={ai:.0f}: below timer resolution{note}")
        return {
            "M": M,
            "N": N,
            "K": K,
            "tag": tag,
            "ai": ai,
            "status": "sub_resolution",
        }

    tops_b = ops / (med_b * 1e-6) / 1e12
    tops_a = ops / (med_a * 1e-6) / 1e12
    spr_b = (hi_b - lo_b) / med_b * 100.0
    spr_a = (hi_a - lo_a) / med_a * 100.0
    reliable = bool(rel_a and rel_b)
    warn = "" if reliable else "  [!] launch-bound: rate unreliable"
    print(
        f"  {M}x{N}x{K} [{tag}] AI={ai:.0f} op/B:\n"
        f"      B(f16cmp) {med_b:8.1f}us (±{spr_b:3.0f}%, {it_b}it) "
        f"{tops_b:6.2f} TOP/s {tops_b / _PEAK_F16_TF * 100:4.1f}%pk | "
        f"A(int8)  {med_a:8.1f}us (±{spr_a:3.0f}%, {it_a}it) "
        f"{tops_a:6.2f} TOP/s {tops_a / _PEAK_I8_TOPS * 100:4.1f}%pk | "
        f"A/B {med_b / med_a:.2f}x{note}{warn}"
    )
    return {
        "M": M,
        "N": N,
        "K": K,
        "tag": tag,
        "ai": round(ai, 1),
        "path_b": {
            "us": round(med_b, 2),
            "tops": round(tops_b, 2),
            "spread_pct": round(spr_b, 1),
        },
        "path_a": {
            "us": round(med_a, 2),
            "tops": round(tops_a, 2),
            "spread_pct": round(spr_a, 1),
        },
        "a_over_b": round(med_b / med_a, 3),
        "reliable": reliable,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--m", type=int, default=0, help="single shape M (0 -> perf suite)")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--scale-a", type=float, default=0.05)
    p.add_argument("--scale-b", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument(
        "--min-ms", type=float, default=8.0, help="auto-scale iters to this floor"
    )
    p.add_argument(
        "--reps", type=int, default=7, help="measurements per kernel; median reported"
    )
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument(
        "--verify", action="store_true", help="numpy-verify each perf shape too"
    )
    args = p.parse_args()

    for d, name in ((args.m, "M"), (args.n, "N"), (args.k, "K")):
        if d and d % 16:
            raise SystemExit(f"{name}={d} must be a multiple of 16")

    if args.m and args.n and args.k:
        shapes = [(args.m, args.n, args.k, "custom")]
    else:
        shapes = _PERF_SHAPES

    rt = Runtime()
    art_b = compile_kernel(
        build_wmma_gemm_int8(WmmaGemmInt8Spec(name=f"b_{args.arch}"), arch=args.arch),
        arch=args.arch,
    )
    art_a = compile_kernel(
        build_wmma_gemm_iu8_dequant(
            WmmaGemmIu8DequantSpec(name=f"a_{args.arch}"), arch=args.arch
        ),
        arch=args.arch,
    )
    mod_b = rt.load_module(art_b.hsaco)
    mod_a = rt.load_module(art_a.hsaco)
    fn_b = mod_b.get_function(art_b.kernel_name)
    fn_a = mod_a.get_function(art_a.kernel_name)

    print(
        f"[{args.arch}] int8 GEMM Path A (true int8 -> f16) vs Path B (int8 -> f16 compute) | "
        f"min_ms={args.min_ms} reps={args.reps} | op count=2*M*N*K"
    )
    gate = _correctness_gate(
        rt, fn_b, fn_a, sa=args.scale_a, sb=args.scale_b, tol=args.tol
    )
    print(f"  correctness gate (128^3, both kernels): {'PASS' if gate else 'FAIL'}")
    if not gate:
        return 1

    timing = {"min_ms": args.min_ms, "max_iters": args.max_iters, "reps": args.reps}
    rows = []
    for M, N, K, tag in shapes:
        row = _bench_shape(
            rt,
            fn_b,
            fn_a,
            M,
            N,
            K,
            tag,
            sa=args.scale_a,
            sb=args.scale_b,
            tol=args.tol,
            verify=args.verify,
            timing=timing,
        )
        if row:
            rows.append(row)

    _write_data(
        "05_int8_perf_a_vs_b",
        {
            "arch": args.arch,
            "path_a": "wmma_gemm_iu8_dequant (true int8 i32-acc -> f16)",
            "path_b": "wmma_gemm_int8 (int8 storage / f16 compute)",
            "peak_f16_tf": _PEAK_F16_TF,
            "peak_i8_tops": _PEAK_I8_TOPS,
            "min_ms": args.min_ms,
            "reps": args.reps,
            "correctness_gate_pass": gate,
            "results": rows,
        },
    )
    mod_a.unload()
    mod_b.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
