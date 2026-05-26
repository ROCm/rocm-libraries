#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Quick head-to-head: gemm_decode P0 vs AITER skinny GEMM kernels at the
M=1 N=8192 K=7168 BF16 smoke shape (and a small sweep around it).

For each kernel we report:
  - kernel time (us, median over `repeat` iterations after `warmup`)
  - achieved HBM bandwidth (GB/s)
  - achieved compute throughput (TFLOPS)

The gemm_decode number is the median wallclock of the
`benchmark_gemm_decode_universal_bf16_smallm_default` executable; AITER
numbers are timed directly from Python with hipEvent-backed PyTorch timers.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# Force AITER import from the local checkout.
AITER_ROOT = Path("/home/AMD/samremes/dev/aiter")
sys.path.insert(0, str(AITER_ROOT))

import torch
import aiter
from aiter.jit.utils.chip_info import get_cu_num


CK_BUILD = Path("/home/AMD/samremes/dev/rocm-libraries/projects/composablekernel/build")
GEMM_DECODE_BF16_EXE       = CK_BUILD / "bin" / "benchmark_gemm_decode_universal_bf16_smallm_default"
GEMM_DECODE_FP8_EXE        = CK_BUILD / "bin" / "benchmark_gemm_decode_universal_fp8_smallm_pertensor"
GEMM_DECODE_BLOCKSCALE_EXE = CK_BUILD / "bin" / "benchmark_gemm_decode_blockscale_fp8_smallm_dsv3"


SHAPES = [
    # (M, N, K)
    (1, 8192, 7168),   # design-doc smoke shape
    (1, 4096, 7168),
    (1, 12288, 4096),
    (4, 8192, 7168),
    (1, 7168, 7168),   # DSV3 a8w8_blockscale smoke shape
]


def _bytes_bw(M: int, N: int, K: int, bytes_per_elem: int = 2) -> float:
    """Decode-shape memory traffic in bytes (A + B + C)."""
    return float(M * K + N * K + M * N) * bytes_per_elem


def _flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def time_callable(fn, warmup: int = 100, repeat: int = 200) -> float:
    """Return median per-iteration time in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.median(times_ms) / 1e3


def bench_aiter_skinny(name: str, M: int, N: int, K: int, repeat: int = 100):
    """name in {'wvSpltK', 'LLMM1', 'wv_splitk_small_fp16_bf16'}."""
    device = "cuda"
    dtype = torch.bfloat16
    # AITER convention: skinny_gemm(inp, weights) → wvSpltK(weights, inp, out, inp.shape[0], cu_count).
    # inp: [M, K], weights: [N, K], out: [M, N].
    inp = torch.randn(M, K, dtype=dtype, device=device) * 0.1
    weights = torch.randn(N, K, dtype=dtype, device=device) * 0.1
    out = torch.empty(M, N, dtype=dtype, device=device)
    cu = get_cu_num()

    if name == "wvSpltK":
        fn = lambda: aiter.wvSpltK(weights, inp, out, M, cu)
    elif name == "LLMM1":
        # LLMM1 takes rows_per_block (4 in tuned_gemm.py); only valid for M=1.
        if M != 1:
            return None
        fn = lambda: aiter.LLMM1(weights, inp, out, 4)
    elif name == "wv_splitk_small":
        fn = lambda: aiter.wv_splitk_small_fp16_bf16(weights, inp, out, M, cu)
    else:
        raise ValueError(name)

    # Sanity-check the kernel actually ran (catches misconfigured shapes).
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as ex:
        return {"error": str(ex)}

    sec = time_callable(fn, warmup=20, repeat=repeat)
    us = sec * 1e6
    bw = _bytes_bw(M, N, K) / sec / 1e9
    tflops = _flops(M, N, K) / sec / 1e12
    return {"us": us, "gbs": bw, "tflops": tflops}


def bench_gemm_decode(M: int, N: int, K: int, exe: Path = GEMM_DECODE_BF16_EXE,
                      repeat: int = 200, k_batch: int = 1, trials: int = 3):
    """Take median over `trials` separate process launches to wash out cold-start jitter."""
    if not exe.exists():
        return {"error": f"missing {exe}"}
    cmd = [
        str(exe),
        f"-m={M}",
        f"-n={N}",
        f"-k={K}",
        f"-split_k={k_batch}",
        "-warmup=100",
        f"-repeat={repeat}",
        "-metric=2",
    ]
    samples_us, samples_tflops, samples_gbs = [], [], []
    for _ in range(trials):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as ex:
            return {"error": ex.stderr.strip() or ex.stdout.strip() or str(ex)}
        line = res.stdout.strip().splitlines()[0]
        m = re.search(r"\|\s*([\d.]+)\s*ms,\s*([\d.]+)\s*TFLOP/s,\s*([\d.]+)\s*GB/s", line)
        if not m:
            return {"error": f"unparsable: {line}"}
        ms, tflops, gbs = map(float, m.groups())
        samples_us.append(ms * 1e3)
        samples_tflops.append(tflops)
        samples_gbs.append(gbs)
    return {"us": statistics.median(samples_us),
            "gbs": statistics.median(samples_gbs),
            "tflops": statistics.median(samples_tflops)}


def bench_aiter_blockscale(M: int, N: int, K: int, repeat: int = 100):
    """AITER FP8 a8w8 blockscale GEMM (cktile path).

    Uses Block2D<1, 128> X / Block2D<128, 128> W, the DSV3 convention. The
    tuned CSV inside AITER chooses the kernel; we do not override.
    """
    device = "cuda"
    fp8_dtype = torch.float8_e4m3fn
    if M % 1 != 0 or N % 128 != 0 or K % 128 != 0:
        return None

    inp = torch.randn(M, K, dtype=torch.float32, device=device).clamp_(-1, 1).to(fp8_dtype)
    weights = torch.randn(N, K, dtype=torch.float32, device=device).clamp_(-1, 1).to(fp8_dtype)
    x_scale = torch.full((M, K // 128), 0.1, dtype=torch.float32, device=device)
    w_scale = torch.full((N // 128, K // 128), 0.1, dtype=torch.float32, device=device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    fn = lambda: aiter.gemm_a8w8_blockscale(inp, weights, x_scale, w_scale,
                                            dtype=torch.bfloat16, out=out)
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as ex:
        return {"error": str(ex)}

    sec = time_callable(fn, warmup=20, repeat=repeat)
    us = sec * 1e6
    bytes_total = float(M * K + N * K) * 1.0 + float(M * N) * 2.0
    bw = bytes_total / sec / 1e9
    tflops = _flops(M, N, K) / sec / 1e12
    return {"us": us, "gbs": bw, "tflops": tflops}


def bench_aiter_wvSplitKQ(M: int, N: int, K: int, repeat: int = 100):
    """AITER PerTensor FP8 -> BF16 GEMM (wvSplitKQ).

    Memory traffic accounting uses 1 byte per FP8 input and 2 bytes per BF16
    output, plus 4 bytes for each scale scalar (negligible).
    """
    device = "cuda"
    fp8_dtype = torch.float8_e4m3fnuz
    try:
        # gfx950 uses the OCP variant.
        fp8_dtype = torch.float8_e4m3fn
    except AttributeError:
        pass

    inp = torch.randn(M, K, dtype=torch.float32, device=device).clamp_(-1, 1).to(fp8_dtype)
    weights = torch.randn(N, K, dtype=torch.float32, device=device).clamp_(-1, 1).to(fp8_dtype)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    scale_a = torch.tensor([0.125], dtype=torch.float32, device=device)
    scale_b = torch.tensor([0.0625], dtype=torch.float32, device=device)
    cu = get_cu_num()

    fn = lambda: aiter.wvSplitKQ(weights, inp, out, scale_a, scale_b, cu)
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as ex:
        return {"error": str(ex)}

    sec = time_callable(fn, warmup=20, repeat=repeat)
    us = sec * 1e6
    bytes_total = float(M * K + N * K) * 1.0 + float(M * N) * 2.0
    bw = bytes_total / sec / 1e9
    tflops = _flops(M, N, K) / sec / 1e12
    return {"us": us, "gbs": bw, "tflops": tflops}


def main() -> int:
    print(f"Device: {torch.cuda.get_device_name(0)}")
    cu = get_cu_num()
    print(f"CU count: {cu}\n")

    header = f"{'shape (M,N,K)':<22} {'kernel':<28} {'us':>8} {'TFLOP/s':>10} {'GB/s':>10}"
    print(header)
    print("-" * len(header))

    for (M, N, K) in SHAPES:
        shape_str = f"({M},{N},{K})"

        for k_batch in (1, 2):
            r = bench_gemm_decode(M, N, K, exe=GEMM_DECODE_BF16_EXE, k_batch=k_batch)
            tag = f"gemm_decode BF16 (kb={k_batch})"
            if "error" in r:
                print(f"{shape_str:<22} {tag:<28} {'-':>8} {'-':>10} {'-':>10}  ({r['error']})")
            else:
                print(f"{shape_str:<22} {tag:<28} {r['us']:8.2f} {r['tflops']:10.2f} {r['gbs']:10.1f}")

        for kname in ("wvSpltK", "LLMM1", "wv_splitk_small"):
            r = bench_aiter_skinny(kname, M, N, K)
            if r is None:
                continue
            tag = f"aiter::{kname}"
            if "error" in r:
                print(f"{shape_str:<22} {tag:<28} {'-':>8} {'-':>10} {'-':>10}  ({r['error']})")
            else:
                print(f"{shape_str:<22} {tag:<28} {r['us']:8.2f} {r['tflops']:10.2f} {r['gbs']:10.1f}")
        print()

        # FP8 PerTensor head-to-head (P0b vs AITER wvSplitKQ).
        for k_batch in (1, 2):
            r = bench_gemm_decode(M, N, K, exe=GEMM_DECODE_FP8_EXE, k_batch=k_batch)
            tag = f"gemm_decode FP8 PT (kb={k_batch})"
            if "error" in r:
                print(f"{shape_str:<22} {tag:<28} {'-':>8} {'-':>10} {'-':>10}  ({r['error']})")
            else:
                print(f"{shape_str:<22} {tag:<28} {r['us']:8.2f} {r['tflops']:10.2f} {r['gbs']:10.1f}")

        r = bench_aiter_wvSplitKQ(M, N, K)
        tag = "aiter::wvSplitKQ (FP8 PT)"
        if "error" in r:
            print(f"{shape_str:<22} {tag:<28} {'-':>8} {'-':>10} {'-':>10}  ({r['error']})")
        else:
            print(f"{shape_str:<22} {tag:<28} {r['us']:8.2f} {r['tflops']:10.2f} {r['gbs']:10.1f}")
        print()

        # FP8 blockscale head-to-head (P1 vs AITER gemm_a8w8_blockscale).
        # Skip shapes that don't divide the DSV3 block sizes (X = 1x128,
        # W = 128x128) - the bench harness reports these as missing.
        if N % 128 == 0 and K % 128 == 0:
            for k_batch in (1, 2):
                r = bench_gemm_decode(M, N, K, exe=GEMM_DECODE_BLOCKSCALE_EXE, k_batch=k_batch)
                tag = f"gemm_decode BS (kb={k_batch})"
                if "error" in r:
                    print(f"{shape_str:<22} {tag:<28} {'-':>8} {'-':>10} {'-':>10}  ({r['error']})")
                else:
                    print(f"{shape_str:<22} {tag:<28} {r['us']:8.2f} {r['tflops']:10.2f} {r['gbs']:10.1f}")

            r = bench_aiter_blockscale(M, N, K)
            if r is not None:
                tag = "aiter::a8w8_blockscale"
                if "error" in r:
                    print(f"{shape_str:<22} {tag:<28} {'-':>8} {'-':>10} {'-':>10}  ({r['error']})")
                else:
                    print(f"{shape_str:<22} {tag:<28} {r['us']:8.2f} {r['tflops']:10.2f} {r['gbs']:10.1f}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
