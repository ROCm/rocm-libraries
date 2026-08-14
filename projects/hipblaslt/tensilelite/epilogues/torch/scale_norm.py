# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import os
import torch
import triton
import triton.language as tl
from torch.profiler import profile, record_function, ProfilerActivity

# Tile widths used as Triton constexpr parameters; kept module-level so
# they are easy to tune, but not changed at runtime.
_BLOCK_N: int = 4096
_BLOCK_ND: int = 64


@triton.jit
def scaleNormKernel(
    cPtr, dPtr, outPtr,
    m, n, nD,
    invD, eps,
    cStrideM, cStrideN,
    dStrideM, dStrideN,
    outStrideM, outStrideN,
    blockN: tl.constexpr,
    blockNd: tl.constexpr,
):
    row = tl.program_id(0)

    # Load D in fixed-size blocks so arbitrarily large nD needs no constexpr-sized allocation.
    rowSum = tl.zeros((), dtype=tl.float32)
    for start in range(0, nD, blockNd):
        offs = start + tl.arange(0, blockNd)
        mask = offs < nD
        vals = tl.load(dPtr + row * dStrideM + offs * dStrideN, mask=mask, other=0.0)
        rowSum += tl.sum(vals, axis=0)

    scale = 1.0 / tl.sqrt(invD * rowSum + eps)

    # Block over n so arbitrary column counts work without requiring a power-of-two width.
    for start in range(0, n, blockN):
        offs = start + tl.arange(0, blockN)
        mask = offs < n
        c = tl.load(cPtr + row * cStrideM + offs * cStrideN, mask=mask, other=0.0)
        result = c.to(tl.float32) * scale
        tl.store(outPtr + row * outStrideM + offs * outStrideN, result.to(tl.bfloat16), mask=mask)


def scaleNorm(c: torch.Tensor, d: torch.Tensor, invD: float, eps: float) -> torch.Tensor:
    m, n = c.shape
    nD = d.shape[1]
    out = torch.empty_like(c)
    grid = (m,)
    scaleNormKernel[grid](
        c, d, out,
        m, n, nD,
        invD, eps,
        c.stride(0), c.stride(1),
        d.stride(0), d.stride(1),
        out.stride(0), out.stride(1),
        blockN=_BLOCK_N,
        blockNd=_BLOCK_ND,
    )
    return out


def checkCorrectness(c: torch.Tensor, d: torch.Tensor, invD: float, eps: float, out: torch.Tensor):
    rowSums = d.sum(dim=1, keepdim=True)
    scale = torch.rsqrt(invD * rowSums + eps)
    ref = (c.float() * scale).to(torch.bfloat16)
    try:
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        raise


def runBenchmark(
    c: torch.Tensor, d: torch.Tensor, invD: float, eps: float,
    warmup: int, steps: int, m: int, n: int, nD: int,
):
    for _ in range(warmup):
        scaleNorm(c, d, invD, eps)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(steps):
            with record_function("scale_norm"):
                scaleNorm(c, d, invD, eps)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    if not (torch.cuda.is_available() or getattr(torch, "hip", None) and torch.hip.is_available()):
        raise SystemExit("error: no CUDA/ROCm GPU available")

    parser = argparse.ArgumentParser(description="scale_norm benchmark")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("nd", type=int)
    parser.add_argument("--inv-d", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations before profiling")
    parser.add_argument("--steps", type=int, default=10, help="profiled iterations")
    parser.add_argument("--dump-ir", metavar="DIR", help="dump Triton IR stages (ttir, ttgir, llir, amdgcn, hsaco) to DIR")
    args = parser.parse_args()

    if args.dump_ir:
        os.makedirs(args.dump_ir, exist_ok=True)
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        os.environ["TRITON_DUMP_DIR"] = args.dump_ir

    m, n, nD = args.m, args.n, args.nd
    invD = args.inv_d
    eps = args.eps

    c = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    d = torch.rand(m, nD, dtype=torch.float32, device="cuda")

    out = scaleNorm(c, d, invD, eps)
    checkCorrectness(c, d, invD, eps, out)
    runBenchmark(c, d, invD, eps, args.warmup, args.steps, m, n, nD)


if __name__ == "__main__":
    main()
