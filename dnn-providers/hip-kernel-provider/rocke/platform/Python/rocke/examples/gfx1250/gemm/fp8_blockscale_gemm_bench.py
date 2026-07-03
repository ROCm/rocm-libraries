# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-arch fp8 block-scaled GEMM throughput bench (TFLOPs).

Times the fp8 block-scaled expert GEMM -- the dominant compute in fused MoE --
on the running device and reports effective TFLOPs (= 2*M*N*K / time). Uses the
arch-appropriate validated kernel:

  * gfx950 (CDNA4 / MFMA): instances/common/block_scale_gemm (abquant fp8e4m3)
  * gfx1250 (gfx1250 / WMMA): instances/gfx1250/block_scaled_gemm (K=64 fp8 WMMA)

This is a compute-throughput micro-bench (random bytes, no correctness check;
correctness is covered by the *_verify examples). Use a large M to be
matrix-bound rather than launch/latency-bound.

  PYTHONPATH=Python HIP_VISIBLE_DEVICES=<idx> python3 -m \
      rocke.examples.gfx1250.gemm.fp8_blockscale_gemm_bench --m 4096 --n 2048 --k 2048
"""

from __future__ import annotations

import argparse
import struct
import time


def _build(arch, M, N, K, block_k):
    from rocke.helpers import compile_kernel

    if arch == "gfx1250":
        from rocke.instances.gfx1250.block_scaled_gemm import (
            BlockScaledGemmSpec,
            block_scaled_gemm_grid,
            build_block_scaled_gemm,
        )

        spec = BlockScaledGemmSpec(
            name="bench",
            M=M,
            N=N,
            K=K,
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            scale_dtype="fp32",
            block_k=block_k,
        )
        art = compile_kernel(build_block_scaled_gemm(spec, arch=arch), arch=arch)
        grid = block_scaled_gemm_grid(spec)
        return art, grid, spec.block_size, 2  # C bf16 = 2 bytes

    from rocke.instances.common.block_scale_gemm import (
        BlockScaleGemmSpec,
        block_scale_gemm_grid,
        build_block_scale_gemm,
    )

    spec = BlockScaleGemmSpec(
        M=M,
        N=N,
        K=K,
        quant_mode="abquant",
        mantissa_dtype="fp8e4m3",
        group_size_mnk=(1, 1, block_k),
        name="bench",
    )
    art = compile_kernel(build_block_scale_gemm(spec, arch=arch), arch=arch)
    grid = block_scale_gemm_grid(spec)
    return art, grid, spec.block_size, 4  # C f32 = 4 bytes


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    from rocke.runtime.hip_module import Runtime, get_device_arch

    arch = get_device_arch(0)
    M, N, K, BK = args.m, args.n, args.k, args.block_k
    groups = (K + BK - 1) // BK

    art, grid, block_size, c_bytes = _build(arch, M, N, K, BK)
    rt = Runtime()
    fn = rt.load_module(art.hsaco).get_function(art.kernel_name)

    def z(nbytes):
        d = rt.alloc(nbytes)
        rt.memset(d, 0, nbytes)
        return d

    dA, dB = z(M * K), z(N * K)  # fp8 bytes
    dAs, dBs = z(M * groups * 4), z(groups * N * 4)
    dC = z(M * N * c_bytes)
    packed = struct.pack("<QQQQQiii", dA, dB, dAs, dBs, dC, M, N, K)
    block = (block_size, 1, 1)

    for _ in range(args.warmup):
        rt.launch(fn, grid, block, packed)
    rt.sync()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        rt.launch(fn, grid, block, packed)
    rt.sync()
    dt = time.perf_counter() - t0

    ms = dt / args.iters * 1e3
    flop = 2.0 * M * N * K
    tflops = flop * args.iters / dt / 1e12
    for d in (dA, dB, dAs, dBs, dC):
        rt.free(d)
    print(
        f"[{arch}] fp8 block-scale GEMM M{M} N{N} K{K} bk{BK}: "
        f"{ms:.4f} ms/iter  {tflops:.2f} TFLOPs  (grid={grid} block={block_size})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
