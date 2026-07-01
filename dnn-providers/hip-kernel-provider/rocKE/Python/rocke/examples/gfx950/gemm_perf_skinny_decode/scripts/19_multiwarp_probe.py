#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 19: isolate whether multi-warp breakage is DTLA-specific or general.

Step 16 saw warp_n=2 produce garbage with DTLA on. If non-DTLA also
breaks, the bug is in MFMA/epilogue lane mapping. If only DTLA breaks,
the load-distribution is the problem.
"""

from __future__ import annotations
import ctypes
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5])
)  # .../rocke/.. = python root

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rocke.instances.common.gemm_universal import (  # noqa: E402
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)
from rocke.helpers import (
    compile_kernel,
    make_gemm_manifest,
    write_artifact,
)  # noqa: E402
from rocke.runtime.hip_module import Runtime  # noqa: E402

M, N, K = 2, 4096, 4096


def build(label, *, tile_m, tile_n, tile_k, warp_m, warp_n, direct_to_lds):
    tile = TileSpec(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = TraitSpec(
        pipeline="mem",
        scheduler="interwave",
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        direct_to_lds=direct_to_lds,
        dtl_cache_a=0,
        dtl_cache_b=0,
    )
    data = DataSpec(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32", layout="RCR"
    )
    spec = UniversalGemmSpec(
        name=f"{label}__m{M}n{N}k{K}", tile=tile, trait=trait, data=data
    )
    art = compile_kernel(build_universal_gemm(spec), isa="amdgcn-amd-amdhsa--gfx950")
    sub = ROOT / "build_multiwarp_probe" / label
    sub.mkdir(parents=True, exist_ok=True)
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=tile_m,
        block_n=tile_n,
        block_k=tile_k,
        threads_per_block=spec.block_size,
        default_shape=(M, N, K),
        warmup_iters=1,
        timed_iters=1,
        atoms=["mfma_f32_16x16x32_bf16"],
    )
    write_artifact(art, sub, manifest)
    return art, manifest


def check(art, manifest, label):
    rng = np.random.default_rng(0xC0FFEE)
    A_f32 = rng.integers(-3, 4, size=(M, K), dtype=np.int16).astype(np.float32)
    B_f32 = rng.integers(-3, 4, size=(N, K), dtype=np.int16).astype(np.float32)
    A_u16 = (
        torch.from_numpy(A_f32)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    B_u16 = (
        torch.from_numpy(B_f32)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    A_bf32 = (
        torch.from_numpy(A_u16.view(np.int16))
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )
    B_bf32 = (
        torch.from_numpy(B_u16.view(np.int16))
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )
    ref = (
        torch.from_numpy(A_bf32 @ B_bf32.T).to(torch.bfloat16).to(torch.float32).numpy()
    )

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    A_dev = rt.alloc(A_u16.nbytes)
    B_dev = rt.alloc(B_u16.nbytes)
    C_dev = rt.alloc(M * N * 2)
    rt.memcpy_h2d(
        A_dev, (ctypes.c_uint8 * A_u16.nbytes).from_buffer(A_u16), A_u16.nbytes
    )
    rt.memcpy_h2d(
        B_dev, (ctypes.c_uint8 * B_u16.nbytes).from_buffer(B_u16), B_u16.nbytes
    )
    rt.memset(C_dev, 0, M * N * 2)
    args = struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
    bm = int(manifest["block_m"])
    bn = int(manifest["block_n"])
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)
    rt.launch(fn, grid, block, args, stream=0)
    rt.stream_sync(0)
    out_buf = (ctypes.c_uint8 * (M * N * 2))()
    rt.memcpy_d2h(out_buf, C_dev, M * N * 2)
    C_out = (
        torch.from_numpy(np.frombuffer(bytes(out_buf), dtype=np.int16).copy())
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
        .reshape(M, N)
    )
    diff = C_out - ref
    max_abs = float(np.abs(diff).max())
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(C_dev)
    # Show per-column-region max errors so we can localise which warp's output is wrong
    print(f"  {label}: max_abs={max_abs}")
    if max_abs > 1.0:
        for c0 in range(0, min(N, 128), 16):
            seg = np.abs(diff[:, c0 : c0 + 16]).max()
            print(
                f"    cols[{c0:4d}..{c0 + 16:4d}] max_abs={seg:.1f}  ref_sample={ref[0, c0]:.1f}  got={C_out[0, c0]:.1f}"
            )
    return max_abs


def main():
    cases = [
        # (label, tile_m, tile_n, tile_k, warp_m, warp_n, dtla)
        ("baseline_dtl", 16, 16, 512, 1, 1, True),  # known-good
        ("baseline_nodtl", 16, 16, 512, 1, 1, False),  # ref: non-DTL path single warp
        ("mw_n2_nodtl", 16, 32, 512, 1, 2, False),  # non-DTL multi-warp
        ("mw_n2_dtl", 16, 32, 512, 1, 2, True),  # DTL multi-warp
        ("mw_n4_nodtl", 16, 64, 512, 1, 4, False),
    ]
    for label, tm, tn, tk, wm, wn, dtla in cases:
        try:
            art, mf = build(
                label,
                tile_m=tm,
                tile_n=tn,
                tile_k=tk,
                warp_m=wm,
                warp_n=wn,
                direct_to_lds=dtla,
            )
            check(art, mf, label)
        except Exception as e:
            print(f"  {label}: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
