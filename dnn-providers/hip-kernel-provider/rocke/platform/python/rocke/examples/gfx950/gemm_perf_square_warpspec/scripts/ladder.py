# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reproduce the README performance ladder for square fp16 GEMM on gfx950.

Rungs A–F, each building on the previous, measured in a single interleaved
session so all ratios are comparable (same clock state).  Run from the repo
root with rocke on PYTHONPATH:

    HIP_VISIBLE_DEVICES=0 python scripts/ladder.py

Expected output (absolute TF/s varies with clock; ratios are stable):

  A  plain compv4                     ~590 TF   ~0.31x rocBLAS
  B  + direct-to-LDS                  ~650 TF   ~0.34x
  C  + depth-2 prefetch (1 barrier)  ~1040 TF   ~0.54x
  D  + 4-way swizzle                 ~1420 TF   ~0.74x
  E  + element swizzle (0 conflict)  ~1450 TF   ~0.75x
  F  + cache_b=ALL                   ~1550 TF   ~0.81x
     rocBLAS fp16 reference           ~1920 TF    1.00x
"""

import os
import time
import ctypes
import struct
import statistics as st
import numpy as np
import torch
from rocke.helpers import compile_kernel
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)
from rocke.runtime.hip_module import Runtime

S = 8192
TILE = dict(
    tile_m=256,
    tile_n=256,
    tile_k=64,
    warp_m=4,
    warp_n=4,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)
DATA = DataSpec(
    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32", layout="RCR"
)
ARCH = "gfx950"
FLOP = 2 * S**3
ITERS = 50
CYCLES = 10  # interleaved cycles per rung


def _u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def build_kernel(rt, Ad, Bd, Cd, **trait_kw):
    """Compile one kernel variant and return a zero-arg launch callable."""
    os.environ.pop("CK_SWZ_R", None)
    os.environ.pop("CK_SWZ_W", None)
    os.environ.pop("CK_SWZ_L", None)
    swz = trait_kw.pop("swz_rwl", None)
    if swz:
        os.environ["CK_SWZ_R"], os.environ["CK_SWZ_W"], os.environ["CK_SWZ_L"] = (
            str(swz[0]),
            str(swz[1]),
            str(swz[2]),
        )
    tile = TileSpec(**TILE)
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        pad_m=True,
        pad_n=True,
        pad_k=True,
        **trait_kw,
    )
    spec = UniversalGemmSpec(name="g", tile=tile, trait=trait, data=DATA, wave_size=64)
    ok, msg = is_valid_spec(spec, ARCH)
    assert ok, msg
    art = compile_kernel(build_universal_gemm(spec, arch=ARCH), arch=ARCH)
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    args = struct.pack("<QQQiii", Ad, Bd, Cd, S, S, S)
    grid = ((S + 255) // 256, (S + 255) // 256, 1)
    blk = (spec.block_size, 1, 1)
    return lambda: rt.launch(fn, grid, blk, args)


def check(launch, rt, Ad, Bd, Cd, ref):
    """Return relative max error (should be ~0 for fp16 with small ints)."""
    rt.memset(Cd, 0, S * S * 2)
    launch()
    rt.sync()
    out = np.empty(S * S, dtype=np.int16)
    rt.memcpy_d2h(_u8(out), Cd, S * S * 2)
    C = torch.tensor(out, device="cuda").view(torch.float16).view(S, S).float()
    return (C - ref).abs().max().item() / max(1.0, ref.abs().max().item())


def burst(fn, it=ITERS):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return FLOP / ((time.time() - t) / it) / 1e12


def main():
    rt = Runtime()
    rng = np.random.default_rng(0)
    A = torch.tensor(
        rng.integers(-2, 3, (S, S)).astype(np.float32),
        dtype=torch.float16,
        device="cuda",
    )
    Bm = torch.tensor(
        rng.integers(-2, 3, (S, S)).astype(np.float32),
        dtype=torch.float16,
        device="cuda",
    )
    ref = A.float() @ Bm.float().t()

    Ad = rt.alloc(S * S * 2)
    Bd = rt.alloc(S * S * 2)
    Cd = rt.alloc(S * S * 2)
    rt.memcpy_h2d(
        Ad, _u8(np.ascontiguousarray(A.view(torch.int16).cpu().numpy())), S * S * 2
    )
    rt.memcpy_h2d(
        Bd, _u8(np.ascontiguousarray(Bm.view(torch.int16).cpu().numpy())), S * S * 2
    )

    def rb():
        return torch.matmul(A, Bm.t())

    rungs = [
        (
            "A  plain compv4",
            dict(direct_to_lds=False, dtl_prefetch=False, lds_swizzle=False),
        ),
        (
            "B  + direct-to-LDS",
            dict(direct_to_lds=True, dtl_prefetch=False, lds_swizzle=False),
        ),
        (
            "C  + depth-2 prefetch",
            dict(direct_to_lds=True, dtl_prefetch=True, lds_swizzle=False),
        ),
        (
            "D  + 4-way swizzle",
            dict(
                direct_to_lds=True,
                dtl_prefetch=True,
                lds_swizzle=True,
                swz_rwl=(3, 1, 4),
            ),
        ),
        (
            "E  + element swizzle (0 conflict)",
            dict(direct_to_lds=True, dtl_prefetch=True, lds_swizzle=True),
        ),
        (
            "F  + cache_b=ALL",
            dict(
                direct_to_lds=True,
                dtl_prefetch=True,
                lds_swizzle=True,
                dtl_cache_a=0,
                dtl_cache_b=0,
            ),
        ),
    ]

    # Build all kernels upfront.
    print("Building kernels...")
    launches = []
    for label, kw in rungs:
        fn = build_kernel(rt, Ad, Bd, Cd, **kw)
        err = check(fn, rt, Ad, Bd, Cd, ref)
        ok = "ok" if err < 1e-2 else f"WRONG({err:.1e})"
        print(f"  {label:<30}  correctness: {ok}")
        launches.append(fn)
    os.environ.pop("CK_SWZ_R", None)
    os.environ.pop("CK_SWZ_W", None)
    os.environ.pop("CK_SWZ_L", None)

    # Warm up everything.
    print("\nWarming up...")
    for fn in launches + [rb]:
        for _ in range(60):
            fn()
    torch.cuda.synchronize()

    # Interleaved measurement.
    all_fns = launches + [rb]
    all_names = [label for label, _ in rungs] + ["   rocBLAS fp16 (ref)"]
    acc = {n: [] for n in all_names}
    for _ in range(CYCLES):
        for name, fn in zip(all_names, all_fns):
            acc[name].append(burst(fn))

    rbm = st.median(acc["   rocBLAS fp16 (ref)"])
    print(f"\n{'Rung':<30}  {'TF/s':>7}  {'vs rocBLAS':>10}")
    print("-" * 54)
    for name in all_names:
        v = acc[name]
        tf = st.median(v)
        print(f"  {name:<30}  {tf:>7.0f}  {tf / rbm:>9.3f}x")


if __name__ == "__main__":
    main()
