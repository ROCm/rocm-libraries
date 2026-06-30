# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1151 true-int8 WMMA GEMM with f16 dequant output.

The "Path A" rung: int8×int8 -> int32 via the hardware ``wmma_i32_16x16x16_iu8``
atom (upstream's native-int atom), then a per-tensor symmetric dequant in the
epilogue -> f16 (``instances/gfx1151/wmma_gemm_iu8_dequant.py``). This is the
true-int8-compute counterpart of step 03 (int8 storage / f16 compute) and the
f16-output sibling of upstream's i32-out ``wmma_gemm_iu8.py``.

A/B are random asymmetric int8 stored as raw bytes; the kernel reads them as
i32-packed (4 int8 per i32 — the iu8 fragment ABI), so the device buffer is the
*same bytes* step 03 uploads. A transposed lane map or a wrong dequant index
fails verify, so this also re-confirms the iu8 ABI through the f16 epilogue.

Reference: ``C = (A.f32 * scale_a) @ (B.f32 * scale_b).T -> f16``. The i32
accumulation is exact; only the final f16 store rounds, so parity is judged with
``np.allclose`` (rtol/atol = ``--tol``). Must run on a gfx1151 device.

    python scripts/04_iu8_dequant_verify.py --m 16 --n 16 --k 16
    python scripts/04_iu8_dequant_verify.py --m 128 --n 128 --k 128
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
from rocke.instances.gfx1151.wmma_gemm_iu8_dequant import (  # noqa: E402
    WmmaGemmIu8DequantSpec,
    build_wmma_gemm_iu8_dequant,
    wmma_gemm_iu8_dequant_grid,
)
from rocke.runtime.hip_module import Runtime  # noqa: E402


def _write_data(name: str, payload: dict) -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / f"{name}.json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=128)
    p.add_argument(
        "--scale-a", type=float, default=0.05, help="per-tensor A dequant scale"
    )
    p.add_argument(
        "--scale-b", type=float, default=0.05, help="per-tensor B dequant scale"
    )
    p.add_argument("--tol", type=float, default=2e-2, help="np.allclose rtol and atol")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    import numpy as np

    for d, name in ((args.m, "M"), (args.n, "N"), (args.k, "K")):
        if d % 16:
            raise SystemExit(
                f"{name}={d} must be a multiple of 16 (WMMA 16x16x16 tile)"
            )

    spec = WmmaGemmIu8DequantSpec(name=f"wmma_gemm_iu8_dequant_{args.arch}")
    art = compile_kernel(
        build_wmma_gemm_iu8_dequant(spec, arch=args.arch), arch=args.arch
    )
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
        f"total={art.timings.get('total', 0):.1f}ms"
    )

    if args.no_verify:
        print(f"[{args.arch}] build OK (verify skipped)")
        return 0

    M, N, K = args.m, args.n, args.k
    sa, sb = float(args.scale_a), float(args.scale_b)
    rng = np.random.default_rng(0xC0FFEE)

    # Random asymmetric int8 (RCR: A is M×K, B is N×K). Stored as raw bytes; the
    # kernel reads them i32-packed (4 int8/i32). Small magnitudes keep the
    # dequantized result f16-friendly so the test reflects kernel/ABI correctness.
    A = rng.integers(-8, 9, size=(M, K), dtype=np.int8)
    B = rng.integers(-8, 9, size=(N, K), dtype=np.int8)
    C = np.zeros((M, N), dtype=np.float16)

    grid = wmma_gemm_iu8_dequant_grid(M, N)
    block = (spec.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    ad = rt.alloc(A.nbytes)
    bd = rt.alloc(B.nbytes)
    cd = rt.alloc(C.nbytes)
    rt.memcpy_h2d(ad, u8(A), A.nbytes)
    rt.memcpy_h2d(bd, u8(B), B.nbytes)
    rt.memset(cd, 0, C.nbytes)

    packed = struct.pack("<QQQiiiff", ad, bd, cd, M, N, K, sa, sb)
    rt.launch(fn, grid, block, packed)
    rt.sync()
    rt.memcpy_d2h(u8(C), cd, C.nbytes)

    ref = ((A.astype(np.float32) * sa) @ (B.astype(np.float32) * sb).T).astype(
        np.float16
    )

    diff = np.abs(C.astype(np.float32) - ref.astype(np.float32))
    max_abs = float(diff.max())
    bad = int(
        np.count_nonzero(diff > (args.tol + args.tol * np.abs(ref.astype(np.float32))))
    )
    for ptr in (ad, bd, cd):
        rt.free(ptr)
    module.unload()

    ok = bool(
        np.allclose(
            C.astype(np.float32), ref.astype(np.float32), rtol=args.tol, atol=args.tol
        )
    )
    tag = "PASS (true-int8 dequant OK)" if ok else "FAIL"
    print(
        f"[{args.arch}] WMMA iu8->f16 GEMM {M}x{N}x{K} (scale_a={sa}, scale_b={sb}): "
        f"max_abs_diff={max_abs:.3e} bad={bad}/{C.size} tol={args.tol:.0e} -> {tag}"
    )
    _write_data(
        "04_iu8_dequant_verify",
        {
            "kernel": "wmma_gemm_iu8_dequant (Path A: true int8 i32-acc, f16 dequant out)",
            "atom": "wmma_i32_16x16x16_iu8",
            "shape": {"M": M, "N": N, "K": K},
            "scale_a": sa,
            "scale_b": sb,
            "tol": args.tol,
            "max_abs_diff": max_abs,
            "bad": bad,
            "size": int(C.size),
            "pass": ok,
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
