#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end split-K via the GEMM dispatcher for decode shapes.

The split-K body in ``instances/common/gemm_universal.py`` is the proven win on
skinny / tall-N decode GEMMs (~1.6x rocBLAS at M=2, N=4096, K=4096 bf16). This
script wires that body into the *dispatcher* so the decode matmuls -- qkv_proj /
o_proj / gate_up / down -- pick it up automatically:

  1. build a ``GemmRequest`` per decode shape;
  2. ``dispatch_gemm_bf16`` selects a candidate and the split-K heuristic
     (``rocke.helpers.split_k``) folds a valid split degree into the spec when
     the base grid leaves the device idle (square/large shapes stay split_k=1);
  3. honor the kernel contract: when split_k > 1 allocate a zero-init f32
     workspace ``[M, N]``, launch the dispatcher's ``(N_tiles, M_tiles, split_k)``
     grid, then cast the workspace back to the output dtype.

The heuristic is opt-in / default-safe and overridable via
``ROCKE_GEMM_SPLIT_K`` (``auto`` | ``off`` | ``<n>``). The dry-run path (no GPU)
prints the dispatch decision + grid for every shape and verifies the square
shape stays split_k=1; pass ``--run`` for a quick on-GPU numeric sanity check.

Run (heuristic dry-run, no GPU):
  PYTHONPATH=Python python \
    Python/rocke/examples/gfx950/gemm_perf_skinny_decode/scripts/24_splitk_dispatch.py

Run (GPU numeric sanity, defer the perf sweep to 23_splitk_universal.py):
  ROCKE_LLVM_FLAVOR=llvm22 sudo -n -E env HIP_VISIBLE_DEVICES=0 \
    PYTHONPATH=Python <venv>/bin/python \
    Python/rocke/examples/gfx950/gemm_perf_skinny_decode/scripts/24_splitk_dispatch.py --run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from rocke.dispatch.gemm import dispatch_gemm_bf16  # noqa: E402
from rocke.dispatch.gemm.bf16_rcr import (
    build_kernel as build_kernel_bf16,
)  # noqa: E402
from rocke.dispatch.gemm.common import GemmRequest  # noqa: E402

ARCH = "gfx950"

# (name, M, N, K) -- decode (M tiny) matmuls that leave the device idle, plus a
# square shape that must stay split_k=1.
SHAPES = [
    ("qkv_proj", 2, 2560, 2048),
    ("o_proj", 2, 2048, 2048),
    ("gate_up", 2, 4096, 4096),
    ("down", 2, 4096, 4096),
    ("square_4096", 4096, 4096, 4096),
]


def dispatch_one(name: str, M: int, N: int, K: int):
    req = GemmRequest(M=M, N=N, K=K, arch=ARCH, dtype="bf16", layout="RCR")
    result = dispatch_gemm_bf16(req)
    spec = result.spec
    return req, result, spec


def report_dispatch():
    # The square-shape-stays-1 invariant only holds under the auto heuristic; a
    # forced env degree (ROCKE_GEMM_SPLIT_K=<n>) intentionally overrides it.
    auto = os.environ.get("ROCKE_GEMM_SPLIT_K", "auto").strip().lower() in ("", "auto")
    print(
        f"{'shape':>12} {'M':>5} {'N':>6} {'K':>6} "
        f"{'tile':>12} {'split_k':>7} {'grid':>16}"
    )
    square_ok = True
    for name, M, N, K in SHAPES:
        req, result, spec = dispatch_one(name, M, N, K)
        t = spec.tile
        sk = spec.trait.split_k
        tile = f"{t.tile_m}x{t.tile_n}x{t.tile_k}"
        print(
            f"{name:>12} {M:>5} {N:>6} {K:>6} {tile:>12} {sk:>7} {str(result.grid):>16}"
        )
        if auto and name.startswith("square") and sk != 1:
            square_ok = False
            print(f"  ERROR: square shape {name} got split_k={sk} (expected 1)")
    print()
    if auto:
        print("square-shape-stays-1:", "PASS" if square_ok else "FAIL")
    else:
        print("square-shape-stays-1: N/A (env override active)")
    return square_ok


def run_numeric():
    """One quick on-GPU numeric sanity check per shape (no perf sweep)."""
    import struct

    import torch

    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    assert torch.cuda.is_available(), "ROCm-torch required for --run"
    dev = "cuda"
    torch.manual_seed(0)
    rt = Runtime()

    def ptr(t):
        return int(t.data_ptr())

    for name, M, N, K in SHAPES:
        req, result, spec = dispatch_one(name, M, N, K)
        sk = spec.trait.split_k
        kernel = build_kernel_bf16(result)
        art = compile_kernel(kernel, isa=f"amdgcn-amd-amdhsa--{ARCH}")
        mod = rt.load_module(art.hsaco)
        fn = mod.get_function(art.kernel_name)

        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=dev)

        # Workspace contract: split_k > 1 -> zero-init f32 [M, N], cast back.
        if sk > 1:
            cf32 = torch.zeros(M, N, dtype=torch.float32, device=dev)
            out_buf = cf32
        else:
            out_buf = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

        args = struct.pack("<QQQiii", ptr(A), ptr(W), ptr(out_buf), M, N, K)
        rt.launch(fn, result.grid, result.block, args, stream=0)
        rt.stream_sync(0)
        out = out_buf.to(torch.bfloat16) if sk > 1 else out_buf

        ref = torch.matmul(A.float(), W.float().t()).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs()
        rel = float(diff.max()) / float(ref.float().abs().max() + 1e-9)
        status = "PASS" if rel < 5e-2 else "FAIL"
        print(f"[numeric] {name:>12} split_k={sk} rel={rel:.4e} -> {status}")


def main():
    run = "--run" in sys.argv[1:]
    ok = report_dispatch()
    if run:
        print()
        run_numeric()
    else:
        print(
            "\n(dry run; pass --run for an on-GPU numeric sanity check. "
            "Perf sweep lives in 23_splitk_universal.py.)"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
