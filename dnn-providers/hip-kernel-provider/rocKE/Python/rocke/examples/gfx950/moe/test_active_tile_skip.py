# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone parity + perf test for ``trait.active_tile_skip=True``.

Two cases are checked, each with the same A/B/C tensors and a
``sorted_token_ids`` array shaped ``(batch * slot_size,)``:

1. **All tiles active** — every entry of ``sorted_token_ids`` is 0.
   The active-tile-skip kernel must produce bitwise the same output
   as the canonical kernel.
2. **Half tiles inactive** — the first half of each batch's bucket
   has ``sorted_token_ids == -1`` for full tiles. The active-tile-skip
   kernel must leave the corresponding rows of ``C`` UNCHANGED while
   producing the canonical result for the other half. Zero-init C
   before launch and compare ``C[..., active_rows]`` against the
   canonical output.

Run::

    cd <repo>/dnn-providers/hip-kernel-provider/rocKE/Python
    python rocke/examples/moe/test_active_tile_skip.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if "rocke" not in sys.modules:
    HERE = Path(__file__).resolve()
    pkg_root = HERE.parents[4]
    sys.path.insert(0, str(pkg_root))

import torch  # noqa: E402

from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from rocke.instances.common.batched_gemm import (  # noqa: E402
    BatchedGemmSpec,
    batched_gemm_grid,
    batched_gemm_signature,
    build_batched_gemm,
)
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec  # noqa: E402
from rocke.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from rocke.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


def build_launcher(spec: BatchedGemmSpec) -> KernelLauncher:
    k = build_batched_gemm(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    return KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=batched_gemm_signature(spec)
    )


def run(
    Bsz: int = 8,
    M: int = 32,
    N: int = 4096,
    K: int = 4096,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 64,
    iters: int = 80,
    warmup: int = 20,
) -> int:
    if not torch.cuda.is_available():
        print("skipping: no CUDA")
        return 0
    torch.manual_seed(0)
    # warp_m / warp_n implied by tile_m / tile_n with 32x32 atom.
    warp_m = max(1, tile_m // 32)
    warp_n = max(1, tile_n // 32)
    tile = TileSpec(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )
    spec_base = BatchedGemmSpec(
        name="att_base",
        tile=tile,
        trait=TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="default",
            pad_m=True,
            pad_n=True,
        ),
    )
    spec_att = BatchedGemmSpec(
        name="att_skip",
        tile=tile,
        trait=TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            active_tile_skip=True,
        ),
    )
    print(f"[shape] B={Bsz} M={M} N={N} K={K}, tile=({tile_m},{tile_n},{tile_k})")
    A = torch.randn(Bsz, M, K, dtype=torch.float16, device="cuda")
    Bm = torch.randn(Bsz, N, K, dtype=torch.float16, device="cuda")

    grid = batched_gemm_grid(Bsz, M, N, spec_base)
    cfg = LaunchConfig(grid=grid, block=(spec_base.block_size, 1, 1))

    launcher_base = build_launcher(spec_base)
    launcher_att = build_launcher(spec_att)

    def base_args(C):
        return {
            "A": A,
            "B": Bm,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
            "stride_a": M * K,
            "stride_b": N * K,
            "stride_c": M * N,
        }

    # slot_size must equal M for our purposes (tile_m * 1 m-tile-per-expert).
    slot_size = M

    def sti_full(v):
        return torch.full((Bsz * slot_size,), int(v), dtype=torch.int32, device="cuda")

    # ---- Case 1: all tiles active ----
    C_base = torch.zeros(Bsz, M, N, dtype=torch.float16, device="cuda")
    launcher_base(base_args(C_base), config=cfg)
    synchronize_and_release()

    sti_all_active = sti_full(0)  # token_id 0 (>= 0) for every row -> all active
    C_att = torch.zeros(Bsz, M, N, dtype=torch.float16, device="cuda")
    args_att = dict(
        base_args(C_att), SortedTokenIds=sti_all_active, slot_size=slot_size
    )
    launcher_att(args_att, config=cfg)
    synchronize_and_release()
    delta_all = float((C_base - C_att).abs().max().cpu())
    print(f"[parity] all-active   max|base - att|={delta_all:.4f}  (need 0.0)")

    # ---- Case 2: every other expert tile inactive ----
    sti_half = sti_full(0)
    # Mark half the experts (even ones) as fully inactive.
    inactive_experts = list(range(0, Bsz, 2))
    for e in inactive_experts:
        sti_half[e * slot_size : (e + 1) * slot_size] = -1
    expected = C_base.clone()
    for e in inactive_experts:
        expected[e].zero_()
    C_att2 = torch.zeros(Bsz, M, N, dtype=torch.float16, device="cuda")
    args_att2 = dict(base_args(C_att2), SortedTokenIds=sti_half, slot_size=slot_size)
    launcher_att(args_att2, config=cfg)
    synchronize_and_release()
    delta_half = float((expected - C_att2).abs().max().cpu())
    inactive_actually_zero = float(C_att2[inactive_experts].abs().max().cpu())
    print(f"[parity] half-skipped max|expected - att|={delta_half:.4f}  (need 0.0)")
    print(
        f"[parity] half-skipped inactive-rows max|C|={inactive_actually_zero:.4f}  (need 0.0)"
    )

    # ---- Perf: all-active vs all-inactive ----
    base_us = (
        time_launches(
            lambda: launcher_base(base_args(C_base), config=cfg),
            warmup=warmup,
            iters=iters,
        )
        * 1e-3
    )
    att_active_us = (
        time_launches(
            lambda: launcher_att(args_att, config=cfg),
            warmup=warmup,
            iters=iters,
        )
        * 1e-3
    )
    sti_all_inactive = sti_full(-1)
    args_all_inact = dict(
        base_args(C_att2), SortedTokenIds=sti_all_inactive, slot_size=slot_size
    )
    att_inactive_us = (
        time_launches(
            lambda: launcher_att(args_all_inact, config=cfg),
            warmup=warmup,
            iters=iters,
        )
        * 1e-3
    )

    print(f"[time] base (no skip)      = {base_us * 1e6:8.2f} us")
    print(f"[time] att, all-active     = {att_active_us * 1e6:8.2f} us")
    print(f"[time] att, all-inactive   = {att_inactive_us * 1e6:8.2f} us")
    if delta_all > 1e-3:
        print("[FAIL] all-active parity broken")
        return 1
    if delta_half > 1e-3 or inactive_actually_zero > 1e-3:
        print("[FAIL] half-skip parity broken")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--tile_m", type=int, default=32)
    ap.add_argument("--tile_n", type=int, default=128)
    ap.add_argument("--tile_k", type=int, default=64)
    args = ap.parse_args()
    return run(args.B, args.M, args.N, args.K, args.tile_m, args.tile_n, args.tile_k)


if __name__ == "__main__":
    sys.exit(main())
