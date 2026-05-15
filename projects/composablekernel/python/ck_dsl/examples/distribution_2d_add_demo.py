# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Demo: 2D distribution-driven tile add.

This kernel demonstrates the **2D X / 2-contributor P** pattern of
:class:`ck_dsl.helpers.TileDistributionEncoding`. It computes
``C[h, w] = A[h, w] + B[h, w]`` for ``(H, W)`` row-major matrices,
processing one ``(TM, TN)`` tile per CTA.

The distribution decomposes the 2D tile as:

* ``Hs[0] = (TM,)``                — one level: each lane owns one row
* ``Hs[1] = (TN_GROUPS, VEC)``     — two levels: TN_GROUPS lane sub-groups,
                                     each handling VEC contiguous columns
* ``P0`` is the *linearised* lane index; it contributes to **two**
  H buckets via a 2-element ``Ps2RHs[0]`` sub-sequence:
  ``(major=1, minor=0)`` for the row dim,
  ``(major=2, minor=0)`` for the column-group dim.
* ``Y0`` is the per-thread vector along the column dim.

The caller passes ``ps=[[lane_row, lane_col_group]]`` -- one
sub-sequence per P dim, each entry of which is one of P0's H
contributors in encoding order.

Compare to ``ck_dsl.instances.elementwise``: that kernel uses the
simpler 1D ``sweep_row_chunks`` form; this demo expresses the same
semantics through the full CK Tile distribution surface, which scales
naturally to non-trivial tile decompositions (multi-warp / multi-MFMA
distributions) where the simpler sweep helper is insufficient.

Run with::

    sudo -n /workspace/dsl_bake_off/venv/bin/python3 \\
        python/ck_dsl/examples/distribution_2d_add_demo.py [--H ...] [--W ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from ck_dsl.core.ir import F16, I32, IRBuilder, PtrType  # noqa: E402
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from ck_dsl.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from ck_dsl.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
)
from ck_dsl.helpers import (  # noqa: E402
    TileDistributionEncoding,
    make_global_view,
    make_load_store_traits,
    make_static_distributed_tensor,
    make_static_tile_distribution,
    make_tile_window,
)


def build_2d_add(
    *,
    tile_m: int = 32,
    tile_n: int = 64,
    block_size: int = 256,
    vec: int = 8,
):
    if tile_m * (tile_n // vec) != block_size:
        raise ValueError(
            f"tile geometry mismatch: tile_m({tile_m}) * (tile_n/vec)"
            f"({tile_n // vec}) = {tile_m * (tile_n // vec)} != block_size({block_size})"
        )
    tn_groups = tile_n // vec  # lane sub-groups per row

    # Encoding: 2D X, 2D-contributor P, 1D Y (the per-thread vector).
    encoding = TileDistributionEncoding(
        Hs=((tile_m,), (tn_groups, vec)),
        Ps2RHs_major=((1, 2),),
        Ps2RHs_minor=((0, 0),),
        Ys2RHs_major=(2,),
        Ys2RHs_minor=(1,),
    )
    distribution = make_static_tile_distribution(encoding)
    traits = make_load_store_traits(distribution)
    assert traits.scalar_per_vector == vec
    assert distribution.X_lengths == (tile_m, tile_n)
    assert distribution.Y_lengths == (vec,)

    b = IRBuilder(f"dist2d_add_TM{tile_m}_TN{tile_n}_bs{block_size}_v{vec}")
    b.kernel.attrs["max_workgroup_size"] = block_size

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    # H is part of the launch signature but the kernel body doesn't
    # consume it directly -- the row count is implicit in the launch
    # grid and the per-tile origin. We declare it so the launcher's
    # arg-packer sees the canonical (A, B, C, H, W) ABI.
    _ = b.param("H", I32)  # noqa: F841 - ABI param
    W = b.param("W", I32)

    tid = b.thread_id_x()
    c_tn_groups = b.const_i32(tn_groups)
    lane_row = b.div(tid, c_tn_groups)  # 0..tile_m-1
    lane_col_group = b.mod(tid, c_tn_groups)  # 0..tn_groups-1

    bx = b.block_id_x()
    by = b.block_id_y()
    h0 = b.mul(by, b.const_i32(tile_m))
    w0 = b.mul(bx, b.const_i32(tile_n))

    # Three global views with a runtime row stride W; tile origins are
    # the per-block (h0, w0) corner.
    a_view = make_global_view(A, shape=(tile_m, tile_n), dtype=F16, strides=(W, 1))
    b_view = make_global_view(Bp, shape=(tile_m, tile_n), dtype=F16, strides=(W, 1))
    c_view = make_global_view(C, shape=(tile_m, tile_n), dtype=F16, strides=(W, 1))
    a_tile = make_tile_window(a_view, lengths=(tile_m, tile_n), origin=(h0, w0))
    b_tile = make_tile_window(b_view, lengths=(tile_m, tile_n), origin=(h0, w0))
    c_tile = make_tile_window(c_view, lengths=(tile_m, tile_n), origin=(h0, w0))

    # Distribution-driven load: each thread receives a 1D Y space of
    # length `vec` (one vector load per access).
    a_dt = a_tile.load(
        b, distribution=distribution, ps=[[lane_row, lane_col_group]], traits=traits
    )
    b_dt = b_tile.load(
        b, distribution=distribution, ps=[[lane_row, lane_col_group]], traits=traits
    )

    # Sweep Y -> add and accumulate into a fresh distributed tensor.
    # We compute in f32 (the loaders promoted on the way in) then
    # demote on store.
    c_dt = make_static_distributed_tensor(distribution, dtype=F16)
    for y in distribution.iterate_ys():
        c_dt.set(y, b.fadd(a_dt.get(y), b_dt.get(y)))

    c_tile.store(b, c_dt, ps=[[lane_row, lane_col_group]], traits=traits)

    return b.kernel, tile_m, tile_n, block_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--vec", type=int, default=8)
    args = parser.parse_args()
    if args.H % args.tile_m or args.W % args.tile_n:
        print(
            f"H ({args.H}) and W ({args.W}) must be divisible by tile_m ({args.tile_m}) "
            f"and tile_n ({args.tile_n})",
            file=sys.stderr,
        )
        return 1
    if not torch.cuda.is_available():
        print("HIP device unavailable; exiting", file=sys.stderr)
        return 1

    block_size = args.tile_m * (args.tile_n // args.vec)
    kernel, _, _, _ = build_2d_add(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        block_size=block_size,
        vec=args.vec,
    )
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(kernel))
    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kernel.name,
        signature=[
            {"name": "A", "type": "ptr<f16, global>"},
            {"name": "B", "type": "ptr<f16, global>"},
            {"name": "C", "type": "ptr<f16, global>"},
            {"name": "H", "type": "i32"},
            {"name": "W", "type": "i32"},
        ],
    )

    torch.manual_seed(0)
    A = torch.randn(args.H, args.W, dtype=torch.float16, device="cuda")
    B = torch.randn(args.H, args.W, dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    grid = (args.W // args.tile_n, args.H // args.tile_m, 1)
    launcher(
        {"A": A, "B": B, "C": C, "H": args.H, "W": args.W},
        config=LaunchConfig(grid=grid, block=(block_size, 1, 1)),
    )
    synchronize_and_release()

    ref = (A.float() + B.float()).half()
    diff = (C.float() - ref.float()).abs().max().item()
    print(
        f"2D distribution-driven add  "
        f"H={args.H} W={args.W} tile=({args.tile_m},{args.tile_n}) vec={args.vec}  "
        f"max_abs={diff:.3e}"
    )
    return 0 if diff == 0.0 else 2


if __name__ == "__main__":
    sys.exit(main())
