# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Demo: row-wise sum reduce driven entirely by a CK Tile-style distribution.

This kernel is functionally identical to the production
:mod:`ck_dsl.instances.reduce` builder, but is expressed using the
full CK Tile abstraction stack:

* :class:`ck_dsl.helpers.TileDistributionEncoding` declares the
  ``(Rs, Hs, Ps2RHs, Ys2RHs)`` mapping that decomposes a 1D row of
  ``N`` elements as ``CHUNKS * BLOCK_SIZE * VEC``.
* :func:`make_static_tile_distribution` wraps it with the runtime
  emission logic.
* :func:`make_load_store_traits` picks the vector dim and width
  (``vec_dim_y = 1``, ``scalar_per_vector = VEC`` for the natural
  encoding here).
* :func:`load_tile` issues the per-thread vector loads through the
  :class:`TileWindow` and packs the results into a
  :class:`StaticDistributedTensor`.
* The kernel sweeps the Y space to fold the partial sum, runs a
  block LDS-tree reduction via :func:`block_lds_reduce`, and writes
  one f16 per row via :func:`store_scalar_from_f32`.

This file is intentionally an **example** rather than a production
instance, because the simpler :mod:`ck_dsl.instances.reduce` builder
covers the same workload with less verbose code. The point here is
to show that the full distribution path is wired end-to-end and
matches a torch reference bit-exactly for the canonical row-reduce
encoding.

Run with::

    sudo -n /workspace/dsl_bake_off/venv/bin/python3 \\
        python/ck_dsl/examples/distribution_reduce_demo.py [--M ...] [--N ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from ck_dsl.core.ir import F16, F32, I32, IRBuilder, PtrType  # noqa: E402
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from ck_dsl.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from ck_dsl.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
)
from ck_dsl.helpers import (  # noqa: E402
    TileDistributionEncoding,
    block_lds_reduce,
    load_tile,
    make_lds_view,
    make_load_store_traits,
    make_naive_tensor_view_packed,
    make_static_tile_distribution,
    make_tile_window,
    store_scalar_from_f32,
)


def build_distribution_reduce(*, N: int, block_size: int = 256, vec: int = 8):
    """Build the row-sum kernel using the full distribution path."""
    if N % (block_size * vec):
        raise ValueError(
            f"N={N} must be divisible by block_size*vec={block_size * vec}"
        )
    chunks = N // (block_size * vec)

    # CK Tile encoding: 1D X = chunks * block_size * vec.
    # P0 -> X0 level 1 (block_size lanes);  Y0 -> level 0 (chunks);
    # Y1 -> level 2 (vec elements per thread).
    encoding = TileDistributionEncoding(
        Hs=((chunks, block_size, vec),),
        Ps2RHs_major=((1,),),
        Ps2RHs_minor=((1,),),
        Ys2RHs_major=(1, 1),
        Ys2RHs_minor=(0, 2),
    )
    distribution = make_static_tile_distribution(encoding)
    traits = make_load_store_traits(distribution)
    assert traits.scalar_per_vector == vec
    assert traits.num_access == chunks

    b = IRBuilder(f"dist_reduce_N{N}_b{block_size}_v{vec}")
    b.kernel.attrs["max_workgroup_size"] = block_size

    X = b.param("X", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 - ABI symmetry; equals grid_x
    _ = b.param("N", I32)  # noqa: F841 - validated by caller; equals encoding.X_lengths[0]

    tid = b.thread_id_x()
    row = b.block_id_x()

    # View the input as 1D (since the distribution is 1D). Origin
    # moves to the start of this row.
    x_view = make_naive_tensor_view_packed(X, shape=(N,), dtype=F16)
    row_off = b.mul(row, b.const_i32(N))
    x_tile = make_tile_window(x_view, lengths=(N,), origin=(row_off,))

    # load_tile produces a StaticDistributedTensor of f32 scalars,
    # one per Y position. The distribution drives the iteration
    # order (snake traversal across the non-vector Y dim) and the
    # vector width.
    distributed_x = load_tile(
        b, x_tile, distribution=distribution, ps=[[tid]], traits=traits
    )

    # Sweep Y space to fold the partial sum. The encoding guarantees
    # that every X element this thread "owns" appears exactly once
    # in distributed_x.storage.
    acc = b.const_f32(0.0)
    for y in distribution.iterate_ys():
        acc = b.fadd(acc, distributed_x.get(y))

    # Cross-thread reduction via LDS tree.
    lds = make_lds_view(b, dtype=F32, shape=(block_size,), name_hint="lds_red").base
    total = block_lds_reduce(b, acc, lds, tid, block_size=block_size, combine="sum")

    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        store_scalar_from_f32(b, Y, row, total, dtype="f16")

    return b.kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--vec", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("HIP device unavailable; exiting", file=sys.stderr)
        return 1

    kernel = build_distribution_reduce(
        N=args.N, block_size=args.block_size, vec=args.vec
    )
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(kernel))
    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kernel.name,
        signature=[
            {"name": "X", "type": "ptr<f16, global>"},
            {"name": "Y", "type": "ptr<f16, global>"},
            {"name": "M", "type": "i32"},
            {"name": "N", "type": "i32"},
        ],
    )

    torch.manual_seed(0)
    X = torch.randn(args.M, args.N, dtype=torch.float16, device="cuda") * 0.1
    Y = torch.zeros(args.M, dtype=torch.float16, device="cuda")
    launcher(
        {"X": X, "Y": Y, "M": args.M, "N": args.N},
        config=LaunchConfig(grid=(args.M, 1, 1), block=(args.block_size, 1, 1)),
    )
    synchronize_and_release()

    ref = X.float().sum(-1).half()
    diff = (Y.float() - ref.float()).abs().max().item()
    print(
        f"distribution-driven reduce  "
        f"M={args.M} N={args.N} bs={args.block_size} vec={args.vec}  "
        f"max_abs={diff:.3e}"
    )
    return 0 if diff == 0.0 else 2


if __name__ == "__main__":
    sys.exit(main())
