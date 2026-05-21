# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 37: 2D transpose (CK Tile parity).

CK Tile counterpart: ``example/ck_tile/37_transpose``. Builds the
LDS-staged 2D transpose kernel from :mod:`ck_dsl.instances.transpose`.
Kernel signature:

    (X: ptr, Y: ptr, H: i32, W: i32)

Layout: ``X[H, W] -> Y[W, H]``. Each CTA handles one ``(tile_m, tile_n)``
tile via an LDS-staged transpose that avoids bank conflicts (the
``lds_pad`` knob).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "python"))

from ck_dsl.helpers import (  # noqa: E402
    compile_kernel,
    make_simple_op_manifest,
    write_artifact,
)
from ck_dsl.instances.transpose import (  # noqa: E402
    Transpose2DSpec,
    build_transpose2d,
    transpose2d_signature,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    parser.add_argument("--h", type=int, default=2048)
    parser.add_argument("--w", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--vec", type=int, default=8)
    args = parser.parse_args()

    spec = Transpose2DSpec(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        vec=args.vec,
        dtype=args.dtype,
    )
    kernel = build_transpose2d(spec)
    artifact = compile_kernel(kernel, isa=args.isa)

    # Grid: (ceil(H / tile_m), ceil(W / tile_n), 1).
    gx = (args.h + spec.tile_m - 1) // spec.tile_m
    gy = (args.w + spec.tile_n - 1) // spec.tile_n

    manifest = make_simple_op_manifest(
        artifact=artifact,
        kind="transpose_fp16",
        op="transpose",
        dtype=args.dtype,
        threads_per_block=spec.block_size,
        default_shape=[args.h, args.w],
        args_signature=transpose2d_signature(spec),
        block_m=spec.tile_m,
        block_n=spec.tile_n,
        grid_explicit=[gx, gy, 1],
        atoms=["tile.lds_swizzle"],
        notes=(
            "CK Tile 37_transpose parity: LDS-staged 2D transpose, "
            "bank-conflict-free via lds_pad. Verified bit-exact "
            "(``bad=0``) vs a numpy reference."
        ),
    )
    paths = write_artifact(artifact, Path(args.output_dir), manifest)
    print(
        f"emitted {paths['hsaco']} ({artifact.hsaco_bytes} bytes) in "
        f"{artifact.timings['total']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
