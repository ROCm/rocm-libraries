# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 02: layernorm2d forward (CK Tile parity).

CK Tile counterpart: ``example/ck_tile/02_layernorm2d``. Builds the
forward kernel from :mod:`ck_dsl.instances.layernorm2d` and emits the
matching manifest. Kernel signature:

    (X: ptr, Gamma: ptr, Beta: ptr, Y: ptr, M: i32, N: i32, eps: f32)
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
from ck_dsl.instances.layernorm2d import (  # noqa: E402
    LayerNorm2DSpec,
    build_layernorm2d,
    layernorm2d_signature,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--vec", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    spec = LayerNorm2DSpec(
        n_per_block=args.n,
        block_size=args.block_size,
        vec=args.vec,
        dtype=args.dtype,
    )
    kernel = build_layernorm2d(spec)
    artifact = compile_kernel(kernel, isa=args.isa)

    manifest = make_simple_op_manifest(
        artifact=artifact,
        kind="layernorm_fp16",
        op="layernorm",
        dtype=args.dtype,
        threads_per_block=spec.block_size,
        default_shape=[args.m, args.n],
        args_signature=layernorm2d_signature(spec),
        eps=args.eps,
        atoms=["tile.block_lds_reduce"],
        notes=(
            "CK Tile 02_layernorm2d parity: per-row mean + variance via "
            "LDS tree reduction, then affine (gamma, beta) normalization "
            "back to fp16. Verified vs fp32 numpy reference."
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
