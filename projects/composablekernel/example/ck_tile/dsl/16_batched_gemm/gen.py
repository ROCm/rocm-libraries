# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 16: batched GEMM (CK Tile parity).

CK Tile counterpart: ``example/ck_tile/16_batched_gemm``. Builds the
batched universal GEMM body from :mod:`ck_dsl.instances.batched_gemm`.

Layout is RCR:

    A[batch, M, K] @ B[batch, N, K]^T -> C[batch, M, N]

The kernel reads ``block_id_z`` as the batch index and applies the
runtime strides ``stride_a = M*K``, ``stride_b = N*K``,
``stride_c = M*N``.
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
    make_gemm_manifest,
    write_artifact,
)
from ck_dsl.instances import (  # noqa: E402
    BatchedGemmSpec,
    TileSpec,
    TraitSpec,
    batched_gemm_signature,
    build_batched_gemm,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    args = parser.parse_args()

    spec = BatchedGemmSpec(
        name="ck_dsl_ex16_batched_gemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"),
        batch_size=args.batch,
    )

    kernel = build_batched_gemm(spec)
    artifact = compile_kernel(kernel, isa=args.isa)
    manifest = make_gemm_manifest(
        artifact=artifact,
        block_m=spec.tile.tile_m,
        block_n=spec.tile.tile_n,
        block_k=spec.tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(args.m, args.n, args.k),
        args_signature=batched_gemm_signature(spec),
        atoms=["tile.mfma_f32_32x32x16_f16"],
        notes=(
            "CK Tile 16_batched_gemm parity: universal GEMM body in "
            "batched mode, where block_id.z chooses the batch and "
            "runtime element strides offset A/B/C."
        ),
        extra={
            "kind": "batched_gemm_fp16",
            "default_shape": [args.batch, args.m, args.n, args.k],
        },
    )
    paths = write_artifact(artifact, Path(args.output_dir), manifest)
    print(
        f"emitted {paths['hsaco']} ({artifact.hsaco_bytes} bytes) in "
        f"{artifact.timings['total']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
