# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 05: row reduction (CK Tile parity).

CK Tile counterpart: ``example/ck_tile/05_reduce``. Builds the
row-reduction kernel from :mod:`ck_dsl.instances.reduce`: one CTA
per row, vec-wide chunks, LDS tree reduction, scalar write per row.
The default op is ``sum``; pass ``--op`` for ``mean`` / ``max``.
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
from ck_dsl.instances.reduce import (  # noqa: E402
    Reduce2DSpec,
    build_reduce2d,
    reduce2d_signature,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument("--op", default="sum", choices=("sum", "mean", "max"))
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--vec", type=int, default=4)
    args = parser.parse_args()

    spec = Reduce2DSpec(
        n_per_block=args.n,
        op=args.op,
        block_size=args.block_size,
        vec=args.vec,
        dtype=args.dtype,
    )
    kernel = build_reduce2d(spec)
    artifact = compile_kernel(kernel, isa=args.isa)

    manifest = make_simple_op_manifest(
        artifact=artifact,
        kind="reduce_fp16",
        op=args.op,
        dtype=args.dtype,
        threads_per_block=spec.block_size,
        default_shape=[args.m, args.n],
        args_signature=reduce2d_signature(spec),
        atoms=["tile.block_lds_reduce"],
        notes=(
            "CK Tile 05_reduce parity: row-wise reduce(X[m,n]) -> Y[m]. "
            "One CTA per row, vec-wide chunks, LDS tree reduction. "
            "Verified vs a fp32 numpy reference within fp16 ULP."
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
