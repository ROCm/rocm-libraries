# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 21: elementwise (CK Tile parity).

CK Tile counterpart: ``example/ck_tile/21_elementwise``. Emits one
elementwise kernel through :func:`ck_dsl.instances.build_elementwise`
and writes the matching manifest. The default op is ``gelu_tanh``
(transformer-style activation); pass ``--op`` to change it. Both
unary (``relu``, ``gelu_tanh``, ``silu``, ``neg``, ``abs``, ``exp2``,
``copy``) and binary (``add``, ``sub``, ``mul``, ``max``, ``min``)
ops are supported.
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
from ck_dsl.instances.elementwise import (  # noqa: E402
    ElementwiseSpec,
    build_elementwise,
    elementwise_signature,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument("--op", default="gelu_tanh")
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--vec", type=int, default=8)
    parser.add_argument("--n", type=int, default=1 << 20)
    args = parser.parse_args()

    spec = ElementwiseSpec(
        op=args.op,
        dtype=args.dtype,
        block_size=args.block_size,
        vec=args.vec,
    )
    kernel = build_elementwise(spec)
    artifact = compile_kernel(kernel, isa=args.isa)

    manifest = make_simple_op_manifest(
        artifact=artifact,
        kind="elementwise_fp16",
        op=args.op,
        dtype=args.dtype,
        threads_per_block=spec.block_size,
        default_shape=[args.n],
        elems_per_block=spec.elems_per_block(),
        is_binary=spec.is_binary(),
        args_signature=elementwise_signature(spec),
        atoms=["tile.pointwise"],
        notes=(
            "CK Tile 21_elementwise parity: vectorised pointwise kernel "
            "with vec-wide global loads/stores and an f32-internal "
            "compute path. Validates bit-exact vs a numpy fp32 reference."
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
