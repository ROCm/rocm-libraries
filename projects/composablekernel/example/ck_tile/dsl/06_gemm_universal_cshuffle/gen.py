# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 06: universal GEMM builder + cshuffle epilogue.

First example that goes through `ck_dsl.instances.gemm_universal` —
the same code path the sweep harness drives for every dispatcher
config. The spec mirrors CK's `_base_fp16_rcr_compute` 2x2 entry
(`tile=(128,128,32), warp_grid=(2,2,1), warp_tile=(32,32,16),
pipeline=compv4, scheduler=intrawave, epilogue=cshuffle, block_size=256`)
so the comparison against the dispatcher is direct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "python"))

from ck_dsl.helpers import compile_kernel, make_gemm_manifest, write_artifact  # noqa: E402
from ck_dsl.instances import (  # noqa: E402
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    args = parser.parse_args()

    spec = UniversalGemmSpec(
        name="ck_dsl_ex06_universal_cshuffle",
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
        trait=TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="cshuffle",
        ),
    )

    kernel = build_universal_gemm(spec)
    artifact = compile_kernel(kernel, isa=args.isa)

    manifest = make_gemm_manifest(
        artifact=artifact,
        block_m=spec.tile.tile_m,
        block_n=spec.tile.tile_n,
        block_k=spec.tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(3328, 4096, 4096),
        warmup_iters=5,
        timed_iters=100,
        atoms=["tile.mfma_f32_32x32x16_f16"],
        notes=(
            "Universal-builder path. Same tile geometry as CK's "
            "preselected_fp16_rcr_compute 2x2 entry; uses the cshuffle "
            "epilogue to LDS-stage fp16 accumulators before "
            "wide-vector global stores (runbook §9.3 lever)."
        ),
        extra={
            "pipeline": spec.trait.pipeline,
            "epilogue": spec.trait.epilogue,
            "scheduler": spec.trait.scheduler,
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
