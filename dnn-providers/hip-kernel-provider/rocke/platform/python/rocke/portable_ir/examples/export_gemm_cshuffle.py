#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# export_gemm_cshuffle.py -- roll the universal GEMM (CShuffle epilogue) over
# tile_n into ONE parametric recipe, and emit either the recipe JSON or the
# Python-lowered reference .ll for a given tile_n.
#
# tile_n is structural: the K-loop carries mfmas_m x mfmas_n accumulators (a
# variable loop-carry FAN) and the CShuffle epilogue writes them, all scaling
# with tile_n. The roller separates the shared A-tile from the per-lane B-tiles
# by data-flow (cone labeling + scratchpad store->load matching), rolls the
# per-lane phases, and parameterizes the scaling smem buffer shapes / barrier
# counts. The C recipe VM expands this per tile_n -> HSACO byte-identical to
# production. Replay it with `rocke_portable_ir_replay_cli --recipe <file>`, or
# in-process via rocke.portable_ir.src.online.
#
#   export_gemm_cshuffle.py --emit recipe
#   export_gemm_cshuffle.py --emit ll --TN 128 [--arch gfx950]
import argparse
import json
import sys

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)
from rocke.portable_ir.src.roll import roll


def build(tn: int, arch: str = "gfx950"):
    spec = UniversalGemmSpec(
        name=f"g{tn}",
        tile=TileSpec(16, tn, 16, 1, 1, 1, 16, 16, 16),
        trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        data=DataSpec(),
        wave_size=64,
        block_size=64,
    )
    return build_universal_gemm(spec, arch=arch)


def make_recipe() -> dict:
    r = roll(
        build,
        axis="TN",
        sample_points=[32, 64],
        holdout_points=[128, 256],
        spec_decl=[{"name": "TN", "kind": "int"}],
    )
    if not r.ok:
        raise SystemExit(f"roll failed: {r.reason}")
    return r.recipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe", "ll"], default="recipe")
    ap.add_argument("--TN", type=int, default=64)
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()
    if args.emit == "recipe":
        sys.stdout.write(json.dumps(make_recipe()))
    else:
        sys.stdout.write(
            lower_kernel_to_llvm(build(args.TN, args.arch), arch=args.arch)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
