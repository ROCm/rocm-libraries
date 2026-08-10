# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Co-located, build-time producer for the gemm_wmma_universal family: emit a
# gfx1151 *tiled* WMMA GEMM (RCR) .co (HSACO) sweep, for f16 and bf16, into
# argv[1]. Runtime never touches ck_dsl -- the per-family CMakeLists runs this at
# build time to drop the .co next to the checked-in family.json (family.json is
# the source of truth; this script emits .co ONLY).
#
# Run standalone from rocKE's python dir with PYTHONPATH=. e.g.:
#   cd /home/brpepers/rocKE/projects/composablekernel/python
#   PYTHONPATH=. python3 produce_gemm_universal_co.py <out_dir>
#
# WHY THIS FILE EXISTS (vs. produce_gemm_wmma_co.py): that producer builds the
# no-LDS one-wave-per-16x16-tile *reference* kernel -- correct but ~3-12x slower
# than hipBLASLt. This one calls rocKE's real tiled builder
# `ck_dsl.instances.common.gemm_universal.build_universal_gemm`, which emits an
# LDS-staged, register-blocked, multi-WMMA-per-workgroup GEMM (verified on gfx1151
# upstream via examples/common/universal_gemm_verify.py). We use ck_dsl purely as a
# *library* -- NO rocKE edit -- exactly mirroring that example's known-good spec.
#
# TILE/PIPELINE SWEEP: emits a sweep of tile/warp/pipeline configs (bigger tiles
# quarter the A/B global traffic; wmma_v1 is the alternate compute schedule) so
# the engine's TuneCache can measure-and-pick the fastest per shape. Configs the
# gfx1151 WMMA gate (`GemmPipelinePolicy.validate`) rejects are skipped with a
# printed reason. Symbols are auto-derived and MUST match family.json's co_file
# names (ugemm_gfx1151_{fp16,bf16}_t..._w..._wt16x16x16_..._intrawave_default).
#
# LAUNCH CONTRACT (encoded in family.json): block (block_size,1,1); grid
# (ceil(N/tile_n), ceil(M/tile_m), 1) grid_order "NM" (block_id.x -> N-tile --
# the OPPOSITE of the reference kernel; getting this backwards silently computes
# garbage); ABI (A,B,C ptrs; M,N,K i32); LDS static; workspace 0; RCR layout.
# A/B loads are unmasked and the K-loop has no tail, so each kernel REQUIRES
# M%tile_m==0, N%tile_n==0, K%tile_k==0 (its family.json entry constrains M/N/K
# to those exact tile multiples).

import os
import sys

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers.compile import compile_kernel
from ck_dsl.instances import GemmPipelinePolicy
from ck_dsl.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)

ARCH = "gfx1151"

# f16 uses the "fp16" spelling in the spec/symbol (matches the upstream example);
# bf16 is "bf16". The catalog dtype *constraint* tokens ("f16"/"bf16") live in
# family.json, not here.
_DTYPES = ["fp16", "bf16"]

# The sweep. Each row is (tile_m, tile_n, tile_k, warp_m, warp_n, pipeline). The
# per-warp register-blocking repeat is derived = tile_/(warp_*warp_tile_). Bigger
# tiles = fewer redundant A/B global loads (the LTX-shape lever); wmma_v1 is the
# alternate schedule. Invalid-on-gfx1151 rows are skipped by the policy gate.
_SWEEP = [
    (64, 64, 32, 2, 2, "mem"),
    (128, 128, 32, 4, 4, "mem"),
    (128, 128, 32, 2, 2, "mem"),
    (128, 64, 32, 4, 2, "mem"),
    (128, 128, 32, 4, 4, "wmma_v1"),
    (64, 64, 32, 2, 2, "wmma_v1"),
]


def _pick_atom(target: ArchTarget, dtype: str):
    """Largest-K 16x16 WMMA atom for the dtype (gfx1151 has only 16x16x16).

    Copied from examples/common/universal_gemm_verify.py:_pick_atom so we resolve
    the atom exactly as the verified example does."""
    family = "wmma" if target.wave_size == 32 else "mma"
    op = target.mma.select_largest_k(
        family=family, a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=16, n=16
    )
    if op is None:
        raise SystemExit(f"no 16x16 {family} atom for {dtype} on {target.gfx}")
    return (op.m, op.n, op.k)


def _build_spec(dtype, target, tm, tn, tk, wm, wn, pipeline):
    """Build a UniversalGemmSpec for one sweep row. pad_* left False because the
    catalog constrains to exact tile multiples (no partial tiles ever)."""
    wtm, wtn, wtk = _pick_atom(target, dtype)
    tile = TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=max(tk, wtk),
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
        warp_tile_m=wtm,
        warp_tile_n=wtn,
        warp_tile_k=wtk,
    )
    trait = TraitSpec(
        pipeline=pipeline,  # only mem/wmma_v1 are legal on the gfx1151 WMMA path
        scheduler="intrawave",
        epilogue="default",  # cshuffle is MFMA-only on this arch
        pad_m=False,
        pad_n=False,
        pad_k=False,
    )
    data = DataSpec(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=dtype,
        dtype_acc="fp32",
        layout="RCR",
    )
    return UniversalGemmSpec(
        name=f"ugemm_{ARCH}",
        tile=tile,
        trait=trait,
        data=data,
        wave_size=target.wave_size,  # 32 on gfx1151
    )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    target = ArchTarget.from_gfx(ARCH)
    policy = GemmPipelinePolicy()

    seen = set()
    for dtype in _DTYPES:
        for tm, tn, tk, wm, wn, pipeline in _SWEEP:
            spec = _build_spec(dtype, target, tm, tn, tk, wm, wn, pipeline)
            res = policy.validate(target, spec)
            if not res.ok:
                print(
                    f"SKIP {dtype} t{tm}x{tn}x{tk} w{wm}x{wn} {pipeline}: {res.reason}"
                )
                continue

            artifact = compile_kernel(build_universal_gemm(spec, arch=ARCH), arch=ARCH)
            symbol = artifact.kernel_name
            if symbol in seen:
                continue  # dedupe identical configs
            seen.add(symbol)

            out_path = os.path.join(out_dir, symbol + ".co")
            with open(out_path, "wb") as f:
                f.write(artifact.hsaco)
            print(
                f"dtype={dtype} symbol={symbol} block_size={spec.block_size} "
                f"tile={tm}x{tn}x{tk} warp={wm}x{wn} pipeline={pipeline} "
                f"bytes={len(artifact.hsaco)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
