#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_coverage.py -- tiered rolling status across representative kernels.
#
# For each tier it records concrete traces from an UNMODIFIED production builder,
# rolls over one structural axis, and verifies the parametric recipe with the
# recipe_expand oracle at sampled AND held-out shapes. Reports ROLLED (with
# compression) or, for kernels beyond the current roller, a precise FALLBACK
# reason -- the roller is safe-by-construction (the oracle rejects any bad roll),
# so a fallback is "not compressed", never "wrong".
#
#   T1 small op       : qk_block (vec8 head dot)              over head_size D
#   T2 GEMM           : gemm_universal (k-atom nest)          over tile_n
#   T3 attention      : unified-attention 2D (the Section-3 kernel) over head_size
#
#   python3 -m rocke.portable_ir.drivers.roll_coverage
#
# The tiers differ in how their kernel is reached, and that is the point of
# running them together: T2 is a spec-driven production kernel, while T1
# and T3 are example builders taking plain arguments with no spec dataclass at
# all. All three go through the same roll_kernel entry point, which is the driver
# every other one here is built on, so this doubles as the check that it fits
# kernels that do not follow the conventions its module lookup assumes.

from typing import Any, Dict, List

from rocke.portable_ir.drivers.roll_kernel import Kernel, recipe_ops, roll

ARCH = "gfx950"


def _tiers():
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )
    from rocke.portable_ir.examples import export_mha, qk_block

    # T2's name carries its own axis, so the roller has to fit the kernel symbol
    # as well as the constants -- `name=f"g{tn}"` is what makes that non-trivial.
    gemm = Kernel(
        label="gemm_universal",
        spec_cls=UniversalGemmSpec,
        build_fn=build_universal_gemm,
        make_spec=lambda **kw: UniversalGemmSpec(
            name=f"g{kw['tile_n']}",
            tile=TileSpec(16, kw["tile_n"], 16, 1, 1, 1, 16, 16, 16),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            data=DataSpec(),
            wave_size=64,
            block_size=64,
        ),
    )
    return [
        (
            "T1",
            "qk_block",
            dict(
                kernel=lambda D: qk_block.build_qk_block(D, "f16"),
                axes={"D": [64, 128]},
                holdout={"D": [256, 192, 96]},
                structural="D",
                extra_spec={"dtype": "f16"},
            ),
        ),
        (
            "T2",
            "gemm_universal (CShuffle)",
            dict(
                kernel=gemm,
                axes={"tile_n": [32, 64]},
                holdout={"tile_n": [128, 256]},
                structural="tile_n",
            ),
        ),
        (
            "T3",
            "unified-attention-2d",
            dict(
                kernel=lambda D: export_mha.build("fp16", D, 2048, 1, 32, 1),
                axes={"D": [64, 128]},
                holdout={"D": [256, 192, 96, 512]},
                structural="D",
                extra_spec={"dtype": "fp16"},
            ),
        ),
    ]


def _report(r) -> str:
    """One line per tier: what the recipe covers, and what it saved to cover it."""
    if not r.rolled:
        return f"FALLBACK: {r.reason}"
    kib = len(r.cbor or b"") / 1024
    concrete = sum(r.trace_bytes) / 1024
    return (
        f"ROLLED  {recipe_ops(r.recipe)} ops covers {len(r.points)} shapes "
        f"from {r.n_recorded} traces; {kib:.1f}KiB vs {concrete:.1f}KiB concrete "
        f"({concrete / max(kib, 1e-9):.0f}x)"
    )


def run_coverage() -> List[Dict[str, Any]]:
    rows = []
    for tier, label, kw in _tiers():
        try:
            r = roll(arch=ARCH, quiet=True, **kw)
            rows.append(
                {
                    "tier": tier,
                    "label": label,
                    "ok": r.rolled,
                    "report": _report(r),
                    "result": r,
                    "error": None,
                }
            )
        except Exception as e:  # noqa: BLE001 - record, don't abort the matrix
            rows.append(
                {
                    "tier": tier,
                    "label": label,
                    "ok": False,
                    "report": f"ERROR: {type(e).__name__}: {e}",
                    "result": None,
                    "error": e,
                }
            )
    return rows


def main() -> int:
    rows = run_coverage()
    for row in rows:
        print(f"  [{row['tier']}] {row['label']:<24} {row['report']}")
    rolled = sum(1 for r in rows if r["ok"])
    print("-" * 76)
    print(
        f"rolled+verified: {rolled}/{len(rows)}   "
        f"(fallbacks are safe: the oracle rejects any non-byte-identical roll)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
