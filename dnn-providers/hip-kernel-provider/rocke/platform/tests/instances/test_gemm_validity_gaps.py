# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression tests for three gaps between our validator and upstream's.

All three were found by diffing ``is_valid_spec`` against upstream PR 12658
rather than by chasing a symptom, and all three live in the validator rather
than the emitter -- which is why the byte-identity gate could not see them and
why every other test in this directory missed them.

1. **The cshuffle LDS budget was over-charged.** The gate charged ``AB + C``
   while the emitter's smem packer aliases the C staging tile onto the A/B
   pool, so peak usage is ``max(AB, C)``. A 256x256 C tile is 128 KiB, enough
   to push a geometry that runs fine past the cap -- it silently removed the
   deepest tiles from every cshuffle sweep, and a perf conclusion was drawn
   from the resulting (smaller) champion before anyone noticed.

2. **``lds_swizzle`` was admitted on the WMMA path**, where the store side
   permutes the LDS column and ``_emit_wmma_phase`` never applies the inverse.
   It emits, runs fast, and returns wrong answers, so a sweep ranks it as a
   winner while it is incorrect.

3. **``tdm`` was admitted on MFMA targets.** The ``has_tdm`` check sat inside
   the ``family == "wmma"`` block, which an MFMA target never enters. Not a
   correctness hole -- the lowerer raises -- but the spec reaches the build and
   is recorded as a build failure rather than an invalid spec.

Assertions 1 and 2 are differential against the emitter rather than against a
constant: (1) compares the gate's verdict with the LDS the emitter actually
allocates, so it still holds if the pool packer changes; (2) checks the trait
is refused rather than checking a byte count.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

ARCH = "gfx1250"


def _fp8_spec(*, tile_k: int = 256, epilogue: str = "cshuffle", **trait_kw):
    """The fp8 square-sweep champion geometry, parameterised."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="validity_probe",
        tile=TileSpec(
            tile_m=256,
            tile_n=256,
            tile_k=tile_k,
            warp_m=4,
            warp_n=8,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=64,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue=epilogue,
            **{"tdm": True, "tdm_depth": 2, "lds_k_pad": 48, **trait_kw},
        ),
        data=DataSpec(
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
        block_size=1024,
    )


def _emitted_lds_bytes(spec, arch: str = ARCH) -> int:
    """Total bytes of ``addrspace(3)`` globals the emitter actually allocates."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common.gemm_universal import build_universal_gemm

    ll = lower_kernel_to_llvm(build_universal_gemm(spec, arch=arch), arch=arch)
    width = {"i8": 1, "i16": 2, "i32": 4, "half": 2, "bfloat": 2, "float": 4}
    total = 0
    for m in re.finditer(
        r"addrspace\(3\) global \[(\d+) x (i8|i16|i32|half|bfloat|float)\]", ll
    ):
        total += int(m.group(1)) * width[m.group(2)]
    return total


class TestCshuffleLdsBudget(unittest.TestCase):
    def test_gate_matches_what_the_emitter_allocates(self):
        """The budget must model the alias, not the sum.

        Asserted against the emitter's own output rather than a constant: the
        aliasing is a property of the smem pool packer, so if that changes this
        test should follow it instead of freezing today's number.
        """
        from rocke.core.arch import ArchTarget
        from rocke.instances.common.gemm_universal import _ab_lds_plan, is_valid_spec

        spec = _fp8_spec()
        ok, why = is_valid_spec(spec, arch=ARCH)
        self.assertTrue(ok, f"aliased cshuffle spec rejected: {why}")

        emitted = _emitted_lds_bytes(spec)
        ab_single, _, _ = _ab_lds_plan(spec, ARCH)
        c_bytes = spec.tile.tile_m * spec.tile.tile_n * 2

        self.assertLess(
            emitted,
            ab_single * 2 + c_bytes,
            "emitter is NOT aliasing the C tile onto A/B; if that is "
            "deliberate the budget must go back to charging the sum",
        )
        self.assertLessEqual(
            emitted,
            ArchTarget.from_gfx(ARCH).lds_capacity_bytes,
            "emitted LDS exceeds the cap, so the gate is now too permissive",
        )

    def test_the_sum_would_have_rejected_this_spec(self):
        """Pins that the fix is load-bearing on a geometry anyone would run.

        Without this the first test could pass on a spec that fits either way,
        and the regression it guards would come back unnoticed.
        """
        from rocke.core.arch import ArchTarget
        from rocke.instances.common.gemm_universal import _ab_lds_plan

        spec = _fp8_spec()
        ab_single, _, _ = _ab_lds_plan(spec, ARCH)
        ab_bytes = ab_single * 2  # tdm_depth=2
        c_bytes = spec.tile.tile_m * spec.tile.tile_n * 2
        cap = ArchTarget.from_gfx(ARCH).lds_capacity_bytes

        self.assertGreater(
            ab_bytes + c_bytes, cap, "this geometry no longer exercises the fix"
        )
        self.assertLessEqual(max(ab_bytes, c_bytes), cap)

    def test_cshuffle_no_alias_is_still_charged_additively(self):
        """The opt-out really does cost both tiles, so the gate must charge both."""
        from rocke.instances.common.gemm_universal import is_valid_spec

        spec = _fp8_spec(cshuffle_no_alias=True)
        ok, why = is_valid_spec(spec, arch=ARCH)
        self.assertFalse(
            ok,
            "cshuffle_no_alias marks the C tile exclusive, so this geometry "
            "cannot fit; admitting it means the gate stopped modelling the "
            "opt-out",
        )
        self.assertIn("LDS budget", why)

    def test_default_epilogue_budget_is_unchanged(self):
        """The alias branch must not touch specs that have no C tile."""
        from rocke.instances.common.gemm_universal import _ab_lds_plan, is_valid_spec

        spec = _fp8_spec(epilogue="default")
        ok, why = is_valid_spec(spec, arch=ARCH)
        self.assertTrue(ok, why)
        ab_single, _, _ = _ab_lds_plan(spec, ARCH)
        self.assertEqual(_emitted_lds_bytes(spec), ab_single * 2)


class TestWmmaRejectsLdsSwizzle(unittest.TestCase):
    def test_rejected_on_the_wmma_path(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for tdm in (True, False):
            with self.subTest(tdm=tdm):
                spec = _fp8_spec(
                    epilogue="default", lds_swizzle=True, tdm_depth=2 if tdm else 1
                )
                spec = replace(spec, trait=replace(spec.trait, tdm=tdm))
                ok, why = is_valid_spec(spec, arch=ARCH)
                self.assertFalse(
                    ok,
                    "lds_swizzle is wrong on the WMMA read path (no _swz_col "
                    "in _emit_wmma_phase) and inert under TDM; it must not be "
                    "admitted either way",
                )
                self.assertIn("lds_swizzle", why)

    def test_still_allowed_on_the_mfma_path(self):
        """Family scope, not a global gate -- it is bit-exact and pays on CDNA."""
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            is_valid_spec,
        )

        spec = UniversalGemmSpec(
            name="mfma_swz",
            tile=TileSpec(
                tile_m=128, tile_n=128, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
                warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
                lds_swizzle=True,
            ),
            data=DataSpec(
                dtype_a="bf16", dtype_b="bf16", dtype_c="bf16",
                dtype_acc="fp32", layout="RCR",
            ),
            wave_size=64,
        )
        ok, why = is_valid_spec(spec, arch="gfx950")
        self.assertTrue(ok, f"lds_swizzle must stay valid on MFMA: {why}")


class TestTdmGateRunsBeforeTheFamilyBlock(unittest.TestCase):
    def test_mfma_target_rejects_tdm(self):
        """gfx950 has no mover; the gate must say so instead of the lowerer."""
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            is_valid_spec,
        )

        spec = UniversalGemmSpec(
            name="mfma_tdm",
            tile=TileSpec(
                tile_m=128, tile_n=128, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
                warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
                tdm=True, tdm_depth=2,
            ),
            data=DataSpec(
                dtype_a="bf16", dtype_b="bf16", dtype_c="bf16",
                dtype_acc="fp32", layout="RCR",
            ),
            wave_size=64,
        )
        ok, why = is_valid_spec(spec, arch="gfx950")
        self.assertFalse(ok, "an MFMA target must not admit tdm=True")
        self.assertIn("Tensor Data Mover", why)

    def test_gfx1250_still_admits_tdm(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        ok, why = is_valid_spec(_fp8_spec(epilogue="default"), arch=ARCH)
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
