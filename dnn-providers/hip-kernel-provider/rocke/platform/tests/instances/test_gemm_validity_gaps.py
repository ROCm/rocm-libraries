# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression tests for three gaps between our validator and upstream's.

All three were found by diffing ``is_valid_spec`` against the reference
validator this port follows rather than by chasing a symptom, and all three live in the validator rather
than the emitter -- which is why the byte-identity gate could not see them and
why every other test in this directory missed them.

1. **The cshuffle LDS budget was over-charged.** The gate charged ``AB + C``
   while the emitter's smem packer aliases the C staging tile onto the A/B
   pool, so peak usage is ``max(AB, C)``. A 256x256 C tile is 128 KiB, enough
   to push a geometry that runs fine past the cap -- it silently removed the
   deepest tiles from every cshuffle sweep, and a perf conclusion was drawn
   from the resulting (smaller) champion before anyone noticed. The gate
   counts ``max(AB, C)`` on gfx1250 only, where it has run on silicon.

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

    # tensor_load_to_lds needs the llvm23 flavor; pin it rather than take the
    # environment's, as core/test_gfx1250_tdm.py does.
    ll = lower_kernel_to_llvm(
        build_universal_gemm(spec, arch=arch), arch=arch, llvm_flavor="llvm23"
    )
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
                tile_m=128,
                tile_n=128,
                tile_k=64,
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
                lds_swizzle=True,
            ),
            data=DataSpec(
                dtype_a="bf16",
                dtype_b="bf16",
                dtype_c="bf16",
                dtype_acc="fp32",
                layout="RCR",
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
                tile_m=128,
                tile_n=128,
                tile_k=64,
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
                tdm=True,
                tdm_depth=2,
            ),
            data=DataSpec(
                dtype_a="bf16",
                dtype_b="bf16",
                dtype_c="bf16",
                dtype_acc="fp32",
                layout="RCR",
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


def _mfma_fp8_spec(dtype: str = "fp8e4m3", c_dtype: str = "bf16"):
    """An fp8 GEMM on a wave64 MFMA atom: valid on gfx942/gfx950 in every
    respect except the fp8 A/B dtype, so the arch gate is the only thing that
    can reject it."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="fp8_on_mfma",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=128,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="mem", epilogue="default"),
        data=DataSpec(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=c_dtype,
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=64,
        block_size=256,
        batched=False,
    )


class TestFp8IsGfx1250Only(unittest.TestCase):
    """fp8/bf8 A/B is built and verified on gfx1250 only.

    gfx942 and gfx950 carry fp8 MFMA atoms, so without an explicit gate the
    catalog check admits them: gfx942 would read the OCP-encoded verify inputs
    as FNUZ, and the DirectToLDS path assumes 2-byte elements.
    """

    def test_fp8_rejected_on_mfma_targets(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for arch in ("gfx942", "gfx950"):
            for dtype in ("fp8e4m3", "bf8e5m2"):
                for c_dtype in ("bf16", "fp16"):
                    with self.subTest(arch=arch, dtype=dtype, c_dtype=c_dtype):
                        ok, why = is_valid_spec(_mfma_fp8_spec(dtype, c_dtype), arch)
                        self.assertFalse(ok)
                        self.assertEqual(
                            why,
                            f"fp8/bf8 A/B GEMM is not supported on {arch} (supported on: gfx1250)",
                        )

    def test_the_gate_is_the_only_reason(self):
        # Counterfactual: the same geometry in bf16 is valid on gfx950, so the
        # rejection above comes from the dtype gate, not from the tile.
        from rocke.instances.common.gemm_universal import is_valid_spec

        ok, why = is_valid_spec(_mfma_fp8_spec("bf16", "bf16"), "gfx950")
        self.assertTrue(ok, why)

    def test_fp8_still_valid_on_gfx1250(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for c_dtype in ("bf16", "fp16"):
            with self.subTest(c_dtype=c_dtype):
                spec = replace(
                    _fp8_spec(epilogue="default"),
                    data=replace(_fp8_spec().data, dtype_c=c_dtype),
                )
                ok, why = is_valid_spec(spec, ARCH)
                self.assertTrue(ok, why)


class TestAtomWithoutLaneMapIsRejected(unittest.TestCase):
    """An atom the catalog lists but whose A/B lane maps are not verified.

    The catalog lists a 16x16x128 bf8 WMMA row with no operand maps. A
    validator that accepts it hands the build an op with no A/B map to load
    through; it is rejected up front, with the same reason as the C++ engine.
    """

    @staticmethod
    def _spec(dtype: str, warp_tile_k: int):
        base = _fp8_spec(epilogue="default")
        return replace(
            base,
            tile=replace(base.tile, warp_tile_k=warp_tile_k),
            data=replace(base.data, dtype_a=dtype, dtype_b=dtype),
        )

    def test_bf8_16x16x128_rejected(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        ok, why = is_valid_spec(self._spec("bf8e5m2", 128), ARCH)
        self.assertFalse(ok)
        self.assertEqual(
            why,
            "no verified A/B lane map for bf8e5m2 warp_tile (16, 16, 128) on gfx1250",
        )

    def test_the_map_is_the_only_reason(self):
        # Counterfactual: fp8 on the same atom and bf8 on 16x16x64 are valid.
        from rocke.instances.common.gemm_universal import is_valid_spec

        for dtype, k in (("fp8e4m3", 128), ("bf8e5m2", 64)):
            with self.subTest(dtype=dtype, k=k):
                ok, why = is_valid_spec(self._spec(dtype, k), ARCH)
                self.assertTrue(ok, why)

    def test_every_admitted_gfx1250_atom_has_maps(self):
        # What the validator now guarantees: an admitted atom always builds its
        # A/B fragments, so the builder never meets the missing-map error.
        from rocke.instances.common.gemm_universal import (
            _resolve_mma_op,
            is_valid_spec,
        )

        for dtype in ("fp8e4m3", "bf8e5m2"):
            for k in (64, 128):
                spec = self._spec(dtype, k)
                if is_valid_spec(spec, ARCH)[0]:
                    with self.subTest(dtype=dtype, k=k):
                        self.assertTrue(_resolve_mma_op(spec, ARCH).has_ab_layouts())


def _bf16_spec(arch_tile, *, epilogue="default", **trait_kw):
    """A bf16 universal GEMM with the given (tile, warps, warp tile) geometry."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    (tm, tn, tk), (wm, wn), (wtm, wtn, wtk), pipeline, wave = arch_tile
    return UniversalGemmSpec(
        name="scope_probe",
        tile=TileSpec(
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_k=1,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
        ),
        trait=TraitSpec(
            pipeline=pipeline, scheduler="intrawave", epilogue=epilogue, **trait_kw
        ),
        data=DataSpec(
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_c="bf16",
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=wave,
    )


# gfx950 MFMA, 256x256x64 cshuffle: AB = 64 KiB, C = 128 KiB. The sum (192 KiB)
# exceeds the 160 KiB cap; the max (128 KiB) would not.
CDNA_CSHUFFLE = ((256, 256, 64), (2, 2), (32, 32, 16), "compv3", 64)
# gfx1151 WMMA, the 16x16x16 atom on the mem pipeline.
RDNA_WMMA = ((64, 64, 32), (2, 2), (16, 16, 16), "mem", 32)


class TestTargetScope(unittest.TestCase):
    """The aliased cshuffle budget is a gfx1250 rule; the WMMA lds_swizzle
    reject covers every WMMA target (gfx1151 returns wrong outputs with it)."""

    def test_cdna_keeps_the_summed_cshuffle_budget(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        ok, why = is_valid_spec(
            _bf16_spec(CDNA_CSHUFFLE, epilogue="cshuffle"), "gfx950"
        )
        self.assertFalse(ok, "max(AB, C) admitted a CDNA cshuffle spec")
        self.assertIn("LDS budget", why)

    def test_rdna_wmma_rejects_lds_swizzle(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        self.assertTrue(is_valid_spec(_bf16_spec(RDNA_WMMA), "gfx1151")[0])
        ok, why = is_valid_spec(_bf16_spec(RDNA_WMMA, lds_swizzle=True), "gfx1151")
        self.assertFalse(ok)
        self.assertIn("lds_swizzle", why)


class TestAbLoadElemBytesBounds(unittest.TestCase):
    """Python-only, so it runs without the C++ binding."""

    def test_out_of_range_widths_and_tdm_are_rejected(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = _fp8_spec()  # TDM
        no_tdm = replace(base, trait=replace(base.trait, tdm=False, tdm_depth=1))
        # Both bases are valid on their own, so a reject below is the width's.
        for ok_spec in (base, no_tdm):
            self.assertTrue(is_valid_spec(ok_spec, arch=ARCH)[0])
        for label, spec0, width in (
            ("0", no_tdm, 0),
            ("-1", no_tdm, -1),
            ("32", no_tdm, 32),
            ("2 with tdm", base, 2),
        ):
            with self.subTest(label):
                spec = replace(
                    spec0, trait=replace(spec0.trait, ab_load_elem_bytes=width)
                )
                ok, why = is_valid_spec(spec, arch=ARCH)
                self.assertFalse(ok, label)
                self.assertIn("ab_load_elem_bytes", why)


class TestTdmIssuersAreLaunchedWaves(unittest.TestCase):
    """A TDM fill must be issued by a wave the block launches.

    ``block_size`` is ``warp_m * warp_n`` waves; ``warp_k`` does not add any.
    Counting ``warp_k`` made a one-wave ``warp_k=2`` block hand the B fill to
    wave 1, which does not exist, so B was never loaded.
    """

    def test_warp_k_2_gates_no_fill_on_an_unlaunched_wave(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import (
            build_universal_gemm,
            is_valid_spec,
        )

        base = _fp8_spec(tile_k=128, epilogue="default")
        spec = replace(
            base,
            tile=replace(base.tile, tile_m=16, tile_n=16, warp_m=1, warp_n=1, warp_k=2),
            trait=replace(base.trait, tdm_depth=1, lds_k_pad=0),
            block_size=32,
        )
        ok, why = is_valid_spec(spec, arch=ARCH)
        self.assertTrue(ok, why)
        waves = spec.block_size // spec.wave_size
        self.assertEqual(waves, 1)

        ll = lower_kernel_to_llvm(
            build_universal_gemm(spec, arch=ARCH), arch=ARCH, llvm_flavor="llvm23"
        )
        self.assertIn("tensor.load.to.lds(", ll)
        # The wave id is readfirstlane(tid / wave_size).
        wave_ids = re.findall(
            r"(%[\w.]+) = call i32 @llvm\.amdgcn\.readfirstlane\.i32\(i32 %div",
            ll,
        )
        for wid in wave_ids:
            for m in re.finditer(rf"icmp eq i32 {re.escape(wid)}, (\d+)\b", ll):
                self.assertLess(
                    int(m.group(1)),
                    waves,
                    f"a fill is gated on wave {m.group(1)}; the block has {waves}",
                )


if __name__ == "__main__":
    unittest.main()
