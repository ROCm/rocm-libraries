# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Element-path + arch-gating tests for the fused-MoE dispatcher family."""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget
from rocke.dispatch.families.moe import (
    MOE_REGISTRY,
    MoeRequest,
    dispatch_moe,
    moe_candidates,
)

# Every (arch, dtype) the family claims to dispatch.
_ARCHES = ("gfx942", "gfx950")
_DTYPES = ("fp16", "bf16", "fp8")


def _moe(arch="gfx950", dtype="fp16", **kw):
    base = dict(
        num_tokens=128,
        hidden=7168,
        intermediate=2048,
        num_experts=256,
        top_k=8,
        arch=arch,
        dtype=dtype,
    )
    base.update(kw)
    return MoeRequest(**base)


def _emitted_lds_bytes(kernel, arch: str) -> int:
    """Bytes the smem packer reserves for ``kernel`` -- the number the hardware
    budget is actually spent on, not a spec-side estimate."""
    from rocke.core import lower_llvm

    low = lower_llvm._Lowerer(kernel, arch=arch)
    low._collect_smem(kernel.body)
    low._compute_smem_layout()
    return low._smem_pool_size


class TestMoeDispatch(unittest.TestCase):
    def test_fp16_selects_f16_mega(self):
        r = dispatch_moe(_moe(dtype="fp16"))
        self.assertEqual(r.candidate.spec_id, "mega_f16")

    def test_bf16_selects_f16_mega(self):
        r = dispatch_moe(_moe(dtype="bf16"))
        self.assertEqual(r.candidate.spec_id, "mega_f16")
        self.assertEqual(r.spec.dtype, "bf16")

    def test_fp8_selects_fp8_mega(self):
        r = dispatch_moe(_moe(dtype="fp8"))
        self.assertEqual(r.candidate.spec_id, "mega_fp8")
        # fp8 hero atom K=128.
        self.assertEqual(r.spec.gate_up_k, 128)

    def test_rejects_unknown_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(dtype="i8"))

    def test_rejects_topk_gt_experts(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(num_experts=4, top_k=8))

    def test_rejects_rdna_arch(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx1151"))

    def test_rejects_wave32_cdna_arch(self):
        # gfx1250 is CDNA at wave32 and has no MFMA; a family-level "is CDNA"
        # gate would let it through, which is why the capability lists arches.
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx1250"))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_moe(_moe(arch="gfx000"))

    def test_candidates_dtype_exclusive(self):
        req = _moe(dtype="fp8")
        supported = [c for c in moe_candidates() if c.admits(req)[0]]
        self.assertEqual([c.spec_id for c in supported], ["mega_fp8"])

    def test_unique_candidate_names(self):
        names = [c.name for c in moe_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_block_size_default(self):
        r = dispatch_moe(_moe(dtype="fp16"))
        # warp_m=1 * warp_n=4 * wave_size=64 = 256.
        self.assertEqual(r.spec.block_size, 256)


class TestGfx950SpecIsUnchanged(unittest.TestCase):
    """The gfx942 registration must not move gfx950.

    Pinned field by field rather than by hash: a hash tells you something
    changed, these tell you what.
    """

    def test_f16_spec_keeps_the_shipped_geometry(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                spec = dispatch_moe(_moe(arch="gfx950", dtype=dtype)).spec
                self.assertEqual(spec.name, f"moe_{dtype}")
                self.assertEqual(spec.tile_m, 16)
                self.assertEqual(spec.tile_n_inter, 256)
                self.assertEqual(spec.tile_k_gu, 32)
                self.assertEqual(spec.tile_n_down, 256)
                self.assertEqual(spec.tile_k_down, 64)
                self.assertEqual(spec.warp_tile_k, 32)

    def test_fp8_spec_keeps_the_shipped_geometry(self):
        spec = dispatch_moe(_moe(arch="gfx950", dtype="fp8")).spec
        self.assertEqual(spec.name, "moe_fp8")
        self.assertEqual((spec.gate_up_k, spec.down_k), (128, 128))
        self.assertEqual(spec.tile_n_down, 256)
        self.assertTrue(spec.use_dtla)


class TestGfx942Retile(unittest.TestCase):
    """gfx942 (CDNA3) reaches the mega through two forced departures.

    Neither is a tuning preference: the shipped f16 atom is CDNA4-only, and the
    shipped tiling's LDS pool does not fit CDNA3.
    """

    def test_f16_takes_the_gfx942_atom_and_down_tile(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                r = dispatch_moe(_moe(arch="gfx942", dtype=dtype))
                self.assertEqual(r.candidate.spec_id, "mega_f16")
                # 16x16x16 is gfx942's widest f16/bf16 MFMA.
                self.assertEqual(r.spec.warp_tile_k, 16)
                # Halves Bd_smem: 74752 B does not fit CDNA3's 65536 B.
                self.assertEqual(r.spec.tile_n_down, 128)
                # Everything else is the shipped geometry.
                self.assertEqual(r.spec.tile_m, 16)
                self.assertEqual(r.spec.tile_n_inter, 256)
                self.assertEqual(r.spec.tile_k_gu, 32)
                self.assertEqual(r.spec.tile_k_down, 64)

    def test_fp8_takes_the_catalog_atom_not_the_hero_atom(self):
        r = dispatch_moe(_moe(arch="gfx942", dtype="fp8"))
        self.assertEqual(r.candidate.spec_id, "mega_fp8")
        # 16x16x128 is a gfx950-only scaled intrinsic; 16x16x32 is on both.
        self.assertEqual((r.spec.gate_up_k, r.spec.down_k), (32, 32))

    def test_fp8_does_not_reuse_the_hero_kernel_symbol(self):
        # FusedMegaKernelSpecFp8.kernel_name() encodes only the block tile, so
        # the K=32 and K=128 kernels would collide on one entry-point symbol
        # without a distinct spec name.
        names = {
            arch: dispatch_moe(_moe(arch=arch, dtype="fp8")).build().name
            for arch in _ARCHES
        }
        self.assertEqual(len(set(names.values())), 2, names)

    def test_the_shipped_f16_atom_is_still_refused_where_it_does_not_exist(self):
        # The predicate gates on the atom the SELECTED spec names, so this is
        # the check that would fail if the gfx942 spec ever regained K=32.
        target = ArchTarget.from_gfx("gfx942")
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                self.assertFalse(
                    target.mma.has_shape(
                        family="mma",
                        a_dtype=dtype,
                        b_dtype=dtype,
                        c_dtype="fp32",
                        m=16,
                        n=16,
                        k=32,
                    )
                )


class TestSelectedSpecIsRunnable(unittest.TestCase):
    """Whatever the dispatcher picks must build and fit the target's LDS.

    This is the invariant the gfx942 registration turns on: the shipped tiling
    emits a 74,752 B pool, which is 9,216 B over CDNA3's budget and would show
    up as a kernel-load failure on device rather than a dispatch rejection.
    """

    def test_every_selection_builds(self):
        for arch in _ARCHES:
            for dtype in _DTYPES:
                with self.subTest(arch=arch, dtype=dtype):
                    kernel = dispatch_moe(_moe(arch=arch, dtype=dtype)).build()
                    self.assertTrue(getattr(kernel, "name", ""))

    def test_every_selection_fits_the_target_lds_budget(self):
        for arch in _ARCHES:
            cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
            for dtype in _DTYPES:
                with self.subTest(arch=arch, dtype=dtype):
                    kernel = dispatch_moe(_moe(arch=arch, dtype=dtype)).build()
                    self.assertLessEqual(_emitted_lds_bytes(kernel, arch), cap)

    def test_gfx942_lds_is_the_measured_retile_cost(self):
        # Pinned so a future retile has to restate what it costs.
        expected = {"fp16": 58368, "bf16": 58368, "fp8": 37008}
        for dtype, want in expected.items():
            with self.subTest(dtype=dtype):
                kernel = dispatch_moe(_moe(arch="gfx942", dtype=dtype)).build()
                self.assertEqual(_emitted_lds_bytes(kernel, "gfx942"), want)

    def test_every_selection_targets_the_requested_arch(self):
        # A builder ignoring `arch` would emit one kernel for both.
        for dtype in _DTYPES:
            with self.subTest(dtype=dtype):
                names = {
                    arch: dispatch_moe(_moe(arch=arch, dtype=dtype)).build().name
                    for arch in _ARCHES
                }
                self.assertEqual(len(set(names.values())), len(_ARCHES), names)

    def test_every_selection_lowers_to_an_mfma_that_exists_on_the_target(self):
        # The end of the chain the arch gate exists to protect: gfx942 must not
        # emit a CDNA4 atom, and must not emit the scaled f8f6f4 intrinsic it
        # has no instruction for.
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        expected = {
            ("gfx950", "fp16"): "mfma.f32.16x16x32.f16",
            ("gfx950", "bf16"): "mfma.f32.16x16x32.bf16",
            ("gfx950", "fp8"): "mfma.scale.f32.16x16x128.f8f6f4",
            ("gfx942", "fp16"): "mfma.f32.16x16x16f16",
            ("gfx942", "bf16"): "mfma.f32.16x16x16bf16.1k",
            ("gfx942", "fp8"): "mfma.f32.16x16x32.fp8.fp8",
        }
        for (arch, dtype), intrinsic in expected.items():
            with self.subTest(arch=arch, dtype=dtype):
                kernel = dispatch_moe(_moe(arch=arch, dtype=dtype)).build()
                ll = lower_kernel_to_llvm(kernel, arch=arch)
                self.assertIn(intrinsic, ll)
                if arch == "gfx942":
                    # The two gfx950-only forms, spelled out so a regression
                    # names itself rather than showing up as a missing string.
                    self.assertNotIn("16x16x32.f16", ll)
                    self.assertNotIn("mfma.scale.", ll)


class TestShapeCoverage(unittest.TestCase):
    """The tile is static and unpredicated, so a ragged shape is a rejection.

    ``grid.x`` is ``ceil(I / tile_n_inter)`` but the down k-loop contracts a
    constant ``tile_n_inter``, and the H_out loop steps ``tile_n_down`` into an
    epilogue whose atomic add is unguarded. Admitting a shape the tile cannot
    divide is a silent out-of-bounds access on device.
    """

    def test_ragged_intermediate_is_refused_with_a_stated_reason(self):
        for arch in _ARCHES:
            with self.subTest(arch=arch):
                with self.assertRaises(ValueError) as ctx:
                    dispatch_moe(_moe(arch=arch, dtype="bf16", intermediate=1408))
                self.assertIn("tile_n_inter", str(ctx.exception))

    def test_ragged_hidden_is_refused_with_a_stated_reason(self):
        # 2880 leaves 64 against both down tiles (256 on gfx950, 128 on
        # gfx942); the default intermediate divides, so only H can be at fault.
        for arch in _ARCHES:
            with self.subTest(arch=arch):
                with self.assertRaises(ValueError) as ctx:
                    dispatch_moe(_moe(arch=arch, dtype="bf16", hidden=2880))
                self.assertIn("tile_n_down", str(ctx.exception))

    def test_the_gpt_oss_shape_is_refused_rather_than_silently_served(self):
        # H = I = 2880 divides neither tile on either arch. It used to be
        # admitted and handed the shipped tile regardless of shape.
        for arch in _ARCHES:
            req = _moe(arch=arch, dtype="bf16", hidden=2880, intermediate=2880)
            with self.subTest(arch=arch):
                self.assertRaises(ValueError, dispatch_moe, req)

    def test_shapes_the_tile_divides_are_still_admitted(self):
        for arch in _ARCHES:
            for dtype in _DTYPES:
                with self.subTest(arch=arch, dtype=dtype):
                    self.assertTrue(
                        dispatch_moe(
                            _moe(
                                arch=arch,
                                dtype=dtype,
                                hidden=4096,
                                intermediate=14336,
                            )
                        )
                    )


class TestSpecIdentity(unittest.TestCase):
    """``compile_key`` is documented as an HSACO cache key, so two selections
    that compile to different binaries must not share one."""

    def test_compile_key_separates_every_dispatchable_selection(self):
        keys = {}
        for arch in _ARCHES:
            for dtype in _DTYPES:
                key = dispatch_moe(_moe(arch=arch, dtype=dtype)).kernel_id.compile_key
                keys.setdefault(key, []).append((arch, dtype))
        collisions = {k: v for k, v in keys.items() if len(v) > 1}
        self.assertEqual(collisions, {})
        self.assertEqual(len(keys), len(_ARCHES) * len(_DTYPES))

    def test_fp16_and_bf16_do_not_share_a_compiled_binary(self):
        # They build different MFMA intrinsics and different entry points.
        for arch in _ARCHES:
            with self.subTest(arch=arch):
                a = dispatch_moe(_moe(arch=arch, dtype="fp16"))
                b = dispatch_moe(_moe(arch=arch, dtype="bf16"))
                self.assertNotEqual(a.kernel_id.compile_key, b.kernel_id.compile_key)
                self.assertNotEqual(a.build().name, b.build().name)

    def test_spec_id_stays_arch_independent(self):
        # One candidate per element path spanning both arches, so a caller
        # pinning a spec_id does not have to know which arch it is on.
        for arch in _ARCHES:
            with self.subTest(arch=arch):
                r = dispatch_moe(_moe(arch=arch, dtype="bf16", spec_id="mega_f16"))
                self.assertEqual(r.candidate.name, "moe_fused_mega_f16")


class TestDeclaredCoverage(unittest.TestCase):
    def test_both_element_paths_are_declared_on_both_arches(self):
        for arch in _ARCHES:
            with self.subTest(arch=arch):
                self.assertEqual(
                    {c.spec_id for c in MOE_REGISTRY.for_arch(arch)},
                    {"mega_f16", "mega_fp8"},
                )

    def test_no_other_arch_is_declared(self):
        declared = {a for c in moe_candidates() for a in c.capability.arches}
        self.assertEqual(declared, set(_ARCHES))


if __name__ == "__main__":
    unittest.main()
