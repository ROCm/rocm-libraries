# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the gfx1201 ``direct_a`` GEMM pipeline.

The spec gate, the host launch contract, and the loop structure that makes the
single LDS buffer safe. Numerical correctness on hardware is
``numeric/test_gfx1201_gemm_direct_a_numeric.py``; Python/C++ byte identity is
the ``gemm`` differential family (configs 45-48).
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.instances.common.gemm_direct_a import direct_a_launch_supported
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    wmma_lds_swizzle_row_shift,
)

ARCH = "gfx1201"


def _spec(tile=(128, 64, 64, 4, 1), data=None, **trait):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="direct_a",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, 16, 16, 16),
        trait=TraitSpec(
            **{
                "pipeline": "direct_a",
                "scheduler": "intrawave",
                "epilogue": "default",
                **trait,
            }
        ),
        data=data or DataSpec(),
        wave_size=32,
    )


def _ops(region, names):
    for op in region.ops:
        if op.name in names:
            yield op.name
        for child in op.regions:
            yield from _ops(child, names)


class TestDirectASpecGate(unittest.TestCase):
    def test_measured_tiles_are_valid(self):
        for tile in (
            (128, 64, 64, 4, 1),
            (128, 96, 64, 4, 1),
            (64, 64, 64, 4, 1),
            (32, 32, 128, 2, 1),
            (16, 16, 128, 1, 1),
        ):
            with self.subTest(tile=tile):
                self.assertEqual(is_valid_spec(_spec(tile), ARCH), (True, "ok"))

    def test_bf16_inputs_and_fp32_output_are_valid_and_named(self):
        names = set()
        for data in (
            DataSpec(),
            DataSpec("bf16", "bf16", "bf16"),
            DataSpec("fp16", "fp16", "fp32"),
            DataSpec("bf16", "bf16", "fp32"),
        ):
            with self.subTest(data=data):
                spec = _spec(data=data)
                self.assertEqual(is_valid_spec(spec, ARCH), (True, "ok"))
                names.add(build_universal_gemm(spec, arch=ARCH).name)
        self.assertEqual(len(names), 4)

    def test_rejects_shared_m_rows_other_targets_and_other_dtypes(self):
        cases = (
            (_spec((128, 64, 64, 2, 2)), ARCH, "warp_n=1"),
            (_spec(), "gfx1151", "gfx1201"),
            (_spec(data=DataSpec("fp16", "bf16", "fp16")), ARCH, "same type"),
            (_spec(data=DataSpec("fp16", "fp16", "bf16")), ARCH, "A/B type or FP32"),
            (_spec(epilogue="cshuffle"), ARCH, "default epilogue"),
        )
        for spec, arch, reason in cases:
            with self.subTest(reason=reason):
                ok, why = is_valid_spec(spec, arch)
                self.assertFalse(ok)
                self.assertIn(reason, why)

    def test_rejects_traits_it_would_silently_ignore(self):
        for trait in (
            dict(pad_m=True),
            dict(lds_store_after_mma=True),
            dict(lds_db_allow_1wg=True),
            dict(global_addressing="auto"),
            dict(split_k=2),
        ):
            with self.subTest(trait=trait):
                self.assertFalse(is_valid_spec(_spec(**trait), ARCH)[0])

    def test_rejects_tiles_that_do_not_fit_the_register_file(self):
        # The last three built and spilled in the K loop on gfx1201.
        for tile in (
            (128, 128, 64, 4, 1),
            (16, 64, 128, 1, 1),
            (32, 32, 256, 2, 1),
            (16, 16, 256, 1, 1),
        ):
            with self.subTest(tile=tile):
                ok, why = is_valid_spec(_spec(tile), ARCH)
                self.assertFalse(ok)
                self.assertIn("VGPR", why)

    def test_row_strides_are_a_direct_a_trait_with_two_more_arguments(self):
        packed = build_universal_gemm(_spec(), arch=ARCH)
        strided = build_universal_gemm(_spec(row_strides=True), arch=ARCH)
        self.assertEqual(
            [p.name for p in strided.params],
            [p.name for p in packed.params] + ["lda", "ldb"],
        )
        self.assertNotEqual(packed.name, strided.name)
        other = replace(
            _spec((128, 128, 32, 2, 2)),
            trait=TraitSpec(pipeline="compv4", row_strides=True),
        )
        ok, why = is_valid_spec(other, ARCH)
        self.assertFalse(ok)
        self.assertIn("direct_a", why)

    def test_build_refuses_what_the_gate_refuses(self):
        with self.assertRaisesRegex(ValueError, "warp_n=1"):
            build_universal_gemm(_spec((128, 64, 64, 2, 2)), arch=ARCH)


class TestDirectALaunchContract(unittest.TestCase):
    def test_partial_m_n_and_k_tiles_launch_but_partial_k_atoms_do_not(self):
        spec = _spec()
        self.assertEqual(direct_a_launch_supported(spec, 1000, 37, 128), (True, "ok"))
        self.assertEqual(direct_a_launch_supported(spec, 1024, 1024, 96), (True, "ok"))
        self.assertEqual(direct_a_launch_supported(spec, 64, 64, 16), (True, "ok"))
        self.assertFalse(direct_a_launch_supported(spec, 1024, 1024, 100)[0])
        self.assertFalse(direct_a_launch_supported(spec, 0, 64, 64)[0])

    def test_row_strides_need_the_trait_and_cover_k(self):
        packed, strided = _spec(), _spec(row_strides=True)
        self.assertFalse(direct_a_launch_supported(packed, 512, 512, 256, 288, 256)[0])
        self.assertTrue(direct_a_launch_supported(packed, 512, 512, 256, 256, 256)[0])
        self.assertTrue(direct_a_launch_supported(strided, 512, 512, 256, 288, 320)[0])
        self.assertTrue(direct_a_launch_supported(strided, 512, 512, 256)[0])
        self.assertFalse(direct_a_launch_supported(strided, 512, 512, 256, 240, 256)[0])
        self.assertFalse(direct_a_launch_supported(strided, 512, 512, 256, 256, 255)[0])

    def test_span_limit_counts_the_row_strides(self):
        strided = _spec(row_strides=True)
        k = 1 << 19
        # 2 * 1024 * lda bytes: K fits, twice K as the A stride is 2 GiB.
        self.assertTrue(direct_a_launch_supported(strided, 1024, 64, k, k, k)[0])
        self.assertFalse(direct_a_launch_supported(strided, 1024, 64, k, 2 * k, k)[0])
        self.assertFalse(direct_a_launch_supported(strided, 64, 1024, k, k, 2 * k)[0])

    def test_span_limit_counts_whole_tiles(self):
        spec = _spec()
        k = 1 << 20
        # 2 * 1024 * K bytes is exactly 2 GiB; one tile less fits.
        self.assertTrue(direct_a_launch_supported(spec, 896, 64, k)[0])
        self.assertFalse(direct_a_launch_supported(spec, 1024, 64, k)[0])
        # M = 897 rounds up to the 1024-row span even though 897 rows fit.
        self.assertFalse(direct_a_launch_supported(spec, 897, 64, k)[0])
        self.assertFalse(direct_a_launch_supported(spec, 64, 1024, k)[0])
        self.assertFalse(direct_a_launch_supported(spec, 46336, 46336, 64)[0])

    def test_fp32_output_halves_the_c_span_limit(self):
        half = _spec()
        wide = _spec(data=DataSpec("fp16", "fp16", "fp32"))
        self.assertTrue(direct_a_launch_supported(half, 30000, 30000, 64)[0])
        self.assertFalse(direct_a_launch_supported(wide, 30000, 30000, 64)[0])


class TestDirectALoop(unittest.TestCase):
    def _loop_body(self, spec):
        kernel = build_universal_gemm(spec, arch=ARCH)
        # The second loop is the one-tile remainder loop: one K tile per body.
        loop = [op for op in kernel.body.ops if op.name == "scf.for"][1]
        return list(
            _ops(
                loop.regions[0],
                {"tile.sync", "tile.smem_store_vN", "tile.mma", "tile.smem_load_vN"},
            )
        )

    def test_next_b_tile_is_stored_between_two_barriers_after_the_last_reads(self):
        for tile in ((128, 64, 64, 4, 1), (128, 96, 64, 4, 1)):
            with self.subTest(tile=tile):
                body = self._loop_body(_spec(tile, lds_swizzle=True))
                syncs = [i for i, name in enumerate(body) if name == "tile.sync"]
                stores = [
                    i for i, name in enumerate(body) if name == "tile.smem_store_vN"
                ]
                loads = [
                    i for i, name in enumerate(body) if name == "tile.smem_load_vN"
                ]
                self.assertEqual(len(syncs), 2)
                # Publish barrier, every LDS read of this tile, overwrite
                # barrier, then the stores: no read can see the next tile.
                self.assertEqual(syncs[0], 0)
                self.assertLess(max(loads), syncs[1])
                self.assertLess(syncs[1], min(stores))
                # The last K atom's matrix operations run after the stores.
                self.assertGreater(body[max(stores) :].count("tile.mma"), 0)

    def test_stagger_and_raster_change_the_kernel_and_its_name(self):
        base = _spec()
        variants = [
            base,
            replace(base, trait=replace(base.trait, group_m=4)),
            replace(base, trait=replace(base.trait, k_stagger=8, k_stagger_step=4)),
        ]
        self.assertEqual(len({s.kernel_name() for s in variants}), 3)

    def test_lds_swizzle_keys_on_the_128_byte_window(self):
        # Rows that share a 128-byte window share a key; a row that fills or
        # does not divide the window is its own key.
        for row_bytes, shift in ((128, 0), (256, 0), (64, 1), (32, 2), (96, 0)):
            with self.subTest(row_bytes=row_bytes):
                self.assertEqual(wmma_lds_swizzle_row_shift(row_bytes), shift)


if __name__ == "__main__":
    unittest.main()
