# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the universal-GEMM persistent (grid-stride) tile loop.

The feature mirrors CK Tile's ``UsePersistentKernel``: the launch grid is
sized to the device instead of the problem, and each CTA walks the flattened
output-tile strip with a ``block_id += grid_size`` stride. These tests pin the
three things the rest of the stack depends on -- the validity contract, the
host grid, and the shape of the emitted loop -- without needing a GPU.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    persistent_ctas_for_device,
    universal_gemm_grid,
)

ARCH = "gfx950"
CTAS = 304


def _spec(**trait_overrides) -> UniversalGemmSpec:
    trait = {
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "epilogue": "default",
        **trait_overrides,
    }
    return UniversalGemmSpec(
        name="gemm_tile_loop_test",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        wave_size=64,
    )


class TestPersistentValidity(unittest.TestCase):
    def test_persistent_needs_a_cta_count(self):
        ok, why = is_valid_spec(_spec(persistent=True), arch=ARCH)
        self.assertFalse(ok)
        self.assertIn("persistent_ctas", why)

    def test_persistent_with_cta_count_is_valid(self):
        ok, why = is_valid_spec(
            _spec(persistent=True, persistent_ctas=CTAS), arch=ARCH
        )
        self.assertTrue(ok, why)

    def test_rejected_combinations(self):
        for label, overrides in (
            ("split_k", {"split_k": 4}),
            ("active_tile_skip", {"active_tile_skip": True}),
            ("wsp3", {"pipeline": "wsp3"}),
        ):
            with self.subTest(combo=label):
                ok, why = is_valid_spec(
                    _spec(persistent=True, persistent_ctas=CTAS, **overrides),
                    arch=ARCH,
                )
                self.assertFalse(ok, f"expected {label} to be rejected")
                self.assertIn("persistent", why)

    def test_cta_count_is_in_the_kernel_name(self):
        # Two CTA counts are two different kernels (the stride is baked in), so
        # they must not collide in the artifact cache.
        a = _spec(persistent=True, persistent_ctas=256).kernel_name()
        b = _spec(persistent=True, persistent_ctas=CTAS).kernel_name()
        self.assertNotEqual(a, b)
        self.assertTrue(b.endswith(f"_pers{CTAS}"))
        self.assertNotIn("pers", _spec().kernel_name())


class TestPersistentGrid(unittest.TestCase):
    def test_non_persistent_grid_is_tile_count(self):
        self.assertEqual(universal_gemm_grid(_spec(), 4096, 2048), (16, 32, 1))

    def test_persistent_grid_is_the_cta_count(self):
        spec = _spec(persistent=True, persistent_ctas=CTAS)
        # Independent of the problem: that is the whole point.
        self.assertEqual(universal_gemm_grid(spec, 4096, 2048), (CTAS, 1, 1))
        self.assertEqual(universal_gemm_grid(spec, 128, 128), (CTAS, 1, 1))

    def test_batched_persistent_keeps_batch_in_z(self):
        spec = replace(
            _spec(persistent=True, persistent_ctas=CTAS),
            batched=True,
        )
        self.assertEqual(universal_gemm_grid(spec, 512, 512, batch=7), (CTAS, 1, 7))

    def test_cta_count_helper_falls_back_without_a_device(self):
        self.assertEqual(
            persistent_ctas_for_device(num_cus=80, blocks_per_cu=2), 160
        )
        self.assertEqual(
            persistent_ctas_for_device(num_cus=0, default_num_cus=64), 64
        )
        with self.assertRaises(ValueError):
            persistent_ctas_for_device(num_cus=80, blocks_per_cu=0)


class TestPersistentEmission(unittest.TestCase):
    """The emitted loop, checked at the LLVM-IR level."""

    @staticmethod
    def _ll(**trait_overrides) -> str:
        spec = _spec(**trait_overrides)
        return lower_kernel_to_llvm(build_universal_gemm(spec, arch=ARCH), arch=ARCH)

    def test_abi_is_unchanged(self):
        # The grid-stride step is a codegen constant, so a persistent kernel
        # takes no extra argument and needs no host-side counter workspace.
        pattern = re.compile(r"define amdgpu_kernel void @\S+\((.*?)\) #", re.DOTALL)
        plain = pattern.search(self._ll()).group(1)
        pers = pattern.search(
            self._ll(persistent=True, persistent_ctas=CTAS)
        ).group(1)
        self.assertEqual(
            [p.split()[-1] for p in plain.split(",")],
            [p.split()[-1] for p in pers.split(",")],
        )

    def test_loop_strides_by_the_cta_count_from_blockidx(self):
        ll = self._ll(persistent=True, persistent_ctas=CTAS)
        self.assertRegex(
            ll, r"%tile_idx = phi i32 \[ %bid\d+, %entry \]", "loop starts at blockIdx"
        )
        self.assertRegex(
            ll, rf"add nsw i32 %tile_idx, {CTAS}\b", "loop strides by persistent_ctas"
        )

    def test_accumulators_are_re_zeroed_per_tile(self):
        # The K-loop's accumulator phis must take their entry value from the
        # zero vector on every trip of the outer tile loop, not carry the
        # previous tile's partial sums.
        ll = self._ll(persistent=True, persistent_ctas=CTAS)
        zero = re.search(r"(%cz\d+) = select i1 true, <\d+ x float> zeroinitializer", ll)
        self.assertIsNotNone(zero, "no zero accumulator emitted")
        accs = re.findall(r"%acc_m\d+_n\d+ = phi <\d+ x float> \[ (%cz\d+), ", ll)
        self.assertTrue(accs, "no accumulator phis found")
        self.assertEqual(set(accs), {zero.group(1)})

    def test_inter_tile_lds_barrier_is_emitted(self):
        # One extra barrier versus the non-persistent body: the loop-top guard
        # that keeps the next tile's LDS writes off the previous tile's reads.
        def barriers(ll: str) -> int:
            return ll.count("@llvm.amdgcn.s.barrier()") - ll.count(
                "declare void @llvm.amdgcn.s.barrier()"
            )

        self.assertEqual(
            barriers(self._ll(persistent=True, persistent_ctas=CTAS)),
            barriers(self._ll()) + 1,
        )

    def test_chiplet_swizzle_composes(self):
        ll = self._ll(
            persistent=True, persistent_ctas=CTAS, chiplet_swizzle=True
        )
        self.assertRegex(ll, rf"add nsw i32 %tile_idx, {CTAS}\b")
        # The XCD remap keys off the loop induction variable, so blockIdx.y --
        # which the non-persistent swizzle flattens in -- must be gone.
        self.assertNotIn("workgroup.id.y()", ll.split("declare", 1)[-1])


if __name__ == "__main__":
    unittest.main()
