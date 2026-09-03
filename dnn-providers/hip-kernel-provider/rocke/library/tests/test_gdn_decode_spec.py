# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN decode spec rules and emission, without a GPU.

Covers the ``spec -> IR`` direction: which specs the validator admits, and that
every admitted spec actually lowers to LLVM IR. Lowering needs no device and no
comgr, so this whole file runs on a CPU box -- deliberately, because a family
whose only tests need a GPU contributes nothing to a CPU test lane.

The numeric behaviour of the emitted kernel is covered separately by the
on-device test; this file only asserts that the rules hold and that emission
does not fall over.
"""

from __future__ import annotations

import dataclasses as dc
import unittest

from kernels.gfx950.gdn_decode import (
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)

ARCH = "gfx950"


def _lower(spec: GdnDecodeSpec, arch: str = ARCH, flavor: str = "llvm20") -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        build_gdn_decode(spec, arch=arch), arch=arch, llvm_flavor=flavor
    )


class TestSpecAdmission(unittest.TestCase):
    def test_default_spec_is_valid(self):
        ok, why = is_valid_spec(GdnDecodeSpec(), arch=ARCH)
        self.assertTrue(ok, why)

    def test_simple_reference_path_is_valid(self):
        ok, why = is_valid_spec(GdnDecodeSpec(simple=True), arch=ARCH)
        self.assertTrue(ok, why)

    def test_value_heads_must_be_a_multiple_of_key_heads(self):
        ok, why = is_valid_spec(dc.replace(GdnDecodeSpec(), num_v_heads=33), arch=ARCH)
        self.assertFalse(ok)
        self.assertIn("divisible", why)

    def test_head_dims_must_suit_the_vector_width(self):
        # 16-byte vector loads mean head dims must be multiples of 8; a dim that
        # is not would silently drop the tail of every row.
        ok, why = is_valid_spec(dc.replace(GdnDecodeSpec(), head_k_dim=127), arch=ARCH)
        self.assertFalse(ok)
        self.assertIn("multiples", why)

    def test_block_size_over_the_hardware_limit_is_rejected(self):
        ok, why = is_valid_spec(dc.replace(GdnDecodeSpec(), num_warps=64), arch=ARCH)
        self.assertFalse(ok)
        self.assertIn("max_threads_per_block", why)

    def test_wave_must_divide_by_k_lanes(self):
        ok, why = is_valid_spec(
            dc.replace(GdnDecodeSpec(), warp_threads_k=24), arch=ARCH
        )
        self.assertFalse(ok)

    def test_v_split_must_divide_the_head(self):
        ok, why = is_valid_spec(
            dc.replace(GdnDecodeSpec(), blocks_per_v_dim=7), arch=ARCH
        )
        self.assertFalse(ok)

    def test_unknown_arch_is_rejected_not_crashed(self):
        ok, why = is_valid_spec(GdnDecodeSpec(), arch="gfx000")
        self.assertFalse(ok)
        self.assertTrue(why)

    def test_unsupported_dtype_is_rejected(self):
        ok, why = is_valid_spec(dc.replace(GdnDecodeSpec(), dtype="f32"), arch=ARCH)
        self.assertFalse(ok)


class TestBuilderRejectsInvalidSpecs(unittest.TestCase):
    def test_build_refuses_an_invalid_spec(self):
        with self.assertRaises(ValueError):
            build_gdn_decode(dc.replace(GdnDecodeSpec(), num_v_heads=33), arch=ARCH)

    def test_build_validates_against_the_requested_arch(self):
        # The arch must reach the validator, or a build for one target would be
        # checked against another's limits.
        with self.assertRaises(ValueError) as ctx:
            build_gdn_decode(GdnDecodeSpec(), arch="gfx000")
        self.assertIn("gfx000", str(ctx.exception))


class TestKernelNameIdentity(unittest.TestCase):
    """The name is the compile/launcher cache key, so it must be injective."""

    def test_every_codegen_field_reaches_the_name(self):
        base = GdnDecodeSpec()
        variants = {
            "base": base,
            "state_dtype": dc.replace(base, state_dtype="f16"),
            "dtype": dc.replace(base, dtype="f16"),
            "wave_size": dc.replace(base, wave_size=32),
            "num_warps": dc.replace(base, num_warps=4),
            "warp_threads_k": dc.replace(base, warp_threads_k=8),
            "blocks_per_v_dim": dc.replace(base, blocks_per_v_dim=4),
            "use_qk_l2norm": dc.replace(base, use_qk_l2norm=False),
            "simple": dc.replace(base, simple=True),
            "num_k_heads": dc.replace(base, num_k_heads=8),
            "head_k_dim": dc.replace(base, head_k_dim=64),
        }
        names = {}
        for label, spec in variants.items():
            name = spec.kernel_name()
            self.assertNotIn(
                name,
                names,
                f"{label!r} collides with {names.get(name)!r} on {name!r}; "
                "two different kernels would share one cache entry",
            )
            names[name] = label

    def test_default_spec_name_is_stable(self):
        # Pinned so a rename is a deliberate, visible change rather than a
        # silent cache miss for every existing caller.
        self.assertEqual(
            GdnDecodeSpec().kernel_name(),
            "rocke_gdn_decode_bf16_kh16_vh32_dk128_dv128_w2k16b8_l2",
        )


class TestLaunchShape(unittest.TestCase):
    def test_grid_scales_with_batch_heads_and_v_split(self):
        spec = GdnDecodeSpec()
        for batch in (1, 7, 64):
            expected = batch * spec.num_v_heads * spec.blocks_per_v_dim
            self.assertEqual(gdn_decode_grid(batch, spec)[0], expected)

    def test_simple_path_does_not_split_the_v_dimension(self):
        spec = GdnDecodeSpec(simple=True)
        self.assertEqual(gdn_decode_grid(4, spec)[0], 4 * spec.num_v_heads)

    def test_signature_matches_the_kernel_arguments(self):
        sig = gdn_decode_signature(GdnDecodeSpec())
        names = [a["name"] for a in sig]
        self.assertEqual(
            names,
            [
                "query",
                "key",
                "value",
                "a",
                "b",
                "dt_bias",
                "A_log",
                "read_indices",
                "write_indices",
                "state",
                "out",
                "batch_size",
            ],
        )
        self.assertEqual(names[-1], "batch_size")
        self.assertEqual(sig[-1]["type"], "i32")


class TestEmission(unittest.TestCase):
    """Every admitted spec must actually lower. No GPU, no comgr."""

    def test_default_spec_emits_a_kernel(self):
        llvm = _lower(GdnDecodeSpec())
        self.assertIn("define amdgpu_kernel", llvm)
        self.assertIn(GdnDecodeSpec().kernel_name(), llvm)

    def test_both_builder_paths_emit(self):
        for simple in (False, True):
            with self.subTest(simple=simple):
                llvm = _lower(GdnDecodeSpec(simple=simple))
                self.assertIn("define amdgpu_kernel", llvm)

    def test_every_tuned_tile_emits(self):
        from dispatch.gdn.gfx950 import _TUNED_TILES

        for _, tile, spec_id in _TUNED_TILES:
            with self.subTest(spec_id=spec_id):
                spec = dc.replace(
                    GdnDecodeSpec(),
                    num_warps=tile[0],
                    warp_threads_k=tile[1],
                    blocks_per_v_dim=tile[2],
                )
                self.assertIn("define amdgpu_kernel", _lower(spec))

    def test_distinct_tiles_emit_distinct_code(self):
        # If two tiles produced identical IR the tuning table would be choosing
        # between kernels that are actually the same.
        a = _lower(
            dc.replace(
                GdnDecodeSpec(), num_warps=1, warp_threads_k=8, blocks_per_v_dim=1
            )
        )
        b = _lower(
            dc.replace(
                GdnDecodeSpec(), num_warps=8, warp_threads_k=16, blocks_per_v_dim=1
            )
        )
        self.assertNotEqual(a, b)

    def test_emission_is_deterministic(self):
        self.assertEqual(_lower(GdnDecodeSpec()), _lower(GdnDecodeSpec()))


if __name__ == "__main__":
    unittest.main()
