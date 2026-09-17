# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN decode spec rules and emission, without a GPU.

Covers the ``spec -> IR`` direction: which specs the validator admits, and that
the specs this family actually ships lower to LLVM IR -- the four tuned tiles
plus the default, the ``simple`` reference body and the ``no_l2norm`` variant.
That is a CURATED list, not the admitted space: the validator admits 54 legal
tile combinations at the reference shape, times the dtype axis, and this file
does not walk either. Lowering needs no device and no comgr, so the whole file
runs on a CPU box -- deliberately, because a family whose only tests need a GPU
contributes nothing to a CPU test lane.

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

    def test_a_wave32_target_is_rejected_for_a_wave64_spec(self):
        # gfx1151 is RDNA and runs wave32, while the lane layout and the xor
        # butterfly both take the wave width as given. Before this check the
        # validator answered (True, "") here -- it verified wave_size against
        # warp_threads_k but never against the target, so the one arch family
        # the kernel cannot serve looked buildable.
        ok, why = is_valid_spec(GdnDecodeSpec(), arch="gfx1151")
        self.assertFalse(ok)
        self.assertIn("wave size", why)

    def test_unsupported_dtype_is_rejected(self):
        ok, why = is_valid_spec(dc.replace(GdnDecodeSpec(), dtype="f32"), arch=ARCH)
        self.assertFalse(ok)

    def test_nonpositive_geometry_is_rejected_cleanly(self):
        # A zero geometry field must come back as a reason, never a
        # ZeroDivisionError from a downstream ``%`` divisibility check -- the
        # validator's contract is "if I say yes, it builds; if no, here's why".
        for field in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "num_warps",
            "warp_threads_k",
            "blocks_per_v_dim",
        ):
            ok, why = is_valid_spec(
                dc.replace(GdnDecodeSpec(), **{field: 0}), arch=ARCH
            )
            self.assertFalse(ok, field)
            self.assertIn("positive", why)


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
            "gate_kind": dc.replace(base, gate_kind="kda"),
            "lower_bound": dc.replace(base, gate_kind="kda", lower_bound=-3.0),
            "fuse_gate": dc.replace(base, gate_kind="kda", fuse_gate=False),
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


class TestGateKind(unittest.TestCase):
    """The KDA gate kind must be purely additive to the GDN one.

    GDN applies one scalar decay per head; KDA applies a per-channel DK-vector
    decay. GDN is the special case of KDA where every channel shares a value,
    so one emitter serves both -- but only if selecting the general case cannot
    disturb the special one. These tests pin that boundary.
    """

    def test_defaults_select_the_gdn_gate(self):
        spec = GdnDecodeSpec()
        self.assertEqual(spec.gate_kind, "gdn")
        self.assertEqual(spec.lower_bound, -5.0)
        self.assertTrue(spec.fuse_gate)

    def test_defaults_contribute_nothing_to_the_name(self):
        # test_default_spec_name_is_stable pins the exact string; this states
        # the reason that string must not move, so a failure reads as intent
        # rather than as an unexplained constant mismatch.
        name = GdnDecodeSpec().kernel_name()
        for token in ("kda", "lb", "nofg"):
            self.assertNotIn(token, name)

    def test_lower_bound_cannot_move_a_gdn_name(self):
        # lower_bound has no effect on the GDN path, so it must not reach the
        # cache key there -- otherwise two byte-identical kernels get two names.
        base = GdnDecodeSpec()
        self.assertEqual(
            dc.replace(base, lower_bound=-3.0).kernel_name(), base.kernel_name()
        )

    def test_lower_bound_reaches_only_a_fused_kda_name(self):
        fused = dc.replace(GdnDecodeSpec(), gate_kind="kda")
        self.assertNotIn("lb", fused.kernel_name())
        self.assertIn("lb-3", dc.replace(fused, lower_bound=-3.0).kernel_name())

        raw = dc.replace(fused, fuse_gate=False)
        self.assertEqual(
            dc.replace(raw, lower_bound=-3.0).kernel_name(),
            dc.replace(raw, lower_bound=0.0).kernel_name(),
        )

    def test_rejects_unknown_gate_kind(self):
        ok, msg = is_valid_spec(dc.replace(GdnDecodeSpec(), gate_kind="mamba"), ARCH)
        self.assertFalse(ok)
        self.assertIn("gate_kind", msg)

    def test_rejects_precomputed_gate_for_gdn(self):
        spec = dc.replace(GdnDecodeSpec(), gate_kind="gdn", fuse_gate=False)
        ok, msg = is_valid_spec(spec, ARCH)
        self.assertFalse(ok)
        self.assertIn("requires fuse_gate=True", msg)

    def test_fused_kda_requires_finite_negative_lower_bound(self):
        for bound in (0.0, 1.0, float("nan"), float("inf"), float("-inf")):
            spec = dc.replace(
                GdnDecodeSpec(),
                gate_kind="kda",
                fuse_gate=True,
                lower_bound=bound,
            )
            ok, msg = is_valid_spec(spec, ARCH)
            self.assertFalse(ok, f"lower_bound={bound} must be rejected")
            self.assertIn("finite negative", msg)

    def test_raw_kda_ignores_lower_bound(self):
        for bound in (0.0, 1.0, float("nan"), float("inf")):
            spec = dc.replace(
                GdnDecodeSpec(),
                gate_kind="kda",
                fuse_gate=False,
                lower_bound=bound,
            )
            ok, why = is_valid_spec(spec, ARCH)
            self.assertTrue(ok, why)

    def test_gdn_ignores_lower_bound(self):
        ok, why = is_valid_spec(
            dc.replace(GdnDecodeSpec(), lower_bound=float("nan")), ARCH
        )
        self.assertTrue(ok, why)


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
        # Both tables, each with its own gate kind: a tile is only ever selected
        # together with the gate it was tuned for, so that is how it is checked.
        from dispatch.gdn.gfx950 import _TUNED_TILES_GDN, _TUNED_TILES_KDA

        for gate_kind, table in (
            ("gdn", _TUNED_TILES_GDN),
            ("kda", _TUNED_TILES_KDA),
        ):
            for _, tile, spec_id in table:
                with self.subTest(spec_id=spec_id, gate_kind=gate_kind):
                    spec = dc.replace(
                        GdnDecodeSpec(),
                        gate_kind=gate_kind,
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
