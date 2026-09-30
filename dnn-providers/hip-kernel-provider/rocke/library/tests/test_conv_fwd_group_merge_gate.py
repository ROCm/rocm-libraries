# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for forward group merging (``group_merge``).

No GPU and no comgr: everything here is spec construction, the validity
predicate, and the lowered IR. The numerical counterpart is
``TestConvFwdGroupMergeNumerics`` in ``test_conv_fwd_correctness.py``, which
needs a CDNA device -- merging computes ``Gm x`` redundant MACs and relies on a
diagonal mask to zero them, and no host-side check can prove that cancels.

The gate has to agree between :meth:`ImplicitGemmConvSpec.validate` and
:func:`is_valid_spec`. Both delegate to the single
:func:`fwd_group_merge_available`, because two independently maintained copies
of a gate is exactly how the wgrad family came to have a dispatcher that
admitted specs the builder then rejected.
"""

from __future__ import annotations

import re
import unittest


def _count_vector_buffer_loads(ll: str) -> int:
    """Number of *vector-typed* raw buffer loads in the lowered IR.

    A 128-bit ``buffer_load_dwordx4`` lowers to ``...buffer.load.v4i32`` (= 8
    bf16); scalar loads lower to ``...buffer.load.f16`` / ``i16``. Counting the
    vector variants tells us the merged-axis vectorised load fired.
    """
    return len(re.findall(r"amdgcn\.raw\.(?:ptr\.)?buffer\.load\.v\d+\w+", ll))


def _count_vector_buffer_stores(ll: str) -> int:
    """Same, for stores.

    Worth asserting separately from loads: this arrangement puts the diagonal
    mask on the *B load* rather than the output, so all ``Gm`` N-lanes hold real
    outputs at consecutive channels and the store widens too. CK Tile, which
    merges onto GemmM and GemmN, has to gather a diagonal out of its C tile and
    consequently ships ``VectorSizeC = 1`` at ``Gm = 32``. If a future change
    reintroduces a store-side mask, loads would still widen and only this
    assertion would notice.
    """
    return len(re.findall(r"amdgcn\.raw\.(?:ptr\.)?buffer\.store\.v\d+\w+", ll))


class TestConvFwdGroupMergeGate(unittest.TestCase):
    """Host-side gate for ``group_merge`` (no GPU needed)."""

    def _spec(self, **kw):
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

        base = dict(
            problem=ConvProblem(
                N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64
            ),
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            wave_size=64,
            pipeline="mem",
            epilogue="cshuffle",
        )
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_default_is_one_and_always_admitted(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        spec = self._spec()
        self.assertEqual(spec.group_merge, 1)
        ok, why = is_valid_spec(spec, arch="gfx950")
        self.assertTrue(ok, why)

    def test_merged_dims_track_group_merge(self):
        # grid_* is what the tile covers; ``problem`` stays the true extent that
        # sizes the tensors. Conflating them is the silent wrong-answer bug this
        # whole path is prone to -- the true N_gemm is 1 on depthwise, so a
        # bound taken from it discards Gm-1 of every Gm real outputs.
        spec = self._spec(group_merge=8)
        p = spec.problem
        self.assertEqual(spec.grid_M, p.M)  # M does not merge
        self.assertEqual(spec.grid_N_gemm, 8)  # == Gm
        self.assertEqual(spec.grid_K_gemm, p.Y * p.X * 8)
        self.assertEqual(spec.grid_groups, 8)  # 64 / 8

        base = self._spec()
        self.assertEqual(base.grid_N_gemm, base.problem.N_gemm)
        self.assertEqual(base.grid_K_gemm, base.problem.K_gemm)
        self.assertEqual(base.grid_groups, base.problem.groups)

    def test_grid_launches_over_merged_dims(self):
        # Host/device divergence here is the highest-severity failure mode:
        # launching ``groups`` CTAs in z with gn=1 under merge means Gm x
        # redundant CTAs racing on the same output.
        from kernels.common.conv_implicit_gemm import implicit_gemm_conv_grid

        spec = self._spec(group_merge=8)
        gx, gy, gz = implicit_gemm_conv_grid(spec)
        self.assertEqual(gz, 8)
        self.assertEqual(gx, 1)  # N_gemm=8 fits one tile_n=64 column
        self.assertEqual(gy, (spec.grid_M + spec.tile_m - 1) // spec.tile_m)

    def test_validate_and_predicate_agree(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        cases = [
            dict(group_merge=3),  # not a power-of-two degree
            dict(group_merge=128),  # not a supported degree
            dict(group_merge=8, tile_n=4),  # Gm > tile_n
            dict(group_merge=8, wave_size=32),  # MFMA-only
            dict(group_merge=8, async_dma=True),  # chunk-granular predicate
            dict(group_merge=8, vector_size_b=8),  # B must stay scalar
        ]
        for kw in cases:
            spec = self._spec(**kw)
            ok, why = is_valid_spec(spec, arch="gfx950")
            self.assertFalse(ok, f"{kw} should be rejected")
            with self.assertRaises(ValueError, msg=f"{kw} must raise"):
                spec.validate()

    def test_non_depthwise_is_gated_off(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import is_valid_spec

        p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=8)
        ok, why = is_valid_spec(self._spec(problem=p, group_merge=4), arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("depthwise", why)

    def test_group_merge_must_divide_groups(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import is_valid_spec

        # G=144 is a real corpus shape: divisible by 16 but not by 32.
        p = ConvProblem(
            N=1, Hi=56, Wi=56, C=144, K=144, Y=3, X=3, pH=1, pW=1, groups=144
        )
        ok, _ = is_valid_spec(self._spec(problem=p, group_merge=16), arch="gfx950")
        self.assertTrue(ok)
        ok, why = is_valid_spec(self._spec(problem=p, group_merge=32), arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("divide", why)

    def test_pointwise_is_gated_off(self):
        # Y == X == 1 is where merging is *most* attractive, so this is a
        # deliberate deferral rather than a corner case: the flat pointwise
        # path builds ``valid`` from scratch and has no slot for the diagonal.
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import is_valid_spec

        p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=1, X=1, groups=64)
        ok, why = is_valid_spec(self._spec(problem=p, group_merge=8), arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("pointwise", why)

    def test_kernel_names_are_tagged_per_degree(self):
        # The compile cache keys on kernel.name. Untagged, a Gm sweep would
        # measure one binary N times.
        names = {self._spec(group_merge=g).kernel_name() for g in (1, 2, 4, 8, 16)}
        self.assertEqual(len(names), 5, f"kernel names collide: {names}")
        self.assertNotIn("gm1", self._spec().kernel_name())

    def test_merged_build_widens_the_load_and_the_store(self):
        # The whole point of merging: on depthwise the free axis is one element
        # wide, so every access is scalar. Merging makes it a run of Gm.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv

        scalar = lower_kernel_to_llvm(
            build_implicit_gemm_conv(self._spec(), arch="gfx950")
        )
        merged = lower_kernel_to_llvm(
            build_implicit_gemm_conv(self._spec(group_merge=8), arch="gfx950")
        )
        self.assertEqual(
            _count_vector_buffer_loads(scalar),
            0,
            "depthwise group_merge=1 should have no vector loads to begin with",
        )
        self.assertGreater(
            _count_vector_buffer_loads(merged),
            0,
            "group_merge=8 must vectorise the activation loads",
        )
        self.assertEqual(_count_vector_buffer_stores(scalar), 0)
        self.assertGreater(
            _count_vector_buffer_stores(merged),
            0,
            "the diagonal mask lives on the B load, so the store must be dense",
        )

    def test_merged_b_load_stays_scalar(self):
        # B is only 1/Gm dense and the mask is per-element; a vector load takes
        # one predicate for the whole run and cannot express it. The count of
        # scalar loads must therefore not drop when merging.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv

        def scalar_loads(ll):
            return len(
                re.findall(r"amdgcn\.raw\.(?:ptr\.)?buffer\.load\.[if]\d+\b", ll)
            )

        merged = lower_kernel_to_llvm(
            build_implicit_gemm_conv(self._spec(group_merge=8), arch="gfx950")
        )
        self.assertGreater(
            scalar_loads(merged), 0, "B must still issue scalar loads under merge"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
