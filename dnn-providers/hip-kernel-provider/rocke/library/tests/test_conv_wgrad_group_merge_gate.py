# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for wgrad group merging (``group_merge``).

No GPU and no comgr: everything here is spec construction, the validity
predicate, and the lowered IR. The numerical counterpart lives in
``test_conv_wgrad_correctness.py``, which needs a CDNA device.

The gate has to agree between :meth:`WgradConvSpec.validate` and
:func:`is_valid_wgrad_spec` -- a spec the dispatcher admits and the builder
then rejects is a failure mode this family has produced before.
"""

from __future__ import annotations

import re
import unittest


def _count_vector_buffer_loads(ll: str) -> int:
    """Number of *vector-typed* raw buffer loads in the lowered IR.

    A 128-bit ``buffer_load_dwordx4`` lowers to ``...buffer.load.v4i32`` (= 8
    fp16); scalar loads lower to ``...buffer.load.f16`` / ``i16``. Counting the
    vector variants tells us the free-axis vectorised load fired.
    """
    return len(re.findall(r"amdgcn\.raw\.(?:ptr\.)?buffer\.load\.v\d+\w+", ll))


class TestWgradGroupMergeGate(unittest.TestCase):
    """Host-side gate for ``group_merge`` (no GPU needed).

    The gate has to agree between :meth:`WgradConvSpec.validate` and
    :func:`is_valid_wgrad_spec`: a spec the dispatcher admits and the builder
    then rejects is the exact failure this family has produced before.
    """

    def _spec(self, **kw):
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        base = dict(
            problem=ConvProblem(
                N=2, Hi=12, Wi=12, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64
            ),
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            tile_m=16,
            tile_n=128,
            tile_k=32,
            warp_m=1,
            warp_n=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=64,
            pipeline="mem",
            epilogue="cshuffle",
            split_k=1,
            two_stage=False,
            lds_k_outer=True,
        )
        base.update(kw)
        return WgradConvSpec(**base)

    def test_default_is_one_and_always_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = self._spec()
        self.assertEqual(spec.group_merge, 1)
        ok, why = is_valid_wgrad_spec(spec, arch="gfx950")
        self.assertTrue(ok, why)

    def test_merged_dims_track_group_merge(self):
        # grid_* is what the tile covers; wg_* stays the true per-group extent
        # that sizes dW and the workspace. Conflating them is the silent
        # wrong-answer bug this whole path is prone to.
        spec = self._spec(group_merge=8)
        self.assertEqual((spec.wg_M, spec.wg_N), (1, 9))
        self.assertEqual((spec.grid_M, spec.grid_N), (8, 72))
        self.assertEqual(spec.grid_groups, 8)
        base = self._spec()
        self.assertEqual((base.grid_M, base.grid_N), (base.wg_M, base.wg_N))
        self.assertEqual(base.grid_groups, base.problem.groups)

    def test_validate_and_predicate_agree(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        cases = [
            dict(group_merge=3),  # not a supported degree
            dict(group_merge=128),  # not a supported degree
            dict(group_merge=8, tile_n=32, split_k=1, two_stage=False),
            # Atomic split-K has no diagonal mask; two-stage does.
            dict(group_merge=8, two_stage=False, split_k=4),
            dict(group_merge=8, wave_size=32, split_k=1, two_stage=False),
        ]
        for kw in cases:
            spec = self._spec(**kw)
            ok, why = is_valid_wgrad_spec(spec, arch="gfx950")
            self.assertFalse(ok, f"{kw} should be rejected")
            with self.assertRaises(ValueError, msg=f"{kw} must raise"):
                spec.validate()

    def test_non_depthwise_is_gated_off(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        p = ConvProblem(N=2, Hi=12, Wi=12, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=8)
        ok, why = is_valid_wgrad_spec(
            self._spec(problem=p, group_merge=4), arch="gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("depthwise", why)

    def test_group_merge_must_divide_groups(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        from kernels.common._conv_implicit_gemm_common import ConvProblem

        p = ConvProblem(N=2, Hi=12, Wi=12, C=24, K=24, Y=3, X=3, pH=1, pW=1, groups=24)
        ok, why = is_valid_wgrad_spec(
            self._spec(problem=p, group_merge=16), arch="gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("divide", why)

    def test_kernel_name_distinguishes_degrees(self):
        # The compile cache keys on kernel.name. Untagged, a Gm sweep would
        # measure one binary N times.
        names = {self._spec(group_merge=g).kernel_name() for g in (1, 2, 4, 8)}
        self.assertEqual(len(names), 4, f"kernel names collide: {names}")
        self.assertNotIn("gm1", self._spec().kernel_name())

    def test_merged_build_widens_the_load(self):
        # The whole point of merging: the depthwise free axis is one element
        # wide, so every load is scalar. Merging makes it a run of Gm.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
        )

        scalar = lower_kernel_to_llvm(
            build_implicit_gemm_conv_wgrad(self._spec(), arch="gfx950")
        )
        merged = lower_kernel_to_llvm(
            build_implicit_gemm_conv_wgrad(self._spec(group_merge=8), arch="gfx950")
        )
        self.assertEqual(
            _count_vector_buffer_loads(scalar),
            0,
            "depthwise group_merge=1 should have no vector loads to begin with",
        )
        self.assertGreater(
            _count_vector_buffer_loads(merged),
            0,
            "group_merge=8 must vectorise the dY/X loads",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
