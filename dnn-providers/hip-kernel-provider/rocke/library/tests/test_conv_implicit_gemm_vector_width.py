# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression gate for the fp32-width-8 vector-size admission rule.

``vector_width_reason`` (library/kernels/common/_conv_implicit_gemm_common.py)
and its C++ twin ``rocke_conv_vector_width_ok``
(platform/cpp/instances/common/conv_implicit_gemm.cpp) cap any explicit
``vector_size_*`` at one 16-byte per-lane access. An fp32 operand at width 8
is 32 bytes and must be rejected for every implicit-GEMM conv direction
(forward, wgrad, dgrad); fp16/bf16 at width 8 is exactly 16 bytes and must
stay admitted. This rule previously had no coverage, so a future edit to
either validator (or a drift between the two) would go unnoticed -- this is
why both the shared predicate and the C++ engine are exercised directly
here, not just the Python-side ``is_valid_*`` gates.

No GPU and no comgr needed for the predicate/admission tests. The
cross-engine check builds and lowers through the C++ engine binding, which
only needs the installed ``rocke_engine`` extension (skipped if absent).
"""

from __future__ import annotations

import unittest


class TestVectorWidthReason(unittest.TestCase):
    """Direct unit coverage of the predicate both engines mirror."""

    def test_fp32_width_8_rejected(self):
        from kernels.common._conv_implicit_gemm_common import vector_width_reason

        why = vector_width_reason((("c", 8, "fp32"),))
        self.assertIsNotNone(why)
        self.assertIn("vector_size_c=8", why)
        self.assertIn("32 bytes", why)

    def test_fp16_bf16_width_8_admitted(self):
        from kernels.common._conv_implicit_gemm_common import vector_width_reason

        for dtype in ("fp16", "bf16"):
            why = vector_width_reason((("a", 8, dtype),))
            self.assertIsNone(why, f"{dtype} width 8 (16 bytes) should fit")

    def test_fp32_width_4_is_the_boundary(self):
        from kernels.common._conv_implicit_gemm_common import vector_width_reason

        self.assertIsNone(vector_width_reason((("b", 4, "fp32"),)))
        self.assertIsNotNone(vector_width_reason((("b", 5, "fp32"),)))

    def test_unset_width_is_always_admitted(self):
        from kernels.common._conv_implicit_gemm_common import vector_width_reason

        self.assertIsNone(vector_width_reason((("a", None, "fp32"),)))


class _SharedGeometry:
    """A geometry every direction admits with ``vector_size_c=1``."""

    def _problem(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        return ConvProblem(
            N=4,
            Hi=14,
            Wi=14,
            C=128,
            K=128,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            pH=1,
            pW=1,
            dH=1,
            dW=1,
        )


class TestForwardAdmission(_SharedGeometry, unittest.TestCase):
    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

        base = dict(
            problem=self._problem(),
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
        )
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_fp32_vector_size_c_8_rejected(self):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, is_valid_spec

        spec = self._spec(
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
            vector_size_c=8,
        )
        ok, why = is_valid_spec(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("vector_size_c=8", why)

    def test_fp16_vector_size_c_8_admitted(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        spec = self._spec(vector_size_c=8)
        ok, why = is_valid_spec(spec, arch="gfx950")
        self.assertTrue(ok, why)


class TestWgradAdmission(_SharedGeometry, unittest.TestCase):
    def _spec(self, **kw):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        base = dict(
            problem=self._problem(),
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
        )
        base.update(kw)
        return WgradConvSpec(**base)

    def test_fp32_vector_size_c_8_rejected(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = self._spec(
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
            vector_size_c=8,
        )
        ok, why = is_valid_wgrad_spec(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("vector_size_c=8", why)

    def test_fp16_vector_size_c_8_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = self._spec(vector_size_c=8)
        ok, why = is_valid_wgrad_spec(spec, arch="gfx950")
        self.assertTrue(ok, why)


class TestDgradAdmission(_SharedGeometry, unittest.TestCase):
    def _spec(self, **kw):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

        base = dict(
            problem=self._problem(),
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
        )
        base.update(kw)
        return DgradConvSpec(**base)

    def test_fp32_vector_size_c_8_rejected(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        spec = self._spec(
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
            vector_size_c=8,
        )
        ok, why = is_valid_dgrad_spec(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("vector_size_c=8", why)

    def test_fp16_vector_size_c_8_admitted(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        spec = self._spec(vector_size_c=8)
        ok, why = is_valid_dgrad_spec(spec, arch="gfx950")
        self.assertTrue(ok, why)


class TestCppTwinAgrees(_SharedGeometry, unittest.TestCase):
    """Exercise the C++ validator directly so it cannot silently drift from
    the Python one -- the exact gap the review flagged.
    """

    def setUp(self):
        try:
            from rocke.core.backend import _import_engine

            self._eng = _import_engine()
        except Exception as e:  # noqa: BLE001 -- extension not built here
            self.skipTest(f"rocke_engine extension unavailable: {e}")

    def _dict(self, *, dtype_d: str, vector_size_c: int) -> dict:
        # Built directly (not via conv_implicit_gemm_spec_to_dict, which only
        # forwards dtype_* from top-level spec attributes -- ImplicitGemmConvSpec
        # keeps them under spec.data.* -- so it never actually carries a
        # non-default dtype across the binding) to reach the C++ validator with
        # the exact dtype/width this test cares about.
        return dict(
            problem=dict(
                N=4,
                Hi=14,
                Wi=14,
                C=128,
                K=128,
                Y=3,
                X=3,
                sH=1,
                sW=1,
                pH=1,
                pW=1,
                dH=1,
                dW=1,
            ),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_d=dtype_d,
            vector_size_c=vector_size_c,
        )

    def test_cpp_engine_rejects_fp32_width_8(self):
        d = self._dict(dtype_d="fp32", vector_size_c=8)
        with self.assertRaises(Exception):
            self._eng.conv_implicit_gemm_lower_llvm(d, arch="gfx950")

    def test_cpp_and_python_agree_on_fp16_width_8(self):
        from kernels.common.conv_implicit_gemm import (
            ConvDataSpec,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = ImplicitGemmConvSpec(
            problem=self._problem(),
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            vector_size_c=8,
        )
        py_ll = lower_kernel_to_llvm(
            build_implicit_gemm_conv(spec, arch="gfx950"), arch="gfx950"
        )
        d = self._dict(dtype_d="fp16", vector_size_c=8)
        cpp_ll = self._eng.conv_implicit_gemm_lower_llvm(d, arch="gfx950")
        self.assertEqual(py_ll, cpp_ll)


if __name__ == "__main__":
    unittest.main(verbosity=2)
