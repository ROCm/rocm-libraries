# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused contracts for stable, AMDGPU-lowerable f32 ``math.tanh``."""

from __future__ import annotations

import unittest

import numpy as np

from rocke.core.ir import BF16, F16, F32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import ERROR, verify
from rocke.helpers.activations import _tanh_via_exp2
from rocke.helpers.attention import apply_softcap_scalar


# A deterministic 2,000,001-point f32 probe over [-20, 20] measured one f32
# epsilon of maximum absolute error against numpy.tanh on this formulation.
_TANH_F32_MAX_ABS_ERROR = np.finfo(np.float32).eps


def _fmuladd_f32(a, b, c):
    """Evaluate a fused f32 multiply-add with one final rounding."""
    with np.errstate(invalid="ignore"):
        return (
            np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
            + np.asarray(c, dtype=np.float64)
        ).astype(np.float32)


def _stable_tanh_f32(x):
    """Evaluate the lowering's operation sequence with f32 rounding."""
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    sign = bits & np.uint32(0x80000000)
    abs_values = (bits & np.uint32(0x7FFFFFFF)).view(np.float32)

    y2 = abs_values * abs_values
    p = _fmuladd_f32(
        y2,
        np.float32(float.fromhex("-0x1.758e7ap-8")),
        np.float32(float.fromhex("0x1.521192p-6")),
    )
    p = _fmuladd_f32(y2, p, np.float32(float.fromhex("-0x1.b8389cp-5")))
    p = _fmuladd_f32(y2, p, np.float32(float.fromhex("0x1.110704p-3")))
    p = _fmuladd_f32(y2, p, np.float32(float.fromhex("-0x1.555532p-2")))
    polynomial = _fmuladd_f32(y2, abs_values * p, abs_values)

    with np.errstate(invalid="ignore", over="ignore"):
        scaled = np.float32(2.0 * 1.4426950408889634) * abs_values
        exp = np.exp2(scaled, dtype=np.float32)
        reciprocal = np.float32(1.0) / (exp + np.float32(1.0))
        exponential = _fmuladd_f32(np.float32(-2.0), reciprocal, np.float32(1.0))
    magnitude = np.where(abs_values < np.float32(0.625), polynomial, exponential)
    return (magnitude.view(np.uint32) | sign).view(np.float32)


def _tanh_kernel():
    b = IRBuilder("tanh_f32")
    x = b.param("x", F32)
    out = b.param("out", PtrType(F32, "global"))
    b.store(b.tanh(x), out, b.const_i32(0))
    b.ret()
    return b.kernel


class TestTanhBuilderContract(unittest.TestCase):
    def test_direct_f16_and_bf16_are_rejected(self):
        for dtype in (F16, BF16):
            with self.subTest(dtype=dtype.name):
                b = IRBuilder(f"tanh_{dtype.name}")
                x = b.param("x", dtype)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^math\.tanh requires f32 operand, got {dtype.name}$",
                ):
                    b.tanh(x)

    def test_verifier_rejects_a_narrow_serialized_form(self):
        for dtype in (F16, BF16):
            with self.subTest(dtype=dtype.name):
                b = IRBuilder(f"tanh_verify_{dtype.name}")
                x = b.param("x", dtype)
                b._op("math.tanh", [x], [dtype], result_name_hint="tanh")
                errors = [d.message for d in verify(b.kernel) if d.severity == ERROR]
                self.assertIn(
                    f"math.tanh requires f32 operand, got {dtype.name}", errors
                )


class TestTanhLowering(unittest.TestCase):
    def test_f32_lowering_uses_exp2_and_bitwise_sign_restore(self):
        llvm = _lower_kernel_to_llvm_python(
            _tanh_kernel(), arch="gfx950", llvm_flavor="llvm20"
        )
        self.assertNotIn("llvm.tanh", llvm)
        self.assertIn("declare float @llvm.exp2.f32(float)", llvm)
        self.assertIn("declare float @llvm.amdgcn.rcp.f32(float)", llvm)
        self.assertIn("declare float @llvm.fmuladd.f32(float, float, float)", llvm)
        self.assertIn("fcmp olt float", llvm)
        self.assertIn("select i1", llvm)
        self.assertIn("and i32", llvm)
        self.assertIn("-2147483648", llvm)
        self.assertIn("2147483647", llvm)
        self.assertIn("or i32", llvm)

    def test_activation_helper_routes_through_math_tanh(self):
        b = IRBuilder("tanh_helper")
        x = b.param("x", F32)
        result = _tanh_via_exp2(b, x)
        self.assertEqual(result.op.name, "math.tanh")
        self.assertEqual([op.name for op in b.kernel.body.ops], ["math.tanh"])

    def test_softcap_routes_through_lowerable_tanh(self):
        b = IRBuilder("softcap")
        score = b.param("score", F32)
        softcap = b.param("softcap", F32)
        apply_softcap_scalar(b, score, softcap)
        llvm = _lower_kernel_to_llvm_python(
            b.kernel, arch="gfx950", llvm_flavor="llvm20"
        )
        self.assertIn("@llvm.exp2.f32", llvm)
        self.assertNotIn("llvm.tanh", llvm)


class TestTanhNumerics(unittest.TestCase):
    def test_special_values(self):
        values = np.array(
            [-np.inf, -100.0, -0.0, 0.0, 100.0, np.inf, np.nan],
            dtype=np.float32,
        )
        actual = _stable_tanh_f32(values)

        np.testing.assert_array_equal(actual[:2], np.array([-1.0, -1.0], np.float32))
        self.assertEqual(actual[2].view(np.uint32), np.uint32(0x80000000))
        self.assertEqual(actual[3].view(np.uint32), np.uint32(0x00000000))
        np.testing.assert_array_equal(actual[4:6], np.array([1.0, 1.0], np.float32))
        self.assertTrue(np.isnan(actual[6]))

    def test_tiny_values_are_bit_exact(self):
        positive = np.array(
            [
                np.float32(2.0**-27),
                np.float32(2.0**-26),
                np.finfo(np.float32).tiny,
                np.nextafter(np.float32(0.0), np.float32(1.0)),
            ],
            dtype=np.float32,
        )
        values = np.concatenate((positive, -positive))
        actual = _stable_tanh_f32(values)

        np.testing.assert_array_equal(actual.view(np.uint32), values.view(np.uint32))

    def test_piecewise_boundary(self):
        cutoff = np.float32(0.625)
        positive = np.array(
            [
                np.nextafter(cutoff, np.float32(0.0)),
                cutoff,
                np.nextafter(cutoff, np.float32(np.inf)),
            ],
            dtype=np.float32,
        )
        values = np.concatenate((positive, -positive))
        actual = _stable_tanh_f32(values)
        reference = np.tanh(values, dtype=np.float32)

        np.testing.assert_allclose(
            actual,
            reference,
            rtol=0.0,
            atol=_TANH_F32_MAX_ABS_ERROR,
        )

    def test_measured_f32_error_bound(self):
        values = np.linspace(-20.0, 20.0, 2_000_001, dtype=np.float32)
        actual = _stable_tanh_f32(values)
        reference = np.tanh(values, dtype=np.float32)
        max_abs_error = np.max(np.abs(actual - reference))
        self.assertLessEqual(max_abs_error, _TANH_F32_MAX_ABS_ERROR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
