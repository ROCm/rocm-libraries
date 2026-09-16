# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Flavor-specific ABI regression test for the gfx950 scaled MFMA."""

from __future__ import annotations

import unittest

from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import IRBuilder, KernelDef
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_INTRINSIC = "@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4"
_LLVM20_22_DECL = (
    f"declare <4 x float> {_INTRINSIC}("
    "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
    "i32 immarg, i32 immarg, i32, i32 immarg, i32, i32 immarg)"
)
_LLVM20_22_CALL = (
    f"  %mxacc9 = call <4 x float> {_INTRINSIC}("
    "<8 x i32> %splat2, <8 x i32> %splat4, <4 x float> %splat6, "
    "i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 5, i32 0)"
)
_LLVM23_DECL = (
    f"declare <4 x float> {_INTRINSIC}("
    "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
    "i32 immarg, i32, i32 immarg, i32)"
)
_LLVM23_CALL = (
    f"  %mxacc9 = call <4 x float> {_INTRINSIC}("
    "<8 x i32> %splat2, <8 x i32> %splat4, <4 x float> %splat6, "
    "i32 0, i32 0, i32 0, i32 3, i32 0, i32 5)"
)
_EXPECTED_BY_FLAVOR = {
    "llvm20": (_LLVM20_22_DECL, _LLVM20_22_CALL),
    "llvm22": (_LLVM20_22_DECL, _LLVM20_22_CALL),
    "llvm23": (_LLVM23_DECL, _LLVM23_CALL),
}


def _scaled_mfma_kernel() -> KernelDef:
    builder = IRBuilder("scaled_mfma_llvm23")
    builder.kernel.attrs["max_workgroup_size"] = 64
    a = builder.vector_splat(builder.const_i32(1), 8)
    b = builder.vector_splat(builder.const_i32(2), 8)
    c = builder.vector_splat(builder.const_f32(0.0), 4)
    builder.mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, builder.const_i32(3), builder.const_i32(5)
    )
    builder.ret()
    return builder.kernel


class TestScaledMfmaLLVM23(unittest.TestCase):
    def test_flavor_specific_declaration_and_call_shape(self):
        kernel = _scaled_mfma_kernel()
        for flavor, (declaration, call) in _EXPECTED_BY_FLAVOR.items():
            with self.subTest(flavor=flavor):
                python_llvm = _lower_kernel_to_llvm_python(
                    kernel, arch="gfx950", llvm_flavor=flavor
                )
                intrinsic_lines = [
                    line for line in python_llvm.splitlines() if _INTRINSIC in line
                ]
                self.assertEqual(intrinsic_lines, [declaration, call])

    def test_python_cpp_byte_identity_for_every_flavor(self):
        kernel = _scaled_mfma_kernel()
        serialized = serialize(kernel)
        try:
            _lower_via_cpp_engine(serialized, "gfx950", "llvm20")
        except BackendError as exc:
            self.skipTest(f"C++ engine unavailable: {exc}")

        for flavor in _EXPECTED_BY_FLAVOR:
            with self.subTest(flavor=flavor):
                python_llvm = _lower_kernel_to_llvm_python(
                    kernel, arch="gfx950", llvm_flavor=flavor
                )
                cpp_llvm = _lower_via_cpp_engine(serialized, "gfx950", flavor)
                self.assertEqual(cpp_llvm, python_llvm)


if __name__ == "__main__":
    unittest.main()
