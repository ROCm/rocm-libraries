# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the dtype allowlists in ``TensorView.load_vec``.

``load_vec`` gates on ``self.dtype.name`` in two places -- once for the LDS
address space and once for global -- and raises ``NotImplementedError`` for
anything it does not recognize.  The gfx1250 fp8 GEMM path widened both
gates to admit ``fp8e4m3`` and ``bf8e5m2``; this file pins that.

What is actually asserted
-------------------------
succeeds
    Both 8-bit float types return a ``vec<...x16>`` handle from both the
    LDS and the global arm, and the kernel lowers to a 16-byte vector load
    (``<16 x i8>``, in ``addrspace(3)`` for LDS).

still rejected
    ``i8`` -- an 8-bit type deliberately *not* on either allowlist -- keeps
    raising ``NotImplementedError``.  Without this counterfactual an
    assertion that fp8 works says nothing about the gate: a ``load_vec``
    that admitted every dtype would pass the positive cases too.

Only the two ``load_vec`` gates are in scope.  The other f16/bf16 checks in
``tensor_view.py`` are scalar / elem-byte / store paths with no fp8
counterpart, and are intentionally left alone.

These tests are CPU-only, torch-free, and have no GPU dependency.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import BF8E5M2, FP8E4M3, I8, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.tensor_view import make_lds_view, make_naive_tensor_view_packed

# The gfx1250 fp8 GEMM loads A/B in 16-byte chunks, so n=16 is the width
# that actually matters; the docstring on load_vec allows {2, 4, 8, 16}.
_VEC_N = 16
_SHAPE = (64, 128)
_FP8_TYPES = (FP8E4M3, BF8E5M2)


def _builder(name: str) -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    return b


def _global_load_vec(dtype, n: int = _VEC_N):
    """Emit one global ``load_vec`` and lower; returns (handle, llvm_ir)."""
    b = _builder(f"global_{dtype.name}")
    ptr = b.param("X", PtrType(dtype, "global"), align=16)
    view = make_naive_tensor_view_packed(ptr, shape=_SHAPE, dtype=dtype)
    handle = view.load_vec(b, [b.const_i32(0), b.thread_id_x()], n)
    return handle, lower_kernel_to_llvm(b.kernel)


def _lds_load_vec(dtype, n: int = _VEC_N):
    """Emit one LDS ``load_vec`` and lower; returns (handle, llvm_ir)."""
    b = _builder(f"lds_{dtype.name}")
    view = make_lds_view(b, dtype=dtype, shape=_SHAPE)
    handle = view.load_vec(b, [b.const_i32(0), b.thread_id_x()], n)
    return handle, lower_kernel_to_llvm(b.kernel)


class TestLoadVecFp8Allowed(unittest.TestCase):
    """fp8e4m3 / bf8e5m2 pass both gates and lower to a 16-byte load."""

    def test_global_fp8_returns_vector_handle(self):
        for dtype in _FP8_TYPES:
            with self.subTest(dtype=dtype.name):
                handle, _ll = _global_load_vec(dtype)
                self.assertEqual(handle.type.count, _VEC_N)
                self.assertIs(handle.type.elem, dtype)

    def test_global_fp8_lowers_to_16xi8_load(self):
        for dtype in _FP8_TYPES:
            with self.subTest(dtype=dtype.name):
                _handle, ll = _global_load_vec(dtype)
                self.assertIn("load <16 x i8>", ll)

    def test_lds_fp8_returns_vector_handle(self):
        for dtype in _FP8_TYPES:
            with self.subTest(dtype=dtype.name):
                handle, _ll = _lds_load_vec(dtype)
                self.assertEqual(handle.type.count, _VEC_N)
                self.assertIs(handle.type.elem, dtype)

    def test_lds_fp8_lowers_to_16xi8_load_in_addrspace_3(self):
        for dtype in _FP8_TYPES:
            with self.subTest(dtype=dtype.name):
                _handle, ll = _lds_load_vec(dtype)
                self.assertIn("load <16 x i8>", ll)
                self.assertIn("addrspace(3)", ll)


class TestLoadVecUnlistedDtypeRejected(unittest.TestCase):
    """The gate is still a gate: i8 is 8-bit and still refused.

    This is the counterfactual for the class above.  ``i8`` has the same
    element width as fp8e4m3, so a widening that keyed off byte size rather
    than the dtype name would let it through.
    """

    def test_global_i8_raises(self):
        with self.assertRaises(NotImplementedError) as cm:
            _global_load_vec(I8)
        self.assertIn("i8", str(cm.exception))

    def test_lds_i8_raises(self):
        with self.assertRaises(NotImplementedError) as cm:
            _lds_load_vec(I8)
        self.assertIn("i8", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
