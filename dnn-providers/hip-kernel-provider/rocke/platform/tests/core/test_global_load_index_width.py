# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The scalar typed ``global_load`` GEP takes its index operand's own width.

An i64 element index (a paged KV cache past 2^31 bytes) must lower to an
``i64`` GEP index, as ``global_load_vN`` already does. An i32 index keeps the
``i32`` GEP, so every existing kernel lowers unchanged. The C++ engine twin is
``test_global_load_index_width.cpp``.

No GPU: pure text lowering.

Run:  PYTHONPATH=python python3 tests/core/test_global_load_index_width.py
"""

from __future__ import annotations

import unittest

from rocke.core.ir import BF16, I32, I64, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _kernel(idx_type, vector: bool):
    b = IRBuilder("gl_index_width")
    src = b.param("src", PtrType(BF16, "global"))
    dst = b.param("dst", PtrType(BF16, "global"))
    idx = b.param("idx", idx_type)
    if vector:
        v = b.global_load_vN(src, idx, BF16, 2, align=4)
        v = b.vec_extract(v, 0)
    else:
        v = b.global_load(src, idx, BF16, align=2)
    b.global_store(dst, b.const_i32(0), v, align=2)
    b.ret()
    return b.kernel


class TestGlobalLoadIndexWidth(unittest.TestCase):
    def _gep(self, idx_type, vector: bool) -> str:
        llvm = lower_kernel_to_llvm(_kernel(idx_type, vector), arch="gfx942")
        lines = [ln for ln in llvm.splitlines() if "ptr addrspace(1) %src" in ln]
        self.assertEqual(len(lines), 1, llvm)
        return lines[0]

    def test_scalar_i32_index_keeps_i32_gep(self):
        self.assertIn("ptr addrspace(1) %src, i32 %idx", self._gep(I32, False))

    def test_scalar_i64_index_lowers_to_i64_gep(self):
        self.assertIn("ptr addrspace(1) %src, i64 %idx", self._gep(I64, False))

    def test_scalar_matches_vector_index_width(self):
        for idx_type in (I32, I64):
            with self.subTest(idx=idx_type.name):
                scalar = self._gep(idx_type, False).split("%src, ")[1]
                vector = self._gep(idx_type, True).split("%src, ")[1]
                self.assertEqual(scalar, vector)


if __name__ == "__main__":
    unittest.main()
