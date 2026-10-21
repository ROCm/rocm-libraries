# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tile.buffer_atomic_fadd_f32: builder validation and LLVM lowering.

The op carries the raw cache-policy immediate of
``llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32``; on gfx1250 the value
CPOL_ATOMIC_CASCADE_DEV selects ``th:TH_ATOMIC_CASCADE_RT scope:SCOPE_DEV``.
"""

import unittest

from rocke.core.ir import CPOL_ATOMIC_CASCADE_DEV, F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _kernel(cachepolicy: int):
    b = IRBuilder(f"buffer_atomic_fadd_cp{cachepolicy}")
    out = b.param("out", PtrType(F32, "global"))
    nbytes = b.param("nbytes", I32)
    rsrc = b.buffer_rsrc(out, nbytes)
    off = b.mul(b.thread_id_x(), b.const_i32(4))
    b.buffer_atomic_fadd_f32(
        rsrc, off, b.const_i32(0), b.const_f32(1.0), cachepolicy=cachepolicy
    )
    return b.kernel


class TestBufferAtomicFaddF32(unittest.TestCase):
    def test_cascade_constant(self):
        # th = TH_ATOMIC_CASCADE (4), scope = SCOPE_DEV (2) << 3.
        self.assertEqual(CPOL_ATOMIC_CASCADE_DEV, 20)

    def test_lowers_to_buffer_atomic_with_cachepolicy(self):
        for cp in (0, CPOL_ATOMIC_CASCADE_DEV):
            with self.subTest(cachepolicy=cp):
                ll = lower_kernel_to_llvm(_kernel(cp), arch="gfx1250")
                self.assertIn(
                    "declare float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(", ll
                )
                self.assertIn(
                    "call float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float ", ll
                )
                self.assertIn(f", i32 {cp})", ll)

    def test_rejects_out_of_range_cachepolicy(self):
        b = IRBuilder("bad_cp")
        out = b.param("out", PtrType(F32, "global"))
        rsrc = b.buffer_rsrc(out, b.const_i32(4))
        with self.assertRaises(ValueError):
            b.buffer_atomic_fadd_f32(
                rsrc, b.const_i32(0), b.const_i32(0), b.const_f32(1.0), cachepolicy=32
            )


if __name__ == "__main__":
    unittest.main()
