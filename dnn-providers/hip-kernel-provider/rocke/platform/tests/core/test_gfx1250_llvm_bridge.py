# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused validation for the standalone gfx1250 LLVM 23 bridge."""

from __future__ import annotations

import unittest

from rocke.core.ir import F32, I16, I32, I64, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _build_all():
    b = IRBuilder("gfx1250_bridge")
    src = b.param("src", PtrType(I16, "global"), readonly=True, align=16)
    dst = b.param("dst", PtrType(I16, "global"), align=16)
    barrier = b.smem_alloc(I64, [1], name_hint="barrier")
    local = b.smem_addr_of(barrier)
    members = b.const_i32(2)

    b.s_wait_tensorcnt(3)
    b.s_barrier_signal(1)
    b.s_barrier_wait(2)
    b.s_barrier_init(local, members)
    b.s_barrier_signal_var(local, members)
    b.s_barrier_join(local)
    b.s_wakeup_barrier(local)
    b.s_barrier_leave(4)

    b.s_delay_alu(0x1234)
    b.s_wait_alu(0x2345)
    b.s_clause(0x3456)
    b.s_wait_xcnt(0x4567)

    b.global_store_async_from_lds(
        dst, local, width_bytes=16, offset_bytes=-4, cachepolicy=31
    )
    b.global_load_tr16_b128(src, dtype=I16)

    d4 = b.zero_vec(I32, 4)
    d8 = b.zero_vec(I32, 8)
    b.tensor_load_to_lds(d4, d8, d4, d4, d8, cachepolicy=5)
    b.tensor_store_from_lds(d4, d8, d4, d4, d8, cachepolicy=6)
    b.ret()
    return b.kernel


def _lower(kernel, *, arch="gfx1250", flavor="llvm23"):
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


class TestGfx1250LlvmBridge(unittest.TestCase):
    def test_exact_control_intrinsics_and_inline_asm(self):
        llvm = _lower(_build_all())
        expected = (
            "call void @llvm.amdgcn.s.wait.tensorcnt(i16 3)",
            "call void @llvm.amdgcn.s.barrier.signal(i32 1)",
            "call void @llvm.amdgcn.s.barrier.wait(i16 2)",
            "call void @llvm.amdgcn.s.barrier.init(ptr addrspace(3)",
            "call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3)",
            "call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3)",
            "call void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3)",
            "call void @llvm.amdgcn.s.barrier.leave(i16 4)",
            'call void asm sideeffect "s_delay_alu 4660", ""()',
            'call void asm sideeffect "s_wait_alu 9029", ""()',
            'call void asm sideeffect "s_clause 13398", ""()',
            'call void asm sideeffect "s_wait_xcnt 17767", ""()',
        )
        for text in expected:
            with self.subTest(text=text):
                self.assertIn(text, llvm)
        self.assertIn(
            "declare void @llvm.amdgcn.s.barrier.signal.var("
            "ptr addrspace(3) nocapture, i32)",
            llvm,
        )

    def test_exact_memory_intrinsics(self):
        llvm = _lower(_build_all())
        self.assertIn(
            "call void @llvm.amdgcn.global.store.async.from.lds.b128("
            "ptr addrspace(1) %dst, ptr addrspace(3)",
            llvm,
        )
        self.assertIn(", i32 -4, i32 31)", llvm)
        self.assertIn(
            "call <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16("
            "ptr addrspace(1) %src)",
            llvm,
        )
        self.assertIn(
            "call void @llvm.amdgcn.tensor.load.to.lds(" "<4 x i32>",
            llvm,
        )
        self.assertIn("<8 x i32>", llvm)
        self.assertIn(", i32 5)", llvm)
        self.assertIn(
            "declare void @llvm.amdgcn.tensor.store.from.lds("
            "<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg)",
            llvm,
        )

    def test_serialization_roundtrip_preserves_all_ops(self):
        kernel = _build_all()
        text = serialize(kernel)
        parsed = parse(text)
        self.assertEqual(text, serialize(parsed))
        self.assertEqual(_lower(kernel), _lower(parsed))

    def test_unsupported_arch_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires gfx1250"):
            _lower(_build_all(), arch="gfx1201")

    def test_pre_llvm23_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires LLVM flavor llvm23"):
            _lower(_build_all(), flavor="llvm22")

    def test_immediate_ranges(self):
        for method in ("s_wait_tensorcnt", "s_barrier_wait", "s_barrier_leave"):
            with self.subTest(method=method):
                b = IRBuilder("bad")
                with self.assertRaises(ValueError):
                    getattr(b, method)(65536)
        for method in ("s_delay_alu", "s_wait_alu", "s_clause", "s_wait_xcnt"):
            with self.subTest(method=method):
                b = IRBuilder("bad")
                with self.assertRaises(ValueError):
                    getattr(b, method)(-1)

        b = IRBuilder("bad")
        dst = b.param("dst", PtrType(I16, "global"))
        smem = b.smem_alloc(I64, [1])
        local = b.smem_addr_of(smem)
        with self.assertRaises(ValueError):
            b.global_store_async_from_lds(dst, local, width_bytes=2, cachepolicy=0)
        with self.assertRaises(ValueError):
            b.global_store_async_from_lds(dst, local, width_bytes=4, cachepolicy=32)

    def test_operand_types(self):
        b = IRBuilder("bad_types")
        bad_ptr = b.param("bad", PtrType(F32, "global"))
        local = b.const_i64(0)
        with self.assertRaises(TypeError):
            b.s_barrier_init(local, b.const_i64(2))
        with self.assertRaises(TypeError):
            b.global_store_async_from_lds(b.const_i64(0), local, width_bytes=4)
        with self.assertRaises(TypeError):
            b.global_load_tr16_b128(bad_ptr, dtype=F32)
        d4 = b.zero_vec(I32, 4)
        with self.assertRaises(TypeError):
            b.tensor_load_to_lds(d4, d4, d4, d4, d4)


if __name__ == "__main__":
    unittest.main()
