# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text validation for the gfx1250 data-prefetch ops (Phase 1).

The ISA half of the proof lives in
``rocke/examples/gfx1250/isa_features/data_prefetch_verify.py``, which needs an
llvm23 toolchain; these tests pin the LLVM the lowering hands it.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import (
    F32,
    HW_REG_MODE,
    I32,
    I64,
    MODE_SCALAR_PREFETCH_EN_BIT,
    IRBuilder,
    PtrType,
    hwreg,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _build_all():
    b = IRBuilder("gfx1250_prefetch")
    src = b.param("src", PtrType(F32, "global"), readonly=True, align=16)
    table = b.param("table", PtrType(I32, "global"), addr_space="constant")
    flat = b.param("flat", PtrType(F32, "private"))
    nbytes = b.param("nbytes", I32)
    lines = b.const_i32(4)

    b.enable_scalar_prefetch()
    b.s_prefetch_data(src, lines)
    b.s_prefetch_data(table, lines)
    b.s_prefetch_data(flat, lines)
    rsrc = b.buffer_rsrc(src, nbytes)
    b.s_buffer_prefetch_data(rsrc, lines, offset=256)
    b.global_prefetch(src, cachepolicy=3)
    b.flat_prefetch(flat, cachepolicy=5)
    b.ret()
    return b.kernel


def _lower(kernel, *, arch="gfx1250", flavor="llvm23"):
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


class TestHwreg(unittest.TestCase):
    def test_packing(self):
        self.assertEqual(hwreg(1, 0, 32), 1 | (31 << 11))
        self.assertEqual(hwreg(HW_REG_MODE, MODE_SCALAR_PREFETCH_EN_BIT, 1), 1537)
        self.assertEqual(hwreg(63, 31, 1), 63 | (31 << 6))

    def test_ranges(self):
        for args in ((64, 0, 1), (-1, 0, 1), (1, 32, 1), (1, 0, 0), (1, 24, 9)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    hwreg(*args)


class TestGfx1250Prefetch(unittest.TestCase):
    def test_exact_calls(self):
        llvm = _lower(_build_all())
        expected = (
            "call void @llvm.amdgcn.s.setreg(i32 1537, i32 1)",
            "call void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1) %src, i32 4)",
            "call void @llvm.amdgcn.s.prefetch.data.p4(ptr addrspace(4) %table, i32 4)",
            "call void @llvm.amdgcn.s.prefetch.data.p0(ptr %flat, i32 4)",
            "call void @llvm.amdgcn.s.buffer.prefetch.data(ptr addrspace(8) %rsrc",
            "call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %src, i32 3)",
            "call void @llvm.amdgcn.flat.prefetch(ptr %flat, i32 5)",
        )
        for text in expected:
            with self.subTest(text=text):
                self.assertIn(text, llvm)
        self.assertRegex(
            llvm,
            r"call void @llvm\.amdgcn\.s\.buffer\.prefetch\.data\("
            r"ptr addrspace\(8\) %\S+, i32 256, i32 4\)",
        )

    def test_exact_declares(self):
        llvm = _lower(_build_all())
        expected = (
            "declare void @llvm.amdgcn.s.setreg(i32 immarg, i32)",
            "declare void @llvm.amdgcn.s.prefetch.data.p0(ptr, i32)",
            "declare void @llvm.amdgcn.s.prefetch.data.p1(ptr addrspace(1), i32)",
            "declare void @llvm.amdgcn.s.prefetch.data.p4(ptr addrspace(4), i32)",
            "declare void @llvm.amdgcn.s.buffer.prefetch.data("
            "ptr addrspace(8), i32 immarg, i32)",
            "declare void @llvm.amdgcn.global.prefetch(ptr addrspace(1), i32 immarg)",
            "declare void @llvm.amdgcn.flat.prefetch(ptr, i32 immarg)",
        )
        for text in expected:
            with self.subTest(text=text):
                self.assertEqual(llvm.count(text), 1)

    def test_unused_ops_declare_nothing(self):
        b = IRBuilder("only_global")
        src = b.param("src", PtrType(F32, "global"))
        b.global_prefetch(src)
        b.ret()
        llvm = _lower(b.kernel)
        self.assertIn("call void @llvm.amdgcn.global.prefetch(", llvm)
        for name in ("s.setreg", "s.prefetch.data", "s.buffer.prefetch", "flat.prefetch"):
            with self.subTest(name=name):
                self.assertNotIn(f"@llvm.amdgcn.{name}", llvm)

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

    def test_each_op_is_gated(self):
        def one(emit):
            b = IRBuilder("gate")
            src = b.param("src", PtrType(F32, "global"))
            flat = b.param("flat", PtrType(F32, "private"))
            emit(b, src, flat)
            b.ret()
            return b.kernel

        cases = {
            "s_setreg": lambda b, s, f: b.s_setreg(1537, b.const_i32(1)),
            "s_prefetch_data": lambda b, s, f: b.s_prefetch_data(s, b.const_i32(1)),
            "s_buffer_prefetch_data": lambda b, s, f: b.s_buffer_prefetch_data(
                b.buffer_rsrc(s, b.const_i32(64)), b.const_i32(1)
            ),
            "global_prefetch": lambda b, s, f: b.global_prefetch(s),
            "flat_prefetch": lambda b, s, f: b.flat_prefetch(f),
        }
        for name, emit in cases.items():
            with self.subTest(op=name):
                with self.assertRaisesRegex(ValueError, f"{name} requires gfx1250"):
                    _lower(one(emit), arch="gfx950", flavor="llvm22")

    def test_immediate_ranges(self):
        b = IRBuilder("bad")
        src = b.param("src", PtrType(F32, "global"))
        one = b.const_i32(1)
        with self.assertRaises(ValueError):
            b.s_setreg(65536, one)
        with self.assertRaises(ValueError):
            b.s_setreg(-1, one)
        with self.assertRaises(ValueError):
            b.global_prefetch(src, cachepolicy=32)
        rsrc = b.buffer_rsrc(src, b.const_i32(64))
        with self.assertRaises(ValueError):
            b.s_buffer_prefetch_data(rsrc, one, offset=1 << 31)

    def test_lowerer_rechecks_immediates(self):
        # Serialized IR reaches the lowerer without passing through the builder.
        b = IRBuilder("bad_attr")
        src = b.param("src", PtrType(F32, "global"))
        b.global_prefetch(src)
        b.s_setreg(1537, b.const_i32(1))
        b.ret()
        kernel = b.kernel
        ops = [op for op in kernel.body.ops if op.name.startswith("tile.")]
        ops[0].attrs["cachepolicy"] = 40
        with self.assertRaisesRegex(ValueError, "cachepolicy must be in 0..31"):
            _lower(kernel)
        ops[0].attrs["cachepolicy"] = 0
        ops[1].attrs["simm16"] = 70000
        with self.assertRaisesRegex(ValueError, "simm16 must fit"):
            _lower(kernel)

    def test_operand_types(self):
        b = IRBuilder("bad_types")
        src = b.param("src", PtrType(F32, "global"))
        flat = b.param("flat", PtrType(F32, "private"))
        one = b.const_i32(1)
        with self.assertRaises(TypeError):
            b.s_setreg(1537, b.const_i64(1))
        with self.assertRaises(TypeError):
            b.s_prefetch_data(b.const_i64(0), one)
        with self.assertRaises(TypeError):
            b.s_prefetch_data(src, b.const_i64(1))
        with self.assertRaises(TypeError):
            b.s_buffer_prefetch_data(b.zero_vec(I32, 8), one)
        with self.assertRaises(TypeError):
            b.global_prefetch(flat)
        with self.assertRaises(TypeError):
            b.flat_prefetch(src)

    def test_global_prefetch_rejects_param_moved_to_constant(self):
        b = IRBuilder("moved")
        table = b.param("table", PtrType(I32, "global"), addr_space="constant")
        b.global_prefetch(table)
        b.ret()
        with self.assertRaisesRegex(TypeError, "global_prefetch ptr must be a global"):
            _lower(b.kernel)

    def test_s_prefetch_data_rejects_lds_pointer(self):
        b = IRBuilder("lds")
        p = b.param("p", PtrType(I64, "lds"))
        b.s_prefetch_data(p, b.const_i32(1))
        b.ret()
        with self.assertRaisesRegex(ValueError, "s_prefetch_data: pointer operand"):
            _lower(b.kernel)

    def test_distinct_from_instruction_prefetch(self):
        b = IRBuilder("both")
        src = b.param("src", PtrType(F32, "global"))
        n = b.const_i32(2)
        b.s_prefetch_inst(src, n)
        b.s_prefetch_data(src, n)
        b.ret()
        llvm = _lower(b.kernel)
        self.assertIn("call void @llvm.amdgcn.s.prefetch.inst.p1(", llvm)
        self.assertIn("call void @llvm.amdgcn.s.prefetch.data.p1(", llvm)


if __name__ == "__main__":
    unittest.main()
