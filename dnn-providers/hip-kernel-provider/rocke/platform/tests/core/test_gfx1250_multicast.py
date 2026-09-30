# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text validation for the gfx1250 cluster multicast loads (Phase 4).

The ISA half of the proof lives in
``rocke/examples/gfx1250/isa_features/multicast_verify.py``, which needs an
llvm23 toolchain; these tests pin the LLVM the lowering hands it.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, VectorType, is_pure_op_name
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _build_all():
    b = IRBuilder("gfx1250_multicast")
    src = b.param("src", PtrType(I32, "global"), readonly=True, align=16)
    dst = b.param("dst", PtrType(I32, "global"), align=16)
    mask = b.param("mask", I32)
    smem = b.smem_alloc(I32, [64], name_hint="stage")
    local = b.smem_addr_of(smem)

    v1 = b.cluster_load(src, mask)
    v2 = b.cluster_load(src, mask, width_bytes=8, cachepolicy=1)
    v4 = b.cluster_load(src, mask, width_bytes=16, cachepolicy=8)
    zero = b.const_i32(0)
    b.global_store(dst, zero, v1)
    b.global_store(dst, zero, b.vec_extract(v2, 1))
    b.global_store(dst, zero, b.vec_extract(v4, 3))
    b.cluster_load_async_to_lds(src, local, mask, width_bytes=1)
    b.cluster_load_async_to_lds(src, local, mask, width_bytes=4, offset_bytes=16)
    b.cluster_load_async_to_lds(src, local, mask, width_bytes=8, cachepolicy=3)
    b.cluster_load_async_to_lds(
        src, local, mask, width_bytes=16, offset_bytes=-32, cachepolicy=31
    )
    b.ret()
    return b.kernel


def _lower(kernel, *, arch="gfx1250", flavor="llvm23"):
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


class TestGfx1250Multicast(unittest.TestCase):
    def test_exact_sync_calls(self):
        llvm = _lower(_build_all())
        patterns = (
            r"= call i32 @llvm\.amdgcn\.cluster\.load\.b32\.i32\("
            r"ptr addrspace\(1\) %src, i32 0, i32 %mask\)",
            r"= call <2 x i32> @llvm\.amdgcn\.cluster\.load\.b64\.v2i32\("
            r"ptr addrspace\(1\) %src, i32 1, i32 %mask\)",
            r"= call <4 x i32> @llvm\.amdgcn\.cluster\.load\.b128\.v4i32\("
            r"ptr addrspace\(1\) %src, i32 8, i32 %mask\)",
        )
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertRegex(llvm, pattern)

    def test_exact_async_calls(self):
        llvm = _lower(_build_all())
        patterns = (
            (r"b8", r"i32 0, i32 0"),
            (r"b32", r"i32 16, i32 0"),
            (r"b64", r"i32 0, i32 3"),
            (r"b128", r"i32 -32, i32 31"),
        )
        for suffix, imms in patterns:
            with self.subTest(suffix=suffix):
                self.assertRegex(
                    llvm,
                    rf"call void @llvm\.amdgcn\.cluster\.load\.async\.to\.lds\.{suffix}\("
                    rf"ptr addrspace\(1\) %src, ptr addrspace\(3\) %\S+, {imms}, "
                    rf"i32 %mask\)",
                )

    def test_exact_declares(self):
        llvm = _lower(_build_all())
        expected = (
            "declare i32 @llvm.amdgcn.cluster.load.b32.i32("
            "ptr addrspace(1), i32 immarg, i32)",
            "declare <2 x i32> @llvm.amdgcn.cluster.load.b64.v2i32("
            "ptr addrspace(1), i32 immarg, i32)",
            "declare <4 x i32> @llvm.amdgcn.cluster.load.b128.v4i32("
            "ptr addrspace(1), i32 immarg, i32)",
        )
        for text in expected:
            with self.subTest(text=text):
                self.assertEqual(llvm.count(text), 1)
        for suffix in ("b8", "b32", "b64", "b128"):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    llvm.count(
                        f"declare void @llvm.amdgcn.cluster.load.async.to.lds.{suffix}("
                        "ptr addrspace(1), ptr addrspace(3), i32 immarg, i32 immarg, i32)"
                    ),
                    1,
                )

    def test_unused_widths_declare_nothing(self):
        b = IRBuilder("only_b32")
        src = b.param("src", PtrType(I32, "global"))
        dst = b.param("dst", PtrType(I32, "global"))
        b.global_store(dst, b.const_i32(0), b.cluster_load(src, b.const_i32(3)))
        b.ret()
        llvm = _lower(b.kernel)
        self.assertIn("@llvm.amdgcn.cluster.load.b32.i32(", llvm)
        for name in ("cluster.load.b64", "cluster.load.b128", "cluster.load.async"):
            with self.subTest(name=name):
                self.assertNotIn(f"@llvm.amdgcn.{name}", llvm)

    def test_result_types(self):
        b = IRBuilder("types")
        src = b.param("src", PtrType(I32, "global"))
        mask = b.const_i32(1)
        self.assertIs(b.cluster_load(src, mask).type, I32)
        self.assertEqual(
            b.cluster_load(src, mask, width_bytes=8).type, VectorType(I32, 2)
        )
        self.assertEqual(
            b.cluster_load(src, mask, width_bytes=16).type, VectorType(I32, 4)
        )

    def test_ops_are_impure(self):
        # Multicast writes other workgroups' registers/LDS; DCE must keep them.
        self.assertFalse(is_pure_op_name("tile.cluster_load"))
        self.assertFalse(is_pure_op_name("tile.cluster_load_async_to_lds"))

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
            src = b.param("src", PtrType(I32, "global"))
            local = b.smem_addr_of(b.smem_alloc(I32, [4]))
            emit(b, src, local)
            b.ret()
            return b.kernel

        cases = {
            "cluster_load": lambda b, s, l: b.cluster_load(s, b.const_i32(1)),
            "cluster_load_async_to_lds": lambda b, s, l: b.cluster_load_async_to_lds(
                s, l, b.const_i32(1), width_bytes=4
            ),
        }
        for name, emit in cases.items():
            with self.subTest(op=name):
                with self.assertRaisesRegex(ValueError, f"{name} requires gfx1250"):
                    _lower(one(emit), arch="gfx950", flavor="llvm22")

    def test_builder_rejects(self):
        b = IRBuilder("bad")
        src = b.param("src", PtrType(I32, "global"))
        flat = b.param("flat", PtrType(F32, "private"))
        local = b.smem_addr_of(b.smem_alloc(I32, [4]))
        mask = b.const_i32(1)
        for width in (1, 2, 32):
            with self.subTest(sync_width=width):
                with self.assertRaises(ValueError):
                    b.cluster_load(src, mask, width_bytes=width)
        for width in (2, 32):
            with self.subTest(async_width=width):
                with self.assertRaises(ValueError):
                    b.cluster_load_async_to_lds(src, local, mask, width_bytes=width)
        with self.assertRaises(ValueError):
            b.cluster_load(src, mask, cachepolicy=32)
        with self.assertRaises(ValueError):
            b.cluster_load_async_to_lds(src, local, mask, width_bytes=4, cachepolicy=-1)
        with self.assertRaises(ValueError):
            b.cluster_load_async_to_lds(
                src, local, mask, width_bytes=4, offset_bytes=1 << 31
            )
        with self.assertRaises(TypeError):
            b.cluster_load(flat, mask)
        with self.assertRaises(TypeError):
            b.cluster_load(src, b.const_i64(1))
        with self.assertRaises(TypeError):
            b.cluster_load_async_to_lds(flat, local, mask, width_bytes=4)
        with self.assertRaises(TypeError):
            b.cluster_load_async_to_lds(src, src, mask, width_bytes=4)
        with self.assertRaises(TypeError):
            b.cluster_load_async_to_lds(src, local, b.const_i64(1), width_bytes=4)

    def test_lowerer_rechecks_attrs(self):
        # Serialized IR reaches the lowerer without passing through the builder.
        kernel = _build_all()
        ops = [op for op in kernel.body.ops if op.name.startswith("tile.cluster_load")]
        sync, async_b8 = ops[0], ops[3]
        cases = (
            (sync, "cachepolicy", 40, "cluster_load cachepolicy must be in 0..31"),
            (sync, "width_bytes", 2, "cluster_load width_bytes must be 4, 8, or 16"),
            (sync, "width_bytes", 8, "cluster_load result must be vec<i32x2>, got i32"),
            (
                async_b8,
                "width_bytes",
                2,
                "cluster_load_async_to_lds width_bytes must be 1, 4, 8, or 16",
            ),
            (
                async_b8,
                "offset_bytes",
                1 << 31,
                "cluster_load_async_to_lds offset_bytes must fit signed i32",
            ),
            (
                async_b8,
                "cachepolicy",
                32,
                "cluster_load_async_to_lds cachepolicy must be in 0..31",
            ),
        )
        for op, key, value, message in cases:
            with self.subTest(key=key, value=value):
                saved = op.attrs[key]
                op.attrs[key] = value
                try:
                    with self.assertRaisesRegex((ValueError, TypeError), message):
                        _lower(kernel)
                finally:
                    op.attrs[key] = saved
        _lower(kernel)

    def test_lds_i64_address_accepted(self):
        b = IRBuilder("i64_lds")
        src = b.param("src", PtrType(I32, "global"))
        addr = b.param("lds_addr", I64)
        b.cluster_load_async_to_lds(src, addr, b.const_i32(1), width_bytes=4)
        b.ret()
        llvm = _lower(b.kernel)
        self.assertRegex(
            llvm, r"inttoptr i64 %lds_addr to ptr addrspace\(3\)|ptr addrspace\(3\)"
        )
        self.assertIn("@llvm.amdgcn.cluster.load.async.to.lds.b32(", llvm)


if __name__ == "__main__":
    unittest.main()
