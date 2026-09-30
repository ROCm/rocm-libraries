# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text validation for the gfx1250 TDM descriptor path (Phase 5).

``IRBuilder.tdm_descriptor_2d`` packs the rank-2 D# groups that feed
``tensor_load_to_lds`` / ``tensor_store_from_lds``. The ISA half of the proof
lives in ``rocke/examples/gfx1250/isa_features/tdm_verify.py``, which needs an
llvm23 toolchain; these tests pin the LLVM the lowering hands it.
"""

from __future__ import annotations

import re
import unittest

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, VectorType, is_pure_op_name
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


def _build_round_trip(**descriptor):
    """global -> LDS -> global through one 64x16 i32 tile."""
    b = IRBuilder("gfx1250_tdm")
    src = b.param("src", PtrType(I32, "global"), readonly=True, align=16)
    dst = b.param("dst", PtrType(I32, "global"), align=16)
    dim0 = b.param("dim0", I32)
    dim1 = b.param("dim1", I32)
    stride = b.param("stride", I32)
    lds = b.smem_addr_of(b.smem_alloc(I32, [64 * 16], name_hint="tile"))
    shape = dict(
        elem_bytes=4,
        tensor_dim0=dim0,
        tensor_dim1=dim1,
        row_stride=stride,
        tile_dim0=64,
        tile_dim1=16,
    )
    shape.update(descriptor)
    b.tensor_load_to_lds(*b.tdm_descriptor_2d(src, lds, **shape))
    b.s_wait_tensorcnt(0)
    b.tensor_store_from_lds(*b.tdm_descriptor_2d(dst, lds, **shape))
    b.s_wait_tensorcnt(0)
    b.ret()
    return b.kernel


def _lower(kernel, *, arch="gfx1250", flavor="llvm23"):
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


def _readfirstlane_args(llvm):
    return re.findall(r"@llvm\.amdgcn\.readfirstlane\.i32\(i32 (\S+)\)", llvm)


class TestGfx1250Tdm(unittest.TestCase):
    def test_group0_address_packing(self):
        llvm = _lower(_build_round_trip())
        for name in ("src", "dst"):
            with self.subTest(ptr=name):
                self.assertRegex(
                    llvm, rf"%gaddr\d+ = ptrtoint ptr addrspace\(1\) %{name} to i64"
                )
        # hi[24:0] of the global address, type field (bits 31:30) = 2.
        self.assertRegex(llvm, r"= lshr i64 %gaddr\d+, 32")
        self.assertRegex(llvm, r"= and i32 %tr\d+, 33554431")
        self.assertRegex(llvm, r"= or i32 %and\d+, -2147483648")
        # The smem i64 address is truncated to the 32-bit LDS byte address.
        self.assertRegex(llvm, r"%tr\d+ = trunc i64 %lds_addr\d+ to i32")

    def test_group1_shape_packing(self):
        llvm = _lower(_build_round_trip())
        self.assertRegex(llvm, r"= shl i32 %dim0, 16")
        self.assertRegex(llvm, r"= lshr i32 %dim0, 16")
        self.assertRegex(llvm, r"= shl i32 %dim1, 16")
        self.assertRegex(llvm, r"= lshr i32 %dim1, 16")
        # tile_dim0=64 lands in the high half of dword 3.
        self.assertRegex(llvm, r"= or i32 %lshr\d+, 4194304")

    def test_every_descriptor_dword_is_wave_uniform(self):
        args = _readfirstlane_args(_lower(_build_round_trip()))
        # Two descriptors x (4 group-0 + 8 group-1) dwords.
        self.assertEqual(len(args), 24)
        # Group 0 dword 0 = 1; group 1 = flags(elem 4 B -> data_size 2),
        # ..., tile_dim1, row_stride, dim-1 stride 1<<16, 0.
        self.assertEqual(args[0], "1")
        self.assertEqual(args[4], str(2 << 16))
        self.assertEqual(args[8:12], ["16", "%stride", "65536", "0"])
        # The store descriptor carries the same constant words.
        self.assertEqual(args[12], "1")
        self.assertEqual(args[16], args[4])
        self.assertEqual(args[20:24], args[8:12])

    def test_transfer_calls_take_the_packed_groups(self):
        llvm = _lower(_build_round_trip())
        for intrinsic in ("tensor.load.to.lds", "tensor.store.from.lds"):
            with self.subTest(intrinsic=intrinsic):
                self.assertRegex(
                    llvm,
                    rf"call void @llvm\.amdgcn\.{re.escape(intrinsic)}\("
                    r"<4 x i32> %vp\d+, <8 x i32> %vp\d+, <4 x i32> (%cz\d+), "
                    r"<4 x i32> \1, <8 x i32> %cz\d+, i32 0\)",
                )
        self.assertEqual(
            llvm.count("call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)"), 2
        )

    def test_flag_word_variants(self):
        cases = (
            # elem 2 B, wg mask 3, pad on (interval 2, amount 5).
            (
                dict(elem_bytes=2, workgroup_mask=3, pad_interval=2, pad_amount=5),
                "177274883",
            ),
            # elem 8 B, pad interval 7 / amount 127 sets bit 31: printed signed.
            (dict(elem_bytes=8, pad_interval=7, pad_amount=127), "-2949120"),
            # elem 1 B, pad enabled with interval 0 still sets bit 20.
            (dict(elem_bytes=1, pad_interval=0), str(1 << 20)),
        )
        for descriptor, flags in cases:
            with self.subTest(**descriptor):
                self.assertEqual(
                    _readfirstlane_args(_lower(_build_round_trip(**descriptor)))[4],
                    flags,
                )

    def test_tile_dim0_high_bit_printed_signed(self):
        llvm = _lower(_build_round_trip(tile_dim0=0xFFFF))
        self.assertRegex(llvm, r"= or i32 %lshr\d+, -65536")

    def test_i32_lds_address_used_directly(self):
        b = IRBuilder("i32_lds")
        src = b.param("src", PtrType(I32, "global"))
        lds = b.param("lds", I32)
        n = b.const_i32(64)
        groups = b.tdm_descriptor_2d(
            src,
            lds,
            elem_bytes=4,
            tensor_dim0=n,
            tensor_dim1=n,
            row_stride=n,
            tile_dim0=64,
            tile_dim1=1,
        )
        b.tensor_load_to_lds(*groups)
        b.ret()
        llvm = _lower(b.kernel)
        self.assertNotIn("trunc i64 %lds", llvm)
        self.assertEqual(_readfirstlane_args(llvm)[1], "%lds")

    def test_group_types(self):
        b = IRBuilder("types")
        src = b.param("src", PtrType(I32, "global"))
        n = b.const_i32(8)
        groups = b.tdm_descriptor_2d(
            src,
            b.const_i32(0),
            elem_bytes=4,
            tensor_dim0=n,
            tensor_dim1=n,
            row_stride=n,
            tile_dim0=8,
            tile_dim1=8,
        )
        self.assertEqual(
            [g.type for g in groups],
            [
                VectorType(I32, 4),
                VectorType(I32, 8),
                VectorType(I32, 4),
                VectorType(I32, 4),
                VectorType(I32, 8),
            ],
        )
        self.assertIs(groups[2], groups[3])

    def test_global_ptr_to_i64(self):
        b = IRBuilder("p2i")
        src = b.param("src", PtrType(I32, "global"))
        dst = b.param("dst", PtrType(I64, "global"))
        b.global_store(dst, b.const_i32(0), b.global_ptr_to_i64(src))
        b.ret()
        llvm = _lower(b.kernel, arch="gfx950", flavor="llvm22")
        self.assertRegex(llvm, r"%gaddr\d+ = ptrtoint ptr addrspace\(1\) %src to i64")
        self.assertIs(b.global_ptr_to_i64(src).type, I64)
        with self.assertRaises(TypeError):
            b.global_ptr_to_i64(b.param("flat", PtrType(F32, "private")))

    def test_purity(self):
        # The address cast is side-effect free; the transfers are not.
        self.assertTrue(is_pure_op_name("tile.global_ptr_to_i64"))
        self.assertFalse(is_pure_op_name("tile.tensor_load_to_lds"))
        self.assertFalse(is_pure_op_name("tile.tensor_store_from_lds"))

    def test_serialization_roundtrip(self):
        kernel = _build_round_trip(workgroup_mask=5, pad_interval=1, pad_amount=3)
        text = serialize(kernel)
        parsed = parse(text)
        self.assertEqual(text, serialize(parsed))
        self.assertEqual(_lower(kernel), _lower(parsed))

    def test_transfers_are_gated(self):
        for arch, flavor, message in (
            ("gfx950", "llvm22", "requires gfx1250"),
            ("gfx1201", "llvm23", "requires gfx1250"),
            ("gfx1250", "llvm22", "requires LLVM flavor llvm23"),
        ):
            with self.subTest(arch=arch, flavor=flavor):
                with self.assertRaisesRegex(
                    ValueError, f"tensor_load_to_lds {message}"
                ):
                    _lower(_build_round_trip(), arch=arch, flavor=flavor)

    def test_builder_rejects(self):
        b = IRBuilder("bad")
        src = b.param("src", PtrType(I32, "global"))
        flat = b.param("flat", PtrType(F32, "private"))
        n = b.const_i32(8)
        lds = b.const_i32(0)

        def build(ptr=src, addr=lds, **overrides):
            args = dict(
                elem_bytes=4,
                tensor_dim0=n,
                tensor_dim1=n,
                row_stride=n,
                tile_dim0=8,
                tile_dim1=8,
            )
            args.update(overrides)
            b.tdm_descriptor_2d(ptr, addr, **args)

        type_errors = (
            ("global_ptr must be a global pointer", dict(ptr=flat)),
            ("lds_addr must be i64 or i32", dict(addr=b.const_f32(0.0))),
            ("tensor_dim0 must be i32", dict(tensor_dim0=b.const_i64(8))),
            ("tensor_dim1 must be i32", dict(tensor_dim1=b.const_i64(8))),
            ("row_stride must be i32", dict(row_stride=b.const_i64(8))),
        )
        for message, overrides in type_errors:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
                    build(**overrides)
        value_errors = (
            ("elem_bytes must be 1, 2, 4, or 8", dict(elem_bytes=3)),
            ("tile_dim0 must be in 1..65535", dict(tile_dim0=0)),
            ("tile_dim0 must be in 1..65535", dict(tile_dim0=1 << 16)),
            ("tile_dim1 must be in 1..65535", dict(tile_dim1=0)),
            ("workgroup_mask must be in 0..65535", dict(workgroup_mask=1 << 16)),
            ("workgroup_mask must be in 0..65535", dict(workgroup_mask=-1)),
            ("pad_interval must be in 0..7", dict(pad_interval=8)),
            ("pad_amount must be in 0..127", dict(pad_interval=0, pad_amount=128)),
            ("pad_amount .* needs pad_interval", dict(pad_amount=1)),
        )
        for message, overrides in value_errors:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build(**overrides)


if __name__ == "__main__":
    unittest.main()
