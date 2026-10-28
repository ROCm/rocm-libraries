# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One-instruction unit test for v_wmma_scale_f32_32x16x128_f4.

The atom is outside the 16x16 ``wmma_scaled`` family. The host checks assert
the HIP builtin and the LLVM intrinsic. The device check launches one wave
and compares the 32x16 FP32 tile with an independent reference.
"""

from __future__ import annotations

import struct
import unittest

import numpy as np

from rocke.core.arch import ArchTarget
from rocke.core.arch.wmma_scale import (
    gfx1250_scaled_wmma,
    is_wmma_scale_f32_32x16x128_f4,
)
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.compile import compile_kernel
from rocke.runtime.hip_module import Runtime, get_device_arch

_OP_ID = "wmma_scale_f32_32x16x128_f4"
_BUILTIN = "__builtin_amdgcn_wmma_scale_f32_32x16x128_f4"
_INTRINSIC = "llvm.amdgcn.wmma.scale.f32.32x16x128.f4"
_E8M0_ONE = 0x7F7F7F7F
_FP4_ONE = 2


class _Atom:
    op_id = _OP_ID
    c_frag_len = 16
    c_dtype = "fp32"


def _a_mk(lane: int, elem: int) -> tuple[int, int]:
    """A register map: ``K{2, 32} M{16} L{K1M0} V{M1K2K0}``."""
    m0 = lane % 16
    k1 = lane // 16
    k0 = elem % 32
    k2 = (elem // 32) % 2
    m1 = elem // 64
    return m1 * 16 + m0, (k2 * 2 + k1) * 32 + k0


def _b_kn(lane: int, elem: int) -> tuple[int, int]:
    """B register map: ``K{2, 32} L{K1N} V{K2K0}``."""
    col = lane % 16
    k1 = lane // 16
    k0 = elem % 32
    k2 = (elem // 32) % 2
    return (k2 * 2 + k1) * 32 + k0, col


def _c_mn(lane: int, slot: int) -> tuple[int, int]:
    """C register map: ``M{2, 8} L{M1N} V{M2M0}``."""
    col = lane % 16
    m1 = lane // 16
    m0 = slot % 8
    m2 = slot // 8
    return m2 * 16 + m1 * 8 + m0, col


def _build_unit_kernel():
    b = IRBuilder("wmma_scale_f32_32x16x128_f4_unit")
    a_ptr = b.param("A", PtrType(I32, "global"), readonly=True)
    b_ptr = b.param("B", PtrType(I32, "global"), readonly=True)
    c_ptr = b.param("C", PtrType(F32, "global"))
    lane = b.thread_id_x()
    a_base = b.mul(lane, b.const_i32(16))
    a = b.vec_concat(
        b.global_load_vN(a_ptr, a_base, I32, 8, align=4),
        b.global_load_vN(a_ptr, b.add(a_base, b.const_i32(8)), I32, 8, align=4),
    )
    b_frag = b.global_load_vN(b_ptr, b.mul(lane, b.const_i32(8)), I32, 8, align=4)
    scale = b.const_i32(_E8M0_ONE)
    acc = b.mma(_Atom(), a, b_frag, b.zero_vec_f32(16), scale, scale)
    n = b.const_i32(16)
    m_lane = b.div(lane, n)
    col = b.mod(lane, n)
    for slot in range(16):
        row = b.add(
            b.add(b.const_i32((slot // 8) * 16), b.mul(m_lane, b.const_i32(8))),
            b.const_i32(slot % 8),
        )
        b.global_store(c_ptr, b.add(b.mul(row, n), col), b.vec_extract(acc, slot), align=4)
    return b.kernel


def _pack_words(n_elem: int, code_at) -> np.ndarray:
    words = np.zeros((32, n_elem // 8), dtype=np.uint32)
    for lane in range(32):
        for elem in range(n_elem):
            code = int(code_at(lane, elem)) & 0xF
            word, nibble = divmod(elem, 8)
            words[lane, word] |= np.uint32(code) << np.uint32(4 * nibble)
    return words


class TestGfx1250WmmaScale32x16x128F4(unittest.TestCase):
    def test_atom_stays_outside_the_16x16_family(self):
        self.assertTrue(is_wmma_scale_f32_32x16x128_f4(_OP_ID))
        self.assertIsNone(gfx1250_scaled_wmma(_OP_ID))
        atoms = [
            op
            for op in ArchTarget.from_gfx("gfx1250").mma.ops
            if op.family == "wmma_scaled"
        ]
        self.assertEqual(len(atoms), 86)
        self.assertNotIn(_OP_ID, {op.op_id for op in atoms})

    def test_hip_and_llvm_name_the_instruction(self):
        kernel = _build_unit_kernel()
        hip = lower_kernel_to_hip(kernel, arch="gfx1250")
        llvm = _lower_kernel_to_llvm_python(
            kernel, llvm_flavor="llvm23", arch="gfx1250"
        )
        self.assertIn(_BUILTIN, hip)
        self.assertIn(_INTRINSIC, llvm)
        self.assertIn("i32 1, i32 0,", llvm)

    def test_one_instruction_matches_reference_on_gfx1250(self):
        arch = get_device_arch(0)
        if arch != "gfx1250":
            self.skipTest(f"visible HIP device 0 must be gfx1250, got {arch!r}")
        rng = np.random.default_rng(0x32F4)
        a_codes = rng.integers(0, 2, size=(32, 128), dtype=np.uint8) * _FP4_ONE
        b_codes = rng.integers(0, 2, size=(16, 128), dtype=np.uint8) * _FP4_ONE
        def b_code(lane, elem):
            k, col = _b_kn(lane, elem)
            return b_codes[col, k]

        a_words = _pack_words(128, lambda lane, elem: a_codes[_a_mk(lane, elem)])
        b_words = _pack_words(64, b_code)
        a_value = (a_codes == _FP4_ONE).astype(np.float32)
        b_value = (b_codes == _FP4_ONE).astype(np.float32)
        expected = a_value @ b_value.T

        kernel = _build_unit_kernel()
        art = compile_kernel(kernel, arch="gfx1250", backend="python")
        rt = Runtime()
        module = rt.load_module(art.hsaco)
        try:
            fn = module.get_function(art.kernel_name)
            out = np.full((32, 16), np.nan, dtype=np.float32)
            allocations = []
            for array in (a_words, b_words, out):
                ptr = rt.alloc(array.nbytes)
                allocations.append(ptr)
                host = (ctypes_from(array),)
                rt.memcpy_h2d(ptr, host[0], array.nbytes)
            rt.launch(fn, (1, 1, 1), (32, 1, 1), struct.pack("<QQQ", *allocations))
            rt.sync()
            raw = (ctypes_from(out),)
            rt.memcpy_d2h(raw[0], allocations[-1], out.nbytes)
            got = np.frombuffer(bytes(raw[0]), dtype=np.float32).reshape(32, 16)
        finally:
            module.unload()
            for ptr in allocations:
                rt.free(ptr)
        np.testing.assert_array_equal(got, expected)


def ctypes_from(array: np.ndarray):
    import ctypes

    array = np.ascontiguousarray(array)
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer_copy(array)


if __name__ == "__main__":
    unittest.main()
