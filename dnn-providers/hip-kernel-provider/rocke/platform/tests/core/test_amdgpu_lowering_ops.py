# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Core-lowering unit tests for the AMDGPU op additions that back the gfx950
grouped / ragged GEMM family.

These are engine-level (not kernel-family) contracts, so they live in
``tests/core``:

* ``lower_hip._op_tile_inline_asm`` -- general GCC-style inline asm in the HIP
  backend, with LLVM ``$N`` -> GCC ``%N`` placeholder translation and a memory
  clobber for side-effecting asm.
* ``lower_hip`` typed vector loads/stores -- the element-type prefix map must
  honor i32/f32/i8 for both the LDS and the global vN ops (the prior f16-only
  map silently reinterpreted them as f16), and reject anything unmapped.
* ``lower_hip`` arch seam -- an explicitly passed arch must resolve in the arch
  catalog; only an omitted arch falls back to the gfx950 baseline.
* ``lower_llvm._op_memref_global_atomic_add_pk_bf16`` -- lowers to a generic
  ``atomicrmw fadd <2 x bfloat>`` (the ``llvm.amdgcn.global.atomic.fadd.v2bf16``
  intrinsic does not exist in the shipping ROCm LLVM).
* ``IRBuilder.ds_read_tr16_*`` result-name hint -- must be ``dtr16`` (not
  ``tr16``) so the fresh-name counter can never collide with an ``arith.trunc``
  result (``tr`` + id).

No GPU: pure text lowering (``lower_kernel_to_hip`` / ``lower_kernel_to_llvm``)
and IRBuilder-level checks.

Run:  PYTHONPATH=python python3 tests/core/test_amdgpu_lowering_ops.py
"""

from __future__ import annotations

import unittest

from rocke.core.ir import BF16, I32, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm


class TestPackedBf16AtomicLowering(unittest.TestCase):
    def _kernel(self):
        b = IRBuilder("pk_bf16_atomic")
        p = b.param("p", PtrType(BF16, "global"))
        idx = b.param("idx", I32)
        val = b.global_load_vN(p, idx, BF16, 2, align=4)  # <2 x bf16>
        b.global_atomic_add_pk_bf16(p, idx, val)
        b.ret()
        return b.kernel

    def test_lowers_to_generic_atomicrmw(self):
        ir = lower_kernel_to_llvm(self._kernel(), arch="gfx950")
        self.assertIn("atomicrmw fadd ptr addrspace(1)", ir)
        self.assertIn("<2 x bfloat>", ir)
        # Device-local HBM contract metadata drives HW pk_add selection.
        self.assertIn("amdgpu.no.fine.grained.memory", ir)

    def test_does_not_emit_nonexistent_intrinsic(self):
        # This intrinsic is not in the shipping ROCm LLVM; emitting it was the bug.
        ir = lower_kernel_to_llvm(self._kernel(), arch="gfx950")
        self.assertNotIn("llvm.amdgcn.global.atomic.fadd.v2bf16", ir)


class TestInlineAsmHipLowering(unittest.TestCase):
    def test_emits_volatile_and_translates_placeholders(self):
        b = IRBuilder("asm_probe")
        out = b.param("out", PtrType(I32, "global"))
        raw = b.inline_asm(
            "ds_read_b32 $0, $1 offset:0",
            "=v,v",
            [b.const_i32(0)],
            result_type=I32,
            sideeffect=True,
        )
        b.store(raw, out, b.const_i32(0))
        b.ret()

        hip = lower_kernel_to_hip(b.kernel, arch="gfx950")
        self.assertIn("asm volatile", hip)
        # LLVM $N placeholders must be rewritten to GCC-style %N.
        self.assertIn("%0", hip)
        self.assertIn("%1", hip)
        self.assertNotIn("$0", hip)
        # Side-effecting asm gets a memory clobber so it orders vs LDS traffic.
        self.assertIn('"memory"', hip)


class TestSmemTypedVectorLoadHip(unittest.TestCase):
    def _i32_roundtrip_kernel(self):
        b = IRBuilder("smem_i32")
        out = b.param("out", PtrType(I32, "global"))
        buf = b.smem_alloc(I32, [16, 4], name_hint="s")
        v = b.global_load_vN(out, b.const_i32(0), I32, 4, align=16)  # <4 x i32>
        b.smem_store_vN(buf, [b.const_i32(0), b.const_i32(0)], v, 4)
        loaded = b.smem_load_vN(buf, b.const_i32(0), b.const_i32(0), dtype=I32, n=4)
        b.global_store_vN(out, b.const_i32(0), loaded, n=4, align=16)
        b.ret()
        return b.kernel

    def test_i32_lds_vector_load_uses_i32_prefix_not_f16(self):
        hip = lower_kernel_to_hip(self._i32_roundtrip_kernel(), arch="gfx950")
        # The i32 LDS buffer must be loaded through an i32 vector view, not
        # reinterpreted as f16 (the prior f16-only prefix map's bug). Target the
        # LDS load specifically (`f16x` alone always appears in the prologue
        # typedefs, so a bare absence check would be meaningless).
        self.assertIn("__shared__ int s", hip)  # i32 LDS storage
        self.assertIn("reinterpret_cast<const i32x4*>", hip)  # typed i32 LDS load
        self.assertNotIn("reinterpret_cast<const f16x4*>(&s", hip)  # not as f16

    def test_i32_global_vector_ops_use_i32_prefix_not_f16(self):
        # Same contract on the global side: viewing an i32 load as f16x4 and then
        # assigning it to an i32x4 lvalue *converts* the four values instead of
        # moving the bytes.
        hip = lower_kernel_to_hip(self._i32_roundtrip_kernel(), arch="gfx950")
        self.assertIn("reinterpret_cast<const i32x4*>(out + ", hip)
        self.assertIn("reinterpret_cast<i32x4*>(out + ", hip)
        self.assertNotIn("f16x4*>(out + ", hip)

    def test_bf16_global_vector_ops_use_bf16_prefix(self):
        b = IRBuilder("global_bf16")
        p = b.param("p", PtrType(BF16, "global"))
        v = b.global_load_vN(p, b.const_i32(0), BF16, 4, align=8)
        b.global_store_vN(p, b.const_i32(0), v, n=4, align=8)
        b.ret()

        hip = lower_kernel_to_hip(b.kernel, arch="gfx950")
        self.assertIn("reinterpret_cast<const bf16x4*>(p + ", hip)
        self.assertIn("reinterpret_cast<bf16x4*>(p + ", hip)

    def test_unsupported_elem_type_raises_instead_of_falling_back(self):
        from rocke.core.lower_hip import _vec_prefix

        with self.assertRaises(NotImplementedError) as cm:
            _vec_prefix("f64", "global_load_vN")
        self.assertIn("global_load_vN", str(cm.exception))
        self.assertIn("f64", str(cm.exception))


class TestHipArchSeamValidation(unittest.TestCase):
    """An arch the caller passed explicitly must resolve or fail loudly; only an
    omitted arch takes the gfx950 baseline."""

    def _kernel(self):
        b = IRBuilder("arch_seam")
        out = b.param("out", PtrType(I32, "global"))
        b.store(b.const_i32(1), out, b.const_i32(0))
        b.ret()
        return b.kernel

    def test_unknown_arch_raises(self):
        with self.assertRaises(KeyError):
            lower_kernel_to_hip(self._kernel(), arch="gfx999")

    def test_empty_arch_raises(self):
        with self.assertRaises(ValueError):
            lower_kernel_to_hip(self._kernel(), arch="")

    def test_omitted_arch_uses_baseline(self):
        self.assertEqual(
            lower_kernel_to_hip(self._kernel()),
            lower_kernel_to_hip(self._kernel(), arch="gfx950"),
        )


class TestDsReadTr16NameHint(unittest.TestCase):
    def test_tr16_result_name_avoids_trunc_collision(self):
        b = IRBuilder("tr16_name")
        buf = b.smem_alloc(BF16, [16, 32], name_hint="s")
        v = b.ds_read_tr16_b64(buf, b.const_i32(0), b.const_i32(0), dtype=BF16)
        # Hint is "dtr16", so the name can never collide with a trunc result
        # ("tr" + id): e.g. trunc id 16631 and tr16 id 631 both -> "tr16631".
        self.assertTrue(v.name.startswith("%dtr16"), v.name)
        self.assertFalse(v.name.startswith("%tr16"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
