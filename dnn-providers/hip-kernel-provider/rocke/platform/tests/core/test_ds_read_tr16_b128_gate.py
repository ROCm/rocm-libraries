# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The wide transpose-LDS read lowers on gfx1250 only.

LLVM has no ``llvm.amdgcn.ds.read.tr16.b128``. A declaration for it passes the
verifier and fails at link time, so every other target must raise instead.
"""

from __future__ import annotations

import unittest

from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import BF16, F16, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import (
    LLVM_FLAVORS,
    _decls_for_flavor,
    _lower_kernel_to_llvm_python,
)

_DEAD_KEY = "ds.read.tr16.b128"


def _kernel(elem=F16):
    b = IRBuilder("tr16_b128")
    b.kernel.attrs["max_workgroup_size"] = 32
    out = b.param("out", PtrType(elem, "global"), noalias=True, align=16)
    smem = b.smem_alloc(elem, [64, 8], name_hint="tile")
    tid = b.thread_id_x()
    v = b.ds_read_tr16_b128(smem, tid, b.const_i32(0), dtype=elem)
    b.global_store(out, tid, v)
    b.ret()
    return b.kernel


class TestWideTransposeRead(unittest.TestCase):
    def test_no_flavor_declares_the_dead_intrinsic(self):
        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                self.assertNotIn(_DEAD_KEY, _decls_for_flavor(flavor))

    def test_targets_without_the_instruction_raise(self):
        for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
            with self.subTest(arch=arch):
                with self.assertRaises(NotImplementedError) as ctx:
                    _lower_kernel_to_llvm_python(
                        _kernel(), arch=arch, llvm_flavor="llvm23"
                    )
                self.assertIn("ds_read_tr16_b128 is not available", str(ctx.exception))

    def test_gfx1250_keeps_the_element_typed_intrinsic(self):
        for elem, suffix in ((F16, "v8f16"), (BF16, "v8bf16")):
            with self.subTest(elem=elem.name):
                ir = _lower_kernel_to_llvm_python(
                    _kernel(elem), arch="gfx1250", llvm_flavor="llvm23"
                )
                self.assertIn(f"@llvm.amdgcn.ds.load.tr16.b128.{suffix}(", ir)
                self.assertNotIn("@llvm.amdgcn.ds.read.tr16.b128", ir)

    def test_hip_lowering_rejects_the_op(self):
        with self.assertRaises(NotImplementedError):
            lower_kernel_to_hip(_kernel(), arch="gfx950")

    def test_cpp_engine_agrees(self):
        probe = serialize(_kernel())
        try:
            _lower_via_cpp_engine(probe, "gfx1250", "llvm23")
        except BackendError as exc:
            self.skipTest(f"C++ engine unavailable: {exc}")
        with self.assertRaises(RuntimeError) as ctx:
            _lower_via_cpp_engine(probe, "gfx950", "llvm23")
        self.assertIn("ds_read_tr16_b128 is not available", str(ctx.exception))
        self.assertEqual(
            _lower_via_cpp_engine(probe, "gfx1250", "llvm23"),
            _lower_kernel_to_llvm_python(
                _kernel(), arch="gfx1250", llvm_flavor="llvm23"
            ),
        )


if __name__ == "__main__":
    unittest.main()
