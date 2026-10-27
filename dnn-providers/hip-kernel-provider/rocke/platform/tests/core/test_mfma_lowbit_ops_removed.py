# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dedicated fp4 and fp6 MFMA ops are removed.

Each op called an LLVM intrinsic that does not exist. The call passed `opt
-passes=verify` and `clang -S`. It failed only at link time. LLVM has one
f8f6f4 MFMA family. Use `mfma_scale_f32_16x16x128_f8f6f4` or
`mfma_f32_16x16x128_fp4fp6fp8` with `a_format` and `b_format` set to
`fp4e2m1`, `fp6e2m3`, or `fp6e3m2`.

These tests pin the removal in every layer, so a lowering handler cannot return
without a working intrinsic.
"""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget
from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import F32, IRBuilder, KernelDef, VectorType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import (
    LLVM_FLAVORS,
    _decls_for_flavor,
    _lower_kernel_to_llvm_python,
)
from rocke.helpers.atoms import MFMA_ATOMS, MfmaAtom

# The op id is also the name of the builder method.
# Each row holds the op id, the declaration key, and the atom factory name.
_REMOVED = (
    ("mfma_f32_16x16x128_fp4", "mfma.f32.16x16x128.fp4", "fp4_16x16x128"),
    ("mfma_f32_16x16x96_fp6", "mfma.f32.16x16x96.fp6", "fp6_16x16x96"),
)


def _probe_kernel() -> KernelDef:
    """A kernel without an MMA op. It shows that the C++ engine is available."""
    b = IRBuilder("removed_op_probe")
    b.kernel.attrs["max_workgroup_size"] = 64
    b.ret()
    return b.kernel


def _raw_kernel(op_id: str) -> KernelDef:
    """A kernel that carries the op without the builder checks.

    Serialized IR reaches the lowerers the same way, so each lowerer must
    reject the op on its own.
    """
    b = IRBuilder("removed_op")
    b.kernel.attrs["max_workgroup_size"] = 64
    a = b.vector_splat(b.const_i32(1), 8)
    acc = b.vector_splat(b.const_f32(0.0), 4)
    b._op(
        "tile.mma",
        [a, a, acc],
        [VectorType(F32, 4)],
        attrs={"op_id": op_id},
        result_name_hint="acc",
    )
    b.ret()
    return b.kernel


class TestRemovedOps(unittest.TestCase):
    def test_the_builder_has_no_method_for_the_op(self):
        for op_id, _decl, _factory in _REMOVED:
            with self.subTest(op_id=op_id):
                self.assertFalse(hasattr(IRBuilder, op_id))

    def test_the_builder_refuses_the_op_id(self):
        for op_id, _decl, _factory in _REMOVED:
            with self.subTest(op_id=op_id):
                b = IRBuilder("removed")
                a = b.vector_splat(b.const_i32(1), 8)
                acc = b.vector_splat(b.const_f32(0.0), 4)
                with self.assertRaises(ValueError):
                    b.mma(op_id, a, a, acc)

    def test_the_catalogs_have_no_entry_for_the_op(self):
        gfx950 = ArchTarget.from_gfx("gfx950")
        for op_id, _decl, factory in _REMOVED:
            with self.subTest(op_id=op_id):
                self.assertIsNone(gfx950.mma.by_op_id(op_id))
                self.assertFalse(hasattr(MfmaAtom, factory))
                self.assertNotIn(op_id, [atom.name for atom in MFMA_ATOMS])

    def test_no_flavor_declares_the_intrinsic(self):
        for _op_id, decl, _factory in _REMOVED:
            for flavor in LLVM_FLAVORS:
                with self.subTest(decl=decl, flavor=flavor):
                    self.assertNotIn(decl, _decls_for_flavor(flavor))

    def test_every_lowerer_rejects_a_serialized_op(self):
        for op_id, _decl, _factory in _REMOVED:
            kernel = _raw_kernel(op_id)
            with self.subTest(op_id=op_id, engine="python"):
                with self.assertRaises(NotImplementedError) as ctx:
                    _lower_kernel_to_llvm_python(
                        kernel, arch="gfx950", llvm_flavor="llvm23"
                    )
                self.assertIn(op_id, str(ctx.exception))
            with self.subTest(op_id=op_id, engine="hip"):
                with self.assertRaises(NotImplementedError) as ctx:
                    lower_kernel_to_hip(kernel, arch="gfx950")
                self.assertIn(op_id, str(ctx.exception))

    def test_the_cpp_engine_rejects_a_serialized_op(self):
        try:
            _lower_via_cpp_engine(serialize(_probe_kernel()), "gfx950", "llvm23")
        except BackendError as exc:
            self.skipTest(f"C++ engine unavailable: {exc}")
        for op_id, _decl, _factory in _REMOVED:
            with self.subTest(op_id=op_id), self.assertRaises(RuntimeError) as ctx:
                _lower_via_cpp_engine(serialize(_raw_kernel(op_id)), "gfx950", "llvm23")
            self.assertIn(op_id, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
