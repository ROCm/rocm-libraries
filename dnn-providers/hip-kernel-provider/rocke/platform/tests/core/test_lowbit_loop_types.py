# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused coverage for low-bit dynamic ``scf.for`` loop-carried types."""

from __future__ import annotations

import re
import unittest

from rocke.core.ir import (
    BF8E5M2,
    FP8E4M3,
    I8,
    I16,
    I32,
    IRBuilder,
    PtrType,
    SmemType,
    Type,
    VectorType,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python


_LOWBIT_TYPES = (
    ("i8", I8, "i8"),
    ("i16", I16, "i16"),
    ("fp8e4m3", FP8E4M3, "i8"),
    ("bf8e5m2", BF8E5M2, "i8"),
    ("vec_i8", VectorType(I8, 4), "<4 x i8>"),
    ("vec_i16", VectorType(I16, 4), "<4 x i16>"),
    ("vec_fp8e4m3", VectorType(FP8E4M3, 4), "<4 x i8>"),
    ("vec_bf8e5m2", VectorType(BF8E5M2, 4), "<4 x i8>"),
)


def _build_loop(carry_types=_LOWBIT_TYPES):
    builder = IRBuilder("lowbit_loop_types")
    lower = builder.param("lower", I32)
    upper = builder.param("upper", I32)
    step = builder.param("step", I32)
    iter_args = [
        (f"carry_{name}", builder.param(f"init_{name}", carry_type))
        for name, carry_type, _ in carry_types
    ]
    loop = builder.scf_for_iter(lower, upper, step, iter_args, iv_name="iteration")
    with loop as (_, carries):
        yielded = list(carries)
        if len(yielded) > 1:
            yielded[1] = builder.add(yielded[1], yielded[1])
        builder.scf_yield(*yielded)
    builder.ret()
    return builder.kernel, loop.op


class TestLowbitLoopTypes(unittest.TestCase):
    def test_phi_latch_and_exit_use_complete_lowbit_types(self):
        kernel, loop_op = _build_loop()
        llvm = _lower_kernel_to_llvm_python(
            kernel, arch="gfx950", llvm_flavor="llvm20"
        )

        for (name, _, llvm_type), result in zip(_LOWBIT_TYPES, loop_op.results):
            with self.subTest(carry=name):
                carry = f"%carry_{name}"
                self.assertRegex(
                    llvm,
                    re.compile(
                        rf"^  {re.escape(carry)} = phi {re.escape(llvm_type)} ",
                        re.MULTILINE,
                    ),
                )
                self.assertRegex(
                    llvm,
                    re.compile(
                        rf"^  {re.escape(carry)}\.next\.for\.header(?:\.\d+)? = "
                        rf"bitcast {re.escape(llvm_type)} .* to {re.escape(llvm_type)}$",
                        re.MULTILINE,
                    ),
                )
                self.assertIn(
                    f"  {result.name} = bitcast {llvm_type} {carry} to {llvm_type}",
                    llvm,
                )

        self.assertRegex(
            llvm,
            re.compile(
                r"^  %add\d+ = add nsw i16 %carry_i16, %carry_i16$", re.MULTILINE
            ),
        )

    def test_serialization_round_trip_is_stable(self):
        kernel, _ = _build_loop()
        encoded = serialize(kernel)
        for _, carry_type, _ in _LOWBIT_TYPES:
            with self.subTest(carry_type=carry_type.name):
                self.assertIn(carry_type.name, encoded)
        reparsed = parse(encoded)
        self.assertEqual(serialize(reparsed), encoded)
        self.assertEqual(
            _lower_kernel_to_llvm_python(reparsed, arch="gfx950", llvm_flavor="llvm20"),
            _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor="llvm20"),
        )

    def test_python_cpp_bytes_match_for_each_llvm_flavor(self):
        try:
            import rocke_engine
        except ImportError as exc:
            self.skipTest(f"rocke_engine extension not built: {exc}")

        kernel, _ = _build_loop()
        encoded = serialize(kernel)
        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                python_llvm = _lower_kernel_to_llvm_python(
                    kernel, arch="gfx950", llvm_flavor=flavor
                )
                cpp_llvm = rocke_engine.lower_serialized_ir(
                    encoded, arch="gfx950", flavor=flavor
                )
                self.assertEqual(cpp_llvm, python_llvm)

    def test_illegal_loop_carried_types_are_rejected_clearly(self):
        illegal_types = (
            (Type("i7"), "has no LLVM mapping"),
            (VectorType(I8, 0), "must have a positive width"),
            (PtrType(I8, "global"), "expected a scalar or vector of scalar values"),
            (SmemType(I8, (4,)), "expected a scalar or vector of scalar values"),
            (
                VectorType(VectorType(I8, 2), 2),
                "expected a scalar or vector of scalar values",
            ),
        )
        for carry_type, message in illegal_types:
            with self.subTest(carry_type=carry_type.name):
                kernel, _ = _build_loop((("bad", carry_type, ""),))
                with self.assertRaisesRegex(
                    (ValueError, NotImplementedError), re.escape(message)
                ):
                    _lower_kernel_to_llvm_python(
                        kernel, arch="gfx950", llvm_flavor="llvm20"
                    )

    def test_malformed_serialized_type_metadata_is_rejected_clearly(self):
        kernel, loop_op = _build_loop((("i8", I8, "i8"),))
        loop_op.attrs["iter_args"][0]["type"] = "vec<i8x>"
        with self.assertRaisesRegex(
            ValueError, "metadata type 'vec<i8x>' does not match init type 'i8'"
        ):
            _lower_kernel_to_llvm_python(
                kernel, arch="gfx950", llvm_flavor="llvm20"
            )

    def test_cpp_rejects_illegal_type_and_malformed_metadata_clearly(self):
        try:
            import rocke_engine
        except ImportError as exc:
            self.skipTest(f"rocke_engine extension not built: {exc}")

        illegal_types = (
            (Type("i7"), "no LLVM mapping for type i7"),
            (VectorType(I8, 0), "must have a positive width"),
            (PtrType(I8, "global"), "expected a scalar or vector of scalar values"),
            (SmemType(I8, (4,)), "expected a scalar or vector of scalar values"),
            (
                VectorType(VectorType(I8, 2), 2),
                "expected a scalar or vector of scalar values",
            ),
        )
        for carry_type, message in illegal_types:
            with self.subTest(carry_type=carry_type.name):
                kernel, _ = _build_loop((("bad", carry_type, ""),))
                with self.assertRaisesRegex(RuntimeError, re.escape(message)):
                    rocke_engine.lower_serialized_ir(
                        serialize(kernel), arch="gfx950", flavor="llvm20"
                    )

        kernel, loop_op = _build_loop((("i8", I8, "i8"),))
        loop_op.attrs["iter_args"][0]["type"] = "vec<i8x>"
        with self.assertRaisesRegex(
            RuntimeError,
            "malformed vector type",
        ):
            rocke_engine.lower_serialized_ir(
                serialize(kernel), arch="gfx950", flavor="llvm20"
            )


if __name__ == "__main__":
    unittest.main()
