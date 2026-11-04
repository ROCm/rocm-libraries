# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Low-bit carries and metadata validation in both ``scf.for`` lowering paths."""

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


def _build_loop(carry_types=_LOWBIT_TYPES, *, trip_count=None, unroll=False):
    builder = IRBuilder("lowbit_loop_types")
    if trip_count is None:
        lower = builder.param("lower", I32)
        upper = builder.param("upper", I32)
        step = builder.param("step", I32)
    else:
        lower = builder.const_i32(0)
        upper = builder.const_i32(trip_count)
        step = builder.const_i32(1)
    iter_args = [
        (f"carry_{name}", builder.param(f"init_{name}", carry_type))
        for name, carry_type, _ in carry_types
    ]
    if iter_args:
        loop = builder.scf_for_iter(
            lower, upper, step, iter_args, iv_name="iteration", unroll=unroll
        )
    else:
        loop = builder.scf_for(lower, upper, step, iv_name="iteration")
        loop.op.attrs["unroll"] = unroll
    with loop:
        yielded = list(loop.iter_vars)
        if len(yielded) > 1:
            yielded[1] = builder.add(yielded[1], yielded[1])
        builder.scf_yield(*yielded)
    builder.ret()
    return builder.kernel, loop.op


class TestLowbitLoopTypes(unittest.TestCase):
    def test_unrolled_lowbit_carries_use_initial_or_final_yielded_values(self):
        for trip_count in (0, 1, 3):
            with self.subTest(trip_count=trip_count):
                kernel, loop_op = _build_loop(trip_count=trip_count, unroll=True)
                llvm = _lower_kernel_to_llvm_python(
                    kernel, arch="gfx950", llvm_flavor="llvm20"
                )
                self.assertNotIn("for.header", llvm)
                self.assertNotIn(" phi ", llvm)
                updates = re.findall(
                    r"^  (%\S+) = add nsw i16 (%\S+), (%\S+)$", llvm, re.MULTILINE
                )
                self.assertEqual(len(updates), trip_count)
                final_i16 = "%init_i16"
                for result, lhs, rhs in updates:
                    self.assertEqual((lhs, rhs), (final_i16, final_i16))
                    final_i16 = result
                for (name, _, llvm_type), result in zip(_LOWBIT_TYPES, loop_op.results):
                    expected = final_i16 if name == "i16" else f"%init_{name}"
                    self.assertIn(
                        f"  {result.name} = bitcast {llvm_type} {expected} to {llvm_type}",
                        llvm,
                    )

    def test_phi_latch_and_exit_use_complete_lowbit_types(self):
        kernel, loop_op = _build_loop()
        llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor="llvm20")

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

        for trip_count in (None, 0, 1, 3):
            kernel, _ = _build_loop(
                trip_count=trip_count, unroll=trip_count is not None
            )
            encoded = serialize(kernel)
            for flavor in LLVM_FLAVORS:
                with self.subTest(trip_count=trip_count, flavor=flavor):
                    python_llvm = _lower_kernel_to_llvm_python(
                        kernel, arch="gfx950", llvm_flavor=flavor
                    )
                    cpp_llvm = rocke_engine.lower_serialized_ir(
                        encoded, arch="gfx950", flavor=flavor
                    )
                    self.assertEqual(cpp_llvm, python_llvm)

    def _check_metadata_counts(self, lower, error_type):
        for unroll in (False, True):
            for num_carries in (0, 1):
                carry_types = _LOWBIT_TYPES[:num_carries]
                kernel, _ = _build_loop(carry_types, trip_count=1, unroll=unroll)
                # Plain scf_for has no iter_args or num_iter_args metadata.
                self.assertIn("ret void", lower(kernel))
                cases = [
                    (
                        "extra",
                        [{"name": "%extra", "type": "i8"}] * (num_carries + 1),
                        f"scf.for declares {num_carries} iter_args but has {num_carries + 1} metadata entries",
                    ),
                    (
                        "not_list",
                        "invalid",
                        "scf.for iter_args metadata must be a list",
                    ),
                ]
                if num_carries:
                    cases.extend(
                        [
                            (
                                "empty",
                                [],
                                "scf.for declares 1 iter_args but has 0 metadata entries",
                            ),
                            (
                                "absent",
                                None,
                                "scf.for declares 1 iter_args but has 0 metadata entries",
                            ),
                        ]
                    )
                for case, metadata, message in cases:
                    with self.subTest(
                        unroll=unroll, num_carries=num_carries, case=case
                    ):
                        kernel, loop_op = _build_loop(
                            carry_types, trip_count=1, unroll=unroll
                        )
                        # Keep operands resolvable so rejection reaches the lowerer,
                        # rather than failing on missing block names in the parser.
                        loop_op.regions[0].ops[-1].operands = loop_op.operands[3:]
                        if metadata is None:
                            del loop_op.attrs["iter_args"]
                        else:
                            loop_op.attrs["iter_args"] = metadata
                        with self.assertRaisesRegex(error_type, re.escape(message)):
                            lower(kernel)

    def test_python_validates_metadata_counts_in_both_loop_paths(self):
        self._check_metadata_counts(
            lambda kernel: _lower_kernel_to_llvm_python(
                kernel, arch="gfx950", llvm_flavor="llvm20"
            ),
            ValueError,
        )

    def test_cpp_validates_metadata_counts_in_both_loop_paths(self):
        try:
            import rocke_engine
        except ImportError as exc:
            self.skipTest(f"rocke_engine extension not built: {exc}")
        self._check_metadata_counts(
            lambda kernel: rocke_engine.lower_serialized_ir(
                serialize(kernel), arch="gfx950", flavor="llvm20"
            ),
            RuntimeError,
        )

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
            _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor="llvm20")

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
