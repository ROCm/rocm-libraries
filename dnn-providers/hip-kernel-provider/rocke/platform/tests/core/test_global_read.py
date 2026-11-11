# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric checks of authored address arithmetic, without GPU allocations.

Interpret the emitted integer operations with real signed widths, select the
emitted runtime branch, and observe the complete load's addresses. This makes
near-2GiB and >4GiB cases practical without allocating giant host buffers.
Python/C byte parity is covered by instances/parity/global_read_emit.{py,c}.
"""
import unittest

from rocke.core.ir import BF16, F16, F32, F64, I32, I64, IRBuilder, PtrType
from rocke.helpers import (
    GlobalReadSpec,
    TensorDescriptor,
    emit_global_read_paths,
    make_global_view,
)


class AddressMachine:
    def __init__(self, params):
        self.values = {"%" + k: v for k, v in params.items()}
        self.loads = []
        self.branches = []

    def region(self, region):
        for op in region.ops:
            args = [self.values[v.name] for v in op.operands]
            name = op.name
            if name == "arith.constant":
                result = op.attrs["value"]
            elif name == "arith.add":
                result = args[0] + args[1]
            elif name == "arith.sub":
                result = args[0] - args[1]
            elif name == "arith.mul":
                result = args[0] * args[1]
            elif name == "arith.smin":
                result = min(args)
            elif name == "arith.smax":
                result = max(args)
            elif name in ("arith.sext", "arith.trunc"):
                result = args[0]
            elif name == "arith.and":
                result = args[0] & args[1]
            elif name == "arith.or":
                result = args[0] | args[1]
            elif name == "arith.select":
                result = args[1] if args[0] else args[2]
            elif name == "arith.cmp":
                a, b = args
                result = {
                    "eq": a == b,
                    "ne": a != b,
                    "ge": a >= b,
                    "le": a <= b,
                    "lt": a < b,
                    "gt": a > b,
                }[op.attrs["pred"]]
            elif name == "gpu.thread_id":
                result = self.values.get("%tid", 0)
            elif name == "tile.global_ptr_offset":
                result = args[0] + args[1]
            elif name == "tile.buffer_rsrc":
                result = tuple(args)
            elif name in ("scf.if_else", "scf.if"):
                taken = bool(args[0])
                self.branches.append(taken)
                if taken or name == "scf.if_else":
                    self.region(op.regions[0 if taken else 1])
                continue
            elif name.startswith("memref.global_load") or name.startswith(
                "tile.buffer_load"
            ):
                dtype = op.attrs["elem_type"]
                size = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4, "f64": 8}[dtype]
                n = (
                    op.results[0].type.count
                    if hasattr(op.results[0].type, "count")
                    else 1
                )
                if name.startswith("tile.buffer"):
                    (base, bound), off, scalar = args
                    address = base + off + scalar
                    assert 0 <= off + scalar and off + scalar + size * n <= bound
                else:
                    address = args[0] + args[1] * size
                self.loads.append(tuple(address + i * size for i in range(n)))
                result = 0
            elif name in ("func.return", "gpu.barrier", "scf.yield"):
                continue
            else:
                raise AssertionError(f"unsupported numeric op {name}")
            if op.results:
                value = op.results[0]
                if value.type == I64:
                    # Overflow in span/bind/global delta arithmetic must not be
                    # hidden by the evaluator's unbounded Python integers.
                    assert -(1 << 63) <= result < (1 << 63), (name, result)
                if value.type == I32:
                    result = ((result + (1 << 31)) % (1 << 32)) - (1 << 31)
                self.values[value.name] = result


def authored(
    dtype=F16,
    *,
    addressing="auto",
    negative=False,
    divergent=False,
    vector=8,
    zero_trip=False,
    two_inputs=False,
):
    b = IRBuilder("prepared_numeric")
    base = b.param("base", PtrType(dtype, "global"))
    rows = b.param("rows", I64)
    stride = b.param("stride", I64)
    row = b.param("row", I64)
    col = b.param("col", I64)
    delta_row = b.param("delta_row", I64)
    delta_col = b.param("delta_col", I64)
    if divergent:
        rows = b.add(rows, b.sext(b.thread_id_x(), I64))
    view = make_global_view(
        base, (1, 1), dtype, strides=(-64 if negative else stride, 1)
    )
    specs = [GlobalReadSpec(view, (rows, 64))]
    if two_inputs:
        other_rows = b.param("other_rows", I64)
        specs.append(GlobalReadSpec(view, (other_rows, 64)))

    def body(plans):
        cursor = plans[0].bind(b, (row, col))
        if not zero_trip:
            if vector:
                cursor.load_vec(b, (delta_row, delta_col), n=vector)
            else:
                cursor.load_scalar(b, (delta_row, delta_col))

    emit_global_read_paths(b, specs, body, addressing=addressing)
    return b.kernel


class GlobalReadNumericTests(unittest.TestCase):
    def run_case(self, kernel, **params):
        data = dict(
            base=0x123400000000,
            rows=2,
            stride=64,
            row=1,
            col=0,
            delta_row=0,
            delta_col=0,
        )
        data.update(params)
        machine = AddressMachine(data)
        machine.region(kernel.body)
        return machine

    def test_near_signed_buffer_limit_and_beyond_four_gib(self):
        kernel = authored()
        # Last complete 8-half vector in a span just below INT32_MAX bytes.
        fast = self.run_case(kernel, rows=16777215, row=16777214, col=56)
        self.assertEqual(fast.branches, [True])
        self.assertEqual(
            fast.loads[0], tuple(0x123400000000 + 2147483504 + 2 * i for i in range(8))
        )
        wide = self.run_case(kernel, rows=33554434, row=33554433, col=56)
        self.assertEqual(wide.branches, [False])
        self.assertEqual(
            wide.loads[0], tuple(0x123400000000 + 4294967536 + 2 * i for i in range(8))
        )

    def test_exact_limit_boundary(self):
        kernel = authored()
        self.assertEqual(self.run_case(kernel, rows=16777215).branches, [True])
        self.assertEqual(self.run_case(kernel, rows=16777216).branches, [False])

    def test_extreme_span_metadata_cannot_wrap_into_buffer(self):
        kernel = authored(zero_trip=True)
        for rows, stride in (
            ((1 << 63) - 1, (1 << 63) - 1),
            (2, (1 << 63) - 1),
            (-1, 64),
            (2, -64),
        ):
            with self.subTest(rows=rows, stride=stride):
                machine = self.run_case(kernel, rows=rows, stride=stride, row=0)
                self.assertEqual(machine.branches, [False])
                self.assertEqual(machine.loads, [])
        # An empty tensor has nothing to bound: it takes the full-width path.
        empty = self.run_case(kernel, rows=0, row=0)
        self.assertEqual(empty.branches, [False])
        self.assertEqual(empty.loads, [])

    def test_signed_negative_stride_and_delta(self):
        negative = self.run_case(authored(negative=True), row=2, col=16, delta_row=-1)
        self.assertEqual(negative.branches, [])
        self.assertEqual(negative.loads[0][0], 0x123400000000 - 96)
        for mode in ("auto", "global64"):
            result = self.run_case(
                authored(addressing=mode), row=1, col=16, delta_row=-1, delta_col=-8
            )
            self.assertEqual(
                result.loads[0], tuple(0x123400000010 + 2 * i for i in range(8))
            )

    def test_all_spans_control_the_whole_body(self):
        kernel = authored(two_inputs=True)
        machine = self.run_case(kernel, other_rows=33554434)
        self.assertEqual(machine.branches, [False])
        self.assertEqual(machine.loads[0][0], 0x123400000080)

    def test_nonuniform_metadata_never_creates_uniform_branch(self):
        result = self.run_case(authored(divergent=True))
        self.assertEqual(result.branches, [])
        self.assertEqual(result.loads[0][0], 0x123400000080)

    def test_real_vector_and_scalar_dtype_address_units(self):
        for dtype, size, width in (
            (F16, 2, 8),
            (BF16, 2, 8),
            (F32, 4, 4),
            (I32, 4, 4),
            (F64, 8, 2),
        ):
            for mode in ("auto", "global64"):
                for vector in (0, width):
                    with self.subTest(dtype=dtype.name, mode=mode, vector=vector):
                        machine = self.run_case(
                            authored(dtype, addressing=mode, vector=vector), col=8
                        )
                        expected = tuple(
                            0x123400000000 + (72 + i) * size for i in range(vector or 1)
                        )
                        self.assertEqual(machine.loads[0], expected)

    def test_wide_descriptor_extends_before_products(self):
        b = IRBuilder("wide_descriptor")
        row = b.param("row", I32)
        col = b.param("col", I32)
        stride = b.param("stride", I32)
        desc = TensorDescriptor.with_strides((1, 1), (stride, -1), F16)
        off = desc.offset_i64(b, (row, col))
        machine = AddressMachine(dict(row=1073741825, col=7, stride=8))
        machine.region(b.kernel.body)
        self.assertEqual(machine.values[off.name], 8589934593)

    def test_rejects_unsafe_mode_rank_and_unsupported_vector(self):
        b = IRBuilder("invalid_prepared")
        base = b.param("base", PtrType(F16, "global"))
        view = make_global_view(base, (1, 64), F16)
        with self.assertRaises(ValueError):
            GlobalReadSpec(view, (64,))
        spec = GlobalReadSpec(view, (1, 64))
        with self.assertRaises(ValueError):
            emit_global_read_paths(b, (spec,), lambda plans: None, addressing="buffer")
        # Neither path invents scalar loads to hide an unsupported vector.
        with self.assertRaises(ValueError):
            authored(vector=3)


if __name__ == "__main__":
    unittest.main()
