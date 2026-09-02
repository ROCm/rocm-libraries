# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression coverage for helpers that must distinguish MMA C from D."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I32, IRBuilder, VectorType
from rocke.core.isa.backend import Gfx11RdnaBackend
from rocke.helpers.atoms import MfmaAtom, WmmaAtom
from rocke.helpers.distribution import WmmaTensor, store_wmma_acc, wmma_mma


class _Layout:
    def __init__(self, role: str):
        self.role = role
        self.slots = []

    def coord(self, _builder, _lane, slot):
        self.slots.append(slot)
        return slot, slot + 10


class _Atom:
    a_per_lane = 1
    b_per_lane = 1
    c_per_lane = 2
    d_per_lane = 3
    dtype_c = "i32"
    dtype_d = "f32"

    def __init__(self):
        self.c_map = _Layout("c")
        self.d_map = _Layout("d")

    def zero_acc(self, _builder):
        return "zero-c"

    def emit(self, _builder, _a, _b, _c):
        return "result-d"

    def a_layout(self, _arch):
        return _Layout("a")

    def b_layout(self, _arch):
        return _Layout("b")

    def c_layout(self, _arch):
        return self.c_map

    def d_layout(self, _arch):
        return self.d_map


class _StoreBuilder:
    def const_i32(self, value):
        return value

    def add(self, a, b):
        return a + b

    def vec_extract(self, value, slot):
        return value[slot]

    def cast_f32_to(self, value, _dtype):
        return value


class _Window:
    dtype = F32

    def __init__(self):
        self.stores = []

    def store_scalar(self, _builder, *indices, value, align=None):
        self.stores.append((indices, value, align))


class _EmitBlock:
    def __init__(self):
        self.lines = []

    def emit(self, line):
        self.lines.append(line)


class _Lowerer:
    def __init__(self):
        self.block = _EmitBlock()

    def _need(self, _key):
        pass

    def _operand(self, value):
        return value.name

    def _current(self):
        return self.block


class TestFourRoleAtomHelpers(unittest.TestCase):
    def test_zero_acc_uses_c_dtype_and_width(self):
        builder = IRBuilder("four_role_zero")
        mfma = MfmaAtom(1, 1, 1, 1, 1, 2, 5, "f16", "i32", "f32", "synthetic")
        wmma = WmmaAtom(1, 1, 1, 1, 1, 2, 5, "f16", "i32", "f32", "synthetic")

        for atom in (mfma, wmma):
            with self.subTest(atom=type(atom).__name__):
                zero = atom.zero_acc(builder)
                self.assertEqual(zero.type.count, 2)
                self.assertIs(zero.type.elem, I32)

    def test_wmma_tensor_transitions_from_c_to_d(self):
        atom = _Atom()
        c = WmmaTensor.zero_acc(object(), atom)
        self.assertEqual(c.role, "c")
        self.assertEqual(c.num_slots, 2)
        self.assertIs(c._layout(), atom.c_map)

        d = wmma_mma(
            object(),
            WmmaTensor(atom, "a", "a"),
            WmmaTensor(atom, "b", "b"),
            c,
        )
        self.assertEqual(d.role, "d")
        self.assertEqual(d.num_slots, 3)
        self.assertIs(d._layout(), atom.d_map)

        with self.assertRaisesRegex(ValueError, "C and D fragment types differ"):
            wmma_mma(
                object(),
                WmmaTensor(atom, "a", "a"),
                WmmaTensor(atom, "b", "b"),
                d,
            )

    def test_store_uses_d_layout_and_d_slot_count(self):
        atom = _Atom()
        window = _Window()
        store_wmma_acc(_StoreBuilder(), window, atom, lane=0, acc=[1.0, 2.0, 3.0])

        self.assertEqual(atom.c_map.slots, [])
        self.assertEqual(atom.d_map.slots, [0, 1, 2])
        self.assertEqual(len(window.stores), 3)

    def test_wmma_lowering_types_c_argument_and_d_result_independently(self):
        lowerer = _Lowerer()
        op = SimpleNamespace(
            name="tile.wmma_f32_16x16x16_f16",
            operands=[
                SimpleNamespace(name="%a", type=VectorType(F32, 16)),
                SimpleNamespace(name="%b", type=VectorType(F32, 16)),
                SimpleNamespace(name="%c", type=VectorType(I32, 2)),
            ],
            result=SimpleNamespace(name="%d", type=VectorType(F32, 5)),
        )

        Gfx11RdnaBackend(ArchTarget.from_gfx("gfx1151")).emit_wmma(lowerer, op)

        call = lowerer.block.lines[-1]
        self.assertIn("%d = call <5 x float>", call)
        self.assertIn("<2 x i32> %c", call)


if __name__ == "__main__":
    unittest.main()
