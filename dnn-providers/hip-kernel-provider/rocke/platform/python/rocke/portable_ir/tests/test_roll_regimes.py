# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for regime specialization (roll_regimes.py): SEVERAL recipes covering one
# axis, when the kernel's structure changes partway along it.
#
# The property that matters is not "it splits". It is that it splits in exactly
# the right places -- no more (splitting a uniform axis throws away compression)
# and no fewer (merging across a real branch would produce a recipe that lies).
# So the tests pin both directions, plus the case where no split can help.
#
#   python3 -m unittest rocke.portable_ir.tests.test_roll_regimes

import unittest

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.portable_ir.src.roll_regimes import (
    legal_values,
    regime_report,
    roll_regimes,
)
from rocke.portable_ir.utils.recipe_expand import expand_recipe, recipes_equiv


def build_uniform(N):
    """Same shape everywhere: only the trip count moves. One recipe should do."""
    b = IRBuilder(f"uni_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.global_load_f32(A, tid, align=4)
    for i in range(N):
        acc = b.fadd(acc, b.global_load_f32(A, b.add(tid, b.const_i32(i)), align=4))
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


def build_branch(N, threshold=8):
    """A compile-time branch: past `threshold` the kernel emits a different op."""
    b = IRBuilder(f"br_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.global_load_f32(A, tid, align=4)
    for i in range(N):
        nxt = b.global_load_f32(A, b.add(tid, b.const_i32(i)), align=4)
        acc = b.fmul(acc, nxt) if N >= threshold else b.fadd(acc, nxt)
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


def build_every_value_different(N):
    """Structure changes at EVERY value -- no threshold exists to find."""
    b = IRBuilder(f"chaotic_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.global_load_f32(A, tid, align=4)
    for i in range(N):
        nxt = b.global_load_f32(A, b.add(tid, b.const_i32(i)), align=4)
        acc = b.fmul(acc, nxt) if (N + i) % 2 else b.fadd(acc, nxt)
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


class TestLegalValues(unittest.TestCase):
    def test_the_kernels_own_validation_decides_the_domain(self):
        """Legality comes from the spec, not from a range hardcoded in a driver."""

        class Spec:
            def __init__(self, block=64):
                if block % 32 or not 32 <= block <= 256:
                    raise ValueError(f"bad block {block}")
                self.block = block

        got = legal_values("block", range(8, 400, 8), lambda **kw: Spec(**kw))
        self.assertEqual(got, [32, 64, 96, 128, 160, 192, 224, 256])

    def test_a_tiny_legal_set_is_visible_before_any_rolling_effort(self):
        """The point of asking first: an axis with 2 values cannot repay a recipe."""

        class Spec:
            def __init__(self, head_size=128):
                if head_size not in (64, 128):
                    raise ValueError("head_size must be 64 or 128")

        self.assertEqual(legal_values("head_size", range(16, 513, 16), Spec), [64, 128])


class TestRegimes(unittest.TestCase):
    def test_a_uniform_axis_is_not_split(self):
        """Over-splitting is a silent cost: it throws compression away."""
        r = roll_regimes(build_uniform, axis="N", values=[2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(r.n_recipes, 1)
        self.assertEqual(r.regimes[0].values, [2, 3, 4, 5, 6, 7, 8])

    def test_a_compile_time_branch_splits_exactly_at_the_threshold(self):
        r = roll_regimes(build_branch, axis="N", values=[2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(r.n_recipes, 2)
        self.assertEqual(r.regimes[0].values, [2, 3, 4, 5, 6, 7])
        self.assertEqual(r.regimes[1].values, [8, 9, 10])
        self.assertTrue(all(x.rolled for x in r.regimes))

    def test_every_regime_reproduces_every_value_it_claims(self):
        """The whole point of specializing is to keep byte-exactness while doing it."""
        r = roll_regimes(build_branch, axis="N", values=[2, 3, 4, 5, 6, 7, 8, 9, 10])
        for reg in r.regimes:
            for v in reg.values:
                _, concrete = record_kernel(lambda: build_branch(v))
                exp = expand_recipe(reg.recipe, {"N": v})
                self.assertTrue(recipes_equiv(exp, concrete), f"N={v} mismatch")

    def test_a_moved_threshold_moves_the_boundary(self):
        """The split is discovered from traces, not from a constant in the roller."""
        r = roll_regimes(
            lambda v: build_branch(v, threshold=6),
            axis="N",
            values=[2, 3, 4, 5, 6, 7, 8],
        )
        self.assertEqual(r.regimes[0].values, [2, 3, 4, 5])
        self.assertEqual(r.regimes[1].values, [6, 7, 8])

    def test_an_axis_with_no_threshold_degrades_to_concrete(self):
        """When structure changes at every value, regimes cannot help -- say so."""
        r = roll_regimes(build_every_value_different, axis="N", values=[3, 4, 5, 6])
        self.assertEqual(r.n_rolled, 0)
        self.assertIn("concrete", regime_report(r))

    def test_lookup_finds_the_recipe_for_a_value(self):
        r = roll_regimes(build_branch, axis="N", values=[2, 3, 4, 8, 9, 10])
        self.assertIsNot(r.recipe_for(3), r.recipe_for(9))
        self.assertIs(r.recipe_for(2), r.recipe_for(4))
        self.assertIsNone(r.recipe_for(999))

    def test_two_values_are_required_to_infer_anything(self):
        with self.assertRaises(ValueError):
            roll_regimes(build_uniform, axis="N", values=[4])


if __name__ == "__main__":
    unittest.main()
