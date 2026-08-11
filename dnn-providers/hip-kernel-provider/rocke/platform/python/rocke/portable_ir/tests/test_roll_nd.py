# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for multi-axis rolling (roll_nd.py): ONE recipe covering a CROSS PRODUCT
# of non-reduction axes, rather than one recipe per axis.
#
# The interesting cases are the refusals. A cross term (a constant that scales
# with S*T) fits one-axis-at-a-time probes *perfectly* -- three points in general
# position determine an affine model -- and only shows up as wrong at an interior
# point of the grid. So these tests check the verification sweep has teeth there,
# not just at the extrapolated holdouts.
#
#   python3 -m unittest rocke.portable_ir.tests.test_roll_nd

import unittest

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.portable_ir.src.roll_nd import roll_nd
from rocke.portable_ir.src.roller import affine_intexpr, affine_solve, merge_intexpr
from rocke.portable_ir.utils.recipe_expand import (
    equiv_reason,
    expand_recipe,
    recipes_equiv,
)


# --------------------------------------------------------------------------
# synthetic kernels (recording auto-discovers this module)
# --------------------------------------------------------------------------
def build_shape(S, T):
    """Structure fixed; TWO shape axes move constants only. The 'constants only'
    shape of five of the seven real gated axes."""
    b = IRBuilder(f"shape_S{S}_T{T}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    x = b.global_load_f32(A, b.add(tid, b.const_i32(S)), align=4)
    y = b.global_load_f32(A, b.add(tid, b.const_i32(2 * T + 5)), align=4)
    b.global_store(C, b.add(tid, b.const_i32(S + 3 * T)), b.fadd(x, y), align=4)
    b.ret()
    return b.kernel


def build_cross(S, T):
    """A constant that scales with the PRODUCT of the two axes -- affine in
    neither, but consistent with any one-axis-at-a-time probe set."""
    b = IRBuilder(f"cross_S{S}_T{T}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    x = b.global_load_f32(A, b.add(tid, b.const_i32(S * T)), align=4)
    b.global_store(C, tid, x, align=4)
    b.ret()
    return b.kernel


def build_mixed(N, S):
    """A STRUCTURAL axis (N unrolls a run) plus a shape axis (S moves constants),
    including a base offset that depends on BOTH -- the case that forces the
    structural merge to reconcile already-parametric intexpr trees rather than
    plain integers."""
    b = IRBuilder(f"mixed_N{N}_S{S}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    base = b.add(tid, b.const_i32(N * 8 + S * 2))
    acc = b.const_f32(0.0)
    for i in range(N):
        acc = b.fadd(
            acc, b.global_load_f32(A, b.add(base, b.const_i32(i * 4)), align=4)
        )
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


def build_ladder_shape(N, S):
    """A structural axis whose per-iteration LADDER constant also depends on a
    shape axis (`i*4 + S`). Documents a composition limit: after annotation the
    constant is an intexpr, so the ladder step (which looks for plain ints) skips
    it and the roll is refused."""
    b = IRBuilder(f"ls_N{N}_S{S}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.const_f32(0.0)
    for i in range(N):
        acc = b.fadd(
            acc, b.global_load_f32(A, b.add(tid, b.const_i32(i * 4 + S)), align=4)
        )
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


def build_tail_after_run(N, S):
    """A run followed by a tail whose leading ops REPEAT the run block's
    signatures in a rotated order (`load,fadd,const,add`), so the only run
    candidate the detector offers starts mid-block and swallows the tail's
    constant. Kept as a test so this stays a documented refusal."""
    b = IRBuilder(f"tail_N{N}_S{S}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.const_f32(0.0)
    for i in range(N):
        acc = b.fadd(acc, b.global_load_f32(A, b.add(tid, b.const_i32(i * 4)), align=4))
    b.global_store(C, b.add(tid, b.const_i32(N * 8 + S * 2)), acc, align=4)
    b.ret()
    return b.kernel


def build_div(S):
    """A count expressed as S//8 -- a unit-fraction coefficient, which must come
    out as a `div` intexpr rather than being refused."""
    b = IRBuilder(f"div_S{S}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    x = b.global_load_f32(A, b.add(tid, b.const_i32(S // 8)), align=4)
    b.global_store(C, tid, x, align=4)
    b.ret()
    return b.kernel


class TestAffineSolver(unittest.TestCase):
    def test_exact_fit_and_rejection(self):
        pts = [(8, 64), (16, 64), (8, 128)]
        sol = affine_solve(
            pts, [5 + 2 * 8 + 3 * 64, 5 + 2 * 16 + 3 * 64, 5 + 2 * 8 + 3 * 128]
        )
        self.assertIsNotNone(sol)
        self.assertEqual([int(c) for c in sol], [5, 2, 3])
        self.assertEqual(
            affine_intexpr(["S", "T"], sol),
            {
                "add": [
                    {"add": [{"mul": [{"spec": "S"}, 2]}, {"mul": [{"spec": "T"}, 3]}]},
                    5,
                ]
            },
        )
        # inconsistent system (4 points, no affine model) -> None
        self.assertIsNone(affine_solve(pts + [(16, 128)], [1, 2, 3, 99]))

    def test_unit_fraction_becomes_div(self):
        sol = affine_solve([(64,), (128,)], [8, 16])
        self.assertEqual(affine_intexpr(["S"], sol), {"div": [{"spec": "S"}, 8]})

    def test_non_unit_fraction_refused(self):
        # v = 3S/8 fits exactly but floor division cannot express it.
        sol = affine_solve([(64,), (128,)], [24, 48])
        self.assertIsNotNone(sol)
        self.assertIsNone(affine_intexpr(["S"], sol))

    def test_merge_intexpr_fits_leaves(self):
        # same tree shape, one differing leaf -> fitted in the structural axis
        a = {"add": [{"mul": [{"spec": "S"}, 2]}, 16]}
        b = {"add": [{"mul": [{"spec": "S"}, 2]}, 32]}
        self.assertEqual(
            merge_intexpr(a, b, "N", 2, 4),
            {"add": [{"mul": [{"spec": "S"}, 2]}, {"mul": [{"spec": "N"}, 8]}]},
        )
        # differing shapes cannot be merged
        self.assertIsNone(
            merge_intexpr({"spec": "S"}, {"mul": [{"spec": "S"}, 2]}, "N", 2, 4)
        )


class TestRollNdShapeAxes(unittest.TestCase):
    def test_two_shape_axes_cover_cross_product(self):
        r = roll_nd(
            build_shape,
            axes={"S": [8, 16], "T": [64, 128]},
            holdout_points=[{"S": 32, "T": 256}, {"S": 1, "T": 7}],
        )
        self.assertTrue(r.ok, r.reason)
        # 4 grid points + 2 holdouts verified, from only 3 recorded traces
        self.assertEqual(len(r.points), 6)
        self.assertEqual(r.n_recorded, 3)
        # and it extrapolates to a point no one asked about
        _, con = record_kernel(lambda: build_shape(99, 5))
        exp = expand_recipe(r.recipe, {"S": 99, "T": 5})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_div_axis(self):
        r = roll_nd(build_div, axes={"S": [64, 128]}, holdout_points=[{"S": 512}])
        self.assertTrue(r.ok, r.reason)

    def test_cross_term_refused_at_interior_point(self):
        """S*T fits every one-axis probe, so this must be caught by the grid sweep."""
        r = roll_nd(build_cross, axes={"S": [8, 16], "T": [64, 128]})
        self.assertFalse(r.ok)
        self.assertIn("verify failed", r.reason)
        # specifically at the interior point, not a holdout (there are none here)
        self.assertIn("'S': 16", r.reason)
        self.assertIn("'T': 128", r.reason)

    def test_name_must_be_reconstructible(self):
        """A kernel name that does not carry an axis is refused rather than
        emitted with a wrong symbol at every other point."""

        def build_badname(S):
            b = IRBuilder("fixed_name")  # name ignores S
            A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
            C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
            tid = b.thread_id_x()
            b.global_store(
                C,
                tid,
                b.global_load_f32(A, b.add(tid, b.const_i32(S)), align=4),
                align=4,
            )
            b.ret()
            return b.kernel

        r = roll_nd(build_badname, axes={"S": [8, 16]})
        # the name is constant, so it is reconstructible; the recipe is valid
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.recipe["kernel_name_fmt"], "fixed_name")

    def test_name_carrying_a_derived_quantity_is_refused(self):
        """A name built from a DERIVED quantity (3S+1) cannot be reconstructed by
        substitution, and the program oracle cannot see kernel names at all -- so
        this must be caught here or a wrong symbol ships. (Single-axis `roll` does
        NOT catch it; see the scaling plan's pitfalls.)"""

        def build_derived(S):
            b = IRBuilder(f"derived_g{S * 3 + 1}")
            A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
            C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
            tid = b.thread_id_x()
            b.global_store(
                C,
                tid,
                b.global_load_f32(A, b.add(tid, b.const_i32(S)), align=4),
                align=4,
            )
            b.ret()
            return b.kernel

        r = roll_nd(build_derived, axes={"S": [2, 3]})
        self.assertFalse(r.ok)
        self.assertIn("derived quantity", r.reason)
        # and the escape hatch works: state the format explicitly
        r2 = roll_nd(build_derived, axes={"S": [2, 3]}, name_fmt="derived_g{g}")
        self.assertFalse(r2.ok)  # {g} is not an axis, so it still cannot verify

    def test_point_must_be_a_dict_not_a_bare_value(self):
        """`roll` takes bare axis values, `roll_nd` takes a dict per point. The
        mix-up is reported as such instead of failing inside verification."""
        with self.assertRaises(ValueError) as cm:
            roll_nd(
                build_shape, axes={"S": [8, 16], "T": [64, 128]}, holdout_points=[32]
            )
        self.assertIn("single-axis form", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            roll_nd(
                build_shape,
                axes={"S": [8, 16], "T": [64, 128]},
                holdout_points=[{"S": 32}],
            )
        self.assertIn("missing axes ['T']", str(cm.exception))


class TestRollNdMixed(unittest.TestCase):
    def test_structural_plus_shape_axis(self):
        """A structural axis rolled to a static_for WHILE a shape axis moves
        constants -- including one constant that depends on both."""
        r = roll_nd(
            build_mixed,
            axes={"N": [2, 3], "S": [8, 16]},
            structural_axis="N",
            holdout_points=[{"N": 7, "S": 32}, {"N": 5, "S": 3}],
        )
        self.assertTrue(r.ok, r.reason)
        prog = r.recipe["program"]
        self.assertTrue(
            any(i.get("op") == "static_for" for i in prog),
            "the structural axis should still compress to a static_for:\n"
            + "\n".join(str(i.get("op")) for i in prog),
        )
        _, con = record_kernel(lambda: build_mixed(9, 5))
        exp = expand_recipe(r.recipe, {"N": 9, "S": 5})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_shape_axis_inside_the_ladder_refuses(self):
        """The mixed case composes only while the shape axis stays OUT of the
        unrolled block's per-iteration constants. Inside, the ladder inference
        skips the (now parametric) constant and the oracle rejects the result --
        a missed roll, never a wrong one."""
        r = roll_nd(
            build_ladder_shape,
            axes={"N": [2, 3], "S": [8, 16]},
            structural_axis="N",
            holdout_points=[{"N": 5, "S": 32}],
        )
        self.assertFalse(r.ok)
        self.assertIn("verify failed", r.reason)

    def test_structural_axis_must_be_declared(self):
        """Left undeclared, a structural axis is reported as such rather than
        silently mis-modeled as a constant axis."""
        r = roll_nd(build_mixed, axes={"N": [2, 3], "S": [8, 16]})
        self.assertFalse(r.ok)
        self.assertIn("not constants-only", r.reason)

    def test_rotated_run_boundary_refuses_rather_than_mis_rolls(self):
        """A pre-existing run-detection limitation, pinned here because a WRONG
        roll would be far worse than a missed one.

        When the tail after a run repeats the block's signatures in a rotated
        order, the only candidate the detector offers starts mid-block, so the
        inferred ladder covers a constant that belongs to the tail. The oracle
        catches it and the roll is declined. Single-axis `roll` refuses the same
        shape identically, so this is not specific to multi-axis rolling; fixing
        it means teaching `_run_candidates` about block phase."""
        r = roll_nd(
            build_tail_after_run,
            axes={"N": [2, 3], "S": [8, 16]},
            structural_axis="N",
        )
        self.assertFalse(r.ok)
        self.assertIn("verify failed", r.reason)


if __name__ == "__main__":
    unittest.main()
