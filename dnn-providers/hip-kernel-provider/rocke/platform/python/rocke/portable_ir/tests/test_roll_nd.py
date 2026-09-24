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

import json
import math
import unittest

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.portable_ir.src.roll_nd import roll_nd
from rocke.portable_ir.src.roller import (
    affine_intexpr,
    affine_solve,
    fit_slot,
    merge_intexpr,
)
from rocke.portable_ir.utils.recipe_expand import (
    equiv_reason,
    eval_intexpr,
    expand_recipe,
    magic_division_constants,
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


def build_magic(C):
    """A kernel that divides by the axis, via the DSL's own strength reduction.

    The emitted constants are a `ceil(log2 C)` shift and a multiplier keyed on
    C's odd part, so no polynomial in C fits either. Uses the production helper
    rather than hand-written numbers, so the test tracks the real idiom."""
    from rocke.helpers.transforms import calculate_magic_numbers, do_magic_division

    b = IRBuilder(f"magic_C{C}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    Out = b.param("Out", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    mult, shift = calculate_magic_numbers(C)
    q = do_magic_division(b, tid, mult, shift)
    b.global_store(Out, q, b.global_load_f32(A, q, align=4), align=4)
    b.ret()
    return b.kernel


def build_reciprocal(block):
    """A block COUNT that is reciprocal in the block SIZE (`512 // block`) --
    the shape of constant attention emits for its KV loop."""
    b = IRBuilder(f"recip_b{block}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    Out = b.param("Out", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    x = b.global_load_f32(A, b.add(tid, b.const_i32(512 // block)), align=4)
    b.global_store(Out, b.add(tid, b.const_i32(block)), x, align=4)
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
        this must be caught here or a wrong symbol ships. Single-axis `roll` now
        inherits the check by delegating here; its own test pins that."""

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


class TestBinarySearchTripCount(unittest.TestCase):
    """`ceil(log2(n+1))` -- the trip count of a binary search over `n` items.

    Found by sweeping kernels/gfx950: all three tiled attention kernels size their
    sequence-lookup loop this way, and `num_seqs` refused for want of the model
    even though the recipe schema could already express it. `magic_shift` computes
    `ceil(log2 x)` and both evaluators recurse into its operand, so the fix was to
    hypothesise the offset, not to add a primitive."""

    def _iters(self, n):
        return max(1, int(math.ceil(math.log2(n + 1))))

    def test_the_offset_form_is_recognised(self):
        pts = [(16,), (32,), (48,)]
        expr = fit_slot(["num_seqs"], pts, [self._iters(p[0]) for p in pts])[0]
        self.assertEqual(expr, {"magic_shift": {"add": [{"spec": "num_seqs"}, 1]}})

    def test_it_extrapolates_far_past_the_samples(self):
        """The point of a parameter-free model: fitted on three points, right on
        values nowhere near them."""
        pts = [(16,), (32,), (48,)]
        expr = fit_slot(["num_seqs"], pts, [self._iters(p[0]) for p in pts])[0]
        for n in (1, 7, 64, 100, 511, 512, 4096, 100000):
            self.assertEqual(
                eval_intexpr(expr, {}, {"num_seqs": n}, {}), self._iters(n), f"n={n}"
            )

    def test_a_plain_shift_is_not_explained_as_an_offset_one(self):
        """Offset 0 is tried first, so the exact shift keeps the simpler form."""
        pts = [(24,), (48,), (96,)]
        vals = [magic_division_constants(p[0])[1] for p in pts]
        self.assertEqual(fit_slot(["d"], pts, vals)[0], {"magic_shift": {"spec": "d"}})

    def test_an_unrelated_logarithm_is_still_refused(self):
        """Two offsets is a small hypothesis space on purpose; it must not become a
        general log fitter that explains anything vaguely logarithmic."""
        pts = [(16,), (32,), (48,)]
        expr, why = fit_slot(["n"], pts, [self._iters(p[0]) + 3 for p in pts])
        self.assertIsNone(expr)
        self.assertIn("no candidate model", why)


class TestCandidateModels(unittest.TestCase):
    """The non-affine candidates, at the level of `fit_slot` and end to end.

    Affine covers what a kernel computes FROM a shape. These cover constants a
    code generator CHOSE given a shape, which follow their own generating rule."""

    def test_magic_constants_match_the_dsl_helper(self):
        """Three implementations have to agree: the DSL emits these, the Python
        expander regenerates them, and the C VM mirrors the expander. Pin the two
        Python ones against each other here; the C VM is pinned by
        `test_recipe_roller.py::test_standalone_cli_regenerates_magic_division_constants`.
        """
        from rocke.helpers.transforms import calculate_magic_numbers

        for d in [1, 2, 3, 5, 7, 8, 9, 64, 96, 128, 160, 192, 224, 256, 384, 1000]:
            mult, shift = calculate_magic_numbers(d)
            want = (mult - (1 << 32) if mult >= (1 << 31) else mult, shift)
            self.assertEqual(magic_division_constants(d), want, f"divisor {d}")

    def test_magic_constants_reproduce_the_division(self):
        """The whole point of the pair: it computes `n // d` exactly."""
        for d in (3, 64, 96, 160, 224):
            m, s = magic_division_constants(d)
            m &= 0xFFFFFFFF  # umul_hi reads the bit pattern as unsigned
            for n in (1, 7, 100, 1000, 12345, 999983, 2**31):
                self.assertEqual((((n * m) >> 32) + n) >> s, n // d, f"{n}//{d}")

    def test_fit_slot_prefers_the_simplest_exact_model(self):
        """Candidates are tried simplest-first, so a slot that IS affine never
        gets described as something more exotic."""
        pts = [(64,), (96,), (128,)]
        expr, why = fit_slot(["C"], pts, [128, 192, 256])
        self.assertEqual(expr, {"mul": [{"spec": "C"}, 2]}, why)

    def test_fit_slot_finds_magic_operands(self):
        pts = [(64,), (96,), (128,)]
        shifts = [magic_division_constants(p[0])[1] for p in pts]
        mults = [magic_division_constants(p[0])[0] for p in pts]
        self.assertEqual(
            fit_slot(["C"], pts, shifts)[0], {"magic_shift": {"spec": "C"}}
        )
        self.assertEqual(
            fit_slot(["C"], pts, mults)[0], {"magic_multiplier": {"spec": "C"}}
        )

    def test_fit_slot_finds_a_reciprocal(self):
        pts = [(32,), (64,), (128,)]
        expr, why = fit_slot(["b"], pts, [16, 8, 4])
        self.assertEqual(expr, {"div": [512, {"spec": "b"}]}, why)

    def test_unit_fraction_slope_allows_a_non_zero_intercept(self):
        """`x/d + k` is expressible, so refusing it when `k != 0` cost coverage for
        nothing. `div` floors either way, so two samples are a guess with or
        without an intercept, and both are settled by the held-out points."""
        from rocke.portable_ir.src.roller import _linear_expr

        self.assertEqual(
            _linear_expr("w", 4, 3, 8, 4),
            {"add": [{"div": [{"spec": "w"}, 4]}, 2]},
        )

    def test_fit_slot_still_refuses_the_unexplainable(self):
        """A candidate library must not become 'fit anything'. A quadratic is not
        in the hypothesis class, and saying so is the correct answer."""
        pts = [(2,), (3,), (5,)]
        expr, why = fit_slot(["N"], pts, [4, 9, 25])
        self.assertIsNone(expr)
        self.assertIn("fits no candidate model", why)

    def test_magic_axis_rolls_at_held_out_odd_parts(self):
        """End to end: the axis that no curve fits. The holdouts have odd parts
        3, 5 and 7, whose multipliers share no value with any sampled point."""
        r = roll_nd(
            build_magic,
            axes={"C": [64, 96, 128]},
            holdout_points=[{"C": 192}, {"C": 160}, {"C": 224}, {"C": 384}],
        )
        self.assertTrue(r.ok, r.reason)
        prog = json.dumps(r.recipe["program"])
        self.assertIn("magic_multiplier", prog)
        self.assertIn("magic_shift", prog)

    def test_magic_axis_sampled_only_on_powers_of_two_is_refused(self):
        """The sampling trap, pinned. Every power of two has multiplier 1, so the
        slot looks INVARIANT and gets frozen -- inference cannot know it is being
        starved, and only a non-power-of-two holdout exposes it."""
        r = roll_nd(build_magic, axes={"C": [64, 128]}, holdout_points=[{"C": 96}])
        self.assertFalse(r.ok)
        self.assertIn("verify failed", r.reason)

    def test_reciprocal_axis_rolls(self):
        r = roll_nd(
            build_reciprocal,
            axes={"block": [32, 64, 128]},
            holdout_points=[{"block": 256}, {"block": 512}],
        )
        self.assertTrue(r.ok, r.reason)
        self.assertIn('"div": [512', json.dumps(r.recipe["program"]))

    def test_two_samples_cannot_separate_a_line_from_a_reciprocal(self):
        """Ambiguity is resolved by ORDER, so ordering can pick the wrong model.

        `512 div b` at b=32,64 gives 16,8 -- and the line `24 - b/4` passes through
        both. Two samples contain no evidence to prefer either; simplest-first
        means the line would win whenever it is expressible. Here it is not (a
        non-unit fraction), so the reciprocal is reached and the roll succeeds --
        but the general lesson is that a third sample, not a cleverer solver, is
        what makes this unambiguous."""
        two = fit_slot(["b"], [(32,), (64,)], [16, 8])[0]
        self.assertEqual(two, {"div": [512, {"spec": "b"}]})
        # the line is genuinely exact on those two points
        self.assertEqual([24 - 32 // 4, 24 - 64 // 4], [16, 8])
        # a third sample rejects it outright: 24 - 128/4 = -8, not 4
        self.assertIsNone(affine_solve([(32,), (64,), (128,)], [16, 8, 4]))
        three = fit_slot(["b"], [(32,), (64,), (128,)], [16, 8, 4])[0]
        self.assertEqual(three, {"div": [512, {"spec": "b"}]})

    def test_cross_term_fits_once_an_interior_point_is_recorded(self):
        """The same kernel that must be REFUSED without holdouts (above) rolls
        once escalation is permitted, and the recipe says `S*T` outright."""
        r = roll_nd(
            build_cross,
            axes={"S": [8, 16], "T": [64, 128]},
            holdout_points=[{"S": 32, "T": 256}, {"S": 24, "T": 192}],
        )
        self.assertTrue(r.ok, r.reason)
        self.assertIn(
            '{"mul": [{"spec": "S"}, {"spec": "T"}]}', json.dumps(r.recipe["program"])
        )
        # escalation costs one extra recording, not the whole grid
        self.assertEqual(r.n_recorded, 4)

    def test_cross_term_without_holdouts_says_why(self):
        """Fitting products consumes grid points that verification relies on, so
        refuse rather than silently weaken the evidence."""
        r = roll_nd(build_cross, axes={"S": [8, 16], "T": [64, 128]})
        self.assertFalse(r.ok)
        self.assertIn("holdout_points", r.reason)


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
