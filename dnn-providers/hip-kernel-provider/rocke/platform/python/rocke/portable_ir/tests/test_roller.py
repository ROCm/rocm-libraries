# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the scalable roller (roller.py / roll.py) and its device-free oracle
# (recipe_expand.py). The oracle replays a parametric recipe at concrete spec
# values and checks structural equivalence (modulo SSA renaming) to an
# independently-recorded concrete recipe -- the byte-identity proxy for HSACO.
#
#   python3 -m unittest rocke.portable_ir.tests.test_roller

import unittest

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.portable_ir.examples import export_mha, qk_block
from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe
from rocke.portable_ir.utils.recipe_expand import (
    equiv_reason,
    expand_recipe,
    recipes_equiv,
)
from rocke.portable_ir.src.roll import roll, roll_report


# --------------------------------------------------------------------------
# synthetic kernels (defined here; recording auto-discovers this module)
# --------------------------------------------------------------------------
def build_multi(N):
    """TWO loop-carries (sum, product) + TWO index ladders -- beyond the bespoke
    single-accumulator roller."""
    b = IRBuilder(f"multi_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    s = b.const_f32(0.0)
    p = b.const_f32(1.0)
    for i in range(N):
        off = b.const_i32(i * 4)
        off2 = b.const_i32(i * 4 + 100)
        x = b.global_load_f32(A, b.add(tid, off), align=4)
        y = b.global_load_f32(A, b.add(tid, off2), align=4)
        s = b.fadd(s, x)
        p = b.fmul(p, y)
    r = b.fadd(s, p)
    b.global_store(C, tid, r, align=4)
    b.ret()
    return b.kernel


def build_appearing(N):
    """A run whose trip count is `N-1`, so it is ABSENT at the smallest N.

    This is the shape of attention's softmax reduction over `block_n`: combining
    k values takes k-1 steps, so the run has zero copies at the smallest sample.
    A second, ordinary run follows it, so the two growing regions sit back to
    back -- the same layout the real kernel has."""
    b = IRBuilder(f"appear_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.global_load_f32(A, tid, align=4)
    for i in range(N - 1):
        acc = b.fmax(acc, b.global_load_f32(A, b.add(tid, b.const_i32(i)), align=4))
    acc = b.fmul(acc, acc)
    for i in range(N):
        acc = b.fadd(
            acc, b.global_load_f32(A, b.add(tid, b.const_i32(50 + i)), align=4)
        )
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


def build_two_runs(N):
    """TWO independent unrolled loops over N at one level -> exercises multi-run
    rolling (find first run, roll, recurse on the remainder)."""
    b = IRBuilder(f"two_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    a1 = b.const_f32(0.0)
    for i in range(N):
        a1 = b.fadd(a1, b.global_load_f32(A, b.add(tid, b.const_i32(i * 4)), align=4))
    m = b.fmul(a1, a1)
    a2 = b.const_f32(1.0)
    for j in range(N):
        a2 = b.fmul(a2, b.global_load_f32(A, b.add(m, b.const_i32(j * 8 + 7)), align=4))
    b.global_store(C, tid, b.fadd(a1, a2), align=4)
    b.ret()
    return b.kernel


def build_phased(N):
    """A GEMM-like phased fan: phase 1 produces per-lane partials, phase 2
    consumes them (INTER-RUN per-lane value flow), then a reduction epilogue
    over the results. Exercises output lane-refs + inter-run flow + lane-ref
    runs together."""
    from rocke.core.ir import I32

    b = IRBuilder(f"ph_{N}")
    C = b.param("C", PtrType(I32, "global"), writeonly=True, align=16)
    n = b.param("n", I32)
    tid = b.thread_id_x()
    lo = b.const_i32(0)
    st = b.const_i32(1)
    z = b.const_i32(0)
    loop = b.scf_for_iter(lo, n, st, [(f"acc{i}", z) for i in range(N)], iv_name="k")
    with loop as (k, accs):
        parts = [b.mul(k, b.const_i32(i * 3 + 1)) for i in range(N)]
        news = [b.add(accs[i], parts[i]) for i in range(N)]
        b.scf_yield(*news)
    s = loop.results[0]
    for i in range(1, N):
        s = b.add(s, loop.results[i])
    b.global_store(C, tid, s, align=4)
    b.ret()
    return b.kernel


def build_quad(N):
    """k = N*N is quadratic; a two-point linear fit must FAIL held-out
    verification -> graceful fallback (no roll)."""
    b = IRBuilder(f"quad_{N}")
    A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
    tid = b.thread_id_x()
    k = b.const_i32(N * N)
    acc = b.const_f32(0.0)
    for i in range(N):
        c = b.const_i32(i)
        x = b.global_load_f32(A, b.add(tid, b.add(c, k)), align=4)
        acc = b.fadd(acc, x)
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


class TestExpanderOracle(unittest.TestCase):
    def test_equiv_has_teeth(self):
        c64 = kerneldef_to_recipe(export_mha.build("fp16", 64, 2048, 1, 32, 1))
        c128 = kerneldef_to_recipe(export_mha.build("fp16", 128, 2048, 1, 32, 1))
        self.assertTrue(recipes_equiv(c64, c64))
        self.assertFalse(recipes_equiv(c64, c128))  # different unroll counts


class TestRollAttention(unittest.TestCase):
    def test_headsize_roll_and_holdout(self):
        r = roll(
            lambda D: export_mha.build("fp16", D, 2048, 1, 32, 1),
            axis="D",
            sample_points=[64, 128],
            holdout_points=[256, 192, 96, 512],
            spec_decl=[{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}],
            extra_spec={"dtype": "fp16"},
        )
        self.assertTrue(r.ok, r.reason)
        # one recipe expands byte-equivalently at an UNSAMPLED head dim
        exp = expand_recipe(r.recipe, {"D": 384, "dtype": "fp16"})
        con = kerneldef_to_recipe(export_mha.build("fp16", 384, 2048, 1, 32, 1))
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))


class TestRollQkBlock(unittest.TestCase):
    def test_qk_block_roll(self):
        r = roll(
            lambda D: qk_block.build_qk_block(D, "f16"),
            axis="D",
            sample_points=[64, 128],
            holdout_points=[256, 192, 320],
            spec_decl=[{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}],
            extra_spec={"dtype": "f16"},
        )
        self.assertTrue(r.ok, r.reason)


class TestRollGeneralization(unittest.TestCase):
    def test_multi_carry_and_ladders(self):
        r = roll(
            build_multi, axis="N", sample_points=[2, 3], holdout_points=[1, 5, 8, 16]
        )
        self.assertTrue(r.ok, r.reason)
        # spot-check a held-out N expands correctly
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, con = record_kernel(lambda: build_multi(7))
        exp = expand_recipe(r.recipe, {"N": 7})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_multiple_independent_runs(self):
        """Two separate unrolled loops over N at one level both roll (multi-run)."""
        r = roll(
            build_two_runs, axis="N", sample_points=[2, 3], holdout_points=[1, 4, 7, 11]
        )
        self.assertTrue(r.ok, r.reason)
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, con = record_kernel(lambda: build_two_runs(9))
        exp = expand_recipe(r.recipe, {"N": 9})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_inter_run_per_lane_flow(self):
        """A phased fan (per-lane partials produced in one run, consumed by the
        next) -> the GEMM-CShuffle dataflow shape (output lane-refs feeding a
        downstream run + lane-ref reduction)."""
        r = roll(
            build_phased, axis="N", sample_points=[2, 3], holdout_points=[1, 4, 5, 8]
        )
        self.assertTrue(r.ok, r.reason)
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, con = record_kernel(lambda: build_phased(7))
        exp = expand_recipe(r.recipe, {"N": 7})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_run_absent_at_the_smaller_sample_is_a_sample_choice_problem(self):
        """A run with trip count `N-1` has ZERO copies at N=1, and a run seen zero
        or one time carries no evidence of how one iteration feeds the next --
        `_roll_run` reads the loop-carry by diffing block 0 against block 1. So
        this is not a detector weakness; it is the samples. The refusal has to say
        so, because the obvious reading ("the detector missed a pattern") sends you
        off to improve the wrong thing."""
        bad = roll(build_appearing, axis="N", sample_points=[1, 2], holdout_points=[4])
        self.assertFalse(bad.ok)
        self.assertIn("ABSENT at the smaller axis value", bad.reason)
        self.assertIn("SAMPLE CHOICE", bad.reason)

    def test_the_same_kernel_rolls_once_the_run_is_sampled_twice(self):
        """The other half of the pair above: nothing about the kernel changed, only
        which points were recorded. At N=2,3 the run has 1 and 2 copies, which is
        the minimum that makes a loop carry visible."""
        r = roll(
            build_appearing,
            axis="N",
            sample_points=[2, 3],
            holdout_points=[4, 5, 9, 17],
        )
        self.assertTrue(r.ok, r.reason)
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, con = record_kernel(lambda: build_appearing(12))
        exp = expand_recipe(r.recipe, {"N": 12})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_advice_names_points_that_actually_work(self):
        """The suggestion is only worth printing if following it succeeds."""
        from rocke.portable_ir.src.roller import _sample_advice

        self.assertIn("[2, 3]", _sample_advice(1, 2))
        self.assertTrue(
            roll(build_appearing, axis="N", sample_points=[2, 3], holdout_points=[8]).ok
        )

    def test_nonlinear_falls_back(self):
        r = roll(build_quad, axis="N", sample_points=[2, 3], holdout_points=[4, 5])
        self.assertFalse(r.ok)
        self.assertIn("verify failed", r.reason)


def build_fan(L):
    """A variable loop-carry fan: a runtime scf.for carrying L accumulators (the
    T4 pattern, e.g. deep_fused_conv_pool's per-output-row accumulators). Lane
    count L is the structural axis."""
    from rocke.core.ir import I32

    b = IRBuilder(f"fan_{L}")
    C = b.param("C", PtrType(I32, "global"), writeonly=True, align=16)
    n = b.param("n", I32)
    tid = b.thread_id_x()
    lo = b.const_i32(0)
    st = b.const_i32(1)
    z = b.const_i32(0)
    loop = b.scf_for_iter(lo, n, st, [(f"acc{i}", z) for i in range(L)], iv_name="k")
    with loop as (k, accs):
        news = [b.add(accs[i], b.mul(k, b.const_i32(i))) for i in range(L)]
        b.scf_yield(*news)
    s = loop.results[0]
    for i in range(1, L):
        s = b.add(s, loop.results[i])
    b.global_store(C, tid, s, align=4)
    b.ret()
    return b.kernel


def build_fan_simple(L):
    """A clean variable loop-carry fan whose results are used directly (no
    reduction over results) -- the case the auto fan-roller handles today."""
    from rocke.core.ir import I32

    b = IRBuilder(f"fans_{L}")
    C = b.param("C", PtrType(I32, "global"), writeonly=True, align=16)
    n = b.param("n", I32)
    tid = b.thread_id_x()
    lo = b.const_i32(0)
    st = b.const_i32(1)
    z = b.const_i32(0)
    loop = b.scf_for_iter(lo, n, st, [(f"acc{i}", z) for i in range(L)], iv_name="k")
    with loop as (k, accs):
        news = [b.add(accs[i], b.mul(k, b.const_i32(i))) for i in range(L)]
        b.scf_yield(*news)
    b.global_store(C, tid, loop.results[0], align=4)
    b.ret()
    return b.kernel


def _parametric_fan_recipe(prologue, lob, stb, zb, tidb):
    lanes = {"for": {"var": "lane", "lo": 0, "hi": {"spec": "L"}, "step": 1}}
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": "fan_{L}",
        "spec": [{"name": "L", "kind": "int"}],
        "attrs": {},
        "program": prologue
        + [
            {
                "op": "scf_for",
                "iv": "k",
                "lo": lob,
                "hi": "n",
                "step": stb,
                "iter": [{**lanes, "name": "acc{lane}", "init": zb}],
                "results": [{**lanes, "name": "res{lane}"}],
                "unroll": False,
                "elide_trailing_barrier": True,
                "body": [
                    {
                        "op": "static_for",
                        "var": "lane",
                        "lo": 0,
                        "hi": {"spec": "L"},
                        "step": 1,
                        "body": [
                            {"op": "const_i32", "bind": "ci", "val": {"var": "lane"}},
                            {
                                "op": "emit",
                                "opcode": "arith.mul",
                                "in": ["k", "ci"],
                                "out": {"bind": "prod", "type": "i32"},
                            },
                            {
                                "op": "emit",
                                "opcode": "arith.add",
                                "in": ["acc{lane}", "prod"],
                                "out": {"bind": "new{lane}", "type": "i32"},
                            },
                        ],
                    },
                    {
                        "op": "emit",
                        "opcode": "scf.yield",
                        "in": [{**lanes, "name": "new{lane}"}],
                        "attrs": {"num": {"t": "i", "v": {"spec": "L"}}},
                    },
                ],
            },
            {"op": "alias", "bind": "S", "from": "res0"},
            {
                "op": "static_for",
                "var": "lane",
                "lo": 1,
                "hi": {"spec": "L"},
                "step": 1,
                "body": [
                    {
                        "op": "emit",
                        "opcode": "arith.add",
                        "in": ["S", "res{lane}"],
                        "out": {"bind": "sx", "type": "i32"},
                    },
                    {"op": "alias", "bind": "S", "from": "sx"},
                ],
            },
            {
                "op": "emit",
                "opcode": "memref.global_store_typed",
                "in": ["C", tidb, "S"],
                "attrs": {
                    "align": {"t": "i", "v": 4},
                    "elem_type": {"t": "s", "v": "i32"},
                },
            },
            {"op": "ret"},
        ],
    }


class TestParametricFanExpander(unittest.TestCase):
    """The expander's parametric scf.for (rolled iter-args/results + format names)
    reproduces a variable loop-carry fan byte-equivalently across lane counts --
    the representation the T4 auto-roller targets."""

    def test_fan_expands_across_lane_counts(self):
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, c2 = record_kernel(lambda: build_fan(2))
        pre = c2["program"][:6]
        param = _parametric_fan_recipe(
            pre,
            pre[3]["out"]["bind"],
            pre[4]["out"]["bind"],
            pre[5]["out"]["bind"],
            pre[2]["out"]["bind"],
        )
        for L in (2, 3, 4, 5, 8, 16):
            with self.subTest(L=L):
                _, con = record_kernel(lambda L=L: build_fan(L))
                exp = expand_recipe(param, {"L": L})
                self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_auto_roll_fan_with_reduction(self):
        """Fan + a reduction/epilogue over ALL results -> exercises lane-ref-aware
        runs (the CShuffle-over-results pattern), with the lane offset."""
        r = roll(
            build_fan, axis="L", sample_points=[2, 3], holdout_points=[1, 4, 5, 8, 16]
        )
        self.assertTrue(r.ok, r.reason)
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, con = record_kernel(lambda: build_fan(9))
        exp = expand_recipe(r.recipe, {"L": 9})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))

    def test_auto_roll_variable_fan(self):
        """The roller AUTO-detects a variable loop-carry fan (scf.for iter-arg
        arity scaling with the axis) and rolls iter-args + per-lane body + yield,
        verified at held-out lane counts."""
        r = roll(
            build_fan_simple,
            axis="L",
            sample_points=[2, 3],
            holdout_points=[1, 4, 5, 8, 16],
        )
        self.assertTrue(r.ok, r.reason)
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, con = record_kernel(lambda: build_fan_simple(11))
        exp = expand_recipe(r.recipe, {"L": 11})
        self.assertTrue(recipes_equiv(exp, con), equiv_reason(exp, con))


class TestRollGemmCShuffle(unittest.TestCase):
    """The real GEMM+CShuffle kernel rolls over tile_n via lane-label
    segmentation (cone + scratchpad), inter-phase per-lane flow, and type/attr
    parameterization (the smem buffer shape + sched-group counts scale)."""

    @staticmethod
    def _gemm(tn):
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
        )

        spec = UniversalGemmSpec(
            name=f"g{tn}",
            tile=TileSpec(16, tn, 16, 1, 1, 1, 16, 16, 16),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            data=DataSpec(),
            wave_size=64,
            block_size=64,
        )
        return build_universal_gemm(spec, arch="gfx950")

    def test_gemm_cshuffle_rolls(self):
        r = roll(
            self._gemm,
            axis="TN",
            sample_points=[32, 64],
            holdout_points=[128, 256],
            spec_decl=[{"name": "TN", "kind": "int"}],
        )
        self.assertTrue(r.ok, r.reason)
        from rocke.portable_ir.src.recording_builder import record_kernel

        for tn in (96, 192):  # unsampled multiples
            _, con = record_kernel(lambda tn=tn: self._gemm(tn))
            exp = expand_recipe(r.recipe, {"TN": tn})
            self.assertTrue(
                recipes_equiv(exp, con), f"tn={tn}: {equiv_reason(exp, con)}"
            )


class TestLaneAnalysis(unittest.TestCase):
    """The data-flow analyses that find loop-dependent (per-lane) actions:
    cone labeling separates shared vs per-lane by meaning, and scratchpad
    matching bridges the store->LDS->load memory hop."""

    def test_gemm_lane_labels(self):
        from rocke.portable_ir.src import roller
        from rocke.portable_ir.src.recording_builder import record_kernel

        _, rb = record_kernel(lambda: TestRollGemmCShuffle._gemm(64))
        sf = next(i for i in rb["program"] if i["op"] == "scf_for")
        body = sf["body"]
        labels = roller.lane_label_body(body, body[-1]["in"], 4)
        # The A-tile chain is shared; each B-tile chain is its own lane; the
        # side-effecting B stores inherit their lane through LDS memory.
        by_op = {}
        for j, instr in enumerate(body):
            op = instr.get("opcode", "")
            if op in (
                "memref.global_load_vN",
                "tile.smem_store_vN",
                "tile.smem_load_v4",
                "tile.mma",
            ):
                by_op.setdefault(op, []).append(labels.get(j))
        # 4 mma -> lanes 0..3
        self.assertEqual(by_op["tile.mma"], [0, 1, 2, 3])
        # smem loads: 1 shared (A) + 4 per-lane (B)
        self.assertEqual(
            sorted(str(x) for x in by_op["tile.smem_load_v4"]),
            ["0", "1", "2", "3", "S"],
        )
        # B stores (side effects) inherit lanes via the scratchpad edge
        self.assertEqual(
            sorted(str(x) for x in by_op["tile.smem_store_vN"]),
            ["0", "1", "2", "3", "S"],
        )


class TestRollCoverageTiers(unittest.TestCase):
    """Tiered roll status. Rolling is safe-by-construction: roll() only returns
    a recipe that the oracle verified at sampled AND held-out shapes, so every
    ok result is byte-equivalent and fallbacks are 'not compressed', never wrong."""

    def test_coverage_runs_and_t3_rolls(self):
        from rocke.portable_ir.drivers.roll_coverage import run_coverage

        rows = {r["tier"]: r for r in run_coverage()}
        for tier in ("T1", "T2", "T3", "T4"):
            self.assertIn(tier, rows)
            self.assertIsNone(rows[tier]["error"], rows[tier]["report"])
        # T1 (small op) and T3 (the production unified-attention 2D) must roll.
        self.assertTrue(rows["T1"]["ok"], rows["T1"]["report"])
        self.assertTrue(rows["T3"]["ok"], rows["T3"]["report"])


class TestRollDelegatesToRollNd(unittest.TestCase):
    """`roll` is a thin wrapper over `roll_nd` with one axis.

    Worth pinning because the wrapper is what makes the two entry points agree on
    safety. When `roll` had its own implementation it skipped the kernel-name
    check, so the convenient API was the weak one: a recipe whose symbol did not
    track the axis passed every check `roll` made and shipped the wrong symbol.
    Re-splitting them would quietly reopen that hole, and nothing else would fail.
    """

    def _build_derived(self, S):
        """Named after a DERIVED quantity, so the name cannot be reconstructed by
        substituting the axis -- `derived_g7` at S=2, `derived_g10` at S=3."""
        b = IRBuilder(f"derived_g{S * 3 + 1}")
        A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
        C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
        tid = b.thread_id_x()
        b.global_store(
            C, tid, b.global_load_f32(A, b.add(tid, b.const_i32(S)), align=4), align=4
        )
        b.ret()
        return b.kernel

    def test_roll_refuses_a_name_it_cannot_reconstruct(self):
        r = roll(
            self._build_derived, axis="S", sample_points=[2, 3], holdout_points=[9]
        )
        self.assertFalse(r.ok, "roll accepted a recipe that emits the wrong symbol")
        self.assertIn("derived quantity", r.reason)

    def test_roll_still_returns_traces_keyed_by_axis_value(self):
        """The wrapper must not leak roll_nd's tuple-keyed points to callers."""

        def build(N):
            b = IRBuilder(f"uni_{N}")
            A = b.param("A", PtrType(F32, "global"), readonly=True, align=16)
            C = b.param("C", PtrType(F32, "global"), writeonly=True, align=16)
            tid = b.thread_id_x()
            acc = b.global_load_f32(A, tid, align=4)
            for i in range(N):
                acc = b.fadd(
                    acc, b.global_load_f32(A, b.add(tid, b.const_i32(i)), align=4)
                )
            b.global_store(C, tid, acc, align=4)
            b.ret()
            return b.kernel

        r = roll(build, axis="N", sample_points=[2, 3], holdout_points=[7])
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(sorted(r.traces), [2, 3, 7])
        self.assertIn("3 shapes", roll_report(r))


if __name__ == "__main__":
    unittest.main()
