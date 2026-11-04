# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Drift guard for RecordingIRBuilder vs IRBuilder.
#
# RecordingIRBuilder records a recipe by intercepting IRBuilder's op emission
# (`_emit`) and region management (`push_region`/`pop_region`). If a future
# IRBuilder change routes ops around `_emit`, alters region nesting, or changes
# the Op/Param/Value/Region structure, the *live-recorded* recipe will no longer
# match the *built* KernelDef and these tests fail -- alerting developers that
# RecordingIRBuilder must be updated.
#
# The oracle (`_kernel_to_program`) is an INDEPENDENT post-hoc walk of the final
# KernelDef. Comparing it to the live recording verifies the interception
# (region routing, nothing dropped/added) stays faithful. A separate op-count
# check guards against silently dropped/extra ops even if the per-op shape were
# wrong.
#
#   python3 -m unittest rocke.portable_ir.tests.test_recording_builder

import unittest

from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.core.ir_print import print_ir
from rocke.portable_ir.src.recording_builder import (
    RecordingIRBuilder,
    kernel_to_recipe,
)


# --------------------------------------------------------------------------
# Builder-agnostic kernels (take a builder `b`), exercising the op surface.
# --------------------------------------------------------------------------
def build_scalar(b):
    c = b.const_i32(1)
    b.add(c, c)
    b.ret()


def build_memory(b):
    A = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    v = b.global_load_f32(A, tid, align=4)
    b.global_store(C, tid, v, align=4)
    b.ret()


def build_vector(b):
    C = b.param("C", PtrType(F16, "global"))
    s = b.const_f32(2.0)
    v = b.vector_splat(s, 4)
    w = b.vector_add(v, v)
    h = b.vec_trunc_f32_to_f16(w)
    e = b.vec_extract(h, 0)
    tid = b.thread_id_x()
    b.store_f16(C, tid, e)
    b.ret()


def build_forloop(b):
    C = b.param("C", PtrType(F32, "global"))
    lo = b.const_i32(0)
    hi = b.const_i32(16)
    st = b.const_i32(1)
    acc0 = b.const_f32(0.0)
    loop = b.scf_for_iter(
        lo,
        hi,
        st,
        [("acc", acc0)],
        iv_name="k0",
        unroll=False,
        elide_trailing_barrier=True,
    )
    with loop as (k0, (acc,)):
        one = b.const_f32(1.0)
        n = b.fadd(acc, one)
        b.scf_yield(n)
    tid = b.thread_id_x()
    b.global_store(C, tid, loop.results[0], align=4)
    b.ret()


def build_scf_if(b):
    C = b.param("C", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    z = b.const_i32(0)
    cond = b.cmp_gt(n, z)
    guard = b.scf_if(cond)
    with guard:
        v = b.const_f32(1.0)
        b.global_store(C, tid, v, align=4)
    b.ret()


def build_nested(b):
    """scf.for containing an scf.if -- exercises region nesting."""
    C = b.param("C", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    lo = b.const_i32(0)
    st = b.const_i32(1)
    acc0 = b.const_f32(0.0)
    loop = b.scf_for_iter(lo, n, st, [("acc", acc0)], iv_name="k")
    with loop as (k, (acc,)):
        z = b.const_i32(0)
        cond = b.cmp_gt(k, z)
        guard = b.scf_if(cond)
        with guard:
            one = b.const_f32(1.0)
            b.global_store(C, tid, one, align=4)
        b.scf_yield(acc)
    b.global_store(C, tid, loop.results[0], align=4)
    b.ret()


def build_if_else(b):
    """scf.if_else with values defined (and used) inside each arm, plus a
    barrier in both arms -- the reason the op exists."""
    A = b.param("A", PtrType(F32, "global"))
    C = b.param("C", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    cond = b.cmp_lt(tid, n)
    with b.scf_if_else(cond) as (then_ctx, else_ctx):
        with then_ctx:
            v = b.global_load_f32(A, tid, align=4)
            two = b.const_f32(2.0)
            w = b.fmul(v, two)
            b.global_store(C, tid, w, align=4)
            b.sync()
        with else_ctx:
            z = b.const_f32(0.0)
            b.global_store(C, tid, z, align=4)
            b.sync()
    b.ret()


def build_if_else_nested(b):
    """if_else inside the else arm of another if_else, with an scf.if inside the
    then arm: arm routing must survive nesting in both directions."""
    C = b.param("C", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    z = b.const_i32(0)
    outer = b.cmp_lt(tid, n)
    with b.scf_if_else(outer) as (then_ctx, else_ctx):
        with then_ctx:
            pos = b.cmp_gt(tid, z)
            with b.scf_if(pos):
                one = b.const_f32(1.0)
                b.global_store(C, tid, one, align=4)
        with else_ctx:
            twice = b.add(n, n)
            inner = b.cmp_lt(tid, twice)
            with b.scf_if_else(inner) as (t2, e2):
                with t2:
                    two = b.const_f32(2.0)
                    b.global_store(C, tid, two, align=4)
                with e2:
                    three = b.const_f32(3.0)
                    b.global_store(C, tid, three, align=4)
    b.ret()


def build_if_else_in_for(b):
    """scf.if_else inside an scf.for body, using the loop's iv and carry."""
    C = b.param("C", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    lo = b.const_i32(0)
    st = b.const_i32(1)
    acc0 = b.const_f32(0.0)
    loop = b.scf_for_iter(lo, n, st, [("acc", acc0)], iv_name="k")
    with loop as (k, (acc,)):
        odd = b.cmp_lt(k, tid)
        with b.scf_if_else(odd) as (then_ctx, else_ctx):
            with then_ctx:
                b.global_store(C, k, acc, align=4)
            with else_ctx:
                one = b.const_f32(1.0)
                bumped = b.fadd(acc, one)
                b.global_store(C, k, bumped, align=4)
        b.scf_yield(acc)
    b.global_store(C, tid, loop.results[0], align=4)
    b.ret()


def build_inline_asm(b):
    C = b.param("C", PtrType(F32, "global"))
    tid = b.thread_id_x()
    x = b.const_f32(1.0)
    r = b.inline_asm("v_mov_b32 $0, $1", "=v,v", [x], result_type=F32)
    b.global_store(C, tid, r, align=4)
    b.ret()


def build_inline_asm_multi(b):
    C = b.param("C", PtrType(F32, "global"))
    tid = b.thread_id_x()
    a = b.const_i32(7)
    outs = b.inline_asm_multi(
        "v_swap_b32 $0, $1, $2, $2", "=v,=v,v", [a], result_types=[I32, I32]
    )
    # touch a result so it's not dead; recorder cares only about emission.
    _ = outs[0]
    b.ret()


def build_if_else_results(b):
    """scf.if_else that yields a value; portable IR refuses to carry it."""
    C = b.param("C", PtrType(F32, "global"))
    n = b.param("n", I32)
    tid = b.thread_id_x()
    branch = b.scf_if_else(b.cmp_lt(tid, n), result_types=[F32])
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(b.const_f32(1.0))
        with else_ctx:
            b.scf_yield(b.const_f32(2.0))
    b.global_store(C, tid, branch.results[0], align=4)
    b.ret()


# Kept out of KERNELS: recording it is refused, not replayed.
IF_ELSE_RESULTS_MESSAGE = (
    "portable IR does not support an scf.if_else with results; "
    "lower the kernel directly"
)


KERNELS = {
    "scalar": build_scalar,
    "memory": build_memory,
    "vector": build_vector,
    "forloop": build_forloop,
    "scf_if": build_scf_if,
    "nested": build_nested,
    "if_else": build_if_else,
    "if_else_nested": build_if_else_nested,
    "if_else_in_for": build_if_else_in_for,
    "inline_asm": build_inline_asm,
    "inline_asm_multi": build_inline_asm_multi,
}


# --------------------------------------------------------------------------
# Independent op-count oracle (per-op-shape agnostic).
# --------------------------------------------------------------------------
def _count_ops(ops):
    n = 0
    for op in ops:
        n += 1
        for r in op.regions:
            n += _count_ops(r.ops)
    return n


def _count_instrs(prog):
    n = 0
    for i in prog:
        if i.get("op") in ("emit", "scf_for", "scf_if", "scf_if_else", "ret"):
            n += 1
        for key in ("body", "then", "else"):
            if key in i:
                n += _count_instrs(i[key])
    return n


class TestRecordingBuilder(unittest.TestCase):
    def test_refuses_an_if_else_with_results(self):
        """The recipe's scf_if_else carries no result types, so a recorded
        result-bearing if/else would replay with its results unbound: the live
        recorder and both post-hoc walks refuse it instead."""
        from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe

        with self.assertRaises(NotImplementedError) as live:
            build_if_else_results(RecordingIRBuilder("if_else_results"))
        self.assertEqual(str(live.exception), IF_ELSE_RESULTS_MESSAGE)
        plain = IRBuilder("if_else_results")
        build_if_else_results(plain)
        for walk in (kernel_to_recipe, kerneldef_to_recipe):
            with self.subTest(walk=walk.__name__):
                with self.assertRaises(NotImplementedError) as posthoc:
                    walk(plain.kernel)
                self.assertEqual(str(posthoc.exception), IF_ELSE_RESULTS_MESSAGE)

    def test_non_invasive(self):
        """RecordingIRBuilder must build the IDENTICAL KernelDef as IRBuilder."""
        for name, fn in KERNELS.items():
            with self.subTest(kernel=name):
                plain = IRBuilder(name)
                fn(plain)
                rec = RecordingIRBuilder(name)
                fn(rec)
                self.assertEqual(
                    print_ir(plain.kernel),
                    print_ir(rec.kernel),
                    f"{name}: subclassing altered the built kernel",
                )

    def test_recipe_matches_kernel(self):
        """Live-recorded recipe must match an independent post-hoc walk of the
        built KernelDef (catches interception / region-routing drift)."""
        for name, fn in KERNELS.items():
            with self.subTest(kernel=name):
                rb = RecordingIRBuilder(name)
                fn(rb)
                recipe = rb.recipe()
                expected = kernel_to_recipe(rb.kernel)
                self.assertEqual(
                    recipe, expected, f"{name}: recorded recipe diverged from KernelDef"
                )

    def test_no_ops_dropped(self):
        """Op count in the recipe must equal op count in the KernelDef
        (independent of per-op shape) -- nothing dropped or duplicated."""
        for name, fn in KERNELS.items():
            with self.subTest(kernel=name):
                rb = RecordingIRBuilder(name)
                fn(rb)
                body_instrs = [
                    i for i in rb.recipe()["program"] if i.get("op") != "param"
                ]
                self.assertEqual(
                    _count_instrs(body_instrs),
                    _count_ops(rb.kernel.body.ops),
                    f"{name}: op count mismatch (recorder dropped/added ops)",
                )

    def test_params_captured(self):
        rb = RecordingIRBuilder("memory")
        build_memory(rb)
        params = [i for i in rb.recipe()["program"] if i.get("op") == "param"]
        self.assertEqual([p["name"] for p in params], ["A", "C"])
        self.assertEqual(
            params[0]["attrs"], {"noalias": True, "readonly": True, "align": 16}
        )

    def test_multi_result_captured(self):
        """N-result ops (inline_asm_multi) must record all results via 'outs'."""
        rb = RecordingIRBuilder("inline_asm_multi")
        build_inline_asm_multi(rb)
        asm = [
            i
            for i in rb.recipe()["program"]
            if i.get("op") == "emit" and i["opcode"] == "tile.inline_asm"
        ]
        self.assertEqual(len(asm), 1)
        self.assertIn("outs", asm[0], "multi-result asm must use 'outs'")
        self.assertEqual(len(asm[0]["outs"]), 2)

    def test_region_nesting(self):
        """The scf.if inside the scf.for body must be nested in the recipe."""
        rb = RecordingIRBuilder("nested")
        build_nested(rb)
        body = [i for i in rb.recipe()["program"] if i.get("op") != "param"]
        scf_for = next(i for i in body if i["op"] == "scf_for")
        self.assertTrue(
            any(i["op"] == "scf_if" for i in scf_for["body"]),
            "scf.if was not nested inside the scf.for body",
        )

    def test_if_else_records_both_arms(self):
        """scf.if_else -> one scf_if_else instr carrying BOTH arms, each holding
        exactly the ops (and the value definitions) built inside it."""
        rb = RecordingIRBuilder("if_else")
        build_if_else(rb)
        body = [i for i in rb.recipe()["program"] if i.get("op") != "param"]
        ie = [i for i in body if i["op"] == "scf_if_else"]
        self.assertEqual(len(ie), 1)
        ie = ie[0]
        self.assertEqual(set(ie), {"op", "cond", "then", "else"})
        then_ops = [i.get("opcode") for i in ie["then"]]
        else_ops = [i.get("opcode") for i in ie["else"]]
        self.assertIn("arith.fmul", then_ops)
        self.assertNotIn("arith.fmul", else_ops)
        self.assertEqual(then_ops[-1], else_ops[-1], "both arms end in the barrier")
        # The cond is the cmp emitted just before, by name.
        cmp = body[body.index(ie) - 1]
        self.assertEqual(cmp["out"]["bind"], ie["cond"])
        # A value defined in an arm is bound there, not at the top level.
        top_binds = {i["out"]["bind"] for i in body if "out" in i}
        arm_binds = {i["out"]["bind"] for i in ie["then"] + ie["else"] if "out" in i}
        self.assertFalse(top_binds & arm_binds)

    def test_if_else_nesting(self):
        rb = RecordingIRBuilder("if_else_nested")
        build_if_else_nested(rb)
        body = [i for i in rb.recipe()["program"] if i.get("op") != "param"]
        outer = next(i for i in body if i["op"] == "scf_if_else")
        self.assertTrue(any(i["op"] == "scf_if" for i in outer["then"]))
        self.assertFalse(any(i["op"] == "scf_if_else" for i in outer["then"]))
        inner = next(i for i in outer["else"] if i["op"] == "scf_if_else")
        self.assertEqual(len(inner["then"]), 2)
        self.assertEqual(len(inner["else"]), 2)

        rb = RecordingIRBuilder("if_else_in_for")
        build_if_else_in_for(rb)
        body = [i for i in rb.recipe()["program"] if i.get("op") != "param"]
        scf_for = next(i for i in body if i["op"] == "scf_for")
        ie = next(i for i in scf_for["body"] if i["op"] == "scf_if_else")
        self.assertEqual(ie["then"][0]["in"][1], scf_for["iv"])

    def test_if_else_matches_legacy_walk(self):
        """kerneldef_to_recipe (the older post-hoc walk) spells scf_if_else the
        same way the recorder does."""
        from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe

        for name in ("if_else", "if_else_nested", "if_else_in_for"):
            with self.subTest(kernel=name):
                rb = RecordingIRBuilder(name)
                KERNELS[name](rb)
                self.assertEqual(rb.recipe(), kerneldef_to_recipe(rb.kernel))

    def test_if_else_expands_and_needs_level_three(self):
        """The pure-Python VM mirror replays scf_if_else to an equivalent
        program, and the ABI stamp says an engine must know the op."""
        from rocke.portable_ir.src import abi
        from rocke.portable_ir.utils import recipe_expand

        for name in ("if_else", "if_else_nested", "if_else_in_for"):
            with self.subTest(kernel=name):
                rb = RecordingIRBuilder(name)
                KERNELS[name](rb)
                recipe = rb.recipe()
                flat = recipe_expand.expand_recipe(recipe, {})
                self.assertEqual(recipe_expand.equiv_reason(flat, recipe), "")
                self.assertEqual(abi.recipe_min_reader(recipe), 3)
        # Dropping the else arm is a detected difference, not a silent match.
        rb = RecordingIRBuilder("if_else")
        build_if_else(rb)
        a = rb.recipe()
        b = kernel_to_recipe(rb.kernel)
        next(i for i in b["program"] if i.get("op") == "scf_if_else")["else"] = []
        self.assertNotEqual(recipe_expand.equiv_reason(a, b), "")

    def test_hand_recorder_if_else(self):
        """RecipeRecorder.scf_if_else places the instr where the block opens and
        routes each arm into its own body, like IRBuilder.scf_if_else."""
        from rocke.portable_ir.src.recipe_recorder import PTR_F32, RecipeRecorder
        from rocke.portable_ir.utils import recipe_expand

        rec = RecipeRecorder("hand_if_else", [])
        C = rec.param("C", PTR_F32)
        n = rec.param("n", "i32")
        tid = rec.thread_id_x()
        cond = rec.cmp_gt(n, rec.const_i32(0))
        with rec.scf_if_else(cond) as (then_ctx, else_ctx):
            with then_ctx:
                rec.store(C, tid, rec.const_f32(1.0))
            with else_ctx:
                rec.store(C, tid, rec.const_f32(2.0))
        rec.ret()
        recipe = rec.recipe()
        prog = recipe["program"]
        ie = [i for i in prog if i["op"] == "scf_if_else"]
        self.assertEqual(len(ie), 1)
        self.assertEqual(prog[prog.index(ie[0]) + 1]["op"], "ret")
        self.assertEqual(ie[0]["cond"], cond.reg)
        self.assertEqual([len(ie[0]["then"]), len(ie[0]["else"])], [2, 2])
        flat = recipe_expand.expand_recipe(recipe, {})
        self.assertEqual(recipe_expand.equiv_reason(flat, recipe), "")


def _online_lib():
    from rocke.portable_ir.src import online

    return online.resolve_library_path(build=False)


@unittest.skipIf(_online_lib() is None, "no shared librocke; run online.build_lib()")
class TestIfElseReplay(unittest.TestCase):
    """With a source-matched engine at hand, both replay paths rebuild the
    scf.if_else kernels to the Python lowerer's .ll, byte for byte."""

    def test_replay_is_byte_identical(self):
        import os

        from rocke.core import ir_export
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.portable_ir.src import online, recipe_bundle

        saved = {
            k: os.environ.get(k) for k in ("ROCKE_ONLINE_LIB", "ROCKE_LLVM_FLAVOR")
        }
        os.environ["ROCKE_ONLINE_LIB"] = _online_lib()
        os.environ["ROCKE_LLVM_FLAVOR"] = "llvm20"
        try:
            online.load()
            for name in ("if_else", "if_else_nested", "if_else_in_for"):
                for arch in ("gfx942", "gfx950"):
                    rb = RecordingIRBuilder(name)
                    KERNELS[name](rb)
                    py = lower_kernel_to_llvm(
                        rb.kernel, llvm_flavor="llvm20", arch=arch
                    )
                    with self.subTest(kernel=name, arch=arch, path="recipe"):
                        cbor = recipe_bundle.cbor_encode(rb.recipe())
                        vm, _ = online.recipe_cbor_to_llvm(cbor, arch=arch)
                        self.assertEqual(vm, py)
                    with self.subTest(kernel=name, arch=arch, path="ir_json"):
                        text = ir_export.export_kernel_ir_json(rb.kernel)
                        eng, _ = online.ir_json_to_llvm(text, arch=arch)
                        self.assertEqual(eng, py)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_engine_refuses_an_if_else_with_results(self):
        """Both C readers refuse an if/else that yields values with the
        recorder's text: the IR JSON of one (whose results the JSON lists) and
        a recipe whose scf_if_else lists results (no recorder writes one)."""
        import os

        from rocke.core import ir_export
        from rocke.portable_ir.src import online, recipe_bundle

        saved = os.environ.get("ROCKE_ONLINE_LIB")
        os.environ["ROCKE_ONLINE_LIB"] = _online_lib()
        try:
            online.load()
            plain = IRBuilder("if_else_results")
            build_if_else_results(plain)
            for arch in ("gfx942", "gfx950"):
                with self.subTest(path="ir_json", arch=arch):
                    text = ir_export.export_kernel_ir_json(plain.kernel)
                    with self.assertRaises(RuntimeError) as imported:
                        online.ir_json_to_llvm(text, arch=arch)
                    self.assertIn(IF_ELSE_RESULTS_MESSAGE, str(imported.exception))
                with self.subTest(path="recipe", arch=arch):
                    rb = RecordingIRBuilder("if_else_carry")
                    build_if_else(rb)
                    recipe = rb.recipe()
                    (instr,) = [
                        i for i in recipe["program"] if i["op"] == "scf_if_else"
                    ]
                    instr["results"] = ["if0"]
                    cbor = recipe_bundle.cbor_encode(recipe)
                    with self.assertRaises(RuntimeError) as replayed:
                        online.recipe_cbor_to_llvm(cbor, arch=arch)
                    self.assertIn(IF_ELSE_RESULTS_MESSAGE, str(replayed.exception))
        finally:
            if saved is None:
                os.environ.pop("ROCKE_ONLINE_LIB", None)
            else:
                os.environ["ROCKE_ONLINE_LIB"] = saved


def _production_cases():
    """Unmodified production builders + their defining module."""
    from rocke.instances.common import elementwise
    from rocke.instances.common import reduce as reduce_mod

    # attention_unified lives in the rocke LIBRARY tree (kernels/), not platform.
    from kernels.common import attention_unified
    from rocke.portable_ir.examples import export_mha

    return [
        (
            "attn2d_fp16_d128",
            attention_unified,
            lambda: export_mha.build("fp16", 128, 2048, 1, 32, 1),
        ),
        (
            "elementwise_silu",
            elementwise,
            lambda: elementwise.build_elementwise(
                elementwise.ElementwiseSpec(
                    op="silu", dtype="bf16", block_size=64, vec=8
                )
            ),
        ),
        (
            "reduce_sum_f16",
            reduce_mod,
            lambda: reduce_mod.build_reduce2d(
                reduce_mod.Reduce2DSpec(
                    n_per_block=4096,
                    op="sum",
                    block_size=256,
                    vec=4,
                    dtype="f16",
                    wave_size=64,
                )
            ),
        ),
    ]


class TestRecordingProductionKernels(unittest.TestCase):
    """Wire RecordingIRBuilder into real production builders (untouched) and
    assert the live recording matches the built KernelDef."""

    def test_records_production_kernels(self):
        from rocke.portable_ir.src.recording_builder import (
            kernel_to_recipe,
            record_kernel,
        )

        for name, module, build in _production_cases():
            with self.subTest(kernel=name):
                kernel, recorded = record_kernel(build, module)
                self.assertEqual(
                    recorded,
                    kernel_to_recipe(kernel),
                    f"{name}: live recording diverged from KernelDef",
                )

    def test_matches_legacy_recipe(self):
        """Recorder agrees with the byte-identity-proven kerneldef_to_recipe walk
        (for kernels without N-result ops -> identical JSON, hence identical HSACO)."""
        from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe
        from rocke.portable_ir.src.recording_builder import record_kernel

        for name, module, build in _production_cases():
            with self.subTest(kernel=name):
                kernel, recorded = record_kernel(build, module)
                self.assertEqual(
                    recorded,
                    kerneldef_to_recipe(kernel),
                    f"{name}: recorded recipe != legacy recipe",
                )


class TestRecordCoverage(unittest.TestCase):
    """Surface-wide gate: the recorder must faithfully capture every production
    kernel the reuse harness can build (0 recorder failures)."""

    def test_no_recorder_failures_across_surface(self):
        import os

        from rocke.portable_ir.drivers import record_coverage

        if not record_coverage._PARITY_DIRS:
            self.skipTest("parity emitter dirs not present")
        paths = sorted(
            os.path.join(d, f)
            for d in record_coverage._PARITY_DIRS
            for f in os.listdir(d)
            if f.endswith("_emit.py")
        )
        if not paths:
            self.skipTest("no parity emitters found")

        failures, npass = [], 0
        for path in paths:
            label = os.path.basename(path)[: -len("_emit.py")]
            try:
                mod = record_coverage._load_module(path)
                status, detail = record_coverage._record_one(mod)
            except Exception as e:  # noqa: BLE001
                status, detail = "FAIL", f"harness error: {e}"
            if status == "FAIL":
                failures.append(f"{label}: {detail}")
            elif status == "PASS":
                npass += 1

        self.assertEqual(failures, [], "recorder gaps:\n" + "\n".join(failures))
        # Guard against the harness silently degrading to ~0 recorded kernels.
        self.assertGreaterEqual(npass, 40, f"only {npass} kernels recorded")


if __name__ == "__main__":
    unittest.main()
