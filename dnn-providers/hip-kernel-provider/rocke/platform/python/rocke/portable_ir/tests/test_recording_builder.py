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


KERNELS = {
    "scalar": build_scalar,
    "memory": build_memory,
    "vector": build_vector,
    "forloop": build_forloop,
    "scf_if": build_scf_if,
    "nested": build_nested,
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
        if i.get("op") in ("emit", "scf_for", "scf_if", "ret"):
            n += 1
        for key in ("body", "then", "else"):
            if key in i:
                n += _count_instrs(i[key])
    return n


class TestRecordingBuilder(unittest.TestCase):
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
