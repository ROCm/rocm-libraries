# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the compilability baseline in drivers/hsaco_parity.py -- the part
# that decides whether a CI run is a pass or a failure.
#
# The gate has to hold two things apart. Some kernels in the tree cannot be
# compiled at all (gfx950-only MFMA intrinsics asked to codegen for gfx942, one
# kernel that exhausts memory in comgr). Failing on those would make the gate
# red from its first run, and a permanently red gate gets switched off. But a
# kernel that STOPS compiling is exactly the regression the gate exists to
# catch. So the known-broken set is pinned by name, and only movement away from
# it fails.
#
# These are the tests for that decision, kept separate from the sweep itself so
# they need no comgr, no engine binary and no GPU: the sweep's verdict logic is
# the part that must be right, and it is pure.
#
#   python3 -m unittest rocke.portable_ir.tests.test_hsaco_gate

import json
import os
import unittest

from rocke.portable_ir.drivers.hsaco_parity import (
    _BASELINE,
    _auto_cap_gb,
    _llvm_reason,
    baseline_improvements,
    check_baseline,
)

# One arch's worth of pinned expectations: two kernels that cannot compile, one
# the lowerer declines on purpose, and 40 that do compile.
BASE = {
    "gfx942": {
        "compared": 40,
        "uncompilable": {"mfma_gemm": "Cannot select: intrinsic", "moe_fp8": "OOM"},
        "refused": ["gfx1151_wmma_gemm"],
    }
}
AS_PINNED = {
    "mfma_gemm": "uncompilable",
    "moe_fp8": "uncompilable",
    "gfx1151_wmma_gemm": "refused",
}


class TestBaselineVerdict(unittest.TestCase):
    def test_the_pinned_state_passes(self):
        """The known-broken set, unchanged, is not a failure."""
        self.assertEqual(check_baseline("gfx942", {"cmp": 40}, AS_PINNED, BASE), [])

    def test_a_newly_uncompilable_kernel_fails(self):
        """The regression this gate exists to catch. One cause should produce
        one message: the kernel is also missing from the comparison count, but
        naming it twice makes a CI log harder to read, not more informative."""
        seen = dict(AS_PINNED, batched_gemm="uncompilable")
        fails = check_baseline("gfx942", {"cmp": 39}, seen, BASE)
        self.assertEqual(len(fails), 1)
        self.assertIn("batched_gemm", fails[0])
        self.assertIn("newly fail to compile", fails[0])

    def test_a_newly_declined_kernel_fails(self):
        """Coverage lost to the lowerer declining is still coverage lost."""
        seen = dict(AS_PINNED, batched_gemm="refused")
        fails = check_baseline("gfx942", {"cmp": 39}, seen, BASE)
        self.assertTrue(any("newly declined" in f for f in fails))

    def test_shrinking_coverage_fails_even_with_no_new_breakage(self):
        """Catches a kernel disappearing from the sweep entirely, which leaves
        no name behind to compare and so is invisible to the set difference."""
        fails = check_baseline("gfx942", {"cmp": 31}, AS_PINNED, BASE)
        self.assertEqual(len(fails), 1)
        self.assertIn("coverage shrank", fails[0])

    def test_a_swap_is_caught_though_the_count_is_unchanged(self):
        """One kernel fixed and another broken keeps every total the same. This
        is why the baseline pins names and not just numbers."""
        seen = {k: v for k, v in AS_PINNED.items() if k != "mfma_gemm"}
        seen["batched_gemm"] = "uncompilable"
        fails = check_baseline("gfx942", {"cmp": 40}, seen, BASE)
        self.assertTrue(any("batched_gemm" in f for f in fails))

    def test_a_missing_arch_entry_fails_rather_than_silently_passing(self):
        """An unpinned arch has nothing to compare against, so a run against it
        proves nothing; saying so beats reporting a pass."""
        fails = check_baseline("gfx1201", {"cmp": 5}, {}, BASE)
        self.assertEqual(len(fails), 1)
        self.assertIn("no baseline entry", fails[0])

    def test_a_fixed_kernel_is_reported_but_does_not_fail(self):
        """A kernel getting fixed must never block a merge; it should ask for a
        baseline update."""
        seen = {k: v for k, v in AS_PINNED.items() if k != "moe_fp8"}
        self.assertEqual(check_baseline("gfx942", {"cmp": 41}, seen, BASE), [])
        notes = baseline_improvements("gfx942", seen, BASE)
        self.assertEqual(len(notes), 1)
        self.assertIn("moe_fp8", notes[0])
        self.assertIn("drop it from the baseline", notes[0])


class TestCrashReasons(unittest.TestCase):
    """LLVM prints its diagnosis and then aborts, so the reason has to be
    recovered from the dead child's stderr."""

    def test_it_finds_the_llvm_error(self):
        err = "some progress chatter\nLLVM ERROR: Cannot select: intrinsic %foo\n"
        self.assertEqual(_llvm_reason(err), "LLVM ERROR: Cannot select: intrinsic %foo")

    def test_it_prefers_the_last_diagnosis(self):
        """The final message is the fatal one; earlier ones were survivable."""
        err = "error: early warning\nLLVM ERROR: out of memory\n"
        self.assertEqual(_llvm_reason(err), "LLVM ERROR: out of memory")

    def test_it_recognises_the_allocation_failure(self):
        """The memory-exhaustion path prints this instead of LLVM ERROR."""
        self.assertEqual(_llvm_reason("Allocation failed\n"), "Allocation failed")

    def test_silence_yields_no_reason_rather_than_a_wrong_one(self):
        self.assertEqual(_llvm_reason("\n  \n"), "")


class TestMemoryCap(unittest.TestCase):
    def test_the_cap_is_sane_on_this_machine(self):
        """A fixed cap that fits the author's machine can be lethal on a CI
        runner, so it is derived. Bounds, not an exact value: the point is that
        it is usable wherever it runs."""
        cap = _auto_cap_gb()
        self.assertGreaterEqual(cap, 4)
        self.assertLessEqual(cap, 48)


class TestShippedBaseline(unittest.TestCase):
    def test_it_parses_and_names_every_kernel_it_excuses(self):
        """The checked-in baseline is the gate's definition of acceptable. A
        malformed one, or one that excuses a kernel without saying why, would
        weaken the gate quietly."""
        self.assertTrue(os.path.exists(_BASELINE), _BASELINE)
        with open(_BASELINE) as f:
            base = json.load(f)
        self.assertIn("gfx950", base)
        self.assertIn("gfx942", base)
        for arch in ("gfx950", "gfx942"):
            entry = base[arch]
            self.assertGreater(entry["compared"], 0)
            for kernel, why in entry["uncompilable"].items():
                self.assertTrue(
                    why.strip(), f"{arch}:{kernel} excused without a reason"
                )
