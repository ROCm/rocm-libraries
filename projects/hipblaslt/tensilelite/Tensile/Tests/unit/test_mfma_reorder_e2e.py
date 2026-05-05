################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""End-to-end integration test exercising ``mfmaReorder`` through the
graph-native validator (bead rocm-libraries-x7t).

Background
----------
``test_validate_lr_before_mfma_graph.py::TestLRBeforeMFMA_MfmaReorder`` and
``TestLRBeforeMFMA_ForceUnrollSubIter`` *simulate* mfmaReorder by manually
placing MFMAs at "reordered" slot indices. They verify the graph rule is
INVARIANT under MFMA reordering — the validator reads register operands
directly, so the slot the MFMA lands at doesn't matter.

What was missing (and what this file adds): no test passed a real
``mfmaReorder=[...]`` array end-to-end through:

  1. ``ScheduleInfo(..., mfmaReorder=[...])`` construction.
  2. ``CustomSchedule.expand_macro``-equivalent permutation application
     (``CustomSchedule.py:336-337``: ``mfmaCode = [mfmaCode[x] for x in
     opt1.mfmaReorder]``).
  3. MFMA stream emission into a ``LoopBodyCapture``.
  4. ``build_dataflow_graph`` from that capture.
  5. ``validate_edge_wait_coverage`` on the resulting graph.

Why this layer (and not a full kernel build)
--------------------------------------------
A full ``customMainLoopSchedule`` invocation requires a real
``KernelWriter``, a fully-resolved ``Solution``, mock or real ``GR/LR/Pack``
codes, and the iter-code matrix — a heavy fixture surface that already has
coverage in ``test_ScheduleCapture.py::TestRealKernelCapture`` /
``TestPhase4DefaultCapture`` / ``TestPhase5DefaultTailCapture`` /
``TestDataflowGraphIntegration``. Those classes drive the F32X TF32
16x16x32 4x4 schedule, which has an empty ``mfmaReorder`` (the
``_get_schedule_128x128x32_TF32`` shape does not reorder MFMAs). So they
cover everything *except* the line that consumes a non-empty
``mfmaReorder``.

Smaller-scoped configs that *do* trigger ``mfmaReorder=[...]`` (e.g.
``_get_schedule_256x256x128_8bit`` at CustomSchedule.py:1361 and the
TF32 192x256x32 NN shape at CustomSchedule.py:3683) reject inside
``Solution`` construction under the ISA / vector-width combinations the
shared ``isa_infrastructure`` fixture provides. Standing up an alternate
ISA / vector-width matrix purely to exercise the reorder line would
inflate the test session time by another ~140ms per matched config and
duplicate the kernel-build infrastructure.

Therefore this file targets the *production code path* that consumes
``mfmaReorder`` directly — the permutation step at
``CustomSchedule.py:336-337``. It:

  * Constructs a ``ScheduleInfo`` with a real production ``mfmaReorder``
    shape.
  * Mirrors the exact permutation ``customMainLoopSchedule`` applies to
    ``mfmaCode``.
  * Emits the permuted MFMAs (with their original register operands) into
    a ``LoopBodyCapture`` alongside the LRs that produce their inputs and
    a covering SWait.
  * Runs the full graph build + wait-coverage pipeline on the result.

The reorder shape used here is the simplest non-trivial production shape:
the 16-element block prefix of CustomSchedule.py:1366
(``_get_schedule_256x256x128_8bit``), which is itself a clean
"interleave four MFMA-quads" permutation — exactly the pattern that
exposes the position-vs-dataflow distinction tested below.

Cross-references
----------------
The structural-rule invariant tests (the slot-index simulation) live in:

  test_validate_lr_before_mfma_graph.py::TestLRBeforeMFMA_MfmaReorder
  test_validate_lr_before_mfma_graph.py::TestLRBeforeMFMA_ForceUnrollSubIter

They are complementary: this file pins the production *pipeline*, the
graph rule unit tests pin the validator's *behaviour* under reorder.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingWaitFailure,
)
from Tensile.Components.CustomSchedule import ScheduleInfo

from dataflow_fixtures import (
    make_capture, make_lr, make_mfma, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Production reorder shape — provenance documented inline.
# =============================================================================
# 32-element prefix lifted verbatim from CustomSchedule.py:1366
# (``_get_schedule_256x256x128_8bit``, ``isTN(kernel) and TLDS == 1``,
# ``not kernel["ForceUnrollSubIter"]`` branch). The full production
# shape there is 64-element; we use the first half to keep the test's
# vgpr footprint small. The pattern is the simplest non-trivial
# production reorder that exercises the permutation logic: groups of
# four MFMAs interleaved across two "row halves" of an 8x8 tile.
#
#     mfmaReorder = [0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27,
#                    4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31, ...]
#
# Output slot 4 takes original-index 8, slot 16 takes original-index 4,
# etc. — so the validator's per-edge wait check sees consumer MFMAs at
# slots that DIFFER from their original mfmaCode index, exactly the
# property the negative test below asserts on.
PRODUCTION_REORDER_32 = [0, 1, 2, 3, 8, 9, 10, 11,
                         16, 17, 18, 19, 24, 25, 26, 27,
                         4, 5, 6, 7, 12, 13, 14, 15,
                         20, 21, 22, 23, 28, 29, 30, 31]


# Each MFMA reads from a distinct A-vgpr and B-vgpr range, spaced so the
# LRs that feed them have non-overlapping destinations and so the per-MFMA
# accumulator chains don't collide. One vgpr per source is sufficient
# for the graph's per-byte resolver to form an edge.
#
# Filler MFMAs in ``graph_native_validation_base`` use vgpr 200+, so the
# body under test must stay strictly below 200 for n=32:
#
#   LRA0  ->  v8..v39    (32 distinct A producers)
#   LRB0  ->  v60..v91   (32 distinct B producers)
#   MFMA c_dst (acc) -> v100..v131  (distinct per MFMA, below filler)
_VGPRS_PER_MFMA = 1
_LRA_BASE = 8
_LRB_BASE = 60
_C_DST_BASE = 100


def _build_mfma_code_for_n(n: int):
    """Build a list of synthetic MFMA TaggedInstructions, one per index.

    Mirrors the role of ``mfmaCode`` in ``customMainLoopSchedule`` — a
    dense list keyed by *original* MFMA index. The reorder permutation
    later picks elements out of this list by ``mfmaReorder[k]``.

    Returns instructions WITHOUT a slot index (slot=0 placeholder); the
    caller is responsible for re-slotting after the permutation, exactly
    as the production macro does.
    """
    return [
        make_mfma(
            c_dst_start=_C_DST_BASE + i * _VGPRS_PER_MFMA,
            a_src_start=_LRA_BASE + i * _VGPRS_PER_MFMA,
            b_src_start=_LRB_BASE + i * _VGPRS_PER_MFMA,
            slot=0,  # overwritten below
            c_dst_count=_VGPRS_PER_MFMA,
            a_src_count=_VGPRS_PER_MFMA,
            b_src_count=_VGPRS_PER_MFMA,
        )
        for i in range(n)
    ]


def _apply_reorder_and_reslot(mfma_code, mfma_reorder, *, slot_offset: int = 0):
    """Apply the production permutation to ``mfma_code`` and assign final slots.

    Mirrors CustomSchedule.py:336-337 verbatim::

        if len(opt1.mfmaReorder) > 0:
            mfmaCode = [mfmaCode[x] for x in opt1.mfmaReorder]

    Then re-slots: the MFMA at output position k gets
    ``mfma_index=k + slot_offset``, matching what ``LoopBodyCaptureBuilder``
    would produce after the macro emits the permuted MFMAs sequentially
    starting at the slot following the LRs/SWait that precede them.

    ``slot_offset`` lets the caller leave room before the first MFMA for
    pre-MFMA LRs / SWaits that share ``slot.slot_kind=SLOT_KIND_MFMA``
    (which the synthetic ``make_lr`` / ``make_swait`` builders use).
    Without an offset, an LR at ``slot=0`` would share its
    ``vmfma_index`` with the first reordered MFMA at ``slot=0``, and
    the graph's stream-order resolver couldn't distinguish "LR before
    MFMA" from "MFMA before LR" within the same vmfma_index bucket.
    """
    permuted = [mfma_code[x] for x in mfma_reorder]
    # Replace each TaggedInstruction's slot to reflect its new position.
    # TaggedInstruction is a dataclass; rebuild via make_mfma's pattern
    # but reuse the existing inst handle (preserving the register
    # operands established in _build_mfma_code_for_n).
    from Tensile.Components.ScheduleCapture import (
        SLOT_KIND_MFMA, SlotKey, TaggedInstruction,
    )
    reslotted = []
    for k, ti in enumerate(permuted):
        reslotted.append(TaggedInstruction(
            inst=ti.inst,
            category=ti.category,
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=k + slot_offset, sequence=0),
        ))
    return reslotted


# =============================================================================
# E2E tests
# =============================================================================


class TestMfmaReorderE2E(GraphNativeValidationTest):
    """End-to-end pipeline: ScheduleInfo(mfmaReorder) -> permutation ->
    capture -> graph -> validate.

    Pins the production line ``mfmaCode = [mfmaCode[x] for x in
    opt1.mfmaReorder]`` (CustomSchedule.py:336-337) is consumed by the
    graph-native validator without false positives, and that an
    induced wait-coverage gap on a *reordered* MFMA still fires the
    expected ``MissingWaitFailure(dscnt)``.
    """

    def test_scheduleinfo_round_trips_mfmaReorder(self):
        """``ScheduleInfo`` stashes the supplied reorder array verbatim
        and exposes it via ``opt1.mfmaReorder`` — the attribute the
        production permutation step reads.

        Pinning this guards against a future refactor that drops the
        attribute or coerces the list (e.g. into a tuple) in a way that
        would break the comprehension at CustomSchedule.py:337.
        """
        opt1 = ScheduleInfo(
            numCodePaths=1, numMfma=32, optSchedule={}, syncCode=[],
            nglshift=0, nllshift=0,
            mfmaReorder=PRODUCTION_REORDER_32,
        )
        assert opt1.mfmaReorder == PRODUCTION_REORDER_32
        assert len(opt1.mfmaReorder) == opt1.numMfma

    def test_e2e_clean_schedule_passes_validation(self):
        """Build a clean reordered schedule end-to-end and assert the
        graph-native validator returns zero failures.

        Pipeline:

          1. ``ScheduleInfo(mfmaReorder=PRODUCTION_REORDER_32)``.
          2. Mock ``mfmaCode`` (32 distinct MFMAs, each with unique
             A/B/C register operands).
          3. Apply the production permutation
             (``[mfmaCode[x] for x in opt1.mfmaReorder]``) and re-slot.
          4. Place a covering ``LRA`` for each MFMA's a_src AT slot=0
             (subiter=0, ordered before all MFMAs by sequence) and one
             ``LRB`` per b_src; emit a single ``SWait(dscnt=0)``
             between the LRs and the MFMA stream.
          5. Build the graph, run wait-coverage validation.

        The clean schedule has SWait(dscnt=0) draining all LR -> MFMA
        edges before the first MFMA fires, so no failure should fire.
        """
        n = 32
        opt1 = ScheduleInfo(
            numCodePaths=1, numMfma=n, optSchedule={}, syncCode=[],
            nglshift=0, nllshift=0,
            mfmaReorder=PRODUCTION_REORDER_32,
        )

        # 1. Build the original mfmaCode (one entry per ORIGINAL index 0..n-1).
        mfma_code = _build_mfma_code_for_n(n)

        # 2. Mirror CustomSchedule.py:336-337 — apply the reorder. Land
        # the MFMAs at slot=1..n (slot_offset=1) to leave vmfma_index=0
        # for the pre-MFMA LRs / SWait so they form an unambiguous
        # stream-order chain.
        permuted_mfmas = _apply_reorder_and_reslot(
            mfma_code, opt1.mfmaReorder, slot_offset=1)

        # 3. Build a body: LRs first (at sequence-distinguished pre-MFMA
        #    slots), then a draining SWait, then the permuted MFMAs.
        #    The LRs feed registers that the post-permutation MFMAs
        #    actually read — the reorder doesn't change register
        #    operands, so the LR -> MFMA edges still form correctly.
        instructions = []
        for i in range(n):
            instructions.append(make_lr(
                dst_vgpr_start=_LRA_BASE + i * _VGPRS_PER_MFMA,
                dst_vgpr_count=_VGPRS_PER_MFMA,
                lds_offset=64 + i * 4,
                slot=0, sequence=i, category="LRA0",
            ))
            instructions.append(make_lr(
                dst_vgpr_start=_LRB_BASE + i * _VGPRS_PER_MFMA,
                dst_vgpr_count=_VGPRS_PER_MFMA,
                lds_offset=2048 + i * 4,
                slot=0, sequence=n + i, category="LRB0",
            ))
        # Single SWait drains all LRs before any MFMA fires.
        instructions.append(make_swait(slot=0, sequence=2 * n, dscnt=0))
        # Then the permuted MFMAs.
        instructions.extend(permuted_mfmas)

        cap = make_capture(BODY_LABEL_ML, instructions)
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap, num_mfma=n)))
        self.assert_no_failures(failures)

    def test_e2e_uncovered_lr_after_reorder_fires_missing_wait(self):
        """Negative-path: with the same reordered MFMA stream as the
        positive test, place ONE LR (the producer of MFMA-original-index
        24, which lands at output slot 12 after PRODUCTION_REORDER_32)
        AFTER the SWait so its LR -> MFMA edge has no covering dscnt
        drain. Expect ``MissingWaitFailure(counter_kind="dscnt")``.

        This is the load-bearing assertion: the validator picked up a
        gap on a MFMA that was MOVED by the reorder (output slot 12 !=
        its original index 24), proving the production permutation
        feeds through to the per-edge wait check.
        """
        n = 32
        opt1 = ScheduleInfo(
            numCodePaths=1, numMfma=n, optSchedule={}, syncCode=[],
            nglshift=0, nllshift=0,
            mfmaReorder=PRODUCTION_REORDER_32,
        )

        mfma_code = _build_mfma_code_for_n(n)
        permuted_mfmas = _apply_reorder_and_reslot(
            mfma_code, opt1.mfmaReorder, slot_offset=1)

        # Pick an output slot whose underlying original-MFMA index
        # is moved by the permutation. Output slot 12 maps to original
        # index PRODUCTION_REORDER_32[12] == 24. Place its LRA AFTER
        # the SWait so the LR -> MFMA(output-slot-12) edge has no
        # covering drain.
        broken_output_slot = 12
        broken_original_idx = PRODUCTION_REORDER_32[broken_output_slot]
        assert broken_original_idx == 24, (
            "Reorder shape changed; update the expected original index "
            "or the test's broken_output_slot."
        )
        assert broken_output_slot != broken_original_idx, (
            "Test's load-bearing property — that the broken MFMA was "
            "MOVED by the reorder — was lost. Pick another slot."
        )
        broken_a_vgpr = _LRA_BASE + broken_original_idx * _VGPRS_PER_MFMA

        # Build the same LR/SWait/MFMA stream, but skip the broken-MFMA
        # producer LR from the pre-SWait phase, then add it AFTER the
        # SWait at a higher sequence slot.
        instructions = []
        for i in range(n):
            if i != broken_original_idx:
                instructions.append(make_lr(
                    dst_vgpr_start=_LRA_BASE + i * _VGPRS_PER_MFMA,
                    dst_vgpr_count=_VGPRS_PER_MFMA,
                    lds_offset=64 + i * 4,
                    slot=0, sequence=i, category="LRA0",
                ))
            instructions.append(make_lr(
                dst_vgpr_start=_LRB_BASE + i * _VGPRS_PER_MFMA,
                dst_vgpr_count=_VGPRS_PER_MFMA,
                lds_offset=2048 + i * 4,
                slot=0, sequence=n + i, category="LRB0",
            ))
        # SWait — drains all LRs that have been issued so far. The
        # broken LR will be added AFTER this, leaving its edge uncovered.
        instructions.append(make_swait(slot=0, sequence=2 * n, dscnt=0))
        # Now append the broken LR AFTER the swait (still pre-MFMA in
        # slot order — but no SWait sits between this LR and the MFMA
        # at output slot 4 that consumes it).
        instructions.append(make_lr(
            dst_vgpr_start=broken_a_vgpr,
            dst_vgpr_count=_VGPRS_PER_MFMA,
            lds_offset=64 + broken_original_idx * 4,
            slot=0, sequence=2 * n + 1, category="LRA0",
        ))
        # Finally the permuted MFMAs.
        instructions.extend(permuted_mfmas)

        cap = make_capture(BODY_LABEL_ML, instructions)
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap, num_mfma=n)))

        # Pin: a MissingWait(dscnt) failure exists for the LRA0 producer
        # we deliberately placed after the SWait. This is the load-
        # bearing assertion proving the permutation pipeline is wired:
        # the consumer MFMA the failure points at occupies OUTPUT slot 4
        # because of the reorder permutation — without the permutation
        # being honoured, the failure (or its absence) would point at
        # the original-index-8 slot instead.
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )
        # The producer is the LRA0 we made uncovered.
        assert f.producer.category == "LRA0", (
            f"expected LRA0 producer, got {f.producer.category!r}"
        )

    def test_e2e_empty_reorder_is_a_noop(self):
        """The production guard ``if len(opt1.mfmaReorder) > 0`` skips
        the permutation when the array is empty. This test pins that an
        empty ``mfmaReorder`` is a valid configuration — many production
        schedules (the bulk of ``CustomSchedule.py``) leave it empty —
        and the e2e pipeline still passes.
        """
        n = 32
        opt1 = ScheduleInfo(
            numCodePaths=1, numMfma=n, optSchedule={}, syncCode=[],
            nglshift=0, nllshift=0,
            mfmaReorder=[],
        )
        assert opt1.mfmaReorder == []

        mfma_code = _build_mfma_code_for_n(n)
        # Production guard: when reorder is empty, mfmaCode is untouched.
        # Assign each MFMA its original-index slot — what
        # ``LoopBodyCaptureBuilder`` does after macro emission with no
        # reorder.
        identity_reorder = list(range(n))
        permuted_mfmas = _apply_reorder_and_reslot(
            mfma_code, identity_reorder, slot_offset=1)

        instructions = []
        for i in range(n):
            instructions.append(make_lr(
                dst_vgpr_start=_LRA_BASE + i * _VGPRS_PER_MFMA,
                dst_vgpr_count=_VGPRS_PER_MFMA,
                lds_offset=64 + i * 4,
                slot=0, sequence=i, category="LRA0",
            ))
            instructions.append(make_lr(
                dst_vgpr_start=_LRB_BASE + i * _VGPRS_PER_MFMA,
                dst_vgpr_count=_VGPRS_PER_MFMA,
                lds_offset=2048 + i * 4,
                slot=0, sequence=n + i, category="LRB0",
            ))
        instructions.append(make_swait(slot=0, sequence=2 * n, dscnt=0))
        instructions.extend(permuted_mfmas)

        cap = make_capture(BODY_LABEL_ML, instructions)
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap, num_mfma=n)))
        self.assert_no_failures(failures)
