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
"""Coverage for the unregistered-ISA ArchProfile fallback (rocm-libraries-zkzw).

Hard rules pinned by these tests:
  - `_resolve_arch_profile_for_isa` returns `None` (NOT
    `_DEFAULT_CDNA4_ARCH_PROFILE`) when the ISA tuple is not registered
    in `_ARCH_PROFILES_BY_ISA`.
  - The unregistered-ISA warning includes the ISA tuple verbatim so it is
    grep-able from build logs.
  - There is no separate `unregistered_isa` carrier-level field. A
    `FourPartCapture` (and its derived `DataflowGraph`) with
    `arch_profile is None` is the only way to express "skip timing
    validation"; there is no silent CDNA-4 fallback path anywhere.
  - Timing helpers (`_quad_cycle_gap_ok`, `_cvt_to_mfma_gap_ok`,
    `_mfma_pack_to_cvt_gap_ok`) short-circuit when the carrying graph
    has `arch_profile is None`, returning a `TimingCheck` whose
    `result` is `TimingResult.ARCH_NOT_SUPPORTED` and
    `observed`/`required` are 0 so callers emit no
    `TimingTooCloseFailure`.
  - Non-timing validation (cross-graph diff, wait coverage edges that
    don't go through the four timing helpers, SCC, MiddlePack
    interleave) still runs on unregistered-ISA kernels — only timing checks
    are suppressed.
  - The warning fires on EVERY call (no process-wide de-dup): a build
    that validates N kernels for the same uncharacterized arch fires N
    warnings.
"""

import pytest

from rocisa.container import vgpr
from rocisa.instruction import VXorB32

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    FourPartCapture,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)
from Tensile.Components.CMSValidator import (
    ArchProfile,
    TimingCheck,
    TimingResult,
    TimingTooCloseFailure,
    _ARCH_PROFILES_BY_ISA,
    _DEFAULT_CDNA4_ARCH_PROFILE,
    _cvt_to_mfma_gap_ok,
    _mfma_pack_to_cvt_gap_ok,
    _quad_cycle_gap_ok,
    _resolve_arch_profile,
    _resolve_arch_profile_for_isa,
    build_dataflow_graph,
    validate_edge_wait_coverage,
)

from dataflow_fixtures import (
    make_lr, make_mfma, make_swait, make_capture,
)


# A tuple guaranteed not to be in `_ARCH_PROFILES_BY_ISA`. Picked to
# look like a plausible-but-fake ISA so grepping logs for the literal
# turns up the warning without colliding with any real arch.
_UNREGISTERED_ISA = (99, 99, 99)


def _tag(inst, *, category: str, mfma_index: int, sequence: int) -> TaggedInstruction:
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=mfma_index, sequence=sequence),
    )


def _wrap(ml_capture, *, arch_profile=None):
    """Wrap a single main-loop capture into a FourPartCapture, filling the
    other 3 bodies with a no-op MFMA so build_dataflow_graph's
    non-empty-body precondition is satisfied. `arch_profile=None` is the
    unregistered-ISA case (timing checks skipped); pass
    `_DEFAULT_CDNA4_ARCH_PROFILE` for fixtures that need timing checks
    to fire."""
    _FILLER_RANGES = {
        BODY_LABEL_ML_PREV: (200, 204, 208),
        BODY_LABEL_NGL:     (220, 224, 228),
        BODY_LABEL_NLL:     (240, 244, 248),
    }

    def _filler(label):
        c, a, b = _FILLER_RANGES[label]
        return make_capture(label, [make_mfma(
            c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
        )])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: _filler(BODY_LABEL_NGL)},
        n_ll={0: _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=arch_profile,
    )


def _zero_gap_mfma_to_alu_capture():
    """Build a capture that produces a TimingTooCloseFailure when validated
    against CDNA 4 timing constants. Mirrors
    test_dataflow_graph_register_gaps.test_mfma_to_alu_consumer_zero_gap_emits_timing_too_close
    so the unregistered-ISA test asserts the SAME schedule produces NO failure
    once timing is skipped."""
    alu_consumer = VXorB32(dst=vgpr(20, 1), src0=vgpr(0, 1), src1=vgpr(21, 1))
    return make_capture(BODY_LABEL_ML, [
        make_lr(8, 4, 64, slot=0, category="LRA0"),
        make_swait(slot=1, dscnt=0),
        make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                  slot=2, sequence=0, a_src_count=2),
        # Same vmfma_index, sub_index 1 — zero gap. ALU consumer reads v0
        # which overlaps the MFMA producer's accumulator v0..v3.
        _tag(alu_consumer, category="PackA0", mfma_index=2, sequence=1),
    ])


# =============================================================================
# 1. Resolver: known ISA returns the registered profile (regression pin).
# =============================================================================


class TestResolverKnownISA:
    """Sanity: a known ISA still resolves to its registered profile."""

    def test_cdna4_isa_resolves_to_default_profile(self):
        profile = _resolve_arch_profile_for_isa((9, 5, 0))
        assert profile is _DEFAULT_CDNA4_ARCH_PROFILE, (
            f"Known ISA (9, 5, 0) must resolve to the registered CDNA 4 "
            f"profile. Got {profile!r}."
        )

    def test_none_isa_resolves_to_default_profile(self):
        """`isa is None` keeps the historical bit-identical CDNA 4 path
        (covered for callers that haven't plumbed ISA info through —
        every existing test fixture relies on this fallback)."""
        profile = _resolve_arch_profile_for_isa(None)
        assert profile is _DEFAULT_CDNA4_ARCH_PROFILE


# =============================================================================
# 2. Resolver: unregistered ISA returns None and warns with the ISA verbatim.
# =============================================================================


class TestResolverUnknownISA:
    """`_resolve_arch_profile_for_isa` must return None and warn when the
    ISA is not registered."""

    def test_unregistered_isa_returns_none_not_cdna4(self):
        """The CDNA-4 silent fallback is removed: unregistered ISA returns
        None so callers can detect the case and skip timing checks."""
        # Pre-condition: ensure the ISA is genuinely unknown.
        assert _UNREGISTERED_ISA not in _ARCH_PROFILES_BY_ISA

        profile = _resolve_arch_profile_for_isa(_UNREGISTERED_ISA)
        assert profile is None, (
            f"Unregistered ISA must return None (no silent CDNA-4 fallback). "
            f"Got {profile!r}."
        )

    def test_unregistered_isa_warning_contains_isa_verbatim(self, capsys):
        """The warning must include the ISA tuple verbatim so it is
        grep-able from build logs."""
        _resolve_arch_profile_for_isa(_UNREGISTERED_ISA)
        captured = capsys.readouterr()
        # The literal `repr((99, 99, 99))` representation must appear
        # somewhere in the warning so log-grepping for the ISA tuple
        # finds it.
        assert repr(_UNREGISTERED_ISA) in captured.out, (
            f"Warning must contain the ISA tuple verbatim "
            f"({repr(_UNREGISTERED_ISA)!r}). Got stdout: {captured.out!r}"
        )
        # Tensile's standard warning prefix must be present so log
        # filters that already match Tensile warnings catch this one.
        assert "WARNING" in captured.out

    def test_unregistered_isa_warning_fires_every_call(self, capsys):
        """Warning is emitted on EVERY call (no process-wide de-dup).
        A build that validates N kernels for the same uncharacterized
        arch sees N warnings — that is the directed behavior so callers
        cannot accidentally lose visibility into how many kernels fell
        into the unregistered-ISA path."""
        _resolve_arch_profile_for_isa(_UNREGISTERED_ISA)
        first = capsys.readouterr().out
        _resolve_arch_profile_for_isa(_UNREGISTERED_ISA)
        second = capsys.readouterr().out
        _resolve_arch_profile_for_isa(_UNREGISTERED_ISA)
        third = capsys.readouterr().out
        assert "WARNING" in first
        assert "WARNING" in second, (
            f"Second call for the same unregistered ISA must re-emit the "
            f"warning (no de-dup). Got stdout: {second!r}"
        )
        assert "WARNING" in third, (
            f"Third call for the same unregistered ISA must re-emit the "
            f"warning (no de-dup). Got stdout: {third!r}"
        )
        # Each call must contain the ISA tuple verbatim.
        assert repr(_UNREGISTERED_ISA) in first
        assert repr(_UNREGISTERED_ISA) in second
        assert repr(_UNREGISTERED_ISA) in third


# =============================================================================
# 3. _resolve_arch_profile: graph-side detection of arch_profile=None.
# =============================================================================


class TestGraphSideTimingSkipDetection:
    """An `arch_profile=None` FourPartCapture builds into a DataflowGraph
    whose `arch_profile` is also None; `_resolve_arch_profile` reports
    None for that graph. Timing helpers then short-circuit on the same
    predicate."""

    def test_build_dataflow_graph_propagates_none_arch_profile(self):
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        assert g.arch_profile is None, (
            f"build_dataflow_graph must copy `arch_profile=None` from the "
            f"FourPartCapture to the resulting DataflowGraph. "
            f"Got {g.arch_profile!r}."
        )

    def test_build_dataflow_graph_propagates_explicit_arch_profile(self):
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE))
        assert g.arch_profile is _DEFAULT_CDNA4_ARCH_PROFILE

    def test_resolve_arch_profile_returns_none_for_skipped_graph(self):
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        assert _resolve_arch_profile(g) is None, (
            "An unregistered-ISA graph (arch_profile=None) must surface as None "
            "from the general arch-profile resolver so the four pair-specific "
            "helpers can short-circuit on the same predicate."
        )


# =============================================================================
# 4. Helpers short-circuit on unregistered ISA — no TimingTooCloseFailure emitted.
# =============================================================================


class TestTimingHelpersShortCircuit:
    """The four pair-specific timing helpers must short-circuit when the
    carrying graph has `arch_profile is None`."""

    def test_quad_cycle_gap_ok_returns_arch_not_supported_on_unregistered_isa(self):
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        # Pick the same-slot MFMA→ALU edge created from the fixture.
        edge = next(
            (e for e in g.edges
             if getattr(e.producer, "category", None) == "MFMA"
             and getattr(e.consumer, "category", None) == "PackA0"),
            None,
        )
        assert edge is not None, (
            "Fixture must produce an MFMA→ALU edge for this assertion to "
            "be meaningful."
        )
        check = _quad_cycle_gap_ok(
            edge.producer, edge.consumer, 0, graph=g
        )
        assert isinstance(check, TimingCheck), (
            f"Helper must return a TimingCheck dataclass. Got "
            f"{type(check).__name__}: {check!r}"
        )
        assert check.result is TimingResult.ARCH_NOT_SUPPORTED, (
            f"Unregistered-ISA graph must report ARCH_NOT_SUPPORTED from "
            f"_quad_cycle_gap_ok regardless of the actual cycle gap. "
            f"Got result={check.result!r}."
        )
        assert check.observed == 0 and check.required == 0, (
            f"Short-circuit returns observed=0/required=0; got "
            f"observed={check.observed}, required={check.required}."
        )

    def test_cvt_to_mfma_gap_ok_returns_arch_not_supported_on_unregistered_isa(self):
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        mfma_nodes = [n for n in g.nodes.values() if n.category == "MFMA"]
        # The producer/consumer don't have to satisfy the CVT->MFMA
        # routing predicate — the helper short-circuits before it
        # consults instruction shape. We pass any two real nodes so
        # the short-circuit return path is exercised.
        producer = mfma_nodes[0]
        consumer = mfma_nodes[0]
        check = _cvt_to_mfma_gap_ok(producer, consumer, g)
        assert check == TimingCheck.arch_not_supported()

    def test_mfma_pack_to_cvt_gap_ok_returns_arch_not_supported_on_unregistered_isa(self):
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        mfma_nodes = [n for n in g.nodes.values() if n.category == "MFMA"]
        producer = mfma_nodes[0]
        consumer = mfma_nodes[0]
        check = _mfma_pack_to_cvt_gap_ok(producer, consumer, g)
        assert check == TimingCheck.arch_not_supported()


# =============================================================================
# 5. End-to-end: validate_edge_wait_coverage emits NO TimingTooCloseFailure
#    for the zero-gap MFMA->ALU schedule when ISA is unknown, but emits one
#    for the same schedule when ISA is known (CDNA 4).
# =============================================================================


class TestEndToEndTimingSuppression:
    """Pin the wire-up: the same schedule that fails on CDNA 4 timing
    constants must pass when the ISA is unknown — and ONLY the timing
    failure should disappear, not unrelated dataflow checks.
    """

    def test_known_isa_emits_timing_too_close_failure(self):
        """Baseline: CDNA 4 path produces TimingTooCloseFailure on the
        zero-gap MFMA->ALU schedule. Mirrors the existing regression
        pin in test_dataflow_graph_register_gaps."""
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE))
        failures = validate_edge_wait_coverage(g)
        timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
            and getattr(f.producer, "category", None) == "MFMA"
        ]
        assert timing, (
            f"Baseline pin: known ISA must emit TimingTooCloseFailure "
            f"on the zero-gap MFMA->ALU edge. Got: "
            f"{[type(f).__name__ for f in failures]}"
        )

    def test_unregistered_isa_emits_no_timing_too_close_failure(self):
        """Same schedule, but with `arch_profile=None` on the capture
        (the unregistered-ISA case): the timing helpers short-circuit, so
        NO TimingTooCloseFailure is emitted. Other failures (if any)
        are unaffected."""
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        failures = validate_edge_wait_coverage(g)
        timing = [
            f for f in failures
            if isinstance(f, TimingTooCloseFailure)
        ]
        assert not timing, (
            f"Unregistered-ISA path must emit NO TimingTooCloseFailure "
            f"(timing helpers short-circuit). Got: "
            f"{[type(f).__name__ for f in failures]}"
        )

    def test_unregistered_isa_non_timing_validation_still_runs(self):
        """The non-timing portion of `validate_edge_wait_coverage` must
        still run — only timing-specific failures are suppressed.
        We compare the per-edge-classification path of the same
        captures. Each capture is built afresh because rocisa
        instruction wrappers carry per-instance reads/writes that the
        graph builder mutates indirectly through `_populate_wrapper`,
        and the graph holds references to the underlying TaggedInstructions
        (sharing one capture between two graphs ties their FIFO state
        through `_all_nodes_in_order`).

        We construct an explicit MissingWaitFailure scenario (drop the
        SWait between LR and MFMA): the dscnt FIFO observes one
        outstanding LR with no covering wait. Both the known-ISA and
        unregistered-ISA paths must emit the MissingWaitFailure — this is a
        non-timing wait-coverage check, NOT a timing check, so the
        unregistered-ISA short-circuit must not suppress it."""
        # MissingWait scenario: LR producer + MFMA consumer with no
        # covering SWait dscnt. Pure wait-coverage failure — independent
        # of any quad-cycle timing constants.
        def _missing_wait_capture():
            return make_capture(BODY_LABEL_ML, [
                make_lr(8, 4, 64, slot=0, category="LRA0"),
                # No SWait here.
                make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                          slot=2, sequence=0, a_src_count=2),
            ])

        g_known = build_dataflow_graph(
            _wrap(_missing_wait_capture(), arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE))
        g_unknown = build_dataflow_graph(
            _wrap(_missing_wait_capture(), arch_profile=None))

        non_timing_known = [
            type(f).__name__ for f in validate_edge_wait_coverage(g_known)
            if not isinstance(f, TimingTooCloseFailure)
        ]
        non_timing_unknown = [
            type(f).__name__ for f in validate_edge_wait_coverage(g_unknown)
            if not isinstance(f, TimingTooCloseFailure)
        ]
        # Both paths must surface the MissingWaitFailure on the
        # uncovered LR -> MFMA edge.
        assert "MissingWaitFailure" in non_timing_known, (
            f"Baseline pin: known ISA must emit MissingWaitFailure on "
            f"the uncovered LR->MFMA edge. Got: {non_timing_known}"
        )
        assert sorted(non_timing_known) == sorted(non_timing_unknown), (
            f"Non-timing failures must be the same on both paths "
            f"(unregistered-ISA only suppresses timing checks). "
            f"known={non_timing_known}, unknown={non_timing_unknown}."
        )

    def test_unregistered_isa_graph_construction_succeeds(self):
        """Cross-graph diff invariant: build_dataflow_graph produces a
        well-formed graph (nodes, edges, captures populated) even when
        `arch_profile is None`. This pins that the unregistered-ISA branch
        does not short-circuit graph construction itself — only the
        downstream timing helpers."""
        cap = _zero_gap_mfma_to_alu_capture()
        g = build_dataflow_graph(_wrap(cap, arch_profile=None))
        assert g.nodes, "Graph must contain nodes on unregistered-ISA path"
        assert g.captures, "Graph must contain captures on unregistered-ISA path"
        # MFMA node must still be present; arch-skip is purely a timing
        # property, not an instruction-classification property.
        mfma_nodes = [n for n in g.nodes.values() if n.category == "MFMA"]
        assert mfma_nodes, "MFMA node must still be classified on unregistered-ISA path"
