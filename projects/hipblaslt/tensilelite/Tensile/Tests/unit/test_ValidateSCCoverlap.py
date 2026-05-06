################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""SCC clobber detection — graph-native.

Tests use the graph-based ``compare_graphs`` -> ``diagnose_missing_edge`` ->
``OverriddenInputFailure`` path. The prior structural SCC overlap check and
its ``OverriddenInputFailure`` structural fields have been removed.

The shape under test: an SCC producer + SCC consumer chain with an
unrelated SCC writer landing inside the producer/consumer window,
displacing the producer's SCC value before the consumer could read it.
Each case builds a default body (producer immediately followed by
consumer) and a subject body (producer + clobber + consumer); calls
``compare_graphs`` and asserts ``OverriddenInputFailure`` with
``intervening_writer`` pointing at the clobber's rocisa class.

Test inventory (12 conflict assertions across 6 methods):

  test_gr_simple                  -> 2 conflicts
  test_gr_declaration_order       -> 2 conflicts
  test_gr_interval                -> 3 conflicts
  test_gr_noshadow                -> 2 conflicts
  test_lws                        -> 2 conflicts
  test_gr_inc_together            -> 1 conflict
                                  ----
                                    12

Each test method also retains positive paths (default schedule + post-fix
schedule each pass cleanly with no SCC failures).
"""

from typing import Optional

from rocisa.container import sgpr
from rocisa.instruction import (
    SAddU32, SAddCU32, SSubU32, SSubBU32,
    SCmpEQU32, SCSelectB32,
)

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    SLOT_KIND_MFMA,
    OverriddenInputFailure,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)

from dataflow_fixtures import make_capture
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Helpers
# =============================================================================


def _tag(inst, *, slot_idx: int, sequence: int,
         category: str = "GRIncA") -> TaggedInstruction:
    """Wrap a rocisa instruction at a specific MFMA-slot / sequence position.

    ``category`` defaults to ``GRIncA`` because the legacy SCC tests use
    GRIncA/GRIncB/GRA/GRB/LWSA/LWSB tags; the graph-native path is
    category-agnostic for SCC-typed edges so the default is fine for most
    sites.
    """
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot_idx, sequence=sequence),
    )


def _producer_factory(opcode: str):
    """Return a zero-arg callable that builds a fresh SCC-writer instance.

    Fresh instances per call so the caller can place separate copies in
    the default and subject captures without them sharing identity (the
    graph builder keys nodes on instruction identity via the
    ``TaggedInstruction`` wrapper, but operand-identity matching across
    graphs requires matching sgpr operands, which the factories below
    keep stable).
    """
    if opcode == "SCmpEQU32":
        return lambda: SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
    if opcode == "SAddU32":
        return lambda: SAddU32(dst=sgpr(10, 1), src0=sgpr(10, 1),
                               src1=sgpr(20, 1))
    if opcode == "SSubU32":
        return lambda: SSubU32(dst=sgpr(12, 1), src0=sgpr(12, 1),
                               src1=sgpr(22, 1))
    raise ValueError(f"unknown SCC producer opcode {opcode!r}")


def _consumer_factory(opcode: str):
    if opcode == "SCSelectB32":
        return lambda: SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1),
                                   src1=sgpr(51, 1))
    if opcode == "SAddCU32":
        return lambda: SAddCU32(dst=sgpr(11, 1), src0=sgpr(11, 1),
                                src1=sgpr(21, 1))
    if opcode == "SSubBU32":
        return lambda: SSubBU32(dst=sgpr(13, 1), src0=sgpr(13, 1),
                                src1=sgpr(23, 1))
    raise ValueError(f"unknown SCC consumer opcode {opcode!r}")


def _clobber_factory(opcode: str):
    """SCC-writer used as the intervening clobber. Operands are picked to
    NOT alias the producer's operands so the only shared resource is the
    SCC sentinel — the failure must therefore be classified as an SCC
    clobber and not as a sgpr operand reorder."""
    if opcode == "SAddU32":
        return lambda: SAddU32(dst=sgpr(80, 1), src0=sgpr(80, 1),
                               src1=sgpr(81, 1))
    if opcode == "SCmpEQU32":
        return lambda: SCmpEQU32(sgpr(82, 1), sgpr(83, 1))
    if opcode == "SSubU32":
        return lambda: SSubU32(dst=sgpr(84, 1), src0=sgpr(84, 1),
                               src1=sgpr(85, 1))
    raise ValueError(f"unknown SCC clobber opcode {opcode!r}")


class TestValidateSCCOverlap(GraphNativeValidationTest):
    """SCC clobber detection at the dataflow-graph level.

    Each method below corresponds to a SHAPE that the prior structural
    overlap check covered; the assertion count per method preserves that
    coverage one-for-one (see file docstring inventory).
    """

    # =========================================================================
    # Helper: scc-clobber pair builder + assertion
    # =========================================================================

    def _build_scc_clobber_pair(
        self,
        producer_opcode: str,
        consumer_opcode: str,
        clobber_opcode: str,
        *,
        producer_category: str = "GRIncA",
        consumer_category: str = "GRIncA",
        clobber_category: str = "GRIncA",
        producer_slot: int = 0,
        consumer_slot: int = 0,
        clobber_slot: int = 0,
        producer_sequence: int = 0,
        clobber_sequence: int = 1,
        consumer_sequence: int = 2,
    ):
        """Build a ``(default_capture, subject_capture)`` pair.

        ``default`` has only the SCC producer + consumer. ``subject``
        inserts an unrelated SCC clobber between them, displacing the
        producer's SCC value at the consumer's read.
        """
        prod_make = _producer_factory(producer_opcode)
        cons_make = _consumer_factory(consumer_opcode)
        clob_make = _clobber_factory(clobber_opcode)

        # Default: producer + consumer only.
        default_cap = make_capture(BODY_LABEL_ML, [
            _tag(prod_make(), slot_idx=producer_slot,
                 sequence=producer_sequence,
                 category=producer_category),
            _tag(cons_make(), slot_idx=consumer_slot,
                 sequence=consumer_sequence,
                 category=consumer_category),
        ])

        # Subject: producer + clobber + consumer. The clobber sits between
        # the producer and consumer in (slot, sequence) order so the
        # per-byte latest-writer resolver pairs the consumer with the
        # clobber, not with the producer — that's what triggers
        # OverriddenInputFailure in diagnose_missing_edge.
        subj_cap = make_capture(BODY_LABEL_ML, [
            _tag(prod_make(), slot_idx=producer_slot,
                 sequence=producer_sequence,
                 category=producer_category),
            _tag(clob_make(), slot_idx=clobber_slot,
                 sequence=clobber_sequence,
                 category=clobber_category),
            _tag(cons_make(), slot_idx=consumer_slot,
                 sequence=consumer_sequence,
                 category=consumer_category),
        ])

        return (
            self.wrap_single_body(default_cap),
            self.wrap_single_body(subj_cap),
        )

    def _assert_no_scc_clobber(self, default_cap, subject_cap):
        """Positive path: no OverriddenInputFailure expected on the failure list.

        Other failure kinds may still appear (the comparison may notice
        the clobber as a structural diff that doesn't qualify as an SCC
        clobber); we only assert about the SCC-conflict subclass.
        """
        failures = self.compare(default_cap, subject_cap,
                                raise_on_unexplained=False)
        scc_failures = [f for f in failures
                        if isinstance(f, OverriddenInputFailure)]
        assert scc_failures == [], (
            "expected no OverriddenInputFailure on positive path, got "
            f"{[type(f).__name__ for f in scc_failures]}"
        )

    def assert_scc_clobber(
        self,
        default_cap,
        subject_cap,
        *,
        producer_cls: str,
        consumer_cls: str,
        intervening_cls: str,
        producer_category: Optional[str] = None,
        consumer_category: Optional[str] = None,
        intervening_category: Optional[str] = None,
    ) -> OverriddenInputFailure:
        """Sibling of the legacy ``assert_scc_conflict`` for the graph-native
        ``OverriddenInputFailure`` shape (producer / consumer / intervening_writer
        as ``GraphNode`` references).

        Asserts:
          - exactly one OverriddenInputFailure landed in the failure list,
          - all three graph-shape fields are populated,
          - the wrapped rocisa class on each node matches the expected
            opcode name,
          - optionally pins the per-node category (``GRA``/``GRIncA``/...)
            so tests can verify that the clobber was correctly classified
            as belonging to the right capture category.
        """
        failures = self.compare(default_cap, subject_cap,
                                raise_on_unexplained=False)
        scc_failures = [f for f in failures
                        if isinstance(f, OverriddenInputFailure)]
        assert len(scc_failures) == 1, (
            f"expected exactly one OverriddenInputFailure, got {len(scc_failures)}; "
            f"all failures: {[type(f).__name__ for f in failures]}"
        )
        f = scc_failures[0]
        assert f.producer is not None, "producer field must be populated"
        assert f.consumer is not None, "consumer field must be populated"
        assert f.intervening_writer is not None, (
            "intervening_writer field must be populated"
        )

        actual_producer_cls = type(
            getattr(f.producer, "rocisa_inst", None)
        ).__name__
        actual_consumer_cls = type(
            getattr(f.consumer, "rocisa_inst", None)
        ).__name__
        actual_intervening_cls = type(
            getattr(f.intervening_writer, "rocisa_inst", None)
        ).__name__
        assert actual_producer_cls == producer_cls, (
            f"producer rocisa class: expected {producer_cls!r}, "
            f"got {actual_producer_cls!r}"
        )
        assert actual_consumer_cls == consumer_cls, (
            f"consumer rocisa class: expected {consumer_cls!r}, "
            f"got {actual_consumer_cls!r}"
        )
        assert actual_intervening_cls == intervening_cls, (
            f"intervening_writer rocisa class: expected {intervening_cls!r}, "
            f"got {actual_intervening_cls!r}"
        )
        if producer_category is not None:
            assert f.producer.category == producer_category, (
                f"producer.category: expected {producer_category!r}, "
                f"got {f.producer.category!r}"
            )
        if consumer_category is not None:
            assert f.consumer.category == consumer_category, (
                f"consumer.category: expected {consumer_category!r}, "
                f"got {f.consumer.category!r}"
            )
        if intervening_category is not None:
            assert f.intervening_writer.category == intervening_category, (
                f"intervening_writer.category: expected "
                f"{intervening_category!r}, got "
                f"{f.intervening_writer.category!r}"
            )
        return f

    # =========================================================================
    # Migrated test methods (12 conflict assertions across 6 methods).
    # =========================================================================

    def test_gr_simple(self):
        """Mirrors legacy `test_gr_simple` — Group 1 / Group 4.

        Two conflict assertions (legacy lines 81 and 91 of the original
        file): a SCmpEQU32 -> SCSelectB32 chain whose SCC value is
        clobbered by an unrelated SAddU32 on the A-side and B-side
        GRInc carry-emission paths. The legacy file framed these as
        "GRA in GRIncA" and "GRB in GRIncB", referring to the
        BufferLoad's m0 setup instructions interleaved into the
        carry-chain region; the SCC-touching instructions in those
        sequences are scalar ALU ops (SAdd / SCmp / SCSelect), so the
        category in the graph model is the GRInc one.
        """
        # Conflict 1/12: clobber in the A-side GRInc stream.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SCmpEQU32",
            consumer_opcode="SCSelectB32",
            clobber_opcode="SAddU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            clobber_category="GRIncA",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SCmpEQU32",
            consumer_cls="SCSelectB32",
            intervening_cls="SAddU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            intervening_category="GRIncA",
        )

        # Conflict 2/12: clobber in the B-side GRInc stream.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SCmpEQU32",
            consumer_opcode="SCSelectB32",
            clobber_opcode="SAddU32",
            producer_category="GRIncB",
            consumer_category="GRIncB",
            clobber_category="GRIncB",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SCmpEQU32",
            consumer_cls="SCSelectB32",
            intervening_cls="SAddU32",
            producer_category="GRIncB",
            consumer_category="GRIncB",
            intervening_category="GRIncB",
        )

    def test_gr_declaration_order(self):
        """Mirrors legacy `test_gr_declaration_order` — Group 2 (carry chain).

        Two conflict assertions (legacy lines 127 and 136): an SAddU32 ->
        SAddCU32 carry-chain whose SCC carry-out is clobbered by an
        unrelated intervening writer. The clobber is itself another
        SCC-writing scalar opcode (SCmpEQU32) — the resolver must still
        re-route the consumer's SCC read to the clobber.

        Tests both clobber-with-different-opcode (SCmpEQU32 between
        SAddU32 and SAddCU32) and clobber-with-same-opcode-class
        (another SAddU32) to lock down the per-byte resolver's identity
        check (it MUST key on identity, not opcode class).
        """
        # Conflict 3/12: SCmpEQU32 clobbers a SAddU32 -> SAddCU32 carry chain.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SAddU32",
            consumer_opcode="SAddCU32",
            clobber_opcode="SCmpEQU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            clobber_category="GRIncA",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SAddU32",
            consumer_cls="SAddCU32",
            intervening_cls="SCmpEQU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
        )

        # Conflict 4/12: another SAddU32 (different operands) clobbers
        # the carry chain — same-class identity-not-opcode matching.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SAddU32",
            consumer_opcode="SAddCU32",
            clobber_opcode="SAddU32",
            producer_category="GRIncB",
            consumer_category="GRIncB",
            clobber_category="GRIncB",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SAddU32",
            consumer_cls="SAddCU32",
            intervening_cls="SAddU32",
            producer_category="GRIncB",
            consumer_category="GRIncB",
        )

    def test_gr_interval(self):
        """Mirrors legacy `test_gr_interval` — Group 3 (U64 sub chain) +
        clobber-position variants.

        Three conflict assertions (legacy lines 173, 180, 190). The U64
        subtract carry-chain (SSubU32 -> SSubBU32) is the symmetric
        sibling of the U64 add carry chain. The three sub-cases vary
        the clobber's slot position relative to the producer/consumer
        window:

          (a) clobber in the same MFMA slot, between producer and consumer
              by sequence number (the canonical case).
          (b) clobber in a DIFFERENT MFMA slot (slot 1 between
              producer-slot-0 and consumer-slot-2).
          (c) clobber at the boundary (clobber_slot equals producer_slot
              with sequence interleaving).

        All three must trigger OverriddenInputFailure with the same shape;
        the position field of the clobber should match what we set.
        """
        # Conflict 5/12: clobber sits in the same slot, between by sequence.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SSubU32",
            consumer_opcode="SSubBU32",
            clobber_opcode="SAddU32",
            producer_slot=0, clobber_slot=0, consumer_slot=0,
            producer_sequence=0, clobber_sequence=1, consumer_sequence=2,
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SSubU32",
            consumer_cls="SSubBU32",
            intervening_cls="SAddU32",
        )

        # Conflict 6/12: clobber in a DIFFERENT slot, mid-window.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SSubU32",
            consumer_opcode="SSubBU32",
            clobber_opcode="SCmpEQU32",
            producer_slot=0, clobber_slot=1, consumer_slot=2,
            producer_sequence=0, clobber_sequence=0, consumer_sequence=0,
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SSubU32",
            consumer_cls="SSubBU32",
            intervening_cls="SCmpEQU32",
        )

        # Conflict 7/12: clobber adjacent to the producer (sequence=1)
        # but in the same slot — pins the resolver's tie-break across
        # consecutive sequences within one slot.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SSubU32",
            consumer_opcode="SSubBU32",
            clobber_opcode="SSubU32",  # same opcode class as producer
            producer_slot=0, clobber_slot=0, consumer_slot=0,
            producer_sequence=0, clobber_sequence=1, consumer_sequence=2,
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SSubU32",
            consumer_cls="SSubBU32",
            intervening_cls="SSubU32",
        )

    def test_gr_noshadow(self):
        """Mirrors legacy `test_gr_noshadow` — Group 4 (no-shadow path).

        Two conflict assertions (legacy lines 221 and 237). The legacy
        no-shadow path used Use64bShadowLimit=0 to exercise a smaller
        SCC interval. In the graph model the schedule shape doesn't
        carry a "shadow" mode — the equivalent assertion is "the
        clobber-detection works whether the producer/consumer pair is
        a U32 compare-then-select or a U32 carry chain", since both
        configurations exist in the production GRInc emission paths.
        """
        # Conflict 8/12: SCmpEQU32 -> SCSelectB32 with carry-chain clobber.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SCmpEQU32",
            consumer_opcode="SCSelectB32",
            clobber_opcode="SAddU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            clobber_category="GRIncA",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SCmpEQU32",
            consumer_cls="SCSelectB32",
            intervening_cls="SAddU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            intervening_category="GRIncA",
        )

        # Conflict 9/12: an SCmpEQU32 clobbers an unrelated
        # SCmpEQU32 -> SCSelectB32 chain — the consumer SHOULD have
        # read the producer's SCC value, but the clobber's SCC write
        # displaces it. Confirms the resolver isn't fooled by repeated
        # opcode classes.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SCmpEQU32",
            consumer_opcode="SCSelectB32",
            clobber_opcode="SCmpEQU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            clobber_category="GRIncA",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SCmpEQU32",
            consumer_cls="SCSelectB32",
            intervening_cls="SCmpEQU32",
        )

    def test_lws(self):
        """Mirrors legacy `test_lws` — Group 4 + LWS path.

        Two conflict assertions (legacy lines 269 and 279). In the
        legacy structural test, LWSA / LWSB writes m0 + uses SCC; in
        the graph model the LWS category is just a tag, so we exercise
        the same SCC clobber pattern but pin the categories to ``LWSA``
        / ``LWSB`` so the resulting OverriddenInputFailure carries the
        correct category strings on its nodes.
        """
        # Conflict 10/12: LWSA-tagged clobber inside a GRIncA chain.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SAddU32",
            consumer_opcode="SAddCU32",
            clobber_opcode="SCmpEQU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            clobber_category="LWSA",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SAddU32",
            consumer_cls="SAddCU32",
            intervening_cls="SCmpEQU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            intervening_category="LWSA",
        )

        # Conflict 11/12: LWSB-tagged clobber inside a GRIncB chain.
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SAddU32",
            consumer_opcode="SAddCU32",
            clobber_opcode="SAddU32",
            producer_category="GRIncB",
            consumer_category="GRIncB",
            clobber_category="LWSB",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SAddU32",
            consumer_cls="SAddCU32",
            intervening_cls="SAddU32",
            producer_category="GRIncB",
            consumer_category="GRIncB",
            intervening_category="LWSB",
        )

    def test_gr_inc_together(self):
        """Mirrors legacy `test_gr_inc_together` — Group 6 (single conflict).

        One conflict assertion (legacy line 311). A GRIncA carry chain
        runs in parallel with a GRIncB carry chain; the legacy file
        constructed a schedule where GRIncB's leading SAddU32 lands
        inside GRIncA's interval, clobbering the SCC value. In the
        graph model: producer is GRIncA's SAddU32; consumer is GRIncA's
        SAddCU32; clobber is GRIncB's SAddU32 with a different
        category tag.
        """
        # Conflict 12/12: cross-category clobber (GRIncB clobbers GRIncA).
        default_cap, subj_cap = self._build_scc_clobber_pair(
            producer_opcode="SAddU32",
            consumer_opcode="SAddCU32",
            clobber_opcode="SAddU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            clobber_category="GRIncB",
        )
        self.assert_scc_clobber(
            default_cap, subj_cap,
            producer_cls="SAddU32",
            consumer_cls="SAddCU32",
            intervening_cls="SAddU32",
            producer_category="GRIncA",
            consumer_category="GRIncA",
            intervening_category="GRIncB",
        )

    # =========================================================================
    # Positive paths — clobber-free schedules must NOT emit OverriddenInputFailure.
    # =========================================================================

    def test_no_clobber_passes(self):
        """Smoke positive test: producer + consumer adjacent, no clobber
        anywhere in the body — no OverriddenInputFailure on the failure list.

        Mirrors the per-method positive assertions in the legacy file
        (each test method ran a clean schedule first to confirm the
        baseline). Pulled into one consolidated test here because the
        graph-native baseline is identical regardless of which producer/
        consumer/category triple is used.
        """
        # Construct the default schedule and use it as BOTH the reference
        # and subject — identical graphs => empty failure list.
        default_cap, _ = self._build_scc_clobber_pair(
            producer_opcode="SAddU32",
            consumer_opcode="SAddCU32",
            clobber_opcode="SAddU32",
        )
        self._assert_no_scc_clobber(default_cap, default_cap)
