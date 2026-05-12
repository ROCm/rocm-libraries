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
"""Capture-pipeline finalize() checks and idMap completeness.

These tests use the REAL LoopBodyCaptureBuilder.finalize() — the entire
purpose is to exercise the finalize-time checks. Synthetic instruction
classes (defined inline) trigger the SMEM/flat/store guards.
"""

from dataclasses import dataclass

import pytest

from Tensile.Components.ScheduleCapture import (
    LoopBodyCaptureBuilder,
    LoopBodyCapture,
    SLOT_KIND_MFMA,
    CaptureWiringError,
    CaptureSMEMError,
    CaptureFlatError,
    CaptureStoreError,
    CaptureIdmapMismatchError,
    assert_idmap_completeness,
)


# =============================================================================
# Stand-in instruction classes whose names match rocisa
# =============================================================================
# The finalize() check uses class-name matching (not isinstance) to stay free
# of hard rocisa imports. So we define classes with the names rocisa uses.


@dataclass
class SLoadB128:
    pass


@dataclass
class SLoadB64:
    pass


@dataclass
class FlatLoadB128:
    pass


@dataclass
class BufferStoreB128:
    pass


@dataclass
class GlobalStoreB128:
    pass


# =============================================================================
# finalize() — wiring (rocisa_inst != None)
# =============================================================================


class TestFinalizeWiring:
    def test_well_formed_capture_returns_loopbody(self):
        """Positive: a well-formed capture (every TaggedInstruction has inst
        non-None, no SMEM/flat/store) finalize()s without raising."""
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr, DSModifiers

        b = LoopBodyCaptureBuilder()
        inst = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0, 1),
                          ds=DSModifiers(offset=64))
        b.append(inst=inst, category="LRA0", subiter=0, mfma_index=0)
        result = b.finalize()
        assert isinstance(result, LoopBodyCapture)
        assert len(result.instructions) == 1

    def test_finalize_raises_capture_wiring_error_when_inst_is_none(self):
        b = LoopBodyCaptureBuilder()
        # Append with inst=None — rocisa wiring failed.
        b.append(inst=None, category="LRA0", subiter=0, mfma_index=0)
        with pytest.raises(CaptureWiringError) as exc:
            b.finalize()
        assert "LRA0" in str(exc.value)


# =============================================================================
# finalize() — SMEM guard
# =============================================================================


class TestFinalizeSMEMGuard:
    def test_smem_load_raises_capture_smem_error(self):
        b = LoopBodyCaptureBuilder()
        b.append(inst=SLoadB128(), category="OTHER",
                 subiter=0, mfma_index=0)
        with pytest.raises(CaptureSMEMError) as exc:
            b.finalize()
        assert "SLoadB128" in str(exc.value)

    def test_smem_load_b64_raises(self):
        b = LoopBodyCaptureBuilder()
        b.append(inst=SLoadB64(), category="OTHER",
                 subiter=0, mfma_index=0)
        with pytest.raises(CaptureSMEMError):
            b.finalize()


# =============================================================================
# finalize() — flat-op guard
# =============================================================================


class TestFinalizeFlatGuard:
    def test_flat_load_raises_capture_flat_error(self):
        b = LoopBodyCaptureBuilder()
        b.append(inst=FlatLoadB128(), category="OTHER",
                 subiter=0, mfma_index=0)
        with pytest.raises(CaptureFlatError) as exc:
            b.finalize()
        assert "FlatLoadB128" in str(exc.value)


# =============================================================================
# finalize() — store guard
# =============================================================================


class TestFinalizeStoreGuard:
    def test_buffer_store_raises_capture_store_error(self):
        b = LoopBodyCaptureBuilder()
        b.append(inst=BufferStoreB128(), category="OTHER",
                 subiter=0, mfma_index=0)
        with pytest.raises(CaptureStoreError) as exc:
            b.finalize()
        assert "BufferStoreB128" in str(exc.value)

    def test_global_store_raises(self):
        b = LoopBodyCaptureBuilder()
        b.append(inst=GlobalStoreB128(), category="OTHER",
                 subiter=0, mfma_index=0)
        with pytest.raises(CaptureStoreError):
            b.finalize()


# =============================================================================
# idMap completeness — pure function
# =============================================================================


class TestIdMapCompleteness:
    def test_matched_dict_and_capture_passes(self):
        from dataflow_fixtures import make_lr, make_swait, make_capture
        from Tensile.Components.ScheduleCapture import BODY_LABEL_ML

        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
        ])
        idmap = {"LRA0": [object(), object()]}  # 2 source instructions
        # Should not raise.
        assert_idmap_completeness(idmap, cap)

    def test_missing_instruction_raises(self):
        from dataflow_fixtures import make_lr, make_capture
        from Tensile.Components.ScheduleCapture import BODY_LABEL_ML

        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
        ])
        # idMap declares 5 LRA0 entries but capture only has 1.
        idmap = {"LRA0": [object()] * 5}
        with pytest.raises(CaptureIdmapMismatchError) as exc:
            assert_idmap_completeness(idmap, cap)
        assert "LRA0" in str(exc.value)
        assert "5" in str(exc.value)
        assert "1" in str(exc.value)

    def test_extra_instruction_raises(self):
        from dataflow_fixtures import make_lr, make_capture
        from Tensile.Components.ScheduleCapture import BODY_LABEL_ML

        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_lr(16, 4, 96, slot=2, category="LRA0"),
        ])
        idmap = {"LRA0": [object(), object()]}  # only 2 declared, 3 captured
        with pytest.raises(CaptureIdmapMismatchError):
            assert_idmap_completeness(idmap, cap)

    def test_sync_count_mismatch_ignored(self):
        """CMS lets the user specify arbitrary numbers of waits — count
        parity isn't a coverage property for SYNC/SNOP categories."""
        from dataflow_fixtures import make_swait, make_lr, make_capture
        from Tensile.Components.ScheduleCapture import BODY_LABEL_ML

        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_swait(slot=1, dscnt=0),
            make_swait(slot=2, dscnt=0),
            make_swait(slot=3, dscnt=0),
            make_swait(slot=4, dscnt=0),
            make_swait(slot=5, dscnt=0),
            make_swait(slot=6, dscnt=0),
        ])
        idmap = {"SYNC": [object(), object(), object()]}  # 3 vs 7 — ignored
        assert_idmap_completeness(idmap, cap)  # no raise

    def test_snop_count_mismatch_ignored(self):
        from dataflow_fixtures import make_capture
        from Tensile.Components.ScheduleCapture import (
            BODY_LABEL_ML, TaggedInstruction, SlotKey, WrappedInstruction,
        )
        from rocisa.instruction import SNop

        snops = [
            TaggedInstruction(
                wrapped=WrappedInstruction(SNop(waitState=0)), category="SNOP",
                slot=SlotKey(0, SLOT_KIND_MFMA, i, 0),
            )
            for i in range(5)
        ]
        cap = make_capture(BODY_LABEL_ML, snops)
        idmap = {"SNOP": [object(), object()]}  # 2 vs 5 — ignored
        assert_idmap_completeness(idmap, cap)


# =============================================================================
# rocm-libraries-vybd (F3) — capture-pipeline body invariants
# =============================================================================
# After F3 deletes the default-side leftover pack[*] / packPre[*] walk in
# `_loopBody` (KernelWriter.py:_captureDefaultSchedule branch), no
# capture-pipeline site should be able to:
#
#   (xbi0) emit the same Python rocisa instance into a single body twice
#          (would surface as `id(rocisa_inst)` collisions across two
#           TaggedInstructions in the same body), OR
#
#   (flpk) emit two distinct Python rocisa objects with identical
#          `WrappedInstruction.canonical_str(...)` under different category
#          tags within a single body (canonical-text cross-tagging).
#
# The canonical-text invariant is strictly stronger than the same-id
# invariant: flpk's pairs have different ids but identical canonical text;
# xbi0's pair has the same id (which trivially yields the same canonical
# text). Both are pinned here because:
#
#   1. xbi0's `id()` invariant is the cheapest specific catch for the
#      double-storage-buffer aliasing shape that pack[storeIdx*N] +
#      pack[1] historically produced.
#
#   2. flpk's canonical-text invariant catches the broader regression
#      surface: any future code path that builds two distinct Python
#      objects (different ids) for the same canonical instruction and
#      tags them with different categories would slip through the
#      `id()`-only check but be caught here.
#
# Both invariants are independent of the identity scheme used downstream
# in `compare_graphs` — they are pure capture-time properties of
# `LoopBodyCapture.instructions`.


def _assert_no_double_capture_in_body(body, body_label):
    """A single body's instructions must not contain two TaggedInstructions
    that wrap the same Python rocisa instance (xbi0 invariant)."""
    seen = {}
    for ti in body.instructions:
        ri = ti.wrapped.rocisa_inst
        if ri is None:
            continue
        rid = id(ri)
        prev = seen.get(rid)
        if prev is not None:
            raise AssertionError(
                f"{body_label}: rocisa_inst id={rid} "
                f"({type(ri).__name__}) appears twice — "
                f"first at slot={prev.slot} cat={prev.category}, "
                f"now at slot={ti.slot} cat={ti.category}. "
                f"This violates the no-double-capture invariant pinned by "
                f"rocm-libraries-xbi0 (the leftover-pack walk used to emit "
                f"the same Python leaf twice via storage-buffer aliasing; "
                f"rocm-libraries-vybd F3 deleted that walk)."
            )
        seen[rid] = ti


def _assert_no_canonical_text_cross_tagged_in_body(body, body_label):
    """A single body's instructions must not contain two TaggedInstructions
    that share the same canonical_str under DIFFERENT category tags (flpk
    invariant — strictly stronger than the same-id invariant above).

    Same canonical_str under the SAME category tag is allowed (e.g. SYNC
    `s_waitcnt(0)` legitimately repeats), so the check is keyed on
    (canonical_str -> first-seen category) and only fires when a later
    TaggedInstruction with the same canonical_str carries a DIFFERENT
    category.
    """
    from Tensile.Components.ScheduleCapture import WrappedInstruction
    seen = {}
    for ti in body.instructions:
        ri = ti.wrapped.rocisa_inst
        if ri is None:
            continue
        canon = WrappedInstruction.canonical_str(ri)
        prev = seen.get(canon)
        if prev is None:
            seen[canon] = ti
            continue
        if prev.category != ti.category:
            raise AssertionError(
                f"{body_label}: canonical text {canon!r} appears under "
                f"cat={ti.category} and earlier under cat={prev.category} "
                f"(slot_kind_now={ti.slot.slot_kind}, "
                f"slot_kind_prev={prev.slot.slot_kind}, "
                f"id_now={id(ri)}, id_prev={id(prev.wrapped.rocisa_inst)}). "
                f"This violates the no-canonical-text-cross-tagging "
                f"invariant pinned by rocm-libraries-flpk (the leftover-pack "
                f"walk used to re-tag distinct Python objects with the same "
                f"canonical text under PackA0 / PackA3 etc.; "
                f"rocm-libraries-vybd F3 deleted that walk)."
            )


class TestNoDoubleCaptureUnit:
    """Unit-level pin (xbi0): builder must not be allowed to silently store
    two TaggedInstructions that share the same rocisa_inst Python id within
    a single body. The post-finalize body's instructions list is walked.
    """

    def test_builder_with_no_aliased_leaves_passes(self):
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr, DSModifiers

        b = LoopBodyCaptureBuilder()
        for i in range(3):
            inst = DSLoadB128(dst=vgpr(8 + 4 * i, 4), src=vgpr(0, 1),
                              ds=DSModifiers(offset=64 + 16 * i))
            b.append(inst=inst, category="LRA0", subiter=0, mfma_index=i)
        cap = b.finalize()
        _assert_no_double_capture_in_body(cap, "synthetic-unit")

    def test_assertion_fires_when_same_inst_appended_twice(self):
        """If the builder somehow accumulates the SAME Python rocisa
        instance twice in a single body, the invariant check must fire.
        This is the symptom xbi0 produced via storage-buffer aliasing in
        the leftover-pack walk; vybd F3 deleted the walk so the symptom is
        no longer reachable from the production capture path, but the
        canary remains here against any future regression that
        re-introduces an aliased emission site."""
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr, DSModifiers

        b = LoopBodyCaptureBuilder()
        # SAME object referenced twice — exactly the aliasing shape xbi0
        # produced via pack[1].add(packCodeA) flowing through the leftover
        # walk twice.
        inst = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0, 1),
                          ds=DSModifiers(offset=64))
        b.append(inst=inst, category="LRA0", subiter=0, mfma_index=0)
        b.append(inst=inst, category="LRA0", subiter=0, mfma_index=1)
        cap = b.finalize()
        with pytest.raises(AssertionError) as exc:
            _assert_no_double_capture_in_body(cap, "synthetic-unit")
        assert "appears twice" in str(exc.value)
        assert "no-double-capture invariant" in str(exc.value)


class TestNoCanonicalTextCrossTaggedUnit:
    """Unit-level pin (flpk, strictly stronger than xbi0): builder must not
    accumulate two distinct Python rocisa objects that share the same
    `canonical_str` under DIFFERENT category tags within a single body.

    The leftover-pack walk historically built a fresh `leftover_idmap` over
    `PackCodeAAllIters[0..LoopIters-1]` and tagged Python objects under
    `PackA{u}` categories that could differ from the per-iter PRE_LOOP
    capture's tags for the canonically-equivalent instruction. vybd F3
    deletes the walk; this canary catches any future site that re-introduces
    the cross-tagging shape.
    """

    def test_builder_with_distinct_canonical_texts_passes(self):
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr, DSModifiers

        b = LoopBodyCaptureBuilder()
        for i in range(3):
            inst = DSLoadB128(dst=vgpr(8 + 4 * i, 4), src=vgpr(0, 1),
                              ds=DSModifiers(offset=64 + 16 * i))
            b.append(inst=inst, category=f"PackA{i}", subiter=0,
                     mfma_index=i)
        cap = b.finalize()
        _assert_no_canonical_text_cross_tagged_in_body(cap, "synthetic-unit")

    def test_same_canonical_text_same_category_passes(self):
        """Same canonical text under the SAME category is allowed (e.g.
        legitimate repeats inside one PackA0 emission group)."""
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr, DSModifiers

        b = LoopBodyCaptureBuilder()
        # Two DISTINCT Python objects with IDENTICAL canonical text.
        inst_a = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0, 1),
                            ds=DSModifiers(offset=64))
        inst_b = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0, 1),
                            ds=DSModifiers(offset=64))
        assert id(inst_a) != id(inst_b)
        b.append(inst=inst_a, category="LRA0", subiter=0, mfma_index=0)
        b.append(inst=inst_b, category="LRA0", subiter=0, mfma_index=1)
        cap = b.finalize()
        # Same category — must pass.
        _assert_no_canonical_text_cross_tagged_in_body(cap, "synthetic-unit")

    def test_assertion_fires_on_canonical_text_cross_tagging(self):
        """The flpk shape: two DISTINCT Python rocisa objects with
        IDENTICAL canonical text under DIFFERENT category tags within a
        single body. xbi0's same-id invariant would NOT catch this (the
        ids differ); the canonical-text invariant must."""
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr, DSModifiers

        b = LoopBodyCaptureBuilder()
        # Two DISTINCT Python objects, same canonical text.
        inst_a = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0, 1),
                            ds=DSModifiers(offset=64))
        inst_b = DSLoadB128(dst=vgpr(8, 4), src=vgpr(0, 1),
                            ds=DSModifiers(offset=64))
        assert id(inst_a) != id(inst_b)
        b.append(inst=inst_a, category="PackA0", subiter=0, mfma_index=0)
        b.append(inst=inst_b, category="PackA3", subiter=0, mfma_index=3)
        cap = b.finalize()
        with pytest.raises(AssertionError) as exc:
            _assert_no_canonical_text_cross_tagged_in_body(cap, "synthetic-unit")
        msg = str(exc.value)
        assert "canonical text" in msg
        assert "PackA0" in msg
        assert "PackA3" in msg

        # Sanity: xbi0's same-id check should NOT fire on this shape — the
        # two Python objects have different ids. This sanity assertion is
        # what makes the canonical-text invariant strictly stronger than
        # the same-id invariant.
        _assert_no_double_capture_in_body(cap, "synthetic-unit")
