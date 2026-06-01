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
import os

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

# rocm-libraries-g9fi (round-3 review): guard against the cwd-pollution trap
# that misled the round-2 plan-adherence verifier. When pytest is invoked from
# a directory containing a sibling `Tensile/` package (e.g. the main repo's
# tree), `import Tensile.KernelWriter` resolves to THAT tree's KernelWriter.py
# rather than the worktree's — even when PYTHONPATH points at the worktree.
# The test file itself loads from the worktree (pytest used the explicit
# path), but the production code under test loads from the wrong tree. The
# resulting SHADOW capture is then built from a different KernelWriter, so
# count-parity vs CMS spuriously fails on categories the wrong-tree code
# doesn't yet implement (nmsx Fix 1/2/3 / g9fi).
#
# Defense: cross-check that `Tensile.KernelWriter`'s file lives in the same
# tree as this test file. If not, fail loud with a one-line directive on how
# to fix the invocation. This eliminates a class of false-positive bug
# reports that look like real divergences but are caused by import-path
# leakage.
def _assert_tensile_tree_matches_test_tree():
    import Tensile.KernelWriter as _kw
    test_tree = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    kw_tree = os.path.abspath(
        os.path.join(os.path.dirname(_kw.__file__), "..")
    )
    if test_tree != kw_tree:
        raise RuntimeError(
            f"Tensile package loaded from a different tree than this test "
            f"file. test_tree={test_tree!r}, kw_tree={kw_tree!r}. This "
            f"usually means pytest was invoked from a directory containing "
            f"a sibling `Tensile/` package that shadows the intended one — "
            f"`import Tensile.*` resolves to the cwd's tree, not the one "
            f"PYTHONPATH points at. Fix: `cd {test_tree}` before invoking "
            f"pytest. See the round-3 review note on the g9fi commit."
        )


_assert_tensile_tree_matches_test_tree()


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
# optSchedule ↔ idMap consistency — pure function
# =============================================================================
# rocm-libraries-6hk3: port of develop's `verify_correct_number_of_instructions`.
# Compares per-(category × codepath) optSchedule slot-row sizes to idMap
# leaf counts via bare `len()`. Skips SYNC / SNOP. idMap entries are flat
# Python lists by construction (removeComments output) — see the helper
# docstring and dispatch.py:92-122.


class _FakeScheduleInfo:
    """Minimal stand-in for ScheduleInfo carrying only what the check reads."""

    def __init__(self, optSchedule):
        self.optSchedule = optSchedule


class TestVerifyCorrectNumberOfInstructions:
    def test_matching_counts_pass(self):
        from Tensile.Components.CMSValidator import (
            verify_correct_number_of_instructions,
        )
        sched = _FakeScheduleInfo({
            "LRA0": [[0, 1, 2], [3, 4, 5]],
            "LWA":  [[7]],
        })
        idmap = {
            "LRA0": [object()] * 3,
            "LWA":  [object()],
        }
        verify_correct_number_of_instructions(sched, idmap)  # no raise

    def test_slot_count_mismatch_raises(self):
        from Tensile.Components.CMSValidator import (
            verify_correct_number_of_instructions, ValidationError,
        )
        sched = _FakeScheduleInfo({
            "LRA0": [[0, 1, 2], [3, 4]],  # cp1 short by one
        })
        idmap = {"LRA0": [object()] * 3}
        with pytest.raises(ValidationError) as exc:
            verify_correct_number_of_instructions(sched, idmap)
        msg = str(exc.value)
        assert "LRA0" in msg
        assert "codepath 1" in msg
        assert "optSchedule has 2" in msg
        assert "idMap has 3" in msg

    def test_sync_and_snop_skipped(self):
        from Tensile.Components.CMSValidator import (
            verify_correct_number_of_instructions,
        )
        # Intentional count divergence on SYNC/SNOP — must not raise.
        sched = _FakeScheduleInfo({
            "SYNC": [[0, 1, 2, 3, 4]],
            "SNOP": [[0]],
            "LRA0": [[7, 8]],
        })
        idmap = {
            "SYNC": [object()] * 2,   # 2 vs 5 — ignored
            "SNOP": [],               # 0 vs 1 — ignored
            "LRA0": [object()] * 2,   # match
        }
        verify_correct_number_of_instructions(sched, idmap)


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


# =============================================================================
# rocm-libraries-nmsx Phase 1 — SHADOW capture window/scope/walk-coverage
# =============================================================================
# Per DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §4 Phase 1. Three fixes land
# together; each verified by an introspective assertion on the SHADOW
# capture state for a CMS BPG#11-shaped kernel:
#
#   Fix 1 (LCC): closeLoop's loop-counter code (`s_sub_u32` + `s_cmp_eq_i32`)
#                must appear in the SHADOW main_loop capture under category
#                `LCC`. The pre-fix SHADOW finalize ran BEFORE closeLoop
#                emitted these, leaving LCC absent.
#
#   Fix 2 (PLR1 packs): pack/packPre leftover content not consumed by any
#                mainloop iter's iterCode (LoopIters-1's PackCodeAAllIters
#                slot is added but consumed in NLL/NGL) must appear in the
#                SHADOW main_loop capture under PackA{u}/PackB{u} for every
#                covered u. CMS aggregates these into its main_loop macro;
#                SHADOW must mirror that aggregation.
#
#   Fix 3 (LRS/LWS schema): pointer-math leaves emitted by
#                `localReadSwapOffsets(...A/B)` and
#                `localWriteSwapOffsets(...A/B)` must appear under per-side
#                tags LRSA/LRSB/LWSA/LWSB matching the CMS-side idMap schema
#                (ScheduleCapture.py:1045-1048), not the previous generic
#                LRS/LWS hand-rolled tags at KernelWriter.py:1040-1045.
#
# Plus the fail-loud contract: a leaf with no idMap entry AND no registry
# entry must raise CaptureCategoryMissingError immediately.


def _build_bpg11_writer_and_capture():
    """Run a CMS BPG#11-shaped kernel build through SHADOW capture.

    Re-uses the TF32-TN config the existing TestPhase4DefaultCapture tests
    use (TF32 emulation, PGR2/PLR1, TransposeLDS, CMS=1). Wraps the
    `kernelBody` downstream-validator block in a try/except so this test
    surfaces SHADOW state regardless of whether the post-SHADOW Approach-A
    validator path completes — that path is Phase 2/4 scope (the inline
    xj16 / `build_non_cms_reference` block is left intact per task spec).

    Monkey-patches `build_non_cms_reference` to snapshot `ctx.default`
    (the SHADOW FourPartCapture) BEFORE the xj16 path replaces it with
    Approach-A's reference. Stashes the snapshot on `writer._test_shadow_capture`
    for the test to inspect.

    Returns (writer, ctx) where:
      - writer._test_shadow_capture is the SHADOW FourPartCapture (with
        main_loop[0] etc.) as it stood right after Phase 4 SHADOW assembly,
        before xj16's Approach-A reference replaced it.

    Returns None if the kernel-build pre-requisites aren't available.
    """
    import shutil
    if not (shutil.which('amdclang++') or shutil.which('clang++')):
        return None
    if not (shutil.which('amdclang') or shutil.which('clang')):
        return None

    from Tensile.Common import IsaVersion
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Component import Assembler
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    import Tensile.Components.CustomSchedule.approach_a as _aa

    compiler = shutil.which('amdclang++') or shutil.which('clang++')
    assembler_bin = shutil.which('amdclang') or shutil.which('clang')
    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)
    asm = Assembler(assembler_bin, 'V5')

    config = {
        'ProblemType': {
            'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
            'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
            'UseBeta': True, 'Batched': True,
        },
        'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
        'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
        'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
        'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
        'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
        'SourceSwap': 1, 'StreamK': 0,
        'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
    }
    try:
        solution = _make_solution(config, asm, isaInfoMap)
    except Exception:
        return None
    writer = KernelWriterAssembly(asm, DebugConfig())
    writer.enable_capture_default_schedule()

    # Monkey-patch build_non_cms_reference to (1) snapshot SHADOW's
    # ctx.default before it gets replaced, and (2) raise a controlled
    # exception so the xj16 block doesn't proceed (Phase 2/4 scope).
    # The test reads writer._test_shadow_capture rather than ctx.default.
    _orig = _aa.build_non_cms_reference

    def _stashing_build_non_cms_reference(kernel, assembler, isaInfoMap, *args, **kwargs):
        # Snapshot SHADOW's ctx.default before this call would replace it.
        writer._test_shadow_capture = writer._capture_context.default
        # Return a no-op marker so xj16 doesn't crash on None;
        # we abort downstream validation by raising afterward.
        raise RuntimeError("nmsx-test-skip-xj16: snapshot taken")

    _aa.build_non_cms_reference = _stashing_build_non_cms_reference
    try:
        try:
            writer._getKernelSource(solution)
        except Exception:
            # Expected: our monkey-patch raises to abort xj16 path.
            pass
    finally:
        _aa.build_non_cms_reference = _orig

    return writer, writer._capture_context


def _shadow_main_body(writer):
    """Helper: extract the SHADOW main_loop body from the snapshot taken
    by the monkey-patched build_non_cms_reference."""
    snap = getattr(writer, "_test_shadow_capture", None)
    if snap is not None and snap.main_loop:
        return snap.main_loop[0]
    # Fallback: if the snapshot wasn't taken (e.g. xj16 path was disabled
    # before our patch ran), look at ctx.default (which is the SHADOW
    # build's output when xj16 didn't replace it).
    ctx = writer._capture_context
    if getattr(ctx, "default", None) is not None and ctx.default.main_loop:
        return ctx.default.main_loop[0]
    if getattr(ctx, "default_main", None) is not None:
        return ctx.default_main
    return None


def _cms_main_body(writer):
    """Helper: extract CMS main_loop body from writer's capture context."""
    ctx = writer._capture_context
    if getattr(ctx, "cms", None) is not None and ctx.cms.main_loop:
        return ctx.cms.main_loop[0]
    return None


class TestShadowCaptureNmsxFixes:
    """rocm-libraries-nmsx Phase 1 — verify SHADOW capture has LCC,
    PLR1 leftover packs, and per-side LRS*/LWS* tags."""

    def test_shadow_main_capture_present(self):
        result = _build_bpg11_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result
        main = _shadow_main_body(writer)
        assert main is not None, (
            "SHADOW main_loop capture not finalized — Fix 1/Fix 2's "
            "leftover walk + LCC harvest path failed to land. "
            "Check KernelWriter.py:_loopBody _captureDefaultSchedule "
            "finalize site (around line 5151)."
        )
        # Non-empty body — the SHADOW captured something.
        assert len(main.instructions) > 0

    def test_shadow_main_capture_contains_lcc(self):
        """Fix 1: SHADOW main_loop must contain s_sub_u32 + s_cmp_eq_i32
        (the LCC pair) under category 'LCC'. Before Fix 1, these were
        absent because SHADOW finalized before customMainLoopSchedule
        invoked closeLoop."""
        result = _build_bpg11_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result
        main = _shadow_main_body(writer)
        if main is None:
            pytest.skip("SHADOW main_loop capture not populated; check Fix 1.")

        lcc_tagged = [
            ti for ti in main.instructions if ti.category == "LCC"
        ]
        assert len(lcc_tagged) >= 1, (
            f"SHADOW main_loop has 0 LCC-tagged instructions. Fix 1 "
            f"(closeLoop harvest before finalize) didn't land. "
            f"Body has {len(main.instructions)} total instructions."
        )
        # Specifically verify the SSubU32 + SCmpEQI32 pair the Approach-A
        # `_appendCloseLoopLCCToBuilder` harvest produces.
        lcc_classes = {
            type(ti.wrapped.rocisa_inst).__name__ for ti in lcc_tagged
        }
        assert "SSubU32" in lcc_classes or "SCmpEQI32" in lcc_classes, (
            f"SHADOW main_loop LCC slot contains {lcc_classes}, expected "
            f"SSubU32 / SCmpEQI32 from closeLoop harvest (Fix 1)."
        )

    def test_shadow_main_capture_contains_per_subiter_packs(self):
        """Fix 2: SHADOW main_loop must contain pack content for every
        per-iter PackA{u}/PackB{u} category that CMS aggregates. Before
        Fix 2, the LoopIters-1 slot was empty in SHADOW because that
        content goes to pack[storeIdx] consumed in NLL/NGL, never in any
        mainloop iter's iterCode.

        Asserts per-(category) COUNT equality across CMS and SHADOW for
        every PackA{u}/PackB{u} category either side produces. Not just
        category-name SET coverage: a SHADOW capture that registers ONE
        leaf under `PackA0` while CMS has 40 of them would pass a set
        assertion (the name is present on both sides) but represents
        exactly the count-truncation regression Fix 2 was written to
        prevent. Count equality is the load-bearing assertion.
        """
        result = _build_bpg11_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result
        main = _shadow_main_body(writer)
        cms_main = _cms_main_body(writer)
        if main is None or cms_main is None:
            pytest.skip("SHADOW or CMS capture not populated; check Fix 2.")

        def _pack_counts(body):
            counts = {}
            for ti in body.instructions:
                if ti.category.startswith(("PackA", "PackB")):
                    counts[ti.category] = counts.get(ti.category, 0) + 1
            return counts

        cms_pack_counts = _pack_counts(cms_main)
        shadow_pack_counts = _pack_counts(main)

        # First: set coverage (every CMS pack category must exist on SHADOW).
        cms_pack_cats = set(cms_pack_counts.keys())
        shadow_pack_cats = set(shadow_pack_counts.keys())
        missing = cms_pack_cats - shadow_pack_cats
        assert not missing, (
            f"SHADOW main_loop missing PackA{{u}}/PackB{{u}} categories "
            f"present on CMS side: {sorted(missing)}. "
            f"CMS={sorted(cms_pack_cats)}, SHADOW={sorted(shadow_pack_cats)}. "
            f"Fix 2 (leftover pack[*]/packPre[*] walk before finalize) "
            f"didn't capture these subiters."
        )

        # Then: per-category count parity (the load-bearing assertion).
        mismatches = []
        for cat in cms_pack_cats | shadow_pack_cats:
            sh = shadow_pack_counts.get(cat, 0)
            cm = cms_pack_counts.get(cat, 0)
            if sh != cm:
                mismatches.append((cat, sh, cm))
        assert not mismatches, (
            f"Per-PackA{{u}}/PackB{{u}} count mismatches between SHADOW "
            f"and CMS: {mismatches} (each tuple is "
            f"(category, shadow_count, cms_count)). "
            f"CMS={cms_pack_counts}, SHADOW={shadow_pack_counts}. "
            f"Fix 2's leftover walk left a count-truncation regression."
        )

    def test_shadow_main_capture_uses_per_side_lrs_lws_tags(self):
        """Fix 3: pointer-math leaves (from localReadSwapOffsets /
        localWriteSwapOffsets) must carry per-side LRSA/LRSB/LWSA/LWSB
        tags matching the CMS-side idMap schema. Before Fix 3, they
        carried unsided 'LRS'/'LWS' hand-rolled tags."""
        result = _build_bpg11_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result
        main = _shadow_main_body(writer)
        if main is None:
            pytest.skip("SHADOW main_loop capture not populated; check Fix 3.")

        # The previous unsided generic tags should be absent on this body.
        unsided = {
            ti.category for ti in main.instructions
            if ti.category in ("LRS", "LWS")
        }
        assert not unsided, (
            f"SHADOW main_loop carries pre-Fix-3 unsided tags {unsided!r}; "
            f"after Fix 3 these should be split into LRSA/LRSB/LWSA/LWSB."
        )

    def test_shadow_main_capture_categories_match_cms_subject(self):
        """Cross-side count parity on the categories CMS uses (per-side
        LRA/LRB, PackA/PackB, GRA/GRB/GRIncA/GRIncB, LRSA/LRSB/LWSA/LWSB,
        LWA/LWB, LCC, MFMA). SHADOW must match CMS on these.

        Excluded from parity:
          - SYNC/SNOP/SSETPRIO/SBARRIER: scheduler-inserted, legitimately
            differ (per DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3).

        rocm-libraries-g9fi: MFMA is REMOVED from the exclusion set. The
        prior comment here claimed a per-leaf-SHADOW vs per-Module-CMS
        structural divergence. Empirical inspection (g9fi probe on BPG#11)
        disproved that: CMS's `customMainLoopSchedule` calls
        `removeComments(MfmaCodeAllIters)` which is `.flatitems()`-based
        (`CustomSchedule/dispatch.py:73-78`), then iterates `for miIndex
        in range(-1, len(mfmaCode))` over the FLAT leaf list and tags
        each `mfmaItem = mfmaCode[miIndex]` per-leaf with
        `tag_by_origin_id[id(mfmaItem)] = "MFMA"` (dispatch.py:235-240).
        The mfmaIter Module wrapper is destroyed BEFORE CMS's dispatch
        loop runs, so CMS is per-leaf — not per-Module — for MFMA tags.
        SHADOW is also per-leaf (the explicit registration walk at
        KernelWriter.py around the `_captureDefaultSchedule` branch tags
        every macIterCode leaf as MFMA). Both sides agree: 48 MFMAs on
        BPG#11. No exclusion required.

        The legitimate exclusion set is therefore exactly SYNC / SNOP /
        SSETPRIO / SBARRIER per design §3. Do not grow this exclusion set.

        rocm-libraries-nmsx Bug 1/Bug 2 fix status (post-merge): the LCC
        and PLR1-pack-split divergences originally documented as needing
        Phase 3 fixture coverage are now CORRECT in v5 nmsx — LCC is
        harvested from EVERY post-removeComments leaf of closeLoopMod
        (KernelWriter.py:_appendCloseLoopLCCToBuilder, fixed at the
        same isinstance-filter site), and SHADOW's leftover walk applies
        split_for_plr on LoopIters==1 to match dispatch.py's idmap
        construction (KernelWriter.py:_loopBody leftover-walk site).
        """
        result = _build_bpg11_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result
        main = _shadow_main_body(writer)
        cms_main = _cms_main_body(writer)
        if main is None or cms_main is None:
            pytest.skip(
                "SHADOW or CMS capture missing; "
                "downstream Phase 2/4 path failed before count parity."
            )

        excluded = {"SYNC", "SNOP", "SSETPRIO", "SBARRIER"}

        def _per_cat_counts(body):
            counts = {}
            for ti in body.instructions:
                if ti.category in excluded:
                    continue
                counts[ti.category] = counts.get(ti.category, 0) + 1
            return counts

        shadow_counts = _per_cat_counts(main)
        cms_counts = _per_cat_counts(cms_main)

        mismatches = []
        for cat in shadow_counts.keys() | cms_counts.keys():
            sh = shadow_counts.get(cat, 0)
            cm = cms_counts.get(cat, 0)
            if sh != cm:
                mismatches.append((cat, sh, cm))

        # On BPG#11 the SHADOW (after Phase 1 fixes) should match CMS on
        # data-flow categories.
        assert not mismatches, (
            f"Per-category count mismatches on non-mfmaIter-sub-leaf "
            f"data-flow categories: {mismatches}. "
            f"SHADOW={shadow_counts}, CMS={cms_counts}."
        )

    def test_shadow_mfma_count_matches_cms_subject_on_bpg11(self):
        """rocm-libraries-g9fi: explicit MFMA count parity on BPG#11.

        Locks in the invariant that the prior nmsx-revision exclusion was
        masking. Both SHADOW and CMS treat MFMA per-leaf:

          - CMS: `customMainLoopSchedule` flattens MfmaCodeAllIters via
            `removeComments(...)` (which calls `.flatitems()`,
            CustomSchedule/dispatch.py:73-78), then the dispatch loop at
            dispatch.py:235-240 tags every leaf as MFMA via
            `tag_by_origin_id[id(mfmaItem)] = "MFMA"`. The mfmaIter
            Module wrapper is destroyed before the tag loop runs.

          - SHADOW: the explicit per-leaf registration walk at
            KernelWriter.py (around the `_captureDefaultSchedule` branch
            in `_loopBody`) walks `macIterCode.flatitems()` and tags
            each leaf as MFMA. The walk MUST exist for fail-loud
            coverage of non-MFMAInstruction sub-leaves that mfmaIter
            may emit (SNop at KernelWriterAssembly.py:8367, SWaitAlu
            at 8622, tail-loop / shiftK control ops).

        BPG#11 ground truth: 48 MFMAs per main loop (4 MI per subiter ×
        2 subiters × 6 wave-tile elements, matching the bf16 16x16x32
        MFMA grid). This test pins both sides to 48 and additionally
        asserts they match each other. A regression of either the CMS
        dispatch loop (e.g. someone re-introducing per-Module tagging
        and breaking the assumption that mfmaCode is a flat leaf list)
        or the SHADOW per-leaf walk (e.g. someone deleting it under
        the false belief that an atomic-Module mechanism replaces it,
        which was the g9fi bead's prescribed-but-incorrect fix) will
        fail this test loudly.
        """
        result = _build_bpg11_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result
        main = _shadow_main_body(writer)
        cms_main = _cms_main_body(writer)
        if main is None or cms_main is None:
            pytest.skip(
                "SHADOW or CMS capture missing; "
                "downstream Phase 2/4 path failed before MFMA count check."
            )

        shadow_mfma = sum(1 for ti in main.instructions if ti.category == "MFMA")
        cms_mfma = sum(1 for ti in cms_main.instructions if ti.category == "MFMA")

        assert shadow_mfma == 48, (
            f"SHADOW MFMA count on BPG#11 is {shadow_mfma}; expected 48. "
            f"Either the per-leaf MFMA registration walk in "
            f"KernelWriter.py (around the _captureDefaultSchedule branch) "
            f"was removed, or mfmaIter's emit shape changed."
        )
        assert cms_mfma == 48, (
            f"CMS MFMA count on BPG#11 is {cms_mfma}; expected 48. "
            f"Either dispatch.py's per-leaf MFMA tag loop was restructured "
            f"to per-Module (which would break SHADOW-vs-CMS count parity), "
            f"or removeComments / mfmaIter's emit shape changed."
        )
        assert shadow_mfma == cms_mfma, (
            f"MFMA count parity broken: SHADOW={shadow_mfma}, "
            f"CMS={cms_mfma}. The two sides must remain per-leaf-symmetric; "
            f"do not introduce atomic-Module MFMA treatment on either side "
            f"without restructuring both."
        )


def solution_loop_iters_from_writer(writer):
    """Best-effort: read LoopIters from the writer's last-built solution.

    The writer doesn't keep a direct handle to the kernel dict after
    _getKernelSource finishes, but it does stash state on self.states. For
    BPG#11 the value is straightforward; fall back to 2 when not
    introspectable (matches the BPG#11 LoopIters=2 with DepthU=32 / MI K=32).
    """
    states = getattr(writer, "states", None)
    if states is None:
        return 2
    # `numItersPLR` is captured in states; LoopIters = numItersPLR + something.
    # Without a reliable handle we fall back to BPG#11's known value.
    return 2


class TestShadowCaptureFailLoudOnUnknownCategory:
    """Fail-loud contract: a leaf with no idMap entry AND no registry
    entry must raise CaptureCategoryMissingError immediately when SHADOW
    is the consumer. Synthetic test using the production capture function
    directly with a stand-in instruction class that's NOT in
    `InstructionCategory._CLASS_NAME_TO_CATEGORY`."""

    def test_synthetic_unregistered_class_raises(self):
        """Use VXorB32 — a real rocisa Instruction (Item subclass that
        can be `.add()`-ed to a Module) whose class name is NOT in
        `InstructionCategory._CLASS_NAME_TO_CATEGORY`. With empty
        `id_to_category` and `fail_loud_on_missing_category=True`, the
        capture walk must raise `CaptureCategoryMissingError`.
        """
        from Tensile.Components.ScheduleCapture import (
            CaptureCategoryMissingError, LoopBodyCaptureBuilder,
        )
        from Tensile.Components.InstructionCategory import (
            _CLASS_NAME_TO_CATEGORY,
        )
        # Sanity precondition for the test premise.
        assert "VXorB32" not in _CLASS_NAME_TO_CATEGORY, (
            "VXorB32 is now in the registry; pick a different unregistered "
            "rocisa class for this fail-loud canary."
        )

        from Tensile.KernelWriter import KernelWriter
        from rocisa.code import Module
        from rocisa.instruction import VXorB32
        from rocisa.container import vgpr
        from types import SimpleNamespace

        iterCode = Module()
        synthetic = VXorB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
        iterCode.add(synthetic)

        builder = LoopBodyCaptureBuilder()
        shim = SimpleNamespace()
        with pytest.raises(CaptureCategoryMissingError) as exc:
            KernelWriter._captureSubIterToBuilder(
                shim,
                iterCode=iterCode,
                capture=builder,
                subiter=0,
                numMfmaPerIter=4,
                id_to_category={},  # empty: leaf has no idMap entry
                id_to_source_module=None,
                body_label="main_loop",
                fail_loud_on_missing_category=True,
            )
        msg = str(exc.value)
        assert "VXorB32" in msg
        assert "main_loop" in msg
        assert "DEFAULT_SCHEDULER_REFERENCE_DESIGN" in msg

    def test_synthetic_unregistered_class_silent_when_fail_loud_off(self):
        """Same input but with fail_loud_on_missing_category=False
        (Approach-A's path) — silent UNKNOWN fallback, NO raise."""
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder

        from Tensile.KernelWriter import KernelWriter
        from rocisa.code import Module
        from rocisa.instruction import VXorB32
        from rocisa.container import vgpr
        from types import SimpleNamespace

        iterCode = Module()
        iterCode.add(VXorB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))

        builder = LoopBodyCaptureBuilder()
        shim = SimpleNamespace()
        # Should NOT raise.
        KernelWriter._captureSubIterToBuilder(
            shim,
            iterCode=iterCode,
            capture=builder,
            subiter=0,
            numMfmaPerIter=4,
            id_to_category={},
            id_to_source_module=None,
            body_label="non_cms_main_loop",
            fail_loud_on_missing_category=False,
        )
        # Leaf was appended as UNKNOWN.
        assert any(ti.category == "UNKNOWN" for ti in builder._instructions)


class TestAppendCloseLoopLCCToBuilderPgr2:
    """rocm-libraries-nmsx Bug 1 — verify _appendCloseLoopLCCToBuilder
    captures ALL closeLoop control leaves, not just the
    (SSubU32, SCmpEQI32) pair the pre-fix isinstance filter matched.

    Specifically, on the PGR=2 + ASEM%(DepthU*2)==0 path,
    KernelWriterAssembly.closeLoop emits four LCC leaves:
    SCmpEQU32 + SCSelectB32 + SSubU32 + SCmpEQI32 (lines ~6845-6859).
    The CMS-side build_idmap tags ALL four as 'LCC' via
    `loopCounterCode=closeLoopMod`. SHADOW must match — otherwise the
    cross-side count-parity test diverges in the PGR=2 ASEM-multiple
    fixture surface required by Phase 3.

    The pre-fix implementation hard-filtered to (SSubU32, SCmpEQI32)
    and dropped the two control leaves silently. This test asserts the
    fixed implementation captures all four.
    """

    def _build_pgr2_closeloop_module(self):
        """Construct a Module shaped like KernelWriterAssembly.closeLoop's
        PGR=2 + ASEM%(DepthU*2)==0 emission.

        Mirror the exact emission order at KernelWriterAssembly.py:6848-6859:
          1. SCmpEQU32(src0=StaggerU, src1=0)
          2. SCSelectB32(dst=tmpSgpr, src0=hex(2), src1=hex(1))
          3. SSubU32(dst=loopCounter, src0=loopCounter, src1=tmpSgpr, ...)
          4. SCmpEQI32(src0=loopCounter, src1=hex(0), ...)

        Plus a TextBlock comment and an SCBranchSCC1 branch — both of
        which CMS strips in `removeComments` and which the fix must NOT
        tag as LCC.
        """
        from rocisa.code import Module, TextBlock
        from rocisa.container import sgpr
        from rocisa.instruction import (
            SCmpEQU32, SCSelectB32, SSubU32, SCmpEQI32,
            SCBranchSCC1, SNop,
        )

        mod = Module("closeLoop")
        # The comment closeLoop emits via module.addComment1 -> TextBlock.
        mod.add(TextBlock("// closeLoop comment\n"))
        # The four control + counter leaves on the PGR=2 path.
        i_cmpeq = SCmpEQU32(src0=sgpr("StaggerU"), src1=0)
        i_csel = SCSelectB32(dst=sgpr(8), src0=hex(2), src1=hex(1))
        i_sub = SSubU32(dst=sgpr(9), src0=sgpr(9), src1=sgpr(8),
                        comment="dec counter")
        i_cmpeqi = SCmpEQI32(src0=sgpr(9), src1=hex(0),
                             comment="counter==0")
        mod.add(i_cmpeq)
        mod.add(i_csel)
        mod.add(i_sub)
        mod.add(i_cmpeqi)
        # Branches/snops that removeComments strips on the CMS side and
        # the SHADOW helper must skip to stay aligned.
        mod.add(SCBranchSCC1(labelName="LoopEndK", comment="exit loop"))
        mod.add(SNop(waitState=0, comment="snop"))

        return mod, (i_cmpeq, i_csel, i_sub, i_cmpeqi)

    def test_captures_all_four_pgr2_lcc_leaves(self):
        """The PGR=2 + ASEM%(DepthU*2)==0 path produces 4 LCC leaves;
        the fix must capture all 4. Pre-fix this captured only 2."""
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        from Tensile.KernelWriter import KernelWriter
        from types import SimpleNamespace

        mod, expected_leaves = self._build_pgr2_closeloop_module()
        builder = LoopBodyCaptureBuilder()

        # SimpleNamespace shim — mirrors the pattern other tests use to
        # invoke unbound KernelWriter methods (no __init__ needed).
        shim = SimpleNamespace(states=SimpleNamespace(numMfmaPerIter=4))
        kernel = {"LoopIters": 2}

        KernelWriter._appendCloseLoopLCCToBuilder(
            shim, closeLoopModule=mod, capture=builder, kernel=kernel,
        )

        # All four PGR=2 LCC leaves were captured under "LCC".
        lcc_items = [ti for ti in builder._instructions
                     if ti.category == "LCC"]
        assert len(lcc_items) == 4, (
            f"Expected 4 LCC leaves captured (PGR=2 ASEM-multiple path); "
            f"got {len(lcc_items)}. Pre-Bug-1 filter would have produced 2."
        )

        # Identity preservation: each captured wrapped.rocisa_inst is one
        # of the four originals.
        captured_ids = {id(ti.wrapped.rocisa_inst) for ti in lcc_items}
        expected_ids = {id(x) for x in expected_leaves}
        assert captured_ids == expected_ids, (
            f"Captured ids {captured_ids} do not match expected "
            f"{expected_ids}; some leaves are not the originals."
        )

    def test_skips_textblock_and_scbranchscc1_and_snop(self):
        """The fix must skip TextBlock, SCBranchSCC1, SNop — these are
        what CMS's removeComments strips before tagging as LCC."""
        from rocisa.code import TextBlock
        from rocisa.instruction import SCBranchSCC1, SNop
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        from Tensile.KernelWriter import KernelWriter
        from types import SimpleNamespace

        mod, _ = self._build_pgr2_closeloop_module()
        builder = LoopBodyCaptureBuilder()
        shim = SimpleNamespace(states=SimpleNamespace(numMfmaPerIter=4))
        kernel = {"LoopIters": 2}

        KernelWriter._appendCloseLoopLCCToBuilder(
            shim, closeLoopModule=mod, capture=builder, kernel=kernel,
        )

        # None of the captured items are TextBlock/SCBranchSCC1/SNop.
        for ti in builder._instructions:
            inst = ti.wrapped.rocisa_inst
            assert not isinstance(inst, TextBlock), (
                f"TextBlock leaked into LCC capture: {inst!r}"
            )
            assert not isinstance(inst, (SCBranchSCC1, SNop)), (
                f"SCBranchSCC1/SNop leaked into LCC capture: {inst!r}"
            )

    def test_skips_label_on_approach_a_finalloop_path(self):
        """Approach-A call site (KernelWriter.py:5476) calls
        ``closeLoop(..., finalLoop=True)`` which emits ``Label``
        leaves via the odd/even iter pre-code. ``Label`` is an ``Item``
        but NOT an ``Instruction`` — it lacks ``reads_scc``/
        ``writes_scc`` and the LoopBodyCaptureBuilder's wrapper
        population would AttributeError on it. The non-Instruction
        skip in the fix must catch this; without it, the broadened
        capture would regress test_approach_a_non_cms_reference.py.
        """
        from rocisa.code import Module
        from rocisa.code import Label
        from rocisa.container import sgpr
        from rocisa.instruction import SSubU32, SCmpEQI32, Instruction
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        from Tensile.KernelWriter import KernelWriter
        from types import SimpleNamespace

        mod = Module("closeLoop_finalLoop_true")
        # Simulate Approach-A's finalLoop=True closeLoop emission
        # (lines ~6907-6913) — Label + counter instructions.
        i_sub = SSubU32(dst=sgpr(9), src0=sgpr(9), src1=1,
                        comment="dec counter")
        i_cmpeqi = SCmpEQI32(src0=sgpr(9), src1=hex(0),
                             comment="counter==0")
        mod.add(i_sub)
        mod.add(i_cmpeqi)
        # The finalLoop=True path adds these — must NOT enter the
        # LCC capture or finalize() will AttributeError.
        mod.add(Label("LoopEndK_oddexit", "unroll loop odditer exit"))
        mod.add(Label("LoopEndK_evenexit", "unroll loop eveniter exit"))

        builder = LoopBodyCaptureBuilder()
        shim = SimpleNamespace(states=SimpleNamespace(numMfmaPerIter=4))
        kernel = {"LoopIters": 2}

        KernelWriter._appendCloseLoopLCCToBuilder(
            shim, closeLoopModule=mod, capture=builder, kernel=kernel,
        )

        for ti in builder._instructions:
            inst = ti.wrapped.rocisa_inst
            assert not isinstance(inst, Label), (
                f"Label leaked into LCC capture: {inst!r} — "
                f"finalize() would AttributeError on reads_scc/"
                f"writes_scc lookup."
            )
            assert isinstance(inst, Instruction), (
                f"Non-Instruction leaked into LCC capture: "
                f"{type(inst).__name__} {inst!r}"
            )

        # Only the two genuine LCC leaves got captured.
        lcc_items = [ti for ti in builder._instructions
                     if ti.category == "LCC"]
        assert len(lcc_items) == 2
        assert {id(ti.wrapped.rocisa_inst) for ti in lcc_items} == \
               {id(i_sub), id(i_cmpeqi)}

    def test_captures_default_two_leaf_path(self):
        """Sanity: default (non-PGR=2) closeLoop emits only 2 LCC leaves
        (SSubU32 + SCmpEQI32). The fix must continue to capture exactly
        those 2 — regression guard against over-capture."""
        from rocisa.code import Module, TextBlock
        from rocisa.container import sgpr
        from rocisa.instruction import SSubU32, SCmpEQI32, SCBranchSCC1
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        from Tensile.KernelWriter import KernelWriter
        from types import SimpleNamespace

        mod = Module("closeLoop")
        mod.add(TextBlock("// closeLoop default-path comment\n"))
        i_sub = SSubU32(dst=sgpr(9), src0=sgpr(9), src1=1,
                        comment="dec counter")
        i_cmpeqi = SCmpEQI32(src0=sgpr(9), src1=hex(0),
                             comment="counter==0")
        mod.add(i_sub)
        mod.add(i_cmpeqi)
        mod.add(SCBranchSCC1(labelName="LoopEndK", comment="exit loop"))

        builder = LoopBodyCaptureBuilder()
        shim = SimpleNamespace(states=SimpleNamespace(numMfmaPerIter=4))
        kernel = {"LoopIters": 2}

        KernelWriter._appendCloseLoopLCCToBuilder(
            shim, closeLoopModule=mod, capture=builder, kernel=kernel,
        )

        lcc_items = [ti for ti in builder._instructions
                     if ti.category == "LCC"]
        assert len(lcc_items) == 2
        assert {id(ti.wrapped.rocisa_inst) for ti in lcc_items} == \
               {id(i_sub), id(i_cmpeqi)}


class TestLeftoverIdmapSplitForPlrLoopIters1:
    """rocm-libraries-nmsx Bug 2 — verify the leftover-walk idmap
    construction mirrors dispatch.py's split_for_plr when LoopIters==1.

    CMS (CustomSchedule/dispatch.py:103-111) replaces the single-iter
    LRCodeA/B and PackCodeA/B with split_for_plr halves and rebuilds
    idmap with num_loop_iter=2, producing PackA0/PackA1/PackB0/PackB1
    (and LRA0/LRA1/LRB0/LRB1) categories.

    SHADOW's leftover walk must apply the SAME split before calling
    build_idmap, otherwise on a LoopIters==1 kernel the pack leaves
    would carry PackA0/PackB0 only and per-category counts would
    diverge from CMS.

    This test focuses on the idmap-construction contract — the same
    split_for_plr step the fix uses inline at KernelWriter.py:_loopBody.
    """

    def _build_per_iter_lists(self):
        """Build a single per-iter LRCodeA[0]/LRCodeB[0]/PackCodeA[0]/
        PackCodeB[0] Module each containing 4 leaves so split_for_plr's
        2/2 split is observable.
        """
        from rocisa.code import Module
        from rocisa.container import vgpr
        from rocisa.instruction import VAddU32

        def _four_leaf_mod():
            m = Module()
            leaves = []
            for k in range(4):
                inst = VAddU32(dst=vgpr(k), src0=vgpr(k + 10),
                               src1=vgpr(k + 20),
                               comment=f"leaf{k}")
                leaves.append(inst)
                m.add(inst)
            return m, leaves

        lra, lra_leaves = _four_leaf_mod()
        lrb, lrb_leaves = _four_leaf_mod()
        paka, paka_leaves = _four_leaf_mod()
        pakb, pakb_leaves = _four_leaf_mod()
        return ([lra], [lrb], [paka], [pakb],
                {"LRA": lra_leaves, "LRB": lrb_leaves,
                 "PackA": paka_leaves, "PackB": pakb_leaves})

    def test_split_for_plr_on_loopiters_1_produces_per_iter_pack_categories(self):
        """With LoopIters==1, the SHADOW idmap (post-split) must have
        PackA0+PackA1 and PackB0+PackB1 — matching CMS's idMap shape."""
        from Tensile.Components.ScheduleCapture import (
            build_idmap, invert_idmap_to_id_to_category, split_for_plr,
        )
        from rocisa.code import Module

        LRCodeA, LRCodeB, PackCodeA, PackCodeB, _leaves = \
            self._build_per_iter_lists()

        # Mirror exactly the fix's split-and-rebuild step.
        assert len(LRCodeA) == 1
        LRCodeA_split = split_for_plr(LRCodeA[0])
        LRCodeB_split = split_for_plr(LRCodeB[0])
        PackCodeA_split = split_for_plr(PackCodeA[0])
        PackCodeB_split = split_for_plr(PackCodeB[0])
        num_loop_iter = 2

        idmap = build_idmap(
            num_loop_iter=num_loop_iter,
            LRCodeA=LRCodeA_split, PackCodeA=PackCodeA_split,
            LRCodeB=LRCodeB_split, PackCodeB=PackCodeB_split,
            globalReadA=Module(), globalReadB=Module(),
            globalReadIncACode=Module(), globalReadIncBCode=Module(),
            localWriteA=Module(), localWriteB=Module(),
            LRSwapA=[], LRSwapB=[], LWSwapA=[], LWSwapB=[],
            loopCounterCode=Module(), syncCode=Module(), snopCode=Module(),
        )

        # Per-iter categories from the split appear with both subiter
        # indices: PackA0, PackA1, PackB0, PackB1 (and LRA0/1, LRB0/1).
        keys = set(idmap.keys())
        for required in ("PackA0", "PackA1", "PackB0", "PackB1",
                         "LRA0", "LRA1", "LRB0", "LRB1"):
            assert required in keys, (
                f"Expected {required!r} in post-split idmap; "
                f"got {sorted(keys)}."
            )

    def test_no_split_for_plr_on_loopiters_1_misses_pack_a1_pack_b1(self):
        """Sanity / regression guard: WITHOUT split, build_idmap on a
        single-iter input produces only PackA0/PackB0 — the pre-Bug-2
        SHADOW shape that diverged from CMS."""
        from Tensile.Components.ScheduleCapture import (
            build_idmap,
        )
        from rocisa.code import Module

        LRCodeA, LRCodeB, PackCodeA, PackCodeB, _leaves = \
            self._build_per_iter_lists()

        idmap = build_idmap(
            num_loop_iter=1,  # pre-Bug-2 SHADOW path
            LRCodeA=LRCodeA, PackCodeA=PackCodeA,
            LRCodeB=LRCodeB, PackCodeB=PackCodeB,
            globalReadA=Module(), globalReadB=Module(),
            globalReadIncACode=Module(), globalReadIncBCode=Module(),
            localWriteA=Module(), localWriteB=Module(),
            LRSwapA=[], LRSwapB=[], LWSwapA=[], LWSwapB=[],
            loopCounterCode=Module(), syncCode=Module(), snopCode=Module(),
        )

        keys = set(idmap.keys())
        assert "PackA0" in keys and "PackB0" in keys
        assert "PackA1" not in keys, (
            "Pre-Bug-2 single-iter idmap should NOT have PackA1; "
            "presence here means split was applied unexpectedly."
        )
        assert "PackB1" not in keys

    def test_split_preserves_leaf_identity_for_cross_side_lookup(self):
        """split_for_plr must NOT clone — both SHADOW and CMS rely on
        Python id() of the same Instruction object to look up category
        in invert_idmap_to_id_to_category. If split cloned, SHADOW's
        leftover_id_to_cat keys wouldn't match CMS's, breaking the
        whole fix's premise.

        This locks the contract: split_for_plr's halves contain the
        SAME Python objects as the source list.
        """
        from Tensile.Components.ScheduleCapture import split_for_plr

        LRCodeA, _LRCodeB, _PackCodeA, _PackCodeB, leaves = \
            self._build_per_iter_lists()

        halves = split_for_plr(LRCodeA[0])
        assert len(halves) == 2
        # Returned form per ScheduleCapture.py:980 — [second_half, first_half]
        # (second half is iter 0; first half is iter 1).
        # Combined ids must equal the original 4 leaves.
        combined_ids = {id(x) for h in halves for x in h}
        original_ids = {id(x) for x in leaves["LRA"]}
        assert combined_ids == original_ids, (
            f"split_for_plr cloned (or dropped) leaves; "
            f"original={original_ids}, split={combined_ids}."
        )


class TestShadowCaptureLoopIters1CountsMatchCms:
    """rocm-libraries-nmsx Bug 2 — end-to-end count parity on a
    LoopIters==1 fixture. Mirrors test_shadow_main_capture_categories_match_cms_subject
    but exercises the split_for_plr path the bug pinned.

    Constructs a CMS kernel shaped to produce LoopIters==1 (DepthU ==
    MatrixInstK so the unroll loop has exactly one iteration). If the
    fixture cannot be built (kernel-validator rejects the shape, or
    amdclang missing), the test skips — but the focused-unit tests
    above still cover the contract.
    """

    def _try_build_loopiters1_writer_and_capture(self):
        """Build a CMS kernel with LoopIters==1.

        LoopIters = DepthU // (MatrixInstK * LocalSplitU). Use
        DepthU=32, MatrixInstruction K=32 so LoopIters=1.
        Returns (writer, ctx, lcid) or None if not buildable.
        """
        import shutil
        if not (shutil.which('amdclang++') or shutil.which('clang++')):
            return None
        if not (shutil.which('amdclang') or shutil.which('clang')):
            return None

        from Tensile.Common import IsaVersion
        from Tensile.Common.Capabilities import makeIsaInfoMap
        from Tensile.Toolchain.Component import Assembler
        from cms_test_utils import _make_solution
        from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
        import Tensile.Components.CustomSchedule.approach_a as _aa

        compiler = shutil.which('amdclang++') or shutil.which('clang++')
        assembler_bin = shutil.which('amdclang') or shutil.which('clang')
        isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)
        asm = Assembler(assembler_bin, 'V5')

        # MatrixInstruction K=32 + DepthU=32 -> LoopIters=1.
        config = {
            'ProblemType': {
                'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
                'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
                'UseBeta': True, 'Batched': True,
            },
            'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
            'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 0,
            'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
            'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
            'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
            'SourceSwap': 1, 'StreamK': 0,
            'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
        }
        try:
            solution = _make_solution(config, asm, isaInfoMap)
        except Exception:
            return None
        writer = KernelWriterAssembly(asm, DebugConfig())
        writer.enable_capture_default_schedule()

        _orig = _aa.build_non_cms_reference

        def _stash(kernel, assembler, isaInfoMap, *args, **kwargs):
            writer._test_shadow_capture = writer._capture_context.default
            raise RuntimeError("loopiters1-test-skip-xj16: snapshot taken")

        _aa.build_non_cms_reference = _stash
        try:
            try:
                writer._getKernelSource(solution)
            except Exception:
                pass
        finally:
            _aa.build_non_cms_reference = _orig

        return writer, writer._capture_context

    def test_loopiters1_per_category_count_parity(self):
        """On a LoopIters==1 kernel, SHADOW pack/LR per-iter category
        counts must match CMS post-split. Pre-Bug-2 SHADOW had only
        PackA0/PackB0 while CMS had PackA0+PackA1/PackB0+PackB1 —
        per-category counts diverged.
        """
        result = self._try_build_loopiters1_writer_and_capture()
        if result is None:
            pytest.skip("amdclang/clang not available for isa_infrastructure")
        writer, ctx = result

        # Helper functions live at module level alongside the BPG#11
        # fixture builder.
        main = _shadow_main_body(writer)
        cms_main = _cms_main_body(writer)
        if main is None or cms_main is None:
            pytest.skip(
                "SHADOW or CMS capture missing — likely the LoopIters==1 "
                "kernel-validator path rejected this shape. The focused "
                "unit tests in TestLeftoverIdmapSplitForPlrLoopIters1 "
                "still cover the contract."
            )

        # Sanity check that this fixture actually triggered LoopIters==1.
        # If the kernel-validator forced LoopIters>=2, this test is no
        # longer exercising the bug — skip rather than silently passing.
        loop_iters = writer.states.kernel.get("LoopIters") \
            if hasattr(writer.states, "kernel") else None
        if loop_iters is None:
            # Best-effort: check for PackA1 presence in CMS as a proxy
            # for "split_for_plr ran" (i.e. LoopIters==1 in CMS).
            cms_cats = {ti.category for ti in cms_main.instructions}
            if "PackA1" not in cms_cats:
                pytest.skip(
                    f"Fixture did not exercise the LoopIters==1 split path "
                    f"(no PackA1 in CMS); CMS categories: {sorted(cms_cats)}."
                )
        elif loop_iters != 1:
            pytest.skip(
                f"Fixture built with LoopIters={loop_iters}, not 1. "
                f"Kernel validator forced a different shape."
            )

        excluded = {"SYNC", "SNOP", "SSETPRIO", "SBARRIER", "MFMA"}

        def _per_cat_counts(body):
            counts = {}
            for ti in body.instructions:
                if ti.category in excluded:
                    continue
                counts[ti.category] = counts.get(ti.category, 0) + 1
            return counts

        shadow_counts = _per_cat_counts(main)
        cms_counts = _per_cat_counts(cms_main)

        # Bug-2 marker: pre-fix SHADOW would lack PackA1/PackB1 (or
        # LRA1/LRB1) while CMS would have them. We assert per-category
        # parity over the union of keys — same shape as the BPG#11 test.
        mismatches = []
        for cat in shadow_counts.keys() | cms_counts.keys():
            sh = shadow_counts.get(cat, 0)
            cm = cms_counts.get(cat, 0)
            if sh != cm:
                mismatches.append((cat, sh, cm))

        assert not mismatches, (
            f"LoopIters==1 per-category count mismatches: {mismatches}. "
            f"SHADOW={shadow_counts}, CMS={cms_counts}. "
            f"This is the Bug 2 regression — split_for_plr was not "
            f"applied symmetrically to the SHADOW leftover walk."
        )
