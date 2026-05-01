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
        from dataflow_fixtures import _FakeLR, _vrange

        b = LoopBodyCaptureBuilder()
        b.append(inst=_FakeLR(dst=_vrange(8, 4), lds_offset=64),
                 category="LRA0", subiter=0, mfma_index=0)
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
            BODY_LABEL_ML, TaggedInstruction, SlotKey,
        )
        from dataclasses import dataclass

        @dataclass
        class _FakeSNop:
            pass

        snops = [
            TaggedInstruction(
                inst=_FakeSNop(), category="SNOP",
                slot=SlotKey(0, SLOT_KIND_MFMA, i, 0),
            )
            for i in range(5)
        ]
        cap = make_capture(BODY_LABEL_ML, snops)
        idmap = {"SNOP": [object(), object()]}  # 2 vs 5 — ignored
        assert_idmap_completeness(idmap, cap)
