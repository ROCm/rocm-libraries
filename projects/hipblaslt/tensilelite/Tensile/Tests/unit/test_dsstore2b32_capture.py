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
"""Real-instruction fixture pinning DSStore2B32 capture-pipeline behavior.

`DSStore2B32` is `ds_store2_b32` — writes TWO independent 32-bit values to
LDS in a single instruction. Unlike the single-src `DSStore*` family
(DSStoreB8/B16/B32/B64/B128/...), the Python ctor signature is
`(dstAddr, src0, src1, ds, comment)` (rocisa/src/instruction/mem.cpp
DSStore2B32 binding). The C++ class derives from `DSStoreInstruction`
which already encodes the 3-slot src layout in its accessor:

    DSStoreInstruction::getSrcParams() -> {dstAddr, src0, src1}
        (rocisa/include/instruction/mem.hpp:883-886)

So the existing `_DSStoreRule.extract` (which delegates to
`_operands_with_slots`) produces three reads for a 2-src DSStore2B32 and
two reads for a single-src DSStoreB32 (the None src1 filters out via
`Register.is_register`) — no per-class extract logic needed. The only
required wiring is adding "DSStore2B32" to `_LW_CLASS_NAMES` so the
class-name dispatch in `_is_lw` claims it.

These tests pin:

  1. `_is_lw(DSStore2B32(...))` is True.
  2. `_DSStoreRule.applies(...)` is True.
  3. `_populate_wrapper` over a DSStore2B32 yields THREE reads
     (dstAddr, src0, src1) and ZERO writes.
  4. The contrast: DSStoreB32 (single src) yields TWO reads.
  5. End-to-end through `LoopBodyCaptureBuilder.finalize()`.

No `_FakeLW` synthetic stand-in is used (per bead 904 — real-instruction
fixtures are the project standard).
"""

import pytest


def _build_dsstore2_b32():
    """Real rocisa DSStore2B32. dstAddr=v50, src0=v8, src1=v12 (each
    1-wide for ds_store2_b32 since each per-lane element is 32 bits)."""
    from rocisa.container import vgpr
    from rocisa.instruction import DSStore2B32
    return DSStore2B32(
        dstAddr=vgpr(50, 1),
        src0=vgpr(8, 1),
        src1=vgpr(12, 1),
    )


def _build_dsstore_b32():
    """Real rocisa DSStoreB32 (single-src) for shape contrast."""
    from rocisa.container import vgpr
    from rocisa.instruction import DSStoreB32
    return DSStoreB32(dstAddr=vgpr(50, 1), src=vgpr(8, 1))


class TestDSStore2B32Dispatch:
    """`DSStore2B32` must be claimed by `_DSStoreRule` via `_is_lw`."""

    def test_dsstore2b32_recognized_as_lw(self):
        from Tensile.Components.ScheduleCapture import _is_lw
        assert _is_lw(_build_dsstore2_b32()), (
            "DSStore2B32 must be in _LW_CLASS_NAMES so the class-name "
            "dispatch routes it to _DSStoreRule. Without this, the "
            "instruction falls through to UNKNOWN and produces no "
            "reads/writes — a silent miscapture for a real LDS write."
        )

    def test_dsstore_rule_applies_to_dsstore2b32(self):
        from Tensile.Components.ScheduleCapture import _DSStoreRule
        rule = _DSStoreRule()
        assert rule.applies(_build_dsstore2_b32()), (
            "_DSStoreRule must claim DSStore2B32 — the C++ accessor "
            "DSStoreInstruction::getSrcParams already returns "
            "{dstAddr, src0, src1}, so the same rule handles both "
            "single-src and 2-src DSStore variants uniformly."
        )


class TestDSStore2B32ReadExtraction:
    """Pin the read-set: BOTH src0 AND src1 vgprs are captured as reads.
    """

    def test_dsstore2b32_yields_three_reads_no_writes(self):
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(_build_dsstore2_b32())
        _populate_wrapper(wrapper)
        # No register dst (data goes to LDS, not a vgpr).
        assert wrapper.writes == ()
        # Three reads: dstAddr (slot 0), src0 (slot 1), src1 (slot 2).
        # This is the load-bearing assertion: if _DSStoreRule failed to
        # surface src1 (the prior single-src assumption), this would
        # collapse to len == 2 and a downstream MFMA reading src1
        # before the store finishes would falsely appear unblocked.
        assert len(wrapper.reads) == 3, (
            f"DSStore2B32 must capture all three src registers "
            f"(dstAddr, src0, src1); got {len(wrapper.reads)} reads: "
            f"{wrapper.reads}"
        )

    def test_dsstore2b32_reads_include_both_src0_and_src1_regs(self):
        """Direct identity check: vgpr(8) and vgpr(12) (the two src
        operands) must both appear in the captured reads."""
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(_build_dsstore2_b32())
        _populate_wrapper(wrapper)

        read_idxs = {
            getattr(r, "regIdx", None) for r in wrapper.reads
        }
        # dstAddr=vgpr(50), src0=vgpr(8), src1=vgpr(12).
        assert 50 in read_idxs, (
            f"DSStore2B32 dstAddr (vgpr 50) missing from reads: {read_idxs}"
        )
        assert 8 in read_idxs, (
            f"DSStore2B32 src0 (vgpr 8) missing from reads: {read_idxs}"
        )
        assert 12 in read_idxs, (
            f"DSStore2B32 src1 (vgpr 12) missing from reads — this is "
            f"the regression the kx1 bead pins: a 2-src DSStore variant "
            f"must not silently drop its second src register. "
            f"Got: {read_idxs}"
        )

    def test_dsstore2b32_read_slots_preserve_positional_indices(self):
        """`_operands_with_slots` returns 0-based positional slots from
        `getSrcParams`. For DSStore2B32 the layout is
        `{dstAddr@0, src0@1, src1@2}` — slots must be (0, 1, 2)."""
        from Tensile.Components.ScheduleCapture import _operands_with_slots
        reads, read_slots, writes, write_slots = _operands_with_slots(
            _build_dsstore2_b32()
        )
        assert read_slots == (0, 1, 2), (
            f"DSStore2B32 read slots must be (0, 1, 2) — dstAddr@0, "
            f"src0@1, src1@2; got {read_slots}"
        )
        assert writes == ()
        assert write_slots == ()

    def test_single_src_dsstore_b32_contrast_yields_two_reads(self):
        """Sanity: single-src DSStoreB32 yields exactly TWO reads
        (dstAddr@0, src@1). The None src1 hole filters out via
        `Register.is_register`. This confirms the same `_DSStoreRule`
        handles both shapes correctly without a special branch."""
        from Tensile.Components.ScheduleCapture import (
            WrappedInstruction, _populate_wrapper,
        )
        wrapper = WrappedInstruction(_build_dsstore_b32())
        _populate_wrapper(wrapper)
        assert wrapper.writes == ()
        assert len(wrapper.reads) == 2, (
            f"Single-src DSStoreB32 must yield 2 reads (dstAddr, src); "
            f"got {len(wrapper.reads)}: {wrapper.reads}"
        )


class TestDSStore2B32EndToEnd:
    """End-to-end through LoopBodyCaptureBuilder."""

    def test_dsstore2b32_finalize_through_capture_pipeline(self):
        """Appending a real DSStore2B32 through LoopBodyCaptureBuilder
        and calling finalize() populates the wrapper without raising
        CaptureUnknownInstructionError / CaptureStoreError. Confirms the
        new dispatch wiring survives the full capture pipeline."""
        from Tensile.Components.ScheduleCapture import LoopBodyCaptureBuilder
        builder = LoopBodyCaptureBuilder()
        builder.append(_build_dsstore2_b32(), category="LWA", subiter=0)
        capture = builder.finalize()

        ti = capture.instructions[0]
        assert ti.wrapped is not None, (
            "LoopBodyCaptureBuilder.finalize() must populate the "
            "TaggedInstruction.wrapped field for DSStore2B32 — without "
            "the _LW_CLASS_NAMES entry, no rule applies and wrapped "
            "stays None (or the instruction routes to UNKNOWN)."
        )
        assert ti.wrapped.writes == ()
        assert len(ti.wrapped.reads) == 3
