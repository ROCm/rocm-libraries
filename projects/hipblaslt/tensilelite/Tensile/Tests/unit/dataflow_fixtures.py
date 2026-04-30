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
"""Synthetic LoopBodyCapture builders for dataflow-graph unit tests.

Bypasses the production category-resolution path (emit_instructions +
tag_by_origin_id + the MFMA fall-through in expand_cms_macro). Constructors
take `category` as an explicit parameter and stuff the result directly into
a fully-built TaggedInstruction.

Why bypass: the dataflow graph builder, barrier collectors, and comparison
classifier depend on TaggedInstruction.category being correctly set. They
don't depend on category-resolution correctness, which is exercised in
test_ScheduleCapture.py and the integration tests.

Stand-in instruction shapes are ducktyped — only the attributes the graph
builder actually reads need to exist.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from rocisa.container import RegisterContainer, RegName, vgpr, sgpr

from Tensile.Components.ScheduleCapture import (
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    LoopBodyCapture,
    BODY_LABEL_TO_LOOP_INDEX,
)


# =============================================================================
# Stand-in instruction classes
# =============================================================================
# These mimic the rocisa instruction class hierarchy enough for the graph
# builder to dispatch on `type(inst).__name__`. Real rocisa classes are
# C++-bound and a pain to construct in tests; these stubs are pure Python.


@dataclass
class _FakeInstBase:
    """Common base; nothing here — subclass marker only for isinstance()."""


@dataclass
class _FakeLR(_FakeInstBase):
    """Stand-in for a DSLoad (LR) instruction."""
    dst: RegisterContainer       # vgpr range written
    lds_offset: int              # immediate LDS offset

    def __str__(self):
        return f"ds_read {self.dst}, lds[{self.lds_offset}]"


@dataclass
class _FakeLW(_FakeInstBase):
    """Stand-in for a DSStore (LW) instruction."""
    src: RegisterContainer
    lds_offset: int

    def __str__(self):
        return f"ds_write lds[{self.lds_offset}], {self.src}"


@dataclass
class _FakeGR(_FakeInstBase):
    """Stand-in for a BufferLoad (GR) instruction."""
    dst: RegisterContainer
    srd: RegisterContainer       # sgpr SRD
    immediate_offset: int

    def __str__(self):
        return f"buffer_load {self.dst}, {self.srd}, off={self.immediate_offset}"


@dataclass
class _FakeMFMA(_FakeInstBase):
    """Stand-in for an MFMA instruction."""
    c_dst: RegisterContainer
    a_src: RegisterContainer
    b_src: RegisterContainer

    def __str__(self):
        return f"v_mfma {self.c_dst}, {self.a_src}, {self.b_src}, {self.c_dst}"


@dataclass
class _FakeSWait(_FakeInstBase):
    """Stand-in for SWaitCnt. Field names match rocisa.SWaitCnt."""
    dscnt: int = -1
    vlcnt: int = -1
    vscnt: int = -1

    def __str__(self):
        return f"s_waitcnt(dscnt={self.dscnt}, vlcnt={self.vlcnt}, vscnt={self.vscnt})"


@dataclass
class _FakeSBarrier(_FakeInstBase):
    """Stand-in for SBarrier — no fields, only its presence matters."""

    def __str__(self):
        return "s_barrier"


# =============================================================================
# Helpers for register construction
# =============================================================================


def _vrange(start: int, count: int = 1) -> RegisterContainer:
    """Build a vgpr range starting at `start` covering `count` registers."""
    return RegisterContainer("vgpr", RegName(f"v{start}", []), start, count)


def _srange(start: int, count: int = 1) -> RegisterContainer:
    return RegisterContainer("sgpr", RegName(f"s{start}", []), start, count)


# =============================================================================
# TaggedInstruction builders
# =============================================================================
# Slot is constructed with iteration=0 (the macro is iteration-flattened in
# real captures too); body_label is encoded in the test via make_capture, not
# on the SlotKey. Slot's `sequence` field defaults to 0; tests can pass a
# different value when they need to disambiguate two instructions at the
# same mfma_index.


def make_lr(dst_vgpr_start: int, dst_vgpr_count: int, lds_offset: int, slot: int,
            *, category: str = "LRA0", sequence: int = 0) -> TaggedInstruction:
    inst = _FakeLR(dst=_vrange(dst_vgpr_start, dst_vgpr_count), lds_offset=lds_offset)
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_lw(src_vgpr_start: int, src_vgpr_count: int, lds_offset: int, slot: int,
            *, category: str = "LWA", sequence: int = 0) -> TaggedInstruction:
    inst = _FakeLW(src=_vrange(src_vgpr_start, src_vgpr_count), lds_offset=lds_offset)
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_gr(dst_vgpr_start: int, dst_vgpr_count: int,
            srd_sgpr_start: int, immediate_offset: int, slot: int,
            *, category: str = "GRA", sequence: int = 0) -> TaggedInstruction:
    inst = _FakeGR(
        dst=_vrange(dst_vgpr_start, dst_vgpr_count),
        srd=_srange(srd_sgpr_start, 4),
        immediate_offset=immediate_offset,
    )
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_mfma(c_dst_start: int, a_src_start: int, b_src_start: int, slot: int,
              *, c_dst_count: int = 4, a_src_count: int = 2, b_src_count: int = 2,
              category: str = "MFMA", sequence: int = 0) -> TaggedInstruction:
    inst = _FakeMFMA(
        c_dst=_vrange(c_dst_start, c_dst_count),
        a_src=_vrange(a_src_start, a_src_count),
        b_src=_vrange(b_src_start, b_src_count),
    )
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_swait(slot: int, *, dscnt: int = -1, vlcnt: int = -1, vscnt: int = -1,
               sequence: int = 0) -> TaggedInstruction:
    inst = _FakeSWait(dscnt=dscnt, vlcnt=vlcnt, vscnt=vscnt)
    return TaggedInstruction(
        inst=inst,
        category="SYNC",
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_sbarrier(slot: int, *, sequence: int = 0) -> TaggedInstruction:
    inst = _FakeSBarrier()
    return TaggedInstruction(
        inst=inst,
        category="BARRIER",
        slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_capture(body_label: str, instructions: Sequence[TaggedInstruction]
                 ) -> LoopBodyCapture:
    """Construct a LoopBodyCapture with the given pre-built TaggedInstructions.

    Bypasses LoopBodyCaptureBuilder.finalize() so capture-pipeline checks
    (rocisa-wiring, SMEM, flat, store, mfma_code shape) don't fire on
    fixture data. Those checks are tested separately in
    test_capture_pipeline_checks.py with the real builder.

    Validates body_label so a typo doesn't silently route to a wrong
    loop_index downstream.
    """
    if body_label not in BODY_LABEL_TO_LOOP_INDEX:
        raise ValueError(
            f"Unknown body_label {body_label!r}. Expected one of "
            f"{sorted(BODY_LABEL_TO_LOOP_INDEX)}."
        )
    return LoopBodyCapture(instructions=list(instructions))
