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
"""LoopBodyCapture builders for dataflow-graph unit tests.

Bypasses the production category-resolution path (emit_instructions +
tag_by_origin_id + the MFMA fall-through in expand_cms_macro). Each
factory takes `category` as an explicit parameter and stuffs the result
directly into a fully-built TaggedInstruction wrapping a real rocisa
instruction instance.

Why bypass: the dataflow graph builder, barrier collectors, and
comparison classifier depend on TaggedInstruction.category being
correctly set. They don't depend on category-resolution correctness,
which is exercised in test_ScheduleCapture.py and the integration tests.
"""

from typing import Optional, Sequence

from rocisa.container import RegisterContainer, RegName, vgpr, sgpr

from Tensile.Components.ScheduleCapture import (
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    LoopBodyCapture,
    BODY_LABEL_TO_LOOP_INDEX,
    WrappedInstruction,
    _populate_wrapper,
)


# =============================================================================
# Helpers for register construction
# =============================================================================


def _vrange(start: int, count: int = 1) -> RegisterContainer:
    """Build a vgpr range starting at `start` covering `count` registers.

    Uses regType="v" — real rocisa's canonical single-character form.
    """
    return RegisterContainer("v", RegName(f"v{start}", []), start, count)


def _srange(start: int, count: int = 1) -> RegisterContainer:
    return RegisterContainer("s", RegName(f"s{start}", []), start, count)


# A high-numbered placeholder vgpr index used for LDS-address operands on
# real `rocisa.DSLoad*` / `DSStore*` instances. Tests don't exercise
# LDS-address dataflow (the LR/LW edge model uses LDS slot offsets, not
# vgpr identity for the address), so wiring every fixture to the same
# placeholder index is safe — the only constraint is that it must not
# collide with vgpr ranges tests use as MFMA / pack / scratch operands.
_LDS_ADDR_PLACEHOLDER_VGPR = 255

# Placeholder vgpr index for BufferLoad's vaddr operand. Same rationale as
# `_LDS_ADDR_PLACEHOLDER_VGPR`: tests use the GR's `dst` for dataflow,
# not its vaddr; a single distinct placeholder index keeps the vaddr from
# colliding with test data ranges.
_BUFFER_LOAD_VADDR_PLACEHOLDER_VGPR = 254


# =============================================================================
# TaggedInstruction builders
# =============================================================================
# Slot is constructed with subiter=0 (the macro is iteration-flattened in
# real captures too); body_label is encoded in the test via make_capture, not
# on the SlotKey. Slot's `sequence` field defaults to 0; tests can pass a
# different value when they need to disambiguate two instructions at the
# same mfma_index.


def make_lr(dst_vgpr_start: int, dst_vgpr_count: int, lds_offset: int, slot: int,
            *, category: str = "LRA0", sequence: int = 0) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping a real `rocisa.DSLoadB128`.

    The B128 width is chosen as a stable canonical LR class — `_DSLoadRule`
    routes every `DSLoadB*` variant identically, and tests don't depend on
    the concrete word size. The LDS-address vgpr (rocisa's `src` argument)
    is sourced from a per-call placeholder vgpr; tests don't exercise
    LDS-address dataflow.
    """
    from rocisa.instruction import DSLoadB128
    from rocisa.container import DSModifiers
    inst = DSLoadB128(
        dst=vgpr(dst_vgpr_start, dst_vgpr_count),
        src=vgpr(_LDS_ADDR_PLACEHOLDER_VGPR, 1),
        ds=DSModifiers(offset=lds_offset),
    )
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_lw(src_vgpr_start: int, src_vgpr_count: int, lds_offset: int, slot: int,
            *, category: str = "LWA", sequence: int = 0) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping a real `rocisa.DSStoreB128`.

    Real DSStore exposes the LDS-address vgpr in `getSrcParams()`. The
    address is sourced from a per-call placeholder vgpr; tests don't
    exercise LDS-address dataflow.
    """
    from rocisa.instruction import DSStoreB128
    from rocisa.container import DSModifiers
    inst = DSStoreB128(
        dstAddr=vgpr(_LDS_ADDR_PLACEHOLDER_VGPR, 1),
        src=vgpr(src_vgpr_start, src_vgpr_count),
        ds=DSModifiers(offset=lds_offset),
    )
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_gr(dst_vgpr_start: int, dst_vgpr_count: int,
            srd_sgpr_start: int, immediate_offset: int, slot: int,
            *, category: str = "GRA", sequence: int = 0) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping a real `rocisa.BufferLoadB128`.

    Real BufferLoad exposes vaddr (vgpr) AND saddr (sgpr SRD) in
    `getSrcParams()`. The vaddr is sourced from a per-call placeholder
    vgpr; tests don't exercise vaddr dataflow.
    """
    from rocisa.instruction import BufferLoadB128
    from rocisa.container import MUBUFModifiers
    mubuf = MUBUFModifiers(offen=True, offset12=immediate_offset)
    inst = BufferLoadB128(
        dst=vgpr(dst_vgpr_start, dst_vgpr_count),
        vaddr=vgpr(_BUFFER_LOAD_VADDR_PLACEHOLDER_VGPR, 1),
        saddr=sgpr(srd_sgpr_start, 4),
        soffset=0,
        mubuf=mubuf,
    )
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_dtl_buffer_load(vaddr_vgpr_start: int, srd_sgpr_start: int,
                         slot: int, *, category: str = "GRA",
                         sequence: int = 0,
                         immediate_offset: int = 0) -> TaggedInstruction:
    """Build a real DTL-mode BufferLoadB128 wrapped in a TaggedInstruction.

    Unlike `make_gr` (which builds a non-DTL `BufferLoadB128` with a vgpr
    `dst`), this factory constructs the DTL flavor: `dst=None` and
    `MUBUFModifiers(lds=True)`. The rocisa C++ constructor sets
    `is_dtl=True` from `mubuf->lds`, which `_BufferLoadRule` reads to
    publish the implicit m0 read alongside the saddr — enabling the
    dataflow graph to form an edge from any prior m0 setter (SMovB32 /
    SAddU32) to this load.

    Use this in any test that needs DTL-aware m0 tracking. Use `make_gr`
    for plain (non-DTL) BufferLoad tests where the load writes a vgpr dst.
    """
    # Lazy imports — rocisa is a heavy C++-bound module; the fake-only
    # tests in this file shouldn't pay its import cost.
    from rocisa.container import MUBUFModifiers
    from rocisa.instruction import BufferLoadB128
    mubuf = MUBUFModifiers(offen=True, offset12=immediate_offset, lds=True)
    inst = BufferLoadB128(
        dst=None,
        vaddr=vgpr(vaddr_vgpr_start, 1),
        saddr=sgpr(srd_sgpr_start, 4),
        soffset=0,
        mubuf=mubuf,
    )
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_mfma(c_dst_start: int, a_src_start: int, b_src_start: int, slot: int,
              *, c_dst_count: int = 4, a_src_count: int = 2, b_src_count: int = 2,
              category: str = "MFMA", sequence: int = 0,
              variant: Optional[list] = None) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping a real `rocisa.MFMAInstruction`.

    `variant` mirrors `rocisa.MFMAInstruction.variant` ([M, N, K, blk]).
    Default `[32, 32, 0, 1]` renders as `v_wmma_..._32x32x0_...` — the
    standard MFMA family for finish-cycle classification. Pass
    `[4, 4, 4, 16]` to model a 4x4 PackMFMA so `_mfma_finish_cycles_for`
    identifies it as the 1-quad-cycle Pack flavor via the `_4x4x`
    substring in the rendered assembly.

    `acc2 == acc` (the in-place RMW shape) is the dominant production
    case and matches the prior `_FakeMFMA` (a, b, acc) source synthesis.
    """
    from rocisa.instruction import MFMAInstruction
    from rocisa.enum import InstType
    full_variant = list(variant) if variant is not None else [32, 32, 0, 1]
    # Pad short test variants to MFMAInstruction's expected [M, N, K, blk].
    while len(full_variant) < 4:
        full_variant.append(0 if len(full_variant) < 3 else 1)
    acc = vgpr(c_dst_start, c_dst_count)
    inst = MFMAInstruction(
        instType=InstType.INST_F32,
        accType=InstType.INST_F32,
        variant=full_variant,
        mfma1k=False,
        acc=acc,
        a=vgpr(a_src_start, a_src_count),
        b=vgpr(b_src_start, b_src_count),
        acc2=acc,
    )
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_swait(slot: int, *, dscnt: int = -1, vlcnt: int = -1, vscnt: int = -1,
               sequence: int = 0) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping a real `rocisa.SWaitCnt`."""
    from rocisa.instruction import SWaitCnt
    inst = SWaitCnt(vlcnt=vlcnt, vscnt=vscnt, dscnt=dscnt)
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category="SYNC",
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_sbarrier(slot: int, *, sequence: int = 0) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping a real `rocisa.SBarrier`."""
    from rocisa.instruction import SBarrier
    inst = SBarrier()
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category="BARRIER",
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_snop(slot: int, *, wait_state: int = 0,
              sequence: int = 0) -> TaggedInstruction:
    """Build a TaggedInstruction wrapping `rocisa.SNop(waitState=...)`.

    `wait_state` controls the additional quad-cycle issue cost added by
    `_min_issue_quad_cycles_for` (returns `1 + wait_state` for SNop).
    See `cumulative_issue_cycles` walk in ScheduleCapture.py — the SNop's
    per-instruction cost is read inline (SNop instances are not graph
    nodes).
    """
    from rocisa.instruction import SNop
    inst = SNop(waitState=wait_state)
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category="SNOP",
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def make_lcc_pair(slot: int, *,
                  loop_counter_name: str = "LoopCounterL",
                  end_counter: int = 0,
                  sequence_start: int = 0
                  ) -> Sequence[TaggedInstruction]:
    """Build the canonical LCC instruction pair (SSubU32 + SCmpEQI32).

    Mirrors `closeLoop(..., finalLoop=False)` sub-branch A from
    `KernelWriterAssembly.py:6883-6892`:

        SSubU32(dst=loopCounter, src0=loopCounter, src1=hex(1))
        SCmpEQI32(src0=loopCounter, src1=hex(endCounter))

    Per `LCC_AUDIT.md` every captured CMS schedule emits exactly this
    2-instruction pair into the LCC bucket; both are 1-quad-cycle SALU
    ops.

    Uses real rocisa classes (no `_Fake*` shim) — `_class_tag_from_category`
    returns "LCC" so the resulting nodes carry the LCC identity tag and
    `_SCCRule` publishes their reads/writes (sgpr loop counter + SCC).
    Both instructions are placed at the same `mfma_index` slot with
    consecutive `sequence` values (matches the dominant CMS layout
    `'LCC' : [[N, N]]` from `CustomSchedule.py` — see audit Step 3).
    """
    from rocisa.instruction import SSubU32, SCmpEQI32
    counter = sgpr(loop_counter_name, 1)
    sub = SSubU32(dst=counter, src0=counter, src1=hex(1))
    cmp_ = SCmpEQI32(src0=counter, src1=hex(end_counter))
    return (
        TaggedInstruction(
            wrapped=WrappedInstruction(sub),
            category="LCC",
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=slot, sequence=sequence_start),
        ),
        TaggedInstruction(
            wrapped=WrappedInstruction(cmp_),
            category="LCC",
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=slot, sequence=sequence_start + 1),
        ),
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
    # Populate wrapped reads/writes for each TaggedInstruction. With wrapped
    # mandatory on TaggedInstruction the wrapper exists at construction, but
    # fixture factories do not call _populate_wrapper. Without this step,
    # build_dataflow_graph would see empty (reads, writes) and skip edge
    # formation — breaking the dataflow tests that rely on these fixtures.
    insts = list(instructions)
    for ti in insts:
        _populate_wrapper(ti.wrapped, category=ti.category)
    return LoopBodyCapture(instructions=insts)
