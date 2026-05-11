################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""
Single source of truth for the Tensile-lite-side rocisa instruction
categorization used by the CMS validator and the dataflow-graph builder.

Per rocm-libraries-009 (re-scoped 2026-05-08): pushing scheduler-role tags
(LR / LW / GR / MFMA / PACK / ...) into rocisa C++ classes is a layer
violation — rocisa only knows the underlying GPU shape ("this is a
DSLoadB128 with these operands"); whether a given DSLoadB128 plays the
"local read" role is a Tensile-lite scheduler abstraction, NOT an
intrinsic rocisa property. Centralization therefore lives entirely
inside Tensile/Components/, with rocisa untouched.

The map keys on `type(inst).__name__` rather than the type object itself
because the existing producers (CMS scheduler, default-side capture
builder) and the existing test fixtures (which often stand in synthetic
classes whose `__name__` matches a real rocisa class but whose identity
does not) both depend on string-name dispatch. Switching to type-keyed
dispatch would break every fixture that uses a stand-in class — the
class-name-string indirection is exactly what kept the captured-graph
machinery rocisa-import-free in the first place.

Single registry: `_CLASS_NAME_TO_CATEGORY` is the only place a class name
maps to a category. The 13 historical `_*_CLASS_NAMES` sets and the 10
`_is_*` discriminator predicates have been collapsed into entries here.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class InstructionCategory(Enum):
    """Scheduler-role categorization of a rocisa instruction class.

    Each value names ONE Tensile-lite-side bucket the dataflow-graph
    builder, the CMS validator's edge-coverage analysis, and the capture
    pipeline's finalize() guards discriminate on. The bucket vocabulary
    is the union of what the historical 13 `_*_CLASS_NAMES` sets and
    the 10 `_is_*` discriminator predicates already split on; no
    speculative buckets are introduced.

    Members:
      LR             — DS-load (local read).
      LW             — DS-store (local write).
      GR             — buffer/global load (vector global read).
      MFMA           — matrix multiply-accumulate.
      SWAIT          — s_waitcnt.
      SBARRIER       — s_barrier.
      SNOP           — s_nop.
      SSETPRIO       — s_setprio (wave priority; no register dataflow).
      CVT_PACK       — TF32-emulation v_cvt_pk_bf16_f32 family.
      MIDDLE_PACK    — TF32-emulation middle-16 v_cvt_f32_bf16 / v_sub_f32 /
                       PVCvtBF16toFP32 / VDot2CF32BF16 family.
      SMEM           — scalar-memory load/store (finalize() guard only —
                       these decrement dscnt and would desync the per-
                       counter FIFO model used by build_dataflow_graph).
      FLAT           — flat load/store (finalize() guard only — these
                       decrement vmcnt+dscnt simultaneously which the
                       per-counter queue model doesn't handle).
      VECTOR_STORE   — buffer/global vector store (finalize() guard only —
                       vscnt is not modeled).
    """
    LR = "LR"
    LW = "LW"
    GR = "GR"
    MFMA = "MFMA"
    SWAIT = "SWAIT"
    SBARRIER = "SBARRIER"
    SNOP = "SNOP"
    SSETPRIO = "SSETPRIO"
    CVT_PACK = "CVT_PACK"
    MIDDLE_PACK = "MIDDLE_PACK"
    SMEM = "SMEM"
    FLAT = "FLAT"
    VECTOR_STORE = "VECTOR_STORE"


# =============================================================================
# Single class-name -> category registry
# =============================================================================
# This is the ONLY place a rocisa class-name string is bound to a category.
# Adding a new rocisa class to a category means adding ONE entry below.
#
# Class-name keying (rather than type-keyed) preserves the rocisa-import-free
# property of the surrounding modules — synthetic test stand-ins whose
# `type(...).__name__` matches a real rocisa class continue to dispatch
# correctly without registering the stand-in's type object.
_CLASS_NAME_TO_CATEGORY: dict = {
    # LR — Real rocisa LR classes plus the generic umbrella for isinstance
    # fallback if needed.
    "DSLoadB32":            InstructionCategory.LR,
    "DSLoadB64":            InstructionCategory.LR,
    "DSLoadB128":           InstructionCategory.LR,
    "DSLoadB256":           InstructionCategory.LR,
    "DSLoadInstruction":    InstructionCategory.LR,

    # LW — Wider DSStore variants from rocisa/include/instruction/mem.hpp;
    # _DSStoreRule already handles them via getParams() slot 0/1. Listed
    # defensively so type-name dispatch does not misclassify them.
    # D16HI / B8HID16 partial-half-word LDS stores ARE real DSStore
    # subclasses — same Python ctor signature as DSStoreB16 — and the
    # "D16HI"/"B8HID16" semantic doesn't change register-side dataflow.
    "DSStoreB8":            InstructionCategory.LW,
    "DSStoreB16":           InstructionCategory.LW,
    "DSStoreB32":           InstructionCategory.LW,
    "DSStoreB64":           InstructionCategory.LW,
    "DSStoreB128":          InstructionCategory.LW,
    "DSStoreU16":           InstructionCategory.LW,
    "DSStoreB96":           InstructionCategory.LW,
    "DSStoreB192":          InstructionCategory.LW,
    "DSStoreB256":          InstructionCategory.LW,
    "DSStoreD16HIB16":      InstructionCategory.LW,
    "DSStoreB8HID16":       InstructionCategory.LW,
    # DSStore2B32 — `ds_store2_b32` writes TWO independent 32-bit values
    # to LDS in a single instruction. Real rocisa subclass of
    # DSStoreInstruction; same generic rule extracts both src registers
    # (kx1: `_DSStoreRule.extract` delegates to `_operands_with_slots`).
    "DSStore2B32":          InstructionCategory.LW,
    "DSStoreInstruction":   InstructionCategory.LW,

    # GR — rocisa BufferLoad / GlobalLoad classes plus their umbrellas.
    "BufferLoadB32":         InstructionCategory.GR,
    "BufferLoadB64":         InstructionCategory.GR,
    "BufferLoadB128":        InstructionCategory.GR,
    "GlobalLoadB32":         InstructionCategory.GR,
    "GlobalLoadB64":         InstructionCategory.GR,
    "GlobalLoadB128":        InstructionCategory.GR,
    "BufferLoadInstruction": InstructionCategory.GR,
    "GlobalLoadInstruction": InstructionCategory.GR,
    "GlobalReadInstruction": InstructionCategory.GR,

    # MFMA
    "MFMAInstruction":       InstructionCategory.MFMA,

    # Scalar-control (SWAIT / SBARRIER / SNOP / SSETPRIO).
    # SSetPrior — wave-priority scalar op with NO register dataflow:
    # constructor takes only `int prior` (rocisa/include/instruction/
    # common.hpp::SSetPrior); getParams() returns `{prior}`. Treated like
    # SNop/SBarrier/SWaitCnt — claimed by `_NoDataflowRule`, excluded from
    # cross-graph data-flow node identity, default 1 quad-cycle issue cost
    # (no wait_state add — unlike SNop). Witnessed under gfx950 HSS
    # MT256x256x64 #2 and BBS Range MT256x256x64 under
    # CMS=1+PGR=2+PLR=1+DTL=T/T.
    "SWaitCnt":              InstructionCategory.SWAIT,
    "SBarrier":              InstructionCategory.SBARRIER,
    "SNop":                  InstructionCategory.SNOP,
    "SSetPrior":             InstructionCategory.SSETPRIO,

    # CVT-pack (TF32 emulation: v_cvt_pk_bf16_f32 family). Production
    # rocisa class is `VCvtPkF32toBF16`; downstream MFMAs reading the
    # CVTPack's output need a 2-quad-cycle settle window (CDNA 4 ISA
    # section 7.6, `_QUAD_CYCLES_CVT_BEFORE_MFMA`).
    "VCvtPkF32toBF16":       InstructionCategory.CVT_PACK,

    # MiddlePack (TF32 emulation: middle-16 of each 24-pack group computing
    # bf16 error terms). Pair-leader / pair-consumer interleaving invariant
    # checked by `validate_middle_pack_pair_interleaving`; no other middle
    # pack from any category may appear between the two halves of a pair
    # in the global stream.
    "PVCvtBF16toFP32":       InstructionCategory.MIDDLE_PACK,
    "VCvtBF16toFP32":        InstructionCategory.MIDDLE_PACK,
    "VSubF32":               InstructionCategory.MIDDLE_PACK,
    "VDot2CF32BF16":         InstructionCategory.MIDDLE_PACK,

    # SMEM — scalar memory; finalize() guard rejects these in CMS bodies
    # (they decrement dscnt and would desync the per-counter FIFO model).
    "SLoadB32":              InstructionCategory.SMEM,
    "SLoadB64":              InstructionCategory.SMEM,
    "SLoadB128":             InstructionCategory.SMEM,
    "SLoadB256":             InstructionCategory.SMEM,
    "SLoadB512":             InstructionCategory.SMEM,
    "SStoreB32":             InstructionCategory.SMEM,
    "SStoreB64":             InstructionCategory.SMEM,
    "SStoreB128":            InstructionCategory.SMEM,
    "SMemLoadInstruction":   InstructionCategory.SMEM,
    "SMemStoreInstruction":  InstructionCategory.SMEM,

    # FLAT — finalize() guard rejects these in CMS bodies (they decrement
    # both vmcnt and dscnt simultaneously which the per-counter queue model
    # doesn't handle).
    "FlatLoadB8":            InstructionCategory.FLAT,
    "FlatLoadB16":           InstructionCategory.FLAT,
    "FlatLoadB32":           InstructionCategory.FLAT,
    "FlatLoadB64":           InstructionCategory.FLAT,
    "FlatLoadB128":          InstructionCategory.FLAT,
    "FlatStoreB8":           InstructionCategory.FLAT,
    "FlatStoreB16":          InstructionCategory.FLAT,
    "FlatStoreB32":          InstructionCategory.FLAT,
    "FlatStoreB64":          InstructionCategory.FLAT,
    "FlatStoreB128":         InstructionCategory.FLAT,
    "FLATReadInstruction":   InstructionCategory.FLAT,
    "FLATStoreInstruction":  InstructionCategory.FLAT,

    # VECTOR_STORE — finalize() guard rejects these in CMS bodies (vscnt
    # not tracked; no current CMS body emits stores).
    "BufferStoreB32":          InstructionCategory.VECTOR_STORE,
    "BufferStoreB64":          InstructionCategory.VECTOR_STORE,
    "BufferStoreB128":         InstructionCategory.VECTOR_STORE,
    "GlobalStoreB32":          InstructionCategory.VECTOR_STORE,
    "GlobalStoreB64":          InstructionCategory.VECTOR_STORE,
    "GlobalStoreB128":         InstructionCategory.VECTOR_STORE,
    "BufferStoreInstruction":  InstructionCategory.VECTOR_STORE,
    "GlobalStoreInstruction":  InstructionCategory.VECTOR_STORE,
}


def category(inst) -> Optional[InstructionCategory]:
    """Return the InstructionCategory for `inst`, or None if unknown.

    Class-name dispatch (`type(inst).__name__`) — see module docstring for
    why string-keyed lookup is preferred over `type(inst)`-keyed lookup.

    Returns None when the class is not registered. Callers that previously
    branched on `_is_lr(inst)` (a bool returning False for non-LR inputs
    AND for unrecognized inputs alike) now branch on
    `category(inst) == InstructionCategory.LR`, which preserves the same
    truth-value behavior.
    """
    return _CLASS_NAME_TO_CATEGORY.get(type(inst).__name__)


def category_of_class_name(cls_name: str) -> Optional[InstructionCategory]:
    """Lookup-by-name escape hatch.

    Used by the snapshot test (which iterates the registry to verify the
    legacy predicate chain produces the same answer for every registered
    class name) and by callers that already have the class name in hand
    (e.g. LoopBodyCaptureBuilder.finalize() which extracts cls_name once
    and dispatches into multiple guards).
    """
    return _CLASS_NAME_TO_CATEGORY.get(cls_name)


def registered_class_names() -> tuple:
    """Snapshot of all class names currently registered.

    Stable input set for the snapshot test in
    `Tests/unit/test_instruction_category_snapshot.py`. Returned as an
    immutable tuple so callers cannot mutate the registry through the
    snapshot.
    """
    return tuple(_CLASS_NAME_TO_CATEGORY.keys())


# =============================================================================
# RDNA3.5 S_DELAY_ALU named gap-class taxonomy
# =============================================================================
# Mirrors the `INSTID_*` enumeration from RDNA3.5 ISA §16.5 (pp. 251-252)
# verbatim — the instruction itself IS the gap and the encoding NAMES the
# class. This is the strong ISA-side signal called out in the s5g1 audit
# memo (`ISA_GAP_GENERALIZATION_AUDIT.md`, R-10..R-13): the named taxonomy
# is the right validator abstraction, NOT a synthesized "quad-cycle" count.
#
# Each member is exactly one `INSTID_*` value the ISA defines for the
# `INSTID0` / `INSTID1` 4-bit fields of `S_DELAY_ALU`'s SIMM16 encoding.
# Hex values are the 4-bit encoded values per §16.5. DO NOT add members
# the ISA does not list — RDNA3.5 §16.5 is the closed source.
#
# Used by `validate_s_delay_alu_coverage` (CMSValidator) — dormant today
# (the kernel emitter does not produce S_DELAY_ALU), structurally
# installed so the moment any emitter starts producing one, the
# encoding/gap consistency check is already in place.
class RdnaSDelayAluClass(Enum):
    """RDNA3.5 §16.5 S_DELAY_ALU `INSTID_*` named gap-class taxonomy.

    Member name ↔ ISA `INSTID_*` symbol (1:1, verbatim). Member value is
    the 4-bit encoded value per §16.5 ("Legal values for the InstID0 and
    InstID1 fields are: ...").
    """
    NO_DEP            = 0x0   # INSTID_NO_DEP — no dependency on any prior instruction.
    VALU_DEP_1        = 0x1   # INSTID_VALU_DEP_1 — VALU dep, 1 instruction back.
    VALU_DEP_2        = 0x2   # INSTID_VALU_DEP_2 — VALU dep, 2 instructions back.
    VALU_DEP_3        = 0x3   # INSTID_VALU_DEP_3 — VALU dep, 3 instructions back.
    VALU_DEP_4        = 0x4   # INSTID_VALU_DEP_4 — VALU dep, 4 instructions back.
    TRANS32_DEP_1     = 0x5   # INSTID_TRANS32_DEP_1 — TRANS32 dep, 1 back.
    TRANS32_DEP_2     = 0x6   # INSTID_TRANS32_DEP_2 — TRANS32 dep, 2 back.
    TRANS32_DEP_3     = 0x7   # INSTID_TRANS32_DEP_3 — TRANS32 dep, 3 back.
    FMA_ACCUM_CYCLE_1 = 0x8   # INSTID_FMA_ACCUM_CYCLE_1 — single-cycle FMA accum penalty (reserved).
    SALU_CYCLE_1      = 0x9   # INSTID_SALU_CYCLE_1 — 1 cycle penalty for prior SALU.
    SALU_CYCLE_2      = 0xa   # INSTID_SALU_CYCLE_2 — 2 cycle penalty for prior SALU.
    SALU_CYCLE_3      = 0xb   # INSTID_SALU_CYCLE_3 — 3 cycle penalty for prior SALU.

    @property
    def required_back_distance(self) -> int:
        """The minimum producer-back distance (in instructions of the
        named family) the encoding asserts. `VALU_DEP_3` claims "the
        VALU producer is 3 instructions back"; the actual producer/consumer
        gap must be >= this to be correctly covered.

        `NO_DEP` returns 0 — degenerate carrier, nothing to verify.
        `FMA_ACCUM_CYCLE_1` returns 1 — reserved single-cycle penalty.
        """
        return _SDELAY_ALU_REQUIRED_BACK_DISTANCE[self]

    @property
    def family(self) -> str:
        """Producer-instruction family the class names: 'VALU', 'TRANS32',
        'SALU', 'FMA_ACCUM', or 'NONE' (NO_DEP)."""
        return _SDELAY_ALU_FAMILY[self]


_SDELAY_ALU_REQUIRED_BACK_DISTANCE: dict = {
    RdnaSDelayAluClass.NO_DEP:            0,
    RdnaSDelayAluClass.VALU_DEP_1:        1,
    RdnaSDelayAluClass.VALU_DEP_2:        2,
    RdnaSDelayAluClass.VALU_DEP_3:        3,
    RdnaSDelayAluClass.VALU_DEP_4:        4,
    RdnaSDelayAluClass.TRANS32_DEP_1:     1,
    RdnaSDelayAluClass.TRANS32_DEP_2:     2,
    RdnaSDelayAluClass.TRANS32_DEP_3:     3,
    RdnaSDelayAluClass.FMA_ACCUM_CYCLE_1: 1,
    RdnaSDelayAluClass.SALU_CYCLE_1:      1,
    RdnaSDelayAluClass.SALU_CYCLE_2:      2,
    RdnaSDelayAluClass.SALU_CYCLE_3:      3,
}


_SDELAY_ALU_FAMILY: dict = {
    RdnaSDelayAluClass.NO_DEP:            "NONE",
    RdnaSDelayAluClass.VALU_DEP_1:        "VALU",
    RdnaSDelayAluClass.VALU_DEP_2:        "VALU",
    RdnaSDelayAluClass.VALU_DEP_3:        "VALU",
    RdnaSDelayAluClass.VALU_DEP_4:        "VALU",
    RdnaSDelayAluClass.TRANS32_DEP_1:     "TRANS32",
    RdnaSDelayAluClass.TRANS32_DEP_2:     "TRANS32",
    RdnaSDelayAluClass.TRANS32_DEP_3:     "TRANS32",
    RdnaSDelayAluClass.FMA_ACCUM_CYCLE_1: "FMA_ACCUM",
    RdnaSDelayAluClass.SALU_CYCLE_1:      "SALU",
    RdnaSDelayAluClass.SALU_CYCLE_2:      "SALU",
    RdnaSDelayAluClass.SALU_CYCLE_3:      "SALU",
}
