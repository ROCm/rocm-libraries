################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
InstructionShape — finer-grained per-instruction taxonomy that the
`(InstructionShape, InstructionShape) -> List[GapRule]` table in
`ArchProfile.gap_rules` keys on (rocm-libraries-vmua).

InstructionShape is a refinement of `InstructionCategory`:

  - `InstructionCategory.MFMA` covers every matrix-multiply-accumulate
    rocisa instance, regardless of its hardware family. The validator's
    timing rules need to discriminate the 4x4 PackMFMA family from the
    standard 16x16 / 32x32 / DGEMM / SMFMA families because the finish-
    cycle counts and cross-family stall thresholds differ. The shape
    enum splits MFMA into hardware-family-aligned values:
      MFMA_4x4            — TF32 4x4 PackMFMA family (1 quad-cycle finish).
      MFMA_STANDARD       — every other MFMA family (3 quad-cycle finish).

  - `InstructionCategory.CVT_PACK` covers the TF32-emulation
    `v_cvt_pk_bf16_f32` family. The shape enum keeps it as one value
    (`CVT_PACK`) since the existing gap rules don't sub-discriminate.

  - `InstructionCategory.{LR,LW,GR,SWAIT,SBARRIER,SNOP,...}` map to
    same-named shape values (1:1) because the gap-rule table doesn't
    sub-discriminate any of them today.

  - Pack-categorized non-MFMA non-CVT instances (the MIDDLE_PACK family)
    map to `MIDDLE_PACK` shape, mirroring `InstructionCategory.MIDDLE_PACK`.

  - Non-Pack rocisa-instance ALU ops (VXor, SMov, VAdd, etc.) that the
    validator's `_is_alu_producer` predicate claims map to `ALU`. These
    are immediate-visibility producers under most rules; the rare
    timing-relevant case (C-9: ALU→MFMA forwarding gap) is encoded as
    a per-pair gap rule below.

The taxonomy here is DRIVEN BY GAP-RULE NEEDS — every shape value is
referenced by at least one rule in `_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules`.
We do NOT speculatively add shapes (e.g. MFMA_F8 sub-bucket, VALU_TRANS,
VOPD_PAIR) that the audit memo §3 sketched but no current gap-rule row
discriminates on. When a future bead adds a rule that needs those, the
shape values land here at that moment.

Placement rationale: InstructionShape lives in its own module rather
than in `InstructionCategory.py` because `shape_of(node)` consults
`WrappedInstruction.is_mfma` / `is_cvt_pack` (rocisa-aware predicates)
to discriminate MFMA-family vs CVT-Pack vs MIDDLE_PACK among Pack-
categorized instances. `InstructionCategory.py` is intentionally
rocisa-import-free (per its module docstring); pulling rocisa-aware
shape derivation into it would violate that property. A new module
sidesteps the issue.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from Tensile.Components.InstructionCategory import (
    InstructionCategory,
    category as _category_of,
)


class InstructionShape(Enum):
    """Per-instruction shape used by `ArchProfile.gap_rules`.

    Each value names ONE per-pair gap-rule discriminator. The naming
    mirrors `InstructionCategory` for the categories that don't sub-
    discriminate (LR / LW / GR / CVT_PACK / MIDDLE_PACK / SWAIT / ...)
    and refines `MFMA` into per-family shapes.

    Members consumed by `_DEFAULT_CDNA4_ARCH_PROFILE.gap_rules`:
      MFMA_4x4        — 4x4 PackMFMA (TF32 emul). Pack* category +
                        MFMAInstruction rocisa class. 1 quad-cycle finish.
      MFMA_STANDARD   — every other MFMA (16x16 / 32x32 / DGEMM / SMFMA /
                        F8F6F4). category=='MFMA'. 3 quad-cycle finish.
      CVT_PACK        — `v_cvt_pk_bf16_f32` (TF32 emul intermediate).
                        Pack* category + VCvtPkF32toBF16 rocisa.
      MIDDLE_PACK     — TF32-emul middle-16 ops (PVCvtBF16toFP32, VSubF32,
                        VCvtBF16toFP32, VDot2CF32BF16). Pack* category +
                        non-MFMA non-CVT rocisa.
      ALU             — Generic ALU producer (`_is_alu_producer`-claimed),
                        non-Pack category. Includes VXor, SMov, VAdd, etc.
      LR              — DS-load (local read).
      LW              — DS-store (local write).
      GR              — Buffer/global vector load.
      SWAIT           — `s_waitcnt`.
      SBARRIER        — `s_barrier`.
      SNOP            — `s_nop`.
      SSETPRIO        — `s_setprio`.
      SMEM            — Scalar memory load/store.
      FLAT            — Flat memory load/store.
      VECTOR_STORE    — Buffer/global vector store.
      OTHER           — Catch-all for unknown shapes (test-fixture stubs
                        with rocisa_inst=None and unrecognized categories).
    """
    MFMA_4x4      = "MFMA_4x4"
    MFMA_STANDARD = "MFMA_STANDARD"
    CVT_PACK      = "CVT_PACK"
    MIDDLE_PACK   = "MIDDLE_PACK"
    ALU           = "ALU"
    LR            = "LR"
    LW            = "LW"
    GR            = "GR"
    SWAIT         = "SWAIT"
    SBARRIER      = "SBARRIER"
    SNOP          = "SNOP"
    SSETPRIO      = "SSETPRIO"
    SMEM          = "SMEM"
    FLAT          = "FLAT"
    VECTOR_STORE  = "VECTOR_STORE"
    OTHER         = "OTHER"


# Map from `InstructionCategory` member to the directly-corresponding
# `InstructionShape` value, used when no sub-discrimination is needed.
# MFMA / CVT_PACK / MIDDLE_PACK are NOT in this map because they go
# through `shape_of()`'s instance-aware path (Pack* category overlaps
# with multiple shapes).
_CATEGORY_TO_SHAPE: dict = {
    InstructionCategory.LR:           InstructionShape.LR,
    InstructionCategory.LW:           InstructionShape.LW,
    InstructionCategory.GR:           InstructionShape.GR,
    InstructionCategory.SWAIT:        InstructionShape.SWAIT,
    InstructionCategory.SBARRIER:     InstructionShape.SBARRIER,
    InstructionCategory.SNOP:         InstructionShape.SNOP,
    InstructionCategory.SSETPRIO:     InstructionShape.SSETPRIO,
    InstructionCategory.SMEM:         InstructionShape.SMEM,
    InstructionCategory.FLAT:         InstructionShape.FLAT,
    InstructionCategory.VECTOR_STORE: InstructionShape.VECTOR_STORE,
}


def shape_of(node) -> InstructionShape:
    """Return the `InstructionShape` of `node`.

    Discrimination order:
      1. Pack* category + MFMA-shaped rocisa  -> MFMA_4x4.
         (PackMFMA carve-out: syntactically MFMAInstruction but
         categorized Pack* because the macro classifier groups them with
         the surrounding pack chain. See `GraphNode.is_mfma_pack_producer`.)
      2. Pack* category + VCvtPkF32toBF16     -> CVT_PACK.
      3. Pack* category + MIDDLE_PACK rocisa   -> MIDDLE_PACK.
      4. category == "MFMA"                    -> MFMA_STANDARD.
      5. Direct `InstructionCategory` lookup   -> matching shape value.
      6. ALU-immediate (rocisa instance is not in any non-ALU category
         per `InstructionCategory`, plus a non-LDS/non-GR fallback for
         test fixtures where `rocisa_inst is None`) -> ALU.
      7. Otherwise                             -> OTHER.

    Parameters:
        node: A `GraphNode`-shaped object (real or duck-typed) carrying
              `category: str` and `rocisa_inst: object | None`.

    Returns:
        The classifying `InstructionShape`.
    """
    cat = getattr(node, "category", "") or ""
    inst = getattr(node, "rocisa_inst", None)

    # Pack* category — three sub-shapes by rocisa class.
    if cat.startswith("Pack"):
        inst_cat = _category_of(inst) if inst is not None else None
        if inst_cat is InstructionCategory.MFMA:
            return InstructionShape.MFMA_4x4
        if inst_cat is InstructionCategory.CVT_PACK:
            return InstructionShape.CVT_PACK
        if inst_cat is InstructionCategory.MIDDLE_PACK:
            return InstructionShape.MIDDLE_PACK
        # Pack-categorized but unrecognized rocisa class; treat as ALU
        # (matches `_is_alu_producer`'s behavior for Pack* with non-MFMA
        # rocisa).
        return InstructionShape.ALU

    # category == "MFMA" with MFMA-shaped rocisa: standard MFMA.
    if cat == "MFMA":
        return InstructionShape.MFMA_STANDARD

    # Mirror `_is_alu_producer`'s discrimination order: when a rocisa
    # instance is present, its CLASS determines the shape (the category
    # bucket is informational, not authoritative). This is the same
    # property that lets a DTL m0 setter (SMovB32 with category="GRA")
    # land in InstructionShape.ALU rather than InstructionShape.GR — the
    # category names which emission group emitted the instruction; the
    # rocisa class names what the instruction is.
    if inst is not None:
        inst_cat = _category_of(inst)
        if inst_cat is not None:
            direct = _CATEGORY_TO_SHAPE.get(inst_cat)
            if direct is not None:
                return direct
            # MFMA / CVT_PACK / MIDDLE_PACK without the Pack* prefix is
            # an unusual shape — treat as their direct shape.
            if inst_cat is InstructionCategory.MFMA:
                return InstructionShape.MFMA_STANDARD
            if inst_cat is InstructionCategory.CVT_PACK:
                return InstructionShape.CVT_PACK
            if inst_cat is InstructionCategory.MIDDLE_PACK:
                return InstructionShape.MIDDLE_PACK
        # rocisa instance whose class is not registered in
        # InstructionCategory: by `_is_alu_producer` semantics this is
        # an ALU instruction (SMov / SAdd / VXor / etc. — the `_NON_ALU_
        # CATEGORIES` check returns False because `_category(inst)` is
        # None, and the predicate then returns True).
        return InstructionShape.ALU

    # rocisa_inst is None — fall back to category. This branch is only
    # reached for test fixtures that don't carry a rocisa instance.
    _LDS_LIKE = ("LRA0", "LRA1", "LRA3", "LRB0", "LRB1", "LRB3",
                 "LWA", "LWB")
    _GR_LIKE  = ("GRA", "GRB", "GR")
    if cat in _LDS_LIKE:
        if cat.startswith("LR"):
            return InstructionShape.LR
        return InstructionShape.LW
    if cat in _GR_LIKE:
        return InstructionShape.GR

    return InstructionShape.OTHER
