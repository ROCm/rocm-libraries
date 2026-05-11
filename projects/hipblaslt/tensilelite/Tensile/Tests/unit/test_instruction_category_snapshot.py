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

"""Snapshot test for the rocm-libraries-009 (re-scoped) consolidation.

Every class name that the historical 13 `_*_CLASS_NAMES` sets recognized
must continue to map to its corresponding `InstructionCategory` after the
consolidation. The legacy predicate chain (`_is_lr`, `_is_lw`, ...) is
captured here as a frozen reference table; the new `category()` lookup
must produce the equivalent answer for every registered class name.

This is the safety-net the bead's acceptance criterion calls for: any
drift between the new central map and the legacy `*_CLASS_NAMES`
membership will fail this test loudly.
"""

import pytest

from Tensile.Components.InstructionCategory import (
    InstructionCategory,
    category,
    category_of_class_name,
    registered_class_names,
)


# Frozen snapshot of the (class_name -> category) pairs that the historical
# 13 `_*_CLASS_NAMES` sets and the 10 `_is_*` discriminators implied. Captured
# from ScheduleCapture.py at HEAD before the consolidation deletes the
# legacy sets. Any divergence between this table and the central registry
# will fail the test.
_LEGACY_SNAPSHOT = {
    # _LR_CLASS_NAMES
    "DSLoadB32":         InstructionCategory.LR,
    "DSLoadB64":         InstructionCategory.LR,
    "DSLoadB128":        InstructionCategory.LR,
    "DSLoadB256":        InstructionCategory.LR,
    "DSLoadInstruction": InstructionCategory.LR,
    # _LW_CLASS_NAMES
    "DSStoreB8":          InstructionCategory.LW,
    "DSStoreB16":         InstructionCategory.LW,
    "DSStoreB32":         InstructionCategory.LW,
    "DSStoreB64":         InstructionCategory.LW,
    "DSStoreB128":        InstructionCategory.LW,
    "DSStoreU16":         InstructionCategory.LW,
    "DSStoreB96":         InstructionCategory.LW,
    "DSStoreB192":        InstructionCategory.LW,
    "DSStoreB256":        InstructionCategory.LW,
    "DSStoreD16HIB16":    InstructionCategory.LW,
    "DSStoreB8HID16":     InstructionCategory.LW,
    "DSStoreInstruction": InstructionCategory.LW,
    # _GR_CLASS_NAMES
    "BufferLoadB32":         InstructionCategory.GR,
    "BufferLoadB64":         InstructionCategory.GR,
    "BufferLoadB128":        InstructionCategory.GR,
    "GlobalLoadB32":         InstructionCategory.GR,
    "GlobalLoadB64":         InstructionCategory.GR,
    "GlobalLoadB128":        InstructionCategory.GR,
    "BufferLoadInstruction": InstructionCategory.GR,
    "GlobalLoadInstruction": InstructionCategory.GR,
    "GlobalReadInstruction": InstructionCategory.GR,
    # _MFMA_CLASS_NAMES
    "MFMAInstruction": InstructionCategory.MFMA,
    # _SWAIT_CLASS_NAMES / _SBARRIER_CLASS_NAMES / _SNOP_CLASS_NAMES /
    # _SSETPRIO_CLASS_NAMES
    "SWaitCnt":  InstructionCategory.SWAIT,
    "SBarrier":  InstructionCategory.SBARRIER,
    "SNop":      InstructionCategory.SNOP,
    "SSetPrior": InstructionCategory.SSETPRIO,
    # _CVT_PACK_CLASS_NAMES
    "VCvtPkF32toBF16": InstructionCategory.CVT_PACK,
    # _MIDDLE_PACK_CLASS_NAMES
    "PVCvtBF16toFP32": InstructionCategory.MIDDLE_PACK,
    "VCvtBF16toFP32":  InstructionCategory.MIDDLE_PACK,
    "VSubF32":         InstructionCategory.MIDDLE_PACK,
    "VDot2CF32BF16":   InstructionCategory.MIDDLE_PACK,
    # _SMEM_CLASS_NAMES
    "SLoadB32":             InstructionCategory.SMEM,
    "SLoadB64":             InstructionCategory.SMEM,
    "SLoadB128":            InstructionCategory.SMEM,
    "SLoadB256":            InstructionCategory.SMEM,
    "SLoadB512":            InstructionCategory.SMEM,
    "SStoreB32":            InstructionCategory.SMEM,
    "SStoreB64":            InstructionCategory.SMEM,
    "SStoreB128":           InstructionCategory.SMEM,
    "SMemLoadInstruction":  InstructionCategory.SMEM,
    "SMemStoreInstruction": InstructionCategory.SMEM,
    # _FLAT_CLASS_NAMES
    "FlatLoadB8":           InstructionCategory.FLAT,
    "FlatLoadB16":          InstructionCategory.FLAT,
    "FlatLoadB32":          InstructionCategory.FLAT,
    "FlatLoadB64":          InstructionCategory.FLAT,
    "FlatLoadB128":         InstructionCategory.FLAT,
    "FlatStoreB8":          InstructionCategory.FLAT,
    "FlatStoreB16":         InstructionCategory.FLAT,
    "FlatStoreB32":         InstructionCategory.FLAT,
    "FlatStoreB64":         InstructionCategory.FLAT,
    "FlatStoreB128":        InstructionCategory.FLAT,
    "FLATReadInstruction":  InstructionCategory.FLAT,
    "FLATStoreInstruction": InstructionCategory.FLAT,
    # _VECTOR_STORE_CLASS_NAMES
    "BufferStoreB32":         InstructionCategory.VECTOR_STORE,
    "BufferStoreB64":         InstructionCategory.VECTOR_STORE,
    "BufferStoreB128":        InstructionCategory.VECTOR_STORE,
    "GlobalStoreB32":         InstructionCategory.VECTOR_STORE,
    "GlobalStoreB64":         InstructionCategory.VECTOR_STORE,
    "GlobalStoreB128":        InstructionCategory.VECTOR_STORE,
    "BufferStoreInstruction": InstructionCategory.VECTOR_STORE,
    "GlobalStoreInstruction": InstructionCategory.VECTOR_STORE,
}


@pytest.mark.parametrize("cls_name,expected_cat", sorted(_LEGACY_SNAPSHOT.items()))
def test_class_name_lookup_matches_legacy(cls_name, expected_cat):
    """category_of_class_name(cls_name) must return the legacy bucket.

    Every class name the historical predicate chain recognized must
    survive the consolidation in the same bucket.
    """
    assert category_of_class_name(cls_name) == expected_cat, (
        f"Class {cls_name!r}: legacy snapshot says {expected_cat.name}, "
        f"central map says {category_of_class_name(cls_name)}."
    )


@pytest.mark.parametrize("cls_name,expected_cat", sorted(_LEGACY_SNAPSHOT.items()))
def test_instance_dispatch_matches_legacy(cls_name, expected_cat):
    """category(inst) on an instance whose type-name matches `cls_name`
    must return the legacy bucket.

    Stand-in instance: a stub class whose `__name__` matches the registered
    class name. This is the same dispatch shape the production capture
    uses (`type(inst).__name__` lookup), so the test exercises the
    instance-dispatch path end-to-end.
    """
    stub_cls = type(cls_name, (), {})
    inst = stub_cls()
    assert category(inst) == expected_cat


def test_unknown_class_returns_none():
    """category(inst) returns None for an unregistered class.

    Preserves the truth-value behavior of the legacy predicates: each
    `_is_*(inst)` returned False for both wrong-bucket inputs AND for
    unrecognized inputs. A None category compares not-equal to every
    InstructionCategory member, matching that semantics.
    """
    class _NotARealRocisaClass:
        pass

    inst = _NotARealRocisaClass()
    assert category(inst) is None
    for member in InstructionCategory:
        assert category(inst) != member


def test_registered_class_names_matches_snapshot():
    """The central registry's key set must be exactly the snapshot's key set.

    Catches additions to the central map that were not reflected in the
    snapshot table — and snapshot drift if the snapshot is stale.
    """
    assert set(registered_class_names()) == set(_LEGACY_SNAPSHOT.keys()), (
        "Central registry diverged from legacy snapshot. "
        f"Only-in-registry: {set(registered_class_names()) - set(_LEGACY_SNAPSHOT)}; "
        f"Only-in-snapshot: {set(_LEGACY_SNAPSHOT) - set(registered_class_names())}."
    )


def test_no_duplicate_class_name_assignments():
    """Each registered class name maps to exactly one category.

    Sanity check on the registry data structure itself; protects against
    accidental copy-paste duplicates that would silently shadow earlier
    entries.
    """
    names = list(registered_class_names())
    assert len(names) == len(set(names)), (
        "Duplicate class names in registered_class_names() — registry is "
        "supposed to be a 1:1 map."
    )
