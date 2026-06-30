################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — StreamK big-cluster codegen characterization for gfx942.

Exercises the three large uncovered clusters in ``Tensile/Components/StreamK.py``:

  * Lines 630-784  — ``storeBranchesCommon`` tree-reduction arm
                      Gate: ``kernel["StreamKFixupTreeReduction"] == 1``
                            AND ``kernel["StreamKAtomic"] == 0``
                      Config: StreamK=[1,2], StreamKFixupTreeReduction=1

  * Lines 2915-3091 — ``StreamKDynamic.graWorkGroup``
                       Gate: ``kernel["StreamK"] == 4`` (StreamKDynamic class)
                       Config: StreamK=4, StreamKAtomic=0

  * Lines 3142-3215 — ``StreamKDynamic.storeBranches``
                       Gate: ``kernel["StreamK"] == 4``
                             AND ``kernel["StreamKAtomic"] == 0`` (early-return
                             at line 3145 skips the body when Atomic==1)
                       Config: StreamK=4, StreamKAtomic=0

CPU-only: no GPU required. The emit harness instantiates rocisa and runs the
Python+rocisa codegen stack without compiling or launching any GPU kernels.

Pure-assert (no snapshot): checks basename prefix, .amdgcn_target presence,
and assembly markers specific to each path (SAtomicInc for dynamic-mode's
atomic increment of the work queue, SK_Fixup_Tree label for the tree-reduction
fixup loop). This makes the tests robust to coverage-jitter from multiprocessing
and avoids snapshot maintenance overhead.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_DYNAMIC_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "streamk_dynamic.yaml",
)

_FIXUP_TREE_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "streamk_fixup_tree.yaml",
)


# ---------------------------------------------------------------------------
# StreamK=4 (Dynamic) — targets lines 2915-3091 and 3142-3215
# ---------------------------------------------------------------------------

def test_r7_streamk_dynamic_emits_assembly():
    """StreamK=4 (Dynamic) emits gfx942 assembly with err==0."""
    results = emit_kernels_from_config(_DYNAMIC_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"


def test_r7_streamk_dynamic_has_atomic_inc():
    """Dynamic StreamK graWorkGroup uses SAtomicInc for work-queue dispatch.

    StreamKDynamic.graWorkGroup (lines 2915-3091) emits an ``s_atomic_add``
    instruction (Tensile's SAtomicInc instruction, rendered as
    ``s_atomic_add`` in GFX9 assembly) to atomically fetch the next work item
    index from the queue.  Confirm that the instruction is present in at least
    one emitted kernel.
    """
    results = emit_kernels_from_config(_DYNAMIC_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel"

    found_atomic = any(
        "s_atomic_add" in (src or "") or "s_atomic_inc" in (src or "")
        for _b, src, _err in results
    )
    assert found_atomic, (
        "Expected at least one kernel with 's_atomic_add' or 's_atomic_inc' "
        "(graWorkGroup atomic work-item fetch, lines 2985-2987); "
        "none found. Dynamic StreamK may not have been selected."
    )


def test_r7_streamk_dynamic_has_sk_done_label():
    """Dynamic StreamK graWorkGroup emits SK_Done label.

    graWorkGroup (lines 2915-3091) creates ``skDone = Label("SK_Done", "")``
    at line 2919.  The canonicalized source contains the label name.
    Confirm it appears so we know the graWorkGroup body ran.
    """
    results = emit_kernels_from_config(_DYNAMIC_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel"

    found = any(
        "SK_Done" in (src or "")
        for _b, src, _err in results
    )
    assert found, (
        "Expected 'SK_Done' label from StreamKDynamic.graWorkGroup "
        "(line 2919); not found. Dynamic StreamK graWorkGroup may not have run."
    )


def test_r7_streamk_dynamic_storeBranches_emits_fixup_loop():
    """Dynamic StreamK storeBranches emits the partial-tile fixup loop.

    StoreBranches (lines 3142-3215) creates ``skFixupLabel`` and emits
    the fixup loop when StreamKAtomic==0 (i.e. the early-return at line 3145
    does NOT trigger). The label "SK_Fixup" appears in the emitted assembly.
    """
    results = emit_kernels_from_config(_DYNAMIC_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel"

    found = any(
        "SK_Fixup" in (src or "")
        for _b, src, _err in results
    )
    assert found, (
        "Expected 'SK_Fixup' label from StreamKDynamic.storeBranches "
        "(line 3149); not found. storeBranches fixup loop may not have emitted."
    )


# ---------------------------------------------------------------------------
# StreamK=[1,2] + StreamKFixupTreeReduction=1 — targets lines 630-784
# ---------------------------------------------------------------------------

def test_r7_streamk_fixup_tree_emits_assembly():
    """StreamKFixupTreeReduction=1 emits gfx942 assembly with err==0."""
    results = emit_kernels_from_config(_FIXUP_TREE_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"


def test_r7_streamk_fixup_tree_has_sk_fixup_tree_label():
    """StreamKFixupTreeReduction=1 emits SK_Fixup_Tree label.

    storeBranchesCommon (lines 629-784) creates
    ``skFixupTreeLabel = Label(...'SK_Fixup_Tree'...)`` at line 630 when
    ``StreamKFixupTreeReduction == 1``.  Verify the label appears in the
    emitted assembly, confirming the tree-reduction arm was entered.
    """
    results = emit_kernels_from_config(_FIXUP_TREE_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel"

    found = any(
        "SK_Fixup_Tree" in (src or "")
        for _b, src, _err in results
    )
    assert found, (
        "Expected 'SK_Fixup_Tree' label from storeBranchesCommon "
        "(line 630); not found. StreamKFixupTreeReduction=1 branch may "
        "not have been entered."
    )


def test_r7_streamk_fixup_tree_has_tree_loop_labels():
    """StreamKFixupTreeReduction=1 emits all three tree-loop control labels.

    storeBranchesCommon emits a tree-reduction fixup loop with three labels:
      - SK_Fixup_TreeLoop_Start (line 631)
      - SK_Fixup_Wait_Flag       (line 632)
      - endFixupLoop             (line 633)
    All three must appear when StreamKFixupTreeReduction==1 is active.
    """
    results = emit_kernels_from_config(_FIXUP_TREE_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel"

    for label in ("SK_Fixup_TreeLoop_Start", "SK_Fixup_Wait_Flag", "endFixupLoop"):
        found = any(label in (src or "") for _b, src, _err in results)
        assert found, (
            f"Expected '{label}' label from storeBranchesCommon tree-reduction "
            f"arm (lines 631-633); not found in any emitted kernel."
        )
