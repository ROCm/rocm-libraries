#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
################################################################################
"""Code-generation contracts for PostLoopStoreInNll (PLSIN).

These tests guard three invariants that the eligibility-only tests in
test_subtile_plsin.py do not cover:

  1. SGPR isolation -- the tolerant defineSgpr free-pool handling is scoped to
     PLSIN kernels; every non-PLSIN kernel keeps the strict (develop) allocator so
     its SGPR layout is unchanged.
  2. The hoisted PostLoopFusedStore predicate is only emitted for fp32-compute
     PLSIN kernels; otherwise computePostLoopFusedStore is a no-op.
  3. The fused store applies alpha for EVERY alpha value (alpha is not a
     fused-store guard) -- asserted structurally on the globalWriteElements call.

Assertions are structural (module emptiness, allocator behavior, call keywords)
rather than full-assembly goldens, so they survive register-numbering churn.
"""

import ast
import os
from types import SimpleNamespace

import pytest

from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from rocisa.code import Module

from Tensile.KernelWriterAssembly import KernelWriterAssembly

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
KWA_PATH = os.path.join(TENSILE_ROOT, "Tensile", "KernelWriterAssembly.py")
LOGICAL_SCHEDULER_PATH = os.path.join(
    TENSILE_ROOT, "Tensile", "Components", "Subtile", "LogicalScheduler.py")


# ── Harness ───────────────────────────────────────────────────────────────


def _bare_kwa(postLoopStoreInNll):
    """A KernelWriterAssembly with only the fields defineSgpr / the fused-store
    predicate touch. Bypasses __init__ (which needs a full solution)."""
    kwa = KernelWriterAssembly.__new__(KernelWriterAssembly)
    kwa.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                defaultPreventOverflow=False, printRP=False)
    kwa.sgprs = {}
    kwa.states = SimpleNamespace(
        postLoopStoreInNll=postLoopStoreInNll,
        freeSgprVarPool=set(),
    )
    return kwa


def _park_and_borrow(kwa, name, numSgprs=1):
    """Park `name` in the free var pool, then borrow its registers as a temp so
    setSgprToInUseState(name) would raise -- the exact StreamK-reuse condition the
    PLSIN defineSgpr path tolerates and the baseline path must reject."""
    idx = kwa.sgprPool.checkOutAligned(numSgprs, numSgprs, name, False)
    kwa.sgprs[name] = idx
    kwa.sgprPool.addFromCheckOut(idx)              # park: Available + tracked
    kwa.states.freeSgprVarPool.add(name)
    borrow = kwa.sgprPool.checkOutAligned(numSgprs, numSgprs, "borrow", False)
    assert borrow == idx, "borrow must reuse the parked registers"
    return idx


# ── 1. SGPR isolation: defineSgpr ───────────────────────────────────────────


def test_define_sgpr_baseline_rejects_borrowed_parked_var():
    # Non-PLSIN kernels keep develop's strict allocator: a parked var whose
    # registers are borrowed must surface as a RuntimeError, never be masked.
    kwa = _bare_kwa(postLoopStoreInNll=False)
    _park_and_borrow(kwa, "SrdWS")
    with pytest.raises(RuntimeError):
        kwa.defineSgpr("SubtileMGuard", 1)


def test_define_sgpr_plsin_tolerates_borrowed_parked_var():
    # PLSIN kernels tolerate the borrowed-parked case (StreamK reuses SrdWS during
    # the main loop while buildSubtileFusedStore defines guard SGPRs).
    kwa = _bare_kwa(postLoopStoreInNll=True)
    borrowedIdx = _park_and_borrow(kwa, "SrdWS")
    ret = kwa.defineSgpr("SubtileMGuard", 1)
    assert ret is not None
    assert "SubtileMGuard" in kwa.sgprs
    # The guard must NOT reuse the borrowed registers.
    assert kwa.sgprs["SubtileMGuard"] != borrowedIdx
    # The borrowed var is left in-use (untouched) -- not double-freed.
    pool = kwa.sgprPool.getPool()
    assert pool[borrowedIdx].status == RegisterPool.Status.InUse


@pytest.mark.parametrize("plsin", [False, True])
def test_define_sgpr_normal_parked_var_restored_in_both_modes(plsin):
    # When no parked var is borrowed, both modes behave identically: the checkout
    # skips the parked registers and restores the parked var to the free state.
    kwa = _bare_kwa(postLoopStoreInNll=plsin)
    idx = kwa.sgprPool.checkOutAligned(2, 2, "SrdWS", False)
    kwa.sgprs["SrdWS"] = idx
    kwa.addSgprVarToPool("SrdWS")  # parked + Available (not borrowed)
    ret = kwa.defineSgpr("SubtileNGuard", 1)
    assert ret is not None
    # New allocation must not land on the parked SrdWS registers.
    guardIdx = kwa.sgprs["SubtileNGuard"]
    assert guardIdx not in (idx, idx + 1)
    # SrdWS stays parked (free var pool + Available) after the define.
    assert "SrdWS" in kwa.states.freeSgprVarPool
    assert kwa.sgprPool.getPool()[idx].status == RegisterPool.Status.Available


# ── 2. Hoisted fused-store predicate emission gate ──────────────────────────


def _dtype(is_single):
    return SimpleNamespace(isSingle=lambda: is_single)


def _predicate_kernel(compute_single):
    return {"ProblemType": {"ComputeDataType": _dtype(compute_single)}}


def test_compute_post_loop_fused_store_empty_when_disabled():
    kwa = _bare_kwa(postLoopStoreInNll=False)
    kwa._plsinDeferredScalePtrLoads = "sentinel"
    mod = kwa.computePostLoopFusedStore(_predicate_kernel(compute_single=True))
    assert list(mod.flatitems()) == []
    # The stale scale-pointer stash is always reset (arbitrary-alpha path).
    assert kwa._plsinDeferredScalePtrLoads is None


def test_compute_post_loop_fused_store_empty_for_non_fp32():
    # PLSIN on but non-fp32 compute -> not _plsinFusedFlagEligible -> no flag.
    kwa = _bare_kwa(postLoopStoreInNll=True)
    kwa._plsinDeferredScalePtrLoads = None
    mod = kwa.computePostLoopFusedStore(_predicate_kernel(compute_single=False))
    assert list(mod.flatitems()) == []


def test_plsin_fused_flag_eligible_requires_plsin_and_fp32():
    assert _bare_kwa(True)._plsinFusedFlagEligible(_predicate_kernel(True)) is True
    assert _bare_kwa(True)._plsinFusedFlagEligible(_predicate_kernel(False)) is False
    assert _bare_kwa(False)._plsinFusedFlagEligible(_predicate_kernel(True)) is False


# ── 3. Arbitrary-alpha fused store (structural source contract) ─────────────


def _function_node(source, func_name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError("function %s not found" % func_name)


def test_short_k_tail_jump_follows_deferred_preloop_setup():
    # K<DepthU must execute PLSIN's deferred LRA/predicate setup before jumping
    # to the tail. The insertion point is therefore the first preloop wait_gr,
    # not the slot immediately after initC.
    source = open(LOGICAL_SCHEDULER_PATH).read()
    func = _function_node(source, "emitMainAndExitLoops")
    body = ast.get_source_segment(source, func)
    deferred_setup_idx = body.index("_lraDeferred")
    tail_insert_idx = body.index("tail_insert_idx = next")
    wait_target_idx = body.index("if em.opType == 'wait_gr'", tail_insert_idx)
    assert deferred_setup_idx < tail_insert_idx < wait_target_idx


def _find_call(func_node, callee_attr):
    for node in ast.walk(func_node):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == callee_attr):
            return node
    return None


def test_build_subtile_fused_store_applies_alpha_for_any_value():
    # Alpha is NOT a fused-store guard: buildSubtileFusedStore must call
    # globalWriteElements with applyAlpha=True (and the beta0/full-tile shape), so
    # the normal scalar-alpha multiply runs for every alpha value.
    source = open(KWA_PATH).read()
    func = _function_node(source, "buildSubtileFusedStore")
    call = _find_call(func, "globalWriteElements")
    assert call is not None, "buildSubtileFusedStore must call globalWriteElements"
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert isinstance(kwargs.get("applyAlpha"), ast.Constant) and kwargs["applyAlpha"].value is True
    assert isinstance(kwargs.get("edge"), ast.Constant) and kwargs["edge"].value is False
    assert isinstance(kwargs.get("noGSUBranch"), ast.Constant) and kwargs["noGSUBranch"].value is True
    # betas=[False] -> beta-0 only (no C read).
    betas = kwargs.get("betas")
    assert isinstance(betas, ast.List) and len(betas.elts) == 1
    assert isinstance(betas.elts[0], ast.Constant) and betas.elts[0].value is False


def test_build_subtile_fused_store_replays_srd_init_for_short_k():
    # Store-init ALU is normally woven into the FUSED loop. numIter<PGR skips that
    # loop at runtime, so the store body must contain a second, uniquely-labelled
    # complete init rather than using an uninitialized / un-offset SrdD.
    source = open(KWA_PATH).read()
    func = _function_node(source, "buildSubtileFusedStore")
    calls = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "buildSubtileStoreInitModule"
    ]
    assert len(calls) >= 2
    suffixes = [
        keyword.value
        for call in calls
        for keyword in call.keywords
        if keyword.arg == "labelSuffix"
    ]
    assert any(
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Add)
        and isinstance(value.right, ast.Constant)
        and value.right.value == "Short"
        for value in suffixes
    ), "short-K SrdD replay must use distinct assembly labels"
    body = ast.get_source_segment(source, func)
    assert 'src0=sgpr("SizesSum+%u"' in body
    assert 'src1=pgr * kernel["DepthU"]' in body
    assert "plsinSrdDInit_numIter" not in body, (
        "short-K guard scratch must not clobber values live into the SrdD remainder")


def test_fused_store_predicate_excludes_alpha():
    # The hoisted predicate must fold only the structural sub-guards; alpha (and
    # scale pointers) must never be ANDed into PostLoopFusedStore.
    source = open(KWA_PATH).read()
    func = _function_node(source, "computePostLoopFusedStore")
    # No SGPR named "Alpha" may feed the flag accumulator in this function body.
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sgpr":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert "Alpha" not in arg.value


def test_fused_store_predicate_routes_non_null_bias_to_plain_nll():
    source = open(KWA_PATH).read()
    func = _function_node(source, "computePostLoopFusedStore")
    body = ast.get_source_segment(source, func)

    assert 'if kernel["ProblemType"]["UseBias"]' in body
    assert 'normalBiasOffset = self.argLoader.getOffset()' in body
    assert "extBiasOffset = (" in body
    assert "PLSIN_LoadExternalBiasPtr" in body
    assert 'SWaitCnt(kmcnt=0, comment="wait for runtime AddressBias")' in body
    assert 'comment="bad |= AddressBias[0]"' in body
    assert 'comment="bad |= AddressBias[1] (non-null -> plain)"' in body
    assert 'sgpr("AddressBias' not in body, (
        "the early guard must not reference the late-defined AddressBias alias")


def test_fused_store_guard_requires_full_tile():
    # The runtime guard must require a COMPLETE MacroTile in M and N (not the
    # relaxed subtile-aligned/NonEdge check), so the branch-free full-tile store is
    # only taken by full-tile owners.
    source = open(KWA_PATH).read()
    func = _function_node(source, "emitFusedStoreGuard")
    edge_checks = [n for n in ast.walk(func)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "checkIsEdgeSubtile"]
    assert edge_checks, "emitFusedStoreGuard must gate on checkIsEdgeSubtile"
    for call in edge_checks:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert isinstance(kwargs.get("requireFullTile"), ast.Constant)
        assert kwargs["requireFullTile"].value is True
