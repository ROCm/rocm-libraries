#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
################################################################################
"""Scheduler contracts for PostLoopStoreInNll (PLSIN).

Covers the NGLL dual-arm scaffold in LogicalScheduler:
  * non-PLSIN kernels get the stock single NGLL (byte-identical passthrough);
  * PLSIN kernels get a front guard + PostLoopInitInNGLL / plainNGLL dual arm;
  * coord-hoist WEAVING is gated on plsinLargeTile (MacroTile <= 256x256), and a
    large tile takes the non-woven INIT arm without touching the coord VGPRs.

The dual-arm assertions use a large MacroTile so the (heavy, VGPR-pool-driven)
coord-weave path is skipped -- keeping the test structural and hermetic.
"""

import ast
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from rocisa.code import Module

from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
LS_PATH = os.path.join(TENSILE_ROOT, "Tensile", "Components", "Subtile", "LogicalScheduler.py")


def _kernel(mt0, mt1):
    return {"MacroTile0": mt0, "MacroTile1": mt1}


def _scheduler_with_loop(loop_return):
    sched = LogicalScheduler.__new__(LogicalScheduler)
    sched._emitLoop = MagicMock(return_value=loop_return)
    return sched


# ── non-PLSIN passthrough ───────────────────────────────────────────────────


def test_ngll_passthrough_when_disabled():
    plain = Module("plainNGLL")
    sched = _scheduler_with_loop(plain)
    writer = MagicMock()
    writer.states = SimpleNamespace(postLoopStoreInNll=False)

    result = sched._emitNgllMaybeFused(writer, _kernel(256, 256), "NGLL", [])

    # Byte-identical passthrough: the exact plain _emitLoop result, no guard.
    assert result is plain
    sched._emitLoop.assert_called_once()
    writer.emitFusedStoreGuard.assert_not_called()


# ── PLSIN dual arm (large tile => no coord weaving) ─────────────────────────


def test_ngll_dual_arm_when_enabled_large_tile():
    # Distinct real Modules for the plain arm and the INIT arm.
    modules = [Module("plainNGLL"), Module("NGLL_INIT")]
    sched = LogicalScheduler.__new__(LogicalScheduler)
    sched._emitLoop = MagicMock(side_effect=modules)

    writer = MagicMock()
    writer.states = SimpleNamespace(postLoopStoreInNll=True,
                                    subtileHoistedWriteIndices=None)
    writer.emitFusedStoreGuard = MagicMock(return_value=Module("guard"))

    result = sched._emitNgllMaybeFused(writer, _kernel(512, 512), "NGLL", [])

    # A dual-arm MaybeFused wrapper, not the bare plain arm.
    assert isinstance(result, Module)
    assert result.name == "NGLL_MaybeFused"
    # Front guard emitted exactly once, as a long branch (spans the fused body).
    writer.emitFusedStoreGuard.assert_called_once()
    _, kwargs = writer.emitFusedStoreGuard.call_args
    assert kwargs.get("longBranch") is True
    # Both arms emitted: plain first, then the INIT arm.
    init_labels = [c.args[2] for c in sched._emitLoop.call_args_list]
    assert "NGLL" in init_labels and "NGLL_INIT" in init_labels
    # Large tile => coord weaving skipped => hoisted indices never populated.
    assert writer.states.subtileHoistedWriteIndices is None


# ── source contracts ────────────────────────────────────────────────────────


def _function_node(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("function %s not found" % name)


def test_front_guard_forces_long_branch():
    # The plain label sits past the whole fused store body, so the front guard must
    # request a 32-bit long branch.
    source = open(LS_PATH).read()
    func = _function_node(source, "_emitFusedFrontGuard")
    found = False
    for node in ast.walk(func):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "emitFusedStoreGuard"):
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            assert isinstance(kwargs.get("longBranch"), ast.Constant)
            assert kwargs["longBranch"].value is True
            found = True
    assert found, "_emitFusedFrontGuard must call emitFusedStoreGuard"
