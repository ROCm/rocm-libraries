# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused-store weave/lend policy unit tests (`_selectPlsinFusedStorePolicy`).

The policy returns ``(weaveGroups, lendTiles)`` for the PLSIN fused NLL store:

* ``weaveGroups == {}``   -> keep the planner weave (store/MFMA overlap preserved)
* ``weaveGroups is None`` -> no weave (full operand lend; nothing to overlap)

The behavior under test is the small-tile keep-weave decision for the DPP store-coalesce:
on MT256x256 (MIWT[0]=8, gfx950, MXF4->bf16, MI16) the floor-4 coalesce batch needs ~12 extra
VGPRs; rather than take the full lend (which forces ``weaveGroups=None`` and kills the overlap,
~5-8% on MI355X), keep the weave and lend ONLY the dead K=0 operand holes -- but only when the
loop actually has holes to lend (numSubIterK>=2 and A/B dead tiles). Everything is driven here
in isolation: ``isMxf4SubtilePath`` and the two register helpers are stubbed so the test pins the
decision logic, not the data-type gate or the register math (those have their own tests).
"""

from types import SimpleNamespace

import pytest

import tensilelite.Components.Subtile.LogicalScheduler as ls_mod
from tensilelite.Components.Subtile.LogicalScheduler import LogicalScheduler

pytestmark = pytest.mark.unit

# Sentinels so the test can see WHICH lend the policy asked for without the real VGPR math.
_HOLES = {"A": {0}, "B": set()}     # has A/B dead tiles -> has_ab_holes True
_NO_HOLES = {"A": set(), "B": set()}  # single-K tile -> no holes


def _scheduler(num_sub_iter_k, dead_ids, monkeypatch, mxf4=True):
    monkeypatch.setattr(ls_mod, "isMxf4SubtilePath", lambda k: mxf4)
    s = object.__new__(LogicalScheduler)
    s.config = SimpleNamespace(numSubIterK=num_sub_iter_k)
    s._deadOperandTileIds = lambda unroll_iter: dead_ids
    # Echo the arg so a test can tell a dead-hole (partial) lend from a full (None) lend.
    s._operandLendVgprs = lambda arg=None: ("lend", arg)
    return s


def _k(mt0=256, mt1=256, miwt0=8, isa=(9, 5, 0), mi_m=16):
    return {"MacroTile0": mt0, "MacroTile1": mt1, "MIWaveTile": [miwt0, miwt0],
            "ISA": isa, "MatrixInstM": mi_m}


# ── small-tile keep-weave (the change under test) ─────────────────────────────

def test_mt256_miwt8_with_holes_keeps_weave_and_partial_lends(monkeypatch):
    s = _scheduler(num_sub_iter_k=2, dead_ids=_HOLES, monkeypatch=monkeypatch)
    weave, lend = s._selectPlsinFusedStorePolicy(_k())
    assert weave == {}                      # weave KEPT (store/MFMA overlap preserved)
    assert lend == ("lend", _HOLES)         # lent ONLY the dead K=0 holes


def test_mt256_miwt8_no_holes_falls_back_to_full_lend(monkeypatch):
    # Single-K tile: no dead holes to borrow, so there is no overlap to protect -> full lend.
    s = _scheduler(num_sub_iter_k=2, dead_ids=_NO_HOLES, monkeypatch=monkeypatch)
    weave, lend = s._selectPlsinFusedStorePolicy(_k())
    assert weave is None                    # no weave
    assert lend == ("lend", None)           # full operand lend


def test_mt256_miwt8_single_subiter_falls_back_to_full_lend(monkeypatch):
    # numSubIterK < 2: the dead-hole lend is not available, so take the full lend.
    s = _scheduler(num_sub_iter_k=1, dead_ids=_HOLES, monkeypatch=monkeypatch)
    weave, lend = s._selectPlsinFusedStorePolicy(_k())
    assert weave is None
    assert lend == ("lend", None)


# ── scope: the keep-weave is gated to the coalesce-firing config ──────────────

def test_non_mxf4_small_tile_keeps_develop_no_lend(monkeypatch):
    # Not on the MXF4 subtile path -> develop's plain weave, no lend (unchanged).
    s = _scheduler(num_sub_iter_k=2, dead_ids=_HOLES, monkeypatch=monkeypatch, mxf4=False)
    assert s._selectPlsinFusedStorePolicy(_k()) == ({}, [])


def test_wrong_wavetile_keeps_develop_no_lend(monkeypatch):
    # MIWT[0] != 8 (the only correctness-verified geometry) -> no keep-weave lend.
    s = _scheduler(num_sub_iter_k=2, dead_ids=_HOLES, monkeypatch=monkeypatch)
    assert s._selectPlsinFusedStorePolicy(_k(miwt0=4)) == ({}, [])


def test_wrong_isa_keeps_develop_no_lend(monkeypatch):
    s = _scheduler(num_sub_iter_k=2, dead_ids=_HOLES, monkeypatch=monkeypatch)
    assert s._selectPlsinFusedStorePolicy(_k(isa=(9, 4, 2))) == ({}, [])


# ── large tile path is unchanged by the small-tile edit ───────────────────────

def test_large_tile_with_holes_keeps_weave(monkeypatch):
    s = _scheduler(num_sub_iter_k=2, dead_ids=_HOLES, monkeypatch=monkeypatch)
    weave, lend = s._selectPlsinFusedStorePolicy(_k(mt0=512, mt1=256))
    assert weave == {}
    assert lend == ("lend", _HOLES)


def test_large_tile_no_holes_full_lends(monkeypatch):
    s = _scheduler(num_sub_iter_k=2, dead_ids=_NO_HOLES, monkeypatch=monkeypatch)
    weave, lend = s._selectPlsinFusedStorePolicy(_k(mt0=512, mt1=256))
    assert weave is None
    assert lend == ("lend", None)
