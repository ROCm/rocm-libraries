# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.RequiredParameters import getRequiredParametersMin
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components.Subtile.StoreInNLL import (
    NLLStoreEvent,
    nonTemporalDFlags,
    planNLLStoreInterleave,
    subtileStoreInNLLRejectionReason,
)


def _bf16():
    dtype = MagicMock()
    dtype.isBFloat16.return_value = True
    return dtype


def _admissible_state():
    return {
        "SubtileStoreInNLL": 1,
        "ISA": (9, 5, 0),
        "UseSubtileImpl": True,
        "SourceSwap": True,
        "CompactLoopStore": False,
        "StreamK": 0,
        "GlobalSplitU": 1,
        "MIWaveTile": [8, 8],
        "MIArchVgpr": False,
        "NonTemporalD": 4,
        "MacroTile0": 256,
        "MacroTile1": 256,
        "AssertFree0ElementMultiple": 256,
        "AssertFree1ElementMultiple": 256,
        "ProblemType": {
            "UseBeta": False,
            "HighPrecisionAccumulate": True,
            "DestDataType": _bf16(),
            "UseBias": False,
            "UseE": False,
            "UseScaleD": False,
            "UseScaleCD": False,
            "UseScaleAlphaVec": False,
            "OutputAmaxD": False,
            "ActivationType": "none",
        },
    }


def test_parameter_is_explicit_serialized_and_disabled_by_default():
    assert validParameters["SubtileStoreInNLL"] == [0, 1]
    assert defaultSolution["SubtileStoreInNLL"] == 0
    assert "SubtileStoreInNLL" in getRequiredParametersMin()


def test_disabled_path_has_no_rejection():
    state = _admissible_state()
    state["SubtileStoreInNLL"] = 0
    assert subtileStoreInNLLRejectionReason(state) is None


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda s: s.update(ISA=(9, 4, 2)), "requires gfx950"),
        (lambda s: s.update(SourceSwap=False), "requires SourceSwap=1"),
        (lambda s: s.update(StreamK=3), "requires StreamK=0"),
        (lambda s: s.update(GlobalSplitU=0), "requires GlobalSplitU=1"),
        (lambda s: s.update(NonTemporalD=0), "requires NonTemporalD bit 0x4"),
        (lambda s: s.update(AssertFree0ElementMultiple=8), "MacroTile-aligned M"),
        (lambda s: s["ProblemType"].update(UseScaleD=True), "does not support scaleD"),
        (lambda s: s["ProblemType"].update(OutputAmaxD=True), "does not support amaxD"),
    ],
)
def test_narrow_scope_validation(mutate, reason):
    state = _admissible_state()
    mutate(state)
    assert reason in subtileStoreInNLLRejectionReason(state)


def test_admissible_scope_is_enabled():
    assert subtileStoreInNLLRejectionReason(_admissible_state()) is None


def test_universal_problem_type_is_allowed_under_explicit_runtime_contract():
    state = _admissible_state()
    state["ProblemType"].update(
        UseBeta=True, UseBias=True, UseScaleAlphaVec=True,
        ActivationType="hipblaslt_all")
    assert subtileStoreInNLLRejectionReason(state) is None


def test_interleave_places_stores_after_final_producer_with_four_issue_gap():
    mfmas = ["a0", "a1", "a0", "a2", "a3", "a4", "a5", "a6"]
    stores = [
        NLLStoreEvent("store", "a0", "store-a0"),
        NLLStoreEvent("store", "a1", "store-a1"),
    ]
    events = planNLLStoreInterleave(mfmas, stores)

    for store in stores:
        store_idx = events.index(store)
        producer_indices = [
            idx for idx, event in enumerate(events)
            if event.kind == "mfma" and event.accumulator == store.accumulator
        ]
        assert store_idx > max(producer_indices)
        intervening = sum(
            event.kind == "mfma"
            for event in events[max(producer_indices) + 1:store_idx]
        )
        assert intervening == 4


def test_interleave_drains_late_store_before_nll_exit():
    store = NLLStoreEvent("store", "late", "convert-and-store")
    events = planNLLStoreInterleave(["early", "late", "other"], [store])
    assert events[-1] == store
    assert events.index(store) > max(
        idx for idx, event in enumerate(events)
        if event.kind == "mfma" and event.accumulator == "late"
    )


def test_non_temporal_d_bit_is_preserved_for_future_mubuf_store():
    assert nonTemporalDFlags(4) == {"glc": False, "slc": False, "nt": True}
    assert nonTemporalDFlags(7) == {"glc": True, "slc": True, "nt": True}
