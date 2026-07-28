import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def _seed_state(pap, tdm):
    return {
        "PrefetchAcrossPersistent": pap,
        "TDMInst": tdm,
        "StaggerU": 999,
        "StaggerUMapping": 888,
        "StaggerUStride": 777,
        "InternalSupportParams": {"SupportCustomStaggerU": True},
    }

def _assert_untouched(state, pap, tdm):
    assert set(state) == {
        "PrefetchAcrossPersistent",
        "TDMInst",
        "StaggerU",
        "StaggerUMapping",
        "StaggerUStride",
        "InternalSupportParams",
    }
    assert state["PrefetchAcrossPersistent"] == pap
    assert state["TDMInst"] == tdm
    assert state["StaggerU"] == 999
    assert state["StaggerUMapping"] == 888
    assert state["StaggerUStride"] == 777
    assert state["InternalSupportParams"] == {"SupportCustomStaggerU": True}

def _assert_disabled(state, pap, tdm):
    assert set(state) == {
        "PrefetchAcrossPersistent",
        "TDMInst",
        "StaggerU",
        "StaggerUMapping",
        "StaggerUStride",
        "InternalSupportParams",
    }
    assert state["PrefetchAcrossPersistent"] == pap
    assert state["TDMInst"] == tdm
    assert state["StaggerU"] == 0
    assert state["StaggerUMapping"] == 0
    assert state["StaggerUStride"] == 0
    assert state["InternalSupportParams"] == {"SupportCustomStaggerU": False}
    assert set(state["InternalSupportParams"]) == {"SupportCustomStaggerU"}

def test_pap_true_tdm_3_disables_stagger():

    state = _seed_state(True, 3)
    S._disableUnsupportedRuntimeStaggerU(state)
    _assert_disabled(state, True, 3)

def test_pap_true_tdm_not_3_leaves_stagger():

    state = _seed_state(True, 0)
    S._disableUnsupportedRuntimeStaggerU(state)
    _assert_untouched(state, True, 0)

def test_pap_false_tdm_3_leaves_stagger():

    state = _seed_state(False, 3)
    S._disableUnsupportedRuntimeStaggerU(state)
    _assert_untouched(state, False, 3)

def test_pap_false_tdm_not_3_leaves_stagger():
    state = _seed_state(False, 0)
    S._disableUnsupportedRuntimeStaggerU(state)
    _assert_untouched(state, False, 0)
