import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def test_disable_runtime_stagger_u_zeros_all_fields():
    state = {
        "StaggerU": 999,
        "StaggerUMapping": 888,
        "StaggerUStride": 777,
        "InternalSupportParams": {"SupportCustomStaggerU": True},
    }
    S._disableRuntimeStaggerU(state)

    assert state["StaggerU"] == 0
    assert state["StaggerU"] is not None
    assert state["StaggerUMapping"] == 0
    assert state["StaggerUMapping"] is not None
    assert state["StaggerUStride"] == 0
    assert state["StaggerUStride"] is not None

    assert state["InternalSupportParams"]["SupportCustomStaggerU"] is False

    assert set(state) == {
        "StaggerU",
        "StaggerUMapping",
        "StaggerUStride",
        "InternalSupportParams",
    }
    assert set(state["InternalSupportParams"]) == {"SupportCustomStaggerU"}

def test_disable_runtime_stagger_u_preserves_only_expected_keys():

    state = {
        "StaggerU": 5,
        "StaggerUMapping": 5,
        "StaggerUStride": 5,
        "InternalSupportParams": {"SupportCustomStaggerU": True, "Other": 42},
    }
    S._disableRuntimeStaggerU(state)

    assert state["StaggerU"] == 0
    assert state["StaggerUMapping"] == 0
    assert state["StaggerUStride"] == 0
    assert state["InternalSupportParams"]["SupportCustomStaggerU"] is False

    assert set(state["InternalSupportParams"]) == {"SupportCustomStaggerU", "Other"}
    assert state["InternalSupportParams"]["Other"] == 42
    assert set(state) == {
        "StaggerU",
        "StaggerUMapping",
        "StaggerUStride",
        "InternalSupportParams",
    }
