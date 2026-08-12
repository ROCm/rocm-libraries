import copy
import importlib
from pathlib import Path

import yaml
import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

_F = Path(__file__).parent.parent / "LibraryIO" / "data" / "logic_gfx942_HSS_BH.yaml"


def _cfg():
    d = list(yaml.safe_load_all(open(_F)))[0]
    cfg = copy.deepcopy(d[5][0])
    cfg["ProblemType"] = copy.deepcopy(d[4])
    return cfg


def _build(assembler, isa_info_map, cfg, src="k.yaml"):
    return S.Solution(cfg, False, False, False, assembler, isa_info_map, srcName=src)


@pytest.fixture(scope="module")
def base_solution(assembler, isa_info_map):
    return _build(assembler, isa_info_map, _cfg(), "base.yaml")


def test_baseline_valid(base_solution):
    assert base_solution._state["Valid"] is True


def test_baseline_empty_custom_kernel_name_gives_none(base_solution):
    assert base_solution._name is None


def test_baseline_assigned_derived_parameters_true(base_solution):
    assert base_solution._state["AssignedDerivedParameters"] is True


def test_baseline_assigned_problem_independent_true(base_solution):
    assert base_solution._state["AssignedProblemIndependentDerivedParameters"] is True


def test_baseline_problemtype_constructed(base_solution):
    assert base_solution._state["ProblemType"] is not None


def test_custom_kernel_name_propagates_to_name(assembler, isa_info_map):
    cfg = _cfg()
    cfg["CustomKernelName"] = "myCustomK"
    sol = _build(assembler, isa_info_map, cfg, "custom.yaml")
    assert sol._name == "myCustomK"


def test_code_object_version_int_cast_to_str(assembler, isa_info_map):
    cfg = _cfg()
    cfg["CodeObjectVersion"] = 5
    sol = _build(assembler, isa_info_map, cfg, "cov.yaml")
    assert sol._state["CodeObjectVersion"] == "5"


def test_internal_support_params_taken_from_config(assembler, isa_info_map):
    cfg = _cfg()
    cfg["InternalSupportParams"] = {"KernArgsVersion": 99}
    sol = _build(assembler, isa_info_map, cfg, "isp.yaml")
    assert sol._state["InternalSupportParams"]["KernArgsVersion"] == 99


def test_srcname_retained(assembler, isa_info_map):
    sol = _build(assembler, isa_info_map, _cfg(), "unique_src.yaml")
    assert sol.srcName == "unique_src.yaml"
