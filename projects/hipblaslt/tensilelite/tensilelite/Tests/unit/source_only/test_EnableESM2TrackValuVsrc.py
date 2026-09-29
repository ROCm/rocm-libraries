# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Test for evaluateEnableESM2TrackValuVsrc() in Solution.py.

The ESM2 VALU-src VA_VDST stamp (EnableESM2TrackValuVsrc) is derived from the
Sparse problem type: on for sparse, off otherwise.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest

_SOLUTION_PY = Path(__file__).resolve().parents[3] / "SolutionStructs" / "Solution.py"


def _func_body() -> str:
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    start = source.find("def evaluateEnableESM2TrackValuVsrc()")
    assert start != -1, "evaluateEnableESM2TrackValuVsrc not found in Solution.py"
    return source[start : start + 500]


def _nested_function(name: str):
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(textwrap.dedent(ast.get_source_segment(source, function)), namespace)
    return namespace[name]


class _SupportedDataType:
    @staticmethod
    def isDouble():
        return False

    @staticmethod
    def isDoubleComplex():
        return False


def _evaluate_stinkytofu_esm2(sparse):
    function = _nested_function("evaluateStinkyTofuESM2")
    isa = (12, 5, 0)
    function.__globals__.update(
        isa=isa,
        isaInfoMap={isa: SimpleNamespace(archCaps={"HasSchedMode": True})},
        state={
            "ProblemType": {
                "Sparse": sparse,
                "MacDataTypeA": _SupportedDataType(),
                "MacDataTypeB": _SupportedDataType(),
                "ComputeDataType": _SupportedDataType(),
            }
        },
    )
    return function()


def test_derives_from_sparse():
    """The flag must be derived from state["ProblemType"]["Sparse"]."""
    assert 'state["ProblemType"]["Sparse"]' in _func_body()


def test_state_key_assigned():
    """state["EnableESM2TrackValuVsrc"] must be assigned from the evaluator."""
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    assert 'state["EnableESM2TrackValuVsrc"] = evaluateEnableESM2TrackValuVsrc()' in source


@pytest.mark.parametrize("sparse", [1, 2])
def test_stinkytofu_esm2_rejects_sparse_problem_types(sparse):
    assert _evaluate_stinkytofu_esm2(sparse) is False


def test_stinkytofu_esm2_accepts_supported_dense_problem_type():
    assert _evaluate_stinkytofu_esm2(0) is True
