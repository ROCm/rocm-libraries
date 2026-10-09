# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared client options must agree for generator objects and packaged metadata."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tensilelite.Common import state
from tensilelite.Contractions import ProblemType
from tensilelite.Utilities.ClientConfig import problemTypeOptions
from tensilelite.Utilities.LibraryIO import writeMessagePack

pytestmark = pytest.mark.unit


def problem_type(**overrides):
    problem = ProblemType.FromOriginalState(
        {
            "TotalIndices": 4,
            "NumIndicesC": 3,
            "IndicesBatch": [2],
            "IndicesFree": [0, 1],
            "IndicesSummation": [3],
            "IndexAssignmentsA": [0, 3, 2],
            "IndexAssignmentsB": [3, 1, 2],
            "ComplexConjugateA": False,
            "ComplexConjugateB": False,
            "TransposeA": False,
            "TransposeB": False,
            "DataType": "H",
            "DestDataType": "S",
            "ComputeDataType": "S",
            "ActivationComputeDataType": "S",
            "Batched": True,
            "BiasSrc": "D",
            **overrides,
        }
    )
    # Solution.FromOriginalState supplies this solution-level field before
    # serializing the problem type, including for ordinary non-MX solutions.
    problem.mxScaleFormat = 0
    return problem


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, {"bias-source": 3, "strided-batched": True, "activation-type": "None"}),
        (
            {
                "UseBias": 1,
                "BiasSrc": "B",
                "BiasDataTypeList": [0],
                "UseScaleAB": "Vector",
                "UseScaleCD": True,
                "UseScaleAlphaVec": 1,
                "ActivationType": "all",
            },
            {
                "bias-source": 1,
                "use-bias": 1,
                "use-scaleAB": "Vector",
                "use-scaleCD": True,
                "use-scaleAlphaVec": 1,
                "activation-type": "All",
            },
        ),
        (
            {"UseE": True, "DataTypeE": "H", "OutputAmaxD": True},
            {"use-e": True, "e-type": "Half", "amaxD-type": "Float"},
        ),
        (
            {"Sparse": 1, "MetadataLayout": 1, "SwizzleTensorA": True, "GroupedGemm": True},
            {"sparse": 1, "metadata-layout": 1, "swizzle-tensor-a": True, "grouped-gemm": True},
        ),
        (
            {"MXBlockA": 32, "MXBlockB": 32, "DataTypeMXSA": "E8", "DataTypeMXSB": "E8"},
            {"mx-a-block": 32, "mx-a-type": "E8", "mx-b-block": 32, "mx-b-type": "E8"},
        ),
    ],
)
def test_generator_and_serialized_problem_options_agree(overrides, expected):
    """Use the real generator serializer, including enum and omitted-scalar handling."""
    problem = problem_type(**overrides)
    options = problemTypeOptions(problem)
    assert problemTypeOptions(state(problem)) == options
    assert options["a-type"] == "Half"
    assert options["alpha-type"] == options["beta-type"] == "Float"
    assert {key: options[key] for key in expected} == expected


def test_candidates_and_binary_helpers_do_not_import_generator(tmp_path):
    """Exercise the standalone API in a fresh process with generator imports forbidden."""
    record = {
        "index": 17,
        "name": "solution-name",
        "kernelName": "kernel-name",
        "problemType": state(problem_type()),
        "problemPredicate": {"type": "And", "value": []},
        "sizeMapping": {"macroTile": [128, 96, 1], "depthU": 64, "globalSplitU": 1},
    }
    writeMessagePack(
        tmp_path / "TensileLibrary_test_gfx942.dat",
        {"solutions": [record], "library": {}},
        indexed=True,
    )
    script = """
import importlib.abc
import sys

class NoGenerator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rocisa" or fullname in {
            "tensilelite.ClientWriter", "tensilelite.LibraryIO",
            "tensilelite.Contractions", "tensilelite.SolutionStructs",
        }:
            raise AssertionError(f"Unexpected generator import: {fullname}")

sys.meta_path.insert(0, NoGenerator())
import yaml
from tensilelite.Utilities.gemm_candidates import (
    load_library_metadata, candidate_gemms, bench_yaml, tensile_configs,
)
library = load_library_metadata(sys.argv[1], target="gfx942", solution_indices=[17])
cases = candidate_gemms(library, 17)
assert len(cases) > 1
assert cases[0].profile.a_type == "Half"
assert yaml.safe_load(bench_yaml(cases))[0]["solution_index"] == 17
configs = tensile_configs(cases)
assert all("solution-start-idx=17\\n" in config for config in configs)
assert sum(config.count("problem-size=") for config in configs) == len(cases)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
