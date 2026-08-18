################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Integration characterization tests for the heavy parse paths that require
live ``Solution`` construction: ``parseLibraryLogicFile`` / ``parseLibraryLogicData``
and ``parseSolutionsFile`` / ``parseSolutionsData``.

These are driven against a **vendored real library-logic file**
(``data/logic_gfx942_HSS_BH.yaml``, a verbatim copy of a production
aquavanjaram/gfx942 GridBased logic file) using the session ``assembler`` +
``isa_info_map`` fixtures. The parsed result holds live objects, so snapshots
capture a **normalised structural summary** (schedule / arch / counts / sorted
type-mismatches / selected ProblemType fields) rather than the objects
themselves — deterministic across runs in the dev container. The solutions-file
path is exercised by a genuine round-trip: parse logic -> ``writeSolutions``
the real solutions -> ``parseSolutionsFile`` them back.

The custom-kernel branch of ``parseLibraryLogicData`` (a solution whose
``CustomKernelName`` is set) is driven by monkeypatching ``getCustomKernelConfig``
in the ``LibraryIO`` namespace, since the vendored file has no custom-kernel
solution (see ``resistance.md``).
"""

import copy
from pathlib import Path
from typing import List

import pytest

import Tensile.LibraryIO as L

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent / "data" / "logic_gfx942_HSS_BH.yaml"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _summarize_logic(logic):
    """A deterministic structural summary of a parsed LibraryLogic."""
    pt = logic.problemType
    return {
        "schedule": logic.schedule,
        "architecture": logic.architecture,
        "n_solutions": len(logic.solutions),
        "exact_logic_len": len(logic.exactLogic) if logic.exactLogic else None,
        "type_mismatches": sorted(str(k) for k in logic.typeMismatches),
        "operationType": pt["OperationType"],
        "n_bias_types": len(pt["BiasDataTypeList"]),
    }


def _raw_dict():
    """parseLibraryLogicList(...) of the vendored fixture (a dict)."""
    data = L.read(str(_FIXTURE), True)
    assert isinstance(data, List)
    return L.parseLibraryLogicList(copy.deepcopy(data), str(_FIXTURE))


# ===========================================================================
# parseLibraryLogicFile / parseLibraryLogicData — the List path
# ===========================================================================

def test_parse_library_logic_file(assembler, isa_info_map, snapshot):
    logic = L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot


def test_parse_library_logic_data_dict_path(assembler, isa_info_map, snapshot):
    # Passing an already-normalised dict skips the isinstance(data, List) branch
    # and exercises the CUCount/MacDataType defaulting on a dict input.
    data = _raw_dict()
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot


def test_parse_library_logic_data_no_cucount_with_datatypes(assembler, isa_info_map, snapshot):
    # A dict without CUCount (-> CUCount defaulting) whose ProblemType already
    # carries DataTypeA/DataTypeB (-> the getRealDataType* else branches).
    data = _raw_dict()
    del data["CUCount"]
    # MacDataTypeA/B already present -> the "not in" guards take their false arm.
    data["ProblemType"]["MacDataTypeA"] = 4
    data["ProblemType"]["MacDataTypeB"] = 4
    data["ProblemType"]["DataTypeA"] = 4
    data["ProblemType"]["DataTypeB"] = 4
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot


def test_parse_library_logic_data_version_warning(assembler, isa_info_map, snapshot):
    # Incompatible MinimumRequiredVersion -> printWarning path (not a reject).
    data = L.read(str(_FIXTURE), True)
    data = copy.deepcopy(data)
    data[0]["MinimumRequiredVersion"] = "1.0.0"
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot


# ===========================================================================
# parseLibraryLogicData — custom-kernel branch (monkeypatched config)
# ===========================================================================

def test_parse_library_logic_data_custom_kernel(assembler, isa_info_map, monkeypatch, snapshot):
    # CustomKernelName set + an (empty) config -> the custom-kernel merge branch
    # runs; getCustomKernelConfig is monkeypatched so no real kernel is needed.
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp: {})
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_kernel"
    # InternalSupportParams present -> the isp-extraction branch also runs.
    data["Solutions"][0]["InternalSupportParams"] = {"KernelLanguage": "Assembly"}
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot


def test_parse_library_logic_data_custom_kernel_bad_mi(assembler, isa_info_map, monkeypatch):
    # A custom-kernel config with a MatrixInstruction of length != 4 -> ValueError
    # before Solution construction (so MI consistency is irrelevant).
    monkeypatch.setattr(
        L, "getCustomKernelConfig", lambda name, isp: {"MatrixInstruction": [1, 2, 3]}
    )
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_kernel"
    with pytest.raises(ValueError):
        L.parseLibraryLogicData(
            data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
        )


# ===========================================================================
# parseSolutionsFile / parseSolutionsData — round-trip from real solutions
# ===========================================================================

@pytest.fixture
def written_solutions(tmp_path, assembler, isa_info_map):
    """Parse the vendored logic, then write its real solutions to a solutions
    file (no problem sizes) for the parse-back tests."""
    logic = L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    p = tmp_path / "sol.yaml"
    L.writeSolutions(str(p), None, None, None, logic.solutions)
    return p


def test_parse_solutions_file_roundtrip(written_solutions, assembler, isa_info_map, snapshot):
    problemSizes, solutions = L.parseSolutionsFile(
        str(written_solutions), assembler, False, False, False, isa_info_map
    )
    assert {
        "n_solutions": len(solutions),
        "problem_sizes_type": type(problemSizes).__name__,
    } == snapshot


def test_parse_solutions_data_with_bias_activation(written_solutions, assembler, isa_info_map, snapshot):
    # Insert BiasTypeArgs + ActivationArgs header entries so parseSolutionsData
    # advances solutionStartIdxInData past both (branches L413-416).
    data = L.read(str(written_solutions))
    data = data[:2] + [
        {"BiasTypeArgs": [0]},
        {"ActivationArgs": [{"Enum": "none"}]},
    ] + data[2:]
    problemSizes, solutions = L.parseSolutionsData(
        data, str(written_solutions), assembler, False, False, False, isa_info_map
    )
    assert {
        "n_solutions": len(solutions),
        "problem_sizes_type": type(problemSizes).__name__,
    } == snapshot


def test_parse_solutions_data_version_warning(written_solutions, assembler, isa_info_map, snapshot):
    # Incompatible MinimumRequiredVersion -> printWarning path in parseSolutionsData.
    data = L.read(str(written_solutions))
    data[0]["MinimumRequiredVersion"] = "1.0.0"
    problemSizes, solutions = L.parseSolutionsData(
        data, str(written_solutions), assembler, False, False, False, isa_info_map
    )
    assert {
        "n_solutions": len(solutions),
        "problem_sizes_type": type(problemSizes).__name__,
    } == snapshot


def test_parse_solutions_data_too_short(assembler, isa_info_map):
    with pytest.raises(SystemExit):
        L.parseSolutionsData(
            [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": []}],
            "tiny.yaml", assembler, False, False, False, isa_info_map,
        )


def test_parse_solutions_data_missing_problem_sizes(assembler, isa_info_map):
    with pytest.raises(SystemExit):
        L.parseSolutionsData(
            [{"MinimumRequiredVersion": "5.0.0"}, {"NotProblemSizes": []}, {"SolutionIndex": 0}],
            "bad.yaml", assembler, False, False, False, isa_info_map,
        )
