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


def _zero_solution_free_size():
    """A parsed-dict variant with no solutions and a FreeSize matching table so
    the heavy per-solution construction path is skipped and only the top-level
    ProblemType (built at the start of parseLibraryLogicData) is exercised.

    FreeSizeLibrary.FromOriginalState tolerates an empty table ([0, 0] -> empty
    range), so the library still constructs with zero solutions."""
    data = _raw_dict()
    data["Solutions"] = []
    data["ExactLogic"] = []
    data["LibraryType"] = "FreeSize"
    data["Library"] = {"indexOrder": None, "table": [0, 0], "distance": None}
    return data


def test_parse_library_logic_data_macdatatypea_present_not_overwritten(assembler, isa_info_map):
    """The 'MacDataTypeA not in ProblemType' guard (line 522) must read the real
    key: a supplied MacDataTypeA distinct from DataType is kept, not overwritten."""
    data = _zero_solution_free_size()
    data["ProblemType"]["MacDataTypeA"] = 0
    data["ProblemType"]["MacDataTypeB"] = 0
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert str(logic.problemType["MacDataTypeA"]) == "S"


def test_parse_library_logic_data_index_info_top_level_print(assembler, isa_info_map, capsys):
    """With printIndexAssignmentInfo=True and no solutions, exactly one
    'IndicesFree:' block is printed: the one from the top-level ProblemType
    (line 543-545)."""
    data = _zero_solution_free_size()
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, True, isa_info_map, False
    )
    assert capsys.readouterr().out.count("IndicesFree:") == 1

_FIXTURE = Path(__file__).parent / "data" / "logic_gfx942_HSS_BH.yaml"


def test_parse_library_logic_data_type_mismatch_recorded(assembler, isa_info_map):
    """A benign ProblemType type mismatch (bool where int expected) is recorded,
    not raised, because the top-level ProblemType is built with
    raiseOnTypeMismatch=False (line 547)."""
    data = _zero_solution_free_size()
    data["ProblemType"]["MetadataLayout"] = True
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert ("MetadataLayout", "bool", "int") in logic.typeMismatches


def test_parse_library_logic_data_type_mismatch_threads_srcfile(assembler, isa_info_map):
    """The top-level ProblemType threads the real srcFile (line 546) into the
    type-mismatch collector, so a recorded mismatch carries the source file."""
    data = _zero_solution_free_size()
    data["ProblemType"]["MetadataLayout"] = True
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert logic.typeMismatches[("MetadataLayout", "bool", "int")]["files"] == {str(_FIXTURE)}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PT_DTYPE_KEYS = (
    "DataType",
    "DataTypeA",
    "DataTypeB",
    "MacDataTypeA",
    "MacDataTypeB",
    "OperationType",
)


def _summarize_problem_type(pt):
    """Stringified datatype-derivation fields of a ProblemType."""
    return {k: str(pt[k]) for k in _PT_DTYPE_KEYS if k in pt}


def _summarize_solution(sol):
    """Deterministic per-solution view of the fields set during
    solutionStateToSolution (ISA/CUCount/DeviceNames/derived-param flags and
    the overwritten ProblemType datatypes)."""
    return {
        "KernelLanguage": sol.get("KernelLanguage"),
        "ISA": str(sol.get("ISA")),
        "CUCount": sol.get("CUCount"),
        "DeviceNames": sol.get("DeviceNames"),
        "CustomKernelName": sol.get("CustomKernelName"),
        "AssignedDerivedParameters": sol.get("AssignedDerivedParameters"),
        "AssignedProblemIndependentDerivedParameters": sol.get(
            "AssignedProblemIndependentDerivedParameters"
        ),
        "problemType": _summarize_problem_type(sol["ProblemType"]),
    }


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
        "problemType": _summarize_problem_type(pt),
        "solutions": [_summarize_solution(s) for s in logic.solutions],
    }


def _raw_dict():
    """parseLibraryLogicList(...) of the vendored fixture (a dict)."""
    data = L.read(str(_FIXTURE), True)
    assert isinstance(data, List)
    return L.parseLibraryLogicList(copy.deepcopy(data), str(_FIXTURE))


def test_parse_library_logic_data_datatypeb_true_arm_real_key(assembler, isa_info_map):
    """DataTypeB-absent true arm (line 534) must write the real 'DataTypeB' key
    into the raw ProblemType dict, not a corrupted key."""
    data = _raw_dict()
    assert "DataTypeB" not in data["ProblemType"]
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert "DataTypeB" in data["ProblemType"]
    assert "XXDataTypeBXX" not in data["ProblemType"]
    assert "datatypeb" not in data["ProblemType"]
    assert "DATATYPEB" not in data["ProblemType"]


def test_parse_library_logic_data_datatypeb_else_arm_real_key(assembler, isa_info_map):
    """DataTypeB-present else arm (line 536) must overwrite the real 'DataTypeB'
    key, never a corrupted key."""
    data = _raw_dict()
    data["ProblemType"]["DataTypeB"] = 4
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert data["ProblemType"]["DataTypeB"] == 4
    assert "XXDataTypeBXX" not in data["ProblemType"]
    assert "datatypeb" not in data["ProblemType"]
    assert "DATATYPEB" not in data["ProblemType"]


def test_parse_library_logic_data_assembly_isa_overwrite(assembler, isa_info_map):
    """The KernelLanguage=='Assembly' guard (line 552) overwrites the solution ISA
    with the architecture-derived ISA (gfx942 -> 9.4.2), replacing any embedded
    value."""
    data = _raw_dict()
    data["Solutions"][0]["ISA"] = [9, 0, 10]
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert str(logic.solutions[0]["ISA"]) == "SemanticVersion(major=9, minor=4, patch=2)"


# ===========================================================================
# parseLibraryLogicFile / parseLibraryLogicData — the List path
# ===========================================================================

def test_parse_library_logic_file(assembler, isa_info_map, snapshot):
    logic = L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot
    assert logic.solutions
    assert all(s.srcName == str(_FIXTURE) for s in logic.solutions)
    assert all(s.splitGSU is False for s in logic.solutions)


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


def test_parse_library_logic_data_version_warning(assembler, isa_info_map, capsys, snapshot):
    # Incompatible MinimumRequiredVersion -> printWarning path (not a reject).
    data = L.read(str(_FIXTURE), True)
    data = copy.deepcopy(data)
    data[0]["MinimumRequiredVersion"] = "1.0.0"
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    # The warning must actually fire and thread srcFile, the offending version,
    # and the running Tensile version into its message (pins the branch guard,
    # the printWarning call, and every .format argument).
    warnings = [
        line for line in capsys.readouterr().out.splitlines()
        if "Tensile::WARNING:" in line and "does not match Tensile version" in line
    ]
    assert len(warnings) == 1
    msg = warnings[0]
    assert str(_FIXTURE) in msg
    assert "1.0.0" in msg
    assert L.__version__ in msg
    assert msg == (
        "Tensile::WARNING: Version = "
        + str(_FIXTURE)
        + " in library logic file 1.0.0 does not match Tensile version = "
        + L.__version__
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
    with pytest.raises(ValueError, match="MatrixInstruction can only be of length 4"):
        L.parseLibraryLogicData(
            data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
        )


def test_parse_library_logic_data_mac_datatypeb_guard(assembler, isa_info_map):
    """A config-supplied MacDataTypeB must be preserved (the 'MacDataTypeB' not
    in guard takes its false arm). Setting it to Float (0/'S') while DataType is
    Half (4/'H') yields the unsupported GEMM type (H, S, S) which raises. A
    mutated guard that is always true overwrites MacDataTypeB with the DataType,
    silently producing the valid (H, H, S) type and never raising."""
    data = _raw_dict()
    data["ProblemType"]["MacDataTypeB"] = 0
    with pytest.raises(
        Exception,
        match=r"This typed-GEMM \(Ti, To, Tc\) = \(H, S, S\) is not supported yet\.",
    ):
        L.parseLibraryLogicData(
            data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
        )


def test_parse_library_logic_data_no_spurious_problemtype_keys(assembler, isa_info_map):
    """The DataTypeB real-type else-arm must assign back to the canonical
    'DataTypeB' key. A mutated assignment target writes the realized value under
    a spurious key, leaving it in the shared problemType state that becomes
    logic.problemType."""
    data = _raw_dict()
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert "XXDataTypeBXX" not in logic.problemType
    assert "datatypeb" not in logic.problemType
    assert "DATATYPEB" not in logic.problemType


def test_parse_library_logic_data_object_wiring(assembler, isa_info_map):
    """Pins the positional/keyword wiring of the inner Solution() call, the
    MasterSolutionLibrary.FromOriginalState() call, and the LibraryLogic()
    construction: splitGSU is the passed False (not None), srcName is the source
    file (not None or ''), the library is a real MasterSolutionLibrary (not None
    or the typeMismatches dict), and the library's solutions carry splitGSU."""
    logic = L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert all(s.splitGSU is False for s in logic.solutions)
    assert all(s.srcName == str(_FIXTURE) for s in logic.solutions)
    assert type(logic.library).__name__ == "MasterSolutionLibrary"
    assert all(
        s.originalSolution.splitGSU is False
        for s in logic.library.solutions.values()
    )


def test_parse_library_logic_data_problemtype_mismatch_not_raised(assembler, isa_info_map):
    """The inner Solution() must be constructed with
    raiseProblemTypeOnTypeMismatch=False so a mistyped ProblemType parameter is
    collected rather than raised. A mutated flag (keyword removed -> default
    True, or set True) makes the inner Solution's ProblemType reconstruction
    raise ConfigTypeError mid-parse."""
    data = _raw_dict()
    data["ProblemType"]["UseBeta"] = 1
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert any("UseBeta" in str(k) for k in logic.typeMismatches)


def test_parse_library_logic_data_index_assignment_info_threaded(assembler, isa_info_map, capsys):
    """printIndexAssignmentInfo=True must reach every ProblemType construction:
    the top-level ProblemType, the inner Solution's ProblemType, and the
    newLibrary solution's ProblemType (3 IndicesFree prints for the single
    fixture solution). None-ing the argument to FromOriginalState silences the
    newLibrary construction, dropping the count to 2."""
    data = _raw_dict()
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, True, isa_info_map, False
    )
    out = capsys.readouterr().out
    assert out.count("IndicesFree:") == 3


def test_parse_library_logic_data_lazy_library_loading(assembler, isa_info_map):
    """lazyLibraryLoading=True must thread into FromOriginalState so the built
    MasterSolutionLibrary populates lazyLibraries. None-ing the argument takes
    the non-lazy libraryOrder and leaves lazyLibraries empty."""
    data = _raw_dict()
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, True
    )
    assert len(logic.library.lazyLibraries) > 0


def test_parse_library_logic_data_logic_file_threaded(assembler, isa_info_map, capsys, monkeypatch):
    """srcFile must be threaded into BOTH invalid-chip-id diagnostics: the
    per-solution one (srcFile passed to Solution, line 594) and the
    library-level one (logicFile=srcFile passed to FromOriginalState, line 611).
    The gfx942 fixture (Device 0049/0050, unsupported) fires the warning for
    each, so the real source path appears exactly twice and '<unknown>' never
    appears. None-ing or dropping either keyword drops one occurrence to
    '<unknown>'. The chip-id gate is monkeypatched on so the diagnostic runs."""
    import Tensile.Hardware

    monkeypatch.setattr(Tensile.Hardware, "supportsChipIdPredicate", lambda gfx: True)
    data = _raw_dict()
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    lines = capsys.readouterr().out.splitlines()
    assert sum(1 for line in lines if line == f"*   File: {_FIXTURE}") == 2
    assert not any(line == "*   File: <unknown>" for line in lines)


def test_parse_library_logic_data_datatype_overwrite_divergence(assembler, isa_info_map, snapshot):
    """MacDataTypeA/B diverge from DataType while DataTypeA/B/DataType stay Half,
    so every nested solutionStateToSolution datatype guard-and-assignment becomes
    observable on the shared logic.problemType. Clean derivation yields
    MacDataTypeA=MacDataTypeB=S and DataTypeA=DataTypeB=DataType=H; each guard/key
    mutant overwrites exactly one field to a different letter."""
    data = _raw_dict()
    data["ProblemType"]["MacDataTypeA"] = 0
    data["ProblemType"]["MacDataTypeB"] = 0
    data["ProblemType"]["DataTypeA"] = 4
    data["ProblemType"]["DataTypeB"] = 4
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert _summarize_logic(logic) == snapshot


def test_parse_library_logic_file_uses_custom_loader(tmp_path, assembler, isa_info_map):
    """parseLibraryLogicFile must read with the custom event loader (read(...,
    True)). An all-caps ``TRUE`` then parses to a Python bool, so ``Batched`` is
    a well-typed bool and is NOT recorded as a str/bool type mismatch. The
    default StrictTypeLoader used when the loader flag is dropped/falsified keeps
    ``TRUE`` a str, which validateProblemTypeParameterTypes flags."""
    text = _FIXTURE.read_text().replace("\n  Batched: true", "\n  Batched: TRUE", 1)
    p = tmp_path / "logic.yaml"
    p.write_text(text)
    logic = L.parseLibraryLogicFile(
        str(p), assembler, False, False, False, isa_info_map, False
    )
    assert ("Batched", "str", "bool") not in logic.typeMismatches


def test_parse_library_logic_file_forwards_reject_flag(tmp_path, assembler, isa_info_map):
    """printSolutionRejectionReason must be forwarded. With it True a rejected
    solution (GSU and StreamK both disabled) makes reject() raise, since the
    fixture solution has a valid SolutionIndex."""
    text = _FIXTURE.read_text().replace("GlobalSplitU: 1", "GlobalSplitU: 0", 1)
    p = tmp_path / "logic.yaml"
    p.write_text(text)
    with pytest.raises(Exception, match="rejection of a LibraryLogic is not expected"):
        L.parseLibraryLogicFile(
            str(p), assembler, False, True, False, isa_info_map, False
        )


def test_parse_library_logic_file_forwards_index_print(assembler, isa_info_map, capsys):
    """printIndexAssignmentInfo must be forwarded to ProblemType derivation so
    the index-assignment lines are emitted to stdout."""
    L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, True, isa_info_map, False
    )
    lines = capsys.readouterr().out.splitlines()
    assert "IndicesFree:  [0, 1]" in lines


def test_parse_library_logic_file_forwards_lazy_loading(assembler, isa_info_map):
    """lazyLibraryLoading must be forwarded to MasterSolutionLibrary.FromOriginalState
    so the placeholder/lazy library map is populated."""
    logic = L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, True
    )
    assert len(logic.library.lazyLibraries) == 1


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
    assert all(s.srcName == str(written_solutions) for s in solutions)
    assert all(s.splitGSU is False for s in solutions)
    assert {
        "n_solutions": len(solutions),
        "problem_sizes_type": type(problemSizes).__name__,
        "solutions": [_summarize_solution(s) for s in solutions],
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
        "solutions": [_summarize_solution(s) for s in solutions],
    } == snapshot


def test_parse_solutions_data_version_warning(written_solutions, assembler, isa_info_map, capsys, snapshot):
    # Incompatible MinimumRequiredVersion -> printWarning path in parseSolutionsData.
    data = L.read(str(written_solutions))
    data[0]["MinimumRequiredVersion"] = "1.0.0"
    problemSizes, solutions = L.parseSolutionsData(
        data, str(written_solutions), assembler, False, False, False, isa_info_map
    )
    warnings = [
        line for line in capsys.readouterr().out.splitlines()
        if "Tensile::WARNING:" in line and "does not match Tensile version" in line
    ]
    assert len(warnings) == 1
    msg = warnings[0]
    assert str(written_solutions) in msg
    assert "1.0.0" in msg
    assert L.__version__ in msg
    assert msg == (
        "Tensile::WARNING: Version = {} in solution file 1.0.0 "
        "does not match Tensile version = {}".format(written_solutions, L.__version__)
    )
    assert {
        "n_solutions": len(solutions),
        "problem_sizes_type": type(problemSizes).__name__,
        "solutions": [_summarize_solution(s) for s in solutions],
    } == snapshot


def test_parse_solutions_data_too_short(assembler, isa_info_map, capsys):
    with pytest.raises(SystemExit) as ei:
        L.parseSolutionsData(
            [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": []}],
            "tiny.yaml", assembler, False, False, False, isa_info_map,
        )
    assert ei.value.code == -1
    assert capsys.readouterr().out == (
        "Tensile::FATAL: Solution file tiny.yaml is missing required fields "
        "(len = 2 < 3\n"
    )


def test_parse_solutions_data_resets_derived_flags(written_solutions, assembler, isa_info_map):
    """parseSolutionsData force-resets both derived-parameter flags to the
    literal ``False`` on each input solution dict before Solution construction,
    so old-version logic yamls are always re-derived."""
    data = L.read(str(written_solutions))
    for entry in data[2:]:
        entry["AssignedProblemIndependentDerivedParameters"] = True
        entry["AssignedDerivedParameters"] = True
    L.parseSolutionsData(
        data, str(written_solutions), assembler, False, False, False, isa_info_map
    )
    for entry in data[2:]:
        assert entry["AssignedProblemIndependentDerivedParameters"] is False
        assert entry["AssignedDerivedParameters"] is False


def test_parse_library_logic_data_short_list_threads_srcfile(assembler, isa_info_map, capsys):
    """A too-short List routes through parseLibraryLogicList, whose FATAL message
    must carry the srcFile argument threaded from parseLibraryLogicData; the
    srcFile->None argument mutant renders 'None' instead of the path."""
    with pytest.raises(SystemExit):
        L.parseLibraryLogicData(
            [{"MinimumRequiredVersion": "5.0.0"}], "SENTINEL_PATH.yaml",
            assembler, False, False, False, isa_info_map, False,
        )
    assert capsys.readouterr().out == (
        "Tensile::FATAL: Library logic file SENTINEL_PATH.yaml "
        "is missing required fields (len = 1 < 9)\n"
    )


def test_parse_solutions_data_solution_args_threaded(
    written_solutions, assembler, isa_info_map, monkeypatch
):
    """The per-solution ``Solution(...)`` call threads splitGSU,
    printSolutionRejectionReason, printIndexAssignmentInfo, the source file, and
    raiseProblemTypeOnTypeMismatch=False through verbatim; a spy on the Solution
    collaborator pins each positional/keyword argument value."""
    real = L.Solution
    captured = {}

    def spy(*args, **kwargs):
        captured["splitGSU"] = args[1]
        captured["prr"] = args[2]
        captured["pia"] = args[3]
        captured["src"] = args[6] if len(args) > 6 else "<MISSING>"
        captured["raise"] = kwargs.get("raiseProblemTypeOnTypeMismatch", "<MISSING>")
        return real(*args, **kwargs)

    monkeypatch.setattr(L, "Solution", spy)
    data = L.read(str(written_solutions))
    L.parseSolutionsData(
        data, str(written_solutions), assembler, False, False, False, isa_info_map
    )
    assert captured["splitGSU"] is False
    assert captured["prr"] is False
    assert captured["pia"] is False
    assert captured["src"] == str(written_solutions)
    assert captured["raise"] is False


def test_parse_solutions_data_missing_problem_sizes(assembler, isa_info_map, capsys):
    with pytest.raises(SystemExit):
        L.parseSolutionsData(
            [{"MinimumRequiredVersion": "5.0.0"}, {"NotProblemSizes": []}, {"SolutionIndex": 0}],
            "bad.yaml", assembler, False, False, False, isa_info_map,
        )
    assert capsys.readouterr().out == (
        "Tensile::FATAL: Solution file bad.yaml doesn't begin with ProblemSizes\n"
    )


def test_parse_solutions_data_problem_sizes_config_threaded(
    written_solutions, assembler, isa_info_map
):
    """data[1][\"ProblemSizes\"] is threaded verbatim into ProblemSizes(...); a
    truthy but unsupported size-type entry reaches the parser and triggers its
    printExit (SystemExit). A mutant that hardcodes the config to None skips the
    `if config:` branch and returns normally."""
    data = L.read(str(written_solutions))
    data[1] = {"ProblemSizes": [{"Foo": 1}]}
    with pytest.raises(SystemExit):
        L.parseSolutionsData(
            data, str(written_solutions), assembler, False, False, False, isa_info_map
        )


def test_parse_solutions_data_bias_only_advances_one(
    written_solutions, assembler, isa_info_map
):
    """A BiasTypeArgs header with no ActivationArgs advances
    solutionStartIdxInData by exactly one, so every real solution entry is still
    parsed; a mutant that advances by two drops the trailing solution."""
    orig = L.read(str(written_solutions))
    n_solutions = len(orig) - 2
    data = orig[:2] + [{"BiasTypeArgs": [0]}] + orig[2:]
    problemSizes, solutions = L.parseSolutionsData(
        data, str(written_solutions), assembler, False, False, False, isa_info_map
    )
    assert len(solutions) == n_solutions


def test_parse_solutions_data_bias_only_no_solutions_crash_site(
    assembler, isa_info_map
):
    """With a BiasTypeArgs header and no solution entries
    (len(data) == solutionStartIdxInData == 3), the ActivationArgs guard must
    short-circuit on ``len(data) > solutionStartIdxInData`` rather than index
    past the end. The original then crashes at ``solutions[0]`` on the empty
    solutions list; a ``>=`` mutant crashes while indexing
    data[solutionStartIdxInData] inside the guard, so the crashing statement
    differs even though both raise IndexError."""
    data = [
        {"MinimumRequiredVersion": "5.0.0"},
        {"ProblemSizes": []},
        {"BiasTypeArgs": [0]},
    ]
    with pytest.raises(IndexError) as ei:
        L.parseSolutionsData(
            data, "biasonly.yaml", assembler, False, False, False, isa_info_map
        )
    assert "solutions[0]" in str(ei.traceback[-1].statement)


def test_parse_solutions_file_rejection_reason_forwarded(
    written_solutions, assembler, isa_info_map, tmp_path
):
    """parseSolutionsFile must forward printSolutionRejectionReason (mutmut_5
    replaces it with None). With a forced General-Batched reject and a present
    SolutionIndex, a truthy flag makes reject() raise; None silences it."""
    data = L.read(str(written_solutions))
    for entry in data[2:]:
        entry["SolutionIndex"] = 0
        entry["ProblemType"]["Batched"] = False
        entry["ProblemType"]["StridedBatched"] = False
    p = tmp_path / "sol_reject.yaml"
    L.writeYAML(str(p), data)
    with pytest.raises(Exception, match="Any rejection of a LibraryLogic is not expected"):
        L.parseSolutionsFile(str(p), assembler, False, True, False, isa_info_map)


def test_parse_solutions_file_index_assignment_forwarded(
    written_solutions, assembler, isa_info_map, capsys
):
    """parseSolutionsFile must forward printIndexAssignmentInfo (mutmut_6
    replaces it with None). A truthy flag prints the index-assignment block
    from ProblemType.assignDerivedParameters; None prints nothing."""
    L.parseSolutionsFile(str(written_solutions), assembler, False, False, True, isa_info_map)
    out = capsys.readouterr().out
    assert "IndicesFree:" in out
    assert "IndexAssignmentsA:" in out


def test_parse_library_logic_data_datatypea_else_targets_exact_key(assembler, isa_info_map):
    """The DataTypeA else-branch assignment must write the exact 'DataTypeA' key.
    A mutated assignment target leaves the DataTypeA value unchanged (identity)
    but adds a stray key to the shared problemType, observable via key presence."""
    data = _raw_dict()
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    pt = logic.problemType
    assert "DataTypeA" in pt
    for stray in ("XXDataTypeAXX", "datatypea", "DATATYPEA"):
        assert stray not in pt


def test_parse_library_logic_data_cucount_preserved(assembler, isa_info_map):
    data = _raw_dict()
    data["CUCount"] = 110
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert len(logic.solutions) == 1
    assert all(s.get("CUCount") == 110 for s in logic.solutions)


def test_parse_library_logic_data_too_short_list_threads_srcfile(assembler, isa_info_map, capsys):
    with pytest.raises(SystemExit):
        L.parseLibraryLogicData(
            [{"MinimumRequiredVersion": "5.0.0"}],
            "SENTINEL_PATH.yaml", assembler, False, False, False, isa_info_map, False,
        )
    assert "SENTINEL_PATH.yaml" in capsys.readouterr().out


def test_parse_library_logic_data_datatype_if_branch_defaults(assembler, isa_info_map):
    data = _raw_dict()
    assert "DataTypeA" not in data["ProblemType"]
    assert "DataTypeB" not in data["ProblemType"]
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert "DataTypeA" in data["ProblemType"]
    assert data["ProblemType"]["DataTypeA"] == data["ProblemType"]["MacDataTypeA"]
    assert "DataTypeB" in data["ProblemType"]
    assert data["ProblemType"]["DataTypeB"] == data["ProblemType"]["MacDataTypeB"]


def test_parse_library_logic_data_datatype_else_branch_normalizes(assembler, isa_info_map):
    data = _raw_dict()
    data["ProblemType"]["MacDataTypeA"] = 4
    data["ProblemType"]["MacDataTypeB"] = 4
    data["ProblemType"]["DataTypeA"] = 0
    data["ProblemType"]["DataTypeB"] = 0
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert data["ProblemType"]["DataTypeA"] == 0
    assert data["ProblemType"]["DataTypeB"] == 0


def test_parse_library_logic_data_datatype_else_no_junk_keys(assembler, isa_info_map):
    data = _raw_dict()
    data["ProblemType"]["MacDataTypeA"] = 4
    data["ProblemType"]["MacDataTypeB"] = 4
    data["ProblemType"]["DataTypeA"] = 4
    data["ProblemType"]["DataTypeB"] = 4
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert "XXDataTypeAXX" not in data["ProblemType"]
    assert "datatypea" not in data["ProblemType"]
    assert "DATATYPEA" not in data["ProblemType"]
    assert data["ProblemType"]["DataTypeA"] == 4


def test_parse_library_logic_data_type_mismatch_threaded(assembler, isa_info_map):
    data = _raw_dict()
    data["ProblemType"]["UseBias"] = True
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert ("UseBias", "bool", "int") in logic.typeMismatches
    L.resetTypeMismatchCollector()


def test_parse_library_logic_data_isa_from_arch(assembler, isa_info_map):
    """The Assembly branch overwrites a solution's ISA with gfxToIsa(ArchitectureName)."""
    data = _raw_dict()
    data["Solutions"][0]["ISA"] = [9, 0, 10]
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert str(logic.solutions[0]["ISA"]) == "SemanticVersion(major=9, minor=4, patch=2)"


def test_parse_library_logic_data_cucount_from_data(assembler, isa_info_map):
    """Each solution's CUCount is copied from the top-level data[\"CUCount\"]."""
    data = _raw_dict()
    data["CUCount"] = 110
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert logic.solutions[0]["CUCount"] == 110


def test_parse_library_logic_data_custom_kernel_default_isp(assembler, isa_info_map, monkeypatch):
    """With no InternalSupportParams, getCustomKernelConfig receives the kernel name and an empty isp dict."""
    seen = {}
    def fake(name, isp):
        seen["name"] = name
        seen["isp"] = isp
        return {}
    monkeypatch.setattr(L, "getCustomKernelConfig", fake)
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_kernel"
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert seen["name"] == "synthetic_kernel"
    assert seen["isp"] == {}


def test_parse_library_logic_data_custom_kernel_isp_extracted(assembler, isa_info_map, monkeypatch):
    """When InternalSupportParams is present it is passed through to getCustomKernelConfig."""
    seen = {}
    def fake(name, isp):
        seen["name"] = name
        seen["isp"] = isp
        return {}
    monkeypatch.setattr(L, "getCustomKernelConfig", fake)
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_kernel"
    data["Solutions"][0]["InternalSupportParams"] = {"KernelLanguage": "Assembly"}
    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert seen["isp"] == {"KernelLanguage": "Assembly"}


def test_parse_library_logic_data_custom_kernel_config_merged(assembler, isa_info_map, monkeypatch):
    """Keys from the custom-kernel config are merged into the solution state."""
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp: {"CUCount": 999})
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_kernel"
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert logic.solutions[0]["CUCount"] == 999


def test_parse_library_logic_data_custom_kernel_mi_len4_ok(assembler, isa_info_map, monkeypatch):
    """A custom-kernel MatrixInstruction of length exactly 4 must not raise."""
    monkeypatch.setattr(
        L, "getCustomKernelConfig", lambda name, isp: {"MatrixInstruction": [16, 16, 16, 1]}
    )
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_kernel"
    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    assert len(logic.solutions) == 1
