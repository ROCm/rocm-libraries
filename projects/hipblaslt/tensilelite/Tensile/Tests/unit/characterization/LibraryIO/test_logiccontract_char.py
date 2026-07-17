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

"""Characterization tests for the library-logic (de)serialisation contract that
does NOT require live ``Solution`` construction:

  * ``parseLibraryLogicList`` — the matching-table -> dict normaliser, across
    FreeSize / Prediction / Matching library types, the dict-vs-str arch field,
    optional PerfMetric, and the two ``printExit`` guards (len<9, missing type).
  * ``rawLibraryLogic`` — positional unpack incl. trailing ``otherFields``.
  * ``createLibraryLogic`` — assembles the serialized tuple from a logicTuple;
    arch dict-vs-str branch (gfx942 + CUCount), exactLogic present/None,
    tileSelection on/off. Round-tripped back through ``parseLibraryLogicList``.
  * ``getCUCount`` — the ``CU`` env path and the ``rocminfo`` subprocess path
    (subprocess.run monkeypatched), plus the failure -> ``printExit``.

All version tokens normalised to ``<VERSION>``.
"""

import pytest

from Tensile import __version__
import Tensile.LibraryIO as L

pytestmark = pytest.mark.unit


def _norm(obj):
    """Recursively replace the embedded Tensile version with a stable token."""
    if isinstance(obj, str):
        return obj.replace(__version__, "<VERSION>")
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_norm(v) for v in obj)
    return obj


# ===========================================================================
# parseLibraryLogicList
# ===========================================================================

def _logic_list(arch="gfx942", libraryType="Matching", *, extra=None, trailing=None):
    """Build a serialized matching-table logic list (data[0..]) with the
    minimum required 9 fields plus optional PerfMetric(10)/LibraryType(11)."""
    data = [
        {"MinimumRequiredVersion": "5.0.0"},  # 0
        "aquavanjaram",                        # 1  ScheduleName
        arch,                                  # 2  arch (str or dict)
        ["Device 0049"],                       # 3  DeviceNames
        {"OperationType": "GEMM"},             # 4  ProblemType
        [{"SolutionIndex": 0}, {"SolutionIndex": 1}],  # 5  Solutions
        [0, 1],                                # 6  indexOrder
        [[[1, 1, 1], [0, 1.0]]],               # 7  exactLogic / table
        None,                                  # 8  rangeLogic
        None,                                  # 9  (reserved)
        "perf-metric",                         # 10 PerfMetric
        libraryType,                           # 11 matching property / type
    ]
    if extra is not None:
        data[10] = extra
    if trailing is not None:
        data += trailing
    return data


def test_parse_logic_list_matching(snapshot):
    assert _norm(L.parseLibraryLogicList(_logic_list(libraryType="0.5"), "src.yaml")) == snapshot


def test_parse_logic_list_freesize(snapshot):
    assert _norm(L.parseLibraryLogicList(_logic_list(libraryType="FreeSize"), "src.yaml")) == snapshot


def test_parse_logic_list_prediction(snapshot):
    assert _norm(L.parseLibraryLogicList(_logic_list(libraryType="Prediction"), "src.yaml")) == snapshot


def test_parse_logic_list_arch_dict(snapshot):
    # data[2] as a dict -> ArchitectureName + CUCount read from it.
    d = _logic_list(arch={"Architecture": "gfx942", "CUCount": 228})
    assert _norm(L.parseLibraryLogicList(d, "src.yaml")) == snapshot


def test_parse_logic_list_no_perfmetric(snapshot):
    # data[10] falsy -> PerfMetric key absent.
    d = _logic_list()
    d[10] = None
    assert _norm(L.parseLibraryLogicList(d, "src.yaml")) == snapshot


def test_parse_logic_list_too_short():
    with pytest.raises(SystemExit):
        L.parseLibraryLogicList([{"MinimumRequiredVersion": "5.0.0"}], "src.yaml")


def test_parse_logic_list_missing_type():
    # data[11] absent/falsy -> missing matching property -> printExit.
    d = _logic_list()
    d[11] = None
    with pytest.raises(SystemExit):
        L.parseLibraryLogicList(d, "src.yaml")


# ===========================================================================
# rawLibraryLogic
# ===========================================================================

def test_raw_library_logic_minimal(snapshot):
    data = [
        "5.0.0", "sched", "gfx942", ["Device 0049"],
        {"OperationType": "GEMM"}, [{"SolutionIndex": 0}],
        [0], [["k", "v"]], None,
    ]
    assert _norm(L.rawLibraryLogic(data)) == snapshot


def test_raw_library_logic_with_other_fields(snapshot):
    data = [
        "5.0.0", "sched", "gfx942", ["Device 0049"],
        {"OperationType": "GEMM"}, [{"SolutionIndex": 0}],
        [0], [["k", "v"]], None,
        "perf", "Matching", {"extra": 1},  # data[9:] -> otherFields
    ]
    assert _norm(L.rawLibraryLogic(data)) == snapshot


# ===========================================================================
# createLibraryLogic — synthetic problemType/solutions, getCUCount controlled
# ===========================================================================

class _Enum:
    def __init__(self, value):
        self.value = value


def _problem_type():
    class _PT:
        state = {
            "OperationType": "GEMM",
            "DataType": _Enum(4),
            "MacDataTypeA": _Enum(4),
            "MacDataTypeB": _Enum(4),
            "DataTypeA": _Enum(4),
            "DataTypeB": _Enum(4),
            "DataTypeE": _Enum(0),
            "DataTypeAmaxD": _Enum(0),
            "DestDataType": _Enum(0),
            "ComputeDataType": _Enum(0),
            "BiasDataTypeList": [_Enum(0)],
            "ActivationComputeDataType": _Enum(0),
            "ActivationType": _Enum("none"),
            "F32XdlMathOp": _Enum(0),
        }
    return _PT()


class _FakeSolution:
    def __init__(self, index):
        self._index = index

    def getAttributes(self):
        return {
            "SolutionIndex": self._index,
            "ISA": (9, 4, 2, "truncated"),
            "ProblemType": "<deleted-by-createLibraryLogic>",
        }


def _logic_tuple(exactLogic, *, tile=False):
    base = [
        _problem_type(),                 # 0 problemType
        [_FakeSolution(0)],              # 1 solutions
        [0, 1],                          # 2 indexOrder
        exactLogic,                      # 3 exactLogic
        None,                            # 4 rangeLogic
    ]
    if tile:
        base += [
            [_FakeSolution(1)],          # 5 tileSelection solutions
            [3, 4],                      # 6 tileSelectionIndices
            "perf-metric",               # 7 PerfMetric
        ]
    else:
        base += [None, None, "perf-metric"]
    return base


def test_create_library_logic_str_arch(monkeypatch, snapshot):
    # CU=304 -> the gfx942-special branch is skipped, arch stays a plain string.
    monkeypatch.setattr(L, "getCUCount", lambda: 304)
    data = L.createLibraryLogic("aquavanjaram", "gfx942", ["Device 0049"], "Matching",
                                _logic_tuple({(1, 1, 1): [0, 1.0]}))
    assert _norm(data) == snapshot


def test_create_library_logic_dict_arch(monkeypatch, snapshot):
    # gfx942 + CUCount != 304 -> arch becomes {Architecture, CUCount}.
    monkeypatch.setattr(L, "getCUCount", lambda: 228)
    data = L.createLibraryLogic("aquavanjaram", "gfx942", ["Device 0049"], "Matching",
                                _logic_tuple({(1, 1, 1): [0, 1.0]}))
    assert _norm(data) == snapshot


def test_create_library_logic_no_exact(monkeypatch, snapshot):
    monkeypatch.setattr(L, "getCUCount", lambda: 304)
    data = L.createLibraryLogic("aquavanjaram", "gfx90a", ["Device 0049"], "Matching",
                                _logic_tuple(None))
    assert _norm(data) == snapshot


def test_create_library_logic_tile_selection(monkeypatch, snapshot):
    monkeypatch.setattr(L, "getCUCount", lambda: 304)
    data = L.createLibraryLogic("aquavanjaram", "gfx90a", ["Device 0049"], "Matching",
                                _logic_tuple({(1, 1, 1): [0, 1.0]}, tile=True))
    assert _norm(data) == snapshot


def test_create_library_logic_with_metadata(monkeypatch, snapshot):
    # ProblemType.state carrying the optional DataTypeMetadata / MXSA / MXSB
    # keys -> the three guarded conversion branches run.
    monkeypatch.setattr(L, "getCUCount", lambda: 304)
    pt = _problem_type()
    pt.state["DataTypeMetadata"] = _Enum(7)
    pt.state["DataTypeMXSA"] = _Enum(8)
    pt.state["DataTypeMXSB"] = _Enum(9)
    lt = _logic_tuple({(1, 1, 1): [0, 1.0]})
    lt[0] = pt
    data = L.createLibraryLogic("aquavanjaram", "gfx950", ["Device 0049"], "Matching", lt)
    assert _norm(data) == snapshot


def test_create_library_logic_roundtrip(monkeypatch, snapshot):
    # createLibraryLogic output is itself a valid matching-table list.
    monkeypatch.setattr(L, "getCUCount", lambda: 304)
    data = L.createLibraryLogic("aquavanjaram", "gfx942", ["Device 0049"], "Matching",
                                _logic_tuple({(1, 1, 1): [0, 1.0]}))
    assert _norm(L.parseLibraryLogicList(data, "roundtrip.yaml")) == snapshot


# ===========================================================================
# getCUCount
# ===========================================================================

def test_get_cu_count_env(monkeypatch):
    monkeypatch.setenv("CU", "304")
    assert L.getCUCount() == 304


def test_get_cu_count_rocminfo(monkeypatch):
    # CU unset -> rocminfo subprocess path; monkeypatch subprocess.run.
    monkeypatch.delenv("CU", raising=False)

    class _Res:
        stdout = b"Compute Unit:            110\n"

    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: _Res())
    assert L.getCUCount() == 110


def test_get_cu_count_rocminfo_no_match(monkeypatch):
    # rocminfo output without a "Compute Unit:" line -> regex misses ->
    # CU stays None -> printExit (covers the 696->701 no-match branch arm).
    monkeypatch.delenv("CU", raising=False)

    class _Res:
        stdout = b"some unrelated output\n"

    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: _Res())
    with pytest.raises(SystemExit):
        L.getCUCount()


def test_get_cu_count_failure(monkeypatch):
    # CU unset + subprocess raises -> exception swallowed -> printExit.
    monkeypatch.delenv("CU", raising=False)

    def _boom(*a, **k):
        raise OSError("no rocminfo")

    monkeypatch.setattr(L.subprocess, "run", _boom)
    with pytest.raises(SystemExit):
        L.getCUCount()
