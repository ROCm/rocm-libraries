################################################################################
# Characterization tests for Tensile.TensileLibLogicToYaml
#
# ADD-ONLY: pins the library-logic -> benchmark-config YAML transformers and the
# TensileLibLogicToYaml orchestrator (LibraryIO read/parse stubbed).
################################################################################
import importlib

import pytest
import yaml

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileLibLogicToYaml")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def test_make_flow():
    assert isinstance(M.makeFlow([1, 2]), M.FlowList)
    assert M.makeFlow(5) == 5


def test_tprint_gated(monkeypatch, capsys):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    M.tPrint(1, "hidden")
    assert capsys.readouterr().out == ""
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 2)
    M.tPrint(1, "shown")
    assert "shown" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# setGlobalParams
# ---------------------------------------------------------------------------
def test_set_global_params_non_i8():
    res = M.setGlobalParams({"MinimumRequiredVersion": "1.2.3"}, {"DataType": "S"})
    assert res["MinimumRequiredVersion"] == "1.2.3"
    assert res["DataInitTypeA"] == 12
    assert res["NumElementsToValidate"] == 0


def test_set_global_params_i8():
    res = M.setGlobalParams({"MinimumRequiredVersion": "1"}, {"DataType": "I8"})
    assert res["DataInitTypeA"] == 3
    assert res["DataInitTypeB"] == 3


# ---------------------------------------------------------------------------
# formProblemTypeYamlData
# ---------------------------------------------------------------------------
def test_form_problem_type_empty_raises():
    with pytest.raises(RuntimeError, match="empty"):
        M.formProblemTypeYamlData({})


def test_form_problem_type_always_and_default_keys():
    state = {
        "OperationType": "GEMM",
        "DataType": "S",
        "TransposeA": True,
        "TransposeB": False,
        "HighPrecisionAccumulate": True,
    }
    # add a default-valued key (skipped) — pick one that is NOT in the
    # always-print set nor OperationType
    always = {"OperationType", "DataType", "DestDataType", "ComputeDataType",
              "HighPrecisionAccumulate", "TransposeA", "TransposeB"}
    key = next(k for k in M.defaultProblemType if k not in always)
    state[key] = M.defaultProblemType[key]  # equals default -> skipped
    data = M.formProblemTypeYamlData(state)
    assert data["OperationType"] == "GEMM"
    assert data["TransposeA"] is True
    assert key not in data  # default value omitted


# ---------------------------------------------------------------------------
# formGroups / form9BitMIInst
# ---------------------------------------------------------------------------
def test_form_groups():
    data = M.formGroups({"a": 1, "b": 2})
    assert data["Groups"][0][0] == {"a": 1, "b": 2}


def test_form_9bit_mi_inst():
    sol = {
        "MIBlock": [1, 2, 3, 4, 5, 6],
        "MIWaveTile": [7, 8],
        "MIWaveGroup": [9, 10],
        "WorkGroup": [16, 16, 1],
        "MIArchVgpr": True,
    }
    groups = M.form9BitMIInst(sol)
    # MIBlock[0:5] + MIWaveTile + MIWaveGroup = 5+2+2 = 9 bits
    assert list(groups["MatrixInstruction"]) == [1, 2, 3, 4, 5, 7, 8, 9, 10]
    assert isinstance(groups["WorkGroup"], M.FlowList)
    assert groups["MIArchVgpr"] is True


def test_form_9bit_mi_inst_empty_raises():
    with pytest.raises(RuntimeError, match="cannot be empty"):
        M.form9BitMIInst({"MIBlock": [], "MIWaveTile": [], "MIWaveGroup": []})


# ---------------------------------------------------------------------------
# formForkParams
# ---------------------------------------------------------------------------
def test_form_fork_params_skip_mi_raises():
    # LATENT BUG (pinned): when MI is skipped the code sets temp="None" (a str)
    # then calls formGroups(temp), which does temp.items() -> AttributeError.
    # So the skipMI / MI-disabled path is currently broken. See DECISIONS D14.
    sol = {"EnableMatrixInstruction": False}
    with pytest.raises(AttributeError):
        M.formForkParams(sol, skipMI=True)


def test_form_fork_params_with_mi():
    sol = {
        "EnableMatrixInstruction": True,
        "MatrixInstruction": [16, 16, 4, 1],
        "MIBlock": [1, 2, 3, 4, 5, 6],
        "MIWaveTile": [7, 8],
        "MIWaveGroup": [9, 10],
        "WorkGroup": [16, 16, 1],
        "MIArchVgpr": False,
    }
    data = M.formForkParams(sol, skipMI=False)
    grp = data["ForkParameters"][-1]["Groups"][0][0]
    assert "MatrixInstruction" in grp


def test_form_fork_params_includes_nondefault_fork_key(monkeypatch):
    # craft a fork key present in defaultBenchmarkCommonParameters with a value
    # differing from its default so it is emitted. MI enabled + skipMI=False to
    # avoid the formGroups("None") bug (see D14).
    monkeypatch.setattr(M, "defaultBenchmarkCommonParameters", [{"GlobalSplitU": [[1]]}])
    sol = {
        "EnableMatrixInstruction": True,
        "MatrixInstruction": [16, 16, 4, 1],
        "MIBlock": [1, 2, 3, 4, 5, 6],
        "MIWaveTile": [7, 8],
        "MIWaveGroup": [9, 10],
        "WorkGroup": [16, 16, 1],
        "MIArchVgpr": False,
        "GlobalSplitU": 4,
    }
    data = M.formForkParams(sol, skipMI=False)
    fork_keys = [k for d in data["ForkParameters"] for k in d]
    assert "GlobalSplitU" in fork_keys


# ---------------------------------------------------------------------------
# formProblemSize
# ---------------------------------------------------------------------------
def test_form_problem_size_with_exact():
    data = M.formProblemSize([([128, 128, 1, 64], [0, 0.9])], 0, {"BiasDataTypeList": ["S"]})
    assert data["BenchmarkJoinParameters"] is None
    sizes = data["BenchmarkFinalParameters"][0]["ProblemSizes"]
    assert list(sizes[0]["Exact"]) == [128, 128, 1, 64]
    assert list(data["BenchmarkFinalParameters"][1]["BiasTypeArgs"]) == ["S"]


def test_form_problem_size_origami_none(capsys, monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 1)
    data = M.formProblemSize(None, 0, {"BiasDataTypeList": []})
    sizes = data["BenchmarkFinalParameters"][0]["ProblemSizes"]
    assert list(sizes[0]["Exact"]) == [1, 1, 1, 1]
    assert "Origami" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# formLibraryLogic
# ---------------------------------------------------------------------------
def test_form_library_logic():
    data = M.formLibraryLogic("sched", ["Device 75a0"], "gfx942")
    assert str(data["ScheduleName"]) == "sched"
    assert str(data["ArchitectureName"]) == "gfx942"
    assert [str(x) for x in data["DeviceNames"]] == ["Device 75a0"]


# ---------------------------------------------------------------------------
# writeToTensileYamlFile
# ---------------------------------------------------------------------------
def test_write_yaml_file_ok(tmp_path, monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    out = tmp_path / "sub" / "config.yaml"
    ret = M.writeToTensileYamlFile(str(out), {"a": 1})
    assert ret == str(out)
    assert yaml.safe_load(out.read_text()) == {"a": 1}


def test_write_yaml_file_error_returns_none(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    # empty filename -> open("") raises FileNotFoundError (OSError) -> None
    assert M.writeToTensileYamlFile("", {"a": 1}) is None


# ---------------------------------------------------------------------------
# TensileLibLogicToYaml orchestrator (LibraryIO stubbed)
# ---------------------------------------------------------------------------
def _fields():
    versionString = {"MinimumRequiredVersion": "1.0"}
    scheduleName = "sched"
    architectureName = "gfx942"
    deviceNames = ["Device 75a0"]
    problemTypeState = {
        "OperationType": "GEMM",
        "DataType": "S",
        "TransposeA": True,
        "TransposeB": False,
        "HighPrecisionAccumulate": True,
        "BiasDataTypeList": ["S"],
    }
    allSolutionStates = [{
        "EnableMatrixInstruction": True,
        "MatrixInstruction": [16, 16, 4, 1],
        "MIBlock": [1, 2, 3, 4, 5, 6],
        "MIWaveTile": [7, 8],
        "MIWaveGroup": [9, 10],
        "WorkGroup": [16, 16, 1],
        "MIArchVgpr": False,
    }]
    indexOrder = None
    exactLogic = [([128, 128, 1, 64], [0, 0.9])]
    rangeLogic = None
    otherFields = None
    return (
        versionString, scheduleName, architectureName, deviceNames, problemTypeState,
        allSolutionStates, indexOrder, exactLogic, rangeLogic, otherFields,
    )


def test_orchestrator_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    monkeypatch.setattr(M.LibraryIO, "readYAML", lambda p: {"nonempty": 1})
    monkeypatch.setattr(M.LibraryIO, "rawLibraryLogic", lambda y: _fields())
    out = tmp_path / "config.yaml"
    ret = M.TensileLibLogicToYaml("logic.yaml", 0, str(out), skipMI=False)
    assert ret == str(out)
    loaded = yaml.safe_load(out.read_text())
    assert "GlobalParameters" in loaded
    assert "BenchmarkProblems" in loaded
    assert loaded["LibraryLogic"]["ArchitectureName"] == "gfx942"


def test_orchestrator_empty_yaml_raises(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    monkeypatch.setattr(M.LibraryIO, "readYAML", lambda p: "")
    with pytest.raises(RuntimeError, match="empty"):
        M.TensileLibLogicToYaml("logic.yaml", 0, "out.yaml", skipMI=True)


# ---------------------------------------------------------------------------
# parseArgs / main
# ---------------------------------------------------------------------------
def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        M.sys, "argv",
        ["prog", "-i", "in.yaml", "-d", "0,3", "-o", "out.yaml", "-s"],
    )
    args = M.parseArgs()
    assert args.indices == "0,3"
    assert args.skipMI is True
    assert args.input.endswith("in.yaml")


def test_main_single_index(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    monkeypatch.setattr(M.sys, "argv", ["prog", "-i", "in.yaml", "-d", "0", "-o", "out.yaml"])
    calls = []
    monkeypatch.setattr(M, "TensileLibLogicToYaml", lambda inp, idx, out, skip: calls.append((idx, out)))
    M.main()
    assert len(calls) == 1
    assert calls[0][0] == 0


def test_main_multi_index_suffixes(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 0)
    monkeypatch.setattr(M.sys, "argv", ["prog", "-i", "in.yaml", "-d", "1,2", "-o", "/tmp/out.yaml"])
    calls = []
    monkeypatch.setattr(M, "TensileLibLogicToYaml", lambda inp, idx, out, skip: calls.append((idx, out)))
    M.main()
    assert [c[0] for c in calls] == [1, 2]
    # multi-id appends _<id> before .yaml
    assert calls[0][1].endswith("_1.yaml")
    assert calls[1][1].endswith("_2.yaml")
