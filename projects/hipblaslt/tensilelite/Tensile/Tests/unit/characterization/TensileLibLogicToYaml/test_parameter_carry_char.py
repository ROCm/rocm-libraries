################################################################################
# Characterization tests for how Tensile.TensileLibLogicToYaml carries a
# solution's parameters into the config it emits.
#
# The emitter keeps no per-parameter list: what a config can set comes from
# validParameters, values are judged by Tensile's own validator, and the
# problem type is reduced by rebuilding it. These tests pin that contract,
# including a round trip through the config-driven solution path. See
# DECISIONS D47 / ADR 0031.
################################################################################
import copy
import importlib
import os

import pytest
import yaml

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileLibLogicToYaml")
LibraryIO = importlib.import_module("Tensile.LibraryIO")

from Tensile.Common.GlobalParameters import defaultInternalSupportParams, defaultSolution
from Tensile.Common.ValidParameters import validParameters, validParametersForArch

_CHAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CODEGEN_DATA = os.path.join(_CHAR_DIR, "_codegen", "data")

#: Dict-format gfx950 logic whose DefaultSolution holds MaxOccupancy 40 and
#: LdsPadMetadata 0, both different from today's defaults.
DICT_LOGIC = os.path.join(_DATA_DIR, "gfx950_HSS_Bias_AH_dict.yaml")
#: Records PrefetchGlobalReadA/B and TDMFuse, which have no registry default.
DECOUPLE_PGR_LOGIC = os.path.join(_CODEGEN_DATA, "gfx1250", "F4_MX_DecouplePGR_TDMFuse.yaml")
#: Sets UseBias without listing BiasDataTypeList.
BIAS_DEFAULT_LOGIC = os.path.join(_CODEGEN_DATA, "gfx90a", "HSS.yaml")
BIAS_ACT_LOGIC = os.path.join(_CODEGEN_DATA, "gfx942", "BBS_BH_Bias_Act.yaml")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 1)


def _load(path):
    return LibraryIO.readYAML(path)


def _problemType(path):
    data = _load(path)
    return data["ProblemType"] if isinstance(data, dict) else data[4]


def _miSolution(**overrides):
    """A solution as a reader returns it: every default filled, MI enabled."""
    sol = dict(defaultSolution)
    sol.update(
        {
            "EnableMatrixInstruction": True,
            "MatrixInstruction": [16, 16, 4, 1],
            "MIBlock": [16, 16, 4, 1, 1, 1],
            "MIWaveTile": [2, 2],
            "MIWaveGroup": [2, 2],
            "WorkGroup": [32, 8, 1],
            "MIArchVgpr": False,
        }
    )
    sol.update(overrides)
    return sol


def _forkKeys(data):
    return {k: v for d in data["ForkParameters"] for k, v in d.items() if k != "Groups"}


def _firstKey(predicate):
    return next(k for k, v in validParameters.items() if isinstance(v, list) and v and predicate(v))


# ---------------------------------------------------------------------------
# coerceParameterValue / isDefaultParameter / parameterRejection
# ---------------------------------------------------------------------------
def test_coerce_int_to_bool_for_bool_parameter():
    key = _firstKey(lambda v: all(type(x) is bool for x in v))
    assert M.coerceParameterValue(key, 1) is True
    assert M.coerceParameterValue(key, 0) is False
    # Not lossless: left for the validator to reject.
    assert M.coerceParameterValue(key, 2) == 2


def test_coerce_bool_to_int_for_int_parameter():
    key = _firstKey(lambda v: all(type(x) is int for x in v))
    assert M.coerceParameterValue(key, True) == 1
    assert type(M.coerceParameterValue(key, True)) is int


def test_coerce_int_to_float_for_float_parameter():
    key = _firstKey(lambda v: all(type(x) is float for x in v))
    assert M.coerceParameterValue(key, 1) == 1.0
    assert type(M.coerceParameterValue(key, 1)) is float


def test_coerce_leaves_matching_sentinel_and_unknown_values():
    key = _firstKey(lambda v: all(type(x) is int for x in v))
    assert M.coerceParameterValue(key, 3) == 3
    assert M.coerceParameterValue("CustomKernel", {"name": "k"}) == {"name": "k"}
    assert M.coerceParameterValue("NotAParameter", 1) == 1


def test_coerce_leaves_unconvertible_type_to_the_validator():
    key = _firstKey(lambda v: all(type(x) is int for x in v))
    assert M.coerceParameterValue(key, "text") == "text"


def test_is_default_parameter_compares_value_and_type():
    key = next(k for k, v in defaultSolution.items() if type(v) is bool)
    assert M.isDefaultParameter(key, defaultSolution[key]) is True
    assert M.isDefaultParameter(key, int(defaultSolution[key])) is False
    assert M.isDefaultParameter("PrefetchGlobalReadA", 1) is False


def test_parameter_rejection_uses_tensile_validator():
    assert M.parameterRejection("StoreVectorWidth", 4, validParameters) is None
    reason = M.parameterRejection("StoreVectorWidth", 999, validParameters)
    assert reason.startswith("Invalid parameter value: StoreVectorWidth = 999")
    assert "Invalid parameter name" in M.parameterRejection("NotAParameter", 1, validParameters)


# ---------------------------------------------------------------------------
# handwrittenCustomKernelName
# ---------------------------------------------------------------------------
def test_generated_custom_kernel_stamp_is_not_handwritten():
    stamp = {"CustomKernel": {"name": "Cijk_Ailk_Bljk_S_MT64x64x16", "generated": True}}
    assert M.handwrittenCustomKernelName(stamp) is None


def test_handwritten_custom_kernel_named_by_its_block():
    sol = {"CustomKernel": {"name": "Custom_Kernel_A"}, "CustomKernelName": "ignored"}
    assert M.handwrittenCustomKernelName(sol) == "Custom_Kernel_A"


def test_handwritten_custom_kernel_named_by_legacy_field():
    assert M.handwrittenCustomKernelName({"CustomKernelName": "Custom_Legacy"}) == "Custom_Legacy"
    assert M.handwrittenCustomKernelName({"CustomKernelName": ""}) is None
    assert M.handwrittenCustomKernelName({}) is None


# ---------------------------------------------------------------------------
# formForkParams
# ---------------------------------------------------------------------------
def test_fork_carries_settable_parameters_without_registry_default():
    for key in ("PrefetchGlobalReadA", "PrefetchGlobalReadB", "TDMFuse"):
        assert key in validParameters and key not in defaultSolution
    sol = _miSolution(PrefetchGlobalReadA=2, PrefetchGlobalReadB=1, TDMFuse=1, TDMInst=3)
    forks = _forkKeys(M.formForkParams(sol, skipMI=False, architectureName="gfx1250"))
    assert list(forks["PrefetchGlobalReadA"]) == [2]
    assert list(forks["PrefetchGlobalReadB"]) == [1]
    assert list(forks["TDMFuse"]) == [1]


def test_fork_omits_defaults_derived_state_and_keys_emitted_elsewhere():
    sol = _miSolution(
        MacroTile0=256,
        LdsNumBytes=4096,
        ISA=[9, 4, 2],
        CustomKernel={"name": "Cijk_stamp", "generated": True},
    )
    data = M.formForkParams(sol, skipMI=False)
    forks = _forkKeys(data)
    for key in ("MacroTile0", "LdsNumBytes", "ISA", "CustomKernel", "MatrixInstruction", "WorkGroup"):
        assert key not in forks
    assert not any(key in forks for key in defaultSolution if sol[key] == defaultSolution[key])
    assert "CustomKernels" not in data


def test_fork_emits_non_default_value_as_single_candidate():
    value = next(v for v in validParameters["DepthU"] if v > 0 and v != defaultSolution["DepthU"])
    forks = _forkKeys(M.formForkParams(_miSolution(DepthU=value), skipMI=False))
    assert list(forks["DepthU"]) == [value]


def test_fork_coerces_recorded_bools_for_the_validator():
    key = next(
        k
        for k, v in defaultSolution.items()
        if type(v) is bool
        and isinstance(validParameters.get(k), list)
        and all(type(x) is bool for x in validParameters[k])
    )
    recorded = int(not defaultSolution[key])
    forks = _forkKeys(M.formForkParams(_miSolution(**{key: recorded}), skipMI=False))
    assert list(forks[key]) == [bool(recorded)]


def test_fork_omits_value_the_validator_rejects(capsys):
    forks = _forkKeys(M.formForkParams(_miSolution(StoreVectorWidth=999), skipMI=False))
    assert "StoreVectorWidth" not in forks
    out = capsys.readouterr().out
    assert "rejects them" in out and "StoreVectorWidth=999" in out


def test_fork_validates_against_the_target_architecture(capsys):
    wide = validParametersForArch("gfx1250")["LdsPadA"]
    extra = [v for v in wide if v not in validParameters["LdsPadA"]]
    if not extra:
        pytest.skip("no LdsPadA value is gfx1250-only")
    sol = _miSolution(LdsPadA=extra[0])
    assert list(_forkKeys(M.formForkParams(sol, False, "gfx1250"))["LdsPadA"]) == [extra[0]]
    assert "LdsPadA" not in _forkKeys(M.formForkParams(sol, False, "gfx942"))
    assert "LdsPadA={}".format(extra[0]) in capsys.readouterr().out


def test_fork_groups_coerce_mi_arch_vgpr():
    data = M.formForkParams(_miSolution(MIArchVgpr=1), skipMI=False)
    group = data["ForkParameters"][-1]["Groups"][0][0]
    assert group["MIArchVgpr"] is True
    assert "MIArchVgpr" not in _forkKeys(data)


def test_fork_handwritten_custom_kernel_becomes_custom_kernels_entry():
    support = {"KernArgsVersion": 2, "SupportUserGSU": False}
    sol = _miSolution(CustomKernelName="Custom_Handwritten", InternalSupportParams=support)
    data = M.formForkParams(sol, skipMI=False)
    assert data["ForkParameters"] is None
    assert data["CustomKernels"] == ["Custom_Handwritten"]
    assert data["InternalSupportParams"] == support


def test_fork_handwritten_custom_kernel_without_support_params():
    data = M.formForkParams({"CustomKernel": {"name": "Custom_B"}}, skipMI=False)
    assert data["CustomKernels"] == ["Custom_B"]
    assert "InternalSupportParams" not in data


def test_fork_warns_when_recorded_support_params_cannot_be_carried(capsys):
    sol = _miSolution(InternalSupportParams={**defaultInternalSupportParams, "KernArgsVersion": 2})
    M.formForkParams(sol, skipMI=False)
    out = capsys.readouterr().out
    assert "InternalSupportParams" in out and "KernArgsVersion=2" in out


def test_fork_quiet_when_support_params_are_the_defaults(capsys):
    M.formForkParams(_miSolution(InternalSupportParams=dict(defaultInternalSupportParams)), skipMI=False)
    assert "InternalSupportParams" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Problem type: built, not compared against a registry
# ---------------------------------------------------------------------------
def test_problem_type_config_keys_cover_registry_and_config_reads():
    keys = M.problemTypeConfigKeys()
    assert set(M.defaultProblemType) <= keys
    assert {"ActivationType", "UseBias", "BiasDataTypeList", "GateResidualDataTypeList"} <= keys


def test_coerce_problem_type_value():
    assert M.coerceProblemTypeValue("TransposeA", 1) is True
    assert M.coerceProblemTypeValue("DataType", 4) == 4


@pytest.mark.parametrize(
    "path", [DICT_LOGIC, DECOUPLE_PGR_LOGIC, BIAS_DEFAULT_LOGIC, BIAS_ACT_LOGIC], ids=os.path.basename
)
def test_problem_type_block_rebuilds_the_logic_problem_type(path):
    block = _problemType(path)
    emitted = M.formProblemTypeYamlData(copy.deepcopy(block))
    config = {k: (list(v) if isinstance(v, list) else v) for k, v in emitted.items()}
    rebuilt = M.buildProblemTypeState(config, strict=True)
    assert rebuilt is not None
    assert rebuilt == M.buildProblemTypeState(M.normalizedLogicProblemType(block), strict=False)


def test_problem_type_keeps_activation_and_drops_derived_keys():
    emitted = M.formProblemTypeYamlData(_problemType(DICT_LOGIC))
    assert emitted["ActivationType"] == "hipblaslt_all"
    assert "Activation" in emitted
    for derived in ("IndexAssignmentsA", "IndexAssignmentsB", "NumIndicesC", "TotalIndices"):
        assert derived not in emitted
    assert list(emitted)[0] == "OperationType"
    for key in M.PROBLEM_TYPE_ALWAYS_EMITTED:
        assert key in emitted


def test_problem_type_unbuildable_block_is_emitted_unreduced(capsys):
    emitted = M.formProblemTypeYamlData({"OperationType": "GEMM", "Batched": True})
    assert emitted == {"OperationType": "GEMM", "Batched": True}
    assert "cannot be rebuilt" in capsys.readouterr().out


def test_build_problem_type_state_reports_rejection_as_none():
    assert M.buildProblemTypeState({"OperationType": "GEMM"}, strict=True) is None


def test_effective_data_types():
    block = _problemType(BIAS_DEFAULT_LOGIC)
    assert "UseBias" in block and "BiasDataTypeList" not in block
    derived = M.effectiveDataTypes(block, "BiasDataTypeList")
    assert derived and all(type(value) is int for value in derived)
    assert M.effectiveDataTypes({"BiasDataTypeList": [0, 7]}, "BiasDataTypeList") == [0, 7]
    assert M.effectiveDataTypes({}, "BiasDataTypeList") == []


# ---------------------------------------------------------------------------
# formProblemSize
# ---------------------------------------------------------------------------
def test_problem_size_without_matching_exact_uses_placeholder(capsys):
    data = M.formProblemSize([([128, 128, 1, 64], [3, 0.9])], 0, {"BiasDataTypeList": [0]})
    assert list(data["BenchmarkFinalParameters"][0]["ProblemSizes"][0]["Exact"]) == [1, 1, 1, 1]
    assert "maps no size to solution 0" in capsys.readouterr().out


def test_problem_size_uses_derived_bias_types():
    block = _problemType(BIAS_DEFAULT_LOGIC)
    data = M.formProblemSize(None, 0, block, problemSizes=[{"Exact": [64, 64, 1, 64]}])
    assert list(data["BenchmarkFinalParameters"][1]["BiasTypeArgs"]) == M.effectiveDataTypes(
        block, "BiasDataTypeList"
    )


def test_problem_size_carries_gate_types():
    data = M.formProblemSize(None, 0, {"GateResidualDataTypeList": [4]}, problemSizes=[])
    assert list(data["BenchmarkFinalParameters"][1]["GateTypeArgs"]) == [4]


# ---------------------------------------------------------------------------
# Readers restore the solution the way Tensile reads it
# ---------------------------------------------------------------------------
def test_library_reader_fills_from_the_file_default_solution():
    data = _load(DICT_LOGIC)
    assert "MIArchVgpr" not in data["Solutions"][0]
    assert data["DefaultSolution"]["MaxOccupancy"] != defaultSolution["MaxOccupancy"]
    source = M.LibraryLogicReader.read(data, 0)
    assert source.solution["MIArchVgpr"] == data["DefaultSolution"]["MIArchVgpr"]
    assert source.solution["MaxOccupancy"] == data["DefaultSolution"]["MaxOccupancy"]
    for key in defaultSolution:
        assert key in source.solution
    # The parsed input is not modified.
    assert "MIArchVgpr" not in data["Solutions"][0]


def test_library_reader_keeps_the_solution_value_over_defaults():
    data = _load(DICT_LOGIC)
    data["Solutions"][0]["MaxOccupancy"] = 32
    assert M.LibraryLogicReader.read(data, 0).solution["MaxOccupancy"] == 32


def test_library_reader_default_solution_lookup():
    assert M.LibraryLogicReader._defaultSolution(None) is None
    assert M.LibraryLogicReader._defaultSolution([None, None, None]) is None
    assert M.LibraryLogicReader._defaultSolution([None, None, None, "x"]) is None
    assert M.LibraryLogicReader._defaultSolution([None, None, None, {"A": 1}]) == {"A": 1}


def test_library_reader_rejects_missing_solution():
    data = _load(DICT_LOGIC)
    with pytest.raises(RuntimeError, match="solution index:5"):
        M.LibraryLogicReader.read(data, 5)
    data["Solutions"][0] = ""
    with pytest.raises(RuntimeError, match="solution index:0"):
        M.LibraryLogicReader.read(data, 0)


def _benchmarkData(**solution):
    state = {"ProblemType": {"DataType": "S"}, "ISA": [9, 4, 2], "SolutionIndex": 7}
    state.update(solution)
    return [{"MinimumRequiredVersion": "5.0.0"}, {"ProblemSizes": [{"Exact": [8, 8, 1, 8]}]}, state]


def test_benchmark_reader_fills_defaults_and_selects_by_index():
    data = _benchmarkData()
    source = M.BenchmarkDataReader.read(data, 7)
    assert source.architectureName == "gfx942"
    assert all(key in source.solution for key in defaultSolution)
    assert "MaxOccupancy" not in data[2]
    assert source.exactLogic == [[[8, 8, 1, 8], [7, 0.0]]]


def test_benchmark_reader_selects_by_position_and_rejects_out_of_range():
    data = _benchmarkData(SolutionIndex=9)
    assert M.BenchmarkDataReader.read(data, 0).solution["SolutionIndex"] == 9
    with pytest.raises(RuntimeError, match="holds 1 solutions"):
        M.BenchmarkDataReader.read(data, 4)


def test_benchmark_reader_requires_problem_type_and_isa():
    with pytest.raises(RuntimeError, match="no ProblemType"):
        M.BenchmarkDataReader.read(_benchmarkData(ProblemType={}), 7)
    with pytest.raises(RuntimeError, match="no ISA"):
        M.BenchmarkDataReader.read(_benchmarkData(ISA=None), 7)


def test_benchmark_reader_does_not_match_other_shapes():
    assert M.BenchmarkDataReader.matches({"Solutions": []}) is False
    assert M.BenchmarkDataReader.matches([{"MinimumRequiredVersion": "5"}, {"ProblemSizes": []}, {}]) is False


def test_read_source_rejects_unknown_format(monkeypatch):
    class Never(M.SourceReader):
        @staticmethod
        def matches(data):
            return False

        @staticmethod
        def read(data, solutionIndex):  # pragma: no cover - never matched
            raise AssertionError

    monkeypatch.setattr(M, "SOURCE_READERS", (Never,))
    with pytest.raises(RuntimeError, match="Unrecognized input file format"):
        M.readSource({}, 0)


# ---------------------------------------------------------------------------
# Build target
# ---------------------------------------------------------------------------
def test_build_target_prefers_a_revision_schedule():
    assert M.buildTarget({"ScheduleName": "gfx1250v0", "ArchitectureName": "gfx1250"}) == "gfx1250v0"
    assert M.buildTarget({"ScheduleName": "aquavanjaram", "ArchitectureName": "gfx942"}) == "gfx942"


def test_add_build_target_names_isa_and_revision():
    params = {}
    M.addBuildTarget(params, "gfx942")
    assert [list(isa) for isa in params["ISA"]] == [[9, 4, 2]]
    assert "Architecture" not in params

    params = {}
    M.addBuildTarget(params, "gfx1250v0")
    assert [list(isa) for isa in params["ISA"]] == [[12, 5, 0]]
    assert params["Architecture"] == "gfx1250v0"


def test_add_build_target_keeps_run_settings():
    params = {"ISA": [[12, 5, 0]], "Architecture": "gfx1250"}
    M.addBuildTarget(params, "gfx1250v0")
    assert params == {"ISA": [[12, 5, 0]], "Architecture": "gfx1250"}

    # A revision of an ISA the run did not build would make Tensile raise.
    params = {"ISA": [[9, 5, 0]]}
    M.addBuildTarget(params, "gfx1250v0")
    assert params == {"ISA": [[9, 5, 0]]}


def test_add_build_target_ignores_unknown_architecture():
    params = {}
    M.addBuildTarget(params, "notanarch")
    assert params == {}


# ---------------------------------------------------------------------------
# End to end: the emitted config regenerates the same kernel
# ---------------------------------------------------------------------------
def test_emitted_config_carries_file_defaults_and_target(tmp_path):
    out = tmp_path / "config.yaml"
    M.TensileLibLogicToYaml(DICT_LOGIC, 0, str(out), False, useRunConfig=False)
    config = yaml.safe_load(out.read_text())
    assert config["GlobalParameters"]["ISA"] == [[9, 5, 0]]
    forks = {k: v for d in config["BenchmarkProblems"][0][1]["ForkParameters"] for k, v in d.items()}
    assert forks["MaxOccupancy"] == [40]
    assert forks["LdsPadMetadata"] == [0]
    assert config["BenchmarkProblems"][0][0]["ActivationType"] == "hipblaslt_all"


@pytest.mark.parametrize(
    "path, index, arch",
    [
        (DECOUPLE_PGR_LOGIC, 0, "gfx1250"),
        (DICT_LOGIC, 0, "gfx950"),
        (BIAS_DEFAULT_LOGIC, 0, "gfx90a"),
        (BIAS_ACT_LOGIC, 0, "gfx942"),
    ],
    ids=lambda p: os.path.basename(p) if isinstance(p, str) and p.endswith(".yaml") else None,
)
def test_emitted_config_regenerates_the_same_kernel(tmp_path, path, index, arch):
    import codegen_harness
    import config_harness
    from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin

    original = codegen_harness.solutions_from_logic(path)[index]
    out = tmp_path / "config.yaml"
    M.TensileLibLogicToYaml(path, index, str(out), False, useRunConfig=False)
    regenerated = config_harness.solutions_from_config(str(out), arch=arch)

    assert len(regenerated) == 1
    original.getKernels()
    regenerated[0].getKernels()
    assert getKernelNameMin(regenerated[0], False) == getKernelNameMin(original, False)
    assert getSolutionNameMin(regenerated[0], False) == getSolutionNameMin(original, False)
