################################################################################
# Characterization tests for how Tensile.TensileLibLogicToYaml recovers the
# settings of the run that produced its input: ClientParameters.ini from the
# benchmark build tree, then the tuning config beside the build directory.
################################################################################
import ast
import importlib
import os

import pytest
import yaml

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileLibLogicToYaml")
LibraryIO = importlib.import_module("Tensile.LibraryIO")
ClientWriter = importlib.import_module("Tensile.ClientWriter")

_CHAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Source of a realistic solution state for the benchmark data built below.
_LOGIC = os.path.join(_CHAR_DIR, "_codegen", "data", "gfx950", "HHS.yaml")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setitem(M.globalParameters, "ClientLogLevel", 1)


# ---------------------------------------------------------------------------
# ClientWriter scan
# ---------------------------------------------------------------------------
def test_client_parameter_map_reverses_client_writer():
    mapping, decoders = M.clientParameterMap()
    assert mapping["num-warmups"] == "NumWarmups"
    assert mapping["init-a"] == "DataInitTypeA"
    assert mapping["device-idx"] == "Device"
    assert decoders["init-a"] == "DataInitName"
    assert decoders["bounds-check"] == "boundsCheckName"
    for excluded in M.CLIENT_PARAMETER_EXCLUSIONS:
        assert excluded not in mapping


@pytest.fixture
def _syntheticClientWriter(monkeypatch):
    source = (
        "def writeClientConfigIni():\n"
        "    alias = globalParameters['AliasKey']\n"
        "    rebound = globalParameters['FirstKey']\n"
        "    rebound = somethingElse\n"
        "    param('alias-key', alias)\n"
        "    param('rebound-key', rebound)\n"
        "    param('two-keys', globalParameters['A'] or globalParameters['B'])\n"
        "    param('enum-key', DataInitName(globalParameters['DataInitTypeC']).name)\n"
        "    param('activation-additional-args', globalParameters['Excluded'])\n"
        "\n"
        "def dataInitParams():\n"
        "    return [('tuple-key', globalParameters['TupleKey'])]\n"
    )
    real = M.inspect.getsource
    monkeypatch.setattr(
        M.inspect, "getsource", lambda obj: source if obj is ClientWriter else real(obj)
    )
    M.clientParameterMap.cache_clear()
    yield
    M.clientParameterMap.cache_clear()


def test_client_parameter_map_resolves_only_unambiguous_settings(_syntheticClientWriter):
    mapping, decoders = M.clientParameterMap()
    assert mapping["alias-key"] == "AliasKey"
    assert mapping["tuple-key"] == "TupleKey"
    assert mapping["enum-key"] == "DataInitTypeC"
    assert decoders["enum-key"] == "DataInitName"
    assert "rebound-key" not in mapping
    assert "two-keys" not in mapping
    assert "activation-additional-args" not in mapping
    # Overrides are always applied.
    for iniKey, globalKey in M.CLIENT_PARAMETER_OVERRIDES.items():
        assert mapping[iniKey] == globalKey


def test_global_parameter_key():
    node = ast.parse("globalParameters['X']", mode="eval").body
    assert M._globalParameterKey(node) == "X"
    assert M._globalParameterKey(ast.parse("other['X']", mode="eval").body) is None
    assert M._globalParameterKey(ast.parse("globalParameters[1]", mode="eval").body) is None


def test_ini_key_value_pairs():
    function = ast.parse("def f():\n    param('a', 1)\n    param(k, 2)\n    x = ('b', 3)\n").body[0]
    pairs = [(key, ast.literal_eval(value)) for key, value in M._iniKeyValuePairs(function)]
    assert pairs == [("a", 1), ("b", 3)]


def test_invert_name_lookup_probes_until_none():
    inverse = M._invertNameLookup("boundsCheckName")
    assert inverse
    for name, mode in inverse.items():
        assert ClientWriter.boundsCheckName(mode) == name


# ---------------------------------------------------------------------------
# Reading ClientParameters.ini
# ---------------------------------------------------------------------------
def test_coerce_client_value_by_decoder_and_registry_type():
    randomName = ClientWriter.DataInitName.Random.name
    assert M._coerceClientValue("DataInitTypeA", randomName, "DataInitName") == (
        ClientWriter.DataInitName.Random.value
    )
    nanMode = M._invertNameLookup("boundsCheckName")
    name = next(iter(nanMode))
    assert M._coerceClientValue("BoundsCheck", name, "boundsCheckName") == nanMode[name]
    assert M._coerceClientValue("KernelTime", "True", None) is True
    assert M._coerceClientValue("KernelTime", "0", None) is False
    assert M._coerceClientValue("NumWarmups", "7", None) == 7
    assert M._coerceClientValue("SkipSlowSolutionRatio", "0.5", None) == 0.5


def test_coerce_client_value_infers_type_without_registry_entry():
    assert M._coerceClientValue("NotAGlobal", "false", None) is False
    assert M._coerceClientValue("NotAGlobal", "3", None) == 3
    assert M._coerceClientValue("NotAGlobal", "2.5", None) == 2.5
    assert M._coerceClientValue("NotAGlobal", "text", None) == "text"


def _writeIni(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_client_parameters(tmp_path, capsys):
    ini = _writeIni(
        tmp_path / M.CLIENT_PARAMETERS_FILENAME,
        [
            "[client]",
            "# comment",
            "; comment",
            "",
            "no-equals-sign",
            "num-warmups=5",
            "num-warmups=9",
            "device-idx=3",
            "init-a={}".format(ClientWriter.DataInitName.Random.name),
            "print-valids=true",
            "num-benchmarks=notanumber",
            "unknown-setting=1",
        ],
    )
    recovered = M.loadClientParameters(str(ini))
    assert recovered["NumWarmups"] == 5
    assert recovered["Device"] == 3
    assert recovered["DataInitTypeA"] == ClientWriter.DataInitName.Random.value
    assert recovered["ValidationPrintValids"] is True
    assert "NumBenchmarks" not in recovered
    assert list(recovered) == ["NumWarmups", "Device", "DataInitTypeA", "ValidationPrintValids"]
    assert "ignoring num-benchmarks='notanumber'" in capsys.readouterr().out


def test_drop_defaults():
    default = M.globalParameterDefaults["NumWarmups"]
    kept = M.dropDefaults({"NumWarmups": default, "SleepPercent": 42, "NotAGlobal": 1})
    assert kept == {"SleepPercent": 42, "NotAGlobal": 1}


# ---------------------------------------------------------------------------
# Locating the build tree's settings
# ---------------------------------------------------------------------------
def _problemDir(tmp_path):
    return tmp_path / "build_tune" / "1_BenchmarkProblems" / "Cijk_Alik_Bljk_HHS_00"


def test_find_client_parameters_next_to_the_step(tmp_path):
    problem = _problemDir(tmp_path)
    data = problem / "Data" / "00_Final.yaml"
    ini = _writeIni(problem / "00_Final" / "source" / M.CLIENT_PARAMETERS_FILENAME, ["a=1"])
    _writeIni(problem / "other" / M.CLIENT_PARAMETERS_FILENAME, ["a=2"])
    assert M.findClientParameters(str(data)) == str(ini)


def test_find_client_parameters_falls_back_to_a_single_config(tmp_path):
    problem = _problemDir(tmp_path)
    data = problem / "Data" / "00_Final.yaml"
    ini = _writeIni(problem / "elsewhere" / M.CLIENT_PARAMETERS_FILENAME, ["a=1"])
    assert M.findClientParameters(str(data)) == str(ini)
    _writeIni(problem / "second" / M.CLIENT_PARAMETERS_FILENAME, ["a=2"])
    assert M.findClientParameters(str(data)) is None


def test_find_client_parameters_only_inside_a_data_directory(tmp_path):
    _writeIni(tmp_path / "source" / M.CLIENT_PARAMETERS_FILENAME, ["a=1"])
    assert M.findClientParameters(str(tmp_path / "logic" / "gfx950_logic.yaml")) is None


def test_find_run_config(tmp_path):
    data = _problemDir(tmp_path) / "Data" / "00_Final.yaml"
    assert M.findRunConfig(str(data)) is None
    config = tmp_path / "tune.yml"
    config.write_text("{}")
    assert M.findRunConfig(str(data)) == str(config)
    assert M.findRunConfig(str(tmp_path / "no_build_dir" / "x.yaml")) is None


def test_load_run_config_splits_sections_and_drops_the_backend(tmp_path, capsys):
    config = tmp_path / "tune.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "GlobalParameters": {"KeepBuildTmp": True},
                "BenchmarkProblems": [],
                "LibraryLogic": {"ScheduleName": "gfx950"},
                "Backend": {"Name": "Ductile"},
                "Ductile": {"Setting": 1},
                "Extra": {"Key": "value"},
            }
        )
    )
    globals_, libraryLogic, extras = M.loadRunConfig(str(config))
    assert globals_ == {"KeepBuildTmp": True}
    assert libraryLogic == {"ScheduleName": "gfx950"}
    assert extras == {"Extra": {"Key": "value"}}
    assert "Not carrying over the Ductile backend" in capsys.readouterr().out


def test_load_run_config_tolerates_unreadable_and_non_mapping_files(tmp_path, capsys):
    broken = tmp_path / "broken.yaml"
    broken.write_text("GlobalParameters: [unclosed\n")
    assert M.loadRunConfig(str(broken)) == ({}, {}, {})
    assert "could not read run config" in capsys.readouterr().out
    listed = tmp_path / "list.yaml"
    listed.write_text("- a\n- b\n")
    assert M.loadRunConfig(str(listed)) == ({}, {}, {})


def test_resolve_run_settings_disabled_or_absent(tmp_path):
    data = tmp_path / "plain" / "logic.yaml"
    assert M.resolveRunSettings(str(data), None, useRunConfig=False) is None
    assert M.resolveRunSettings(str(data), None, useRunConfig=True) is None
    with pytest.raises(RuntimeError, match="Run config not found"):
        M.resolveRunSettings(str(data), str(tmp_path / "missing.yaml"), useRunConfig=True)


def test_resolve_run_settings_client_config_wins(tmp_path):
    problem = _problemDir(tmp_path)
    data = problem / "Data" / "00_Final.yaml"
    _writeIni(
        problem / "00_Final" / "source" / M.CLIENT_PARAMETERS_FILENAME,
        ["num-warmups=5", "sleep-percent={}".format(M.globalParameterDefaults["SleepPercent"])],
    )
    (tmp_path / "tune.yaml").write_text(
        yaml.safe_dump(
            {
                "GlobalParameters": {"NumWarmups": 99, "SleepPercent": 50, "KeepBuildTmp": True},
                "LibraryLogic": {"ScheduleName": "gfx950"},
            }
        )
    )
    settings = M.resolveRunSettings(str(data), None, useRunConfig=True)
    # The client config recorded what ran, defaults included.
    assert settings.globalParameters == {"NumWarmups": 5, "KeepBuildTmp": True}
    assert settings.libraryLogic == {"ScheduleName": "gfx950"}
    assert len(settings.sources) == 2
    assert "1 at default" in settings.describe()


def test_resolve_run_settings_with_an_empty_run_config(tmp_path):
    config = tmp_path / "empty.yaml"
    config.write_text("{}\n")
    assert M.resolveRunSettings(str(tmp_path / "logic.yaml"), str(config), useRunConfig=True) is None


def test_run_settings_describe_without_sources():
    assert M.RunSettings().describe() == "no source"


def test_set_global_params_prefers_recovered_settings(capsys):
    settings = M.RunSettings(
        globalParameters={"MinimumRequiredVersion": "4.0.0", "NumWarmups": 5},
        sources=["client.ini (1 settings, 0 at default)"],
    )
    params = M.setGlobalParams({"MinimumRequiredVersion": "5.0.0"}, {"DataType": "H"}, settings)
    assert params == {"MinimumRequiredVersion": "5.0.0", "NumWarmups": 5}
    assert "GlobalParameters recovered from: client.ini" in capsys.readouterr().out


def test_library_logic_placeholders_for_benchmark_data(capsys):
    source = M.SolutionSource(
        versionString={},
        scheduleName="gfx950",
        architectureName="gfx950",
        deviceNames=[M.FALLBACK_DEVICE_NAME],
        problemType={},
        solution={},
        hasPlaceholderLibraryLogic=True,
    )
    data = M.formLibraryLogic(source)
    assert str(data["ScheduleName"]) == "gfx950"
    assert "placeholders" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End to end from a benchmark build tree
# ---------------------------------------------------------------------------
def _writeBenchmarkTree(tmp_path):
    logic = LibraryIO.readYAML(_LOGIC)
    solution = dict(logic[5][0])
    solution.update({"ProblemType": logic[4], "ISA": [9, 5, 0], "SolutionIndex": 0})
    problem = _problemDir(tmp_path)
    data = problem / "Data" / "00_Final.yaml"
    data.parent.mkdir(parents=True)
    data.write_text(
        yaml.safe_dump(
            [
                {"MinimumRequiredVersion": "5.0.0"},
                {"ProblemSizes": [{"Exact": [256, 512, 1, 128]}]},
                {"BiasTypeArgs": [[0, 4]]},
                solution,
            ]
        )
    )
    _writeIni(problem / "00_Final" / "source" / M.CLIENT_PARAMETERS_FILENAME, ["num-warmups=5"])
    (tmp_path / "tune.yaml").write_text(
        yaml.safe_dump(
            {
                "GlobalParameters": {"KeepBuildTmp": True, "NumWarmups": 99},
                "LibraryLogic": {"ScheduleName": "gfx950", "DeviceNames": ["Device 75a0"]},
                "Backend": {"Name": "Ductile"},
                "Ductile": {"Setting": 1},
                "Extra": {"Key": "value"},
            }
        )
    )
    return data


def test_benchmark_tree_to_config(tmp_path):
    data = _writeBenchmarkTree(tmp_path)
    out = tmp_path / "config.yaml"
    assert M.TensileLibLogicToYaml(str(data), 0, str(out), False) == str(out)
    config = yaml.safe_load(out.read_text())
    assert list(config) == ["GlobalParameters", "BenchmarkProblems", "LibraryLogic", "Extra"]
    params = config["GlobalParameters"]
    assert params["NumWarmups"] == 5 and params["KeepBuildTmp"] is True
    assert params["ISA"] == [[9, 5, 0]]
    assert config["LibraryLogic"] == {
        "ScheduleName": "gfx950",
        "DeviceNames": ["Device 75a0"],
        "ArchitectureName": "gfx950",
    }
    # The header's bias types go with the epilogue settings (DECISIONS D49).
    final = config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"]
    assert final == [{"ProblemSizes": [{"Exact": [256, 512, 1, 128]}]}]


def test_benchmark_tree_keeps_header_bias_types_with_epilogues(tmp_path):
    data = _writeBenchmarkTree(tmp_path)
    out = tmp_path / "config.yaml"
    M.TensileLibLogicToYaml(str(data), 0, str(out), False, keepEpilogues=True)
    final = yaml.safe_load(out.read_text())["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"]
    assert final[1]["BiasTypeArgs"] == [0, 4]


def test_benchmark_tree_without_run_config(tmp_path):
    data = _writeBenchmarkTree(tmp_path)
    out = tmp_path / "config.yaml"
    M.TensileLibLogicToYaml(str(data), 0, str(out), False, useRunConfig=False)
    config = yaml.safe_load(out.read_text())
    assert list(config) == ["GlobalParameters", "BenchmarkProblems", "LibraryLogic"]
    assert config["GlobalParameters"]["KeepBuildTmp"] is False
    assert config["LibraryLogic"]["DeviceNames"] == [M.FALLBACK_DEVICE_NAME]


def test_orchestrator_requires_a_solution_index(monkeypatch):
    monkeypatch.setattr(M.LibraryIO, "readYAML", lambda p: {"nonempty": 1})
    with pytest.raises(RuntimeError, match="At least one solution idx"):
        M.TensileLibLogicToYaml("logic.yaml", "", "out.yaml", skipMI=False)


def test_fork_params_without_matrix_instruction_flag():
    data = M.formForkParams({"WorkGroup": [64, 4, 1]}, skipMI=False)
    group = data["ForkParameters"][-1]["Groups"][0][0]
    assert list(group["WorkGroup"]) == [64, 4, 1]
    assert "MatrixInstruction" not in group
