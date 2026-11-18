# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Global parameters, ini emission and step checks of the benchmark database mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

pytestmark = pytest.mark.unit

from tensilelite import ClientWriter
from tensilelite.BenchmarkProblems import (
    _checkBenchmarkDatabaseSupported,
    _shouldRunBenchmarkClient,
    _removeStaleCollectedResults,
)
from tensilelite.Common.GlobalParameters import (
    assignGlobalParameters,
    benchmarkDatabaseApplies,
    benchmarkDatabaseUnsupportedReason,
    globalParameters,
    restoreDefaultGlobalParameters,
    isBenchmarkDatabaseExplicit,
    preserveGlobalParameters,
    setBenchmarkDatabaseExplicit,
    validateBenchmarkDatabaseParameters,
)
from tensilelite.Common.TypeValidationErrors import ConfigTypeError
from tensilelite.Contractions import ProblemType
from tensilelite.SolutionStructs.Problem import ProblemSizesMockDummy

EMPTY_ISA_INFO = {}


@pytest.fixture(autouse=True)
def restore_global_parameters():
    restoreDefaultGlobalParameters()
    yield
    restoreDefaultGlobalParameters()


def set_explicit_database() -> None:
    """Make BenchmarkDatabase=True count as set by the user, so unsupported options are errors."""
    globalParameters["BenchmarkDatabase"] = True
    setBenchmarkDatabaseExplicit(True)


# ---------------------------------------------------------------------------
# ini emission.
# ---------------------------------------------------------------------------


class _FactorDimArgs:
    factorDims = [0]


def write_ini(
    tmp_path: Path, for_benchmark: bool, fused_a2a: bool = False, icache_flush_args=""
) -> List[str]:
    problem_type = ProblemType.FromOriginalState(
        {
            "OperationType": "GEMM",
            "DataType": "S",
            "DestDataType": "S",
            "ComputeDataType": "S",
            "TransposeA": 0,
            "TransposeB": 0,
            "Batched": True,
            "UseBeta": True,
            "TotalIndices": 4,
            "NumIndicesC": 3,
            "IndicesFree": [0, 1],
            "IndicesBatch": [2],
            "IndicesSummation": [3],
            "IndexAssignmentsA": [0, 3, 2],
            "IndexAssignmentsB": [3, 1, 2],
            "ComplexConjugateA": False,
            "ComplexConjugateB": False,
            "ActivationComputeDataType": "S",
            "UseBias": 1,
            "BiasSrc": "D",
            "HighPrecisionAccumulate": False,
        }
    )
    if fused_a2a:
        problem_type.fusedA2A = "Enabled"
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    path = source / "ClientParameters.ini"
    ClientWriter.writeClientConfigIni(
        forBenchmark=for_benchmark,
        problemSizes=ProblemSizesMockDummy(),
        biasTypeArgs="",
        factorDimArgs=_FactorDimArgs(),
        activationArgs="",
        icacheFlushArgs=icache_flush_args,
        problemType=problem_type,
        sourceDir=str(source),
        codeObjectFiles=["TensileLibrary_gfx1250.co"],
        resultsFileName="/DATA/benchmark.csv",
        parametersFilePath=str(path),
        deviceId=0,
        gfxName="gfx1250",
        libraryFile="/LIB/gfx1250/TensileLibrary.dat",
    )
    return path.read_text().splitlines()


def test_the_mode_is_on_by_default():
    assert globalParameters["BenchmarkDatabase"] is True
    assert isBenchmarkDatabaseExplicit() is False
    assert "BenchmarkDatabaseExplicit" not in globalParameters


def test_ini_has_the_database_next_to_the_results(tmp_path: Path):
    lines = write_ini(tmp_path, for_benchmark=True)
    assert "results-file=/DATA/benchmark.csv" in lines
    assert "benchmark-db=/DATA/benchmark.db" in lines


def test_ini_has_no_database_when_explicitly_disabled(tmp_path: Path):
    globalParameters["BenchmarkDatabase"] = False
    assert not any(line.startswith("benchmark-db") for line in write_ini(tmp_path, True))


@pytest.mark.parametrize("fused_a2a, icache_flush_args", [(True, [False]), (False, [False, True])])
def test_unsupported_group_ini_has_no_database_options(
    fused_a2a, icache_flush_args, tmp_path: Path
):
    lines = write_ini(tmp_path, True, fused_a2a, icache_flush_args)
    assert not any(line.startswith(("benchmark-db", "indirect-")) for line in lines)


def test_supported_group_ini_has_the_database_options_next_to_unsupported_ones(tmp_path: Path):
    write_ini(tmp_path, True, fused_a2a=True)
    lines = write_ini(tmp_path, True, icache_flush_args=[False])
    assert "benchmark-db=/DATA/benchmark.db" in lines
    assert "indirect-failures=True" in lines


def test_library_client_ini_never_has_a_database(tmp_path: Path):
    assert not any(line.startswith("benchmark-db") for line in write_ini(tmp_path, False))


def test_ini_has_the_indirect_failure_options_on_by_default(tmp_path: Path):
    lines = write_ini(tmp_path, for_benchmark=True)
    assert "indirect-failures=True" in lines
    assert "indirect-verification-failures=True" in lines


def test_ini_has_the_flipped_indirect_failure_options(tmp_path: Path):
    globalParameters["BenchmarkIndirectFailures"] = False
    globalParameters["BenchmarkIndirectVerificationFailures"] = False
    lines = write_ini(tmp_path, for_benchmark=True)
    assert "indirect-failures=False" in lines
    assert "indirect-verification-failures=False" in lines


@pytest.mark.parametrize("database, for_benchmark", [(False, True), (True, False)])
def test_ini_has_no_indirect_failure_options_outside_the_database_mode(
    database, for_benchmark, tmp_path: Path
):
    globalParameters["BenchmarkDatabase"] = database
    lines = write_ini(tmp_path, for_benchmark)
    assert not any(line.startswith("indirect-") for line in lines)


@pytest.mark.parametrize(
    "key", ["BenchmarkIndirectFailures", "BenchmarkIndirectVerificationFailures"]
)
def test_indirect_failure_options_must_be_booleans(key):
    with pytest.raises(ConfigTypeError, match=key):
        assignGlobalParameters({key: 1}, EMPTY_ISA_INFO)


# ---------------------------------------------------------------------------
# Parameter validation.
# ---------------------------------------------------------------------------


def exit_message(capsys, overrides=None) -> str:
    with pytest.raises(SystemExit):
        validateBenchmarkDatabaseParameters(overrides or {})
    return capsys.readouterr().out


LEGACY_NOTE = "using the legacy benchmark mode"

UNSUPPORTED_OPTIONS = [
    ({"ParallelGpuExecution": 0}, "ParallelGpuExecution != 1"),
    ({"ParallelGpuExecution": 2}, "ParallelGpuExecution != 1"),
    ({"NumBenchmarks": 2}, "NumBenchmarks > 1"),
    ({"CpuOnly": True}, "CpuOnly"),
]


def test_defaults_are_valid(capsys):
    validateBenchmarkDatabaseParameters({})
    set_explicit_database()
    validateBenchmarkDatabaseParameters({})
    assert globalParameters["BenchmarkDatabase"] is True
    assert capsys.readouterr().out == ""


# The pairs that the prediction does not select are recorded as NOT_SELECTED, so every threshold
# is supported, both when the mode is explicit and when it is the default.
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0, 1.0001, 2.0])
def test_every_prediction_threshold_is_accepted(threshold, explicit, capsys):
    globalParameters["PredictionThreshold"] = threshold
    if explicit:
        set_explicit_database()
    validateBenchmarkDatabaseParameters({})
    assert globalParameters["BenchmarkDatabase"] is True
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0, 2.0])
def test_ini_passes_the_prediction_threshold_unchanged_in_the_database_mode(
    threshold, tmp_path: Path
):
    globalParameters["PredictionThreshold"] = threshold
    lines = write_ini(tmp_path, for_benchmark=True)
    assert "benchmark-db=/DATA/benchmark.db" in lines
    assert f"prediction-threshold={threshold}" in lines


@pytest.mark.parametrize("options, reason", UNSUPPORTED_OPTIONS)
def test_falls_back_to_the_legacy_mode_when_not_explicit(options, reason, capsys):
    assignGlobalParameters(dict(options), EMPTY_ISA_INFO)
    validateBenchmarkDatabaseParameters({})
    assert globalParameters["BenchmarkDatabase"] is False
    note = f"# BenchmarkDatabase does not support {reason}; {LEGACY_NOTE}"
    assert note in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("options, reason", UNSUPPORTED_OPTIONS)
def test_rejects_an_explicit_mode_with_an_unsupported_option(options, reason, capsys):
    assignGlobalParameters({"BenchmarkDatabase": True, **options}, EMPTY_ISA_INFO)
    out = exit_message(capsys)
    assert f"BenchmarkDatabase does not support {reason}" in out
    assert LEGACY_NOTE not in out


@pytest.mark.parametrize("options, reason", UNSUPPORTED_OPTIONS)
def test_keeps_an_explicitly_disabled_mode_silently(options, reason, capsys):
    assignGlobalParameters({"BenchmarkDatabase": False, **options}, EMPTY_ISA_INFO)
    validateBenchmarkDatabaseParameters({})
    assert globalParameters["BenchmarkDatabase"] is False
    assert LEGACY_NOTE not in capsys.readouterr().out


@pytest.mark.parametrize("options, reason", UNSUPPORTED_OPTIONS)
def test_override_of_an_unsupported_option_falls_back(options, reason, capsys):
    assignGlobalParameters({}, EMPTY_ISA_INFO)
    globalParameters.update(options)
    validateBenchmarkDatabaseParameters(options)
    assert globalParameters["BenchmarkDatabase"] is False
    assert f"{reason}; {LEGACY_NOTE}" in capsys.readouterr().out


def test_override_of_the_mode_is_explicit_and_rejects_an_unsupported_option(capsys):
    assignGlobalParameters({"NumBenchmarks": 2}, EMPTY_ISA_INFO)
    globalParameters["BenchmarkDatabase"] = True
    assert "NumBenchmarks > 1" in exit_message(capsys, {"BenchmarkDatabase": True})


def test_assigning_alone_does_not_validate(capsys):
    assignGlobalParameters({"BenchmarkDatabase": True, "NumBenchmarks": 2}, EMPTY_ISA_INFO)
    assignGlobalParameters({"NumBenchmarks": 2}, EMPTY_ISA_INFO)
    assert globalParameters["BenchmarkDatabase"] is True
    assert "BenchmarkDatabase does not support" not in capsys.readouterr().out


def test_all_offenders_are_reported_together(capsys):
    set_explicit_database()
    globalParameters["ParallelGpuExecution"] = 2
    globalParameters["NumBenchmarks"] = 3
    globalParameters["CpuOnly"] = True
    message = exit_message(capsys)
    assert "ParallelGpuExecution != 1, NumBenchmarks > 1, CpuOnly" in message


def test_only_the_violated_options_are_listed(capsys):
    set_explicit_database()
    globalParameters["NumBenchmarks"] = 3
    globalParameters["PredictionThreshold"] = 1.0
    message = exit_message(capsys)
    assert "NumBenchmarks" in message
    assert "ParallelGpuExecution" not in message
    assert "CpuOnly" not in message
    assert "PredictionThreshold" not in message


def test_the_other_options_are_ignored_without_the_mode():
    globalParameters["BenchmarkDatabase"] = False
    globalParameters["ParallelGpuExecution"] = 2
    globalParameters["NumBenchmarks"] = 3
    globalParameters["CpuOnly"] = True
    validateBenchmarkDatabaseParameters({})


@pytest.mark.parametrize("enabled", [False, True])
def test_negative_timeout_is_always_rejected(enabled, capsys):
    globalParameters["BenchmarkDatabase"] = enabled
    globalParameters["BenchmarkSolutionTimeout"] = -1
    assert "BenchmarkSolutionTimeout must not be negative" in exit_message(capsys)


def test_assign_accepts_the_mode():
    assignGlobalParameters(
        {"BenchmarkDatabase": True, "BenchmarkSolutionTimeout": 30}, EMPTY_ISA_INFO
    )
    assert globalParameters["BenchmarkDatabase"] is True
    assert globalParameters["BenchmarkSolutionTimeout"] == 30


def test_assign_rejects_a_non_bool_mode():
    with pytest.raises(ConfigTypeError, match="BenchmarkDatabase"):
        assignGlobalParameters({"BenchmarkDatabase": 1}, EMPTY_ISA_INFO)


def test_assign_rejects_a_float_timeout():
    with pytest.raises(ConfigTypeError, match="BenchmarkSolutionTimeout"):
        assignGlobalParameters({"BenchmarkSolutionTimeout": 1.5}, EMPTY_ISA_INFO)


def test_assign_rejects_a_user_supplied_explicit_marker():
    with pytest.raises(ConfigTypeError, match="BenchmarkDatabaseExplicit"):
        assignGlobalParameters({"BenchmarkDatabaseExplicit": True}, EMPTY_ISA_INFO)


@pytest.mark.parametrize("config, explicit", [({}, False), ({"BenchmarkDatabase": False}, True)])
def test_assign_records_whether_the_mode_was_given(config, explicit):
    assignGlobalParameters(dict(config), EMPTY_ISA_INFO)
    assert isBenchmarkDatabaseExplicit() is explicit


def test_an_override_of_the_mode_makes_it_explicit():
    assignGlobalParameters({}, EMPTY_ISA_INFO)
    validateBenchmarkDatabaseParameters({"BenchmarkDatabase": False})
    assert isBenchmarkDatabaseExplicit() is True


def test_restoring_the_defaults_resets_the_explicit_state():
    setBenchmarkDatabaseExplicit(True)
    restoreDefaultGlobalParameters()
    assert isBenchmarkDatabaseExplicit() is False


def test_preserving_the_parameters_restores_the_explicit_state():
    setBenchmarkDatabaseExplicit(True)
    with preserveGlobalParameters():
        setBenchmarkDatabaseExplicit(False)
    assert isBenchmarkDatabaseExplicit() is True


# ---------------------------------------------------------------------------
# Benchmark step checks.
# ---------------------------------------------------------------------------


UNSUPPORTED_GROUPS = [
    ({"FusedA2A": "Enabled"}, [False], "fused A2A"),
    ({}, [False, True], "multiple ICacheFlush"),
]


@pytest.mark.parametrize("problem_type, icache_flush_args, message", UNSUPPORTED_GROUPS)
def test_unsupported_step_exits_when_the_mode_is_explicit(
    problem_type, icache_flush_args, message, capsys
):
    step = SimpleNamespace(icacheFlushArgs=icache_flush_args)
    set_explicit_database()
    with pytest.raises(SystemExit):
        _checkBenchmarkDatabaseSupported(problem_type, step)
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("problem_type, icache_flush_args, message", UNSUPPORTED_GROUPS)
def test_unsupported_step_falls_back_when_the_mode_is_not_explicit(
    problem_type, icache_flush_args, message, capsys
):
    step = SimpleNamespace(icacheFlushArgs=icache_flush_args)
    _checkBenchmarkDatabaseSupported(problem_type, step)
    out = capsys.readouterr().out
    assert out.startswith("# BenchmarkDatabase does not support")
    assert message in out
    assert out.rstrip().endswith("; using the legacy benchmark mode")
    assert globalParameters["BenchmarkDatabase"] is True


def test_supported_step_is_accepted_in_database_mode(capsys):
    set_explicit_database()
    step = SimpleNamespace(icacheFlushArgs=[False])
    _checkBenchmarkDatabaseSupported({"FusedA2A": "Disabled"}, step)
    assert capsys.readouterr().out == ""


def test_unsupported_step_is_accepted_without_the_mode(capsys):
    globalParameters["BenchmarkDatabase"] = False
    setBenchmarkDatabaseExplicit(True)
    step = SimpleNamespace(icacheFlushArgs=[False, True])
    _checkBenchmarkDatabaseSupported({"FusedA2A": "Enabled"}, step)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "fused_a2a, icache_flush_args, reason",
    [
        (True, [False], "fused A2A"),
        (False, [False, True], "multiple ICacheFlush"),
        (False, [False], None),
        (False, "", None),
        (False, None, None),
    ],
)
def test_unsupported_reason(fused_a2a, icache_flush_args, reason):
    result = benchmarkDatabaseUnsupportedReason(fused_a2a, icache_flush_args)
    assert (result is None) if reason is None else (reason in result)


def test_database_applies_needs_the_global_mode_and_a_supported_group():
    assert benchmarkDatabaseApplies(False, [False])
    assert not benchmarkDatabaseApplies(True, [False])
    assert not benchmarkDatabaseApplies(False, [False, True])
    globalParameters["BenchmarkDatabase"] = False
    assert not benchmarkDatabaseApplies(False, [False])


@pytest.mark.parametrize("force_redo", [False, True])
def test_database_groups_always_run_the_client(force_redo, tmp_path: Path):
    globalParameters["ForceRedoBenchmarkProblems"] = force_redo
    partial_csv = tmp_path / "partial.csv"
    partial_csv.write_text("partial")
    step = SimpleNamespace(icacheFlushArgs=[False])
    assert _shouldRunBenchmarkClient({}, step, str(partial_csv))


def test_legacy_groups_skip_an_existing_csv_without_force_redo(tmp_path: Path):
    globalParameters["BenchmarkDatabase"] = False
    existing = tmp_path / "existing.csv"
    existing.write_text("done")
    step = SimpleNamespace(icacheFlushArgs=[False])
    globalParameters["ForceRedoBenchmarkProblems"] = False
    assert not _shouldRunBenchmarkClient({}, step, str(existing))
    assert _shouldRunBenchmarkClient({}, step, str(tmp_path / "missing.csv"))
    globalParameters["ForceRedoBenchmarkProblems"] = True
    assert _shouldRunBenchmarkClient({}, step, str(existing))


def test_stale_collected_results_are_removed(tmp_path: Path):
    paths = [tmp_path / name for name in ("group.csv", "group.yaml", "group.gsp")]
    for path in paths[:2]:
        path.write_text("old")
    other = tmp_path / "other.csv"
    other.write_text("keep")
    _removeStaleCollectedResults(*paths)
    assert not any(path.exists() for path in paths)
    assert other.exists()
