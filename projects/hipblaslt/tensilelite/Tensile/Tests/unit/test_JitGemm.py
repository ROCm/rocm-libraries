# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from Tensile import JitGemm as JG, SingleSolution as SS


pytestmark = pytest.mark.unit


@pytest.fixture
def prediction_request():
    return {
        "schema_version": 1,
        "architecture": "gfx950",
        "model": "origami.gemm.estimation",
        "problem": {
            "m": 67, "n": 99, "k": 73, "batch": 3,
            "transpose_a": True, "transpose_b": False,
            "data_type": "h", "high_precision_accumulate": True,
            "c_equals_d": False,
            "strides_a": [1, 80, 5360], "strides_b": [1, 80, 7920],
            "strides_c": [1, 70, 6930], "strides_d": [1, 71, 7029],
        },
        "candidates": [
            {"id": 7, "predicted_cycles": 100.0,
             "parameters": {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2],
                            "DepthU": 32, "NonTemporalA": 0, "NonTemporalB": 0}},
            {"id": 2, "predicted_cycles": 120.0,
             "parameters": {"MatrixInstruction": [16, 16, 16, 1, 1, 2, 2, 2, 2],
                            "DepthU": 64, "NonTemporalA": 0, "NonTemporalB": 0}},
        ],
    }


def solution(depth=64):
    return {
        "Valid": True, "MacroTile0": 64, "MacroTile1": 64, "DepthU": depth,
        "StreamK": 0, "GlobalSplitU": 1,
        "AssertFree0ElementMultiple": 1, "AssertFree1ElementMultiple": 1,
        "AssertSummationElementMultiple": 1,
        "GlobalReadVectorWidthA": 1, "GlobalReadVectorWidthB": 1,
        "BufferLoad": True, "BufferStore": True,
        "ProblemType": {"TLUA": False, "TLUB": False},
        "MIWaveTile": [2, 2], "MIWaveGroup": [2, 2], "NumThreads": 256,
        "_GlobalAccumulation": None,
    }


def test_recipe_preserves_descriptors_and_uses_canonical_defaults(prediction_request):
    config = JG._configuration(prediction_request, prediction_request["candidates"][0])
    problem, group = config["BenchmarkProblems"][0]
    assert problem["TransposeA"] and not problem["TransposeB"]
    exact = group["BenchmarkFinalParameters"][0]["ProblemSizes"][0]["Exact"]
    assert exact["sizes"] == [67, 99, 3, 73]
    assert exact["stridesA"] == [1, 80, 5360]
    assert exact["stridesC"] != exact["stridesD"]
    assert {next(iter(item)) for item in group["ForkParameters"]} == JG._PARAMETERS
    assert "BenchmarkCommonParameters" not in group


@pytest.mark.parametrize("latency", [0, -1, float("inf"), float("nan"), sys.float_info.max])
def test_reject_invalid_origami_scores_before_generation(prediction_request, tmp_path, latency):
    prediction_request["candidates"][0]["predicted_cycles"] = latency
    path = tmp_path / "prediction_request.json"
    path.write_text(json.dumps(prediction_request))
    with pytest.raises(SS.SingleSolutionConfigError):
        JG._readRequest(path)


def test_reject_duplicate_or_unmodeled_candidate_parameters(prediction_request, tmp_path):
    prediction_request["candidates"][0]["parameters"]["NoReject"] = True
    path = tmp_path / "prediction_request.json"
    path.write_text(json.dumps(prediction_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="only MI"):
        JG._readRequest(path)


def test_ranked_validation_rejects_first_and_builds_only_winner(prediction_request, tmp_path, monkeypatch):
    source = tmp_path / "prediction_request.json"
    source.write_text(json.dumps(prediction_request))
    calls = []
    builds = []

    def generate(requestPath, outputPath, *, _selection, **options):
        def derive(config, label):
            calls.append(label)
            if len(calls) == 1:
                print("reject: LDS capacity exceeded")
                raise SS.SingleSolutionRejected("LDS capacity")
            return solution()

        configPath, chosen, prediction = _selection[1](derive)
        builds.append(chosen)
        outputPath.mkdir()
        manifest = outputPath / "manifest.json"
        manifest.write_text(json.dumps({"jit_prediction": prediction}))
        return SimpleNamespace(manifestPath=manifest)

    monkeypatch.setattr(SS, "_generateAndBuild", generate)
    output = tmp_path / "result"
    JG.generateAndBuildJitGemm(source, output, architecture="gfx950")
    prediction = json.loads(Path(str(output) + ".prediction.json").read_text())
    assert len(calls) == 2 and len(builds) == 1
    assert prediction["candidate_id"] == 2
    assert prediction["rejections"][0]["candidate_id"] == 7
    assert prediction["rejections"][0]["reason"] == "reject: LDS capacity exceeded"
    assert prediction["default_parameters"]["GlobalSplitUAlgorithm"] == "MultipleBuffer"
    config = yaml.safe_load(Path(str(output) + ".yaml").read_text())
    assert config["BenchmarkProblems"][0][1]["ForkParameters"] == [
        {name: [value]} for name, value in prediction_request["candidates"][1]["parameters"].items()]


def test_unexpected_derivation_failure_does_not_try_next_candidate(prediction_request, tmp_path):
    calls = []

    def derive(config, label):
        calls.append(label)
        raise FileNotFoundError("assembler support resource disappeared")

    with pytest.raises(FileNotFoundError):
        JG._select(prediction_request, tmp_path / "selected.yaml", derive)
    assert len(calls) == 1
    assert not (tmp_path / "selected.yaml").exists()


def test_all_rejected_creates_no_selected_yaml(prediction_request, tmp_path):
    def derive(config, label):
        raise SS.SingleSolutionRejected("coupled parameter constraint")

    with pytest.raises(SS.SingleSolutionConfigError, match="No Origami-ranked candidate"):
        JG._select(prediction_request, tmp_path / "selected.yaml", derive)
    assert not (tmp_path / "selected.yaml").exists()


def test_runtime_shape_constraint_checked_before_compilation(prediction_request):
    candidate = prediction_request["candidates"][1]
    derived = solution()
    derived["AssertSummationElementMultiple"] = 8
    assert "AssertSummationElementMultiple" in JG._problemRejection(derived, prediction_request, candidate)
    derived = solution()
    derived["ProblemType"]["TLUA"] = True
    derived["GlobalReadVectorWidthA"] = 128
    assert "Leading free dimension" in JG._problemRejection(derived, prediction_request, candidate)


def test_strict_validator_propagates_unexpected_errors(monkeypatch):
    from Tensile import BenchmarkProblems as BP

    def fail(*args, **kwargs):
        raise OSError("missing assembly backend resource")

    monkeypatch.setattr(BP, "matrixInstructionToMIParameters", fail)
    config = {"MatrixInstruction": [16, 16, 16, 1, 1, 2, 2, 2, 2],
              "WorkGroup": [16, 16, 1], "ProblemType": {}, "ISA": (9, 5, 0),
              "WavefrontSize": 64}
    with pytest.raises(OSError, match="missing assembly backend resource"):
        BP._build_and_validate_solution(config, None, SimpleNamespace(), {}, strictErrors=True)
