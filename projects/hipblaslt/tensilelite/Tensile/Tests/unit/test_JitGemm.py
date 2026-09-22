# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import shutil
import subprocess
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
        "NonTemporalA": 0, "NonTemporalB": 0,
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


@pytest.mark.parametrize("architecture, dtype, mi_k", [
    ("gfx90a", "h", 16), ("gfx942", "h", 16), ("gfx950", "h", 32),
    ("gfx90a", "s", 4), ("gfx942", "s", 4), ("gfx950", "s", 4),
    ("gfx1250", "h", 32), ("gfx1250", "s", 4),
])
def test_request_uses_architecture_legal_instruction_and_hints(
    prediction_request, tmp_path, architecture, dtype, mi_k
):
    prediction_request["architecture"] = architecture if architecture == "gfx1250" else architecture + ":xnack-"
    prediction_request["problem"]["data_type"] = dtype
    prediction_request["candidates"] = prediction_request["candidates"][:1]
    parameters = prediction_request["candidates"][0]["parameters"]
    parameters["MatrixInstruction"][2] = mi_k
    parameters["NonTemporalB"] = 0 if architecture in ("gfx90a", "gfx1250") else 4
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    request = JG._readRequest(source)
    assert request["candidates"][0]["parameters"] == parameters
    assert not request["problem"]["output_amax_d"]
    assert not request["problem"]["use_scale_cd"]


@pytest.mark.parametrize("architecture, dtype, mi_k, hint", [
    ("gfx90a", "h", 32, 0), ("gfx942", "h", 32, 0),
    ("gfx90a", "s", 16, 0), ("gfx942", "s", 16, 0), ("gfx950", "s", 16, 0),
    ("gfx90a", "h", 16, 4), ("gfx90a", "s", 4, 4),
    ("gfx1250", "h", 16, 0), ("gfx1250", "h", 32, 4), ("gfx1250", "s", 4, 4),
])
def test_request_rejects_architecture_incompatible_recipe(
    prediction_request, tmp_path, architecture, dtype, mi_k, hint
):
    prediction_request["architecture"] = architecture
    prediction_request["problem"]["data_type"] = dtype
    prediction_request["candidates"] = prediction_request["candidates"][:1]
    parameters = prediction_request["candidates"][0]["parameters"]
    parameters["MatrixInstruction"][2] = mi_k
    parameters["NonTemporalA"] = hint
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="matrix-instruction depth|cache hint"):
        JG._readRequest(source)


def test_mutated_cache_hint_rejected_before_emission(prediction_request):
    candidate = prediction_request["candidates"][1]
    candidate["parameters"]["NonTemporalB"] = 4
    assert "NonTemporalB" in JG._problemRejection(solution(), prediction_request, candidate)


@pytest.mark.parametrize("output_amax_d", [False, True])
def test_configuration_preserves_amax_without_scaling(prediction_request, output_amax_d):
    prediction_request["problem"].update(output_amax_d=output_amax_d, use_scale_cd=False)
    config = JG._configuration(prediction_request, prediction_request["candidates"][0])
    problem, group = config["BenchmarkProblems"][0]
    assert problem["OutputAmaxD"] is output_amax_d
    assert problem["UseScaleCD"] is False
    assert {next(iter(item)) for item in group["ForkParameters"]} == JG._PARAMETERS


def test_prediction_rejects_scaling_before_generation(prediction_request, tmp_path):
    prediction_request["problem"]["use_scale_cd"] = True
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="C/D scaling"):
        JG._readRequest(source)


@pytest.mark.parametrize("architecture, dtype, mi_k", [
    ("gfx90a", "h", 16), ("gfx942", "h", 16), ("gfx950", "h", 32),
    ("gfx90a", "s", 4), ("gfx942", "s", 4), ("gfx950", "s", 4),
    ("gfx1250", "h", 32), ("gfx1250", "s", 4),
])
def test_architecture_recipe_cross_compiles_without_gpu(
    prediction_request, tmp_path, architecture, dtype, mi_k
):
    """Compile real target instructions; execution on each architecture belongs in GPU CI."""
    compiler = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    if not Path(compiler).is_file():
        pytest.skip("ROCm compiler is unavailable")
    prediction_request["architecture"] = architecture
    p = prediction_request["problem"]
    p.update(m=128, n=128, k=128, batch=1, transpose_a=False, transpose_b=False,
             data_type=dtype, high_precision_accumulate=(dtype == "h"))
    for tensor in "abcd":
        p[f"strides_{tensor}"] = [1, 128, 16384]
    prediction_request["candidates"] = prediction_request["candidates"][:1]
    parameters = prediction_request["candidates"][0]["parameters"]
    parameters.update(MatrixInstruction=[16, 16, mi_k, 1, 1, 1, 1, 2, 2], DepthU=32,
                      NonTemporalA=0, NonTemporalB=0 if architecture in ("gfx90a", "gfx1250") else 4)
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    output = tmp_path / "cross-compiled"
    completed = subprocess.run(
        [sys.executable, "-m", "Tensile.JitGemm", str(source), str(output),
         "--architecture", architecture, "--cxx-compiler", compiler, "--keep-build-tmp"],
        capture_output=True, text=True, timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    bundle = output / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["architecture"]["resolved"] == architecture
    assert manifest["counts"]["solutions"] == manifest["counts"]["main_kernels"] == 1
    prediction = manifest["jit_prediction"]
    assert prediction["selected_parameters"] == parameters
    assert prediction["resolved_parameters"]["MatrixInstruction"][:3] == [16, 16, mi_k]
    assert prediction["resolved_parameters"]["NonTemporalB"] == parameters["NonTemporalB"]
    assert all((bundle / name).is_file() for name in manifest["code_objects"])
    assembly = "\n".join(path.read_text() for path in bundle.rglob("*.s"))
    assert f"--{architecture}" in assembly
    suffix = "f16" if dtype == "h" else "f32"
    # CDNA2 assembly uses the legacy mnemonic without a separating underscore.
    separator = "" if architecture == "gfx90a" else "_"
    kind = "wmma" if architecture == "gfx1250" else "mfma"
    assert f"v_{kind}_f32_16x16x{mi_k}{separator}{suffix}" in assembly
    assert prediction["resolved_parameters"]["WavefrontSize"] == (32 if architecture == "gfx1250" else 64)


def test_amax_batch_predicate_checked_before_compilation(prediction_request):
    prediction_request["problem"]["output_amax_d"] = True
    candidate = prediction_request["candidates"][1]
    assert "BatchSizeEqual=1" in JG._problemRejection(solution(), prediction_request, candidate)
    prediction_request["problem"]["batch"] = 1
    assert JG._problemRejection(solution(), prediction_request, candidate) is None
