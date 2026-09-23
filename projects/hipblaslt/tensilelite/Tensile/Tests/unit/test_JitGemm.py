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

    with pytest.raises(SS.SingleSolutionConfigError, match="No JIT candidate"):
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
    ("gfx90a", "h", 16, 4), ("gfx90a", "s", 4, 4),
    ("gfx1250", "h", 32, 4), ("gfx1250", "s", 4, 4),
])
def test_request_rejects_architecture_incompatible_cache_hint(
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
    with pytest.raises(SS.SingleSolutionConfigError, match="cache hint"):
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


def test_legacy_scaling_request_preserves_supported_feature(prediction_request, tmp_path):
    prediction_request["problem"]["use_scale_cd"] = True
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    request = JG._readRequest(source)
    assert JG._problemType(request)["UseScaleCD"] is True


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


@pytest.fixture
def canonical_request(prediction_request):
    p = prediction_request["problem"]
    p.pop("data_type")
    p.pop("high_precision_accumulate")
    prediction_request["problem_type"] = {
        "OperationType": "GEMM", "Batched": True, "StridedBatched": True,
        "TransposeA": p["transpose_a"], "TransposeB": p["transpose_b"],
        "DataType": "H", "DataTypeA": "H", "DataTypeB": "H",
        "MacDataTypeA": "H", "MacDataTypeB": "H",
        "DestDataType": "S", "ComputeDataType": "S", "UseBeta": True,
        "HighPrecisionAccumulate": True,
    }
    return prediction_request


@pytest.mark.parametrize("a,b,d,compute,hpa", [
    ("H", "H", "S", "S", True), ("B", "B", "S", "S", True),
    ("I8", "I8", "I", "I", True), ("F8", "B8", "S", "S", True),
    ("D", "D", "D", "D", False), ("C", "C", "C", "C", False),
    ("Z", "Z", "Z", "Z", False),
])
@pytest.mark.parametrize("enum_values", [False, True])
def test_canonical_request_preserves_independent_types(
    canonical_request, tmp_path, a, b, d, compute, hpa, enum_values
):
    from Tensile.Common.DataType import DataType

    pt = canonical_request["problem_type"]
    pt.update(DataType=a, DataTypeA=a, DataTypeB=b, MacDataTypeA=a, MacDataTypeB=b,
              DestDataType=d, ComputeDataType=compute, HighPrecisionAccumulate=hpa)
    if enum_values:
        for key in ("DataType", "DataTypeA", "DataTypeB", "MacDataTypeA", "MacDataTypeB",
                    "DestDataType", "ComputeDataType"):
            pt[key] = DataType(pt[key]).value
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    request = JG._readRequest(path)
    config = JG._configuration(request, request["candidates"][0])
    assert config["BenchmarkProblems"][0][0] == pt


def test_canonical_epilogue_preserves_operation_and_required_arguments(canonical_request, tmp_path):
    pt = canonical_request["problem_type"]
    pt.update(UseScaleAB="Scalar", UseScaleCD=True, UseScaleAlphaVec=1,
              UseBias=1, BiasDataTypeList=["S"], BiasSrc="D",
              Activation=True, ActivationType="hipblaslt_all",
              ActivationComputeDataType="S", ActivationHPA=True,
              UseE=True, DataTypeE="S", Gradient=True,
              OutputAmaxD=True, DataTypeAmaxD="S")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    request = JG._readRequest(path)
    config = JG._configuration(request, request["candidates"][0])
    actual, group = config["BenchmarkProblems"][0]
    assert actual == pt
    final = {key: value for entry in group["BenchmarkFinalParameters"]
             for key, value in entry.items()}
    assert final["BiasTypeArgs"] == ["S"]
    assert final["ActivationArgs"] == [[{"Enum": "none"}]]
    assert "stridesE" not in final["ProblemSizes"][0]["Exact"]


def test_canonical_structure_cannot_disagree_with_descriptors(canonical_request, tmp_path):
    canonical_request["problem_type"]["TransposeA"] = False
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="TransposeA disagrees"):
        JG._readRequest(path)


def test_buffer_predicate_uses_output_type_and_physical_extent(canonical_request):
    p = canonical_request["problem"]
    p["strides_d"][1] = 2**24
    candidate = canonical_request["candidates"][1]
    # FP16 inputs remain addressable, but this FP32 output reaches the 4 GiB limit.
    assert "Tensor D" in JG._problemRejection(solution(), canonical_request, candidate)
    p["sizes_d"] = [67, 32, 3]
    assert JG._problemRejection(solution(), canonical_request, candidate) is None


def test_native_instruction_legality_is_normal_candidate_rejection(canonical_request, tmp_path):
    canonical_request["candidates"][0]["parameters"]["MatrixInstruction"][:4] = [7, 7, 7, 1]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    request = JG._readRequest(path)
    calls = []

    def derive(config, label):
        calls.append(label)
        return solution()

    _, _, metadata = JG._select(request, tmp_path / "selected.yaml", derive)
    assert len(calls) == 1
    assert metadata["candidate_id"] == 2
    assert "absent from Tensile's catalog" in metadata["rejections"][0]["reason"]


def test_model_exhaustion_falls_back_without_fabricated_prediction(canonical_request, tmp_path):
    calls = []

    def derive(config, label):
        pt, group = config["BenchmarkProblems"][0]
        calls.append(pt)
        assert pt == canonical_request["problem_type"]
        if group["ForkParameters"]:
            raise SS.SingleSolutionRejected("unsupported modeled instruction recipe")
        return solution()

    path, _, metadata = JG._select(canonical_request, tmp_path / "selected.yaml", derive)
    assert len(calls) == 3
    assert metadata["model"] == "tensile.defaults"
    assert metadata["predicted_cycles"] is None
    assert metadata["selected_parameters"] == {}
    assert metadata["problem_type"] == canonical_request["problem_type"]
    assert metadata["origami_candidates"] == canonical_request["candidates"]
    assert len(metadata["origami_rejections"]) == 2
    assert "fallback_reason" in metadata["model_assumptions"]
    assert "no latency prediction" in metadata["summary"]
    assert path.is_file()


def test_native_defaults_use_bounded_catalog_without_scores(canonical_request, tmp_path):
    from Tensile.Common.ValidParameters import makeValidMatrixInstructions

    canonical_request["model"] = "tensile.defaults"
    canonical_request["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": {}}]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    request = JG._readRequest(path)
    candidates = JG._defaultCandidates(request)
    assert 1 < len(candidates) <= JG._MAX_CANDIDATES
    assert candidates[0]["parameters"] == {}
    native = {tuple(mi) for mi in makeValidMatrixInstructions() if len(mi) == 4}
    for candidate in candidates[1:]:
        assert candidate["predicted_cycles"] is None
        assert tuple(candidate["parameters"]["MatrixInstruction"][:4]) in native
    request["candidates"][0]["predicted_cycles"] = 1.0
    path.write_text(json.dumps(request))
    with pytest.raises(SS.SingleSolutionConfigError, match="no predicted latency"):
        JG._readRequest(path)


@pytest.mark.parametrize("a,b", [("F4", "F4"), ("F8", "F8"), ("F8", "F4"), ("F4", "F8")])
@pytest.mark.parametrize("architecture,requirements,layout", [
    ("gfx950", {"UseSubtileImpl": True, "LocalReadVectorWidth": 32}, "HostPreSwizzle"),
    ("gfx1250", {"TDMInst": 3, "ScheduleIterAlg": 4}, "InMemorySwizzle"),
])
def test_mx_native_fallback_records_required_implementation(
    canonical_request, tmp_path, architecture, requirements, layout, a, b
):
    canonical_request.update(architecture=architecture, model="tensile.defaults")
    canonical_request["problem_type"].update(
        DataType=a, DataTypeA=a, DataTypeB=b, MacDataTypeA=a, MacDataTypeB=b,
        MXBlockA=32, MXBlockB=32, DataTypeMXSA="E8", DataTypeMXSB="E8")
    canonical_request["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": {}}]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    request = JG._readRequest(path)
    candidates = JG._defaultCandidates(request)
    assert 1 < len(candidates) <= JG._MAX_CANDIDATES
    for candidate in candidates[1:]:
        assert candidate["predicted_cycles"] is None
        assert candidate["parameters"].items() >= requirements.items()
        assert layout in candidate["default_recipe_reason"]
        mi = candidate["parameters"]["MatrixInstruction"]
        assert mi[:4] in ([16, 16, 128, 1], [32, 32, 64, 1])
        if architecture == "gfx950":
            assert candidate["parameters"]["DepthU"] == 2 * mi[2]
        config = JG._configuration(request, candidate)
        assert config["BenchmarkProblems"][0][0] == canonical_request["problem_type"]
    derived = solution()
    derived["UseSubtileImpl"] = False
    changed = {"parameters": {"UseSubtileImpl": True}}
    assert "UseSubtileImpl" in JG._problemRejection(derived, canonical_request, changed)


@pytest.mark.parametrize("zero", ["m", "n", "k"])
def test_zero_dimensions_preserve_real_extents_and_unmodeled_selection(canonical_request, tmp_path, zero):
    request = canonical_request
    request["model"] = "tensile.defaults"
    request["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": {}}]
    p = request["problem"]
    p[zero] = 0
    p["sizes_a"] = [p["k"], p["m"], p["batch"]]
    p["sizes_b"] = [p["k"], p["n"], p["batch"]]
    p["sizes_c"] = p["sizes_d"] = [p["m"], p["n"], p["batch"]]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    request = JG._readRequest(path)
    selected, _, metadata = JG._select(request, tmp_path / "selected.yaml", lambda *_: solution())
    exact = yaml.safe_load(selected.read_text())["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"][0]["Exact"]
    assert exact["sizes"] == [p[key] for key in ("m", "n", "batch", "k")]
    assert metadata["problem"] == p
    assert metadata["model"] == "tensile.defaults"
    assert metadata["predicted_cycles"] is None


def test_zero_k_still_checks_beta_c_output_bounds(canonical_request):
    canonical_request["problem"]["k"] = 0
    derived = solution()
    derived["ProblemType"]["TLUA"] = True
    derived["GlobalReadVectorWidthA"] = 128
    candidate = canonical_request["candidates"][1]
    assert JG._problemRejection(derived, canonical_request, candidate) is None
    canonical_request["problem"]["strides_d"][1] = 2**24
    assert "Tensor D" in JG._problemRejection(derived, canonical_request, candidate)


@pytest.mark.parametrize("field", ["m", "n", "k"])
def test_negative_dimensions_remain_invalid(canonical_request, tmp_path, field):
    canonical_request["problem"][field] = -1
    path = tmp_path / "request.json"
    path.write_text(json.dumps(canonical_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="nonnegative"):
        JG._readRequest(path)


@pytest.fixture
def mx_layout_request(canonical_request):
    request = canonical_request
    request.update(model="tensile.defaults")
    request["problem_type"].update(
        DataType="F4", DataTypeA="F4", DataTypeB="F4", MacDataTypeA="F4", MacDataTypeB="F4",
        MXBlockA=32, MXBlockB=32, DataTypeMXSA="E8", DataTypeMXSB="E8")
    request["problem"].update(
        mx_scale_format="HostPreSwizzle", scale_mode_a="Block_32_UE8M0_32_8_EXT",
        scale_mode_b="Block_32_UE8M0_32_8_EXT")
    request["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": {}}]
    return request


@pytest.mark.parametrize("architecture,layout,mode", [
    ("gfx950:xnack-", "HostPreSwizzle", "Block_32_UE8M0_32_8_EXT"),
    ("gfx1250", "InMemorySwizzle", "Block_32_UE8M0"),
])
def test_descriptor_layout_is_preserved_separately_from_tuning(
    mx_layout_request, tmp_path, architecture, layout, mode
):
    request = mx_layout_request
    request["architecture"] = architecture
    request["problem"].update(mx_scale_format=layout, scale_mode_a=mode, scale_mode_b=mode)
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request))
    request = JG._readRequest(source)
    calls = []

    def derive(config, label):
        calls.append(config)
        pt, group = config["BenchmarkProblems"][0]
        assert "MXScaleFormat" not in pt
        assert group["ForkParameters"] == [{"MXScaleFormat": [layout]}]
        return {**solution(), "MXScaleFormat": layout}

    selected, _, metadata = JG._select(request, tmp_path / "selected.yaml", derive)
    assert len(calls) == 1
    assert yaml.safe_load(selected.read_text()) == calls[0]
    assert metadata["problem"] == request["problem"]
    assert metadata["selected_parameters"] == {}
    assert metadata["predicted_cycles"] is None
    assert metadata["implementation_parameters"] == {"MXScaleFormat": layout}
    assert "MXScaleFormat" not in metadata["default_parameters"]
    assert "MXScaleFormat" not in metadata["derived_parameters"]
    assert metadata["resolved_parameters"]["MXScaleFormat"] == layout


@pytest.mark.parametrize("architecture,layout,reason", [
    ("gfx950", "NoSwizzle", "shared subtile generator"),
    ("gfx950", "InMemorySwizzle", "requires gfx1250"),
    ("gfx1250", "HostPreSwizzle", "requires gfx950"),
    ("gfx1250", "NoSwizzle", "requires MXScaleFormat=InMemorySwizzle"),
    ("gfx942", "HostPreSwizzle", "requires gfx950"),
    ("gfx942", "InMemorySwizzle", "requires gfx1250"),
    ("gfx950", "Auto", "explicit MX scale layout"),
    ("gfx950", None, "explicit MX scale layout"),
])
def test_unsupported_descriptor_layout_rejected_before_derivation(
    mx_layout_request, tmp_path, architecture, layout, reason
):
    mx_layout_request["architecture"] = architecture
    mx_layout_request["problem"]["mx_scale_format"] = layout
    source = tmp_path / "request.json"
    source.write_text(json.dumps(mx_layout_request))
    with pytest.raises(SS.SingleSolutionConfigError, match=reason):
        JG._readRequest(source)


@pytest.mark.parametrize("mode", [9, 1001, "unknown"])
def test_raw_scale_modes_must_use_unambiguous_names(mx_layout_request, tmp_path, mode):
    mx_layout_request["problem"]["scale_mode_a"] = mode
    source = tmp_path / "request.json"
    source.write_text(json.dumps(mx_layout_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="name a descriptor ScalingFormat"):
        JG._readRequest(source)


@pytest.mark.parametrize("tensor", ["a", "b"])
def test_natural_and_preswizzled_scales_cannot_be_reinterpreted(mx_layout_request, tmp_path, tensor):
    mx_layout_request["problem"][f"scale_mode_{tensor}"] = "Block_32_UE8M0"
    source = tmp_path / "request.json"
    source.write_text(json.dumps(mx_layout_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="disagrees with MXScaleFormat"):
        JG._readRequest(source)


@pytest.mark.parametrize("actual", ["NoSwizzle", "InMemorySwizzle", None])
def test_derived_layout_change_rejected_before_emission(mx_layout_request, actual):
    derived = solution()
    if actual is not None:
        derived["MXScaleFormat"] = actual
    reason = JG._problemRejection(derived, mx_layout_request, mx_layout_request["candidates"][0])
    assert "changed the descriptor MXScaleFormat=HostPreSwizzle" in reason


def test_old_mx_schema_retains_default_layout_derivation(mx_layout_request):
    del mx_layout_request["problem"]["mx_scale_format"]
    assert JG._implementationParameters(mx_layout_request) == {}
    config = JG._configuration(mx_layout_request, mx_layout_request["candidates"][0])
    assert config["BenchmarkProblems"][0][1]["ForkParameters"] == []
