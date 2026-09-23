# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
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


def test_rejected_predictions_do_not_add_default_candidates(prediction_request, tmp_path):
    attempted = []

    def reject(config, label):
        attempted.append(label)
        raise SS.SingleSolutionRejected("unsupported supplied recipe")

    output = tmp_path / "selected.yaml"
    with pytest.raises(SS.SingleSolutionConfigError, match="No JIT candidate") as error:
        JG._select(prediction_request, output, reject)
    assert attempted == ["origami.gemm.estimation candidate 7", "origami.gemm.estimation candidate 2"]
    assert "7: unsupported supplied recipe" in str(error.value)
    assert "2: unsupported supplied recipe" in str(error.value)
    assert not output.exists()


def test_additional_tensile_parameters_reach_solution_validation(prediction_request, tmp_path):
    parameters = prediction_request["candidates"][0]["parameters"]
    parameters.update(PrefetchGlobalRead=2, StaggerU=0, GlobalReadVectorWidthA=-1)
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    request = JG._readRequest(source)
    config = JG._configuration(request, request["candidates"][0])
    assert config["BenchmarkProblems"][0][1]["ForkParameters"] == [
        {name: [value]} for name, value in parameters.items()]


def test_empty_default_recipe_is_not_a_prediction(prediction_request, tmp_path):
    prediction_request.update(model="tensile.defaults", candidates=[
        {"id": 0, "predicted_cycles": None, "parameters": {}}])
    source = tmp_path / "request.json"
    source.write_text(json.dumps(prediction_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="supplied tuning parameters"):
        JG._readRequest(source)


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


def solution(depth=64, problem_type=None):
    from Tensile.Common.GlobalParameters import defaultSolution
    from Tensile.SolutionStructs.Problem import ProblemType

    state = copy.deepcopy(defaultSolution)
    state.update({
        "Valid": True, "MacroTile0": 64, "MacroTile1": 64, "DepthU": depth,
        "StreamK": 0, "GlobalSplitU": 1,
        "AssertFree0ElementMultiple": 1, "AssertFree1ElementMultiple": 1,
        "AssertSummationElementMultiple": 1,
        "GlobalReadVectorWidthA": 1, "GlobalReadVectorWidthB": 1,
        "BufferLoad": True, "BufferStore": True,
        "NonTemporalA": 0, "NonTemporalB": 0,
        "MIWaveTile": [2, 2], "MIWaveGroup": [2, 2], "NumThreads": 256,
        "_GlobalAccumulation": None, "PackedC0IndicesX": [0],
        "InternalSupportParams": {"KernArgsVersion": 1},
    })
    state["ProblemType"] = ProblemType(problem_type or {
        "OperationType": "GEMM", "Batched": True, "DataType": "H",
        "DestDataType": "S", "ComputeDataType": "S", "HighPrecisionAccumulate": True,
        "TransposeA": True, "TransposeB": False,
    }, False)
    return state


def problem_rejection(derived, request, candidate=None):
    from Tensile.SolutionStructs.Validators.ProblemSizes import problemSizeRejection

    p = request["problem"]
    return problemSizeRejection(
        derived, [p[key] for key in ("m", "n", "batch", "k")],
        {tensor.upper(): p[f"strides_{tensor}"] for tensor in "abcd"},
        {tensor.upper(): p.get(f"sizes_{tensor}") for tensor in "abcd"})


def test_recipe_preserves_descriptors_and_uses_canonical_defaults(prediction_request):
    config = JG._configuration(prediction_request, prediction_request["candidates"][0])
    problem, group = config["BenchmarkProblems"][0]
    assert problem["TransposeA"] and not problem["TransposeB"]
    exact = group["BenchmarkFinalParameters"][0]["ProblemSizes"][0]["Exact"]
    assert exact["sizes"] == [67, 99, 3, 73]
    assert exact["stridesA"] == [1, 80, 5360]
    assert exact["stridesC"] != exact["stridesD"]
    assert {next(iter(item)) for item in group["ForkParameters"]} == set(prediction_request["candidates"][0]["parameters"])
    assert "BenchmarkCommonParameters" not in group


@pytest.mark.parametrize("latency", [0, -1, float("inf"), float("nan"), sys.float_info.max])
def test_reject_invalid_origami_scores_before_generation(prediction_request, tmp_path, latency):
    prediction_request["candidates"][0]["predicted_cycles"] = latency
    path = tmp_path / "prediction_request.json"
    path.write_text(json.dumps(prediction_request))
    with pytest.raises(SS.SingleSolutionConfigError):
        JG._readRequest(path)


def test_validation_cannot_be_disabled_by_supplied_parameters(prediction_request, tmp_path):
    prediction_request["candidates"][0]["parameters"]["NoReject"] = True
    path = tmp_path / "request.json"
    path.write_text(json.dumps(prediction_request))
    request = JG._readRequest(path)
    config = JG._configuration(request, request["candidates"][0])
    with pytest.raises(SS.SingleSolutionConfigError, match="NoReject cannot disable"):
        SS._singleConfig(config, path)


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
        raise FileNotFoundError("assembler not found")

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
    assert "BoundSizeMultiple" in problem_rejection(derived, prediction_request, candidate)
    derived = solution()
    derived["ProblemType"]["TLUA"] = True
    derived["GlobalReadVectorWidthA"] = 128
    assert "LeadingFree0SizesGreaterOrEqual" in problem_rejection(derived, prediction_request, candidate)


def test_shape_rejection_tries_the_next_supplied_candidate(prediction_request, tmp_path):
    calls = []

    def derive(config, label):
        calls.append(label)
        derived = solution()
        if len(calls) == 1:
            derived["ProblemType"]["TLUA"] = True
            derived["GlobalReadVectorWidthA"] = 128
        return derived

    _, _, metadata = JG._select(prediction_request, tmp_path / "selected.yaml", derive)
    assert len(calls) == 2
    assert metadata["candidate_id"] == 2
    assert "LeadingFree0SizesGreaterOrEqual" in metadata["rejections"][0]["reason"]


@pytest.mark.parametrize("kernargs,streamk,gsu,rejected", [
    (1, 0, 1, False), (1, 0, 2, True), (1, 0, -1, False),
    (1, 1, 2, False), (0, 0, 2, False),
])
def test_workgroup_limit_uses_shared_predicate_gating(prediction_request, kernargs, streamk, gsu, rejected):
    prediction_request["problem"].update(m=64 * 2**24, n=64, k=512, batch=1)
    derived = solution()
    derived.update(StreamK=streamk, GlobalSplitU=gsu, BufferLoad=False, BufferStore=False)
    derived["InternalSupportParams"]["KernArgsVersion"] = kernargs
    reason = problem_rejection(derived, prediction_request)
    assert (reason is not None) is rejected
    if rejected:
        assert "WorkgroupNumberCheck" in reason


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


def test_solution_can_resolve_parameter_sentinels(prediction_request, tmp_path):
    prediction_request["candidates"] = prediction_request["candidates"][:1]
    prediction_request["candidates"][0]["parameters"]["GlobalReadVectorWidthA"] = -1
    _, _, metadata = JG._select(prediction_request, tmp_path / "selected.yaml", lambda *_: solution())
    assert metadata["selected_parameters"]["GlobalReadVectorWidthA"] == -1
    assert metadata["resolved_parameters"]["GlobalReadVectorWidthA"] == 1


@pytest.mark.parametrize("output_amax_d", [False, True])
def test_configuration_preserves_amax_without_scaling(prediction_request, output_amax_d):
    prediction_request["problem"].update(output_amax_d=output_amax_d, use_scale_cd=False)
    config = JG._configuration(prediction_request, prediction_request["candidates"][0])
    problem, group = config["BenchmarkProblems"][0]
    assert problem["OutputAmaxD"] is output_amax_d
    assert problem["UseScaleCD"] is False
    assert {next(iter(item)) for item in group["ForkParameters"]} == set(prediction_request["candidates"][0]["parameters"])


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
    derived = solution(problem_type=JG._problemType(prediction_request))
    derived["BatchSizeEqual"] = 1  # Set by Solution for OutputAmaxD's reduction.
    assert "BatchSizeEqual=1" in problem_rejection(derived, prediction_request, candidate)
    prediction_request["problem"]["batch"] = 1
    assert problem_rejection(derived, prediction_request, candidate) is None


@pytest.fixture
def problem_type_request(prediction_request):
    """Request with independent storage/arithmetic types in Tensile's ProblemType mapping."""
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
def test_problem_type_request_preserves_independent_types(
    problem_type_request, tmp_path, a, b, d, compute, hpa, enum_values
):
    from Tensile.Common.DataType import DataType

    pt = problem_type_request["problem_type"]
    pt.update(DataType=a, DataTypeA=a, DataTypeB=b, MacDataTypeA=a, MacDataTypeB=b,
              DestDataType=d, ComputeDataType=compute, HighPrecisionAccumulate=hpa)
    if enum_values:
        for key in ("DataType", "DataTypeA", "DataTypeB", "MacDataTypeA", "MacDataTypeB",
                    "DestDataType", "ComputeDataType"):
            pt[key] = DataType(pt[key]).value
    path = tmp_path / "request.json"
    path.write_text(json.dumps(problem_type_request))
    request = JG._readRequest(path)
    config = JG._configuration(request, request["candidates"][0])
    assert config["BenchmarkProblems"][0][0] == pt


def test_problem_type_epilogue_preserves_operation_and_required_arguments(problem_type_request, tmp_path):
    pt = problem_type_request["problem_type"]
    pt.update(UseScaleAB="Scalar", UseScaleCD=True, UseScaleAlphaVec=1,
              UseBias=1, BiasDataTypeList=["S"], BiasSrc="D",
              Activation=True, ActivationType="hipblaslt_all",
              ActivationComputeDataType="S", ActivationHPA=True,
              UseE=True, DataTypeE="S", Gradient=True,
              OutputAmaxD=True, DataTypeAmaxD="S")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(problem_type_request))
    request = JG._readRequest(path)
    config = JG._configuration(request, request["candidates"][0])
    actual, group = config["BenchmarkProblems"][0]
    assert actual == pt
    final = {key: value for entry in group["BenchmarkFinalParameters"]
             for key, value in entry.items()}
    assert final["BiasTypeArgs"] == ["S"]
    assert final["ActivationArgs"] == [[{"Enum": "none"}]]
    assert "stridesE" not in final["ProblemSizes"][0]["Exact"]


def test_problem_type_structure_cannot_disagree_with_descriptors(problem_type_request, tmp_path):
    problem_type_request["problem_type"]["TransposeA"] = False
    path = tmp_path / "request.json"
    path.write_text(json.dumps(problem_type_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="TransposeA disagrees"):
        JG._readRequest(path)


def test_buffer_predicate_uses_output_type_and_physical_extent(problem_type_request):
    p = problem_type_request["problem"]
    p["strides_d"][1] = 2**24
    candidate = problem_type_request["candidates"][1]
    # FP16 inputs remain addressable, but this FP32 output reaches the 4 GiB limit.
    assert "BufferStoreOffsetLimitCheck" in problem_rejection(solution(), problem_type_request, candidate)
    p["sizes_d"] = [67, 32, 3]
    assert problem_rejection(solution(), problem_type_request, candidate) is None


@pytest.mark.parametrize("zero", ["m", "n", "k"])
def test_zero_dimensions_preserve_real_extents(problem_type_request, tmp_path, zero):
    request = problem_type_request
    request["candidates"] = request["candidates"][:1]
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
    assert metadata["model"] == request["model"]
    assert metadata["predicted_cycles"] == request["candidates"][0]["predicted_cycles"]


def test_zero_k_still_checks_beta_c_output_bounds(problem_type_request):
    problem_type_request["problem"]["k"] = 0
    derived = solution()
    derived["ProblemType"]["TLUA"] = True
    derived["GlobalReadVectorWidthA"] = 128
    candidate = problem_type_request["candidates"][1]
    assert problem_rejection(derived, problem_type_request, candidate) is None
    problem_type_request["problem"]["strides_d"][1] = 2**24
    assert "BufferStoreOffsetLimitCheck" in problem_rejection(derived, problem_type_request, candidate)


@pytest.mark.parametrize("field", ["m", "n", "k"])
def test_negative_dimensions_remain_invalid(problem_type_request, tmp_path, field):
    problem_type_request["problem"][field] = -1
    path = tmp_path / "request.json"
    path.write_text(json.dumps(problem_type_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="nonnegative"):
        JG._readRequest(path)


@pytest.fixture
def mx_layout_request(problem_type_request):
    request = problem_type_request
    request.update(model="caller.parameters")
    request["problem_type"].update(
        DataType="F4", DataTypeA="F4", DataTypeB="F4", MacDataTypeA="F4", MacDataTypeB="F4",
        MXBlockA=32, MXBlockB=32, DataTypeMXSA="E8", DataTypeMXSB="E8")
    request["problem"].update(
        mx_scale_format="HostPreSwizzle", scale_mode_a="Block_32_UE8M0_32_8_EXT",
        scale_mode_b="Block_32_UE8M0_32_8_EXT")
    request["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": {"StaggerU": 0}}]
    return request


@pytest.mark.parametrize("architecture,layout,mode", [
    ("gfx950:xnack-", "HostPreSwizzle", "Block_32_UE8M0_32_8_EXT"),
    ("gfx1250", "InMemorySwizzle", "Block_32_UE8M0"),
])
@pytest.mark.parametrize("omit_latency", [False, True])
def test_descriptor_layout_is_preserved_separately_from_tuning(
    mx_layout_request, tmp_path, architecture, layout, mode, omit_latency
):
    request = mx_layout_request
    if omit_latency:
        del request["candidates"][0]["predicted_cycles"]
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
        assert group["ForkParameters"] == [{"StaggerU": [0]}, {"MXScaleFormat": [layout]}]
        return {**solution(), "MXScaleFormat": layout}

    selected, _, metadata = JG._select(request, tmp_path / "selected.yaml", derive)
    assert len(calls) == 1
    assert yaml.safe_load(selected.read_text()) == calls[0]
    assert metadata["problem"] == request["problem"]
    assert metadata["selected_parameters"] == {"StaggerU": 0}
    assert metadata["predicted_cycles"] is None
    assert metadata["implementation_parameters"] == {"MXScaleFormat": layout}
    assert "MXScaleFormat" not in metadata["default_parameters"]
    assert "MXScaleFormat" not in metadata["derived_parameters"]
    assert metadata["resolved_parameters"]["MXScaleFormat"] == layout


@pytest.mark.parametrize("layout", ["Auto", None])
def test_descriptor_layout_must_be_explicit(mx_layout_request, tmp_path, layout):
    mx_layout_request["problem"]["mx_scale_format"] = layout
    source = tmp_path / "request.json"
    source.write_text(json.dumps(mx_layout_request))
    with pytest.raises(SS.SingleSolutionConfigError, match="explicit MX scale layout"):
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
    reason = JG._descriptorRejection(derived, mx_layout_request)
    assert "changed the descriptor MXScaleFormat=HostPreSwizzle" in reason


def test_old_mx_schema_retains_default_layout_derivation(mx_layout_request):
    del mx_layout_request["problem"]["mx_scale_format"]
    del mx_layout_request["problem"]["scale_mode_a"]
    del mx_layout_request["problem"]["scale_mode_b"]
    assert JG._implementationParameters(mx_layout_request) == {}
    config = JG._configuration(mx_layout_request, mx_layout_request["candidates"][0])
    assert config["BenchmarkProblems"][0][1]["ForkParameters"] == [{"StaggerU": [0]}]


@pytest.mark.parametrize("architecture, mode, expected", [
    ("gfx950", "Block_32_UE8M0_32_8_EXT", "HostPreSwizzle"),
    ("gfx950", "Block_32_UE8M0", "NoSwizzle"),
    ("gfx1250", "Block_32_UE8M0", "InMemorySwizzle"),
])
def test_descriptor_scale_modes_derive_provider_layout(mx_layout_request, architecture, mode, expected):
    mx_layout_request["architecture"] = architecture
    del mx_layout_request["problem"]["mx_scale_format"]
    mx_layout_request["problem"].update(scale_mode_a=mode, scale_mode_b=mode)
    assert JG._implementationParameters(mx_layout_request) == {"MXScaleFormat": expected}


def test_descriptor_mixed_physical_layouts_are_rejected(mx_layout_request):
    del mx_layout_request["problem"]["mx_scale_format"]
    mx_layout_request["problem"]["scale_mode_b"] = "Block_32_UE8M0"
    with pytest.raises(SS.SingleSolutionConfigError, match="different MX layouts"):
        JG._implementationParameters(mx_layout_request)


def compile_request(request, tmp_path):
    compiler = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    if not Path(compiler).is_file():
        pytest.skip("ROCm compiler is unavailable")
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request))
    output = tmp_path / "compiled"
    completed = subprocess.run(
        [sys.executable, "-m", "Tensile.JitGemm", str(source), str(output),
         "--architecture", request["architecture"], "--cxx-compiler", compiler],
        capture_output=True, text=True, timeout=240, cwd=tmp_path)
    (tmp_path / "compiler.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "bundle/manifest.json").read_text())
    assert manifest["counts"]["solutions"] == manifest["counts"]["main_kernels"] == 1
    assert all((output / "bundle" / name).is_file() for name in manifest["code_objects"])
    return manifest


def test_shared_instruction_validator_rejects_first_candidate_before_build(prediction_request, tmp_path):
    request = prediction_request
    p = request["problem"]
    p.update(m=128, n=128, k=512, batch=1)
    for tensor in "abcd":
        p[f"strides_{tensor}"] = [1, 512 if tensor in "ab" else 128, 65536]
    # K=4 is a catalog instruction, but does not implement FP16 arithmetic.
    request["candidates"][0]["parameters"].update(
        MatrixInstruction=[16, 16, 4, 1, 1, 2, 2, 2, 2], DepthU=32)
    request["candidates"][1]["parameters"].update(
        MatrixInstruction=[16, 16, 32, 1, 1, 2, 2, 2, 2], DepthU=32)
    manifest = compile_request(request, tmp_path)
    selected = manifest["jit_prediction"]
    assert selected["candidate_id"] == 2
    assert [rejection["candidate_id"] for rejection in selected["rejections"]] == [7]
    assert selected["selected_parameters"] == request["candidates"][1]["parameters"]


@pytest.mark.parametrize("a,b,scale,mode", [
    ("F8", "F6", "E8", "Block_32_UE8M0"),
    ("F6", "F8", "E8", "Block_32_UE8M0"),
    ("F4", "F4", "E8", "Block_32_UE8M0"),
    ("F4", "F4", "F8", "Block_32_UE4M3"),
    ("F4", "F4", "E5M3", "Block_32_UE5M3"),
])
def test_mx_operand_and_scale_types_reuse_one_tuning_recipe_cross_compiles(problem_type_request, tmp_path, a, b, scale, mode):
    """Reuse gfx1250 MX tile/transport tuning; compile every legal type swap.

    The common tile/transport recipe comes from the gfx12 MX TDM family.
    The -1 vector widths let Solution derive the widths required by each type.
    FP8/FP6 requires E8 scales; FP4 supports matching E8, E4M3 or E5M3 scales.
    These are compiler checks, not measurements of numerical accuracy or speed.
    """
    request = problem_type_request
    request.update(architecture="gfx1250", model="caller.shared-mx-recipe")
    request["problem_type"].update(
        DataType=a, DataTypeA=a, DataTypeB=b, MacDataTypeA=a, MacDataTypeB=b,
        MXBlockA=32, MXBlockB=32, DataTypeMXSA=scale, DataTypeMXSB=scale)
    p = request["problem"]
    p.update(m=128, n=128, k=512, batch=1, mx_scale_format="InMemorySwizzle",
             scale_mode_a=mode, scale_mode_b=mode)
    for tensor in "abcd":
        p[f"strides_{tensor}"] = [1, 512 if tensor in "ab" else 128, 65536]
    parameters = {
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": 128,
        "TDMInst": 3, "ScheduleIterAlg": 4, "LocalReadVectorWidth": -1,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "VectorWidthA": 1, "VectorWidthB": 1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1,
        "UseSgprForGRO": 0, "ForceDisableShadowInit": True,
    }
    request["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": parameters}]
    manifest = compile_request(request, tmp_path)
    selected = manifest["jit_prediction"]
    assert selected["selected_parameters"] == parameters
    assert selected["problem_type"] == request["problem_type"]
    assert selected["implementation_parameters"] == {"MXScaleFormat": "InMemorySwizzle"}
    assert selected["rejections"] == []
