# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate parameter recipes and build one solution without benchmarking.

Origami runs in the C++ caller. This driver retains its ordering, rejects invalid
recipes before code emission, and uses SingleSolution's normal toolchain/build.
Only caller-supplied recipes are considered. Exhausting them fails the request
with the candidate rejection reasons.
Explicit-YAML callers continue to use Tensile.SingleSolution directly.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import sys
from pathlib import Path

from Tensile import SingleSolution as SS


_DEFAULTS_SOURCE = "Tensile/Common/GlobalParameters.py:defaultBenchmarkCommonParameters"
_MAX_CANDIDATES = 192


def _require(condition, message):
    if not condition:
        raise SS.SingleSolutionConfigError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _implementationParameters(request):
    """Keep physical scale layout separate from modeled tuning parameters."""
    problem = request["problem"]
    modes = {"None", "Scalar", "Vector", "Block_32_UE8M0", "Block_16_UE8M0",
             "Block_32_UE4M3", "Block_16_UE4M3", "Block_32_UE5M3", "Block_16_UE5M3",
             "Block_32_UE8M0_32_8_EXT"}
    for tensor in "ab":
        key = f"scale_mode_{tensor}"
        if key in problem:
            _require(isinstance(problem[key], str) and problem[key] in modes,
                     f"problem.{key} must name a descriptor ScalingFormat")
    # Earlier schema-1 requests omitted physical layout. Preserve their default
    # derivation; descriptor callers now provide the format before type lowering.
    if "mx_scale_format" not in problem:
        return {}
    layout = problem["mx_scale_format"]
    _require(isinstance(layout, str) and layout in
             ("NoSwizzle", "HostPreSwizzle", "InMemorySwizzle"),
             "problem.mx_scale_format must name an explicit MX scale layout")
    problemType = _problemType(request)
    _require(problemType.get("MXBlockA") or problemType.get("MXBlockB"),
             "problem.mx_scale_format requires an MX-scaled operand")
    architecture = request["architecture"].split(":", 1)[0]
    for tensor in "ab":
        mode = problem.get(f"scale_mode_{tensor}", "None")
        if mode.startswith("Block_"):
            expected = ("HostPreSwizzle" if mode == "Block_32_UE8M0_32_8_EXT" else
                        "InMemorySwizzle" if architecture == "gfx1250" else "NoSwizzle")
            _require(layout == expected,
                     f"problem.scale_mode_{tensor}={mode} disagrees with MXScaleFormat={layout}")
    return {"MXScaleFormat": layout}


def _problemType(request):
    """Preserve the caller's Tensile ProblemType mapping.

    Schema-1 requests emitted before problem_type was added described matching
    input/output types with a few scalar fields. Keep those requests readable;
    new callers send the same ProblemType mapping as an explicit Tensile YAML.
    """
    p = request["problem"]
    canonical = request.get("problem_type")
    if canonical is None:
        canonical = {
            "DataType": p["data_type"], "DestDataType": p["data_type"],
            "ComputeDataType": "s",
            "HighPrecisionAccumulate": p["high_precision_accumulate"],
            "UseBeta": True, "UseBias": 0, "Activation": False,
            "ActivationType": "none",
            "OutputAmaxD": p.get("output_amax_d", False),
            "UseScaleCD": p.get("use_scale_cd", False),
        }
    _require(isinstance(canonical, dict), "problem_type must be a Tensile ProblemType mapping")
    canonical = copy.deepcopy(canonical)
    structural = {
        "OperationType": "GEMM", "Batched": True, "StridedBatched": True,
        "TransposeA": p["transpose_a"], "TransposeB": p["transpose_b"],
    }
    for name, value in structural.items():
        _require(canonical.get(name, value) == value,
                 f"problem_type.{name} disagrees with the GEMM descriptors")
        canonical[name] = value
    _require("DataType" in canonical, "problem_type.DataType is required by Tensile")
    return canonical


def _readRequest(path):
    def rejectConstant(value):
        raise SS.SingleSolutionConfigError(f"Nonfinite JSON value: {value}")

    def uniquePairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        request = json.load(stream, parse_constant=rejectConstant, object_pairs_hook=uniquePairs)
    _require(isinstance(request, dict) and request.get("schema_version") == 1,
             "Expected JIT parameter request schema 1")
    _require(isinstance(request.get("model"), str) and request["model"],
             "Missing prediction model name")
    _require(isinstance(request.get("architecture"), str), "Missing target architecture")
    SS._target(request["architecture"], {})
    problem = request.get("problem")
    _require(isinstance(problem, dict), "Missing problem descriptors")
    for key in ("m", "n", "k"):
        _require(_integer(problem.get(key)), f"problem.{key} must be a nonnegative integer")
    _require(_integer(problem.get("batch"), 1), "problem.batch must be a positive integer")
    for key in ("transpose_a", "transpose_b", "c_equals_d"):
        _require(type(problem.get(key)) is bool, f"problem.{key} must be boolean")
    if "problem_type" not in request:
        for key in ("output_amax_d", "use_scale_cd"):
            problem.setdefault(key, False)
        for key in ("high_precision_accumulate", "output_amax_d", "use_scale_cd"):
            _require(type(problem.get(key)) is bool, f"problem.{key} must be boolean")
    _problemType(request)
    _implementationParameters(request)
    for tensor in "abcd":
        strides = problem.get(f"strides_{tensor}")
        _require(isinstance(strides, list) and len(strides) == 3
                 and all(_integer(value) for value in strides) and strides[0] == 1,
                 f"Expected three nonnegative strides with unit stride[0] for tensor {tensor}")
        sizes = problem.get(f"sizes_{tensor}")
        _require(sizes is None or (isinstance(sizes, list) and len(sizes) == 3
                 and all(_integer(value) for value in sizes)),
                 f"Expected nonnegative tensor extents for tensor {tensor}")
    candidates = request.get("candidates")
    _require(isinstance(candidates, list) and 0 < len(candidates) <= _MAX_CANDIDATES,
             f"Expected 1 through {_MAX_CANDIDATES} ranked candidates")
    ids = set()
    for candidate in candidates:
        _require(isinstance(candidate, dict), "Candidate must be a mapping")
        identifier = candidate.get("id")
        _require(_integer(identifier) and identifier not in ids, "Invalid/duplicate candidate id")
        ids.add(identifier)
        latency = candidate.get("predicted_cycles")
        _require(latency is None or (type(latency) in (int, float) and math.isfinite(latency)
                 and 0 < latency < sys.float_info.max), "Invalid predicted latency")
        _require(isinstance(candidate.get("parameters"), dict) and candidate["parameters"],
                 "A JIT candidate requires supplied tuning parameters; no default recipe is selected")
        # Parameter names, types, values, and coupled constraints belong to
        # BenchmarkProcess/Solution, just as they do for an explicit YAML recipe.

    return request


def _configuration(request, candidate):
    problem = request["problem"]
    exact = {"sizes": [problem[key] for key in ("m", "n", "batch", "k")]}
    for tensor in "abcd":
        exact[f"strides{tensor.upper()}"] = list(problem[f"strides_{tensor}"])
    problemType = _problemType(request)
    final = [{"ProblemSizes": [{"Exact": exact}]}]
    if problemType.get("UseBias"):
        from Tensile.SolutionStructs.Problem import ProblemType

        # BenchmarkProcess validates its argument lists even though this path
        # only derives a solution. Use the normal problem type's bias whitelist.
        biasTypes = ProblemType(problemType, False)["BiasDataTypeList"]
        final.append({"BiasTypeArgs": [value.toChar() for value in biasTypes]})
    if problemType.get("ActivationType") in ("all", "hipblaslt_all"):
        final.append({"ActivationArgs": [[{"Enum": "none"}]]})
    # Physical layout comes from descriptors, independently of candidate tuning.
    parameters = {**candidate["parameters"], **_implementationParameters(request)}
    return {
        "GlobalParameters": {
            "PrintLevel": 0,
            "PrintSolutionRejectionReason": True,
            "CEqualD": problem["c_equals_d"],
        },
        "BenchmarkProblems": [[problemType, {
            "ForkParameters": [{name: [copy.deepcopy(value)]}
                               for name, value in parameters.items()],
            "BenchmarkFinalParameters": final,
        }]],
    }


def _descriptorRejection(solution, request):
    # Physical buffer layout is supplied by descriptors, not a tuning choice.
    for name, value in _implementationParameters(request).items():
        if solution.get(name) != value:
            return f"Tensile changed the descriptor {name}={value} to {solution.get(name)}"
    return None


def _select(request, configPath, derive):
    from Tensile.Common import state
    from Tensile.Common.GlobalParameters import defaultSolution
    from Tensile.SolutionStructs.Validators.ProblemSizes import problemSizeRejection
    import yaml

    rejections = []
    candidates = request["candidates"]
    for candidate in candidates:
        config = _configuration(request, candidate)
        diagnostics = io.StringIO()
        try:
            with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
                solution = derive(config, f"{request['model']} candidate {candidate['id']}")
        except SS.SingleSolutionRejected as error:
            captured = diagnostics.getvalue()
            reasons = [line for line in captured.splitlines() if line.startswith("reject:")]
            rejections.append({"candidate_id": candidate["id"],
                               "reason": "\n".join(reasons)[:4096] or str(error),
                               "diagnostics": captured[:4096] + captured[-4096:]})
            continue
        # Any other exception is a request/toolchain/implementation failure. It
        # propagates without trying another candidate or emitting a kernel.
        problem = request["problem"]
        reason = _descriptorRejection(solution, request) or problemSizeRejection(
            solution, [problem[key] for key in ("m", "n", "batch", "k")],
            {tensor.upper(): problem[f"strides_{tensor}"] for tensor in "abcd"},
            {tensor.upper(): problem.get(f"sizes_{tensor}") for tensor in "abcd"})
        if reason:
            rejections.append({"candidate_id": candidate["id"], "reason": reason})
            continue
        with configPath.open("x", encoding="utf-8") as stream:
            stream.write("# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
                         "# SPDX-License-Identifier: MIT\n")
            yaml.safe_dump(config, stream, sort_keys=False)
        implementation = _implementationParameters(request)
        defaults = {name: state(value) for name, value in defaultSolution.items()
                    if name not in candidate["parameters"] and name not in implementation}
        resolved = {name: state(solution[name]) for name in defaultSolution if name in solution}
        for name in ("MacroTile0", "MacroTile1", "MIWaveTile", "MIWaveGroup", "NumThreads",
                     "_GlobalAccumulation"):
            resolved[name] = state(solution[name])
        selected = copy.deepcopy(candidate["parameters"])
        latency = candidate.get("predicted_cycles")
        metadata = {
            "model": request["model"],
            "candidate_id": candidate["id"],
            "predicted_cycles": latency,
            "selected_parameters": selected,
            "implementation_parameters": implementation,
            "ranked_candidates": copy.deepcopy(candidates),
            "rejections": rejections,
            "defaults_source": _DEFAULTS_SOURCE,
            "default_parameters": defaults,
            "resolved_parameters": resolved,
            "derived_parameters": {name: value for name, value in resolved.items()
                                   if name not in selected and name not in implementation
                                   and defaults.get(name) != value},
            "wave_layout_origin": "Selected Tensile recipe; native wave size from Tensile defaults",
            "problem": copy.deepcopy(request["problem"]),
            "problem_type": _problemType(request),
            "hardware": copy.deepcopy(request.get("hardware", {})),
            "model_assumptions": copy.deepcopy(request.get("model_assumptions", {})),
            "summary": (f"{request['model']} candidate {candidate['id']}: "
                        f"{latency:.6g} predicted cycles; "
                        f"{len(rejections)} earlier candidates rejected by Tensile")
                       if latency is not None else
                       (f"{request['model']} candidate {candidate['id']}; no latency prediction; "
                        f"{len(rejections)} earlier candidates rejected by Tensile"),
        }
        return configPath, solution, metadata
    details = "; ".join(f"{item['candidate_id']}: {item['reason']}" for item in rejections)
    raise SS.SingleSolutionConfigError("No JIT candidate supports the problem; all supplied recipes were rejected: " + details)


def generateAndBuildJitGemm(requestPath, outputPath, *, architecture, **options):
    requestPath = Path(requestPath).resolve(strict=True)
    request = _readRequest(requestPath)
    _require(request["architecture"] == architecture, "Prediction and build architectures differ")
    outputPath = Path(outputPath).absolute()
    configPath = Path(str(outputPath) + ".yaml")
    predictionPath = Path(str(outputPath) + ".prediction.json")
    _require(not configPath.exists() and not predictionPath.exists(),
             "Selected YAML or prediction output already exists")
    selection = (_configuration(request, request["candidates"][0]),
                 lambda derive: _select(request, configPath, derive))
    result = SS._generateAndBuild(requestPath, outputPath, architecture=architecture,
                                  _selection=selection, **options)
    manifest = json.loads(result.manifestPath.read_text(encoding="utf-8"))
    with predictionPath.open("x", encoding="utf-8") as stream:
        json.dump(manifest["jit_prediction"], stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requestPath")
    parser.add_argument("outputPath")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--cxx-compiler", dest="cxxCompiler", default="amdclang++")
    parser.add_argument("--offload-bundler", dest="offloadBundler", default="clang-offload-bundler")
    parser.add_argument("--code-object-version", dest="codeObjectVersion", choices=("4", "5", "6"), default="4")
    parser.add_argument("--library-format", dest="libraryFormat", choices=("msgpack", "yaml"), default="msgpack")
    parser.add_argument("--keep-build-tmp", dest="keepBuildTmp", action="store_true")
    try:
        result = generateAndBuildJitGemm(**vars(parser.parse_args(argv)))
    except (SS.SingleSolutionError, OSError, ValueError) as error:
        print(f"JIT GEMM build failed: {error}", file=sys.stderr)
        return 1
    print(result.manifestPath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
