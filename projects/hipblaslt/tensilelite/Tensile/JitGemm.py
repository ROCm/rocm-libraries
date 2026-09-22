# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate Origami-ranked parameter recipes and build one solution, without benchmarking.

Origami runs in the C++ caller. This driver retains its ordering, rejects invalid
recipes before code emission, and uses SingleSolution's normal toolchain/build.
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


_PARAMETERS = {"MatrixInstruction", "DepthU", "NonTemporalA", "NonTemporalB"}
_DEFAULTS_SOURCE = "Tensile/Common/GlobalParameters.py:defaultBenchmarkCommonParameters"
_MAX_CANDIDATES = 192


def _require(condition, message):
    if not condition:
        raise SS.SingleSolutionConfigError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


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
    _require(request.get("model") == "origami.gemm.estimation", "Unexpected prediction model")
    _require(isinstance(request.get("architecture"), str), "Missing target architecture")
    problem = request.get("problem")
    _require(isinstance(problem, dict), "Missing problem descriptors")
    for key in ("m", "n", "k", "batch"):
        _require(_integer(problem.get(key), 1), f"problem.{key} must be a positive integer")
    _require(problem.get("data_type") in ("s", "h"), "Expected F32 or FP16 inputs/output")
    for key in ("transpose_a", "transpose_b", "c_equals_d", "high_precision_accumulate"):
        _require(type(problem.get(key)) is bool, f"problem.{key} must be boolean")
    _require(problem["data_type"] != "h" or problem["high_precision_accumulate"],
             "FP16 requires FP32 accumulation")
    for tensor in "abcd":
        strides = problem.get(f"strides_{tensor}")
        _require(isinstance(strides, list) and len(strides) == 3
                 and all(_integer(value) for value in strides) and strides[0] == 1,
                 f"Expected canonical strides for tensor {tensor}")
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
        _require(type(latency) in (int, float) and math.isfinite(latency)
                 and 0 < latency < sys.float_info.max, "Invalid Origami predicted latency")
        parameters = candidate.get("parameters")
        _require(isinstance(parameters, dict) and set(parameters) == _PARAMETERS,
                 "A candidate may specify only MI, DepthU, and A/B cache hints")
        mi = parameters["MatrixInstruction"]
        _require(isinstance(mi, list) and len(mi) == 9
                 and all(_integer(value, 1) for value in mi)
                 and mi[0:2] == [16, 16] and mi[3:5] == [1, 1]
                 and mi[7] * mi[8] == 4, "Expected a four-wave square MFMA recipe")
        _require(_integer(parameters["DepthU"], 1), "DepthU must be positive")
        for key in ("NonTemporalA", "NonTemporalB"):
            _require(type(parameters[key]) is int and parameters[key] in (0, 4),
                     "Unsupported cache hint")
    return request


def _configuration(request, candidate):
    problem = request["problem"]
    exact = {"sizes": [problem[key] for key in ("m", "n", "batch", "k")]}
    for tensor in "abcd":
        exact[f"strides{tensor.upper()}"] = list(problem[f"strides_{tensor}"])
    # These describe the actual operation. Kernel tuning fields below contain only
    # the selected parameters; BenchmarkProcess fills every other canonical default.
    return {
        "GlobalParameters": {
            "PrintLevel": 0,
            "PrintSolutionRejectionReason": True,
            "CEqualD": problem["c_equals_d"],
        },
        "BenchmarkProblems": [[{
            "OperationType": "GEMM",
            "DataType": problem["data_type"],
            "DestDataType": problem["data_type"],
            "ComputeDataType": "s",
            "HighPrecisionAccumulate": problem["high_precision_accumulate"],
            "TransposeA": problem["transpose_a"],
            "TransposeB": problem["transpose_b"],
            "UseBeta": True,
            "UseBias": 0,
            "Batched": True,
            "StridedBatched": True,
            "Activation": False,
            "ActivationType": "none",
        }, {
            "ForkParameters": [{name: [copy.deepcopy(value)]}
                               for name, value in candidate["parameters"].items()],
            "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": exact}]}],
        }]],
    }


def _problemRejection(solution, request, candidate):
    """Check shape-dependent predicates available before compilation.

    The C++ runtime still evaluates the complete normal predicate set when the
    resulting solution is registered. These checks cover the constraints of this
    bounded default recipe family, including vector-load minima and buffer limits.
    """
    p = request["problem"]
    m, n, k, batch = (p[key] for key in ("m", "n", "k", "batch"))
    mi = candidate["parameters"]["MatrixInstruction"]
    expected = (mi[0] * mi[5] * mi[7], mi[1] * mi[6] * mi[8])
    if (solution["MacroTile0"], solution["MacroTile1"]) != expected:
        return "Tensile derived a different macro tile from the modeled recipe"
    if solution["DepthU"] != candidate["parameters"]["DepthU"]:
        return "Tensile derived a different DepthU from the modeled recipe"
    if solution["StreamK"] != 0 or solution["GlobalSplitU"] != 1:
        return "Current Tensile defaults no longer match the unsplit Origami model"
    for value, name in ((m, "AssertFree0ElementMultiple"),
                        (n, "AssertFree1ElementMultiple"),
                        (k, "AssertSummationElementMultiple")):
        if value % solution[name]:
            return f"Problem violates {name}={solution[name]}"
    problemType = solution["ProblemType"]
    bpe = 2 if p["data_type"] == "h" else 4
    for tensor, free, index, size1 in (("A", m, 0, m if p["transpose_a"] else k),
                                       ("B", n, 1, k if p["transpose_b"] else n)):
        width = solution[f"GlobalReadVectorWidth{tensor}"]
        tlu = problemType[f"TLU{tensor}"]
        if tlu and free < width:
            return f"Leading free dimension {tensor} is smaller than vector width {width}"
        if solution["BufferLoad"]:
            depthOrTile = solution["DepthU"] if tlu else solution[f"MacroTile{index}"]
            shift = width if tlu and solution[f"AssertFree{index}ElementMultiple"] < width else 0
            offset = p[f"strides_{tensor.lower()}"][1] * min(depthOrTile, size1) + shift
            if offset * bpe >= 2**32:
                return f"Tensor {tensor} exceeds the buffer-load offset limit"
    for tensor, needed in (("c", solution["BufferLoad"]), ("d", solution["BufferStore"])):
        if needed and p[f"strides_{tensor}"][1] * min(solution["MacroTile1"], n) * bpe >= 2**32:
            return f"Tensor {tensor.upper()} exceeds the buffer offset limit"
    workgroups = ((m + expected[0] - 1) // expected[0]) * ((n + expected[1] - 1) // expected[1]) * batch
    if workgroups > 2**24:
        return "Problem exceeds the normal compressed workgroup-count limit"
    return None


def _select(request, configPath, derive):
    from Tensile.Common import state
    from Tensile.Common.GlobalParameters import defaultSolution
    import yaml

    rejections = []
    for candidate in request["candidates"]:
        config = _configuration(request, candidate)
        diagnostics = io.StringIO()
        try:
            with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
                solution = derive(config, f"Origami candidate {candidate['id']}")
        except SS.SingleSolutionRejected as error:
            captured = diagnostics.getvalue()
            reasons = [line for line in captured.splitlines() if line.startswith("reject:")]
            rejections.append({"candidate_id": candidate["id"],
                               "reason": "\n".join(reasons)[:4096] or str(error),
                               "diagnostics": captured[:4096] + captured[-4096:]})
            continue
        # Any other exception is a request/toolchain/implementation failure. It
        # propagates without trying another candidate or emitting a kernel.
        reason = _problemRejection(solution, request, candidate)
        if reason:
            rejections.append({"candidate_id": candidate["id"], "reason": reason})
            continue
        with configPath.open("x", encoding="utf-8") as stream:
            stream.write("# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
                         "# SPDX-License-Identifier: MIT\n")
            yaml.safe_dump(config, stream, sort_keys=False)
        defaults = {name: state(value) for name, value in defaultSolution.items()
                    if name not in candidate["parameters"]}
        resolved = {name: state(solution[name]) for name in defaultSolution if name in solution}
        for name in ("MacroTile0", "MacroTile1", "MIWaveTile", "MIWaveGroup", "NumThreads",
                     "_GlobalAccumulation"):
            resolved[name] = state(solution[name])
        selected = copy.deepcopy(candidate["parameters"])
        metadata = {
            "model": request["model"],
            "candidate_id": candidate["id"],
            "predicted_cycles": candidate["predicted_cycles"],
            "selected_parameters": selected,
            "ranked_candidates": copy.deepcopy(request["candidates"]),
            "rejections": rejections,
            "defaults_source": _DEFAULTS_SOURCE,
            "default_parameters": defaults,
            "resolved_parameters": resolved,
            "derived_parameters": {name: value for name, value in resolved.items()
                                   if name not in selected and defaults.get(name) != value},
            "wave_layout_origin": "Four-wave MFMA recipe; retained with the selected candidate",
            "problem": copy.deepcopy(request["problem"]),
            "hardware": copy.deepcopy(request.get("hardware", {})),
            "model_assumptions": copy.deepcopy(request.get("model_assumptions", {})),
            "summary": (f"Origami candidate {candidate['id']}: "
                        f"{candidate['predicted_cycles']:.6g} predicted cycles; "
                        f"{len(rejections)} earlier candidates rejected by Tensile"),
        }
        return configPath, solution, metadata
    details = "; ".join(f"{item['candidate_id']}: {item['reason']}" for item in rejections)
    raise SS.SingleSolutionConfigError("No Origami-ranked candidate supports the problem: " + details)


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
