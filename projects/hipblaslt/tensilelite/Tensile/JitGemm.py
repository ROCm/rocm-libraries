# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate parameter recipes and build one solution without benchmarking.

Origami runs in the C++ caller. This driver retains its ordering, rejects invalid
recipes before code emission, and uses SingleSolution's normal toolchain/build.
Operations without a valid modeled recipe use native Tensile defaults and record
that no latency prediction is available.
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
    if architecture == "gfx950" and layout == "NoSwizzle":
        raise SS.SingleSolutionConfigError(
            "gfx950 natural MX scale layout is not implemented by the shared subtile generator: "
            "it multiplies scale strides by 32 and uses pre-swizzled scale loads")
    _require(layout != "HostPreSwizzle" or architecture == "gfx950",
             "MXScaleFormat=HostPreSwizzle requires gfx950")
    _require(layout != "InMemorySwizzle" or architecture == "gfx1250",
             "MXScaleFormat=InMemorySwizzle requires gfx1250")
    _require(architecture != "gfx1250" or layout == "InMemorySwizzle",
             "gfx1250 MX requires MXScaleFormat=InMemorySwizzle")
    for tensor in "ab":
        mode = problem.get(f"scale_mode_{tensor}", "None")
        if mode.startswith("Block_"):
            expected = ("HostPreSwizzle" if mode == "Block_32_UE8M0_32_8_EXT" else
                        "InMemorySwizzle" if architecture == "gfx1250" else "NoSwizzle")
            _require(layout == expected,
                     f"problem.scale_mode_{tensor}={mode} disagrees with MXScaleFormat={layout}")
    return {"MXScaleFormat": layout}


def _problemType(request):
    """Preserve the caller's canonical Tensile operation description.

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
    _require(request.get("model") in ("origami.gemm.estimation", "tensile.defaults"),
             "Unexpected prediction model")
    _require(isinstance(request.get("architecture"), str), "Missing target architecture")
    architecture = request["architecture"].split(":", 1)[0]
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
    problemType = _problemType(request)
    from Tensile.SolutionStructs.Problem import ProblemType
    try:
        # This is Tensile's ordinary type/operation validation. Instruction and
        # coupled tuning constraints are checked when each solution is derived.
        ProblemType(problemType, False)
    except (KeyError, IndexError, ValueError, TypeError, RuntimeError) as error:
        raise SS.SingleSolutionConfigError(f"Invalid problem_type: {error}") from error
    _implementationParameters(request)
    for tensor in "abcd":
        strides = problem.get(f"strides_{tensor}")
        _require(isinstance(strides, list) and len(strides) == 3
                 and all(_integer(value) for value in strides) and strides[0] == 1,
                 f"Expected canonical strides for tensor {tensor}")
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
        if request["model"] == "origami.gemm.estimation":
            _require(type(latency) in (int, float) and math.isfinite(latency)
                     and 0 < latency < sys.float_info.max, "Invalid Origami predicted latency")
        else:
            _require(latency is None, "An unmodeled default candidate has no predicted latency")
        parameters = candidate.get("parameters")
        _require(isinstance(parameters, dict) and set(parameters) <= _PARAMETERS
                 and (request["model"] == "tensile.defaults" or set(parameters) == _PARAMETERS),
                 "A candidate may specify only MI, DepthU, and A/B cache hints")
        if "MatrixInstruction" in parameters:
            mi = parameters["MatrixInstruction"]
            _require(isinstance(mi, list) and len(mi) == 9
                     and all(_integer(value, 1) for value in mi),
                     "Expected a nine-field matrix-instruction recipe")
        if "DepthU" in parameters:
            _require(_integer(parameters["DepthU"], 1), "DepthU must be positive")
        for key in ("NonTemporalA", "NonTemporalB"):
            allowedHints = (0,) if architecture in ("gfx90a", "gfx1250") else (0, 4)
            _require(key not in parameters or
                     (type(parameters[key]) is int and parameters[key] in allowedHints),
                     f"Unsupported {architecture} cache hint")
    return request


def _defaultCandidates(request):
    """Try canonical defaults, then native MI forms from Tensile's own catalog.

    This is a deterministic validity search, without a latency estimate or
    datatype substitution. Full ISA/type/resource checks remain in Tensile.
    """
    from Tensile.Common.ValidParameters import makeValidMatrixInstructions

    result = copy.deepcopy(request["candidates"])
    if request["model"] != "tensile.defaults" or any(c["parameters"] for c in result):
        return result
    instructions = sorted({tuple(mi) for mi in makeValidMatrixInstructions() if len(mi) == 4},
                          key=lambda mi: (mi[3] != 1, mi[0] != 16, mi[1] != 16, mi[2]))
    problemType = _problemType(request)
    modes = [{}]
    if problemType.get("MXBlockA") or problemType.get("MXBlockB"):
        # These are the existing native MX implementations. Derivation checks
        # which transport is supported by the requested operation and target.
        from Tensile.Common.DataType import DataType
        from Tensile.Common.ValidParameters import makeValidMFMA

        # Use the existing instruction catalog for the actual MAC datatype pair.
        a = problemType.get("MacDataTypeA", problemType.get("DataTypeA", problemType["DataType"]))
        b = problemType.get("MacDataTypeB", problemType.get("DataTypeB", problemType["DataType"]))
        pair = DataType(a).toChar() + DataType(b).toChar()
        catalog = makeValidMFMA()
        # FP4 uses only the scaled FP8/FP6/FP4 instruction family. Intersect
        # that family with the actual pair; input datatypes remain unchanged.
        scaledForms = {tuple(mi) for mi in catalog["F4F4"]}
        mxInstructions = [mi for mi in catalog.get(pair, []) if tuple(mi) in scaledForms]
        if mxInstructions:
            instructions = sorted({tuple(mi) for mi in mxInstructions}, key=lambda mi: (mi[0] != 16, mi[2]))
        architecture = request["architecture"].split(":", 1)[0]
        if architecture == "gfx950":
            modes = [{"UseSubtileImpl": True, "LocalReadVectorWidth": 32}]
        elif architecture == "gfx1250":
            modes = [{"TDMInst": 3, "ScheduleIterAlg": 4}]
    identifier = max(c["id"] for c in result) + 1
    for mi in instructions:
        for waveTile in (1, 2):
            for mode in modes:
                mode = copy.deepcopy(mode)
                if mode.get("UseSubtileImpl"):
                    mode["DepthU"] = 2 * mi[2]
                if len(result) == _MAX_CANDIDATES:
                    return result
                result.append({"id": identifier, "predicted_cycles": None,
                               "parameters": {"MatrixInstruction": [*mi, 1, waveTile, waveTile, 2, 2],
                                              **mode},
                               "default_recipe_reason": (
                                   "gfx950 MX uses subtile geometry with two K iterations and LocalReadVectorWidth=32; preserve the native HostPreSwizzle layout"
                                   if mode.get("UseSubtileImpl") else
                                   "gfx1250 MX uses TDM with ScheduleIterAlg=4; preserve the native InMemorySwizzle layout"
                                   if mode.get("TDMInst") else
                                   "Native matrix instruction from Tensile's catalog with canonical defaults")})
                identifier += 1
    return result


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


def _problemRejection(solution, request, candidate):
    """Check shape-dependent predicates available before compilation.

    The C++ runtime still evaluates the complete normal predicate set when the
    resulting solution is registered. These checks cover the constraints of this
    bounded default recipe family, including vector-load minima and buffer limits.
    """
    p = request["problem"]
    m, n, k, batch = (p[key] for key in ("m", "n", "k", "batch"))
    if _problemType(request).get("OutputAmaxD", False) and batch != 1:
        return "Output-amax requires BatchSizeEqual=1 for the current reduction kernel"
    parameters = candidate["parameters"]
    mi = parameters.get("MatrixInstruction")
    expected = (solution["MacroTile0"], solution["MacroTile1"])
    if mi:
        expected = (mi[0] * mi[4] * mi[5] * mi[7],
                    mi[1] * (mi[3] // mi[4]) * mi[6] * mi[8])
        if (solution["MacroTile0"], solution["MacroTile1"]) != expected:
            return "Tensile derived a different macro tile from the requested recipe"
    for name, value in parameters.items():
        if name != "MatrixInstruction" and solution[name] != value:
            return f"Tensile changed the requested {name} parameter"
    for name, value in _implementationParameters(request).items():
        if solution.get(name) != value:
            return f"Tensile changed the descriptor {name}={value} to {solution.get(name)}"
    if request.get("model") == "origami.gemm.estimation" and (
            solution["StreamK"] != 0 or solution["GlobalSplitU"] != 1):
        return "Current Tensile defaults no longer match the unsplit Origami model"
    for value, name in ((m, "AssertFree0ElementMultiple"),
                        (n, "AssertFree1ElementMultiple"),
                        (k, "AssertSummationElementMultiple")):
        if value % solution[name]:
            return f"Problem violates {name}={solution[name]}"
    # No output elements means no tensor accesses. K=0 still reads C and
    # writes D (beta*C), but never reads A or B.
    if m == 0 or n == 0:
        return None
    problemType = solution["ProblemType"]
    from Tensile.Common.DataType import DataType

    requestedType = _problemType(request)
    for tensor, free, index, size1 in (("A", m, 0, m if p["transpose_a"] else k),
                                       ("B", n, 1, k if p["transpose_b"] else n)):
        if k == 0:
            continue
        storageType = problemType.get(f"DataType{tensor}", requestedType.get(
            f"DataType{tensor}", requestedType["DataType"]))
        bpe = DataType(storageType).numBytes()
        sizes = p.get(f"sizes_{tensor.lower()}")
        if sizes is not None:
            size1 = sizes[1]
        width = solution[f"GlobalReadVectorWidth{tensor}"]
        tlu = problemType[f"TLU{tensor}"]
        if tlu and free < width:
            return f"Leading free dimension {tensor} is smaller than vector width {width}"
        if solution["BufferLoad"]:
            depthOrTile = solution.get(f"_DepthU{tensor}", solution["DepthU"]) if tlu else solution[f"MacroTile{index}"]
            shift = width if tlu and solution[f"AssertFree{index}ElementMultiple"] < width else 0
            offset = p[f"strides_{tensor.lower()}"][1] * min(depthOrTile, size1) + shift
            if offset * bpe >= 2**32:
                return f"Tensor {tensor} exceeds the buffer-load offset limit"
    for tensor, needed in (("c", solution["BufferLoad"]), ("d", solution["BufferStore"])):
        bpe = DataType(problemType.get("DestDataType", requestedType.get(
            "DestDataType", requestedType["DataType"]))).numBytes()
        size1 = p.get(f"sizes_{tensor}", [m, n, batch])[1]
        if needed and p[f"strides_{tensor}"][1] * min(solution["MacroTile1"], size1) * bpe >= 2**32:
            return f"Tensor {tensor.upper()} exceeds the buffer offset limit"
    workgroups = ((m + expected[0] - 1) // expected[0]) * ((n + expected[1] - 1) // expected[1]) * batch
    if workgroups > 2**24:
        return "Problem exceeds the normal compressed workgroup-count limit"
    return None


def _select(request, configPath, derive):
    from Tensile.Common import state
    from Tensile.Common.GlobalParameters import defaultSolution
    from Tensile.Common.ValidParameters import makeValidMatrixInstructions
    import yaml

    validInstructions = {tuple(mi) for mi in makeValidMatrixInstructions() if len(mi) == 4}
    rejections = []
    candidates = _defaultCandidates(request)
    for candidate in candidates:
        mi = candidate["parameters"].get("MatrixInstruction")
        if mi and tuple(mi[:4]) not in validInstructions:
            rejections.append({"candidate_id": candidate["id"],
                               "reason": f"MatrixInstruction {mi[:4]} is absent from Tensile's catalog"})
            continue
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
        reason = _problemRejection(solution, request, candidate)
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
        metadata = {
            "model": request["model"],
            "candidate_id": candidate["id"],
            "predicted_cycles": candidate["predicted_cycles"],
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
            "summary": (f"Origami candidate {candidate['id']}: "
                        f"{candidate['predicted_cycles']:.6g} predicted cycles; "
                        f"{len(rejections)} earlier candidates rejected by Tensile")
                       if candidate["predicted_cycles"] is not None else
                       (f"Tensile default candidate {candidate['id']}; no latency prediction; "
                        f"{len(rejections)} earlier candidates rejected by Tensile"),
        }
        if request["model"] == "tensile.defaults":
            metadata["default_recipe_reason"] = candidate.get(
                "default_recipe_reason", "Canonical Tensile defaults")
        return configPath, solution, metadata
    if request["model"] == "origami.gemm.estimation":
        fallback = copy.deepcopy(request)
        fallback["model"] = "tensile.defaults"
        fallback["candidates"] = [{"id": 0, "predicted_cycles": None, "parameters": {}}]
        fallback.setdefault("model_assumptions", {})["fallback_reason"] = (
            "All Origami-ranked recipes were rejected by Tensile; select the first valid native recipe")
        path, solution, metadata = _select(fallback, configPath, derive)
        metadata["origami_candidates"] = copy.deepcopy(request["candidates"])
        metadata["origami_rejections"] = rejections
        return path, solution, metadata
    details = "; ".join(f"{item['candidate_id']}: {item['reason']}" for item in rejections)
    raise SS.SingleSolutionConfigError("No JIT candidate supports the problem: " + details)


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
