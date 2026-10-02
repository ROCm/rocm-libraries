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

from . import JitDebug, SingleSolution as SS


_DEFAULTS_SOURCE = "Tensile/Common/GlobalParameters.py:defaultBenchmarkCommonParameters"
_MAX_CANDIDATES = 192
_MODELED_CONTRACT = "origami.gemm.dp.v1"


def _require(condition, message):
    if not condition:
        raise SS.SingleSolutionConfigError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _validateModeled(request, candidate):
    contract = request.get("modeled_contract")
    if contract is None:
        _require("modeled" not in candidate, "Modeled outputs require a modeled_contract")
        return
    _require(contract == _MODELED_CONTRACT, f"Unsupported modeled contract: {contract}")
    modeled = candidate.get("modeled")
    _require(isinstance(modeled, dict), "Missing Origami modeled outputs")
    mt = modeled.get("macro_tile")
    _require(isinstance(mt, list) and len(mt) == 3 and all(_integer(v, 1) for v in mt),
             "Invalid modeled macro_tile")
    for group, keys in (
            ("workgroup_mapping", ("wgm", "wgmxcc", "wgmxccchunk", "wgmxccsplitk")),
            ("stagger", ("staggerU", "staggerUMapping", "staggerUStrideShift")),
            ("launch", ("stream_k", "grid", "active_cus", "timesteps", "split_factor"))):
        values = modeled.get(group)
        _require(isinstance(values, dict), f"Missing modeled {group}")
        for key in keys:
            value = values.get(key)
            _require(type(value) is int and (value != 0 if key == "wgm" else value >= 0),
                     f"Missing/invalid modeled {group}.{key}")
    launch = modeled["launch"]
    _require(modeled["stagger"]["staggerUStrideShift"] <= 31,
             "Modeled staggerUStrideShift exceeds the runtime argument range")
    problem = request["problem"]
    grid = ((problem["m"] + mt[0] - 1) // mt[0]
            * ((problem["n"] + mt[1] - 1) // mt[1]) * problem["batch"])
    _require(launch.get("reduction") == "none" and launch["stream_k"] == 0
             and launch["split_factor"] == 1 and launch["grid"] == grid,
             "Modeled launch conflicts with the data-parallel contract")
    parameters = candidate["parameters"]
    for name in ("MatrixInstruction", "DepthU", "NonTemporalA", "NonTemporalB"):
        _require(name in parameters, f"Missing modeled parameter {name}; defaults are not predictions")
    mi = parameters["MatrixInstruction"]
    _require(isinstance(mi, list) and len(mi) == 9 and all(_integer(v, 1) for v in mi),
             "Modeled MatrixInstruction must retain the nine-value recipe")
    _require(parameters["DepthU"] == mt[2], "Modeled DepthU differs from macro_tile")


def _modeledParameters(request, candidate):
    """Translate model outputs into Tensile units without changing their meaning."""
    if request.get("modeled_contract") is None:
        return {}
    from .Common.DataType import DataType

    modeled = candidate["modeled"]
    mapping, stagger = modeled["workgroup_mapping"], modeled["stagger"]
    bpe = DataType(_problemType(request)["DataType"]).numBytes()
    stride = modeled["macro_tile"][2] * bpe * (2 ** stagger["staggerUStrideShift"])
    _require(float(stride).is_integer(), "Modeled stagger stride is not an integral byte count")
    # Origami 0 and 1 both disable XCC remapping. Tensile's recipe uses 1 for
    # identity. Group=0 implements whole-grid contiguous grouping, not chunking.
    return {"StreamK": 0, "GlobalSplitU": 1,
            "WorkGroupMapping": mapping["wgm"],
            "WorkGroupMappingXCC": max(1, mapping["wgmxcc"]),
            "WorkGroupMappingXCCGroup": 0,
            "StaggerU": stagger["staggerU"],
            "StaggerUMapping": stagger["staggerUMapping"],
            "StaggerUStride": int(stride)}


def _candidateParameters(request, candidate):
    parameters = dict(candidate["parameters"])
    for name, value in _modeledParameters(request, candidate).items():
        _require(name not in parameters or parameters[name] == value,
                 f"Candidate {name} conflicts with its modeled output")
        parameters[name] = value
    return parameters


def _modeledTransportRejection(request, candidate):
    if request.get("modeled_contract") is None:
        return None
    modeled = candidate["modeled"]
    mapping, stagger = modeled["workgroup_mapping"], modeled["stagger"]
    if mapping["wgmxccchunk"] or mapping["wgmxccsplitk"]:
        return ("Data-parallel Tensile cannot represent Origami "
                f"wgmxccchunk={mapping['wgmxccchunk']}, wgmxccsplitk={mapping['wgmxccsplitk']}; "
                "these outputs require the Stream-K mapping ABI")
    xcc = mapping["wgmxcc"]
    if xcc > 1 and (xcc not in (2, 4, 8, 16, 32) or modeled["launch"]["grid"] % xcc):
        return (f"Data-parallel Tensile cannot represent Origami wgmxcc={xcc} "
                f"for grid={modeled['launch']['grid']} with whole-grid contiguous grouping")
    if abs(mapping["wgm"]) >= 1024:
        return f"Origami wgm={mapping['wgm']} exceeds the data-parallel runtime range"
    if stagger["staggerUStrideShift"] > 31:
        return "Origami staggerUStrideShift exceeds the runtime argument range"
    return None


def _modeledRejection(solution, request, candidate):
    if request.get("modeled_contract") is None:
        return None
    from .Common import state

    expected = {name: value for name, value in candidate["parameters"].items()
                if name in ("MatrixInstruction", "DepthU", "NonTemporalA", "NonTemporalB")}
    expected.update(_modeledParameters(request, candidate))
    # Tensile normalizes the nine-value instruction into MI + wave topology.
    mi = expected.pop("MatrixInstruction")
    expected.update(MatrixInstruction=mi[:4], MIWaveTile=mi[5:7], MIWaveGroup=mi[7:9])
    mt = candidate["modeled"]["macro_tile"]
    expected.update(MacroTile0=mt[0], MacroTile1=mt[1])
    # Byte stride is normalized when stagger is zero. Its semantic output is
    # the loop-iteration shift, which must still equal Origami's prediction.
    expected.pop("StaggerUStride")
    expected["_staggerStrideShift"] = candidate["modeled"]["stagger"]["staggerUStrideShift"]
    for name, value in expected.items():
        actual = state(solution.get(name))
        if actual != value:
            return f"Tensile changed modeled {name}={value} to {actual}"
    support = solution.get("InternalSupportParams", {})
    if solution.get("SpaceFillingAlgo") or solution.get("ClusterDim", [1, 1]) != [1, 1]:
        return "Tensile solution replaces the modeled workgroup mapping with another mapping algorithm"
    if not support.get("SupportCustomWGM") or support.get("KernArgsVersion", 0) < 2:
        return "Tensile solution cannot carry the modeled workgroup mapping at runtime"
    if candidate["modeled"]["stagger"]["staggerU"] and (
            not solution.get("BufferLoad") or not support.get("SupportCustomStaggerU")):
        return "Tensile solution cannot carry the modeled stagger at runtime"
    return None


def _implementationParameters(request):
    """Bind Tensile's layout parameter to the supplied scale buffers.

    Scale modes, types and layout affect performance and belong to the problem
    being predicted. They are fixed for every candidate in this request: this
    selector does not rearrange buffers when a candidate needs another layout.
    The returned values record those descriptor constraints separately from
    the predictor's choices; they do not imply that scaling has no cost.
    """
    problem = request["problem"]
    modes = {"None", "Scalar", "Vector", "Block_32_UE8M0", "Block_16_UE8M0",
             "Block_32_UE4M3", "Block_16_UE4M3", "Block_32_UE5M3", "Block_16_UE5M3",
             "Block_32_UE8M0_32_8_EXT"}
    for tensor in "ab":
        key = f"scale_mode_{tensor}"
        if key in problem:
            _require(isinstance(problem[key], str) and problem[key] in modes,
                     f"problem.{key} must name a descriptor ScalingFormat")
    architecture = request["architecture"].split(":", 1)[0]
    expected = {
        tensor: ("HostPreSwizzle" if mode == "Block_32_UE8M0_32_8_EXT" else
                 "InMemorySwizzle" if architecture == "gfx1250" else "NoSwizzle")
        for tensor in "ab"
        if (mode := problem.get(f"scale_mode_{tensor}", "None")).startswith("Block_")
    }
    # Legacy requests without descriptor scale modes retain default derivation.
    # New descriptor callers let this provider-private translation choose the
    # layout; they do not repeat target-dependent Tensile layout rules in C++.
    if "mx_scale_format" not in problem:
        if not expected:
            return {}
        layouts = set(expected.values())
        _require(len(layouts) == 1, "Descriptor A/B scale modes require different MX layouts")
        layout = next(iter(layouts))
    else:
        layout = problem["mx_scale_format"]
    _require(isinstance(layout, str) and layout in
             ("NoSwizzle", "HostPreSwizzle", "InMemorySwizzle"),
             "problem.mx_scale_format must name an explicit MX scale layout")
    problemType = _problemType(request)
    _require(problemType.get("MXBlockA") or problemType.get("MXBlockB"),
             "problem.mx_scale_format requires an MX-scaled operand")
    for tensor, value in expected.items():
        _require(layout == value,
                 f"problem.scale_mode_{tensor}={problem[f'scale_mode_{tensor}']} "
                 f"disagrees with MXScaleFormat={layout}")
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
    requested = request.get("requested_solutions", 1)
    _require(_integer(requested, 1) and requested <= len(candidates),
             "requested_solutions must be a positive integer no larger than the candidate count")
    excluded = request.get("exclude_kernel_names", [])
    _require(isinstance(excluded, list) and all(isinstance(name, str) and name for name in excluded),
             "exclude_kernel_names must be a list of kernel names")
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
        _validateModeled(request, candidate)
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
        from .SolutionStructs.Problem import ProblemType

        # BenchmarkProcess validates its argument lists even though this path
        # only derives a solution. Use the normal problem type's bias whitelist.
        biasTypes = ProblemType(problemType, False)["BiasDataTypeList"]
        final.append({"BiasTypeArgs": [value.toChar() for value in biasTypes]})
    if problemType.get("ActivationType") in ("all", "hipblaslt_all"):
        final.append({"ActivationArgs": [[{"Enum": "none"}]]})
    # Fill omitted/Auto layout from the request, but preserve explicit predictor
    # choices. Selection rejects concrete conflicts before deriving a solution.
    parameters = _candidateParameters(request, candidate)
    for name, value in _implementationParameters(request).items():
        if parameters.get(name, "Auto") == "Auto":
            parameters[name] = value
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


def _candidateDescriptorRejection(candidate, request):
    for name, required in _implementationParameters(request).items():
        supplied = candidate["parameters"].get(name)
        # Malformed values still go through Tensile's parameter validation.
        if supplied in ("NoSwizzle", "HostPreSwizzle", "InMemorySwizzle") and supplied != required:
            return f"Candidate {name}={supplied} conflicts with descriptor {name}={required}"
    return None


def _select(request, configPath, derive, ranking=None, _debug=JitDebug.NULL):
    """Return the first acceptable candidate as (configPath, solution, metadata).

    A ``ranking`` from _Ranking continues after the candidate it last accepted,
    and after its first acceptance returns None once no candidate is left.
    """
    from .Common import state
    from .Common.GlobalParameters import defaultSolution
    from .SolutionStructs.Validators.ProblemSizes import problemSizeRejection
    import yaml

    ranking = _Ranking() if ranking is None else ranking
    rejections = ranking.rejections
    excluded = set(request.get("exclude_kernel_names", []))
    candidates = request["candidates"]

    def tried(candidate, outcome, reason=None, span=None):
        _debug.candidate(index=ranking.position - 1, of=len(candidates), id=candidate["id"],
                         outcome=outcome, reason=reason, ns=(span or {}).get("ns"))

    while ranking.position < len(candidates):
        candidate = candidates[ranking.position]
        ranking.position += 1
        reason, category = _modeledTransportRejection(request, candidate), "modeled_transport"
        if not reason:
            reason, category = _candidateDescriptorRejection(candidate, request), "descriptor"
        if reason:
            rejections.append({"candidate_id": candidate["id"], "reason": reason})
            tried(candidate, "rejected", category)
            continue
        config = _configuration(request, candidate)
        diagnostics = io.StringIO()
        try:
            with _debug.span("derive", stage=False, rejects=(SS.SingleSolutionRejected,),
                             candidate=candidate["id"]) as span:
                with contextlib.redirect_stdout(diagnostics), \
                        contextlib.redirect_stderr(diagnostics):
                    solution = derive(config, f"{request['model']} candidate {candidate['id']}")
        except SS.SingleSolutionRejected as error:
            captured = diagnostics.getvalue()
            reasons = [line for line in captured.splitlines() if line.startswith("reject:")]
            rejections.append({"candidate_id": candidate["id"],
                               "reason": "\n".join(reasons)[:4096] or str(error),
                               "diagnostics": captured[:4096] + captured[-4096:]})
            tried(candidate, "rejected", "tensile", span)
            continue
        # Any other exception is a request/toolchain/implementation failure. It
        # propagates without trying another candidate or emitting a kernel.
        problem = request["problem"]
        reason, category = _modeledRejection(solution, request, candidate), "modeled"
        if not reason:
            reason, category = _descriptorRejection(solution, request), "descriptor"
        if not reason:
            reason, category = problemSizeRejection(
                solution, [problem[key] for key in ("m", "n", "batch", "k")],
                {tensor.upper(): problem[f"strides_{tensor}"] for tensor in "abcd"},
                {tensor.upper(): problem.get(f"sizes_{tensor}") for tensor in "abcd"}
            ), "problem_size"
        if reason:
            rejections.append({"candidate_id": candidate["id"], "reason": reason})
            tried(candidate, "rejected", category, span)
            continue
        kernel = solution.get("KernelNameMin")
        if kernel in excluded:
            rejections.append({"candidate_id": candidate["id"],
                               "reason": f"Excluded kernel {kernel}"})
            tried(candidate, "rejected", "excluded", span)
            continue
        if kernel in ranking.kernels:
            rejections.append({"candidate_id": candidate["id"],
                               "reason": f"Same kernel as candidate {ranking.kernels[kernel]}"})
            tried(candidate, "rejected", "duplicate", span)
            continue
        with configPath.open("x", encoding="utf-8") as stream:
            stream.write("# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
                         "# SPDX-License-Identifier: MIT\n")
            yaml.safe_dump(config, stream, sort_keys=False)
        implementation = _implementationParameters(request)
        selected = _candidateParameters(request, candidate)
        defaults = {name: state(value) for name, value in defaultSolution.items()
                    if name not in selected and name not in implementation}
        resolved = {name: state(solution[name]) for name in defaultSolution if name in solution}
        for name in ("MacroTile0", "MacroTile1", "MIWaveTile", "MIWaveGroup", "NumThreads",
                     "_GlobalAccumulation", "_staggerStrideShift"):
            if name in solution:
                resolved[name] = state(solution[name])
        latency = candidate.get("predicted_cycles")
        metadata = {
            "model": request["model"],
            "candidate_id": candidate["id"],
            "predicted_cycles": latency,
            "selected_parameters": selected,
            "implementation_parameters": implementation,
            "ranked_candidates": copy.deepcopy(candidates),
            "rejections": copy.deepcopy(rejections),
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
        if request.get("modeled_contract"):
            metadata["modeled_contract"] = request["modeled_contract"]
            metadata["modeled"] = copy.deepcopy(candidate["modeled"])
        ranking.kernels[kernel] = candidate["id"]
        ranking.accepted += 1
        tried(candidate, "selected", span=span)
        return configPath, solution, metadata
    if ranking.accepted:
        return None
    details = "; ".join(f"{item['candidate_id']}: {item['reason']}" for item in rejections)
    raise SS.SingleSolutionConfigError("No JIT candidate supports the problem; all supplied recipes were rejected: " + details)


class _Ranking:
    """Progress through the ranked candidates across the builds of one request."""

    def __init__(self):
        self.position = 0
        self.accepted = 0
        self.kernels = {}  # accepted kernel name -> candidate id
        self.rejections = []


def generateAndBuildJitGemm(requestPath, outputPath, *, architecture, _debug=JitDebug.NULL,
                            **options):
    """Build the best ``requested_solutions`` candidates as ``outputPath/bundle-<rank>``.

    ``outputPath/bundle`` links to ``bundle-0``, whose recipe and prediction are
    also written next to ``outputPath``. Later ranks write ``<outputPath>.<rank>.yaml``.
    Fewer bundles than requested means the remaining candidates were rejected.
    """
    requestPath = Path(requestPath).resolve(strict=True)
    with _debug.span("request_read"):
        request = _readRequest(requestPath)
    _debug.request(requested=request.get("requested_solutions", 1),
                   candidates=len(request["candidates"]))
    _require(request["architecture"] == architecture, "Prediction and build architectures differ")
    outputPath = Path(outputPath).absolute()
    configPath = Path(str(outputPath) + ".yaml")
    predictionPath = Path(str(outputPath) + ".prediction.json")
    _require(not configPath.exists() and not predictionPath.exists(),
             "Selected YAML or prediction output already exists")
    ranking = _Ranking()

    def select(derive):
        path = configPath if not ranking.accepted else Path(f"{outputPath}.{ranking.accepted}.yaml")
        return _select(request, path, derive, ranking, _debug)

    selection = (_configuration(request, request["candidates"][0]), select)
    result = SS._generateAndBuild(requestPath, outputPath, architecture=architecture,
                                  _selection=selection,
                                  _count=request.get("requested_solutions", 1), _debug=_debug,
                                  **options)
    with _debug.span("prediction_write"):
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
    parser.add_argument("--source-only", dest="sourceOnly", action="store_true")
    JitDebug.addArguments(parser)
    args = vars(parser.parse_args(argv))
    debug = JitDebug.fromArguments(parser, args, module="Tensile.JitGemm", mode="prediction")
    status, errorType = "failed", None
    try:
        result = generateAndBuildJitGemm(**args, _debug=debug)
        status = "ok"
    except (SS.SingleSolutionError, OSError, ValueError) as error:
        errorType = type(error).__name__
        print(f"JIT GEMM build failed: {error}", file=sys.stderr)
        return 1
    except BaseException as error:
        errorType = type(error).__name__
        raise
    finally:
        debug.finish(status, errorType)
    print(result.manifestPath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
