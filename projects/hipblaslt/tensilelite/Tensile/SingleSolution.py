# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build one GEMM solution, including its helpers, without benchmarking.

The callable returns a published bundle; ``python -m Tensile.SingleSolution`` is its
CLI. The output directory must not exist. Calls in one interpreter must use one
toolchain and must not overlap other Tensile generation: rocisa and the generator
have process-global state. A fresh subprocess is recommended for each request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from Tensile import __version__


class SingleSolutionError(RuntimeError):
    """A request could not produce a complete single-solution bundle."""


class SingleSolutionConfigError(SingleSolutionError):
    """Invalid or unsupported single-solution configuration."""


class SingleSolutionRejected(SingleSolutionConfigError):
    """A well-formed candidate failed normal solution validation."""


class SingleSolutionBuildError(SingleSolutionError):
    """Toolchain, code generation, compilation or publication failed."""


@dataclass(frozen=True)
class SingleSolutionBuildResult:
    bundlePath: Path
    manifestPath: Path
    codeObjectPaths: tuple[Path, ...]
    mainCodeObjectPath: Path
    libraryPath: Path
    logicalLibraryPath: Path
    kernelName: str
    solutionName: str
    solutionIndex: int
    architecture: str


_generationLock = threading.Lock()
_compilerIdentity = None


def _rejectParameterMode(name, choices, location):
    if name == "CustomKernel":
        raise SingleSolutionConfigError(f"{location}: custom kernels are not supported")
    if name == "NoReject" and any(choices):
        raise SingleSolutionConfigError(f"{location}: NoReject cannot disable solution validation")


def _parameterDict(value, location):
    if value is None:
        return {}
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SingleSolutionConfigError(f"{location} must be a list of parameter mappings")
    result = {}
    for item in value:
        for name, choices in item.items():
            if not isinstance(choices, list) or not choices:
                raise SingleSolutionConfigError(
                    f"{location}.{name} requires a nonempty choice list"
                )
            _rejectParameterMode(name, choices, f"{location}.{name}")
            # Check raw groups before duplicate keys/overrides can hide a mode.
            if name == "Groups":
                for parameterGroup in choices:
                    if isinstance(parameterGroup, list):
                        for grouped in parameterGroup:
                            if isinstance(grouped, dict):
                                for groupedName, groupedValue in grouped.items():
                                    _rejectParameterMode(
                                        groupedName,
                                        (
                                            groupedValue
                                            if isinstance(groupedValue, list)
                                            else [groupedValue]
                                        ),
                                        f"{location}.Groups.{groupedName}",
                                    )
            result[name] = choices
    return result


def _singleConfig(config, source):
    """Bound Groups before BenchmarkProcess can eagerly expand them."""
    from Tensile.BenchmarkStructs import _groupedParameterValueOptions

    if not isinstance(config, dict):
        raise SingleSolutionConfigError(f"{source}: expected a YAML mapping")
    problems = config.get("BenchmarkProblems")
    if not isinstance(problems, list) or len(problems) != 1:
        raise SingleSolutionConfigError("BenchmarkProblems must contain exactly one problem entry")
    entry = problems[0]
    if (
        not isinstance(entry, list)
        or len(entry) != 2
        or any(not isinstance(x, dict) for x in entry)
    ):
        raise SingleSolutionConfigError(
            "BenchmarkProblems[0] requires one problem type and one parameter group"
        )
    problem, group = entry
    prefix = "BenchmarkProblems[0][1]"
    if group.get("CustomKernels") or group.get("InternalSupportParams"):
        raise SingleSolutionConfigError(f"{prefix}: custom kernels are not supported")
    backend = config.get("Backend") or {}
    if (
        not isinstance(backend, dict)
        or backend.get("Name", "tensile") != "tensile"
        or backend.get("Config")
    ):
        raise SingleSolutionConfigError(
            "Only ordinary Tensile parameter configurations are supported"
        )
    for name in (
        "InitialSolutionParameters",
        "BenchmarkForkParameters",
        "JoinParameters",
        "BenchmarkJoinParameters",
    ):
        if group.get(name) is not None:
            raise SingleSolutionConfigError(f"{prefix}.{name} is no longer supported")

    common = _parameterDict(
        group.get("BenchmarkCommonParameters"), f"{prefix}.BenchmarkCommonParameters"
    )
    forks = _parameterDict(group.get("ForkParameters"), f"{prefix}.ForkParameters")
    groups = forks.pop("Groups", [])
    overridden = set()
    for groupIndex, parameterGroup in enumerate(groups):
        location = f"{prefix}.ForkParameters.Groups[{groupIndex}]"
        if not isinstance(parameterGroup, list) or len(parameterGroup) != 1:
            raise SingleSolutionConfigError(
                f"{location} must request exactly one parameter combination"
            )
        grouped = parameterGroup[0]
        if not isinstance(grouped, dict):
            raise SingleSolutionConfigError(f"{location}[0] must be a parameter mapping")
        for name, value in grouped.items():
            options = _groupedParameterValueOptions(name, value)
            if len(options) != 1:
                raise SingleSolutionConfigError(
                    f"{location}[0].{name} must request exactly one value"
                )
            overridden.add(name)
    effective = {**common, **forks}
    for name, choices in effective.items():
        if name not in overridden and len(choices) != 1:
            raise SingleSolutionConfigError(
                f"{prefix}.{name} requests {len(choices)} combinations; exactly one is required"
            )
    globalsConfig = config.get("GlobalParameters") or {}
    if not isinstance(globalsConfig, dict):
        raise SingleSolutionConfigError("GlobalParameters must be a mapping")
    for name in ("GenerateSourcesAndExit", "ForceGenerateKernel", "PythonProfile"):
        if globalsConfig.get(name):
            raise SingleSolutionConfigError(
                f"GlobalParameters.{name} conflicts with building one solution"
            )
    return problem, group, copy.deepcopy(globalsConfig)


def _target(architecture, globalsConfig):
    from Tensile.Common.Architectures import (
        architectureMap,
        baseArchName,
        gfxToCompilerTarget,
        gfxToIsa,
    )

    if not isinstance(architecture, str) or not re.fullmatch(
        r"gfx[0-9a-f]+(?:v0)?(?::(?:xnack|sramecc)[+-])*", architecture
    ):
        raise SingleSolutionConfigError(f"Expected one explicit GPU target, got {architecture!r}")
    base = baseArchName(architecture)
    if base not in architectureMap:
        raise SingleSolutionConfigError(f"Unsupported architecture {architecture!r}")
    isa = gfxToIsa(base)
    configured = globalsConfig.get("Architecture")
    if configured is not None and configured != architecture:
        raise SingleSolutionConfigError(
            "GlobalParameters.Architecture contradicts the requested architecture"
        )
    configuredIsa = globalsConfig.get("ISA")
    if configuredIsa is not None and configuredIsa != [list(isa)]:
        raise SingleSolutionConfigError(
            "GlobalParameters.ISA contradicts the requested architecture"
        )
    suffix = architecture[len(base) :]
    features = suffix.split(":")[1:]
    if len({feature[:-1] for feature in features}) != len(features):
        raise SingleSolutionConfigError("GPU target contains duplicate or contradictory features")
    return isa, gfxToCompilerTarget(base) + suffix


def _sourceRevision():
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _helperDescriptions(helpers):
    """Describe generators without confusing their names with exported symbols."""
    kinds = {
        "KernelWriterBetaOnly": "kernel_family",
        "KernelWriterConversion": "kernel_family",
        "KernelWriterReduction": "kernel_family",
        "KernelWriterActivationEnumHeader": "header",
        "KernelWriterActivationFunction": "device_function",
    }
    return [
        {
            "kind": kinds[type(helper).__name__],
            "generator": type(helper).__name__,
            "name": helper.getKernelName(),
        }
        for helper in helpers
    ]


def _deriveSingleSolution(config, source, architecture, toolchain, debug, isaInfoMap,
                          *, strictErrors=False):
    """Shared singleton derivation; no source emission, compilation, or benchmark."""
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructLazyForkPermutations
    from Tensile.BenchmarkProblems import _generate_single_solution

    problem, group, _ = _singleConfig(config, source)
    try:
        process = BenchmarkProcess(
            problem, group, debug.printIndexAssignmentInfo,
            keyPathPrefix="BenchmarkProblems[0][1]", srcFile=str(source),
            gfxName=architecture,
        )
        step = process[0]
        permutations = list(itertools.islice(
            constructLazyForkPermutations(step.forkParams, step.paramGroups), 2))
        if len(permutations) != 1:
            raise SingleSolutionConfigError("Parameters must expand to exactly one combination")
        solution = _generate_single_solution(
            permutations[0], process.problemType, step.constantParams,
            toolchain.assembler, debug, isaInfoMap, strictErrors=strictErrors,
        )
    except Exception as error:
        if strictErrors:
            # Unexpected failures, including missing resources, must not become
            # candidate rejections in the JIT driver.
            raise
        raise SingleSolutionConfigError(f"{source}: {error}") from error
    if solution is None or not solution["Valid"]:
        raise SingleSolutionRejected(
            "The requested solution is invalid; enable PrintSolutionRejectionReason in YAML for details"
        )
    return solution


def _build(
    configPath,
    staging,
    architecture,
    cxxCompiler,
    offloadBundler,
    codeObjectVersion,
    libraryFormat,
    keepBuildTmp,
    _selection=None,
):
    from Tensile import LibraryIO
    from Tensile.Common import getVerbosity, setVerbosity, state
    from Tensile.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
    from Tensile.Common.GlobalParameters import (
        assignGlobalParameters,
        globalParameters,
        restoreDefaultGlobalParameters,
    )
    from Tensile.Common.Types import makeDebugConfig
    from Tensile.Common.ValidParameters import validParameters
    from Tensile.KernelHelperNaming import KernelHelperEnum, initHelperKernelObjects
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionLibrary import MasterSolutionLibrary
    from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin
    from Tensile.TensileCreateLibrary.Run import writeSolutionsAndKernels
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Source import makeSourceToolchain
    from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain
    from Tensile.resources import copy_static_headers

    config = LibraryIO.read(str(configPath)) if _selection is None else _selection[0]
    _, _, globalsConfig = _singleConfig(config, configPath)
    isa, compilerTarget = _target(architecture, globalsConfig)
    if libraryFormat not in ("msgpack", "yaml"):
        raise SingleSolutionConfigError("libraryFormat must be msgpack or yaml")
    if codeObjectVersion not in ("4", "5", "6"):
        raise SingleSolutionConfigError("codeObjectVersion must be 4, 5 or 6")
    if globalsConfig.get("RuntimeLanguage", "HIP") != "HIP":
        raise SingleSolutionConfigError("The single-solution API supports RuntimeLanguage HIP")

    savedGlobals = copy.deepcopy(globalParameters)
    savedIsa = copy.deepcopy(validParameters["ISA"])
    savedVerbosity = getVerbosity()
    try:
        restoreDefaultGlobalParameters()
        setVerbosity(globalsConfig.get("PrintLevel", 1))
        compiler, bundler, _ = validateToolchain(
            cxxCompiler, offloadBundler, ToolchainDefaults.HIP_CONFIG
        )
        toolchain = makeAssemblyToolchain(compiler, bundler, codeObjectVersion)
        global _compilerIdentity
        identity = (str(Path(compiler).resolve()), tuple(toolchain.assembler.version))
        if _compilerIdentity is not None and _compilerIdentity != identity:
            raise SingleSolutionConfigError("Use a fresh Python process when changing the compiler")
        _compilerIdentity = identity
        isaInfoMap = makeIsaInfoMap([isa], compiler)
        applyArchCapOverrides(isaInfoMap, [architecture])
        globalsConfig.update(
            {
                "RuntimeLanguage": "HIP",
                "CpuThreads": 1,
                "PythonProfile": False,
                "CodeObjectVersion": codeObjectVersion,
                "LibraryFormat": libraryFormat,
                "GenerateSourcesAndExit": False,
                "HardwareMonitor": False,
                "PinClocks": False,
                "KeepBuildTmp": keepBuildTmp,
            }
        )
        assignGlobalParameters(globalsConfig, isaInfoMap)
        globalParameters["StinkyTofuArchName"] = (
            "gfx1250v0" if architecture.split(":")[0] == "gfx1250v0" else ""
        )
        debug = makeDebugConfig(globalsConfig)
        prediction = None
        if _selection is None:
            solution = _deriveSingleSolution(
                config, configPath, architecture, toolchain, debug, isaInfoMap)
        else:
            def derive(candidateConfig, label):
                return _deriveSingleSolution(
                    candidateConfig, label, architecture, toolchain, debug, isaInfoMap,
                    strictErrors=True)
            configPath, solution, prediction = _selection[1](derive)
        helpers = initHelperKernelObjects(solution, KernelHelperEnum.All, str(compiler), isaInfoMap)
        # Match the normal build's helper-family deduplication. One writer may
        # emit several exported kernels, while activation writers emit support.
        helpers = list({helper.getKernelName(): helper for helper in helpers}.values())
        helperDescriptions = _helperDescriptions(helpers)
        sourceToolchain = None
        supportFiles = []
        if helpers:
            supportFiles = copy_static_headers(staging) + ["Kernels.cpp", "Kernels.h"]
            sourceToolchain = makeSourceToolchain(compiler, bundler)
            sourceToolchain.compiler.default_args.append(
                f"-mcode-object-version={codeObjectVersion}"
            )
        solution["SolutionIndex"] = 0
        solution["SolutionNameMin"] = getSolutionNameMin(solution, debug.splitGSU)
        solution["KernelNameMin"] = getKernelNameMin(solution, debug.splitGSU)
        solutions = [solution]
        kernels = solution.getKernels()
        if len(kernels) != 1:
            raise SingleSolutionConfigError("The solution must contain exactly one main kernel")
        outputs, _ = writeSolutionsAndKernels(
            staging,
            toolchain,
            sourceToolchain,
            solutions,
            kernels,
            helpers,
            KernelWriterAssembly(toolchain.assembler, debug),
            debug.splitGSU,
            [compilerTarget],
            errorTolerant=False,
            compress=False,
            removeTemporaries=not keepBuildTmp,
            strict=True,
            assemblyTarget=compilerTarget,
        )
        mainOutputs = [Path(path) for path in outputs if Path(path).suffix == ".co"]
        if len(solutions) != 1 or len(mainOutputs) != 1:
            raise SingleSolutionBuildError(
                "Build did not produce exactly one solution and main code object"
            )
        library = MasterSolutionLibrary.BenchmarkingLibrary(
            solutions,
            toolchain.assembler,
            debug.splitGSU,
            debug.printSolutionRejectionReason,
            debug.printIndexAssignmentInfo,
            isaInfoMap,
        )
        library.applyNaming(debug.splitGSU)
        selected = next(iter(library.solutions.values()))
        mainOutput = mainOutputs[0]
        libraryBase = mainOutput.parent / "TensileLibrary"
        LibraryIO.write(str(libraryBase), state(library), libraryFormat)
        logical = libraryBase.with_suffix(".dat" if libraryFormat == "msgpack" else ".yaml")
        physical = Path(str(logical) + ".zlib") if libraryFormat == "msgpack" else logical
        for artifact in [*map(Path, outputs), physical, *(staging / name for name in supportFiles)]:
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise SingleSolutionBuildError(f"Missing or empty output artifact: {artifact}")
        manifest = {
            "schema_version": 2,
            "architecture": {
                "requested": architecture,
                "resolved": architecture,
                "compiler_target": compilerTarget,
            },
            "main_kernel": {
                "name": selected.kernelName,
                "code_object": str(mainOutput.relative_to(staging)),
            },
            "code_objects": [str(Path(path).relative_to(staging)) for path in outputs],
            "helpers": helperDescriptions,
            "support_files": supportFiles,
            "solution": {
                "index": selected.index,
                "name": selected.name,
                "kernel_name": selected.kernelName,
            },
            "library": {
                "format": libraryFormat,
                "logical_path": str(logical.relative_to(staging)),
                "path": str(physical.relative_to(staging)),
            },
            "counts": {
                "solutions": 1,
                "main_kernels": 1,
                "helper_generators": sum(h["kind"] == "kernel_family" for h in helperDescriptions),
                "support_generators": sum(h["kind"] != "kernel_family" for h in helperDescriptions),
            },
            "provenance": {
                "config_sha256": hashlib.sha256(configPath.read_bytes()).hexdigest(),
                "generator_version": __version__,
                "source_revision": _sourceRevision(),
                "compiler_path": str(compiler),
                "compiler_version": ".".join(map(str, toolchain.assembler.version)),
                "code_object_version": codeObjectVersion,
                "kernargs_version": solution["InternalSupportParams"]["KernArgsVersion"],
                "keep_build_tmp": keepBuildTmp,
            },
        }
        if prediction is not None:
            manifest["jit_prediction"] = prediction
        return manifest
    finally:
        globalParameters.clear()
        globalParameters.update(savedGlobals)
        validParameters["ISA"] = savedIsa
        setVerbosity(savedVerbosity)


def _writeLoaderEnvelope(path: Path, manifest: dict) -> None:
    """Write the private runtime envelope (UTF-8 strings, little-endian lengths)."""
    import struct

    fields = (
        str(manifest["schema_version"]), str(manifest["counts"]["solutions"]),
        str(manifest["counts"]["main_kernels"]), str(manifest["solution"]["index"]),
        manifest["main_kernel"]["name"], manifest["solution"]["kernel_name"],
        manifest["solution"]["name"], manifest["architecture"]["requested"],
        manifest["architecture"]["resolved"], manifest["architecture"]["compiler_target"],
        manifest["library"]["format"], manifest["main_kernel"]["code_object"],
        manifest["library"]["path"], manifest["library"]["logical_path"],
    )
    objects = manifest["code_objects"]
    if not 0 < len(objects) <= 4096:
        raise ValueError("Invalid loader code object count")
    with path.open("wb") as output:
        output.write(b"TLJIT001")
        output.write(struct.pack("<I", len(fields)))

        def write_string(value):
            encoded = value.encode("utf-8")
            if not 0 < len(encoded) <= 1048576 or b"\0" in encoded:
                raise ValueError("Invalid loader field")
            output.write(struct.pack("<I", len(encoded)))
            output.write(encoded)

        for value in fields:
            write_string(value)
        output.write(struct.pack("<I", len(objects)))
        for value in objects:
            write_string(value)


def generateAndBuildSingleSolution(
    configPath: str | Path,
    outputPath: str | Path,
    *,
    architecture: str,
    cxxCompiler: str = "amdclang++",
    offloadBundler: str = "clang-offload-bundler",
    codeObjectVersion: str = "4",
    libraryFormat: str = "msgpack",
    keepBuildTmp: bool = False,
) -> SingleSolutionBuildResult:
    """Build one explicit YAML recipe; no parameter prediction or benchmarking."""
    return _generateAndBuild(
        configPath, outputPath, architecture=architecture, cxxCompiler=cxxCompiler,
        offloadBundler=offloadBundler, codeObjectVersion=codeObjectVersion,
        libraryFormat=libraryFormat, keepBuildTmp=keepBuildTmp)


def _generateAndBuild(
    configPath: str | Path,
    outputPath: str | Path,
    *,
    architecture: str,
    cxxCompiler: str = "amdclang++",
    offloadBundler: str = "clang-offload-bundler",
    codeObjectVersion: str = "4",
    libraryFormat: str = "msgpack",
    keepBuildTmp: bool = False,
    _selection=None,
) -> SingleSolutionBuildResult:
    """Build one YAML-requested solution and publish ``outputPath/bundle`` atomically.

    ``outputPath`` must not exist, including after a previous failed attempt.
    Failure leaves private staging for diagnosis but never publishes a bundle.
    Paths in the manifest are relative to the published bundle, while paths in
    the Python result are absolute. No benchmarking or GPU launch is performed.
    """
    if not _generationLock.acquire(blocking=False):
        raise SingleSolutionError("Concurrent generation in one Python process is not supported")
    try:
        configPath = Path(configPath).resolve(strict=True)
        outputPath = Path(os.path.abspath(outputPath))
        outputPath.mkdir(parents=True, exist_ok=False)
        staging = outputPath / ".staging"
        staging.mkdir()
        manifest = _build(
            configPath,
            staging,
            architecture,
            cxxCompiler,
            offloadBundler,
            codeObjectVersion,
            libraryFormat,
            keepBuildTmp,
            *(() if _selection is None else (_selection,)),
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _writeLoaderEnvelope(staging / "loader.bin", manifest)
        bundle = outputPath / "bundle"
        staging.rename(bundle)
        return SingleSolutionBuildResult(
            bundle,
            bundle / "manifest.json",
            tuple(bundle / path for path in manifest["code_objects"]),
            bundle / manifest["main_kernel"]["code_object"],
            bundle / manifest["library"]["path"],
            bundle / manifest["library"]["logical_path"],
            manifest["main_kernel"]["name"],
            manifest["solution"]["name"],
            manifest["solution"]["index"],
            manifest["architecture"]["resolved"],
        )
    except SingleSolutionError:
        raise
    except SystemExit as error:
        raise SingleSolutionConfigError(
            f"Tensile rejected the request (exit {error.code}); see preceding diagnostics"
        ) from error
    except Exception as error:
        raise SingleSolutionBuildError(str(error)) from error
    finally:
        _generationLock.release()


def main(argv=None):
    """Translate the callable's result/errors into CLI output and an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configPath")
    parser.add_argument("outputPath")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--cxx-compiler", dest="cxxCompiler", default="amdclang++")
    parser.add_argument("--offload-bundler", dest="offloadBundler", default="clang-offload-bundler")
    parser.add_argument(
        "--code-object-version", dest="codeObjectVersion", choices=("4", "5", "6"), default="4"
    )
    parser.add_argument(
        "--library-format", dest="libraryFormat", choices=("msgpack", "yaml"), default="msgpack"
    )
    parser.add_argument("--keep-build-tmp", dest="keepBuildTmp", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generateAndBuildSingleSolution(**vars(args))
    except SingleSolutionError as error:
        print(f"Single-solution build failed: {error}", file=sys.stderr)
        return 1
    print(result.manifestPath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
