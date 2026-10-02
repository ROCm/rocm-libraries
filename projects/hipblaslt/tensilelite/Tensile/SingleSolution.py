# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build one GEMM solution, including its helpers, without benchmarking.

The callable returns a published bundle; ``python -m Tensile.SingleSolution`` is its
CLI. With ``sourceOnly`` (``--source-only``) the bundle holds the solution's
sources and library entry instead of code objects. The output directory must not
exist. Calls in one interpreter must use one toolchain and must not overlap other
Tensile generation: rocisa and the generator have process-global state. A fresh
subprocess is recommended for each request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from . import JitDebug, __version__


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
    codeObjectPaths: tuple[Path, ...]  # empty for a source bundle
    mainCodeObjectPath: Path | None  # None for a source bundle
    libraryPath: Path
    logicalLibraryPath: Path
    kernelName: str
    solutionName: str
    solutionIndex: int
    architecture: str
    sourcePaths: tuple[Path, ...] = ()  # the main assembly first; empty for code objects


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
    from .BenchmarkStructs import _groupedParameterValueOptions

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
    from .Common.Architectures import (
        architectureMap,
        baseArchName,
        compilerTargetOf,
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
    return isa, compilerTargetOf(base) + suffix


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
    from .BenchmarkStructs import BenchmarkProcess, constructLazyForkPermutations
    from .BenchmarkProblems import _generate_single_solution

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
    sourceOnly=False,
    _selection=None,
    _debug=JitDebug.NULL,
):
    with _debug.span("imports"):
        from . import LibraryIO
        from .Common import getVerbosity, setVerbosity, state
        from .Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
        from .Common.GlobalParameters import (
            assignGlobalParameters,
            globalParameters,
            restoreDefaultGlobalParameters,
        )
        from .Common.Types import makeDebugConfig
        from .Common.ValidParameters import validParameters
        from .KernelHelperNaming import KernelHelperEnum, initHelperKernelObjects
        from .KernelWriterAssembly import KernelWriterAssembly
        from .SolutionLibrary import MasterSolutionLibrary
        from .SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin
        from .TensileCreateLibrary.Run import writeSolutionsAndKernels
        from .Toolchain.Assembly import AssemblyToolchain, makeAssemblyToolchain
        from .Toolchain.Component import Assembler
        from .Toolchain.Source import makeSourceToolchain
        from .Toolchain.Validators import ToolchainDefaults, validateToolchain
        from .resources import copy_static_headers

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
        with _debug.span("setup"):
            restoreDefaultGlobalParameters()
            setVerbosity(globalsConfig.get("PrintLevel", 1))
            if sourceOnly:
                # The compiler still probes assembler capabilities; nothing is built.
                compiler, _ = validateToolchain(cxxCompiler, ToolchainDefaults.HIP_CONFIG)
                bundler = None
                toolchain = AssemblyToolchain(Assembler(compiler, codeObjectVersion), None, None)
            else:
                compiler, bundler, _ = validateToolchain(
                    cxxCompiler, offloadBundler, ToolchainDefaults.HIP_CONFIG
                )
                toolchain = makeAssemblyToolchain(compiler, bundler, codeObjectVersion)
            global _compilerIdentity
            identity = (str(Path(compiler).resolve()), tuple(toolchain.assembler.version))
            if _compilerIdentity is not None and _compilerIdentity != identity:
                raise SingleSolutionConfigError(
                    "Use a fresh Python process when changing the compiler")
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
        with _debug.span("select", rejects=(SingleSolutionRejected,)):
            if _selection is None:
                solution = _deriveSingleSolution(
                    config, configPath, architecture, toolchain, debug, isaInfoMap)
            else:
                def derive(candidateConfig, label):
                    derived = _deriveSingleSolution(
                        candidateConfig, label, architecture, toolchain, debug, isaInfoMap,
                        strictErrors=True)
                    # Must match the name the published library gives the kernel
                    # (MasterSolutionLibrary.applyNaming names the kernel view).
                    derived["KernelNameMin"] = getKernelNameMin(
                        derived.getKernels()[0], debug.splitGSU)
                    return derived
                choice = _selection[1](derive)
        if _selection is not None:
            if choice is None:
                return None
            configPath, solution, prediction = choice
        with _debug.span("helpers"):
            helpers = initHelperKernelObjects(
                solution, KernelHelperEnum.All, str(compiler), isaInfoMap)
            # Match the normal build's helper-family deduplication. One writer may
            # emit several exported kernels, while activation writers emit support.
            helpers = list({helper.getKernelName(): helper for helper in helpers}.values())
            helperDescriptions = _helperDescriptions(helpers)
            sourceToolchain = None
            supportFiles = []
            sources = staging / "sources"
            if helpers and sourceOnly:
                supportFiles = copy_static_headers(sources) + ["Kernels.cpp", "Kernels.h"]
            elif helpers:
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
        with _debug.span("kernel_source"):
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
                sourcesOnly=sourceOnly,
            )
        if sourceOnly:
            if len(solutions) != 1 or len(outputs) != 1:
                raise SingleSolutionBuildError(
                    "Generation did not produce exactly one solution and main kernel source"
                )
            with _debug.span("copy_sources"):
                sources.mkdir(exist_ok=True)
                mainOutput = sources / Path(outputs[0]).name
                shutil.copyfile(outputs[0], mainOutput)
                for name in ("Kernels.cpp", "Kernels.h") if helpers else ():
                    (staging / name).replace(sources / name)
                if not keepBuildTmp:
                    shutil.rmtree(staging / "build_tmp")
            outputs = [mainOutput, *(sources / name for name in supportFiles)]
            libraryBase = staging / "library" / "TensileLibrary"
        else:
            mainOutputs = [Path(path) for path in outputs if Path(path).suffix == ".co"]
            if len(solutions) != 1 or len(mainOutputs) != 1:
                raise SingleSolutionBuildError(
                    "Build did not produce exactly one solution and main code object"
                )
            mainOutput = mainOutputs[0]
            libraryBase = mainOutput.parent / "TensileLibrary"
        with _debug.span("library_write"):
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
            LibraryIO.write(str(libraryBase), state(library), libraryFormat)
        logical = libraryBase.with_suffix(".dat" if libraryFormat == "msgpack" else ".yaml")
        physical = Path(str(logical) + ".zlib") if libraryFormat == "msgpack" else logical
        supportPaths = [] if sourceOnly else [staging / name for name in supportFiles]
        for artifact in [*map(Path, outputs), physical, *supportPaths]:
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise SingleSolutionBuildError(f"Missing or empty output artifact: {artifact}")
        target = {
            "requested": architecture,
            "resolved": architecture,
            "compiler_target": compilerTarget,
        }
        mainKernel = {"name": selected.kernelName}
        if sourceOnly:
            # The sources and library entry are the build input; the rest is provenance.
            manifest = {
                "schema_version": 3,
                "mode": "source",
                "architecture": target,
                "main_kernel": mainKernel,
                "sources": [str(path.relative_to(staging)) for path in outputs],
                "helpers": helperDescriptions,
            }
        else:
            mainKernel["code_object"] = str(mainOutput.relative_to(staging))
            manifest = {
                "schema_version": 2,
                "architecture": target,
                "main_kernel": mainKernel,
                "code_objects": [str(Path(path).relative_to(staging)) for path in outputs],
                "helpers": helperDescriptions,
                "support_files": supportFiles,
            }
        with _debug.span("provenance"):
            configSha256 = hashlib.sha256(configPath.read_bytes()).hexdigest()
            sourceRevision = _sourceRevision()
        manifest |= {
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
                "config_sha256": configSha256,
                "generator_version": __version__,
                "source_revision": sourceRevision,
                "compiler_path": str(compiler),
                "compiler_version": ".".join(map(str, toolchain.assembler.version)),
                "code_object_version": codeObjectVersion,
                "kernargs_version": solution["InternalSupportParams"]["KernArgsVersion"],
                "persistent_loop_args_version":
                    solution["InternalSupportParams"].get("PersistentLoopArgsVersion", 0),
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
    sourceOnly: bool = False,
) -> SingleSolutionBuildResult:
    """Build one explicit YAML recipe; no parameter prediction or benchmarking."""
    return _generateAndBuild(
        configPath, outputPath, architecture=architecture, cxxCompiler=cxxCompiler,
        offloadBundler=offloadBundler, codeObjectVersion=codeObjectVersion,
        libraryFormat=libraryFormat, keepBuildTmp=keepBuildTmp, sourceOnly=sourceOnly)


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
    sourceOnly: bool = False,
    _selection=None,
    _count=None,
    _debug=JitDebug.NULL,
) -> SingleSolutionBuildResult:
    """Build one YAML-requested solution and publish ``outputPath/bundle`` atomically.

    ``outputPath`` must not exist, including after a previous failed attempt.
    Failure leaves private staging for diagnosis but never publishes a bundle.
    Paths in the manifest are relative to the published bundle, while paths in
    the Python result are absolute. No benchmarking or GPU launch is performed.
    With ``sourceOnly`` nothing is assembled, compiled, or bundled: the bundle
    holds ``sources/`` and ``library/`` for the caller to build.

    With ``_count``, ``_selection`` is asked for up to that many solutions, one
    build each, until it returns None. They are published together as
    ``outputPath/bundle-<rank>``, and ``outputPath/bundle`` links to
    ``bundle-0``, which the result describes.
    """
    if not _generationLock.acquire(blocking=False):
        raise SingleSolutionError("Concurrent generation in one Python process is not supported")
    try:
        configPath = Path(configPath).resolve(strict=True)
        outputPath = Path(os.path.abspath(outputPath))
        outputPath.mkdir(parents=True, exist_ok=False)
        names = ["bundle"] if _count is None else [f"bundle-{rank}" for rank in range(_count)]
        staged = []
        for rank, name in enumerate(names):
            _debug.bundle(rank)
            staging = outputPath / (".staging" if _count is None else f".staging-{name}")
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
                sourceOnly,
                *(() if _selection is None else (_selection,)),
                _debug=_debug,
            )
            if manifest is None:
                staging.rmdir()
                break
            with _debug.span("manifest_write"):
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
            staged.append((staging, name, manifest))
        _debug.bundle(None)
        if not staged:
            raise SingleSolutionBuildError("No solution was selected")
        bundle = outputPath / "bundle"
        with _debug.span("publish"):
            for staging, name, _ in staged:
                staging.rename(outputPath / name)
            if _count is not None:
                bundle.symlink_to(names[0], target_is_directory=True)
        _debug.published(len(staged))
        manifest = staged[0][2]
        mainCodeObject = manifest["main_kernel"].get("code_object")
        return SingleSolutionBuildResult(
            bundle,
            bundle / "manifest.json",
            tuple(bundle / path for path in manifest.get("code_objects", ())),
            bundle / mainCodeObject if mainCodeObject else None,
            bundle / manifest["library"]["path"],
            bundle / manifest["library"]["logical_path"],
            manifest["main_kernel"]["name"],
            manifest["solution"]["name"],
            manifest["solution"]["index"],
            manifest["architecture"]["resolved"],
            tuple(bundle / path for path in manifest.get("sources", ())),
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
    parser.add_argument("--source-only", dest="sourceOnly", action="store_true")
    JitDebug.addArguments(parser)
    args = vars(parser.parse_args(argv))
    debug = JitDebug.fromArguments(
        parser, args, module="Tensile.SingleSolution", mode="explicit")
    status, errorType = "failed", None
    try:
        debug.request(requested=1)
        result = _generateAndBuild(**args, _debug=debug)
        status = "ok"
    except SingleSolutionError as error:
        errorType = type(error).__name__
        print(f"Single-solution build failed: {error}", file=sys.stderr)
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
