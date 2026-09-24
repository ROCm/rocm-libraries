# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Behavioral contract tests for one-kernel requests and bundle publication."""

import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import msgpack
import pytest
import yaml

from Tensile import LibraryIO, SingleSolution as SS


pytestmark = pytest.mark.unit
CONFIG = Path(__file__).parent / "test_data" / "single_solution.yaml"


@pytest.fixture
def config():
    return LibraryIO.read(str(CONFIG))


@pytest.mark.parametrize("value", [[], [None, None]])
def test_reject_multiple_or_empty_problem_entries(config, value):
    config["BenchmarkProblems"] = value
    with pytest.raises(SS.SingleSolutionConfigError, match="exactly one"):
        SS._singleConfig(config, CONFIG)


@pytest.mark.parametrize("choices", [[], [1, 2]])
def test_reject_empty_or_multiple_requested_choices(config, choices):
    config["BenchmarkProblems"][0][1]["ForkParameters"].append({"GlobalSplitU": choices})
    with pytest.raises(SS.SingleSolutionConfigError):
        SS._singleConfig(config, CONFIG)


def test_empty_matrix_instruction_is_one_candidate(config):
    config["BenchmarkProblems"][0][1]["ForkParameters"] = [{"MatrixInstruction": [[]]}]
    _, group, _ = SS._singleConfig(config, CONFIG)
    assert group["ForkParameters"] == [{"MatrixInstruction": [[]]}]


def test_groups_are_bounded_before_eager_expansion(config, monkeypatch):
    import Tensile.BenchmarkStructs as BS

    def unexpected(*args, **kwargs):
        pytest.fail("must reject before expanding Groups")

    monkeypatch.setattr(BS, "_expandGroupedParameters", unexpected)
    config["BenchmarkProblems"][0][1]["ForkParameters"].append(
        {"Groups": [[{"NonTemporalA": list(range(100000)), "NonTemporalB": list(range(100000))}]]}
    )
    with pytest.raises(SS.SingleSolutionConfigError, match="exactly one value"):
        SS._singleConfig(config, CONFIG)


def test_groups_override_fork_choices_using_existing_semantics(config):
    group = config["BenchmarkProblems"][0][1]
    group["ForkParameters"] = [
        {"GlobalSplitU": [1, 2]},
        {"Groups": [[{"GlobalSplitU": 1, "MatrixInstruction": [16, 16, 16, 1, 1, 2, 1, 4, 1]}]]},
    ]
    SS._singleConfig(config, CONFIG)


def test_custom_wildcard_rejected_without_resource_scan(config, monkeypatch):
    import Tensile.BenchmarkStructs as BS

    monkeypatch.setattr(BS, "getAllCustomKernelNames", lambda: pytest.fail("custom resource scan"))
    config["BenchmarkProblems"][0][1]["CustomKernels"] = ["*"]
    with pytest.raises(SS.SingleSolutionConfigError, match="custom kernels"):
        SS._singleConfig(config, CONFIG)


@pytest.mark.parametrize("location", ["BenchmarkCommonParameters", "ForkParameters", "Groups"])
@pytest.mark.parametrize(
    "name,value", [("NoReject", True), ("CustomKernel", {"name": "handwritten"})]
)
def test_bypass_modes_rejected_before_solution_construction(
    config, tmp_path, monkeypatch, location, name, value
):
    from Tensile import BenchmarkProblems
    from Tensile.KernelWriterAssembly import KernelWriterAssembly

    def unexpected(*args, **kwargs):
        pytest.fail("forbidden parameter mode reached solution construction or custom source")

    monkeypatch.setattr(BenchmarkProblems, "_generate_single_solution", unexpected)
    monkeypatch.setattr(KernelWriterAssembly, "_getCustomKernelSource", unexpected)
    group = config["BenchmarkProblems"][0][1]
    if location == "Groups":
        group["ForkParameters"].append({"Groups": [[{name: value}]]})
    else:
        group.setdefault(location, []).append({name: [value]})
    source = tmp_path / "forbidden.yaml"
    LibraryIO.writeYAML(str(source), config)
    with pytest.raises(SS.SingleSolutionConfigError, match="NoReject|custom kernels"):
        SS.generateAndBuildSingleSolution(source, tmp_path / "output", architecture="gfx942")
    assert not (tmp_path / "output" / "bundle").exists()


def test_explicit_validation_and_overridden_bypass_modes(config):
    forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
    forks.append({"NoReject": [False]})
    SS._singleConfig(config, CONFIG)
    forks.insert(0, {"NoReject": [True]})
    with pytest.raises(SS.SingleSolutionConfigError, match="NoReject"):
        SS._singleConfig(config, CONFIG)


@pytest.mark.parametrize(
    "target", ["all", "gfx942;gfx950", "gfx942:unknown+", "gfx942:xnack+:xnack-"]
)
def test_reject_invalid_or_multiple_targets(target):
    with pytest.raises(SS.SingleSolutionConfigError):
        SS._target(target, {})


def test_target_preserves_features_and_resolves_stepping():
    isa, compiler = SS._target("gfx1250v0:xnack-", {})
    assert tuple(isa) == (12, 5, 0)
    assert compiler == "gfx1250:xnack-"


@pytest.fixture
def stub_build(monkeypatch):
    def build(configPath, staging, *args):
        library = staging / "library" / "gfx942"
        library.mkdir(parents=True)
        (library / "single.co").write_bytes(b"code object")
        (library / "TensileLibrary.dat.zlib").write_bytes(b"library")
        return {
            "schema_version": 2,
            "architecture": {"requested": "gfx942", "resolved": "gfx942", "compiler_target": "gfx942"},
            "counts": {"solutions": 1, "main_kernels": 1},
            "main_kernel": {"name": "single", "code_object": "library/gfx942/single.co"},
            "code_objects": ["library/gfx942/single.co"],
            "solution": {"index": 0, "name": "solution", "kernel_name": "single"},
            "library": {
                "format": "msgpack",
                "path": "library/gfx942/TensileLibrary.dat.zlib",
                "logical_path": "library/gfx942/TensileLibrary.dat",
            },
        }

    monkeypatch.setattr(SS, "_build", build)


def test_result_is_published_atomically_and_paths_survive_move(tmp_path, stub_build):
    output = tmp_path / "output with spaces"
    result = SS.generateAndBuildSingleSolution(CONFIG, output, architecture="gfx942")
    assert result.bundlePath == output / "bundle"
    assert result.mainCodeObjectPath.is_file()
    assert all(path.is_file() for path in result.codeObjectPaths)
    assert result.libraryPath.is_file()
    assert not result.logicalLibraryPath.exists()
    assert not (output / ".staging").exists()
    manifest = json.loads(result.manifestPath.read_text())
    assert manifest["schema_version"] == 2
    assert not Path(manifest["main_kernel"]["code_object"]).is_absolute()
    import struct

    # Decode independently: version and field count precede length-prefixed UTF-8.
    encoded = (result.bundlePath / "loader.bin").read_bytes()
    assert encoded[:8] == b"TLJIT001"
    assert struct.unpack_from("<I", encoded, 8)[0] == 14
    values = []
    offset = 12
    for _ in range(14):
        length = struct.unpack_from("<I", encoded, offset)[0]
        offset += 4
        values.append(encoded[offset:offset + length].decode("utf-8"))
        offset += length
    assert values[4:7] == ["single", "single", "solution"]
    assert values[7:10] == ["gfx942"] * 3
    assert values[10] == "msgpack"
    assert struct.unpack_from("<I", encoded, offset)[0] == 1
    offset += 4
    length = struct.unpack_from("<I", encoded, offset)[0]
    assert encoded[offset + 4:] == b"library/gfx942/single.co"
    assert len(encoded) == offset + 4 + length
    before = result.manifestPath.read_bytes()
    with pytest.raises(SS.SingleSolutionBuildError):
        SS.generateAndBuildSingleSolution(CONFIG, output, architecture="gfx942")
    assert result.manifestPath.read_bytes() == before


@pytest.mark.parametrize("error", [RuntimeError("assembler failed"), SystemExit(-1)])
def test_failed_request_never_publishes_and_does_not_exit_caller(tmp_path, monkeypatch, error):
    def fail(config, staging, *args):
        (staging / "partial.co").write_bytes(b"partial")
        raise error

    monkeypatch.setattr(SS, "_build", fail)
    output = tmp_path / "failed"
    with pytest.raises(SS.SingleSolutionError):
        SS.generateAndBuildSingleSolution(CONFIG, output, architecture="gfx942")
    assert not (output / "bundle").exists()
    assert not list(output.rglob("manifest.json"))
    assert (output / ".staging" / "partial.co").exists()


def test_cli_translates_typed_errors_to_status(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise SS.SingleSolutionConfigError("two requested configurations")

    monkeypatch.setattr(SS, "generateAndBuildSingleSolution", fail)
    assert SS.main([str(CONFIG), "unused", "--architecture", "gfx942"]) == 1
    assert "two requested configurations" in capsys.readouterr().err


def test_tool_version_probe_treats_path_as_literal_argv(tmp_path, monkeypatch):
    from Tensile.Toolchain import Component

    executable = tmp_path / 'compiler $(false) "quoted"'
    executable.write_text("#!/bin/sh\nprintf 'clang version 18.1.2\\n'\n")
    executable.chmod(0o755)
    monkeypatch.setattr(Component, "validateToolchain", lambda path: path)
    assert tuple(Component._getVersion(str(executable), "--version", r"version\s+([\d.]+)")) == (
        18,
        1,
        2,
    )


@pytest.mark.parametrize("parameter", [{"NoSuchParameter": [1]}, {"GlobalSplitU": [True]}])
def test_existing_parameter_validation_runs_before_codegen(
    config, parameter, tmp_path, monkeypatch
):
    from Tensile.TensileCreateLibrary import Run

    monkeypatch.setattr(
        Run, "processKernelSource", lambda *a, **k: pytest.fail("invalid YAML reached codegen")
    )
    config["BenchmarkProblems"][0][1]["ForkParameters"].append(parameter)
    source = tmp_path / "invalid.yaml"
    LibraryIO.writeYAML(str(source), config)
    with pytest.raises(SS.SingleSolutionConfigError):
        SS.generateAndBuildSingleSolution(source, tmp_path / "output", architecture="gfx942")
    assert not (tmp_path / "output" / "bundle").exists()


@pytest.mark.parametrize("failure", ["source", "compile", "unbundle", "empty", "missing"])
def test_helper_build_failures_cannot_publish_a_bundle(tmp_path, monkeypatch, failure):
    from Tensile.TensileCreateLibrary import Run
    from Tensile.Toolchain.Component import Compiler, Bundler
    from Tensile.KernelWriterConversion import KernelWriterConversion

    def fail(*args, **kwargs):
        raise RuntimeError("injected helper failure")

    if failure == "source":
        monkeypatch.setattr(KernelWriterConversion, "getSourceFileString", lambda *a: (1, ""))
    elif failure == "compile":
        monkeypatch.setattr(Compiler, "__call__", fail)
    elif failure == "unbundle":
        monkeypatch.setattr(Bundler, "__call__", fail)
    elif failure == "empty":
        monkeypatch.setattr(Run, "buildSourceCodeObjectFiles", lambda *a, **k: [])
    else:
        monkeypatch.setattr(
            Run, "buildSourceCodeObjectFiles", lambda *a, **k: [tmp_path / "missing.hsaco"]
        )
    with pytest.raises(SS.SingleSolutionBuildError, match="helper|Helper|nonempty code object"):
        SS.generateAndBuildSingleSolution(
            CONFIG.with_name("single_solution_splitk.yaml"),
            tmp_path / "output",
            architecture="gfx942",
        )
    assert not (tmp_path / "output" / "bundle").exists()
    assert not list((tmp_path / "output").rglob("manifest.json"))


def test_real_assembly_failure_cannot_publish_a_bundle(tmp_path, monkeypatch):
    from Tensile.Toolchain.Component import Assembler

    def fail(*args, **kwargs):
        raise RuntimeError("injected assembly failure")

    monkeypatch.setattr(Assembler, "__call__", fail)
    with pytest.raises(SS.SingleSolutionBuildError, match="injected assembly failure"):
        SS.generateAndBuildSingleSolution(CONFIG, tmp_path / "output", architecture="gfx942")
    assert not (tmp_path / "output" / "bundle").exists()


def test_legacy_writer_still_tolerates_assembly_failure_and_builds_helpers(tmp_path, monkeypatch):
    from Tensile.TensileCreateLibrary import Run
    from Tensile.Toolchain.Component import Assembler

    called = []
    writer = Run.writeSolutionsAndKernels

    def legacy_writer(output, asmToolchain, srcToolchain, *args, **kwargs):
        kwargs.pop("strict")
        return writer(
            output, asmToolchain, SimpleNamespace(compiler=None, bundler=None), *args, **kwargs
        )

    def fail(*args, **kwargs):
        raise RuntimeError("tolerated assembly failure")

    monkeypatch.setattr(Assembler, "__call__", fail)
    monkeypatch.setattr(Run, "writeSolutionsAndKernels", legacy_writer)
    monkeypatch.setattr(Run, "writeHelpers", lambda *a: called.append("write helpers"))
    monkeypatch.setattr(
        Run, "buildSourceCodeObjectFiles", lambda *a: called.append("build helpers")
    )
    monkeypatch.setattr(Run, "buildAssemblyCodeObjectFiles", lambda *a: [])
    with pytest.raises(
        SS.SingleSolutionBuildError, match="exactly one solution and main code object"
    ):
        SS.generateAndBuildSingleSolution(CONFIG, tmp_path / "output", architecture="gfx942")
    assert called == ["write helpers", "build helpers"]


@pytest.mark.parametrize("recipe", ["plain", "splitk", "adaptive", "activation"])
def test_real_build_has_one_solution_and_complete_helpers_without_benchmarking(tmp_path, recipe):
    """Cross-compile complete solutions; no GPU or benchmark client is needed."""
    compiler = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    if not Path(compiler).is_file():
        pytest.skip("ROCm compiler is unavailable")
    # Use a subprocess so unrelated unit tests cannot supply stale rocisa caps.
    source = (
        CONFIG
        if recipe in ("plain", "activation")
        else CONFIG.with_name(f"single_solution_{recipe}.yaml")
    )
    if recipe == "activation":
        config = LibraryIO.read(str(source))
        config["BenchmarkProblems"][0][0].update(Activation=True, ActivationType="all")
        config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"].append(
            {"ActivationArgs": [[{"Enum": "relu"}]]}
        )
        source = tmp_path / "activation.yaml"
        LibraryIO.writeYAML(str(source), config)
    output = tmp_path / "real output"
    script = """
import sys
from Tensile import SingleSolution
from Tensile import BenchmarkProblems, ClientWriter, LibraryLogic
from Tensile.TensileCreateLibrary import Run
from Tensile.Toolchain import Source
def forbidden(*args, **kwargs):
    raise AssertionError('benchmark, cache or unnecessary helper compilation was called')
BenchmarkProblems.main = forbidden
BenchmarkProblems.writeBenchmarkFiles = forbidden
ClientWriter.runClient = forbidden
LibraryLogic.main = forbidden
Source.HelperKernelCache = forbidden
if sys.argv[4] == 'plain':
    Run.writeHelpers = forbidden
    Run.buildSourceCodeObjectFiles = forbidden
SingleSolution.generateAndBuildSingleSolution(sys.argv[1], sys.argv[2], architecture='gfx942', cxxCompiler=sys.argv[3], keepBuildTmp=True)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(source), str(output), compiler, recipe],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    bundle = output / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["counts"]["solutions"] == manifest["counts"]["main_kernels"] == 1
    artifacts = {bundle / path for path in manifest["code_objects"]}
    assert artifacts == set(bundle.rglob("*.co")) | set(bundle.rglob("*.hsaco"))
    assert all(path.read_bytes().startswith(b"\x7fELF") for path in artifacts)
    assert all((bundle / path).is_file() for path in manifest["support_files"])
    codeObject = bundle / manifest["main_kernel"]["code_object"]
    assert codeObject.read_bytes().startswith(b"\x7fELF")
    readobj = shutil.which("llvm-readobj") or str(Path(compiler).resolve().parent / "llvm-readobj")
    notes = subprocess.run(
        [readobj, "--notes", str(codeObject)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    metadata = yaml.safe_load(notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0])
    assert len(metadata["amdhsa.kernels"]) == 1
    assert metadata["amdhsa.kernels"][0][".name"] == manifest["main_kernel"]["name"]
    library = msgpack.unpackb(
        zlib.decompress((bundle / manifest["library"]["path"]).read_bytes()), raw=False
    )
    assert len(library["solutions"]) == 1
    assert library["solutions"][0]["kernelName"] == manifest["main_kernel"]["name"]
    assembly = list(bundle.rglob("*.s"))
    assert len(assembly) == 1
    assert assembly[0].read_text().count(".amdhsa_kernel ") == 1
    if recipe == "plain":
        assert manifest["counts"]["helper_generators"] == 0
        assert manifest["counts"]["support_generators"] == 0
        assert not list(bundle.rglob("*.hsaco"))
    elif recipe == "activation":
        assert manifest["counts"]["support_generators"] == 2
        assert manifest["counts"]["helper_generators"] == 0
        assert {helper["kind"] for helper in manifest["helpers"]} == {"header", "device_function"}
    else:
        assert manifest["counts"]["helper_generators"] > 0
        helperObject = next(bundle.rglob("*.hsaco"))
        notes = subprocess.run(
            [readobj, "--notes", str(helperObject)], check=True, capture_output=True, text=True
        ).stdout
        metadata = yaml.safe_load(notes.split("AMDGPU Metadata: ", 1)[1].split("...", 1)[0])
        # One conversion generator emits multiple GSU/vector-width entrypoints.
        assert len(metadata["amdhsa.kernels"]) > manifest["counts"]["helper_generators"]
        assert any("PostGSU" in kernel[".name"] for kernel in metadata["amdhsa.kernels"])
    assert not list(bundle.rglob("ClientParameters.ini"))
    assert not list(bundle.rglob("cache.yaml"))
    assert not list(bundle.rglob("*.csv"))
