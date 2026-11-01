# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Chunked helper-kernel compile (timing_runs/DECISIONS.md topic 6, C; P4.14).

writeHelpers splits the helper kernels into chunks by a hash of their names, each with a header
holding only what its kernels need; compileHelperChunks compiles the chunks with the driver's own
compile jobs in parallel and links them with its link and bundle jobs; buildSourceCodeObjectFiles
compiles Kernels.cpp in one piece if anything in that fails."""

import os
import re
import subprocess
import types
from pathlib import Path

import pytest
from Tensile.Common.DataType import DataType
from Tensile.KernelWriterReduction import KernelWriterReduction
from Tensile.TensileCreateLibrary.Run import copyStaticFiles, writeHelpers
from Tensile.Toolchain import HelperKernelChunks, Source
from Tensile.Toolchain.Component import Bundler, Compiler
from Tensile.Toolchain.HelperKernelChunks import (
    chunkOf,
    compileHelperChunks,
    supportsChunkedCompile,
)

pytestmark = pytest.mark.unit


class _Helper:
    """A helper kernel writer: one kernel, its argument struct + declaration in the header."""

    sharedHeader = False

    def __init__(self, name):
        self.name = name

    def getKernelName(self):
        return self.name

    def getSourceFileString(self):
        return (0, f"\n// definition of {self.name}\n")

    def getHeaderFileString(self):
        return f"struct argument_{self.name};\n// declaration of {self.name}\n"


class _SharedHelper(_Helper):
    """Like the activation helpers: header defines what other kernels use, the source is a
    comment with no trailing newline."""

    sharedHeader = True

    def getSourceFileString(self):
        return (0, "// This is a dummy file.")

    def getHeaderFileString(self):
        return f"// shared {self.name}"


NAMES = [f"Cijk_Kernel{i}" for i in range(12)]


def _helpers(names=NAMES):
    return [_SharedHelper("Enum"), _SharedHelper("Activation")] + [_Helper(n) for n in names]


def _write(tmp_path, helpers, chunkCount):
    out = tmp_path / "out"
    out.mkdir(parents=True)
    chunks = writeHelpers(out, helpers, "Kernels.cpp", "Kernels.h", chunkCount, tmp_path / "chunks")
    return out, chunks


def _files(directory):
    return {p.name: p.read_text() for p in sorted(Path(directory).iterdir())}


def test_chunk_of_is_in_range_and_spreads_kernels():
    chunks = [chunkOf(name, 4) for name in NAMES]
    assert chunks == [chunkOf(name, 4) for name in NAMES]
    assert set(chunks) == {0, 1, 2, 3}
    assert {chunkOf(name, 1) for name in NAMES} == {0}


def test_chunking_does_not_change_kernels_cpp_and_kernels_h(tmp_path):
    one, chunks = _write(tmp_path / "a", _helpers(), 1)
    assert chunks == []
    assert not (tmp_path / "a" / "chunks").exists()
    four, chunks = _write(tmp_path / "b", _helpers(), 4)
    assert len(chunks) == 4
    assert _files(one) == _files(four)


def test_each_chunk_has_its_kernels_and_only_their_declarations(tmp_path):
    helpers = _helpers()
    _, chunks = _write(tmp_path, helpers, 4)
    assert [p.name for p in chunks] == [f"Kernels_chunk{i}.cpp" for i in range(4)]

    for index, chunk in enumerate(chunks):
        source = chunk.read_text()
        header = chunk.with_suffix(".h").read_text()
        own = [n for n in NAMES if chunkOf(n, 4) == index]
        assert f'#include "Kernels_chunk{index}.h"\n' in source
        assert re.findall(r"definition of (\w+)", source) == own
        assert re.findall(r"declaration of (\w+)", header) == own
        assert re.findall(r"struct argument_(\w+)", header) == own
        # The static headers and the shared headers (each on its own line), before the own
        # declarations, as in Kernels.h.
        assert header.index('#include "KernelHeader.h"') < header.index("// shared Enum\n")
        assert header.index("// shared Enum\n") < header.index("// shared Activation\n")
        assert header.index("// shared Activation\n") < header.index("struct argument_")
        # The shared helpers' sources (comments with no newline) go to the first chunk, each
        # on its own line.
        assert source.count("// This is a dummy file.\n") == (2 if index == 0 else 0)
        assert "dummy file.\n// definition" not in source


def test_adding_a_kernel_changes_only_its_chunk(tmp_path):
    _write(tmp_path / "a", _helpers(NAMES), 4)
    added = "Cijk_Kernel_new"
    _write(tmp_path / "b", _helpers(NAMES[:5] + [added] + NAMES[5:]), 4)

    before, after = _files(tmp_path / "a" / "chunks"), _files(tmp_path / "b" / "chunks")
    assert set(before) == set(after)
    changed = {name for name in before if before[name] != after[name]}
    stem = f"Kernels_chunk{chunkOf(added, 4)}"
    assert changed == {f"{stem}.cpp", f"{stem}.h"}


def test_fewer_than_two_used_chunks_compile_in_one_piece(tmp_path):
    name = NAMES[0]
    _, chunks = _write(tmp_path, _helpers([name]), 4)
    assert chunks == []
    assert not (tmp_path / "chunks").exists()


def _compiler(launcher=(), defaultArgs=()):
    return types.SimpleNamespace(launcher=list(launcher), default_args=list(defaultArgs))


@pytest.mark.parametrize(
    "compiler, targets, expected",
    [
        (_compiler(), ["gfx950"], True),
        (_compiler(), ["gfx942:xnack+"], True),
        (_compiler(), ["gfx942", "gfx950"], False),
        (_compiler(launcher=["ccache"]), ["gfx950"], False),
        (_compiler(defaultArgs=["--save-temps"]), ["gfx950"], False),
    ],
)
def test_supports_chunked_compile(compiler, targets, expected):
    assert supportsChunkedCompile(compiler, targets) == expected


def _driverJobs(args):
    """The jobs clang prints for a one-piece compile of args' source (-###), with temporary
    file names derived from the source name."""
    src = Path(args[args.index("-c") - 1]).stem
    out = args[args.index("-o") + 1]
    return [
        ["/rocm/clang-22", "-cc1", "-emit-obj", "-o", f"/tmp/{src}-gfx950-1.o", "-x", "hip", src],
        [
            "/rocm/lld",
            "-flavor",
            "gnu",
            "-shared",
            "-o",
            f"/tmp/{src}-gfx950-2.out",
            f"/tmp/{src}-gfx950-1.o",
        ],
        [
            "/rocm/clang-offload-bundler",
            "-type=o",
            f"-input=/tmp/{src}-gfx950-2.out",
            f"-output={out}",
        ],
    ]


@pytest.fixture
def fakeDriver(monkeypatch):
    """compileHelperChunks with the driver's -### replaced by _driverJobs; records the jobs run
    (compile jobs sorted: they run concurrently)."""
    ran = []
    monkeypatch.setattr(HelperKernelChunks, "driverJobs", _driverJobs)
    monkeypatch.setattr(HelperKernelChunks, "_invoke", lambda job, desc: ran.append(job))
    compiler = types.SimpleNamespace(
        compileArgs=lambda inc, targets, src, obj, pch_path=None: ["clang", src, "-c", "-o", obj]
    )
    return compiler, ran


def test_chunks_are_compiled_with_the_driver_jobs_and_linked_once(tmp_path, fakeDriver):
    compiler, ran = fakeDriver
    chunks = [tmp_path / "Kernels_chunk0.cpp", tmp_path / "Kernels_chunk3.cpp"]
    obj = str(tmp_path / "obj" / "Kernels.o")
    compileHelperChunks(compiler, "inc", "gfx950", chunks, obj)

    objDir = tmp_path / "obj"
    compiles, link, bundle = sorted(ran[:2]), ran[2], ran[3]
    assert len(ran) == 4
    assert [job[job.index("-o") + 1] for job in compiles] == [
        str(objDir / "Kernels_chunk0.o"),
        str(objDir / "Kernels_chunk3.o"),
    ]
    assert [job[-1] for job in compiles] == ["Kernels_chunk0", "Kernels_chunk3"]
    linked = str(objDir / "Kernels.linked.o")
    assert link == [
        "/rocm/lld",
        "-flavor",
        "gnu",
        "-shared",
        "-o",
        linked,
        str(objDir / "Kernels_chunk0.o"),
        str(objDir / "Kernels_chunk3.o"),
    ]
    assert bundle == [
        "/rocm/clang-offload-bundler",
        "-type=o",
        f"-input={linked}",
        f"-output={obj}",
    ]


@pytest.mark.parametrize(
    "jobs",
    [
        lambda jobs: jobs[:2],  # no bundle job
        lambda jobs: [jobs[0], ["/rocm/clang-22", "-cc1as"], *jobs[1:]],  # an extra job
        lambda jobs: [jobs[0], jobs[1][:-1], jobs[2]],  # the link does not read the object
        lambda jobs: [*jobs[:2], jobs[2][:-1]],  # the bundle does not write the object
    ],
)
def test_unexpected_driver_jobs_are_an_error(tmp_path, fakeDriver, monkeypatch, jobs):
    compiler, ran = fakeDriver
    monkeypatch.setattr(HelperKernelChunks, "driverJobs", lambda args: jobs(_driverJobs(args)))
    chunks = [tmp_path / "Kernels_chunk0.cpp", tmp_path / "Kernels_chunk1.cpp"]
    with pytest.raises(RuntimeError):
        compileHelperChunks(compiler, "inc", "gfx950", chunks, str(tmp_path / "Kernels.o"))
    assert ran == []


class _StubCompiler:
    """A compiler that supports chunks; records one-piece compiles."""

    launcher = []
    default_args = []

    def __init__(self):
        self.calls = []

    def __call__(self, includeDir, archs, kernelPath, objPath, pch_path=None):
        self.calls.append(Path(kernelPath).name)
        Path(objPath).write_text("one piece")


class _StubBundler:
    def targets(self, objPath):
        return ["host-x86_64-unknown-linux", "hipv4-amdgcn-amd-amdhsa--gfx950"]

    def __call__(self, target, objPath, coPathRaw):
        Path(coPathRaw).write_text(Path(objPath).read_text())


@pytest.fixture
def stubBuild(tmp_path, monkeypatch):
    """buildSourceCodeObjectFiles with stub tools, no helper cache, PCH or chunk cache. Returns
    build(chunkPaths, archs) -> (one-piece compiles, code object text)."""
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_PCH", "1")
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CHUNK_CACHE", "1")
    kernel = tmp_path / "Kernels.cpp"
    kernel.write_text("// src")

    def build(chunkPaths, archs=("gfx950",)):
        compiler = _StubCompiler()
        coPaths = Source.buildSourceCodeObjectFiles(
            compiler,
            _StubBundler(),
            tmp_path / "dest",
            tmp_path / "obj",
            tmp_path,
            kernel,
            list(archs),
            chunkPaths,
        )
        return compiler.calls, [Path(p).read_text() for p in coPaths]

    return build


def _chunkedCompile(compiler, includeDir, target, chunkPaths, objPath, pchPath=None, cache=None):
    Path(objPath).write_text(f"{len(chunkPaths)} chunks")


def test_chunks_replace_the_one_piece_compile(stubBuild, monkeypatch):
    monkeypatch.setattr(Source, "compileHelperChunks", _chunkedCompile)
    assert stubBuild(["Kernels_chunk0.cpp", "Kernels_chunk2.cpp"]) == ([], ["2 chunks"])


def test_failed_chunked_compile_falls_back_to_one_piece(stubBuild, monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError("chunk 2 does not compile")

    monkeypatch.setattr(Source, "compileHelperChunks", fail)
    assert stubBuild(["Kernels_chunk0.cpp", "Kernels_chunk2.cpp"]) == (
        ["Kernels.cpp"],
        ["one piece"],
    )
    assert (
        "# Helper kernel chunked compile failed, compiling in one piece: chunk 2 does not compile"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "chunkPaths, archs",
    [
        ([], ["gfx950"]),
        (["Kernels_chunk1.cpp"], ["gfx950"]),
        (["a.cpp", "b.cpp"], ["gfx942", "gfx950"]),
    ],
)
def test_no_chunked_compile_without_two_chunks_or_for_two_targets(
    stubBuild, monkeypatch, chunkPaths, archs
):
    monkeypatch.setattr(Source, "compileHelperChunks", _chunkedCompile)
    calls, _ = stubBuild(chunkPaths, archs)
    assert calls == ["Kernels.cpp"]


REAL_COMPILER = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / "amdclang++"
REAL_BUNDLER = REAL_COMPILER.resolve().parent / "clang-offload-bundler"


def _reduction(biasType):
    problemType = {
        "NumIndicesC": 3,
        "DataType": DataType("h"),
        "ComputeDataType": DataType("s"),
        "HighPrecisionAccumulate": True,
        "BiasDataType": DataType(biasType),
    }
    return KernelWriterReduction({"ProblemType": problemType})


def _kernelCode(coPath):
    """{kernel name: disassembly without addresses, up to its last s_endpgm} of a code object.
    What follows s_endpgm is padding to the next function, which depends on the layout."""
    tools = REAL_COMPILER.resolve().parent
    out = subprocess.run(
        [str(tools / "llvm-objdump"), "-d", "--no-show-raw-insn", "--no-leading-addr", str(coPath)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    code, name = {}, None
    for line in out.splitlines():
        match = re.match(r"^<(\w+)>:$", line.strip())
        if match:
            name = match.group(1)
            code[name] = []
        elif name and line.split("//")[0].strip():
            code[name].append(" ".join(line.split("//")[0].split()))
    return {
        name: lines[: len(lines) - lines[::-1].index("s_endpgm")] for name, lines in code.items()
    }


def _realBuilder(tmp_path, monkeypatch, writeHelperFiles):
    """Writes the helpers with writeHelperFiles(tmp_path) (-> chunk sources) next to the static
    headers, and returns (chunks, build): build(name, chunkPaths) compiles them for gfx950 with
    the real compiler and bundler (no helper cache, PCH or chunk cache) and returns
    _kernelCode."""
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_PCH", "1")
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CHUNK_CACHE", "1")
    copyStaticFiles(tmp_path)
    chunks = writeHelperFiles(tmp_path)
    compiler = Compiler(str(REAL_COMPILER), "sha1")
    bundler = Bundler(str(REAL_BUNDLER))

    def build(name, chunkPaths):
        (coPath,) = Source.buildSourceCodeObjectFiles(
            compiler,
            bundler,
            tmp_path / name,
            tmp_path / f"obj_{name}",
            tmp_path,
            tmp_path / "Kernels.cpp",
            ["gfx950"],
            chunkPaths,
        )
        return _kernelCode(coPath)

    return chunks, build


@pytest.fixture
def realBuild(tmp_path, monkeypatch):
    """Helpers with 21 reduction kernels in two chunks (of 3: the three reduction helpers' names
    hash to chunks 2, 1 and 1)."""
    helpers = [_SharedHelper("Enum"), _reduction("s"), _reduction("h"), _reduction("b")]
    return _realBuilder(
        tmp_path,
        monkeypatch,
        lambda out: writeHelpers(out, helpers, "Kernels.cpp", "Kernels.h", 3, out / "chunks"),
    )


needsRocm = pytest.mark.skipif(
    not (REAL_COMPILER.exists() and REAL_BUNDLER.exists()), reason="needs ROCm with gfx950"
)


@needsRocm
def test_real_compiler_chunked_kernels_are_identical(realBuild, capsys):
    chunks, build = realBuild
    assert [p.name for p in chunks] == ["Kernels_chunk1.cpp", "Kernels_chunk2.cpp"]
    onePiece = build("one", [])
    chunked = build("chunked", chunks)
    assert "chunked compile failed" not in capsys.readouterr().out
    assert len(onePiece) == 21
    assert chunked == onePiece


@needsRocm
def test_real_compiler_falls_back_when_a_chunk_does_not_compile(realBuild, capsys):
    chunks, build = realBuild
    onePiece = build("one", [])
    with open(chunks[-1], "a") as chunk:
        chunk.write("this does not compile;\n")
    assert build("chunked", chunks) == onePiece
    assert "# Helper kernel chunked compile failed, compiling in one piece" in (
        capsys.readouterr().out
    )


# I8_GSU: Int8 GEMM with GSU and hipblaslt_all activation; its helpers are the real writers
# BetaOnly, 3 x Conversion (PostGSU16 VW1/2/4), ActivationEnumHeader and ActivationFunction.
_I8_GSU_LOGIC = Path(__file__).parent / "characterization" / "_codegen" / "data" / "gfx950"
_I8_GSU_LOGIC = _I8_GSU_LOGIC / "I8_GSU.yaml"


@needsRocm
def test_real_writers_conversion_kernels_compile_in_chunks_without_the_activation_writers(
    tmp_path, monkeypatch, capsys
):
    """The conversion kernels call the activation device functions and use the activation enum,
    which the activation writers define in their headers (sharedHeader): a chunk with a
    conversion kernel but not the activation writers must still compile, in chunks."""
    from codegen_harness import (
        _isolated_globals,
        _solutions_from_logic_unguarded,
        get_assembler,
        get_isa_info_map,
    )
    from Tensile.KernelWriterActivationFunction import KernelWriterActivationFunction
    from Tensile.KernelWriterConversion import KernelWriterConversion
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelHelperObjects,
        generateKernelObjectsFromSolutions,
    )

    def writeHelperFiles(out):
        # The activation writers read rocisa state: generate and write in isolation.
        with _isolated_globals():
            solutions = _solutions_from_logic_unguarded(_I8_GSU_LOGIC)
            helpers = generateKernelHelperObjects(
                generateKernelObjectsFromSolutions(solutions),
                str(get_assembler().path),
                get_isa_info_map(),
            )
            (activation,) = [h for h in helpers if isinstance(h, KernelWriterActivationFunction)]
            conversions = [h for h in helpers if isinstance(h, KernelWriterConversion)]
            # Some conversion kernel is in another chunk than the activation writer's name
            # would put it, so without sharedHeader that chunk would not compile.
            activationChunk = chunkOf(activation.getKernelName(), 2)
            assert any(chunkOf(c.getKernelName(), 2) != activationChunk for c in conversions)
            return writeHelpers(out, helpers, "Kernels.cpp", "Kernels.h", 2, out / "chunks")

    chunks, build = _realBuilder(tmp_path, monkeypatch, writeHelperFiles)
    assert len(chunks) == 2
    onePiece = build("one", [])
    chunked = build("chunked", chunks)
    assert "chunked compile failed" not in capsys.readouterr().out
    assert sum("PostGSU16" in name for name in onePiece) == 6  # GroupedGemm on and off
    assert chunked == onePiece
