# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-chunk compile cache of the helper kernels (timing_runs/DECISIONS.md topic 6, C; P4.15).

compileHelperChunks takes each chunk's object from HelperKernelChunkCache when its key (chunk
source and header, static headers, compile job without build-directory names, full compiler
version) is cached, compiles the others and stores them after the link."""

import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from Tensile.TensileCreateLibrary.Run import writeHelpers
from Tensile.Toolchain import HelperKernelChunkCache as cacheModule
from Tensile.Toolchain import HelperKernelChunks
from Tensile.Toolchain.HelperKernelCache import _STATIC_HEADER_FILES
from Tensile.Toolchain.HelperKernelChunkCache import HelperKernelChunkCache
from Tensile.Toolchain.HelperKernelChunks import chunkOf, compileHelperChunks
from test_helper_kernel_chunks import (
    _helpers,
    _realBuilder,
    _reduction,
    _SharedHelper,
    needsRocm,
)

pytestmark = pytest.mark.unit

VERSION = "AMD clang version 22.0.0git (roc-7.2.1 26084 f58b06dce1f9)\n"


@pytest.fixture
def cacheDir(tmp_path, monkeypatch):
    """An empty enabled chunk cache in tmp_path, with a fixed compiler version."""
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_CHUNK_CACHE", raising=False)
    monkeypatch.setenv("TENSILE_HELPER_CHUNK_CACHE_DIR", str(tmp_path / "chunk_cache"))
    monkeypatch.setattr(cacheModule, "_compilerVersion", lambda path: VERSION)
    return tmp_path / "chunk_cache"


def _entries(cacheDir):
    return sorted(p.name for p in cacheDir.iterdir()) if cacheDir.is_dir() else []


# ------------------------------------------------------------------------------------------------
# A fake driver: the jobs of a one-piece compile with what really differs between two builds of
# the same chunk (temporary output, -cuid, source path, include and working directory).

FLAGS = ["-triple", "amdgcn-amd-amdhsa", "-O3", "-target-cpu", "gfx950"]


def _driverJobs(args):
    src = args[args.index("-c") - 1]
    inc = args[args.index("-I") + 1]
    out = args[args.index("-o") + 1]
    pch = ["-include-pch", args[args.index("-include-pch") + 1]] if "-include-pch" in args else []
    tmp = f"/tmp/{Path(src).stem}-gfx950-{uuid.uuid4().hex[:6]}"
    compile_ = [
        "/rocm/clang-22",
        "-cc1",
        *FLAGS,
        "-main-file-name",
        Path(src).name,
        f"-fdebug-compilation-dir={os.getcwd()}",
        f"-fcoverage-compilation-dir={os.getcwd()}",
        *pch,
        "-I",
        inc,
        f"-cuid={uuid.uuid4().hex[:16]}",
        "-o",
        f"{tmp}.o",
        "-x",
        "hip",
        src,
    ]
    link = ["/rocm/lld", "-flavor", "gnu", "-shared", "-o", f"{tmp}.out", f"{tmp}.o"]
    bundle = ["/rocm/clang-offload-bundler", "-type=o", f"-input={tmp}.out", f"-output={out}"]
    return [compile_, link, bundle]


def _fakeInvoke(ran):
    """Runs the fake jobs: a compile writes 'object of <source text>', the link concatenates its
    inputs, the bundle copies."""

    def invoke(job, desc):
        out = job[job.index("-o") + 1] if "-o" in job else None
        if job[1] == "-cc1":
            ran.append(("compile", Path(job[-1]).name))
            Path(out).write_text(f"object of {Path(job[-1]).read_text()}")
        elif "lld" in job[0]:
            ran.append(("link",))
            inputs = job[job.index("-o") + 2 :]
            Path(out).write_text("".join(Path(p).read_text() for p in inputs))
        else:
            ran.append(("bundle",))
            src = next(a for a in job if a.startswith("-input=")).split("=", 1)[1]
            dst = next(a for a in job if a.startswith("-output=")).split("=", 1)[1]
            Path(dst).write_text(Path(src).read_text())

    return invoke


class _FakeCompiler:
    path = "/rocm/bin/amdclang++"

    def compileArgs(self, inc, targets, src, obj, pch_path=None):
        pch = [] if pch_path is None else ["-include-pch", pch_path]
        return ["clang", "-I", inc, *pch, src, "-c", "-o", obj]


@pytest.fixture
def fakeBuild(tmp_path, monkeypatch, cacheDir):
    """build(name, helpers, chunkCount=4, pch=None) writes the helpers' chunks and the static
    headers into tmp_path/name, compiles them with the fake driver and the chunk cache from the
    environment, and returns (jobs run, bundled object text)."""
    ran = []
    monkeypatch.setattr(HelperKernelChunks, "driverJobs", _driverJobs)
    monkeypatch.setattr(HelperKernelChunks, "_invoke", _fakeInvoke(ran))

    def build(name, helpers, chunkCount=4, pch=None):
        root = tmp_path / name
        (root / "inc").mkdir(parents=True)
        (root / "obj").mkdir()
        for header in _STATIC_HEADER_FILES:
            (root / "inc" / header).write_text(f"// {header}\n")
        chunks = writeHelpers(
            root / "inc", helpers, "Kernels.cpp", "Kernels.h", chunkCount, root / "chunks"
        )
        obj = root / "obj" / "Kernels.o"
        ran.clear()
        compileHelperChunks(
            _FakeCompiler(),
            str(root / "inc"),
            "gfx950",
            chunks,
            str(obj),
            pch,
            HelperKernelChunkCache(),
        )
        return sorted(ran), obj.read_text()

    return build


NAMES = [f"Cijk_Kernel{i}" for i in range(12)]
ALL_COMPILED = [("bundle",), *[("compile", f"Kernels_chunk{i}.cpp") for i in range(4)], ("link",)]


def test_second_build_takes_every_chunk_from_the_cache(fakeBuild, cacheDir, monkeypatch, capsys):
    ran, cold = fakeBuild("a", _helpers(NAMES))
    assert ran == ALL_COMPILED
    assert len(_entries(cacheDir)) == 4
    assert "# Helper kernel chunk cache: 0 of 4 chunks hit" in capsys.readouterr().out

    # Another build directory and working directory: other paths, -cuid and temporaries.
    monkeypatch.chdir(cacheDir.parent)
    ran, warm = fakeBuild("b", _helpers(NAMES))
    assert ran == [("bundle",), ("link",)]
    assert warm == cold
    assert "# Helper kernel chunk cache: 4 of 4 chunks hit" in capsys.readouterr().out


def test_adding_a_helper_compiles_only_its_chunk(fakeBuild, cacheDir, capsys):
    fakeBuild("a", _helpers(NAMES))
    added = "Cijk_Kernel_new"
    ran, obj = fakeBuild("b", _helpers(NAMES[:5] + [added] + NAMES[5:]))
    assert ran == [("bundle",), ("compile", f"Kernels_chunk{chunkOf(added, 4)}.cpp"), ("link",)]
    assert f"definition of {added}" in obj
    assert len(_entries(cacheDir)) == 5
    assert "# Helper kernel chunk cache: 3 of 4 chunks hit" in capsys.readouterr().out


def test_another_pch_or_flag_is_a_miss(fakeBuild, cacheDir, monkeypatch):
    fakeBuild("a", _helpers(NAMES))
    ran, _ = fakeBuild("b", _helpers(NAMES), pch="/cache/helper_pch/1234/hip.pch")
    assert ran == ALL_COMPILED
    monkeypatch.setattr(sys.modules[__name__], "FLAGS", [*FLAGS, "-D", "X=1"])
    ran, _ = fakeBuild("c", _helpers(NAMES))
    assert ran == ALL_COMPILED
    assert len(_entries(cacheDir)) == 12


def test_another_compiler_build_is_a_miss(fakeBuild, monkeypatch):
    fakeBuild("a", _helpers(NAMES))
    monkeypatch.setattr(cacheModule, "_compilerVersion", lambda path: VERSION + "rebuilt\n")
    ran, _ = fakeBuild("b", _helpers(NAMES))
    assert ran == ALL_COMPILED


def test_disabled_cache_compiles_every_chunk_and_stores_nothing(
    fakeBuild, cacheDir, monkeypatch, capsys
):
    fakeBuild("a", _helpers(NAMES))
    entries = _entries(cacheDir)
    capsys.readouterr()
    for value in ("1", "yes", "ON", "True"):
        monkeypatch.setenv("TENSILE_DISABLE_HELPER_CHUNK_CACHE", value)
        ran, _ = fakeBuild(f"b{value}", _helpers(NAMES + ["Cijk_Kernel_new"]))
        assert ran == ALL_COMPILED
        assert _entries(cacheDir) == entries
        assert "chunk cache" not in capsys.readouterr().out


def test_failed_compile_stores_nothing(fakeBuild, cacheDir, monkeypatch):
    invoke = HelperKernelChunks._invoke

    def failChunk1(job, desc):
        if job[1] == "-cc1" and job[-1].endswith("Kernels_chunk1.cpp"):
            raise RuntimeError("chunk 1 does not compile")
        invoke(job, desc)

    monkeypatch.setattr(HelperKernelChunks, "_invoke", failChunk1)
    with pytest.raises(RuntimeError, match="chunk 1 does not compile"):
        fakeBuild("a", _helpers(NAMES))
    assert _entries(cacheDir) == []


def test_failed_link_discards_the_cached_chunks(fakeBuild, cacheDir, monkeypatch):
    fakeBuild("a", _helpers(NAMES))
    invoke = HelperKernelChunks._invoke

    def failLink(job, desc):
        if "lld" in job[0]:
            raise RuntimeError("truncated object")
        invoke(job, desc)

    monkeypatch.setattr(HelperKernelChunks, "_invoke", failLink)
    added = "Cijk_Kernel_new"
    with pytest.raises(RuntimeError, match="truncated object"):
        fakeBuild("b", _helpers(NAMES + [added]))
    # The three hits are gone and the compiled chunk was not stored (the link failed): only
    # build a's version of the changed chunk is left.
    assert len(_entries(cacheDir)) == 1

    monkeypatch.setattr(HelperKernelChunks, "_invoke", invoke)
    ran, _ = fakeBuild("c", _helpers(NAMES + [added]))
    assert ran == ALL_COMPILED


def test_unreadable_static_header_compiles_without_the_cache(
    fakeBuild, cacheDir, monkeypatch, capsys
):
    monkeypatch.setattr(cacheModule, "_STATIC_HEADER_FILES", ["Missing.h"])
    ran, _ = fakeBuild("a", _helpers(NAMES))
    assert ran == ALL_COMPILED
    assert _entries(cacheDir) == []
    out = capsys.readouterr().out
    assert "# Helper kernel chunk cache not used, computing its keys failed" in out


# ------------------------------------------------------------------------------------------------
# The key and the cache entries.


@pytest.fixture
def chunk(tmp_path):
    """(chunk path, compile job, include dir, static header hashes) of a chunk in tmp_path."""
    inc = tmp_path / "inc"
    inc.mkdir()
    path = tmp_path / "Kernels_chunk0.cpp"
    path.write_text('#include "Kernels_chunk0.h"\n// definition of K\n')
    path.with_suffix(".h").write_text("// declaration of K\n")
    job = _driverJobs(["clang", "-I", str(inc), str(path), "-c", "-o", "out.o"])[0]
    return str(path), job, str(inc), {"KernelHeader.h": "1"}


def _key(path, job, inc, static, version=VERSION):
    return cacheModule._computeChunkKey(path, job, inc, static, version)


def test_key_ignores_build_directory_names(chunk, tmp_path, monkeypatch):
    path, job, inc, static = chunk
    other = tmp_path / "other"
    other.mkdir()
    otherPath = other / Path(path).name
    otherPath.write_text(Path(path).read_text())
    otherPath.with_suffix(".h").write_text(Path(path).with_suffix(".h").read_text())
    otherInc = str(other / "inc")
    monkeypatch.chdir(other)
    otherJob = _driverJobs(["clang", "-I", otherInc, str(otherPath), "-c", "-o", "x.o"])[0]
    assert otherJob != job
    assert _key(str(otherPath), otherJob, otherInc, static) == _key(path, job, inc, static)


@pytest.mark.parametrize(
    "change",
    [
        lambda p, j, i, s, v: (p, j, i, s, v + "x"),  # compiler build
        lambda p, j, i, s, v: (p, j, i, {**s, "KernelHeader.h": "2"}, v),  # static header
        lambda p, j, i, s, v: (p, [*j[:3], "-O2", *j[4:]], i, s, v),  # a flag
        lambda p, j, i, s, v: (p, [*j, "-DX"], i, s, v),  # an extra flag
        lambda p, j, i, s, v: (p, j, "/other/inc", s, v),  # -I is not the include dir
    ],
)
def test_key_changes_with_every_input(chunk, change):
    path, job, inc, static = chunk
    assert _key(*change(path, job, inc, static, VERSION)) != _key(path, job, inc, static)


@pytest.mark.parametrize("suffix", [".cpp", ".h"])
def test_key_changes_with_the_chunk_source_and_header(chunk, suffix):
    path, job, inc, static = chunk
    before = _key(path, job, inc, static)
    with open(Path(path).with_suffix(suffix), "a") as f:
        f.write("// one more kernel\n")
    assert _key(path, job, inc, static) != before


def test_store_and_restore(cacheDir, tmp_path):
    cache = HelperKernelChunkCache()
    obj = tmp_path / "chunk0.o"
    obj.write_bytes(b"object")
    restored = tmp_path / "restored.o"
    assert not cache.restore("k", restored)
    assert not restored.exists()

    cache.store("k", obj)
    assert cache.restore("k", restored)
    assert restored.read_bytes() == b"object"
    obj.write_bytes(b"other")
    cache.store("k", obj)  # an existing entry is kept
    assert cache.restore("k", restored)
    assert restored.read_bytes() == b"object"
    assert _entries(cacheDir) == ["k"]

    cache.discard("k")
    assert _entries(cacheDir) == []
    assert not cache.restore("k", restored)
    cache.discard("k")  # already gone


def test_empty_entry_is_a_miss(cacheDir, tmp_path):
    (cacheDir / "k").mkdir(parents=True)
    (cacheDir / "k" / "chunk.o").write_bytes(b"")
    restored = tmp_path / "restored.o"
    assert not HelperKernelChunkCache().restore("k", restored)
    assert not restored.exists()


def test_concurrent_stores_of_one_key_leave_one_entry(cacheDir, tmp_path, monkeypatch):
    """Threads of one process storing the same key at once (the helper-cache temp names are per
    process only): each uses its own temporary directory, one rename wins, nothing is left."""
    cache = HelperKernelChunkCache()
    threads = 8
    barrier = threading.Barrier(threads)
    rename = Path.rename
    renamed = []

    def renameTogether(self, target):
        renamed.append(self)
        barrier.wait(timeout=10)
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", renameTogether)
    objs = []
    for i in range(threads):
        objs.append(tmp_path / f"obj{i}.o")
        objs[-1].write_bytes(b"object %d" % i)
    workers = [threading.Thread(target=cache.store, args=("k", obj)) for obj in objs]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert len(set(renamed)) == threads
    assert _entries(cacheDir) == ["k"]
    assert (cacheDir / "k" / "chunk.o").read_bytes() in {o.read_bytes() for o in objs}


def test_unwritable_cache_is_not_an_error(cacheDir, tmp_path):
    cacheDir.parent.mkdir(exist_ok=True)
    cacheDir.write_text("a file, not a directory")
    obj = tmp_path / "chunk0.o"
    obj.write_bytes(b"object")
    cache = HelperKernelChunkCache()
    cache.store("k", obj)
    assert not cache.restore("k", tmp_path / "restored.o")


def test_old_entries_and_leftover_temporaries_are_evicted(cacheDir):
    old = time.time() - 31 * 24 * 3600
    for name, mtime in [
        ("old", old),
        ("fresh", None),
        (".tmp_old_1_x", time.time() - 2 * 24 * 3600),
        (".tmp_fresh_1_x", None),
    ]:
        (cacheDir / name).mkdir(parents=True)
        if mtime:
            os.utime(cacheDir / name, (mtime, mtime))
    HelperKernelChunkCache()
    assert _entries(cacheDir) == [".tmp_fresh_1_x", "fresh"]


def test_restore_refreshes_the_entry(cacheDir, tmp_path):
    obj = tmp_path / "chunk0.o"
    obj.write_bytes(b"object")
    HelperKernelChunkCache().store("k", obj)
    old = time.time() - 29 * 24 * 3600
    os.utime(cacheDir / "k", (old, old))
    assert HelperKernelChunkCache().restore("k", tmp_path / "restored.o")
    assert time.time() - (cacheDir / "k").stat().st_mtime < 3600


# ------------------------------------------------------------------------------------------------
# The real compiler.


@needsRocm
def test_real_compiler_cached_chunks_give_identical_kernels(tmp_path, monkeypatch, capsys):
    """Reduction helpers s and h hash to chunks 2 and 1 (of 3), b to chunk 1: a second build in
    another directory takes both chunks from the cache, with identical kernels; adding b compiles
    chunk 1 only, with the kernels of a one-piece compile."""
    helpers = [_SharedHelper("Enum"), _reduction("s"), _reduction("h")]
    builds = {}

    def writer(name, helpers):
        def write(out):
            return writeHelpers(out, helpers, "Kernels.cpp", "Kernels.h", 3, out / "chunks")

        chunks, build = _realBuilder(tmp_path / name, monkeypatch, write)
        builds[name] = (chunks, build)

    writer("a", helpers)
    writer("b", helpers)
    writer("c", [*helpers, _reduction("b")])
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_CHUNK_CACHE")
    monkeypatch.setenv("TENSILE_HELPER_CHUNK_CACHE_DIR", str(tmp_path / "chunk_cache"))

    def run(name):
        chunks, build = builds[name]
        kernels = build("chunked", chunks)
        return kernels, capsys.readouterr().out

    cold, out = run("a")
    assert "# Helper kernel chunk cache: 0 of 2 chunks hit" in out
    warm, out = run("b")
    assert "# Helper kernel chunk cache: 2 of 2 chunks hit" in out
    assert "chunked compile failed" not in out
    assert len(cold) == 14
    assert warm == cold
    assert warm == builds["b"][1]("one", [])

    added, out = run("c")
    assert "# Helper kernel chunk cache: 1 of 2 chunks hit" in out
    assert "chunked compile failed" not in out
    assert len(added) == 21
    assert added == builds["c"][1]("one", [])
    assert len(_entries(tmp_path / "chunk_cache")) == 3
