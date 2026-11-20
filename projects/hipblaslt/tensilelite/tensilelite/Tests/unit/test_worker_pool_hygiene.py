# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""shutdownWorkerPool() ends the loky workers and releaseFreedMemory() is a no-op without glibc."""

import ctypes
import os
import time
from unittest.mock import MagicMock, patch

import pytest

import importlib

# The module itself: tensilelite.Common re-exports joblib's Parallel class under
# the same name, so attribute access would return the class.
Parallel = importlib.import_module("tensilelite.Common.Parallel")
from tensilelite.Common.GlobalParameters import globalParameters


def _pid(_):
    return os.getpid()


def _gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not os.path.exists(f"/proc/{pid}"):
            return True
        time.sleep(0.05)
    return False


def test_loky_still_exposes_the_executor_the_shutdown_reaches():
    # shutdownWorkerPool() reads this private name; a joblib upgrade that
    # moves it would otherwise turn the recycling into a silent no-op.
    from joblib.externals.loky import reusable_executor

    assert hasattr(reusable_executor, "_executor")


def test_shutdown_ends_the_workers_and_the_next_call_respawns(monkeypatch):
    monkeypatch.setitem(globalParameters, "CpuThreads", 2)
    monkeypatch.setitem(globalParameters, "ShowProgressBar", False)
    first = set(
        Parallel.ParallelMap2(
            _pid, [(i,) for i in range(8)], "pids", procs=2, spawn_worker=True
        )
    )
    assert first and os.getpid() not in first
    Parallel.shutdownWorkerPool()
    assert all(_gone(pid) for pid in first), f"workers survived shutdown: {first}"
    second = set(
        Parallel.ParallelMap2(
            _pid, [(i,) for i in range(8)], "pids", procs=2, spawn_worker=True
        )
    )
    assert second and not (second & first)
    Parallel.shutdownWorkerPool()
    Parallel.shutdownWorkerPool()  # a second shutdown without a run in between is harmless


def test_shutdown_warns_when_loky_has_no_executor_attribute(monkeypatch, capsys):
    from joblib.externals.loky import reusable_executor

    monkeypatch.delattr(reusable_executor, "_executor")
    Parallel.shutdownWorkerPool()
    assert "worker pool is not recycled" in capsys.readouterr().out


def test_release_freed_memory_is_a_noop_without_glibc(monkeypatch):
    monkeypatch.setattr(Parallel, "_libc", None)

    def no_libc(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", no_libc)
    Parallel.releaseFreedMemory()
    assert Parallel._libc is False
    Parallel.releaseFreedMemory()  # still a no-op, without retrying the load


class _FakeLibc:
    def __init__(self):
        self.calls = []

    def malloc_trim(self, pad):
        self.calls.append(pad)
        return 1


def test_release_freed_memory_calls_malloc_trim_every_time(monkeypatch):
    fake = _FakeLibc()
    monkeypatch.setattr(Parallel, "_libc", fake)
    Parallel.releaseFreedMemory()
    Parallel.releaseFreedMemory()
    assert fake.calls == [0, 0]


@pytest.mark.skipif(
    not os.path.exists("/lib/x86_64-linux-gnu/libc.so.6")
    and not os.path.exists("/lib64/libc.so.6"),
    reason="glibc only",
)
def test_release_freed_memory_loads_glibc(monkeypatch):
    monkeypatch.setattr(Parallel, "_libc", None)
    Parallel.releaseFreedMemory()
    assert Parallel._libc is not None and Parallel._libc is not False


# ---------------------------------------------------------- driver boundaries
RUN = importlib.import_module("tensilelite.create_library.run")


def _kernel(name):
    values = {
        "KernelLanguage": "Assembly",
        "ISA": (9, 4, 2),
        "WavefrontSize": 64,
        "BaseName": name,
    }
    kernel = MagicMock()
    kernel.__getitem__.side_effect = lambda key: values[key]
    kernel.get.side_effect = lambda key, default=None: values.get(key, default)
    return kernel


def test_codegen_trims_after_every_kernel_and_recycles_the_pool_once(tmp_path):
    events = []
    groups = [
        RUN.KernelCompileGroup(f"k{i}", _kernel(f"k{i}"), (_kernel(f"k{i}"),), f"k{i}")
        for i in range(3)
    ]
    asmPath = tmp_path / "k.s"
    asmPath.write_text("")
    with patch.object(
        RUN, "ParallelMap2", side_effect=lambda fn, objs, *a, **k: [fn(o) for o in objs]
    ), patch.object(
        RUN, "processKernelSource", side_effect=lambda *a, **k: MagicMock()
    ), patch.object(
        RUN,
        "writeAssembly",
        side_effect=lambda path, result: (asmPath, (9, 4, 2), 64, result),
    ), patch.object(
        RUN, "_alignAmdgcnTargetToStepping"
    ), patch.object(
        RUN, "compilerTargetOf", return_value="gfx942"
    ), patch.object(
        RUN, "buildAssemblyCodeObjectFiles", return_value=[]
    ), patch.object(
        RUN, "buildSourceCodeObjectFiles"
    ), patch.object(
        RUN, "writeHelpers"
    ), patch.object(
        RUN, "copyStaticFiles"
    ), patch.object(
        RUN, "releaseFreedMemory", side_effect=lambda: events.append("trim")
    ), patch.object(
        RUN, "shutdownWorkerPool", side_effect=lambda: events.append("shutdown")
    ):
        n, unique, results = RUN.writeSolutionsAndKernelsTCL(
            tmp_path / "out",
            MagicMock(),
            MagicMock(),
            [],
            groups,
            [],
            MagicMock(),
            ["gfx942"],
            removeTemporaries=False,
        )
    assert n == 3 and len(results) == 3
    # one trim per kernel, in the worker function, then one pool shutdown after codegen
    assert events == ["trim", "trim", "trim", "shutdown"]


def test_run_recycles_the_pool_after_logic_loading(tmp_path):
    import shutil

    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap

    logic = tmp_path / "logic"
    logic.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
    isaInfoMap = makeIsaInfoMap([gfxToIsa("gfx942")], clang)
    arguments = {
        "PrintLevel": 1,
        "OutputPath": str(out),
        "CxxCompiler": "/fake/hipcc",
        "CCompiler": "/fake/hipcc",
        "OffloadBundler": "/fake/bundler",
        "Assembler": "/fake/assembler",
        "CodeObjectVersion": "4",
        "BuildIdKind": "sha1",
        "AsmDebug": False,
        "AsanBuild": False,
        "Architecture": "gfx942",
        "LogicPath": str(logic),
        "LogicFormat": "yaml",
        "LibraryFormat": "msgpack",
        "CpuThreads": 1,
        "LazyLibraryLoading": True,
        "GenSolTable": False,
        "Experimental": False,
        "LogicFilter": "*",
        "DisableAsmComments": False,
        "UseCompression": False,
        "KeepBuildTmp": False,
    }
    events = []

    def load(*a, **k):
        events.append("load")
        return [], {"gfx942": MagicMock(solutions={}, lazyLibraries={})}, {}

    def write(*a, **k):
        events.append("write")
        return 0, [], []

    with patch.object(RUN, "parseArguments", return_value=arguments), patch.object(
        RUN, "setVerbosity"
    ), patch.object(
        RUN,
        "validateToolchain",
        return_value=("/fake/hipcc", None, "/fake/bundler", None),
    ), patch.object(
        RUN, "makeIsaInfoMap", return_value=isaInfoMap
    ), patch.object(
        RUN, "assignGlobalParameters"
    ), patch.object(
        RUN, "makeAssemblyToolchain", return_value=MagicMock()
    ), patch.object(
        RUN, "makeSourceToolchain", return_value=MagicMock()
    ), patch.object(
        RUN, "generateLogicDataAndSolutions", side_effect=load
    ), patch.object(
        RUN, "generateKernelObjectsFromSolutions", return_value=[]
    ), patch.object(
        RUN, "generateKernelHelperObjects", return_value=[]
    ), patch.object(
        RUN, "writeSolutionsAndKernelsTCL", side_effect=write
    ), patch.object(
        RUN, "passPostKernelInfoToLibrary"
    ), patch.object(
        RUN.LibraryIO, "write"
    ), patch.object(
        RUN.shutil, "rmtree"
    ), patch.object(
        RUN, "shutdownWorkerPool", side_effect=lambda: events.append("shutdown")
    ):
        RUN.main()
    # the pool that parsed the logic files is recycled before codegen starts
    assert events == ["load", "shutdown", "write"]
