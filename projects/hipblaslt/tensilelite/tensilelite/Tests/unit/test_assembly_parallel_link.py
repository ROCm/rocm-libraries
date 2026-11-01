# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for parallel linking in tensilelite.Toolchain.Assembly and the
Linker's thread limit."""
import threading
from pathlib import Path

import pytest

import tensilelite.Toolchain.Assembly as A
import tensilelite.Toolchain.Component as C
from tensilelite.Toolchain.Assembly import buildAssemblyCodeObjectFiles

pytestmark = pytest.mark.unit


class _RecordingLinker:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = []

    def __call__(self, objFiles, coFileRaw, threads=None):
        with self._lock:
            self.calls.append((Path(coFileRaw).name, len(objFiles), threads))
        Path(coFileRaw).write_text("raw")


class _StubBundler:
    def __init__(self):
        self.calls = []

    def compress(self, raw, co, gfx):
        self.calls.append((Path(raw).name, Path(co).name, gfx))
        Path(co).write_text("co")


def _kernels(objectsPerCodeObject):
    return [
        {"ISA": [9, 4, 2], "BaseName": f"{co}_k{i}", "codeObjectFile": co}
        for co, count in objectsPerCodeObject.items()
        for i in range(count)
    ]


def _build(tmp_path, linker, kernels):
    return buildAssemblyCodeObjectFiles(
        linker, _StubBundler(), kernels, tmp_path / "dest", tmp_path, compress=True
    )


def test_concurrent_links_share_the_cpus(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CPUThreadCount", lambda: 8)
    linker = _RecordingLinker()
    _build(tmp_path, linker, _kernels({"A": 1, "B": 1}))
    assert sorted(threads for _, _, threads in linker.calls) == [4, 4]


def test_links_are_single_threaded_when_jobs_outnumber_cpus(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CPUThreadCount", lambda: 2)
    linker = _RecordingLinker()
    _build(tmp_path, linker, _kernels({"A": 1, "B": 1, "C": 1}))
    assert [threads for _, _, threads in linker.calls] == [1, 1, 1]


def test_largest_links_start_first_and_output_order_is_kept(tmp_path, monkeypatch):
    # One worker runs the links in the order they were submitted.
    monkeypatch.setattr(A, "CPUThreadCount", lambda: 1)
    linker = _RecordingLinker()
    coFiles = _build(tmp_path, linker, _kernels({"Small": 1, "Large": 3, "Medium": 2}))
    assert [name for name, _, _ in linker.calls] == [
        "Large.co.raw",
        "Medium.co.raw",
        "Small.co.raw",
    ]
    assert [p.name for p in coFiles] == ["Small.co", "Large.co", "Medium.co"]


def test_link_and_bundle_compresses_the_linked_code_object(tmp_path):
    linker, bundler = _RecordingLinker(), _StubBundler()
    raw, co = tmp_path / "A.co.raw", tmp_path / "A.co"
    A._linkAndBundle(linker, bundler, True, 3, (["a.o", "b.o"], raw, co, "gfx942"))
    assert linker.calls == [("A.co.raw", 2, 3)]
    assert bundler.calls == [("A.co.raw", "A.co", "gfx942")]
    assert co.read_text() == "co"


def test_link_and_bundle_moves_the_code_object_without_compression(tmp_path):
    linker, bundler = _RecordingLinker(), _StubBundler()
    raw, co = tmp_path / "A.co.raw", tmp_path / "A.co"
    A._linkAndBundle(linker, bundler, False, 1, (["a.o"], raw, co, "gfx942"))
    assert bundler.calls == []
    assert not raw.exists()
    assert co.read_text() == "raw"


def _linker(monkeypatch):
    captured = []
    monkeypatch.setattr(C, "_invoke", lambda args, desc="": captured.append(args))
    linker = C.Linker.__new__(C.Linker)
    linker.default_args = ["clang", "--target=amdgcn-amd-amdhsa"]
    return linker, captured


def test_linker_passes_the_thread_limit_to_lld(monkeypatch):
    linker, captured = _linker(monkeypatch)
    linker(["a.o", "b.o"], "out.co", threads=4)
    assert captured == [
        ["clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--threads=4", "a.o", "b.o", "-o", "out.co"]
    ]


def test_linker_without_a_thread_limit_leaves_lld_default(monkeypatch):
    linker, captured = _linker(monkeypatch)
    linker(["a.o"], "out.co")
    assert captured == [["clang", "--target=amdgcn-amd-amdhsa", "a.o", "-o", "out.co"]]


def test_linker_keeps_the_thread_limit_with_a_response_file(tmp_path, monkeypatch):
    linker, captured = _linker(monkeypatch)
    monkeypatch.setattr(C.Linker, "_use_response_file", lambda self, args: True)
    dest = str(tmp_path / "out.co")
    linker(["a.o"], dest, threads=2)
    assert captured == [
        [
            "clang",
            "--target=amdgcn-amd-amdhsa",
            "-Xlinker",
            "--threads=2",
            "-o",
            dest,
            f"@{dest}.linker_args",
        ]
    ]
