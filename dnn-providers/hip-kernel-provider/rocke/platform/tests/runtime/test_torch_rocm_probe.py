# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only subprocess coverage, collected by installed runtime/tools CTest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from rocke.runtime import runtime_coexistence
from rocke.runtime import torch_rocm_probe as probe
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_UNSEEN = object()
_INCOMPLETE = probe.TorchRocmStatus.INCONCLUSIVE
_MATCH = probe.TorchRocmStatus.SAME_BINARIES_OBSERVED
_MISMATCH = probe.TorchRocmStatus.DIFFERENT_BINARIES_OBSERVED
_Worker = probe.TorchWorkerStatus

# The two binaries have identical loader names and different contents.
# No mapping on either side is neutral when the other library kind is present.
_PAIR_OUTCOMES = {
    (None, None): _UNSEEN,
    (None, "a"): _INCOMPLETE,
    (None, "b"): _INCOMPLETE,
    ("a", None): _INCOMPLETE,
    ("b", None): _INCOMPLETE,
    ("a", "a"): _MATCH,
    ("b", "b"): _MATCH,
    ("a", "b"): _MISMATCH,
    ("b", "a"): _MISMATCH,
}


def _pair_id(pair):
    return "-".join(variant or "empty" for variant in pair)


def _write_library(directory, kind, variant):
    if variant is None:
        return None
    filenames = {
        "hip": ("libamdhip64.so.7", "amdhip64_7.dll"),
        "comgr": ("libamd_comgr.so.3", "amd_comgr_3.dll"),
    }
    directory.mkdir()
    linux_name, windows_name = filenames[kind]
    path = directory / (windows_name if sys.platform == "win32" else linux_name)
    if sys.platform == "linux":
        path.write_bytes(f"fixture image {variant}".encode())
    else:
        import _ctypes
        import _socket

        shutil.copyfile(_ctypes.__file__ if variant == "a" else _socket.__file__, path)
    return path


@pytest.mark.skipif(sys.platform not in {"linux", "win32"}, reason="native inventory")
@pytest.mark.parametrize("kind", ["hip", "comgr"])
@pytest.mark.parametrize("pair", _PAIR_OUTCOMES, ids=_pair_id)
def test_subprocess_compares_mapped_bytes_not_filenames_or_versions(
    tmp_path, kind, pair
):
    # Use fixture mappings on Linux and renamed CPython DLLs on Windows. No
    # ROCm or torch installation, compiler, or GPU is needed.
    parent_lib = _write_library(tmp_path / "parent libs", kind, pair[0])
    child_lib = _write_library(tmp_path / "child libs", kind, pair[1])
    (tmp_path / "fixture_mapping.py").write_text(
        "import ctypes, mmap, os, sys\n"
        "def map_fixture(name):\n"
        "    path = os.environ[name]\n"
        "    if not path:\n"
        "        return None\n"
        "    if sys.platform == 'win32':\n"
        "        return ctypes.CDLL(path)\n"
        "    with open(path, 'rb') as stream:\n"
        "        return mmap.mmap(stream.fileno(), 0, prot=mmap.PROT_READ | mmap.PROT_EXEC)\n",
        encoding="utf-8",
    )
    (tmp_path / "torch.py").write_text(
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from fixture_mapping import map_fixture\n"
        "assert 'rocke' not in sys.modules\n"
        "__version__ = 'fixture'\n"
        "version = SimpleNamespace(hip='fixture HIP version')\n"
        "print('noisy import output')\n"
        "lib = map_fixture('TEST_PROBE_TORCH_LIB')\n",
        encoding="utf-8",
    )
    env = rocke_subprocess_env(python_paths=[tmp_path])
    env.update(
        TEST_PROBE_PARENT_LIB=str(parent_lib) if parent_lib is not None else "",
        TEST_PROBE_TORCH_LIB=str(child_lib) if child_lib is not None else "",
    )
    report = subprocess.run(
        rocke_python_command(
            "-c",
            "import json, sys\n"
            "from dataclasses import asdict\n"
            "from fixture_mapping import map_fixture\n"
            "from rocke.runtime.torch_rocm_probe import probe_torch_rocm\n"
            "lib = map_fixture('TEST_PROBE_PARENT_LIB')\n"
            "assert 'torch' not in sys.modules\n"
            "result = probe_torch_rocm()\n"
            "assert 'torch' not in sys.modules\n"
            "print(json.dumps(asdict(result)))\n",
        ),
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    data = json.loads(report.stdout)
    assert data["worker_status"] == _Worker.IMPORTED
    assert data["torch_version"] == "fixture"
    assert data["torch_hip_version"] == "fixture HIP version"
    assert Path(data["torch_file"]).samefile(tmp_path / "torch.py")
    outcome = _PAIR_OUTCOMES[pair]
    assert data["status"] == (_INCOMPLETE if outcome == _UNSEEN else outcome)
    for key, path in [("parent_libraries", parent_lib), ("torch_libraries", child_lib)]:
        if path is None:
            assert data[key] == []
        else:
            (row,) = data[key]
            assert row["kind"] == kind
            assert Path(row["path"]).samefile(path)
            assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    if outcome == _MISMATCH:
        assert f"{kind}: different binaries" in data["reason"]
        assert data["parent_libraries"][0]["path"] in data["reason"]
        assert data["torch_libraries"][0]["path"] in data["reason"]
    elif outcome == _INCOMPLETE:
        assert f"{kind}: missing" in data["reason"]
    elif outcome == _UNSEEN:
        assert "no ROCm binaries observed" in data["reason"]


@pytest.mark.parametrize("hip", _PAIR_OUTCOMES, ids=_pair_id)
@pytest.mark.parametrize("comgr", _PAIR_OUTCOMES, ids=_pair_id)
def test_combined_hip_and_comgr_evidence(monkeypatch, hip, comgr):
    def inventory(side, side_name):
        return tuple(
            probe.RocmLibrary(kind, f"{side_name}/{kind}", pair[side])
            for kind, pair in [("hip", hip), ("comgr", comgr)]
            if pair[side] is not None
        )

    parent, child = inventory(0, "parent"), inventory(1, "child")

    def report_import(command, **_kwargs):
        # Stub the worker boundary to cover the full matrix without creating 81
        # extra interpreters. The single-kind tests exercise the real worker.
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "status": _Worker.IMPORTED,
                    "version": "fixture",
                    "hip_version": "fixture HIP version",
                    "torch_file": "fixture/torch.py",
                    "libraries": [asdict(item) for item in child],
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(probe, "_snapshot", lambda: (parent, None))
    monkeypatch.setattr(probe.subprocess, "run", report_import)
    report = probe.probe_torch_rocm()
    assert report.worker_status is _Worker.IMPORTED
    assert report.parent_libraries == parent
    assert report.torch_libraries == child
    outcomes = {_PAIR_OUTCOMES[hip], _PAIR_OUTCOMES[comgr]}
    if _MISMATCH in outcomes:
        assert report.status is _MISMATCH
    elif _INCOMPLETE in outcomes or outcomes == {_UNSEEN}:
        assert report.status is _INCOMPLETE
    else:
        assert report.status is _MATCH
    for kind, pair in [("hip", hip), ("comgr", comgr)]:
        outcome = _PAIR_OUTCOMES[pair]
        assert (f"{kind}: different binaries" in report.reason) == (
            outcome == _MISMATCH
        )
        assert (f"{kind}: missing" in report.reason) == (outcome == _INCOMPLETE)


@pytest.mark.parametrize(
    ("source", "status", "reason"),
    [
        (
            "raise ModuleNotFoundError(name='torch')",
            _Worker.TORCH_NOT_INSTALLED,
            "ModuleNotFoundError",
        ),
        (
            "raise ImportError('dependency missing')",
            _Worker.TORCH_IMPORT_ERROR,
            "dependency missing",
        ),
        (
            "import rocke_probe_missing_torch_dependency",
            _Worker.TORCH_IMPORT_ERROR,
            "rocke_probe_missing_torch_dependency",
        ),
        ("import os; os._exit(19)", _Worker.WORKER_ERROR, "status 19"),
    ],
)
def test_import_failures_are_not_binary_mismatches(
    tmp_path, monkeypatch, source, status, reason
):
    (tmp_path / "torch.py").write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = probe.probe_torch_rocm()
    assert result.status is _INCOMPLETE
    assert result.worker_status is status
    assert result.torch_file is None
    assert reason in result.reason


def test_reports_native_crash(monkeypatch):
    def fail(*args, **kwargs):
        kwargs["stderr"].write(b"native loader failure\n")
        return subprocess.CompletedProcess(args[0], -11)

    monkeypatch.setattr(probe.subprocess, "run", fail)
    result = probe.probe_torch_rocm()
    assert result.status is _INCOMPLETE
    assert result.worker_status is _Worker.CRASHED
    assert "-11" in result.reason
    assert result.stderr == "native loader failure"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_observing_or_spawning(monkeypatch, timeout):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid timeout must be rejected before inventory or spawning")

    monkeypatch.setattr(probe, "_snapshot", unexpected)
    monkeypatch.setattr(probe.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        probe.probe_torch_rocm(timeout=timeout)


@pytest.mark.parametrize("timeout", ["30", "-1"])
def test_module_cli(tmp_path, timeout):
    (tmp_path / "torch.py").write_text(
        "__version__ = 'cli fixture'\n", encoding="utf-8"
    )
    result = subprocess.run(
        rocke_python_command(
            "-m", "rocke.runtime.torch_rocm_probe", f"--timeout={timeout}"
        ),
        cwd=tmp_path,
        env=rocke_subprocess_env(python_paths=[tmp_path]),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if timeout == "-1":
        assert result.returncode == 2
        assert not result.stdout
        assert "error: timeout must be positive and finite" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "inconclusive"
    assert report["worker_status"] == "imported"
    assert report["torch_version"] == "cli fixture"
    assert Path(report["torch_file"]).samefile(tmp_path / "torch.py")


def test_reports_torch_from_child_startup_path(tmp_path):
    parent_only = tmp_path / "parent_only"
    child_visible = tmp_path / "child_visible"
    parent_only.mkdir()
    child_visible.mkdir()
    (parent_only / "torch.py").write_text("__version__ = 'parent'\n", encoding="utf-8")
    child_torch = child_visible / "torch.py"
    child_torch.write_text("__version__ = 'child'\n", encoding="utf-8")
    result = subprocess.run(
        rocke_python_command(
            "-c",
            "import importlib.util, json, sys\n"
            "from dataclasses import asdict\n"
            "from pathlib import Path\n"
            "from rocke.runtime.torch_rocm_probe import probe_torch_rocm\n"
            "parent_only = Path(sys.argv[1])\n"
            "sys.path.insert(0, str(parent_only))\n"
            "parent_spec = importlib.util.find_spec('torch')\n"
            "assert Path(parent_spec.origin).samefile(parent_only / 'torch.py')\n"
            "assert 'torch' not in sys.modules\n"
            "report = probe_torch_rocm()\n"
            "assert 'torch' not in sys.modules\n"
            "print(json.dumps(asdict(report)))\n",
            parent_only,
        ),
        cwd=tmp_path,
        env=rocke_subprocess_env(python_paths=[child_visible]),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["worker_status"] == _Worker.IMPORTED
    assert report["torch_version"] == "child"
    assert Path(report["torch_file"]).samefile(child_torch)


def test_bounds_subprocess_runtime(tmp_path, monkeypatch):
    (tmp_path / "torch.py").write_text("import time; time.sleep(60)", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = probe.probe_torch_rocm(timeout=0.5)
    assert result.status is _INCOMPLETE
    assert result.worker_status is _Worker.TIMEOUT


def test_unverifiable_and_ambiguous_binaries_do_not_match():
    known = probe.RocmLibrary("hip", "libamdhip64.so.7", "digest")
    other = probe.RocmLibrary("hip", "other/libamdhip64.so.7", "digest")
    unknown = probe.RocmLibrary("hip", "unreadable/libamdhip64.so.7")
    assert probe._compare((known,), (other,))[0] is _MATCH
    for right in [(), (other, known), (unknown,)]:
        assert probe._compare((known,), right)[0] is _INCOMPLETE
    assert probe._compare((), ())[0] is _INCOMPLETE


@pytest.mark.parametrize(
    ("platform", "kind", "valid", "near_miss"),
    [
        ("linux", "hip", "libamdhip64.so.7", "libamdhip64_debug.so.7"),
        ("linux", "comgr", "libamd_comgr.so.3", "amd_comgr.so.3"),
        ("linux", "llvm", "libLLVM.so.23.0git", "libLLVMCore.so.23"),
        ("linux", "llvm", "libLLVM-23.so", "libLLVM-C.so"),
        ("linux", "clang", "libclang-cpp.so.23", "libclang.so.23"),
        ("win32", "hip", "AMDhip64_7.DLL", "amdhip64_stub.dll"),
        ("win32", "comgr", "amd_comgr0702.dll", "amd_comgr_helper.dll"),
        ("win32", "llvm", "LLVM-23.dll", "LLVMCore.dll"),
        ("win32", "llvm", "LLVM.dll", "LLVM-C.dll"),
        ("win32", "clang", "clang-cpp.dll", "libclang.dll"),
        ("win32", "clang", "libclang-cpp-23.dll", "clang-23.dll"),
    ],
)
def test_inventory_library_names(platform, kind, valid, near_miss):
    classify = runtime_coexistence._mapped_library_kind
    assert classify(valid, platform=platform) == kind
    assert classify(near_miss, platform=platform) is None
    other_platform = "win32" if platform == "linux" else "linux"
    assert classify(valid, platform=other_platform) is None


@pytest.mark.parametrize(
    ("platform", "c_api", "shared_llvm"),
    [
        ("linux", "libLLVM-C.so", "libLLVM.so.23.0git"),
        ("win32", "LLVM-C.dll", "LLVM-23.dll"),
    ],
)
def test_llvm_c_api_does_not_produce_binary_mismatch(platform, c_api, shared_llvm):
    def inventory(name, digest):
        kind = runtime_coexistence._mapped_library_kind(name, platform=platform)
        return (probe.RocmLibrary(kind, name, digest),) if kind else ()

    status, reason = probe._compare(
        inventory(c_api, "c-api-digest"), inventory(shared_llvm, "shared-llvm-digest")
    )
    assert status is _INCOMPLETE
    assert "llvm: missing" in reason


@pytest.mark.skipif(sys.platform != "linux", reason="Linux maps format")
def test_linux_inventory_deduplicates_segments_and_checks_file_identity(tmp_path):
    path = tmp_path / "with space" / "libLLVM.so.23.0git"
    path.parent.mkdir()
    path.write_bytes(b"fixture")
    stat = path.stat()
    dev = f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}"
    lines = [
        f"0000-0500 r-xp 0 {dev} 0\n",  # Anonymous executable mapping.
        f"0500-1000 r--p 0 {dev} {stat.st_ino + 9} {path}\n",
        f"1000-2000 r-xp 0 {dev} {stat.st_ino} {path}\n",
        f"2000-3000 r-xp 1 {dev} {stat.st_ino} {path}\n",
        f"3000-4000 r-xp 0 {dev} {stat.st_ino + 1} {path}\n",
        f"4000-5000 r-xp 0 {dev} {stat.st_ino + 2} {path} (deleted)\n",
    ]
    actual = probe._linux_libraries(lines)
    assert len(actual) == 3
    assert [item.kind for item in actual] == ["llvm"] * 3
    assert actual[0].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual[1].error == "mapped file does not match disk"
    assert actual[2].error == "mapped file was deleted"
