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
from rocke.runtime import torch_rocm_probe as probe
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_UNSEEN = "unseen"
_INCOMPLETE = "inconclusive"
_MATCH = "same_binaries_observed"
_MISMATCH = "different_binaries_observed"

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
    assert data["worker_status"] == "imported"
    assert data["torch_version"] == "fixture"
    assert data["torch_hip_version"] == "fixture HIP version"
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
                    "status": "imported",
                    "version": "fixture",
                    "hip_version": "fixture HIP version",
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
    assert report.worker_status == "imported"
    assert report.parent_libraries == parent
    assert report.torch_libraries == child
    outcomes = {_PAIR_OUTCOMES[hip], _PAIR_OUTCOMES[comgr]}
    if _MISMATCH in outcomes:
        assert report.status == _MISMATCH
    elif _INCOMPLETE in outcomes or outcomes == {_UNSEEN}:
        assert report.status == _INCOMPLETE
    else:
        assert report.status == _MATCH
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
            "torch_not_installed",
            "ModuleNotFoundError",
        ),
        (
            "raise ImportError('dependency missing')",
            "torch_import_error",
            "dependency missing",
        ),
        ("import os; os._exit(19)", "worker_error", "status 19"),
    ],
)
def test_import_failures_are_not_binary_mismatches(
    tmp_path, monkeypatch, source, status, reason
):
    (tmp_path / "torch.py").write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = probe.probe_torch_rocm()
    assert result.status == "inconclusive"
    assert result.worker_status == status
    assert reason in result.reason


def test_reports_native_crash(monkeypatch):
    def fail(*args, **kwargs):
        kwargs["stderr"].write(b"native loader failure\n")
        return subprocess.CompletedProcess(args[0], -11)

    monkeypatch.setattr(probe.subprocess, "run", fail)
    result = probe.probe_torch_rocm()
    assert result.status == "inconclusive"
    assert result.worker_status == "crashed"
    assert "-11" in result.reason
    assert result.stderr == "native loader failure"


def test_bounds_subprocess_runtime(tmp_path, monkeypatch):
    (tmp_path / "torch.py").write_text("import time; time.sleep(60)", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = probe.probe_torch_rocm(timeout=0.5)
    assert result.status == "inconclusive"
    assert result.worker_status == "timeout"


def test_unverifiable_and_ambiguous_binaries_do_not_match():
    known = probe.RocmLibrary("hip", "libamdhip64.so.7", "digest")
    other = probe.RocmLibrary("hip", "other/libamdhip64.so.7", "digest")
    unknown = probe.RocmLibrary("hip", "unreadable/libamdhip64.so.7")
    assert probe._compare((known,), (other,))[0] == "same_binaries_observed"
    for right in [(), (other, known), (unknown,)]:
        assert probe._compare((known,), right)[0] == "inconclusive"
    assert probe._compare((), ())[0] == "inconclusive"


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
