# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Discover the shared SDK compiler without importing or initializing runtimes."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import pytest

from rocke.runtime import comgr
from rocke.runtime import runtime_coexistence as rc


@pytest.fixture
def sdk_site(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    core = site / "_rocm_sdk_core"
    core.mkdir(parents=True)
    # Finding this package must never execute its initialization code.
    (core / "__init__.py").write_text("raise AssertionError('SDK imported')\n")
    monkeypatch.syspath_prepend(str(site))
    for name in ("torch", "rocm_sdk", "_rocm_sdk_core"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("ROCKE_COMGR_LIB", "ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rc, "_PROC_MAPS", str(tmp_path / "no-maps"))
    monkeypatch.setattr(rc, "_rocm_root_libdirs", lambda: [str(tmp_path / "system")])
    monkeypatch.setattr(comgr, "_lib", None)
    importlib.invalidate_caches()
    return core


@pytest.mark.parametrize(
    "windows, relative",
    [
        (False, "lib/libamd_comgr.so.3"),
        (True, "bin/amd_comgr0605.dll"),
        (True, "bin/amd_comgr0701.dll"),
        (True, "bin/amd_comgr0702.dll"),
        (True, "bin/amd_comgr.dll"),
    ],
)
def test_sdk_comgr_is_discovered_without_runtime_imports(
    sdk_site, monkeypatch, windows, relative
):
    library = sdk_site / relative
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_IS_WINDOWS", windows)
    monkeypatch.setattr(comgr, "_IS_WINDOWS", windows)
    attempts = []

    def load(path):
        attempts.append(path)
        return SimpleNamespace(_name=path)

    monkeypatch.setattr(comgr.ctypes, "CDLL", load)
    monkeypatch.setattr(comgr, "_add_dll_dir", lambda path: None)
    assert comgr.resolved_lib_path() == str(library)
    assert comgr._resolve_lib()._name == str(library)
    assert attempts == [str(library)]
    for name in ("torch", "rocm_sdk", "_rocm_sdk_core"):
        assert name not in sys.modules


def test_explicit_override_precedes_sdk_and_preserves_failure_fallback(
    sdk_site, tmp_path, monkeypatch
):
    library = sdk_site / "lib" / "libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    override = str(tmp_path / "unloadable.so")
    monkeypatch.setenv("ROCKE_COMGR_LIB", override)
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    monkeypatch.setattr(comgr, "_IS_WINDOWS", False)
    attempts = []

    def load(path):
        attempts.append(path)
        if path == override:
            raise OSError("override unavailable")
        return SimpleNamespace(_name=path)

    monkeypatch.setattr(comgr.ctypes, "CDLL", load)
    assert comgr._resolve_lib()._name == str(library)
    assert attempts == [override, str(library)]


def test_loaded_torch_comgr_precedes_sdk(sdk_site, tmp_path, monkeypatch):
    library = sdk_site / "lib" / "libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    mapped = str(tmp_path / "torch" / "libamd_comgr.so")
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    monkeypatch.setattr(rc, "_torch_bundled_lib", lambda stem: mapped)
    monkeypatch.setattr(rc, "_torch_comgr_is_stale", lambda path: False)
    assert rc._candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"])[:2] == [
        mapped,
        str(library),
    ]


def test_active_sdk_comgr_is_not_probed_twice(sdk_site, monkeypatch):
    library = sdk_site / "lib" / "libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    monkeypatch.setattr(rc, "_torch_bundled_lib", lambda stem: str(library))
    monkeypatch.setattr(rc, "_mapped_lib", lambda stem: str(library))
    paths = rc._candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"])
    assert paths[0] == str(library)
    assert paths.count(str(library)) == 1


def test_missing_sdk_comgr_keeps_system_fallback(sdk_site, monkeypatch):
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    paths = rc._candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"])
    system = rc._rocm_root_libdirs()[0]
    assert paths[:2] == [
        os.path.join(system, "libamd_comgr.so"),
        os.path.join(system, "libamd_comgr.so.3"),
    ]


def test_sdk_does_not_change_linux_hip_selection(sdk_site, monkeypatch):
    library = sdk_site / "lib" / "libamdhip64.so.7"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    paths = rc._candidate_lib_paths("amdhip64", "ROCKE_HIP_LIB", ["7"])
    assert str(library) not in paths


def test_unloadable_sdk_reports_extra_and_original_error(sdk_site, monkeypatch):
    library = sdk_site / "lib" / "libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    monkeypatch.setattr(comgr, "_IS_WINDOWS", False)

    def fail(path):
        raise OSError("missing compiler dependency")

    monkeypatch.setattr(comgr.ctypes, "CDLL", fail)
    with pytest.raises(comgr.ComgrError, match=r"rocke\[sdk-core\]") as error:
        comgr._load_lib()
    assert str(library) in str(error.value)
    assert "missing compiler dependency" in str(error.value)
    assert "AMD ROCm wheel index" in str(error.value)


def test_auto_lowering_without_compiler_reports_extra(monkeypatch):
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: None)
    with pytest.raises(comgr.ComgrError, match=r"rocke\[sdk-core\]") as error:
        comgr._require_compiler_info()
    assert "offline IR emission" in str(error.value)


def test_sdk_release_lookup_does_not_need_runtime_imports(sdk_site, monkeypatch):
    from importlib import metadata

    library = sdk_site / "lib" / "libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(metadata, "version", lambda name: "7.9.0rc1")
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    assert comgr.resolved_lib_rocm_version() == (7, 9)
    for name in ("torch", "rocm_sdk", "_rocm_sdk_core"):
        assert name not in sys.modules
