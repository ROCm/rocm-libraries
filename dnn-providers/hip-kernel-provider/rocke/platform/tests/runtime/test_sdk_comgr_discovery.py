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
from rocke.core import lower_llvm


@pytest.fixture
def sdk_site(tmp_path, monkeypatch):
    # Linux fixtures are the default; Windows cases select their mode explicitly.
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    monkeypatch.setattr(comgr, "_IS_WINDOWS", False)
    site = tmp_path / "site-packages"
    core = site / "_rocm_sdk_core"
    core.mkdir(parents=True)
    # Finding this package must never execute its initialization code.
    (core / "__init__.py").write_text("raise AssertionError('SDK imported')\n")
    dist = site / "rocm_sdk_core-7.9.0rc1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rocm-sdk-core\nVersion: 7.9.0rc1\n"
    )
    (dist / "RECORD").write_text(
        "".join(
            f"_rocm_sdk_core/{relative},,\n"
            for relative in (
                "lib/libamd_comgr.so.3",
                "lib/libamd_comgr.so",
                "bin/amd_comgr0605.dll",
                "bin/amd_comgr0701.dll",
                "bin/amd_comgr0702.dll",
                "bin/amd_comgr.dll",
            )
        )
    )
    monkeypatch.syspath_prepend(str(site))
    for name in ("torch", "rocm_sdk", "_rocm_sdk_core"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("ROCKE_COMGR_LIB", "ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rc, "_PROC_MAPS", str(tmp_path / "no-maps"))
    monkeypatch.setattr(rc, "_mapped_elf_lib_paths", lambda stem: [])
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


def test_explicit_override_is_only_candidate(sdk_site, tmp_path, monkeypatch):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    override = str(tmp_path / "unloadable.so")
    monkeypatch.setenv("ROCKE_COMGR_LIB", override)
    attempts = []

    def load(path):
        attempts.append(path)
        raise OSError("override unavailable")

    monkeypatch.setattr(comgr.ctypes, "CDLL", load)
    assert comgr.resolved_lib_path() == override
    with pytest.raises(comgr.ComgrError, match="override unavailable"):
        comgr._resolve_lib()
    assert attempts == [override]


@pytest.mark.parametrize("torch_loaded", [False, True])
def test_sdk_is_only_automatic_candidate(sdk_site, tmp_path, monkeypatch, torch_loaded):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    if torch_loaded:
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(__file__=str(tmp_path / "torch/__init__.py")),
        )
    monkeypatch.setattr(
        rc, "_torch_bundled_lib", lambda stem: pytest.fail("COMGR queried torch")
    )
    monkeypatch.setattr(
        rc, "_rocm_root_libdirs", lambda: pytest.fail("COMGR queried system")
    )
    monkeypatch.setenv("ROCKE_COMGR_LIB", "")
    assert rc._candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"]) == [
        str(library)
    ]


def test_missing_sdk_has_no_system_fallback(sdk_site, monkeypatch):
    assert rc._candidate_lib_paths("amd_comgr", "ROCKE_COMGR_LIB", ["3"]) == []
    with pytest.raises(comgr.ComgrError, match="no candidate path"):
        comgr._resolve_lib()


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


@pytest.mark.parametrize(
    "query", [comgr._require_compiler_info, lower_llvm._resolve_llvm_flavor]
)
@pytest.mark.parametrize("windows", [False, True])
def test_required_compiler_query_preserves_sdk_load_failure(
    sdk_site, monkeypatch, query, windows
):
    relative = "bin/amd_comgr.dll" if windows else "lib/libamd_comgr.so.3"
    library = sdk_site / relative
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_IS_WINDOWS", windows)
    monkeypatch.setattr(comgr, "_IS_WINDOWS", windows)
    monkeypatch.setattr(comgr, "_add_dll_dir", lambda path: None)
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    attempts = []

    def fail(path):
        attempts.append(path)
        raise OSError("missing compiler dependency")

    monkeypatch.setattr(comgr.ctypes, "CDLL", fail)
    with pytest.raises(comgr.ComgrError, match="Cannot determine LLVM flavor") as error:
        query()
    assert str(library) in str(error.value)
    assert "missing compiler dependency" in str(error.value)
    assert "offline IR emission" in str(error.value)
    assert isinstance(error.value.__cause__, comgr.ComgrError)
    assert attempts == [str(library)]
    assert comgr._lib is None
    assert comgr.loaded_compiler_info() is None
    assert attempts == [str(library), str(library)]


def test_auto_lowering_without_compiler_reports_extra(sdk_site, monkeypatch):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    with pytest.raises(comgr.ComgrError, match=r"rocke\[sdk-core\]") as error:
        comgr._require_compiler_info()
    assert "offline IR emission" in str(error.value)
    assert "no candidate path" in str(error.value)


def test_sdk_release_lookup_does_not_need_runtime_imports(sdk_site, monkeypatch):
    library = sdk_site / "lib" / "libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    assert comgr.resolved_lib_rocm_version() == (7, 9)
    for name in ("torch", "rocm_sdk", "_rocm_sdk_core"):
        assert name not in sys.modules


@pytest.mark.parametrize("suffix", ["", "_7_9_nonce"])
@pytest.mark.parametrize(
    "windows, relative",
    [(False, "lib/libamd_comgr.so.3"), (True, "bin/amd_comgr0702.dll")],
)
def test_record_owned_suffixed_payload(
    sdk_site, monkeypatch, suffix, windows, relative
):
    core = sdk_site.with_name(sdk_site.name + suffix)
    if suffix:
        sdk_site.rename(core)
    library = core / relative
    library.parent.mkdir()
    library.touch()
    dist = next(core.parent.glob("*.dist-info"))
    (dist / "RECORD").write_text(f"{core.name}/{relative},,\n")
    monkeypatch.setattr(rc, "_IS_WINDOWS", windows)
    assert comgr.resolved_lib_path() == str(library)
    assert comgr.resolved_lib_rocm_version() == (7, 9)
    assert core.name not in sys.modules


@pytest.mark.parametrize("alias", [False, True])
def test_torch_free_mapped_comgr_guard(sdk_site, tmp_path, monkeypatch, alias):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.write_bytes(b"\x7fELF")
    foreign = tmp_path / "libamd_comgr.so.3"
    if alias:
        foreign.symlink_to(library)
    else:
        foreign.write_bytes(b"\x7fELF")
    maps = tmp_path / "maps"
    maps.write_text(f"1000-2000 r-xp 0000 00:00 1 {foreign}\n")
    monkeypatch.setattr(rc, "_PROC_MAPS", str(maps))
    attempts = []
    monkeypatch.setattr(
        comgr.ctypes,
        "CDLL",
        lambda path: attempts.append(path) or SimpleNamespace(_name=path),
    )
    if alias:
        assert comgr._resolve_lib()._name == str(library)
        assert attempts == [str(library)]
    else:
        with pytest.raises(comgr.ComgrError, match="second COMGR"):
            comgr._resolve_lib()
        monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
        for query in (comgr._require_compiler_info, lower_llvm._resolve_llvm_flavor):
            with pytest.raises(comgr.ComgrError, match="second COMGR") as error:
                query()
            assert str(library) in str(error.value)
            assert str(foreign) in str(error.value)
            assert "separate process" in str(error.value)
            assert isinstance(error.value.__cause__, comgr.ComgrError)
        assert comgr.loaded_compiler_info() is None
        assert attempts == []


def test_multiple_mapped_comgrs_do_not_hide_conflict(sdk_site, tmp_path, monkeypatch):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.write_bytes(b"\x7fELF")
    foreign = tmp_path / "libamd_comgr.so.3"
    foreign.write_bytes(b"\x7fELF")
    maps = tmp_path / "maps"
    maps.write_text(
        "".join(f"1000-2000 r-xp 0000 00:00 1 {path}\n" for path in (library, foreign))
    )
    monkeypatch.setattr(rc, "_PROC_MAPS", str(maps))
    monkeypatch.setattr(
        comgr.ctypes, "CDLL", lambda path: pytest.fail("loaded competing compiler")
    )
    with pytest.raises(comgr.ComgrError, match="second COMGR"):
        comgr._resolve_lib()
    with pytest.raises(comgr.ComgrError, match="second COMGR") as error:
        comgr._require_compiler_info()
    assert str(library) in str(error.value)
    assert str(foreign) in str(error.value)
    assert isinstance(error.value.__cause__, comgr.ComgrError)


def test_record_rejects_unowned_or_malformed_paths(sdk_site):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    dist = next(sdk_site.parent.glob("*.dist-info"))
    (dist / "RECORD").write_text(
        "_rocm_sdk_core/../_rocm_sdk_core/lib/libamd_comgr.so.3,,\n_rocm_sdk_core/lib/libamd_comgr.so.debug,,\n"
    )
    assert comgr.resolved_lib_path() is None


def test_compatibility_hook_never_imports_torch(sdk_site, monkeypatch):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setitem(sys.modules, "torch", None)
    assert comgr.prefer_bundled_lib() == (7, 9)
    assert sys.modules["torch"] is None


def test_windows_mapping_introspection_fails_closed(monkeypatch):
    import ctypes
    from ctypes import wintypes

    def enumerate_modules(process, modules, size, needed):
        modules[0] = 1
        needed._obj.value = ctypes.sizeof(wintypes.HMODULE)
        return 1

    def filename(*args):
        return 0

    def current_process():
        return 1

    psapi = SimpleNamespace(
        EnumProcessModules=enumerate_modules, GetModuleFileNameExW=filename
    )
    kernel = SimpleNamespace(GetCurrentProcess=current_process)
    with monkeypatch.context() as patch:
        patch.setattr(rc, "_IS_WINDOWS", True)
        patch.setattr(os, "name", "nt")
        patch.setattr(
            ctypes,
            "WinDLL",
            lambda name, **kwargs: psapi if name == "psapi" else kernel,
            raising=False,
        )
        patch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
        with pytest.raises(OSError, match="cannot identify a mapped runtime library"):
            rc._mapped_lib_paths("amd_comgr")
