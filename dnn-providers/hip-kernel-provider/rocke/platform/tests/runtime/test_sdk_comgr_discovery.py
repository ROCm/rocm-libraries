# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDK ownership/paths stay independent of compiler/runtime initialization."""
from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
from rocke.core import lower_llvm
from rocke.runtime import comgr, runtime_coexistence as rc, _session, _sdk_comgr


@pytest.fixture
def sdk_site(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "_IS_WINDOWS", False)
    monkeypatch.setattr(_session, "session", _session.RuntimeSession())
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", lambda: [])
    site = tmp_path / "site-packages"
    core = site / "_rocm_sdk_core"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text("raise AssertionError('SDK initialized')\n")
    dist = site / "rocm_sdk_core-7.9.0rc1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: rocm-sdk-core\nVersion: 7.9.0rc1\n")
    (dist / "RECORD").write_text(
        "".join(
            f"_rocm_sdk_core/{path},,\n"
            for path in (
                "lib/libamd_comgr.so.3",
                "bin/amd_comgr0702.dll",
                "lib/libamdhip64.so.7",
                "bin/amdhip64_7.dll",
            )
        )
    )
    monkeypatch.syspath_prepend(str(site))
    for name in ("torch", "rocm_sdk", "_rocm_sdk_core"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (
        "ROCKE_COMGR_LIB",
        "ROCKE_HIP_LIB",
        "ROCM_PATH",
        "ROCM_HOME",
        "HIP_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    importlib.invalidate_caches()
    return core


@pytest.mark.parametrize(
    "windows,relative",
    [(False, "lib/libamd_comgr.so.3"), (True, "bin/amd_comgr0702.dll")],
)
@pytest.mark.parametrize("suffix", ["", "_nonce"])
def test_sdk_comgr_is_discovered_without_runtime_imports(
    sdk_site, monkeypatch, windows, relative, suffix
):
    core = sdk_site.with_name(sdk_site.name + suffix)
    if suffix:
        sdk_site.rename(core)
    library = core / relative
    library.parent.mkdir()
    library.touch()
    next(core.parent.glob("*.dist-info")).joinpath("RECORD").write_text(
        f"{core.name}/{relative},,\n"
    )
    monkeypatch.setattr(rc, "_IS_WINDOWS", windows)
    attempts = []
    monkeypatch.setattr(
        comgr.ctypes,
        "CDLL",
        lambda path: attempts.append(path) or SimpleNamespace(_name=path),
    )
    monkeypatch.setattr(comgr, "_add_dll_dir", lambda path: None)
    assert comgr.resolved_lib_path() == str(library)
    assert comgr._resolve_lib()._name == str(library)
    assert attempts == [str(library)]
    assert comgr.resolved_lib_rocm_version() == (7, 9)
    assert all(name not in sys.modules for name in ("torch", "rocm_sdk", core.name))


def test_sdk_owns_hip_for_torch_free_pair(sdk_site, monkeypatch):
    library = sdk_site / "lib/libamdhip64.so.7"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setattr(rc, "_rocm_root_libdirs", lambda: ["system/lib"])
    assert rc._candidate_lib_paths("amdhip64", "ROCKE_HIP_LIB", ["7"])[0] == str(
        library
    )
    assert "torch" not in sys.modules


def test_explicit_override_is_only_candidate(sdk_site, tmp_path, monkeypatch):
    override = str(tmp_path / "bad.so")
    monkeypatch.setenv("ROCKE_COMGR_LIB", override)
    attempts = []

    def fail(path):
        attempts.append(path)
        raise OSError("override unavailable")

    monkeypatch.setattr(comgr.ctypes, "CDLL", fail)
    with pytest.raises(comgr.ComgrError, match="override unavailable"):
        comgr._resolve_lib()
    assert attempts == [override]
    assert _session.session.compiler is None


def test_missing_sdk_has_no_system_fallback(sdk_site, monkeypatch):
    monkeypatch.setattr(rc, "_rocm_root_libdirs", lambda: ["system/lib"])
    assert _sdk_comgr.candidates() == []
    with pytest.raises(comgr.ComgrError, match="no candidate"):
        comgr._resolve_lib()


@pytest.mark.parametrize(
    "query", [comgr._require_compiler_info, lower_llvm._resolve_llvm_flavor]
)
def test_required_compiler_query_preserves_sdk_load_failure(
    sdk_site, monkeypatch, query
):
    library = sdk_site / "lib/libamd_comgr.so.3"
    library.parent.mkdir()
    library.touch()
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)

    def fail(path):
        raise OSError("missing compiler dependency")

    monkeypatch.setattr(comgr.ctypes, "CDLL", fail)
    with pytest.raises(comgr.ComgrError, match="Cannot determine LLVM flavor") as error:
        query()
    assert str(library) in str(error.value)
    assert "missing compiler dependency" in str(error.value)
    assert isinstance(error.value.__cause__, comgr.ComgrError)
    assert _session.session.compiler is None


@pytest.mark.parametrize("suffix", ["", "_nonce"])
@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("relative_site", [False, True])
@pytest.mark.parametrize("hip", [False, True])
def test_metadata_payload_preserves_parent_path_spelling(
    tmp_path, monkeypatch, suffix, windows, relative_site, hip
):
    monkeypatch.chdir(tmp_path)
    site = os.path.join("alias", "..", "site-packages")
    if not relative_site:
        site = os.path.join(str(tmp_path), site)
    if hip:
        relative = "bin/amdhip64_7.dll" if windows else "lib/libamdhip64.so.7"
    else:
        relative = "bin/amd_comgr0702.dll" if windows else "lib/libamd_comgr.so.3"
    entry = f"_rocm_sdk_core{suffix}/{relative}"
    distribution = SimpleNamespace(
        files=[entry],
        locate_file=lambda path: os.path.join(site, *str(path).split("/")),
    )
    monkeypatch.setattr(_sdk_comgr, "_distribution", lambda: distribution)
    expected = os.path.join(
        str(tmp_path), "alias", "..", "site-packages", *entry.split("/")
    )
    seen = []
    monkeypatch.setattr(
        _sdk_comgr.os.path,
        "isfile",
        lambda path: seen.append(path) or path == expected,
    )
    query = _sdk_comgr.hip_candidates if hip else _sdk_comgr.candidates
    assert query(windows=windows) == [expected]
    assert seen == [expected]


def _check_metadata_parent_site(
    sdk_site,
    tmp_path,
    monkeypatch,
    suffix,
    windows,
    relative_site,
    decoy_present,
    *,
    symlink,
):
    deployment = tmp_path / "deployment"
    site = deployment / "physical/site-packages"
    site.parent.mkdir(parents=True)
    sdk_site.parent.rename(site)
    core = site / ("_rocm_sdk_core" + suffix)
    if suffix:
        (site / "_rocm_sdk_core").rename(core)
    relative = "bin/amd_comgr0702.dll" if windows else "lib/libamd_comgr.so.3"
    library = core / relative
    library.parent.mkdir()
    library.touch()
    next(site.glob("*.dist-info")).joinpath("RECORD").write_text(
        f"{core.name}/{relative},,\n"
    )
    child = site.parent / "child"
    child.mkdir()
    alias = child
    if symlink:
        alias = deployment / "alias"
        try:
            alias.symlink_to(child, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlink unavailable: {error}")
    decoy = deployment / "site-packages" / core.name / relative
    if decoy_present:
        decoy.parent.mkdir(parents=True)
        decoy.touch()
    monkeypatch.chdir(tmp_path)
    spelling = os.path.join(str(alias), "..", "site-packages")
    if relative_site:
        spelling = (
            os.path.relpath(alias, tmp_path)
            + os.sep
            + os.path.join("..", "site-packages")
        )
    monkeypatch.syspath_prepend(spelling)
    monkeypatch.setattr(rc, "_IS_WINDOWS", windows)
    raw = _sdk_comgr._distribution().locate_file(f"{core.name}/{relative}")
    assert os.path.samefile(raw, library)
    selected = _sdk_comgr.candidates(windows=windows)
    assert len(selected) == 1 and os.path.isabs(selected[0])
    assert os.path.samefile(selected[0], library)
    assert _sdk_comgr.release_for_path(str(library)) == (7, 9)
    if decoy_present:
        assert _sdk_comgr.release_for_path(str(decoy)) is None
    assert all(name not in sys.modules for name in ("torch", "rocm_sdk", core.name))


@pytest.mark.parametrize("suffix", ["", "_nonce"])
@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("relative_site", [False, True])
@pytest.mark.parametrize("decoy_present", [False, True])
def test_metadata_site_preserves_ordinary_parent_traversal(
    sdk_site, tmp_path, monkeypatch, suffix, windows, relative_site, decoy_present
):
    _check_metadata_parent_site(
        sdk_site,
        tmp_path,
        monkeypatch,
        suffix,
        windows,
        relative_site,
        decoy_present,
        symlink=False,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/parent traversal semantics")
@pytest.mark.parametrize("suffix", ["", "_nonce"])
@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("relative_site", [False, True])
@pytest.mark.parametrize("decoy_present", [False, True])
def test_metadata_site_preserves_symlink_parent_traversal(
    sdk_site, tmp_path, monkeypatch, suffix, windows, relative_site, decoy_present
):
    _check_metadata_parent_site(
        sdk_site,
        tmp_path,
        monkeypatch,
        suffix,
        windows,
        relative_site,
        decoy_present,
        symlink=True,
    )


def test_record_rejects_unowned_names(sdk_site):
    library = sdk_site / "bin/amd_comgr_3.dll"
    library.parent.mkdir()
    library.touch()
    next(sdk_site.parent.glob("*.dist-info")).joinpath("RECORD").write_text(
        "_rocm_sdk_core/bin/amd_comgr_3.dll,,\n_rocm_sdk_core/../_rocm_sdk_core/lib/libamd_comgr.so.3,,\n"
    )
    assert _sdk_comgr.candidates(windows=True) == []
    assert _sdk_comgr.candidates() == []
