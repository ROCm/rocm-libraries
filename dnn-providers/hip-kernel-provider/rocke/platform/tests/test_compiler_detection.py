# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compiler selection follows loaded code and preserves its provenance."""

import ctypes
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from rocke.core import lower_llvm
from rocke.core.ir import IRBuilder
from rocke.runtime import comgr


@pytest.fixture(autouse=True)
def fresh_query(monkeypatch):
    monkeypatch.setattr(comgr, "_llvm_version_lib", None)
    monkeypatch.setattr(comgr, "_compiler_info", None)
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)


def compiler_library(version, name="compiler.so"):
    def query(*outputs):
        for pointer, value in zip(outputs, version):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint))[0] = value

    return SimpleNamespace(
        _name=name,
        LLVMGetVersion=Mock(side_effect=query),
        amd_comgr_get_version=Mock(),
    )


def test_direct_query_ignores_package_metadata_and_reports_loaded_origin(monkeypatch):
    lib = compiler_library((23, 1, 4), "requested-comgr.so")
    monkeypatch.setattr(comgr, "_resolve_lib", lambda: lib)
    monkeypatch.setattr(
        comgr,
        "_library_for_symbol",
        lambda fn: "loaded-llvm.so" if fn is lib.LLVMGetVersion else "loaded-comgr.so",
    )
    with patch.object(
        comgr, "resolved_lib_rocm_version", side_effect=AssertionError("metadata")
    ), patch.object(
        comgr, "_probe_llvm_version", side_effect=AssertionError("unneeded probe")
    ):
        info = comgr.loaded_compiler_info()
        assert asdict(info) == {
            "llvm_version": (23, 1, 4),
            "source": "LLVMGetVersion",
            "requested_comgr": "requested-comgr.so",
            "comgr_path": "loaded-comgr.so",
            "query_library_path": "loaded-llvm.so",
        }
        assert lower_llvm._resolve_llvm_flavor() == "llvm23"
        assert comgr.loaded_compiler_info() is info
        assert lib.LLVMGetVersion.call_count == 1


def test_query_cache_tracks_the_handle_not_its_filename(monkeypatch):
    first = compiler_library((20, 0, 0))
    second = compiler_library((23, 0, 0))
    monkeypatch.setattr(comgr, "_library_for_symbol", lambda fn: None)
    monkeypatch.setattr(comgr, "_resolve_lib", lambda: first)
    assert comgr.loaded_compiler_info().llvm_version == (20, 0, 0)
    monkeypatch.setattr(comgr, "_resolve_lib", lambda: second)
    assert comgr.loaded_compiler_info().llvm_version == (23, 0, 0)


def test_hidden_llvm_uses_same_comgr_and_records_probe(monkeypatch):
    lib = SimpleNamespace(
        _name="static-comgr.so",
        amd_comgr_do_action=Mock(),
        amd_comgr_get_version=Mock(),
    )
    monkeypatch.setattr(comgr, "_resolve_lib", lambda: lib)
    monkeypatch.setattr(comgr, "_library_for_symbol", lambda fn: "static-comgr.so")
    with patch.object(comgr, "_probe_llvm_version", return_value=(20, 0, 0)) as probe:
        info = comgr.loaded_compiler_info()
    probe.assert_called_once_with(lib)
    assert info.llvm_version == (20, 0, 0)
    assert info.source == "COMGR preprocessing"
    assert info.query_library_path == info.comgr_path == "static-comgr.so"


def test_unqueryable_loaded_library_is_unknown_not_another_install(monkeypatch):
    lib = SimpleNamespace(_name="unknown-comgr.so")
    monkeypatch.setattr(comgr, "_resolve_lib", lambda: lib)
    with patch.object(
        comgr, "_probe_llvm_version", side_effect=comgr.ComgrError("probe failed")
    ), patch.object(
        comgr, "resolved_lib_rocm_version", side_effect=AssertionError("metadata")
    ):
        info = comgr.loaded_compiler_info()
        assert info.llvm_version is None
        assert info.source == "unavailable"
        assert lower_llvm._resolve_llvm_flavor() == "llvm22"
        comgr._assert_ir_flavor_matches_lib(
            f'target datalayout = "{lower_llvm._datalayout_for_flavor("llvm23")}"'
        )


def test_unavailable_library_does_not_prevent_offline_emission(monkeypatch):
    with patch.object(comgr, "_resolve_lib", side_effect=comgr.ComgrError("missing")):
        assert comgr.loaded_compiler_info() is None
        assert lower_llvm._resolve_llvm_flavor() == "llvm22"
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    with patch.object(comgr, "_resolve_lib", side_effect=AssertionError("offline")):
        assert lower_llvm._resolve_llvm_flavor() == "llvm23"


def test_auto_lowering_recovers_then_retains_success(monkeypatch):
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    monkeypatch.setattr(comgr, "_lib", None)
    monkeypatch.setattr(comgr, "_library_for_symbol", lambda fn: None)
    library = compiler_library((23, 0, 0))
    load = Mock(side_effect=[comgr.ComgrError("missing"), library])
    monkeypatch.setattr(comgr, "_load_lib", load)
    builder = IRBuilder("compiler_recovery")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.const_i32(1)
    first = lower_llvm.lower_kernel_to_llvm(builder.kernel)
    assert lower_llvm._datalayout_for_flavor("llvm22") in first
    second = lower_llvm.lower_kernel_to_llvm(builder.kernel)
    assert lower_llvm._datalayout_for_flavor("llvm23") in second
    info = comgr.loaded_compiler_info()
    assert lower_llvm.lower_kernel_to_llvm(builder.kernel) == second
    assert comgr.loaded_compiler_info() is info
    assert load.call_count == 2
    library.LLVMGetVersion.assert_called_once()


def test_unqueryable_success_is_retained_without_retry(monkeypatch):
    monkeypatch.setattr(comgr, "_lib", None)
    monkeypatch.setattr(comgr, "_library_for_symbol", lambda fn: None)
    load = Mock(return_value=SimpleNamespace(_name="unknown.so"))
    monkeypatch.setattr(comgr, "_load_lib", load)
    probe = Mock(side_effect=comgr.ComgrError("unqueryable"))
    monkeypatch.setattr(comgr, "_probe_llvm_version", probe)
    info = comgr.loaded_compiler_info()
    assert info.llvm_version is None
    assert comgr.loaded_compiler_info() is info
    assert lower_llvm._resolve_llvm_flavor() == "llvm22"
    load.assert_called_once()
    probe.assert_called_once()


@pytest.mark.parametrize("source", ["api", "environment"])
def test_explicit_lowering_bypasses_library_load(monkeypatch, source):
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    load = Mock(side_effect=AssertionError("explicit lowering loaded COMGR"))
    monkeypatch.setattr(comgr, "_load_lib", load)
    monkeypatch.setattr(comgr, "_lib", None)
    builder = IRBuilder("explicit_flavor")
    builder.const_i32(1)
    kwargs = {"llvm_flavor": "llvm23"} if source == "api" else {}
    if source == "environment":
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    ir = lower_llvm.lower_kernel_to_llvm(builder.kernel, **kwargs)
    assert lower_llvm._datalayout_for_flavor("llvm23") in ir
    load.assert_not_called()


@pytest.mark.parametrize(
    "major,flavor",
    [
        (19, "llvm20"),
        (20, "llvm20"),
        (21, "llvm22"),
        (22, "llvm22"),
        (23, "llvm23"),
        (24, "llvm23"),
    ],
)
def test_llvm_flavor_boundaries(major, flavor):
    assert lower_llvm._flavor_for_llvm(major) == flavor


def test_override_does_not_falsify_guard_evidence(monkeypatch):
    info = comgr.CompilerInfo(
        (20, 0, 0), "COMGR preprocessing", "requested.so", "loaded.so", "loaded.so"
    )
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: info)
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    assert lower_llvm._resolve_llvm_flavor() == "llvm23"
    with pytest.raises(comgr.ComgrError) as error:
        comgr._assert_ir_flavor_matches_lib(
            f'target datalayout = "{lower_llvm._datalayout_for_flavor("llvm23")}"'
        )
    assert info.describe() in str(error.value)


def test_loader_skips_unloadable_candidate_before_querying(monkeypatch):
    lib = compiler_library((23, 0, 0), "second.so")
    monkeypatch.setattr(comgr, "_lib", None)
    monkeypatch.setattr(
        comgr, "_candidate_lib_paths", lambda *args: ["first.so", "second.so"]
    )
    monkeypatch.setattr(comgr, "_add_dll_dir", lambda path: None)
    monkeypatch.setattr(comgr, "_library_for_symbol", lambda fn: "second.so")
    with patch.object(
        ctypes, "CDLL", side_effect=[OSError("cannot load"), lib]
    ) as load:
        info = comgr.loaded_compiler_info()
    assert info.llvm_version == (23, 0, 0)
    assert info.requested_comgr == "second.so"
    assert load.call_count == 2


def test_failed_preprocessing_releases_acquired_handles():
    lib = SimpleNamespace()
    acquired = []
    released = []

    def create(*args):
        value = len(acquired) + 1
        ctypes.cast(args[-1], ctypes.POINTER(comgr._Handle))[0].handle = value
        acquired.append(value)
        return 0

    def destroy(handle):
        released.append(handle.handle)
        return 0

    for name in ("create_data_set", "create_data", "create_action_info"):
        setattr(lib, "amd_comgr_" + name, Mock(side_effect=create))
    for name in ("destroy_data_set", "release_data", "destroy_action_info"):
        setattr(lib, "amd_comgr_" + name, Mock(side_effect=destroy))
    for name in (
        "set_data",
        "set_data_name",
        "data_set_add",
        "get_isa_name",
        "action_info_set_isa_name",
        "action_info_set_language",
        "action_data_get_data",
        "get_data",
    ):
        setattr(lib, "amd_comgr_" + name, Mock(return_value=0))
    lib.amd_comgr_do_action = Mock(return_value=1)
    with pytest.raises(comgr.ComgrError, match="status=1"):
        comgr._probe_llvm_version(lib)
    assert released == list(reversed(acquired))
