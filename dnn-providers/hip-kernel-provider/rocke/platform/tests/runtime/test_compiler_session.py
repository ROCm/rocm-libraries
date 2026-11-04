# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One owner, cold admission and a zero-negotiation warm compilation path."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from rocke.runtime import comgr, _session
from rocke.runtime._compiler_probe import probe_version
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env


@pytest.fixture(autouse=True)
def owner(monkeypatch):
    # Build/installed launchers deliberately forward real ownership. Unit
    # fixtures must declare their own configuration, not inherit that owner.
    for key in ("ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "ROCKE_LLVM_FLAVOR"):
        monkeypatch.delenv(key, raising=False)
    session = _session.RuntimeSession()
    monkeypatch.setattr(_session, "session", session)
    return session


def test_single_flight_and_warm_handle(owner):
    release, entered = threading.Event(), threading.Event()
    handle = SimpleNamespace(_name="selected")
    calls = []

    def load():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return handle

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(owner.resolve, "compiler", load) for _ in range(8)]
        assert entered.wait(5)
        release.set()
        assert all(future.result() is handle for future in futures)
    assert calls == [1]
    for _ in range(100):
        assert owner.resolve("compiler", lambda: pytest.fail("warm reload")) is handle


def test_changed_file_after_load_is_retained_but_not_admitted(owner, monkeypatch):
    from rocke.runtime import _compiler_query

    handle = SimpleNamespace(_name="selected")
    monkeypatch.setattr(comgr, "_candidate_lib_paths", lambda *args: ["selected"])
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", lambda: [])
    monkeypatch.setattr(comgr, "_add_dll_dir", lambda path: None)
    monkeypatch.setattr(comgr.ctypes, "CDLL", Mock(return_value=handle))
    monkeypatch.setattr(
        _compiler_query,
        "file_signature",
        Mock(side_effect=[(1, 2, 3, 4), (1, 9, 3, 4)]),
    )
    with pytest.raises(comgr.ComgrError, match="identity changed"):
        comgr._resolve_lib()
    for _ in range(2):
        with pytest.raises(comgr.ComgrError, match="identity changed"):
            comgr._require_compiler_info()
    assert owner.compiler is handle  # Retain the unusable image; never retry a load.
    comgr.ctypes.CDLL.assert_called_once()


def test_failed_acquisition_is_retryable(owner):
    load = Mock(
        side_effect=[comgr.ComgrError("load failed"), SimpleNamespace(_name="selected")]
    )
    with pytest.raises(comgr.ComgrError, match="load failed"):
        owner.resolve("compiler", load)
    assert owner.compiler is None
    assert owner.resolve("compiler", load)._name == "selected"
    assert load.call_count == 2


def test_failed_acquisition_refreshes_inputs_on_retry(owner, tmp_path, monkeypatch):
    missing, valid = tmp_path / "missing.so", tmp_path / "valid.so"
    valid.touch()
    handle = SimpleNamespace(_name=str(valid))
    attempts = []

    def load(path):
        attempts.append(path)
        if path == str(missing):
            raise OSError("missing compiler")
        return handle

    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", lambda: [])
    monkeypatch.setattr(comgr, "_add_dll_dir", lambda path: None)
    monkeypatch.setattr(comgr.ctypes, "CDLL", load)
    monkeypatch.setenv("ROCKE_COMGR_LIB", str(missing))
    with pytest.raises(comgr.ComgrError, match="missing compiler"):
        comgr._resolve_lib()
    assert owner.compiler is None and owner.hip is None
    monkeypatch.setenv("ROCKE_COMGR_LIB", str(valid))
    assert comgr._resolve_lib() is handle
    assert attempts == [str(missing), str(valid)]
    monkeypatch.setenv("ROCKE_COMGR_LIB", str(missing))
    assert comgr._resolve_lib() is handle
    assert owner.environment["ROCKE_COMGR_LIB"] == str(valid)
    assert attempts == [str(missing), str(valid)]


def test_failed_waiters_share_attempt_inputs_then_retry(owner, monkeypatch):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    calls = []
    waiter_count = 0

    def load():
        calls.append(owner.environment["ROCKE_COMGR_LIB"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            raise comgr.ComgrError("old attempt failed")
        return SimpleNamespace(_name="new")

    monkeypatch.setenv("ROCKE_COMGR_LIB", "old")
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(owner.resolve, "compiler", load)
        assert entered.wait(5)
        real_wait = owner._pending.done.wait

        def wait(*args, **kwargs):
            nonlocal waiter_count
            with owner.lock:
                waiter_count += 1
                if waiter_count == 7:
                    waiting.set()
            return real_wait(*args, **kwargs)

        monkeypatch.setattr(owner._pending.done, "wait", wait)
        others = [pool.submit(owner.resolve, "compiler", load) for _ in range(7)]
        assert waiting.wait(5)
        monkeypatch.setenv("ROCKE_COMGR_LIB", "new")
        release.set()
        for future in [first, *others]:
            with pytest.raises(comgr.ComgrError, match="old attempt failed"):
                future.result()
    assert calls == ["old"]
    assert owner.resolve("compiler", load)._name == "new"
    assert calls == ["old", "new"]


@pytest.mark.parametrize(
    "first,second,key",
    [("hip", "compiler", "ROCKE_COMGR_LIB"), ("compiler", "hip", "ROCKE_HIP_LIB")],
)
def test_second_resource_captures_its_cold_inputs(
    owner, monkeypatch, first, second, key
):
    retained = SimpleNamespace(_name="retained")
    assert owner.resolve(first, lambda: retained) is retained
    monkeypatch.setenv(key, "late-selection")
    seen = []

    def load():
        seen.append(owner.environment[key])
        if len(seen) == 1:
            raise RuntimeError("retryable selection failure")
        return SimpleNamespace(_name=owner.environment[key])

    with pytest.raises(RuntimeError, match="retryable selection failure"):
        owner.resolve(second, load)
    monkeypatch.setenv(key, "repaired-selection")
    selected = owner.resolve(second, load)
    assert selected._name == "repaired-selection"
    assert seen == ["late-selection", "repaired-selection"]
    monkeypatch.setenv(key, "ignored-warm-change")
    assert owner.resolve(second, lambda: pytest.fail("warm reload")) is selected
    assert owner.resolve(first, lambda: pytest.fail("retained reload")) is retained


@pytest.mark.parametrize("kind", ["compiler", "hip"])
def test_late_missing_override_reaches_second_loader(
    owner, tmp_path, monkeypatch, kind
):
    from rocke.runtime import hip_module

    other = "hip" if kind == "compiler" else "compiler"
    retained = SimpleNamespace(_name=str(tmp_path / "retained"))
    owner.resolve(other, lambda: retained)
    key = "ROCKE_COMGR_LIB" if kind == "compiler" else "ROCKE_HIP_LIB"
    missing = str(tmp_path / "late-missing")
    monkeypatch.setenv(key, missing)
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", lambda: [])
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", lambda: [])
    monkeypatch.setattr(_session, "check_pair", lambda hip, compiler: None)
    load = Mock(side_effect=OSError("exact missing file"))
    monkeypatch.setattr(ctypes, "CDLL", load)
    resolve = comgr._resolve_lib if kind == "compiler" else hip_module._resolve_hip
    error = comgr.ComgrError if kind == "compiler" else hip_module.HipError
    with pytest.raises(error, match="late-missing"):
        resolve()
    load.assert_called_once_with(missing)
    assert getattr(owner, kind) is None and getattr(owner, other) is retained


@pytest.mark.parametrize("kind", ["compiler", "initialize", "hip"])
@pytest.mark.parametrize(
    "relative", ["selected.so", "directory with spaces/selected.so"]
)
def test_explicit_paths_are_anchored_before_loading(
    owner, tmp_path, monkeypatch, kind, relative
):
    from rocke.runtime import hip_module

    monkeypatch.chdir(tmp_path)
    expected = os.path.join(str(tmp_path), relative)
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", lambda: [])
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", lambda: [])
    load = Mock(side_effect=OSError("not loadable"))
    monkeypatch.setattr(ctypes, "CDLL", load)
    if kind == "initialize":
        resolve = lambda: comgr.initialize_compiler(relative)
    elif kind == "compiler":
        monkeypatch.setenv("ROCKE_COMGR_LIB", relative)
        resolve = comgr._resolve_lib
    else:
        monkeypatch.setenv("ROCKE_HIP_LIB", relative)
        resolve = hip_module._resolve_hip
    error = hip_module.HipError if kind == "hip" else comgr.ComgrError
    with pytest.raises(error, match="selected.so"):
        resolve()
    load.assert_called_once_with(expected)


@pytest.mark.parametrize(
    "stem,key", [("amd_comgr", "ROCKE_COMGR_LIB"), ("amdhip64", "ROCKE_HIP_LIB")]
)
def test_explicit_candidates_preserve_symlink_parent_traversal(
    tmp_path, monkeypatch, stem, key
):
    from rocke.runtime import runtime_coexistence

    nested = tmp_path / "physical" / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(nested, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")
    physical = nested.parent / "selected.so"
    physical.touch()
    monkeypatch.chdir(tmp_path)
    relative = os.path.join("alias", "..", "selected.so")
    paths = runtime_coexistence._candidate_lib_paths(stem, key, [], {key: relative})
    assert paths == [os.path.join(str(tmp_path), relative)]
    assert os.path.samefile(paths[0], physical)


@pytest.mark.parametrize("explicit", [False, True])
def test_multiple_mapped_hip_providers_are_rejected_even_with_match(
    owner, tmp_path, monkeypatch, explicit
):
    from rocke.runtime import hip_module
    from rocke.runtime._loaded_libraries import LoadedLibrary
    from rocke.runtime._compiler_query import file_signature

    providers = []
    for name in ("selected.so", "foreign.so"):
        path = tmp_path / name
        path.touch()
        providers.append(
            LoadedLibrary(
                SimpleNamespace(_name=str(path)), str(path), file_signature(path)
            )
        )
    selected = providers[0].path
    monkeypatch.setattr(hip_module, "_candidate_lib_paths", lambda *args: [selected])
    if explicit:
        monkeypatch.setenv("ROCKE_HIP_LIB", selected)
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", lambda: providers)
    cleanup = Mock()
    monkeypatch.setattr(comgr._loaded_libraries, "release", cleanup)
    load = Mock(side_effect=AssertionError("new HIP load"))
    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(
        hip_module.HipError, match="multiple mapped HIP providers.*ambiguous"
    ):
        hip_module._resolve_hip()
    load.assert_not_called()
    assert cleanup.call_args_list == [call(item) for item in providers]
    assert owner.hip is None


def test_explicit_reinitialization_preserves_exact_file(owner, tmp_path, monkeypatch):
    selected, different = tmp_path / "selected", tmp_path / "different"
    selected.touch()
    different.touch()
    handle = SimpleNamespace(_name=str(selected))
    monkeypatch.setattr(comgr, "_load_lib", Mock(return_value=handle))
    assert comgr.initialize_compiler(str(selected)) is handle
    assert comgr.initialize_compiler(str(selected)) is handle
    with pytest.raises(comgr.ComgrError, match="already owns"):
        comgr.initialize_compiler(str(different))
    assert owner.compiler is handle
    comgr._load_lib.assert_called_once()


def test_precompiled_manifest_does_not_initialize_compiler(owner, monkeypatch):
    import rocke.run_manifest as runner
    from rocke.runtime import hip_module

    class Stop(Exception):
        pass

    runtime = SimpleNamespace(_name="runtime")
    monkeypatch.setattr(
        comgr, "_resolve_lib", lambda: pytest.fail("manifest compiler bootstrap")
    )
    monkeypatch.setattr(hip_module, "_load_lib", Mock(return_value=runtime))

    def run(*args, **kwargs):
        assert hip_module._resolve_hip() is runtime
        raise Stop

    monkeypatch.setattr(runner, "run_manifest", run)
    with pytest.raises(Stop):
        runner.main(["unused.hsaco", "unused.json"])
    assert owner.compiler is None and owner.hip is runtime
    assert hip_module._resolve_hip() is runtime
    hip_module._load_lib.assert_called_once()


def test_incompatible_new_hip_pair_is_refused_before_load(owner, tmp_path, monkeypatch):
    from rocke.runtime import hip_module

    selected = tmp_path / "selected/compiler.so"
    selected.parent.mkdir()
    selected.touch()
    windows = sys.platform == "win32"
    foreign = (
        tmp_path / "artifact" / ("amd_comgr.dll" if windows else "libamd_comgr.so.3")
    )
    foreign.parent.mkdir()
    foreign.touch()
    hip = foreign.parent / ("amdhip64.dll" if windows else "libamdhip64.so.7")
    hip.touch()
    owner.compiler = SimpleNamespace(_name=str(selected))
    monkeypatch.setenv("ROCKE_HIP_LIB", str(hip))
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", lambda: [])
    monkeypatch.setattr(
        hip_module.ctypes, "CDLL", Mock(side_effect=AssertionError("unsafe HIP load"))
    )
    with pytest.raises(hip_module.HipError, match="coherent exact-file pair"):
        hip_module._resolve_hip()
    hip_module.ctypes.CDLL.assert_not_called()


def test_preloaded_hip_alias_uses_existing_handle(owner, tmp_path, monkeypatch):
    from rocke.runtime import hip_module
    from rocke.runtime._loaded_libraries import LoadedLibrary
    from rocke.runtime._compiler_query import file_signature

    path = tmp_path / "runtime.so"
    path.touch()
    alias = tmp_path / "alias.so"
    os.link(path, alias)
    handle = SimpleNamespace(_name=str(path), _handle=0)
    mapped = LoadedLibrary(handle, str(path), file_signature(path))
    monkeypatch.setenv("ROCKE_HIP_LIB", str(alias))
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", lambda: [mapped])
    monkeypatch.setattr(
        hip_module.ctypes, "CDLL", Mock(side_effect=AssertionError("second HIP"))
    )
    assert hip_module._resolve_hip() is handle
    assert hip_module._resolve_hip() is handle
    hip_module.ctypes.CDLL.assert_not_called()


@pytest.mark.parametrize(
    "mode", ["foreign", "same-file", "alias", "multiple", "changed", "missing"]
)
def test_external_compiler_is_checked_before_new_hip(
    owner, tmp_path, monkeypatch, mode
):
    from rocke.runtime import hip_module
    from rocke.runtime._compiler_query import file_signature
    from rocke.runtime._loaded_libraries import LoadedLibrary

    windows = sys.platform == "win32"
    companion = tmp_path / ("amd_comgr.dll" if windows else "libamd_comgr.so.3")
    hip = tmp_path / ("amdhip64.dll" if windows else "libamdhip64.so.7")
    companion.touch()
    hip.touch()
    foreign = tmp_path / "foreign-compiler"
    foreign.touch()
    selected = companion if mode in ("same-file", "alias") else foreign
    if mode == "alias":
        selected = tmp_path / "compiler-alias"
        os.link(companion, selected)
    signature = file_signature(str(selected))
    if mode == "changed":
        signature = (signature[0], signature[1] + 1, *signature[2:])
    if mode == "missing":
        selected.unlink()
    provider = LoadedLibrary(
        SimpleNamespace(_name=str(selected), _handle=0), str(selected), signature
    )
    providers = [provider]
    if mode == "multiple":
        providers.append(
            LoadedLibrary(
                SimpleNamespace(_handle=0),
                str(companion),
                file_signature(str(companion)),
            )
        )
    inventory = Mock(return_value=providers)
    cleanup = Mock()
    runtime = SimpleNamespace(_name=str(hip))
    load = Mock(return_value=runtime)
    monkeypatch.setenv("ROCKE_HIP_LIB", str(hip))
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", lambda: [])
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", inventory)
    monkeypatch.setattr(comgr._loaded_libraries, "release", cleanup)
    monkeypatch.setattr(hip_module, "_add_dll_dir", lambda path: None)
    monkeypatch.setattr(hip_module.ctypes, "CDLL", load)
    monkeypatch.setattr(
        comgr, "_resolve_lib", lambda: pytest.fail("HIP bootstrapped COMGR")
    )
    if mode in ("same-file", "alias"):
        assert hip_module._resolve_hip() is runtime
        assert hip_module._resolve_hip() is runtime
        load.assert_called_once_with(str(hip))
    else:
        message = (
            "coherent exact-file pair"
            if mode == "foreign"
            else "ambiguous" if mode == "multiple" else "identity"
        )
        with pytest.raises(hip_module.HipError, match=message):
            hip_module._resolve_hip()
        load.assert_not_called()
    inventory.assert_called_once()
    assert cleanup.call_count == len(providers)
    assert owner.compiler is None


@pytest.mark.parametrize("pair", ["coherent", "alias", "foreign"])
@pytest.mark.parametrize("selection", ["same-file", "distinct-same-version"])
def test_mapped_compiler_validates_retained_hip(
    owner, tmp_path, monkeypatch, pair, selection
):
    from rocke.runtime._compiler_query import file_signature
    from rocke.runtime._loaded_libraries import LoadedLibrary

    windows = sys.platform == "win32"
    companion = tmp_path / ("amd_comgr.dll" if windows else "libamd_comgr.so.3")
    hip = tmp_path / ("amdhip64.dll" if windows else "libamdhip64.so.7")
    companion.touch()
    hip.touch()
    provider = companion
    if pair != "coherent":
        provider = tmp_path / "mapped-compiler"
        if pair == "alias":
            os.link(companion, provider)
        else:
            provider.touch()
    selected = provider
    if selection == "distinct-same-version":
        selected = tmp_path / "selected-compiler"
        selected.touch()
    handle = SimpleNamespace(_name=str(provider), _handle=0)
    mapped = LoadedLibrary(handle, str(provider), file_signature(provider))
    owner.hip = SimpleNamespace(_name=str(hip))
    inventory, cleanup = Mock(return_value=[mapped]), Mock()
    info = comgr.CompilerInfo(
        (23, 0, 1), "fixture", str(selected), str(provider), str(provider)
    )
    monkeypatch.setattr(comgr, "_candidate_lib_paths", lambda *args: [str(selected)])
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_compilers", inventory)
    monkeypatch.setattr(comgr._loaded_libraries, "release", cleanup)
    monkeypatch.setattr(comgr, "query_info", Mock(return_value=info))
    monkeypatch.setattr(comgr, "probe_version", Mock(return_value=info))
    monkeypatch.setattr(
        comgr.ctypes, "CDLL", Mock(side_effect=AssertionError("second compiler"))
    )
    if pair == "foreign":
        with pytest.raises(comgr.ComgrError, match="coherent exact-file pair"):
            comgr._resolve_lib()
        assert owner.compiler is None and owner.compiler_info is None
        assert not hasattr(handle, "_rocke_admission")
        cleanup.assert_called_once_with(mapped)
    else:
        assert comgr._resolve_lib() is handle
        assert comgr._resolve_lib() is handle
        assert owner.compiler is handle
        cleanup.assert_not_called()
    inventory.assert_called_once()
    comgr.ctypes.CDLL.assert_not_called()


def test_runtime_inventory_failure_preserves_hip_error_contract(owner, monkeypatch):
    from rocke.runtime import hip_module

    inventory = Mock(side_effect=comgr.ComgrError("runtime inventory unavailable"))
    monkeypatch.setattr(hip_module, "_candidate_lib_paths", lambda *args: ["hip"])
    monkeypatch.setattr(comgr._loaded_libraries, "loaded_runtimes", inventory)
    monkeypatch.setattr(
        hip_module.ctypes, "CDLL", Mock(side_effect=AssertionError("HIP load"))
    )
    with pytest.raises(hip_module.HipError, match="runtime inventory unavailable"):
        hip_module._resolve_hip()
    for query in (
        hip_module.get_device_arch,
        hip_module.get_device_target_id,
        hip_module.get_device_name,
    ):
        assert query() is None
    assert owner.hip is None and owner.compiler is None
    assert inventory.call_count == 4
    hip_module.ctypes.CDLL.assert_not_called()


def test_cached_evidence_bypasses_all_cold_work(owner, monkeypatch):
    def version(*outputs):
        for output, value in zip(outputs, (23, 0, 1)):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_uint))[0] = value

    library = SimpleNamespace(
        _name="selected",
        LLVMGetVersion=Mock(side_effect=version),
        amd_comgr_get_version=Mock(),
    )
    monkeypatch.setattr(comgr, "_load_lib", Mock(return_value=library))
    monkeypatch.setattr(comgr, "_library_for_symbol", lambda fn: "selected")
    info = comgr._require_compiler_info()
    monkeypatch.setattr(comgr, "_load_lib", lambda: pytest.fail("warm load"))
    monkeypatch.setattr(
        comgr._loaded_libraries,
        "loaded_compilers",
        lambda: pytest.fail("warm inventory"),
    )
    monkeypatch.setattr(
        comgr, "probe_version", lambda path: pytest.fail("warm subprocess")
    )
    monkeypatch.setattr(
        comgr, "_candidate_lib_paths", lambda *args: pytest.fail("warm metadata")
    )
    for _ in range(100):
        assert comgr._require_compiler_info() is info
    library.LLVMGetVersion.assert_called_once()


def library(
    root, name="libamd_comgr.so.3", major=23, patch=1, constructor="", version=True
):
    root.mkdir(parents=True, exist_ok=True)
    output = root / name
    source = "#include <stddef.h>\nvoid amd_comgr_get_version(size_t*a,size_t*b){*a=3;*b=4;}\nint amd_comgr_do_action(int a,void*b,void*c,void*d){return 0;}\nvoid fixture_touch(void){}\n"
    if version:
        source += f"void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){{*a={major};*b=0;*c={patch};}}\n"
    source += constructor
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-Wl,-soname,libamd_comgr.so.3",
            "-o",
            str(output),
        ],
        input=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("cc"), reason="ELF loader fixtures"
)
@pytest.mark.parametrize("kind", ["compiler", "initialize", "hip"])
@pytest.mark.parametrize("local", ["missing", "corrupt"])
def test_filename_override_cannot_search_for_another_file(tmp_path, kind, local):
    name = "libamdhip64.so.7" if kind == "hip" else "libamd_comgr.so.3"
    fallback = library(
        tmp_path / "loader",
        name=name,
        constructor="int hipInit(unsigned x){return 0;} int hipGetDeviceCount(int*n){*n=0;return 0;}",
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    if local == "corrupt":
        (cwd / name).write_text("not a shared library")
    code = """
import os,sys
from rocke.runtime import comgr,hip_module,_session
from rocke.runtime._loaded_libraries import loaded_compilers,loaded_runtimes,release
name,kind=sys.argv[1:]
if kind=='hip':
    os.environ['ROCKE_HIP_LIB']=name
    resolve=hip_module._resolve_hip
    error=hip_module.HipError
elif kind=='initialize':
    resolve=lambda:comgr.initialize_compiler(name)
    error=comgr.ComgrError
else:
    os.environ['ROCKE_COMGR_LIB']=name
    resolve=comgr._resolve_lib
    error=comgr.ComgrError
try:resolve()
except error as e:assert os.path.join(os.getcwd(),name) in str(e),str(e)
else:raise AssertionError('filename override loaded a search-path fallback')
assert _session.session.compiler is None and _session.session.hip is None
providers=loaded_runtimes() if kind=='hip' else loaded_compilers()
assert not providers
for item in providers:release(item)
"""
    env = rocke_subprocess_env()
    for key in ("ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "ROCKE_LLVM_FLAVOR"):
        env.pop(key, None)
    env["LD_LIBRARY_PATH"] = str(fallback.parent)
    result = subprocess.run(
        rocke_python_command("-c", code, name, kind),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("cc"), reason="ELF loader fixtures"
)
def test_real_multiple_hip_providers_are_refused(tmp_path):
    symbols = "int hipInit(unsigned x){return 0;} int hipGetDeviceCount(int*n){*n=0;return 0;}"
    paths = [
        library(tmp_path / name, name="libamdhip64.so.7", constructor=symbols)
        for name in ("one", "two")
    ]
    code = """
import ctypes,sys
from rocke.runtime import hip_module
from rocke.runtime._loaded_libraries import loaded_runtimes,release
libs=[ctypes.CDLL(p) for p in sys.argv[1:]]
providers=loaded_runtimes()
assert len(providers)==2
for item in providers:release(item)
hip_module._candidate_lib_paths=lambda *args:[sys.argv[1]]
try:hip_module._resolve_hip()
except hip_module.HipError as e:assert 'multiple mapped HIP providers' in str(e),str(e)
else:raise AssertionError('ambiguous HIP accepted')
"""
    env = rocke_subprocess_env()
    for key in ("ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "ROCKE_LLVM_FLAVOR"):
        env.pop(key, None)
    result = subprocess.run(
        rocke_python_command("-c", code, *map(str, paths)),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def sdk(site, major=23, patch=1, constructor=""):
    output = library(
        site / "_rocm_sdk_core_session/lib",
        major=major,
        patch=patch,
        constructor=constructor,
    )
    dist = site / "rocm_sdk_core-7.9.0rc1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: rocm-sdk-core\nVersion: 7.9.0rc1\n")
    (dist / "RECORD").write_text(
        "_rocm_sdk_core_session/lib/libamd_comgr.so.3,,\n_rocm_sdk_core_session/lib/libamdhip64.so.7,,\n"
    )
    return output


@pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("cc"), reason="ELF compiler fixtures"
)
@pytest.mark.parametrize(
    "mode",
    [
        "match",
        "mismatch",
        "patch-mismatch",
        "explicit",
        "alias",
        "unknown",
        "deleted",
        "replaced",
    ],
)
def test_renamed_preload_admission_and_fast_path(tmp_path, mode):
    foreign = library(tmp_path / "foreign", "compiler.so", version=mode != "unknown")
    site = tmp_path / "site-packages"
    # The candidate must never be opened in the parent containing the foreign
    # compiler. A child does not inherit these mappings, so it can query it.
    trip = (
        "#include <stdio.h>\n#include <string.h>\n#include <unistd.h>\n"
        '__attribute__((constructor))static void trip(){FILE*f=fopen("/proc/self/maps","r");char line[4096];if(!f)return;'
        f"while(fgets(line,sizeof(line),f))if(strstr(line,{json.dumps(str(foreign))}))_exit(86);fclose(f);}}\n"
    )
    candidate = sdk(
        site,
        major=20 if mode == "mismatch" else 23,
        patch=2 if mode == "patch-mismatch" else 1,
        constructor=trip,
    )
    env = rocke_subprocess_env()
    for key in ("ROCKE_COMGR_LIB", "ROCKE_LLVM_FLAVOR"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(site) + os.pathsep + env["PYTHONPATH"]
    script = r"""
import ctypes,os,sys
from rocke.runtime import comgr
from rocke.runtime import _session
retained=ctypes.CDLL(sys.argv[1])
mode=sys.argv[3]
if mode=='explicit':os.environ['ROCKE_COMGR_LIB']=sys.argv[2]
if mode=='alias':os.environ['ROCKE_COMGR_LIB']=sys.argv[1]
if mode in ('deleted','replaced'):
    os.unlink(sys.argv[1])
    if mode=='replaced':os.link(sys.argv[2],sys.argv[1])
if mode in ('match','alias'):
    info=comgr._require_compiler_info()
    assert info.llvm_version==(23,0,1)
    assert os.path.samefile(info.comgr_path,sys.argv[1])
    if mode=='match':assert os.path.samefile(info.requested_comgr,sys.argv[2])
    comgr.probe_version=lambda path:(_ for _ in ()).throw(AssertionError('warm probe'))
    comgr._loaded_libraries.loaded_compilers=lambda:(_ for _ in ()).throw(AssertionError('warm inventory'))
    for _ in range(20):assert comgr._require_compiler_info() is info
else:
    try:comgr._require_compiler_info()
    except comgr.ComgrError as error:
        message=str(error)
        assert any(word in message for word in ('mismatch','exact-file','identity','evidence'))
    else:raise AssertionError('unsafe compiler admitted')
print('OK')
"""
    result = subprocess.run(
        rocke_python_command("-c", script, str(foreign), str(candidate), mode),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "OK"


@pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("cc"),
    reason="ELF runtime pair fixtures",
)
@pytest.mark.parametrize("order", ["compile-first", "hip-first"])
@pytest.mark.parametrize("pair", ["coherent", "foreign"])
def test_preloaded_runtime_compiler_pair_both_orders(tmp_path, order, pair):
    site = tmp_path / "site-packages"
    companion = sdk(site)
    provider = (
        companion
        if pair == "coherent"
        else library(tmp_path / "foreign", "compiler.so")
    )
    hip = companion.parent / "libamdhip64.so.7"
    # No compiler dependency: HIP-only use need not bootstrap COMGR. The SDK
    # metadata still declares the physical companion required for compilation.
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(hip)],
        input="int hipInit(unsigned x){return 0;}\nint hipGetDeviceCount(int*x){*x=0;return 0;}\n",
        check=True,
        capture_output=True,
        text=True,
    )
    env = rocke_subprocess_env()
    for key in ("ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "ROCKE_LLVM_FLAVOR"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(site) + os.pathsep + env["PYTHONPATH"]
    script = r"""
import ctypes,os,sys
from rocke.runtime import comgr,hip_module,_session
provider,hip,order,pair=sys.argv[1:]
external_compiler=ctypes.CDLL(provider)
external_hip=ctypes.CDLL(hip)
if order=='hip-first':
    assert os.path.samefile(hip_module._resolve_hip()._name,hip)
    assert _session.session.compiler is None
try:
    info=comgr._require_compiler_info()
    assert info.llvm_version==(23,0,1)
    assert os.path.samefile(info.comgr_path,provider)
    assert os.path.samefile(hip_module._resolve_hip()._name,hip)
except (comgr.ComgrError,hip_module.HipError) as error:
    assert pair=='foreign',str(error)
    assert 'coherent exact-file pair' in str(error)
    if order=='hip-first':assert _session.session.compiler is None
    else:assert _session.session.hip is None
else:
    assert pair=='coherent','incoherent pair admitted'
    _session.check_pair(hip,info.comgr_path)
    comgr._loaded_libraries.loaded_compilers=lambda:(_ for _ in ()).throw(AssertionError('warm inventory'))
    for _ in range(3):assert comgr._require_compiler_info() is info
print('OK')
"""
    result = subprocess.run(
        rocke_python_command("-c", script, str(provider), str(hip), order, pair),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "OK"


@pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("cc"), reason="ELF probe fixtures"
)
@pytest.mark.parametrize("failure", ["crash", "timeout", "output"])
def test_version_probe_fails_closed(tmp_path, failure):
    actions = {
        "crash": "_exit(17);",
        "timeout": "sleep(30);",
        "output": 'for(int i=0;i<100000;++i)puts("noise");',
    }
    constructor = (
        "#include <unistd.h>\n#include <stdio.h>\n__attribute__((constructor))static void action(){"
        + actions[failure]
        + "}\n"
    )
    selected = library(tmp_path, constructor=constructor)
    with pytest.raises(comgr.ComgrError, match="probe"):
        probe_version(
            str(selected), timeout=0.5 if failure == "timeout" else 5, limit=4096
        )


@pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("cc"),
    reason="ELF runtime pair fixtures",
)
@pytest.mark.parametrize("order", ["compile-first", "hip-first"])
def test_coherent_sdk_pair_both_orders(tmp_path, order):
    site = tmp_path / "site-packages"
    compiler = sdk(site)
    hip = compiler.parent / "libamdhip64.so.7"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-x",
            "none",
            str(compiler),
            # Match owned SDK payloads: bind this fixture's companion before
            # the ambient loader paths needed by the Python interpreter.
            "-Wl,--disable-new-dtags,-rpath,$ORIGIN",
            "-o",
            str(hip),
        ],
        input="extern void fixture_touch(void);\nint hip_fixture(){fixture_touch();return 0;}\n",
        check=True,
        capture_output=True,
        text=True,
    )
    env = rocke_subprocess_env()
    for key in (
        "ROCKE_COMGR_LIB",
        "ROCKE_HIP_LIB",
        "ROCKE_LLVM_FLAVOR",
        "ROCM_PATH",
        "ROCM_HOME",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(site) + os.pathsep + env["PYTHONPATH"]
    script = r"""
import os,sys
from rocke.runtime import comgr,hip_module
if sys.argv[3]=='hip-first': assert hip_module._resolve_hip().hip_fixture()==0
info=comgr._require_compiler_info()
assert info.llvm_version==(23,0,1)
assert os.path.samefile(info.comgr_path,sys.argv[1])
assert os.path.samefile(hip_module._resolve_hip()._name,sys.argv[2])
print('OK')
"""
    result = subprocess.run(
        rocke_python_command("-c", script, str(compiler), str(hip), order),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
