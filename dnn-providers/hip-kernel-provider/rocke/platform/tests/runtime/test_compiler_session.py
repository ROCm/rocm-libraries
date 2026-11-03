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
from unittest.mock import Mock

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
            "-Wl,-rpath,$ORIGIN",
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
