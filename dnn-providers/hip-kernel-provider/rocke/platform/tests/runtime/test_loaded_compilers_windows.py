# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Windows symbol-owner inventory; native execution cases never run in pytest's process."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rocke.runtime import _loaded_libraries
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env


@pytest.mark.parametrize("filename_failure", [False, True])
def test_windows_inventory_uses_forwarded_symbol_owner(
    tmp_path, monkeypatch, filename_failure
):
    path = str(tmp_path / "renamed-compiler.dll")
    open(path, "wb").close()

    def enumerate(process, modules, capacity, needed):
        modules[0], modules[1] = 1, 3
        needed._obj.value = 2 * ctypes.sizeof(ctypes.c_void_p)
        return 1

    def module(flags, address, output):
        output._obj.value = 3
        return 1

    def name(handle, buffer, capacity):
        if filename_failure:
            return 0
        buffer.value = path
        return len(path)

    kernel = SimpleNamespace(
        GetCurrentProcess=Mock(return_value=1),
        GetProcAddress=Mock(
            side_effect=lambda handle, symbol: (
                100 if symbol.endswith(b"version") else 101
            )
        ),
        GetModuleHandleExW=Mock(side_effect=module),
        GetModuleFileNameW=Mock(side_effect=name),
        FreeLibrary=Mock(),
    )
    psapi = SimpleNamespace(EnumProcessModules=Mock(side_effect=enumerate))
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, **kwargs: kernel if name == "kernel32" else psapi,
        raising=False,
    )
    monkeypatch.setattr(
        ctypes, "CDLL", lambda name, handle: SimpleNamespace(_handle=handle, _name=name)
    )
    if filename_failure:
        with pytest.raises(
            _loaded_libraries.ComgrError, match="identify loaded COMGR file"
        ):
            _loaded_libraries._windows()
        kernel.FreeLibrary.assert_called_once_with(3)
    else:
        found = _loaded_libraries._windows()
        assert len(found) == 1 and found[0].path == path
        assert found[0].library._handle == 3
        assert (
            sum(
                call.args[0] == 0x4 for call in kernel.GetModuleHandleExW.call_args_list
            )
            == 1
        )


@pytest.mark.skipif(sys.platform != "win32", reason="actual Windows DLL load execution")
@pytest.mark.parametrize("mode", ["match", "mismatch", "explicit"])
def test_windows_renamed_preload_version_admission(tmp_path, mode):
    compiler = shutil.which("clang++") or shutil.which("amdclang++")
    if not compiler:
        pytest.skip("Windows loader fixture requires the LLVM compiler")
    root = tmp_path / "Jos\u00e9-\u4f8b"
    root.mkdir()
    foreign = root / "compiler.dll"
    site = root / "site-packages"
    bin = site / "_rocm_sdk_core_windows" / "bin"
    bin.mkdir(parents=True)
    candidate = bin / "amd_comgr0702.dll"

    def build(path, major, trip=False):
        source = path.with_suffix(".cpp")
        text = (
            "#include <windows.h>\n#include <stddef.h>\n"
            'extern "C" __declspec(dllexport) void amd_comgr_get_version(size_t*a,size_t*b){*a=3;*b=4;}\n'
            'extern "C" __declspec(dllexport) int amd_comgr_do_action(int a,void*b,void*c,void*d){return 0;}\n'
            f'extern "C" __declspec(dllexport) void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){{*a={major};*b=0;*c=1;}}\n'
        )
        if trip:
            text += 'BOOL WINAPI DllMain(HMODULE,DWORD reason,LPVOID){if(reason==DLL_PROCESS_ATTACH&&GetModuleHandleW(L"compiler.dll"))ExitProcess(86);return TRUE;}\n'
        source.write_text(text)
        subprocess.run(
            [compiler, "-shared", str(source), "-o", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )

    build(foreign, 23)
    build(candidate, 20 if mode == "mismatch" else 23, True)
    dist = site / "rocm_sdk_core-7.9.0rc1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: rocm-sdk-core\nVersion: 7.9.0rc1\n")
    (dist / "RECORD").write_text("_rocm_sdk_core_windows/bin/amd_comgr0702.dll,,\n")
    env = rocke_subprocess_env()
    env.pop("ROCKE_COMGR_LIB", None)
    env["PYTHONPATH"] = str(site) + os.pathsep + env["PYTHONPATH"]
    script = r"""
import ctypes,os,sys
from rocke.runtime import comgr
retained=ctypes.CDLL(sys.argv[1])
if sys.argv[3]=='explicit':os.environ['ROCKE_COMGR_LIB']=sys.argv[2]
if sys.argv[3]=='match':
    info=comgr._require_compiler_info()
    assert info.llvm_version==(23,0,1)
    assert os.path.samefile(info.comgr_path,sys.argv[1])
    assert os.path.samefile(info.requested_comgr,sys.argv[2])
else:
    try:comgr._require_compiler_info()
    except comgr.ComgrError as error:assert any(word in str(error) for word in ('mismatch','exact-file'))
    else:raise AssertionError('incompatible compiler admitted')
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
