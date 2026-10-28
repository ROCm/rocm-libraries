# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Standalone regression test for tasks.py helpers -- run directly with
# `pytest shared/stinkytofu/tests/test_tasks.py`. Not wired into CTest: the
# python_module/tests/ glob in tests/CMakeLists.txt is scoped to tests that
# import the compiled stinkytofu module; this one only exercises tasks.py
# itself and needs no build.

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks


def _touch_exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    path.chmod(0o755)


def test_parse_vcvars_env_parses_key_value_lines():
    stdout = (
        "VSINSTALLDIR=C:\\VS\\2022\\BuildTools\\\nINCLUDE=C:\\Some\\Include\\Path\n"
    )
    env = tasks._parse_vcvars_env(stdout)
    assert env["VSINSTALLDIR"] == "C:\\VS\\2022\\BuildTools\\"
    assert env["INCLUDE"] == "C:\\Some\\Include\\Path"


def test_parse_vcvars_env_tolerates_replacement_characters():
    # Simulates what errors="replace" produces when vcvarsall.bat's banner
    # text contains a byte undecodable under the active code page (e.g. a
    # JIS/Shift-JIS system locale with an English-language VS install) --
    # the exact scenario _setup_msvc_env()'s errors="replace" is meant to
    # survive instead of crashing on.
    stdout = (
        "Copyright �� 2022 banner text\n"
        "� garbage line with no equals sign\n"
        "\n"
        "VSINSTALLDIR=C:\\VS\\2022\\BuildTools\\\n"
        "INCLUDE=C:\\Some\\Include\\Path\n"
    )
    env = tasks._parse_vcvars_env(stdout)
    assert env["VSINSTALLDIR"] == "C:\\VS\\2022\\BuildTools\\"
    assert env["INCLUDE"] == "C:\\Some\\Include\\Path"


def test_parse_vcvars_env_empty_stdout():
    assert tasks._parse_vcvars_env("") == {}


def test_rocm_compiler_pair_comes_from_llvm_bin(tmp_path):
    llvm = tmp_path / "rocm" / "lib" / "llvm" / "bin"
    _touch_exe(llvm / "amdclang++")
    _touch_exe(llvm / "amdclang")

    def which(name):
        if name in ("g++", "gcc", "clang++", "clang"):
            return f"/usr/bin/{name}"
        return None

    cxx, cc, notice = tasks._resolve_build_compilers(
        tmp_path / "rocm", gcc=False, which=which
    )
    assert cxx == (llvm / "amdclang++").as_posix()
    assert cc == (llvm / "amdclang").as_posix()
    assert notice is None


def test_rocm_without_amdclang_does_not_use_the_system_compiler(tmp_path):
    (tmp_path / "rocm" / "bin").mkdir(parents=True)

    def which(name):
        if name in ("clang++", "clang", "g++", "gcc"):
            return f"/usr/bin/{name}"
        return None

    with pytest.raises(SystemExit, match="amdclang"):
        tasks._resolve_build_compilers(tmp_path / "rocm", gcc=False, which=which)


def test_no_rocm_keeps_c_and_cxx_in_the_same_pair(tmp_path):
    def which(name):
        found = {
            "amdclang++": "/opt/rocm/bin/amdclang++",
            "clang++": "/usr/bin/clang++",
            "clang": "/usr/bin/clang",
            "gcc": "/usr/bin/gcc",
        }
        return found.get(name)

    cxx, cc, notice = tasks._resolve_build_compilers(
        tmp_path / "missing", gcc=False, which=which
    )
    assert (cxx, cc) == ("/usr/bin/clang++", "/usr/bin/clang")
    assert notice.startswith("ROCm not found;")


def test_gcc_flag_requires_both_compilers(tmp_path):
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)

    def which(name):
        return "/usr/bin/g++" if name == "g++" else None

    with pytest.raises(SystemExit, match="g\\+\\+ and gcc"):
        tasks._resolve_build_compilers(rocm, gcc=True, which=which)


def test_windows_without_rocm_can_use_msvc(tmp_path):
    def which(name):
        return "C:/VS/cl.exe" if name == "cl" else None

    cxx, cc, notice = tasks._resolve_build_compilers(
        tmp_path / "missing", gcc=False, which=which, platform="win32"
    )
    assert (cxx, cc) == ("C:/VS/cl.exe", "C:/VS/cl.exe")
    assert notice.startswith("ROCm not found;")


def test_detect_rocm_windows_without_install_is_not_an_error(monkeypatch):
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(tasks.shutil, "which", lambda _name: None)
    monkeypatch.setattr(tasks.sys, "platform", "win32")
    assert not tasks._detect_rocm().is_dir()
