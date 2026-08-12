################################################################################
# Characterization tests for Tensile.Toolchain.Validators
#
# ADD-ONLY: pins current behavior of the toolchain component validators.
# Posix-focused (the container is Linux); Windows-only branches are documented
# as resistance rather than exercised (os.name cannot be flipped meaningfully
# because the Windows paths call os.environ["PATHEXT"], _windowsLatestRocmBin,
# etc. on a non-Windows fs). See target.md.
################################################################################
import importlib
import os
import stat

import pytest

pytestmark = pytest.mark.unit

V = importlib.import_module("Tensile.Toolchain.Validators")


# ---------------------------------------------------------------------------
# supported* predicates (pure)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn,name,expected",
    [
        ("supportedCxxCompiler", "amdclang++", True),
        ("supportedCxxCompiler", "clang++", True),
        ("supportedCxxCompiler", "/opt/rocm/bin/amdclang++", True),
        ("supportedCxxCompiler", "amdclang", False),
        ("supportedCxxCompiler", "g++", False),
        ("supportedCCompiler", "amdclang", True),
        ("supportedCCompiler", "clang", True),
        ("supportedCCompiler", "/usr/bin/clang", True),
        ("supportedCCompiler", "amdclang++", False),
        ("supportedOffloadBundler", "clang-offload-bundler", True),
        ("supportedOffloadBundler", "clang", False),
        ("supportedHip", "hipcc", True),
        ("supportedHip", "hipconfig", True),
        ("supportedHip", "hipcc.exe", False),
    ],
)
def test_supported_predicates(fn, name, expected):
    assert getattr(V, fn)(name) is expected


def test_supported_device_enumerator_posix():
    # On posix only rocm_agent_enumerator / amdgpu-arch are accepted.
    assert V.supportedDeviceEnumerator("rocm_agent_enumerator") is True
    assert V.supportedDeviceEnumerator("amdgpu-arch") is True
    assert V.supportedDeviceEnumerator("/opt/rocm/bin/amdgpu-arch") is True
    assert V.supportedDeviceEnumerator("hipinfo") is False


def test_supported_component_matches_basename():
    # _supportedComponent matches both the raw string and Path(component).name
    assert V._supportedComponent("amdclang", ["amdclang"]) is True
    assert V._supportedComponent("/a/b/amdclang", ["amdclang"]) is True
    assert V._supportedComponent("amdclang", ["clang"]) is False


# ---------------------------------------------------------------------------
# _exeExists
# ---------------------------------------------------------------------------
def test_exe_exists_true_false(tmp_path):
    exe = tmp_path / "amdclang++"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    assert V._exeExists(exe) is True

    missing = tmp_path / "nope"
    assert V._exeExists(missing) is False


# ---------------------------------------------------------------------------
# _posixSearchPaths (env-driven, pure)
# ---------------------------------------------------------------------------
def test_posix_search_paths_defaults(monkeypatch):
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("PATH", raising=False)
    paths = V._posixSearchPaths()
    assert V.DEFAULT_ROCM_BIN_PATH_POSIX in paths
    assert V.DEFAULT_ROCM_LLVM_BIN_PATH_POSIX in paths


def test_posix_search_paths_with_rocm_and_path(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "/custom/rocm")
    monkeypatch.setenv("PATH", "/binA" + os.pathsep + "/binB")
    paths = V._posixSearchPaths()
    from pathlib import Path

    assert Path("/custom/rocm/bin") in paths
    assert Path("/custom/rocm/lib/llvm/bin") in paths
    assert Path("/binA") in paths
    assert Path("/binB") in paths
    # ROCM_PATH entries precede the defaults precede PATH entries
    assert paths.index(Path("/custom/rocm/bin")) < paths.index(
        V.DEFAULT_ROCM_BIN_PATH_POSIX
    )
    assert paths.index(V.DEFAULT_ROCM_BIN_PATH_POSIX) < paths.index(Path("/binA"))


# ---------------------------------------------------------------------------
# _validateExecutable
# ---------------------------------------------------------------------------
def _make_exe(d, name):
    p = d / name
    p.write_text("#!/bin/sh\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_validate_executable_absolute_ok(tmp_path):
    exe = _make_exe(tmp_path, "amdclang++")
    assert V._validateExecutable(str(exe), []) == str(exe)


def test_validate_executable_absolute_missing(tmp_path):
    missing = tmp_path / "amdclang++"  # supported name but not created
    with pytest.raises(FileNotFoundError):
        V._validateExecutable(str(missing), [])


def test_validate_executable_unsupported_name(tmp_path):
    exe = _make_exe(tmp_path, "g++")
    with pytest.raises(ValueError):
        V._validateExecutable(str(exe), [tmp_path])


def test_validate_executable_found_in_search_path(tmp_path):
    _make_exe(tmp_path, "amdclang")
    # relative name resolved against searchPaths
    assert V._validateExecutable("amdclang", [tmp_path]) == str(tmp_path / "amdclang")


def test_validate_executable_not_in_any_search_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        V._validateExecutable("amdclang", [tmp_path])


# ---------------------------------------------------------------------------
# validateToolchain (public entry)
# ---------------------------------------------------------------------------
def test_validate_toolchain_no_args():
    with pytest.raises(ValueError):
        V.validateToolchain()


def test_validate_toolchain_single_returns_scalar(tmp_path, monkeypatch):
    _make_exe(tmp_path, "amdclang++")
    monkeypatch.setattr(V, "_posixSearchPaths", lambda: [tmp_path])
    result = V.validateToolchain("amdclang++")
    assert result == str(tmp_path / "amdclang++")
    assert isinstance(result, str)


def test_validate_toolchain_multiple_returns_tuple(tmp_path, monkeypatch):
    _make_exe(tmp_path, "amdclang++")
    _make_exe(tmp_path, "amdclang")
    monkeypatch.setattr(V, "_posixSearchPaths", lambda: [tmp_path])
    result = V.validateToolchain("amdclang++", "amdclang")
    assert isinstance(result, tuple)
    assert result == (str(tmp_path / "amdclang++"), str(tmp_path / "amdclang"))


def test_validate_toolchain_propagates_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_posixSearchPaths", lambda: [tmp_path])
    with pytest.raises(FileNotFoundError):
        validate = V.validateToolchain("amdclang++")
        # generator is lazy for the single-arg path; force evaluation
        _ = validate


# ---------------------------------------------------------------------------
# ToolchainDefaults (NamedTuple class attributes resolved at import on posix)
# ---------------------------------------------------------------------------
def test_toolchain_defaults_posix():
    d = V.ToolchainDefaults
    assert d.CXX_COMPILER == "amdclang++"
    assert d.C_COMPILER == "amdclang"
    assert d.OFFLOAD_BUNDLER == "clang-offload-bundler"
    assert d.ASSEMBLER == "amdclang++"
    assert d.HIP_CONFIG == "hipconfig"
    assert d.DEVICE_ENUMERATOR in ("rocm_agent_enumerator", "amdgpu-arch")


def test_oss_select_posix():
    assert V.osSelect(linux="L", windows="W") == "L"


# ---------------------------------------------------------------------------
# Windows-only helpers — exercised directly where they do not gate on os.name.
# These pin the Windows path-selection logic even though the host is Linux.
# ---------------------------------------------------------------------------
def test_windows_latest_rocm_bin_picks_highest(tmp_path):
    (tmp_path / "5.7").mkdir()
    (tmp_path / "6.0").mkdir()
    (tmp_path / "6.10").mkdir()
    (tmp_path / "not-a-version").mkdir()
    (tmp_path / "5.9.txt").write_text("x")  # not a dir
    latest = V._windowsLatestRocmBin(tmp_path)
    assert latest == tmp_path / "6.10" / "bin"


def test_windows_latest_rocm_bin_none_when_empty(tmp_path):
    assert V._windowsLatestRocmBin(tmp_path) is None


def test_windows_search_paths_direct(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("HIP_PATH", "/hip/root")
    monkeypatch.setenv("PATH", "/winA" + os.pathsep + "/winB")
    paths = V._windowsSearchPaths()
    assert Path("/hip/root/bin") in paths
    assert Path("/winA") in paths
    assert Path("/winB") in paths
    # HIP_PATH precedes PATH entries
    assert paths.index(Path("/hip/root/bin")) < paths.index(Path("/winA"))


def test_windows_search_paths_appends_latest_rocm_bin(tmp_path, monkeypatch):
    from pathlib import Path

    (tmp_path / "6.2").mkdir()
    monkeypatch.setattr(V, "DEFAULT_ROCM_BIN_PATH_WINDOWS", tmp_path)
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.delenv("PATH", raising=False)
    paths = V._windowsSearchPaths()
    assert tmp_path / "6.2" / "bin" in paths


def test_windows_with_extensions_raises_on_posix():
    # os.name is 'posix' in the container -> guard raises.
    with pytest.raises(ValueError):
        V._windowsWithExtensions("amdclang++")


def test_windows_with_extensions_nt(monkeypatch):
    monkeypatch.setattr(V.os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".EXE;.BAT")
    files = V._windowsWithExtensions("amdclang++")
    assert files == ["amdclang++", "amdclang++.exe", "amdclang++.bat"]


@pytest.mark.nt_path_simulation
def test_supported_component_windows_branch(monkeypatch):
    monkeypatch.setattr(V.os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".EXE")
    # targets get extension-expanded; raw 'amdclang++' still matches
    assert V._supportedComponent("amdclang++", ["amdclang++"]) is True
    assert V._supportedComponent("amdclang++.exe", ["amdclang++"]) is True


@pytest.mark.nt_path_simulation
def test_supported_device_enumerator_windows(monkeypatch):
    monkeypatch.setattr(V.os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".EXE")
    assert V.supportedDeviceEnumerator("hipinfo") is True
    assert V.supportedDeviceEnumerator("hipInfo") is True
    assert V.supportedDeviceEnumerator("amdgpu-arch") is False
