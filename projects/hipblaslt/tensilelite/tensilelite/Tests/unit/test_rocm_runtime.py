# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import pytest

import _tensilelite_client_binding as client_binding
import tensilelite
from tensilelite import _rocm, _runtime

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def enable_runtime_validation(monkeypatch):
    monkeypatch.setattr(_rocm, "_ENABLE_ROCM_VERSION_VALIDATION", True)
    monkeypatch.setattr(_rocm, "_ENABLE_PYTHON_ROCM_RUNTIME", True)


def _root(tmp_path: Path, version: str = "7.2.4") -> Path:
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text(version + "\n", encoding="utf-8")
    return root


def _system_rocm(root: Path, source: str = "test") -> _rocm.SystemRocmRoot:
    return _rocm.SystemRocmRoot(root, source)


def _python_core(path: Path, version: str):
    return type(
        "Core",
        (),
        {"__file__": str(path / "__init__.py"), "__version__": version},
    )


def _set_tensilelite_version(monkeypatch, version: str) -> None:
    monkeypatch.setattr(tensilelite, "__version__", version)


@pytest.mark.parametrize(
    ("distribution_version", "expected"),
    [
        ("5.0.0+rocm7.2.4", "7.2.4"),
        ("5.0.0+devrocm10.1.0.dev0.0123456789abcdef", "10.1.0.dev0.0123456789abcdef"),
    ],
)
def test_expected_rocm_version_parses_valid_rocm_tags(distribution_version, expected):
    assert _rocm._expected_rocm_version("tensilelite", distribution_version) == expected


@pytest.mark.parametrize("version", ["5.0.0", "5.0.0+cuda12.0.0"])
def test_expected_rocm_version_rejects_unmatched_distribution(version):
    with pytest.raises(_rocm.TensileLiteRuntimeError):
        _rocm._expected_rocm_version("tensilelite", version)

def test_validate_distribution_uses_therock_ci_before_other_installations(
    tmp_path, monkeypatch
):
    graph_bin = tmp_path / "graph" / "bin"
    llvm_bin = tmp_path / "graph" / "lib" / "llvm" / "bin"
    graph_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)
    monkeypatch.setenv("THEROCK_PACKAGE_VERSION", "10.1.0.dev0+abcdef")
    monkeypatch.setenv("PATH", os.pathsep.join((str(graph_bin), str(llvm_bin))))
    monkeypatch.setattr(
        _rocm,
        "_python_sdk_version",
        lambda: (_ for _ in ()).throw(AssertionError("Python SDK discovery was used")),
    )
    monkeypatch.setattr(
        _rocm,
        "_resolve_system_rocm",
        lambda: (_ for _ in ()).throw(AssertionError("prefix discovery was used")),
    )

    result = _rocm.validate_distribution(
        "tensilelite", "5.0.0+devrocm10.1.0.dev0.abcdef"
    )

    assert isinstance(result, _rocm.TheRockCIRocm)
    assert isinstance(result, _rocm.ValidatedRocm)
    assert result.path == graph_bin.resolve()
    assert result.version == "10.1.0.dev0.abcdef"
    assert result.source == "TheRock CI build"
    assert result.executable_search_paths == (graph_bin.resolve(), llvm_bin.resolve())

@pytest.mark.parametrize("value", ["", "git"])
def test_therock_ci_rejects_non_release_package_identity(monkeypatch, value):
    monkeypatch.setenv("THEROCK_PACKAGE_VERSION", value)

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="TheRock CI requires a release THEROCK_PACKAGE_VERSION",
    ):
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm10.1.0")
def test_validate_distribution_uses_base_info_version_without_python_core(tmp_path, monkeypatch):
    root = _root(tmp_path, "10.1.0")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))

    result = _rocm.validate_distribution(
        "tensilelite", "5.0.0+rocm10.1.0a20260813"
    )

    assert isinstance(result, _rocm.SystemRocm)
    assert isinstance(result, _rocm.ValidatedRocm)
    assert result.path == root
    assert result.version == "10.1.0"
    assert result.source == "test"
    assert result.executable_search_paths == (root / "bin", root / "lib" / "llvm" / "bin")


def test_validate_distribution_reports_mismatch(tmp_path, monkeypatch):
    root = _root(tmp_path, "7.3.0")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root, "active Python rocm_sdk"))
    monkeypatch.setattr(
        _rocm,
        "SystemRocm",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("invalid system ROCm was created")),
    )

    with pytest.raises(_rocm.TensileLiteRuntimeError) as exc_info:
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")

    assert str(exc_info.value) == (
        "tensilelite and ROCm release mismatch.\n"
        "  tensilelite version: 5.0.0+rocm7.2.4\n"
        "  expected ROCm: 7.2.4\n"
        "  found ROCm: 7.3.0\n"
        f"  selected: {root}\n"
        "  selected by: active Python rocm_sdk\n"
        "Install the wheel from the matching ROCm wheel index or select the matching ROCM_PATH."
    )


def test_validate_distribution_reports_python_sdk_mismatch(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    user_scripts = tmp_path / "user" / "bin"
    core_path = tmp_path / "venv" / "site-packages" / "rocm_sdk_core"
    monkeypatch.setitem(sys.modules, "rocm_sdk_core", _python_core(core_path, "7.3.0"))
    monkeypatch.setattr(
        _rocm.sysconfig,
        "get_path",
        lambda name, scheme=None: str(scripts if scheme is None else user_scripts),
    )
    monkeypatch.setattr(
        _rocm,
        "_resolve_system_rocm",
        lambda: (_ for _ in ()).throw(AssertionError("prefix discovery was used")),
    )
    monkeypatch.setattr(
        _rocm,
        "PythonRocm",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("invalid Python ROCm was created")),
    )

    with pytest.raises(_rocm.TensileLiteRuntimeError) as exc_info:
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")

    assert str(exc_info.value) == (
        "tensilelite and ROCm release mismatch.\n"
        "  tensilelite version: 5.0.0+rocm7.2.4\n"
        "  expected ROCm: 7.2.4\n"
        "  found ROCm: 7.3.0\n"
        f"  selected: {core_path.resolve()}\n"
        "  selected by: active Python rocm_sdk_core\n"
        "Install the wheel from the matching ROCm wheel index or select the matching ROCM_PATH."
    )


def test_resolve_system_rocm_prefers_environment(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setenv("ROCM_PATH", str(root))

    result = _rocm._resolve_system_rocm()

    assert result.root == root.resolve()
    assert result.source == "explicit ROCM_PATH"


def test_resolve_system_rocm_rejects_empty_environment(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "")

    with pytest.raises(_rocm.TensileLiteRuntimeError) as exc_info:
        _rocm._resolve_system_rocm()

    assert str(exc_info.value) == (
        "ROCM_PATH is set but empty.\n"
        "Set ROCM_PATH to the matching conventional ROCm installation."
    )


def test_resolve_system_rocm_falls_back_to_path_discovery(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(_rocm, "_path_system_rocm", lambda: _system_rocm(root, "hipconfig on PATH"))
    monkeypatch.setattr(_rocm.Path, "is_dir", lambda path: False if path == _rocm.Path("/opt/rocm") else path.exists())

    result = _rocm._resolve_system_rocm()

    assert result.root == root.resolve()
    assert result.source == "hipconfig on PATH"


def test_path_system_rocm_uses_hipconfig_rocmpath(tmp_path, monkeypatch):
    root = _root(tmp_path)
    which_requests = []
    command_requests = []
    monkeypatch.setattr(
        _rocm.shutil,
        "which",
        lambda name: which_requests.append(name) or "/usr/bin/hipconfig",
    )
    monkeypatch.setattr(
        _rocm.subprocess,
        "run",
        lambda *args, **kwargs: command_requests.append((args, kwargs))
        or _rocm.subprocess.CompletedProcess(args[0], 0, str(root) + "\n", ""),
    )

    result = _rocm._path_system_rocm()

    assert result == _system_rocm(root, "hipconfig on PATH")
    assert which_requests == ["hipconfig"]
    assert command_requests == [
        (
            (["/usr/bin/hipconfig", "--rocmpath"],),
            {"check": True, "capture_output": True, "text": True, "timeout": 5},
        )
    ]


def test_validate_distribution_uses_active_python_core_version(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    user_scripts = tmp_path / "user" / "bin"
    core_path = tmp_path / "venv" / "site-packages" / "rocm_sdk_core"
    scripts.mkdir(parents=True)
    user_scripts.mkdir(parents=True)

    monkeypatch.setitem(
        sys.modules,
        "rocm_sdk_core",
        _python_core(core_path, "10.1.0a20260813"),
    )
    monkeypatch.setattr(
        _rocm.sysconfig,
        "get_path",
        lambda name, scheme=None: str(scripts if scheme is None else user_scripts),
    )
    monkeypatch.setattr(
        _rocm,
        "_resolve_system_rocm",
        lambda: (_ for _ in ()).throw(AssertionError("prefix discovery was used")),
    )

    result = _rocm.validate_distribution(
        "tensilelite", "5.0.0+rocm10.1.0a20260813"
    )

    assert isinstance(result, _rocm.PythonRocm)
    assert isinstance(result, _rocm.ValidatedRocm)
    assert result.path == core_path.resolve()
    assert result.version == "10.1.0a20260813"
    assert result.source == "active Python rocm_sdk_core"
    assert result.executable_search_paths == (scripts.resolve(), user_scripts.resolve())


@pytest.mark.parametrize(
    "user_scheme",
    ["posix_user", "nt_user"],
)
def test_python_sdk_executable_search_paths_include_platform_user_scripts(
    tmp_path, monkeypatch, user_scheme
):
    scripts = tmp_path / "venv" / "bin"
    user_scripts = tmp_path / "user" / "bin"
    calls = []

    def get_path(name, scheme=None):
        calls.append((name, scheme))
        return str(scripts if scheme is None else user_scripts)

    monkeypatch.setattr(
        _rocm.sysconfig,
        "get_preferred_scheme",
        lambda purpose: user_scheme,
    )
    monkeypatch.setattr(_rocm.sysconfig, "get_path", get_path)

    assert _rocm._python_sdk_executable_search_paths() == (
        scripts.resolve(),
        user_scripts.resolve(),
    )
    assert calls == [("scripts", None), ("scripts", user_scheme)]

@pytest.mark.parametrize("primary_exists", [False, True])
def test_python_sdk_executable_search_prefers_primary_script_over_user_script(
    tmp_path, monkeypatch, primary_exists
):
    from tensilelite.Toolchain.Validators import validateToolchain

    scripts = tmp_path / "venv" / "bin"
    user_scripts = tmp_path / "user" / "bin"
    primary = scripts / "amdclang"
    user = user_scripts / "amdclang"
    for candidate in (primary, user):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("#!/bin/sh\n", encoding="utf-8")
        candidate.chmod(0o755)
    if not primary_exists:
        primary.unlink()

    monkeypatch.setattr(
        _runtime,
        "_installation",
        _rocm.PythonRocm(
            version="10.1.0a20260813",
            path=scripts,
            executable_search_paths=(scripts, user_scripts),
        ),
    )

    assert validateToolchain("amdclang") == str(primary if primary_exists else user)
def test_runtime_initialization_does_not_import_rocisa(tmp_path, monkeypatch):
    root = _root(tmp_path)
    client = root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    client_requests = []

    monkeypatch.delitem(sys.modules, "rocisa", raising=False)
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm7.2.4")
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.SystemRocm(
            path=root,
            version="7.2.4",
            source="test",
            executable_search_paths=(root / "bin", root / "lib" / "llvm" / "bin"),
        ),
    )
    monkeypatch.setattr(
        _runtime,
        "default_client_candidate",
        lambda paths, source: client_requests.append((paths, source))
        or client_binding.ClientCandidate(client, "test client"),
    )
    monkeypatch.setattr(
        _runtime,
        "selected_client",
        lambda default_client: default_client(),
    )
    monkeypatch.setattr(_runtime, "validate_client", lambda path, version: None)

    _runtime.initialize()

    assert "rocisa" not in sys.modules
    assert client_requests == []
    assert _runtime.executable_search_paths() == [root / "bin", root / "lib" / "llvm" / "bin"]
    assert _runtime.client_executable() == client
    assert client_requests == [
        ((root / "bin", root / "lib" / "llvm" / "bin"), "test"),
    ]


def test_cli_help_does_not_request_client(monkeypatch):
    from tensilelite import cli

    monkeypatch.setattr(
        _runtime,
        "client_executable",
        lambda: (_ for _ in ()).throw(AssertionError("client requested by --help")),
    )

    assert cli.main(["--help"]) == 0


def test_python_sdk_client_request_fails_when_no_client_script_is_published(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm10.1.0a20260813")
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.PythonRocm(
            version="10.1.0a20260813",
            path=scripts,
            executable_search_paths=(scripts,),
        ),
    )
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: None)

    _runtime.initialize()

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="tensilelite-client was not found in the selected executable paths",
    ):
        _runtime.client_executable()


def test_python_sdk_client_request_uses_explicit_binding_before_sdk_default(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    configured = tmp_path / "configured-client"
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm10.1.0a20260813")
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.PythonRocm(
            version="10.1.0a20260813",
            path=scripts,
            executable_search_paths=(scripts,),
        ),
    )
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: configured)
    monkeypatch.setattr(
        _runtime,
        "default_client_candidate",
        lambda *args: (_ for _ in ()).throw(AssertionError("default client was probed")),
    )
    validated = []
    monkeypatch.setattr(
        _runtime,
        "validate_client",
        lambda path, version: validated.append((path, version)),
    )

    _runtime.initialize()

    assert _runtime.client_executable() == configured
    assert validated == [(configured, "5.0.0+rocm10.1.0a20260813")]

def _initialize_runtime_with_root(root: Path, monkeypatch) -> None:
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm7.2.4")
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.SystemRocm(
            path=root,
            version="7.2.4",
            source="test",
            executable_search_paths=(root / "bin", root / "lib" / "llvm" / "bin"),
        ),
    )
    _runtime.initialize()

@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX test executable")
def test_client_writer_path_request_rejects_missing_explicit_client(tmp_path, monkeypatch):
    from tensilelite import ClientWriter

    root = _root(tmp_path)
    configured = (tmp_path / "missing-client").absolute()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: configured)
    _initialize_runtime_with_root(root, monkeypatch)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="Client path is not a regular file"):
        ClientWriter.getClientExecutablePath()

@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX test executable")
def test_client_writer_path_request_accepts_valid_explicit_client(tmp_path, monkeypatch):
    from tensilelite import ClientWriter

    root = _root(tmp_path)
    client = (tmp_path / "configured-client").absolute()
    client.parent.mkdir(parents=True, exist_ok=True)
    client.write_text("#!/bin/sh\nprintf '5.0.0+rocm7.2.4\\n'\n", encoding="utf-8")
    client.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: client)
    _initialize_runtime_with_root(root, monkeypatch)

    assert ClientWriter.getClientExecutablePath() == str(client)
