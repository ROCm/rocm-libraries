# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

import pytest

import _tensilelite_client_binding as client_binding
import tensilelite
from tensilelite import _rocm, _runtime

pytestmark = pytest.mark.unit


def _root(tmp_path: Path, version: str = "7.2.4") -> Path:
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text(version + "\n", encoding="utf-8")
    return root


def _resolved(root: Path, source: str = "test") -> _rocm.ResolvedRocmRoot:
    return _rocm.ResolvedRocmRoot(root, source)


def _set_tensilelite_version(monkeypatch, version: str) -> None:
    monkeypatch.setattr(tensilelite, "__version__", version)


def test_expected_rocm_version_from_local_tag():
    assert _rocm.expected_rocm_version("tensilelite", "5.0.0+rocm7.2.4") == "7.2.4"


@pytest.mark.parametrize("version", ["5.0.0", "5.0.0+cuda12.0.0"])
def test_expected_rocm_version_rejects_unmatched_distribution(version):
    with pytest.raises(_rocm.TensileLiteRuntimeError):
        _rocm.expected_rocm_version("tensilelite", version)


def test_validate_distribution_uses_base_info_version_without_python_core(tmp_path, monkeypatch):
    root = _root(tmp_path, "10.1.0")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "resolve_rocm_root", lambda: _resolved(root))

    result = _rocm.validate_distribution(
        "tensilelite", "5.0.0+rocm10.1.0a20260813"
    )

    assert isinstance(result, _rocm.PrefixRocm)
    assert result.root == root
    assert result.version == "10.1.0"
    assert result.source == "test"
    assert result.toolchain_paths == (root / "bin", root / "lib" / "llvm" / "bin")


def test_validate_distribution_reports_mismatch(tmp_path, monkeypatch):
    root = _root(tmp_path, "7.3.0")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "resolve_rocm_root", lambda: _resolved(root, "active Python rocm_sdk"))

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="selected by: active Python rocm_sdk",
    ):
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")


def test_resolve_rocm_root_prefers_environment(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setenv("ROCM_PATH", str(root))

    result = _rocm.resolve_rocm_root()

    assert result.root == root.resolve()
    assert result.source == "explicit ROCM_PATH"


def test_resolve_rocm_root_uses_hipconfig_on_path(tmp_path, monkeypatch):
    root = _root(tmp_path)
    hipconfig = tmp_path / "hipconfig"
    hipconfig.write_text("#!/bin/sh\n", encoding="utf-8")
    hipconfig.chmod(0o755)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(_rocm, "_path_rocm_root", lambda: _resolved(root, "hipconfig on PATH"))
    monkeypatch.setattr(_rocm.Path, "is_dir", lambda path: False if path == _rocm.Path("/opt/rocm") else path.exists())

    result = _rocm.resolve_rocm_root()

    assert result.root == root.resolve()
    assert result.source == "hipconfig on PATH"


def test_path_rocm_root_uses_hipconfig_rocmpath(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(_rocm.shutil, "which", lambda name: "/usr/bin/hipconfig")
    monkeypatch.setattr(
        _rocm.subprocess,
        "run",
        lambda *args, **kwargs: _rocm.subprocess.CompletedProcess(
            args[0], 0, str(root) + "\n", ""
        ),
    )

    result = _rocm._path_rocm_root()

    assert result == _resolved(root, "hipconfig on PATH")


def test_python_sdk_version_uses_distribution_version(monkeypatch):
    core_version = "10.1.0a20260813"

    class Core:
        __version__ = core_version

    monkeypatch.setitem(sys.modules, "rocm_sdk_core", Core)

    assert _rocm._python_sdk_version() == core_version


def test_expected_rocm_version_parses_development_tag():
    assert (
        _rocm.expected_rocm_version(
            "tensilelite", "5.0.0+devrocm10.1.0.dev0.0123456789abcdef"
        )
        == "10.1.0.dev0.0123456789abcdef"
    )


def test_validate_distribution_uses_active_python_core_version(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(
        _rocm,
        "_python_sdk_version",
        lambda: "10.1.0a20260813",
    )
    monkeypatch.setattr(_rocm.sysconfig, "get_path", lambda name: str(scripts))
    monkeypatch.setattr(
        _rocm,
        "resolve_rocm_root",
        lambda: (_ for _ in ()).throw(AssertionError("prefix discovery was used")),
    )

    result = _rocm.validate_distribution(
        "tensilelite", "5.0.0+rocm10.1.0a20260813"
    )

    assert isinstance(result, _rocm.PythonSdkRocm)
    assert result.version == "10.1.0a20260813"
    assert result.source == "active Python rocm_sdk_core"
    assert result.toolchain_paths == (scripts.resolve(),)


def test_runtime_reports_external_rocisa_import_failure(monkeypatch):
    def fail_import(name):
        assert name == "rocisa"
        raise ImportError("dependency is unavailable")

    monkeypatch.setattr(_runtime, "import_module", fail_import)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="independently packaged"):
        _runtime.initialize()


def test_runtime_treats_rocisa_as_an_opaque_import(tmp_path, monkeypatch):
    root = _root(tmp_path)
    client = root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    imports = []
    client_requests = []

    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm7.2.4")
    monkeypatch.setattr(_runtime, "import_module", lambda name: imports.append(name) or object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.PrefixRocm(
            root, "7.2.4", "test", (root / "bin", root / "lib" / "llvm" / "bin")
        ),
    )
    monkeypatch.setattr(
        _runtime,
        "selected_client",
        lambda default_client: client_requests.append(default_client())
        or client_binding.ClientCandidate(client, "test client"),
    )
    monkeypatch.setattr(_runtime, "validate_client", lambda path, version: None)

    _runtime.initialize()

    assert imports == ["rocisa"]
    assert client_requests == []
    assert _runtime.toolchain_search_paths() == (root / "bin", root / "lib" / "llvm" / "bin")
    assert _runtime.client_executable() == client
    assert client_requests == [client_binding.ClientCandidate(client, "test")]


def test_cli_help_does_not_request_client(monkeypatch):
    from tensilelite import cli

    monkeypatch.setattr(
        _runtime,
        "client_executable",
        lambda: (_ for _ in ()).throw(AssertionError("client requested by --help")),
    )

    assert cli.main(["--help"]) == 0


def test_python_sdk_client_request_uses_sdk_client_trampoline(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm10.1.0a20260813")
    monkeypatch.setattr(_runtime, "import_module", lambda name: object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.PythonSdkRocm(
            "10.1.0a20260813", (scripts,)
        ),
    )
    monkeypatch.setattr(_runtime, "selected_client", lambda default_client: default_client())
    validated = []
    monkeypatch.setattr(
        _runtime,
        "validate_client",
        lambda path, version: validated.append((path, version)),
    )

    _runtime.initialize()

    executable = scripts / ("tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client")
    assert _runtime.client_executable() == executable
    assert validated == [(executable, "5.0.0+rocm10.1.0a20260813")]


def test_python_sdk_client_request_uses_explicit_binding_before_sdk_default(tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    configured = tmp_path / "configured-client"
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm10.1.0a20260813")
    monkeypatch.setattr(_runtime, "import_module", lambda name: object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.PythonSdkRocm("10.1.0a20260813", (scripts,)),
    )
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: configured)
    monkeypatch.setattr(
        _rocm.PythonSdkRocm,
        "default_client",
        lambda self: (_ for _ in ()).throw(AssertionError("SDK client was probed")),
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
    monkeypatch.setattr(_runtime, "import_module", lambda name: object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.PrefixRocm(
            root, "7.2.4", "test", (root / "bin", root / "lib" / "llvm" / "bin")
        ),
    )
    _runtime.initialize()


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX test executable")
def test_client_writer_path_request_rejects_missing_standard_client(tmp_path, monkeypatch):
    from tensilelite import ClientWriter

    root = _root(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _initialize_runtime_with_root(root, monkeypatch)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="Client path is not a regular file"):
        ClientWriter.getClientExecutablePath()


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX test executable")
def test_client_writer_path_request_accepts_valid_standard_client(tmp_path, monkeypatch):
    from tensilelite import ClientWriter

    root = _root(tmp_path)
    client = root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    client.parent.mkdir(parents=True)
    client.write_text("#!/bin/sh\nprintf '5.0.0+rocm7.2.4\\n'\n", encoding="utf-8")
    client.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _initialize_runtime_with_root(root, monkeypatch)

    assert ClientWriter.getClientExecutablePath() == str(client)
