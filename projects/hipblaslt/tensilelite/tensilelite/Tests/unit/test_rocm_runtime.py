# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess
import sys

import pytest

from tensilelite import _rocm, _runtime


pytestmark = pytest.mark.unit


def _root(tmp_path: Path, version: str = "7.2.4") -> Path:
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text(version + "\n", encoding="utf-8")
    return root


def _resolved(root: Path, source: str = "test") -> _rocm.ResolvedRocmRoot:
    return _rocm.ResolvedRocmRoot(root, source)


def test_expected_rocm_version_from_local_tag():
    assert _rocm.expected_rocm_version("tensilelite", "5.0.0+rocm7.2.4") == "7.2.4"


@pytest.mark.parametrize("version", ["5.0.0", "5.0.0+cuda12.0.0"])
def test_expected_rocm_version_rejects_unmatched_distribution(version):
    with pytest.raises(_rocm.TensileLiteRuntimeError):
        _rocm.expected_rocm_version("tensilelite", version)


def test_validate_distribution_exact_match(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(_rocm, "resolve_rocm_root", lambda: _resolved(root))

    result = _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")

    assert result.root == root
    assert result.version == "7.2.4"
    assert result.source == "test"


def test_validate_distribution_reports_mismatch(tmp_path, monkeypatch):
    root = _root(tmp_path, "7.3.0")
    monkeypatch.setattr(_rocm, "resolve_rocm_root", lambda: _resolved(root, "active Python rocm_sdk"))

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="selected by: active Python rocm_sdk",
    ):
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")


def test_resolve_rocm_root_prefers_environment(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(_rocm, "find_spec", lambda name: None)
    monkeypatch.setenv("ROCM_PATH", str(root))

    result = _rocm.resolve_rocm_root()

    assert result.root == root.resolve()
    assert result.source == "explicit ROCM_PATH"


def test_resolve_rocm_root_prefers_active_python_sdk(tmp_path, monkeypatch):
    sdk_root = _root(tmp_path / "sdk")
    fallback_root = _root(tmp_path / "fallback")
    commands = []

    monkeypatch.setattr(_rocm, "find_spec", lambda name: object())
    monkeypatch.setenv("ROCM_PATH", str(fallback_root))

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=str(sdk_root), stderr="")

    monkeypatch.setattr(_rocm.subprocess, "run", run)

    result = _rocm.resolve_rocm_root()

    assert result.root == sdk_root.resolve()
    assert result.source == "active Python rocm_sdk"
    assert commands == [
        (
            [_rocm.sys.executable, "-m", "rocm_sdk", "path", "--root"],
            {"check": True, "capture_output": True, "text": True, "timeout": 10},
        )
    ]


def test_resolve_rocm_root_does_not_fall_back_from_broken_python_sdk(tmp_path, monkeypatch):
    fallback_root = _root(tmp_path / "fallback")
    monkeypatch.setattr(_rocm, "find_spec", lambda name: object())
    monkeypatch.setenv("ROCM_PATH", str(fallback_root))
    monkeypatch.setattr(
        _rocm.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, args[0], stderr="missing rocm[devel]")
        ),
    )

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="selected by: active Python rocm_sdk",
    ):
        _rocm.resolve_rocm_root()


def test_runtime_reports_external_rocisa_import_failure(monkeypatch):
    def fail_import(name):
        assert name == "rocisa"
        raise ImportError("dependency is unavailable")

    monkeypatch.setattr(_runtime, "import_module", fail_import)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="independently packaged"):
        _runtime.initialize("5.0.0+rocm7.2.4")


def test_runtime_treats_rocisa_as_an_opaque_import(tmp_path, monkeypatch):
    root = _root(tmp_path)
    client = root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    imports = []
    client_requests = []

    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_root", None)
    monkeypatch.setattr(_runtime, "_root_source", None)
    monkeypatch.setattr(_runtime, "_distribution_version", None)
    monkeypatch.setattr(_runtime, "import_module", lambda name: imports.append(name) or object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.ValidatedRocm(root, "7.2.4", "test"),
    )
    monkeypatch.setattr(
        _runtime,
        "selected_client",
        lambda root: client_requests.append(root) or (client, False),
    )
    monkeypatch.setattr(_runtime, "validate_client", lambda path, version: None)

    _runtime.initialize("5.0.0+rocm7.2.4")

    assert imports == ["rocisa"]
    assert client_requests == []
    assert _runtime.client_executable() == client
    assert client_requests == [root]
    assert _runtime.rocm_root() == root


def test_cli_help_does_not_request_client(monkeypatch):
    from tensilelite import cli

    monkeypatch.setattr(
        _runtime,
        "client_executable",
        lambda: (_ for _ in ()).throw(AssertionError("client requested by --help")),
    )

    assert cli.main(["--help"]) == 0


def _initialize_runtime_with_root(root: Path, monkeypatch) -> None:
    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_root", None)
    monkeypatch.setattr(_runtime, "_root_source", None)
    monkeypatch.setattr(_runtime, "_distribution_version", None)
    monkeypatch.setattr(_runtime, "import_module", lambda name: object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.ValidatedRocm(root, "7.2.4", "test"),
    )
    _runtime.initialize("5.0.0+rocm7.2.4")


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
