# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
from pathlib import Path
import subprocess

from packaging.version import Version
import pytest

import _tensilelite_client_binding as binding
import tensilelite_configure_client as configure_client


pytestmark = pytest.mark.unit


class _Distribution:
    version = "5.0.0+rocm7.2.4"

    def __init__(self, package_dir: Path, *, editable: bool = False):
        self.package_dir = package_dir
        self.files = () if editable else (Path("tensilelite/__init__.py"),)
        self.editable = editable

    def locate_file(self, entry):
        return self.package_dir.parent / entry

    def read_text(self, name):
        if name == "direct_url.json" and self.editable:
            return json.dumps({"url": self.package_dir.parent.as_uri()})
        return None


@pytest.mark.parametrize("editable", [False, True])
def test_installation_key_uses_exact_resolved_package_directory(tmp_path, monkeypatch, editable):
    package_dir = (tmp_path / "environment" / "tensilelite").resolve()
    monkeypatch.setattr(binding, "_package_dir", lambda: package_dir)
    monkeypatch.setattr(
        binding.metadata, "distributions", lambda **unused: [_Distribution(package_dir, editable=editable)]
    )

    installation = binding.current_installation()

    assert installation.package_dir == package_dir
    assert installation.identifier == hashlib.sha256(os.fsencode(str(package_dir))).hexdigest()


def test_different_worktrees_have_different_installation_keys(tmp_path, monkeypatch):
    package_dirs = [(tmp_path / name / "tensilelite").resolve() for name in ("one", "two")]
    identifiers = []
    for package_dir in package_dirs:
        monkeypatch.setattr(binding, "_package_dir", lambda package_dir=package_dir: package_dir)
        monkeypatch.setattr(
            binding.metadata, "distributions", lambda **unused: [_Distribution(package_dir)]
        )
        identifiers.append(binding.current_installation().identifier)
    assert identifiers[0] != identifiers[1]


def test_binding_root_and_helper_cache_share_tensilelite_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    installation = binding.Installation(tmp_path / "package", "id", "1.0+rocm1.0.0")
    assert binding.binding_path(installation) == tmp_path / ".tensilelite/bindings/id/client.json"


def test_configure_atomically_replaces_and_reset_removes_one_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    installation = binding.Installation(tmp_path / "package", "id", "5.0.0+rocm7.2.4")
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    monkeypatch.setattr(configure_client, "current_installation", lambda: installation)
    monkeypatch.setattr(configure_client, "validate_client", lambda path, version: None)

    configure_client.configure(client)
    path = binding.binding_path(installation)
    assert json.loads(path.read_text(encoding="utf-8")) == str(client)
    assert list(path.parent.iterdir()) == [path]

    configure_client.reset()
    assert not path.exists()


def test_configured_binding_is_exclusive_even_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    installation = binding.Installation(tmp_path / "package", "id", "5.0.0+rocm7.2.4")
    configured = (tmp_path / "missing-client").absolute()
    path = binding.binding_path(installation)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(str(configured)), encoding="utf-8")
    def default_client():
        raise AssertionError("configured bindings must not resolve a default client")

    selected = binding.selected_client(default_client, installation)

    assert selected == binding.ClientCandidate(configured, "explicit TensileLite client binding")


def test_default_client_candidate_uses_first_existing_executable_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    installation = binding.Installation(tmp_path / "package", "id", "5.0.0+rocm7.2.4")
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    executable = "tensilelite-client.exe" if os.name == "nt" else "tensilelite-client"
    client = fallback / executable
    client.parent.mkdir()
    client.write_text("client", encoding="utf-8")

    def default_client():
        return binding.default_client_candidate((primary, fallback), "test prefix")

    selected = binding.selected_client(default_client, installation)
    assert selected == binding.ClientCandidate(client, "test prefix")


def _client_result(stdout="5.0.0+rocm7.2.4\n", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_client_result(stderr="loader warning"), "wrote to stderr"),
        (_client_result(returncode=3), "status 3"),
        (_client_result(returncode=-9), "signal 9"),
        (_client_result(stdout=""), "exactly one"),
        (_client_result(stdout="one\ntwo\n"), "exactly one"),
        (_client_result(stdout="not a version\n"), "not PEP 440"),
        (_client_result(stdout="5.0.1+rocm7.2.4\n"), "version mismatch"),
    ],
)
def test_client_identity_diagnostics(tmp_path, monkeypatch, result, message):
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    monkeypatch.setattr(binding.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(binding.ClientBindingError, match=message):
        binding.validate_client(client, "5.0.0+rocm7.2.4")


def test_client_identity_accepts_pep440_equivalent_version(tmp_path, monkeypatch):
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    monkeypatch.setattr(binding.subprocess, "run", lambda *args, **kwargs: _client_result())
    binding.validate_client(client, str(Version("5.0.0+rocm7.2.4")))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("missing loader"), "Could not launch"),
        (OSError("bad image"), "loader failed"),
        (subprocess.TimeoutExpired(["client", "--version"], 5), "timed out"),
    ],
)
def test_client_launch_diagnostics(tmp_path, monkeypatch, failure, message):
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(binding.subprocess, "run", fail)
    with pytest.raises(binding.ClientBindingError, match=message):
        binding.validate_client(client, "5.0.0+rocm7.2.4")
