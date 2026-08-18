# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from tensilelite import _rocm, _runtime


pytestmark = pytest.mark.unit


def _root(tmp_path: Path, version: str = "7.2.4") -> Path:
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text(version + "\n", encoding="utf-8")
    return root


def test_expected_rocm_version_from_local_tag():
    assert _rocm.expected_rocm_version("tensilelite", "5.0.0+rocm7.2.4") == "7.2.4"


@pytest.mark.parametrize("version", ["5.0.0", "5.0.0+cuda12.0.0"])
def test_expected_rocm_version_rejects_unmatched_distribution(version):
    with pytest.raises(_rocm.TensileLiteRuntimeError):
        _rocm.expected_rocm_version("tensilelite", version)


def test_validate_distribution_exact_match(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(_rocm, "resolve_rocm_root", lambda: root)

    result = _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")

    assert result.root == root
    assert result.version == "7.2.4"


def test_validate_distribution_reports_mismatch(tmp_path, monkeypatch):
    root = _root(tmp_path, "7.3.0")
    monkeypatch.setattr(_rocm, "resolve_rocm_root", lambda: root)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="expected ROCm: 7.2.4"):
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.2.4")


def test_resolve_rocm_root_prefers_environment(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setenv("ROCM_PATH", str(root))

    assert _rocm.resolve_rocm_root() == root.resolve()


def test_runtime_reports_external_rocisa_import_failure(monkeypatch):
    def fail_import(name):
        assert name == "rocisa"
        raise ImportError("dependency is unavailable")

    monkeypatch.setattr(_runtime, "import_module", fail_import)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="independently packaged"):
        _runtime.validate_runtime("5.0.0+rocm7.2.4")


def test_runtime_treats_rocisa_as_an_opaque_import(tmp_path, monkeypatch):
    root = _root(tmp_path)
    client = root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    client.parent.mkdir(parents=True)
    client.write_text("", encoding="utf-8")
    client.chmod(0o755)
    imports = []

    monkeypatch.setattr(_runtime, "import_module", lambda name: imports.append(name) or object())
    monkeypatch.setattr(
        _runtime,
        "validate_distribution",
        lambda distribution, version: _rocm.ValidatedRocm(root, "7.2.4"),
    )

    result = _runtime.validate_runtime("5.0.0+rocm7.2.4")

    assert imports == ["rocisa"]
    assert result.client == client
