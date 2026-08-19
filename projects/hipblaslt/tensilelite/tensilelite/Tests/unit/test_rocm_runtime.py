# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import builtins
import json
import os
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


_SHA = "0123456789abcdef0123456789abcdef01234567"
_OTHER = "89abcdef0123456789abcdef0123456789abcdef"


def _manifest(root: Path, value) -> None:
    path = root / "share" / "therock" / "therock_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rocm_package_version": value}), encoding="utf-8")


def _package(tmp_path: Path, version: str = "5.0.0+rocm7.2.4", revision: str = _SHA):
    return client_binding.Installation(tmp_path / "package", "id", version, revision, False)

@pytest.mark.parametrize(
    ("distribution_version", "expected"),
    [
        ("5.0.0+rocm7.2.4", "7.2.4"),
        ("5.0.0+devrocm10.1.0.dev0.0123456789abcdef", "10.1.0.dev0.0123456789abcdef"),
    ],
)
def test_expected_rocm_version_parses_valid_rocm_tags(distribution_version, expected):
    assert _rocm._expected_rocm_version("tensilelite", distribution_version) == expected
@pytest.mark.parametrize("version", ["5.0.0+cuda12.0.0", "5.0.0+gnothex"])
def test_expected_rocm_version_rejects_unmatched_distribution(version):
    with pytest.raises(_rocm.TensileLiteRuntimeError):
        _rocm._expected_rocm_version("tensilelite", version)


@pytest.mark.parametrize("version", ["5.0.0", f"5.0.0+g{_SHA}", f"5.0.0+g{_SHA}.dirty"])
def test_development_builds_make_no_rocm_claim(version):
    assert _rocm._expected_rocm_version("tensilelite", version) is None


@pytest.mark.parametrize(
    ("expected", "actual", "base_only", "matches"),
    [
        ("7.12.0", "7.12.0", False, True),
        ("7.12.0", "7.12.0rc1", False, False),
        ("7.12.0a20261110", "7.12.0a20261111", False, False),
        ("7.12.0+bkc.20261110", "7.12.0+bkc.20261111", False, False),
        # TheRock dev identities: the run's commit token is ignored, variants are not.
        (f"7.12.0.dev0+{_SHA}", f"7.12.0.dev0+{_OTHER}", False, True),
        (f"7.12.0.dev0+{_SHA}.asan", f"7.12.0.dev0+{_OTHER}.asan", False, True),
        (f"7.12.0.dev0+{_SHA}.asan", f"7.12.0.dev0+{_OTHER}", False, False),
        (f"7.12.0.dev0+{_SHA}", "7.12.0", False, False),
        (f"7.12.0.dev0+{_SHA}", "7.12.0.dev1+" + _SHA, False, False),
        # Base-only installations compare X.Y.Z only.
        ("7.12.0rc1", "7.12.0", True, True),
        (f"7.12.0.dev0+{_SHA}", "7.12.0", True, True),
        ("7.12.0", "7.13.0", True, False),
    ],
)
def test_identity_comparison_rules(expected, actual, base_only, matches):
    canonical = _rocm._canonical_rocm_version
    assert (
        _rocm.identities_match(canonical(expected), canonical(actual), base_only=base_only)
        is matches
    )

def _graph_path(tmp_path, monkeypatch):
    graph_bin = tmp_path / "graph" / "bin"
    llvm_bin = tmp_path / "graph" / "lib" / "llvm" / "bin"
    graph_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join((str(graph_bin), str(llvm_bin))))
    return graph_bin, llvm_bin


def _forbid_installed_adapters(monkeypatch):
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


def test_marker_selects_therock_ci_before_other_installations(tmp_path, monkeypatch):
    graph_bin, llvm_bin = _graph_path(tmp_path, monkeypatch)
    monkeypatch.setenv("TENSILELITE_ROCM_ADAPTER", "therock-ci")
    monkeypatch.setenv("THEROCK_PACKAGE_VERSION", "10.1.0.dev0+abcdef")
    _forbid_installed_adapters(monkeypatch)

    result = _rocm.validate_distribution(
        "tensilelite", "5.0.0+devrocm10.1.0.dev0.abcdef"
    )

    assert isinstance(result, _rocm.TheRockCIRocm)
    assert isinstance(result, _rocm.ValidatedRocm)
    assert result.path == graph_bin.resolve()
    assert result.version == "10.1.0.dev0.abcdef"
    assert result.identity_kind == "full"
    assert result.source == "TheRock CI build"
    assert result.executable_search_paths == (graph_bin.resolve(), llvm_bin.resolve())


@pytest.mark.parametrize("value", [None, "", "git"])
def test_marker_without_a_real_version_is_a_development_build(tmp_path, monkeypatch, value):
    _graph_path(tmp_path, monkeypatch)
    monkeypatch.setenv("TENSILELITE_ROCM_ADAPTER", "therock-ci")
    if value is None:
        monkeypatch.delenv("THEROCK_PACKAGE_VERSION", raising=False)
    else:
        monkeypatch.setenv("THEROCK_PACKAGE_VERSION", value)
    _forbid_installed_adapters(monkeypatch)

    result = _rocm.validate_distribution("tensilelite", f"5.0.0+g{_SHA}")

    assert isinstance(result, _rocm.TheRockCIRocm)
    assert result.version is None
    assert result.identity_kind is None


def test_marker_cross_checks_a_labelled_wheel_against_the_passed_version(tmp_path, monkeypatch):
    _graph_path(tmp_path, monkeypatch)
    monkeypatch.setenv("TENSILELITE_ROCM_ADAPTER", "therock-ci")
    monkeypatch.setenv("THEROCK_PACKAGE_VERSION", "7.12.0rc1")

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="release mismatch"):
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.12.0rc2")
    assert _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.12.0rc1").version == "7.12.0rc1"


@pytest.mark.parametrize("value", ["", "git", "7.12.0"])
def test_therock_version_alone_never_selects_the_build_adapter(tmp_path, monkeypatch, value):
    """An installed wheel in a shell that happens to set THEROCK_PACKAGE_VERSION."""
    root = _root(tmp_path, "7.12.0")
    monkeypatch.delenv("TENSILELITE_ROCM_ADAPTER", raising=False)
    monkeypatch.setenv("THEROCK_PACKAGE_VERSION", value)
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))

    result = _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.12.0")

    assert isinstance(result, _rocm.SystemRocm)


def test_therock_version_alone_leaves_the_python_sdk_selected(tmp_path, monkeypatch):
    monkeypatch.delenv("TENSILELITE_ROCM_ADAPTER", raising=False)
    monkeypatch.setenv("THEROCK_PACKAGE_VERSION", "7.12.0")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: "7.12.0")
    monkeypatch.setattr(_rocm, "_python_sdk_location", lambda: tmp_path)
    monkeypatch.setattr(_rocm, "_python_sdk_executable_search_paths", lambda: (tmp_path,))

    result = _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.12.0")

    assert isinstance(result, _rocm.PythonRocm)


@pytest.mark.parametrize("value", ["", "TheRock", "therock", "system"])
def test_unknown_adapter_marker_fails(monkeypatch, value):
    monkeypatch.setenv("TENSILELITE_ROCM_ADAPTER", value)

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="is not a TensileLite ROCm adapter"):
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm10.1.0")


@pytest.mark.parametrize(
    ("manifest", "info", "version", "kind"),
    [
        ("7.12.0", "7.12.0", "7.12.0", "full"),
        ("7.12.0rc1", "7.12.0", "7.12.0rc1", "full"),
        ("7.12.0a20261110", "7.12.0", "7.12.0a20261110", "full"),
        (f"7.12.0.dev0+{_SHA}.asan", "7.12.0", f"7.12.0.dev0.{_SHA}.asan", "full"),
        ("git", "7.12.0", "7.12.0", "base"),
        (7, "7.12.0", "7.12.0", "base"),
        (None, "7.12.0-43", "7.12.0", "base"),
    ],
)
def test_system_rocm_identity_prefers_the_therock_manifest(
    tmp_path, monkeypatch, manifest, info, version, kind
):
    root = _root(tmp_path, info)
    if manifest is not None:
        _manifest(root, manifest)
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))

    result = _rocm.validate_distribution("tensilelite", f"5.0.0+g{_SHA}")

    assert (result.version, result.identity_kind) == (version, kind)


def test_system_rocm_malformed_manifest_falls_back_to_info_version(tmp_path, monkeypatch):
    root = _root(tmp_path, "7.12.0")
    path = root / "share" / "therock" / "therock_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))

    result = _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.12.0rc1")

    assert (result.version, result.identity_kind) == ("7.12.0", "base")


def test_system_rocm_rejects_disagreeing_version_records(tmp_path, monkeypatch):
    root = _root(tmp_path, "7.11.0")
    _manifest(root, "7.12.0")
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))

    with pytest.raises(_rocm.TensileLiteRuntimeError, match="Inconsistent ROCm installation") as error:
        _rocm.validate_distribution("tensilelite", f"5.0.0+g{_SHA}")
    assert "therock_manifest.json: 7.12.0" in str(error.value)
    assert ".info/version: 7.11.0" in str(error.value)


@pytest.mark.parametrize(
    ("wheel", "manifest", "ok"),
    [
        ("5.0.0+rocm7.12.0", "7.12.0", True),
        ("5.0.0+rocm7.12.0", "7.12.0a20261110", False),  # release wheel on a nightly
        ("5.0.0+rocm7.12.0a20261110", "7.12.0", False),  # nightly wheel on a release
        (f"5.0.0+devrocm7.12.0.dev0.{_SHA}", f"7.12.0.dev0+{_OTHER}", True),  # reused stage
        (f"5.0.0+devrocm7.12.0.dev0.{_SHA}", "7.12.0", False),
        (f"5.0.0+g{_SHA}", "7.99.0", True),  # no claim
    ],
)
def test_system_rocm_compares_full_identities_on_therock_roots(
    tmp_path, monkeypatch, wheel, manifest, ok
):
    root = _root(tmp_path, manifest.split("a")[0].split(".dev")[0].split("+")[0])
    _manifest(root, manifest)
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))

    if ok:
        _rocm.validate_distribution("tensilelite", wheel)
    else:
        with pytest.raises(_rocm.TensileLiteRuntimeError, match="release mismatch"):
            _rocm.validate_distribution("tensilelite", wheel)


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
    assert result.executable_search_paths == (
        root / "bin",
        root / "lib" / "llvm" / "bin",
        root / "libexec" / "hipblaslt" / "tensilelite",
    )
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


def test_python_sdk_version_returns_none_when_core_package_is_absent(monkeypatch):
    original_import = builtins.__import__

    def import_without_core(name, *args, **kwargs):
        if name == "rocm_sdk_core":
            raise ModuleNotFoundError("No module named 'rocm_sdk_core'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rocm_sdk_core", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_core)

    assert _rocm._python_sdk_version() is None


def test_python_sdk_version_rejects_missing_core_dependency(monkeypatch):
    original_import = builtins.__import__

    def import_with_broken_core(name, *args, **kwargs):
        if name == "rocm_sdk_core":
            raise ModuleNotFoundError(
                "No module named 'rocm_sdk_dependency'", name="rocm_sdk_dependency"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rocm_sdk_core", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_with_broken_core)

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="active Python ROCm core package could not be imported",
    ):
        _rocm._python_sdk_version()

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
    package = _package(tmp_path)
    monkeypatch.setattr(_runtime, "current_installation", lambda: package)
    monkeypatch.setattr(
        _runtime,
        "default_client_candidate",
        lambda installation, paths, source: client_requests.append((paths, source))
        or client_binding.ClientCandidate(client, "test client"),
    )
    monkeypatch.setattr(
        _runtime,
        "selected_client",
        lambda default_client, installation: default_client(),
    )
    monkeypatch.setattr(
        _runtime,
        "validate_client",
        lambda path, installation: client_binding.ClientIdentity(_SHA, "7.2.4 base"),
    )

    _runtime.initialize()

    assert "rocisa" not in sys.modules
    assert client_requests == []
    assert _runtime.executable_search_paths() == [root / "bin", root / "lib" / "llvm" / "bin"]
    assert _runtime.client_executable() == client
    assert client_requests == [
        ((root / "bin", root / "lib" / "llvm" / "bin"), "test"),
    ]

def test_executable_search_paths_requires_explicit_initialization(monkeypatch):
    monkeypatch.setattr(_runtime, "_installation", None)
    monkeypatch.setattr(
        _runtime,
        "initialize",
        lambda: (_ for _ in ()).throw(AssertionError("implicit initialization attempted")),
    )

    with pytest.raises(
        _rocm.TensileLiteRuntimeError,
        match="TensileLite runtime has not been initialized",
    ):
        _runtime.executable_search_paths()
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
    monkeypatch.setattr(_runtime, "current_installation", lambda: _package(tmp_path))

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
    package = _package(tmp_path, "5.0.0+rocm10.1.0a20260813")
    monkeypatch.setattr(_runtime, "current_installation", lambda: package)
    validated = []
    monkeypatch.setattr(
        _runtime,
        "validate_client",
        lambda path, installation: validated.append((path, installation))
        or client_binding.ClientIdentity(_SHA, "10.1.0a20260813 full"),
    )

    _runtime.initialize()

    assert _runtime.client_executable() == configured
    assert validated == [(configured, package)]

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
    client.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        f"  --source-revision) echo {_SHA} ;;\n"
        "  --build-rocm) echo '7.2.4 base' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    client.chmod(0o755)
    package_dir = tmp_path / "package" / "tensilelite"
    package_dir.mkdir(parents=True)
    (package_dir / "_source_revision.py").write_text(
        Path(_rocm.__file__).with_name("_source_revision.py").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        _runtime,
        "current_installation",
        lambda: client_binding.Installation(package_dir, "id", "5.0.0+rocm7.2.4", _SHA, False),
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: client)
    _initialize_runtime_with_root(root, monkeypatch)

    assert ClientWriter.getClientExecutablePath() == str(client)
@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX test executable")
def test_system_rocm_client_request_uses_prefix_client(tmp_path, monkeypatch):
    root = _root(tmp_path)
    client = root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    client.parent.mkdir(parents=True)
    client.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        f"  --source-revision) echo {_SHA} ;;\n"
        "  --build-rocm) echo '7.2.4 base' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    client.chmod(0o755)
    package_dir = tmp_path / "package" / "tensilelite"
    package_dir.mkdir(parents=True)
    (package_dir / "_source_revision.py").write_text(
        Path(_rocm.__file__).with_name("_source_revision.py").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        _runtime,
        "current_installation",
        lambda: client_binding.Installation(package_dir, "id", "5.0.0+rocm7.2.4", _SHA, False),
    )

    monkeypatch.setattr(_runtime, "_client", None)
    monkeypatch.setattr(_runtime, "_installation", None)
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))
    monkeypatch.setattr(client_binding, "read_binding", lambda installation=None: None)
    _set_tensilelite_version(monkeypatch, "5.0.0+rocm7.2.4")

    _runtime.initialize()

    assert _runtime.client_executable() == client


@pytest.fixture(autouse=True)
def enable_runtime_validation(monkeypatch):
    monkeypatch.setattr(_rocm, "_not_checked_warned", False)


def _validated(version, kind="full"):
    return _rocm.SystemRocm(
        path=Path("/rocm"),
        version=version,
        source="test",
        executable_search_paths=(),
        identity_kind=kind,
    )


@pytest.fixture
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(_runtime, "_warned", set())


@pytest.mark.parametrize(
    ("build_rocm", "selected", "kind"),
    [
        ("7.12.0 full", "7.12.0", "full"),
        ("7.12.0rc1 full", "7.12.0rc1", "full"),
        ("7.12.0 base", "7.12.0rc1", "full"),  # base client: base to base
        ("7.12.0rc1 full", "7.12.0", "base"),  # base installation: base to base
        (f"7.12.0.dev0+{_SHA} full", f"7.12.0.dev0.{_OTHER}", "full"),
    ],
)
def test_client_build_rocm_matching_identity_is_silent(fresh_warnings, build_rocm, selected, kind):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _runtime.check_client_build_rocm(build_rocm, _validated(selected, kind), Path("/c"))


def test_client_build_rocm_minor_mismatch_warns_once(fresh_warnings):
    import warnings

    with pytest.warns(_runtime.ClientRocmWarning, match="different ROCm"):
        _runtime.check_client_build_rocm("7.11.0 full", _validated("7.12.0"), Path("/c"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _runtime.check_client_build_rocm("7.11.0 full", _validated("7.12.0"), Path("/c"))


@pytest.mark.parametrize(("build_rocm", "selected", "kind"), [
    ("6.4.0 full", "7.12.0", "full"),
    ("6.4.0 base", "7.12.0rc1", "full"),
    ("8.0.0 full", "7.2.4", "base"),
])
def test_client_build_rocm_major_mismatch_fails(fresh_warnings, build_rocm, selected, kind):
    with pytest.raises(_rocm.TensileLiteRuntimeError, match="different major ROCm version"):
        _runtime.check_client_build_rocm(build_rocm, _validated(selected, kind), Path("/c"))


@pytest.mark.parametrize(("build_rocm", "selected"), [
    ("unknown unknown", "7.12.0"),
    ("garbage", "7.12.0"),
    ("7.12.0 full", None),
])
def test_client_build_rocm_unknown_warns_once(fresh_warnings, build_rocm, selected):
    with pytest.warns(_runtime.ClientRocmWarning, match="could not be checked"):
        _runtime.check_client_build_rocm(build_rocm, _validated(selected), Path("/c"))


# ---------------------------------------------------------------------------
# Strict validation of release-labelled wheels (decisions 4 and 5)
# ---------------------------------------------------------------------------

_CHANNELS = {
    "release": ("5.0.0+rocm7.12.0", "7.12.0"),
    "rc": ("5.0.0+rocm7.12.0rc2", "7.12.0rc2"),
    "nightly": ("5.0.0+rocm7.12.0a20261021", "7.12.0a20261021"),
    "dev": (f"5.0.0+devrocm7.12.0.dev0.{_SHA}", f"7.12.0.dev0+{_SHA}"),
}


def _therock_root(tmp_path, monkeypatch, manifest):
    root = _root(tmp_path, "7.12.0")
    _manifest(root, manifest)
    monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
    monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))
    return root


@pytest.mark.parametrize("wheel_channel", sorted(_CHANNELS))
@pytest.mark.parametrize("installed_channel", sorted(_CHANNELS))
def test_labelled_wheels_match_only_their_own_channel(
    tmp_path, monkeypatch, wheel_channel, installed_channel
):
    _therock_root(tmp_path, monkeypatch, _CHANNELS[installed_channel][1])
    wheel = _CHANNELS[wheel_channel][0]

    if wheel_channel == installed_channel:
        _rocm.validate_distribution("tensilelite", wheel)
    else:
        with pytest.raises(_rocm.TensileLiteRuntimeError, match="release mismatch"):
            _rocm.validate_distribution("tensilelite", wheel)


@pytest.mark.parametrize(
    ("manifest", "wheel", "ok"),
    [
        # The manifest's '+' becomes '.' in the label.
        (f"7.12.0.dev0+{_SHA}", f"5.0.0+devrocm7.12.0.dev0.{_SHA}", True),
        # Stage reuse: the baseline run's manifest, this run's wheel.
        (f"7.12.0.dev0+{_SHA}", f"5.0.0+devrocm7.12.0.dev0.{_OTHER}", True),
        (f"7.12.0.dev0+{_SHA}.asan", f"5.0.0+devrocm7.12.0.dev0.{_OTHER}", False),
        (f"7.12.0.dev0+{_SHA}.asan", f"5.0.0+devrocm7.12.0.dev0.{_OTHER}.asan", True),
        ("7.12.0.dev0", "5.0.0+rocm7.12.0a20261021", False),
        ("7.12.0.dev0", "5.0.0+rocm7.12.0", False),
        # A commit-less custom dev identity still matches a commit-labelled one.
        ("7.12.0.dev0", f"5.0.0+devrocm7.12.0.dev0.{_SHA}", True),
    ],
)
def test_dev_identities_ignore_only_their_commit_token(tmp_path, monkeypatch, manifest, wheel, ok):
    _therock_root(tmp_path, monkeypatch, manifest)

    if ok:
        _rocm.validate_distribution("tensilelite", wheel)
    else:
        with pytest.raises(_rocm.TensileLiteRuntimeError, match="release mismatch"):
            _rocm.validate_distribution("tensilelite", wheel)


def test_development_wheel_warns_once_and_continues(tmp_path, monkeypatch):
    import warnings

    _therock_root(tmp_path, monkeypatch, "7.12.0rc9")

    with pytest.warns(_rocm.RocmNotCheckedWarning, match="not checked") as record:
        result = _rocm.validate_distribution("tensilelite", f"5.0.0+g{_SHA}")
    assert result.version == "7.12.0rc9"
    assert "7.12.0rc9" in str(record[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rocm.validate_distribution("tensilelite", f"5.0.0+g{_SHA}.dirty")


def test_labelled_wheel_does_not_warn(tmp_path, monkeypatch):
    import warnings

    _therock_root(tmp_path, monkeypatch, "7.12.0")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rocm.validate_distribution("tensilelite", "5.0.0+rocm7.12.0")


@pytest.mark.parametrize(
    "therock_version",
    [
        "7.12.0",
        "7.12.0rc2",
        "7.12.0a20261021",
        f"7.12.0.dev0+{_SHA}",
        "7.12.0a20261021+asan",
        "10.1.0a20260811+bkc.20260814",
        "7.12.0.dev0+sha.asan",
    ],
)
@pytest.mark.parametrize("installed_as", ["manifest", "python-sdk", "therock-ci"])
def test_stamped_label_round_trips_through_runtime_validation(
    tmp_path, monkeypatch, therock_version, installed_as
):
    """The label release_metadata stamps for a TheRock version passes the
    runtime check against the same version as the installed ROCm reports it."""
    import runpy

    metadata = runpy.run_path(str(Path(tensilelite.__file__).parents[1] / "release_metadata.py"))
    label = metadata["distribution_version"](_SHA, therock_version)

    if installed_as == "manifest":
        root = _root(tmp_path, therock_version.split("a")[0].split("rc")[0].split(".dev")[0].split("+")[0])
        _manifest(root, therock_version)
        monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: None)
        monkeypatch.setattr(_rocm, "_resolve_system_rocm", lambda: _system_rocm(root))
    elif installed_as == "python-sdk":
        monkeypatch.setattr(_rocm, "_python_sdk_version", lambda: therock_version)
        monkeypatch.setattr(_rocm, "_python_sdk_location", lambda: tmp_path)
        monkeypatch.setattr(_rocm, "_python_sdk_executable_search_paths", lambda: (tmp_path,))
    else:
        _graph_path(tmp_path, monkeypatch)
        monkeypatch.setenv("TENSILELITE_ROCM_ADAPTER", "therock-ci")
        monkeypatch.setenv("THEROCK_PACKAGE_VERSION", therock_version)

    result = _rocm.validate_distribution("tensilelite", label)

    assert result.version == _rocm._canonical_rocm_version(therock_version)


@pytest.mark.parametrize(("build_rocm", "selected", "kind"), [
    ("7.11.0 full", "7.12.0", "full"),
    ("7.12.0rc1 full", "7.12.0", "full"),
    ("7.11.0 base", "7.12.0rc1", "full"),
])
def test_client_build_rocm_mismatch_fails_for_labelled_packages(
    fresh_warnings, build_rocm, selected, kind
):
    with pytest.raises(_rocm.TensileLiteRuntimeError, match="released for a specific ROCm"):
        _runtime.check_client_build_rocm(
            build_rocm, _validated(selected, kind), Path("/c"), strict=True
        )


def test_client_build_rocm_major_mismatch_message_wins_when_strict(fresh_warnings):
    with pytest.raises(_rocm.TensileLiteRuntimeError, match="different major ROCm version"):
        _runtime.check_client_build_rocm("6.4.0 full", _validated("7.12.0"), Path("/c"), strict=True)


def test_client_build_rocm_unknown_still_warns_for_labelled_packages(fresh_warnings):
    with pytest.warns(_runtime.ClientRocmWarning, match="could not be checked"):
        _runtime.check_client_build_rocm(
            "unknown unknown", _validated("7.12.0"), Path("/c"), strict=True
        )
