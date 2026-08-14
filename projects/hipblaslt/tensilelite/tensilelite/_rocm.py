# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Resolve the ROCm installation required by the TensileLite package."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TypeAlias

from _tensilelite_client_binding import ClientCandidate, standard_client_path


class TensileLiteRuntimeError(ImportError):
    """The installed TensileLite wheel and ROCm runtime do not match."""


@dataclass(frozen=True)
class PythonRocm:
    version: str
    toolchain_paths: tuple[Path, ...]
    source: str = "active Python rocm_sdk_core"

    def default_client(self) -> ClientCandidate:
        executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
        return ClientCandidate(
            self.toolchain_paths[0] / executable,
            "active Python rocm_sdk_libraries",
        )


@dataclass(frozen=True)
class SystemRocm:
    root: Path
    version: str
    source: str
    toolchain_paths: tuple[Path, ...]

    def default_client(self) -> ClientCandidate:
        return ClientCandidate(standard_client_path(self.root), self.source)


ValidatedRocm: TypeAlias = PythonRocm | SystemRocm


@dataclass(frozen=True)
class SystemRocmRoot:
    root: Path
    source: str


_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$", re.IGNORECASE)


def canonical_rocm_version(value: str) -> str:
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _RELEASE_RE.fullmatch(value):
        raise TensileLiteRuntimeError(f"Invalid ROCm release value: {value!r}")
    return value


def expected_rocm_version(distribution: str, distribution_version: str | None = None) -> str:
    if distribution_version is None:
        try:
            distribution_version = package_version(distribution)
        except PackageNotFoundError as exc:
            raise TensileLiteRuntimeError(
                f"{distribution} must be installed as a ROCm-versioned wheel; "
                "direct source-tree imports are unsupported."
            ) from exc
    try:
        local = distribution_version.split("+", 1)[1]
    except IndexError as exc:
        raise TensileLiteRuntimeError(
            f"{distribution} {distribution_version!r} has no '+rocmX.Y.Z' release tag."
        ) from exc
    if local.startswith("devrocm"):
        return canonical_rocm_version(local[len("devrocm") :])
    if not local.startswith("rocm"):
        raise TensileLiteRuntimeError(
            f"{distribution} {distribution_version!r} has an invalid ROCm release tag."
        )
    return canonical_rocm_version(local[len("rocm") :])


def rocm_base_version(value: str) -> str:
    match = re.match(r"^([0-9]+(?:\.[0-9]+){2})", value)
    if match is None:
        raise TensileLiteRuntimeError(f"Invalid ROCm release value: {value!r}")
    return match.group(1)


def _validated_system_rocm_root(root: Path, source: str) -> SystemRocmRoot:
    if not root.is_dir():
        raise TensileLiteRuntimeError(
            "ROCm installation not found.\n"
            f"  selected root: {root}\n"
            f"  selected by: {source}"
        )
    return SystemRocmRoot(root.resolve(), source)


def _python_sdk_version() -> str | None:
    try:
        import rocm_sdk_core
    except ModuleNotFoundError:
        return None

    try:
        return rocm_sdk_core.__version__
    except Exception as exc:
        raise TensileLiteRuntimeError(
            "The active Python ROCm core package could not resolve its publication identity.\n"
            "  selected by: active Python rocm_sdk_core\n"
            "Install the matching rocm core package."
        ) from exc


def _python_sdk_toolchain_paths() -> tuple[Path, ...]:
    scripts = sysconfig.get_path("scripts")
    if not scripts:
        raise TensileLiteRuntimeError(
            "The active Python ROCm SDK has no scripts directory for its tool trampolines.\n"
            "  selected by: active Python rocm_sdk_core"
        )
    return (Path(scripts).resolve(),)


def _path_system_rocm() -> SystemRocmRoot | None:
    hipconfig = shutil.which("hipconfig")
    if hipconfig is None:
        return None
    try:
        result = subprocess.run(
            [hipconfig, "--rocmpath"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    root = result.stdout.strip()
    return _validated_system_rocm_root(Path(root), "hipconfig on PATH") if root else None


def resolve_system_rocm() -> SystemRocmRoot:
    explicit = os.environ.get("ROCM_PATH")
    if explicit:
        return _validated_system_rocm_root(Path(explicit).expanduser(), "explicit ROCM_PATH")
    if sys.platform != "win32" and Path("/opt/rocm").is_dir():
        return _validated_system_rocm_root(Path("/opt/rocm"), "/opt/rocm")
    path_root = _path_system_rocm()
    if path_root is not None:
        return path_root
    raise TensileLiteRuntimeError(
        "ROCm installation not found.\n"
        "  selected by: no explicit ROCM_PATH, /opt/rocm, or hipconfig on PATH\n"
        "Set ROCM_PATH to the matching conventional ROCm installation."
    )


def _validate_version(
    distribution: str,
    distribution_version: str | None,
    expected_for_comparison: str,
    validated: ValidatedRocm,
) -> None:
    if validated.version != expected_for_comparison:
        shown_version = distribution_version or package_version(distribution)
        selected = (
            f"  selected root: {validated.root}\n"
            if isinstance(validated, SystemRocm)
            else ""
        )
        raise TensileLiteRuntimeError(
            f"{distribution} and ROCm release mismatch.\n"
            f"  {distribution} version: {shown_version}\n"
            f"  expected ROCm: {expected_for_comparison}\n"
            f"  found ROCm: {validated.version}\n"
            f"{selected}"
            f"  selected by: {validated.source}\n"
            "Install the wheel from the matching ROCm wheel index or select the matching ROCM_PATH."
        )


def _validate_python_sdk(
    distribution: str,
    distribution_version: str | None,
    expected: str,
    python_sdk_version: str,
) -> PythonRocm:
    validated = PythonRocm(
        canonical_rocm_version(python_sdk_version),
        _python_sdk_toolchain_paths(),
    )
    _validate_version(distribution, distribution_version, expected, validated)
    return validated


def _validate_system_rocm(
    distribution: str,
    distribution_version: str | None,
    expected: str,
) -> SystemRocm:
    resolved = resolve_system_rocm()
    version_file = resolved.root / ".info" / "version"
    try:
        actual = canonical_rocm_version(version_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TensileLiteRuntimeError(
            "The resolved ROCm installation has no readable release metadata.\n"
            f"  selected root: {resolved.root}\n"
            f"  selected by: {resolved.source}\n"
            f"  expected file: {version_file}"
        ) from exc
    validated = SystemRocm(
        resolved.root,
        actual,
        resolved.source,
        (resolved.root / "bin", resolved.root / "lib" / "llvm" / "bin"),
    )
    _validate_version(
        distribution,
        distribution_version,
        rocm_base_version(expected),
        validated,
    )
    return validated


def validate_distribution(
    distribution: str, distribution_version: str | None = None
) -> ValidatedRocm:
    expected = expected_rocm_version(distribution, distribution_version)
    
    python_sdk_version = _python_sdk_version()
    if python_sdk_version is not None:
        return _validate_python_sdk(distribution, distribution_version, expected, python_sdk_version)
    
    return _validate_system_rocm(distribution, distribution_version, expected)
