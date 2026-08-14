# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Resolve the ROCm installation required by the TensileLite package."""

from __future__ import annotations

from abc import ABC, abstractmethod
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

from _tensilelite_client_binding import ClientBindingError, ClientCandidate, standard_client_path


# Diagnostic source label for an active Python rocm-sdk-core installation.
_PYTHON_SDK_SOURCE = "active Python rocm_sdk_core"



class TensileLiteRuntimeError(ImportError):
    """The installed TensileLite wheel and ROCm runtime do not match."""


@dataclass(frozen=True)
class ValidatedRocm(ABC):
    """A ROCm installation selected and validated for this process."""

    path: Path
    version: str
    toolchain_paths: tuple[Path, ...]
    source: str

    @abstractmethod
    def default_client(self) -> ClientCandidate:
        """Return the default client candidate for this installation."""


@dataclass(frozen=True)
class PythonRocm(ValidatedRocm):
    source: str = _PYTHON_SDK_SOURCE

    def default_client(self) -> ClientCandidate:
        # Enable the Python SDK default when rocm-sdk-libraries publishes the
        # tensilelite-client console-script trampoline.
        raise ClientBindingError(
            "rocm-sdk-libraries does not publish the tensilelite-client console script. "
            "Configure an explicit TensileLite client binding instead."
        )


class SystemRocm(ValidatedRocm):
    def default_client(self) -> ClientCandidate:
        return ClientCandidate(standard_client_path(self.path), self.source)


@dataclass(frozen=True)
class SystemRocmRoot:
    root: Path
    source: str


# ---------------------------------------------------------------------------
# Public version normalization
# ---------------------------------------------------------------------------

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
        return _canonical_rocm_version(local[len("devrocm") :])
    if not local.startswith("rocm"):
        raise TensileLiteRuntimeError(
            f"{distribution} {distribution_version!r} has an invalid ROCm release tag."
        )
    return _canonical_rocm_version(local[len("rocm") :])


# ---------------------------------------------------------------------------
# Public installation validation
# ---------------------------------------------------------------------------

def validate_distribution(
    distribution: str, distribution_version: str | None = None
) -> ValidatedRocm:
    expected = expected_rocm_version(distribution, distribution_version)
    python_sdk_version = _python_sdk_version()
    if python_sdk_version is not None:
        return _validate_python_sdk(distribution, distribution_version, expected, python_sdk_version)
    return _validate_system_rocm(distribution, distribution_version, expected)


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _validate_compatibility(
    *,
    distribution: str,
    distribution_version: str | None,
    expected_version: str,
    actual_version: str,
    path: Path,
    source: str,
) -> None:
    if actual_version == expected_version:
        return
    shown_version = distribution_version or package_version(distribution)
    raise TensileLiteRuntimeError(
        f"{distribution} and ROCm release mismatch.\n"
        f"  {distribution} version: {shown_version}\n"
        f"  expected ROCm: {expected_version}\n"
        f"  found ROCm: {actual_version}\n"
        f"  selected: {path}\n"
        f"  selected by: {source}\n"
        "Install the wheel from the matching ROCm wheel index or select the matching ROCM_PATH."
    )


# A canonical ROCm release has a three-part numeric base version and may
# include a lowercase publication suffix, such as 10.1.0a20260814.
_RELEASE_RE: re.Pattern[str] = re.compile(
    r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$",
    re.IGNORECASE,
)


def _canonical_rocm_version(value: str) -> str:
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _RELEASE_RE.fullmatch(value):
        raise TensileLiteRuntimeError(f"Invalid ROCm release value: {value!r}")
    return value


def _rocm_base_version(value: str) -> str:
    match = re.match(r"^([0-9]+(?:\.[0-9]+){2})", value)
    if match is None:
        raise TensileLiteRuntimeError(f"Invalid ROCm release value: {value!r}")
    return match.group(1)


# ---------------------------------------------------------------------------
# Python SDK adapter
# ---------------------------------------------------------------------------

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


def _validate_python_sdk(
    distribution: str,
    distribution_version: str | None,
    expected: str,
    python_sdk_version: str,
) -> PythonRocm:
    version = _canonical_rocm_version(python_sdk_version)
    path = _python_sdk_location()
    toolchain_paths = _python_sdk_toolchain_paths()
    _validate_compatibility(
        distribution=distribution,
        distribution_version=distribution_version,
        expected_version=expected,
        actual_version=version,
        path=path,
        source=_PYTHON_SDK_SOURCE,
    )
    return PythonRocm(path=path, version=version, toolchain_paths=toolchain_paths)


def _python_sdk_location() -> Path:
    try:
        import rocm_sdk_core

        location = rocm_sdk_core.__file__
        if location is None:
            raise AttributeError("__file__ is None")
        return Path(location).resolve().parent
    except Exception as exc:
        raise TensileLiteRuntimeError(
            "The active Python ROCm core package has no installed location.\n"
            "  selected by: active Python rocm_sdk_core\n"
            "Install the matching rocm core package."
        ) from exc


def _python_sdk_toolchain_paths() -> tuple[Path, ...]:
    user_scheme = sysconfig.get_preferred_scheme("user")
    script_dirs = (
        sysconfig.get_path("scripts"),
        sysconfig.get_path("scripts", scheme=user_scheme),
    )
    paths = tuple(
        dict.fromkeys(Path(scripts).resolve() for scripts in script_dirs if scripts)
    )
    if not paths:
        raise TensileLiteRuntimeError(
            "The active Python ROCm SDK has no scripts directory for its tool trampolines.\n"
            "  selected by: active Python rocm_sdk_core"
        )
    return paths


# ---------------------------------------------------------------------------
# Conventional-prefix adapter
# ---------------------------------------------------------------------------

def _validate_system_rocm(
    distribution: str,
    distribution_version: str | None,
    expected: str,
) -> SystemRocm:
    resolved = _resolve_system_rocm()
    version_file = resolved.root / ".info" / "version"
    try:
        actual = _canonical_rocm_version(version_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TensileLiteRuntimeError(
            "The resolved ROCm installation has no readable release metadata.\n"
            f"  selected root: {resolved.root}\n"
            f"  selected by: {resolved.source}\n"
            f"  expected file: {version_file}"
        ) from exc
    path = resolved.root
    toolchain_paths = (path / "bin", path / "lib" / "llvm" / "bin")
    _validate_compatibility(
        distribution=distribution,
        distribution_version=distribution_version,
        expected_version=_rocm_base_version(expected),
        actual_version=actual,
        path=path,
        source=resolved.source,
    )
    return SystemRocm(
        path=path,
        version=actual,
        source=resolved.source,
        toolchain_paths=toolchain_paths,
    )


def _resolve_system_rocm() -> SystemRocmRoot:
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


def _validated_system_rocm_root(root: Path, source: str) -> SystemRocmRoot:
    if not root.is_dir():
        raise TensileLiteRuntimeError(
            "ROCm installation not found.\n"
            f"  selected root: {root}\n"
            f"  selected by: {source}"
        )
    return SystemRocmRoot(root.resolve(), source)
