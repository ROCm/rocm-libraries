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
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from . import _build_rocm

# Diagnostic source label for an active Python rocm-sdk-core installation.
_PYTHON_SDK_SOURCE = "active Python rocm_sdk_core"
_THEROCK_CI_SOURCE = "TheRock CI build"

# Internal marker set only by hipBLASLt's CMake, in the environment of its own
# code-generation commands, when it builds under TheRock. It alone selects the
# TheRock build-time adapter, so an installed wheel can never enter that mode.
_ADAPTER_ENV = "TENSILELITE_ROCM_ADAPTER"
_THEROCK_CI_ADAPTER = "therock-ci"

class TensileLiteRuntimeError(ImportError):
    """The installed TensileLite wheel and ROCm runtime do not match."""


@dataclass(frozen=True)
class ValidatedRocm:
    """A ROCm installation selected and validated for this process.

    ``version`` is the selected installation's identity: the full published
    identity (``identity_kind == "full"``), only its ``X.Y.Z`` base when that
    is all the installation records (``"base"``), or None when it is unknown,
    as in a TheRock build that passed no version. It is used for comparisons
    and diagnostics only.
    """

    path: Path
    version: str | None
    executable_search_paths: tuple[Path, ...]
    source: str
    identity_kind: str | None = "full"


@dataclass(frozen=True)
class PythonRocm(ValidatedRocm):
    source: str = _PYTHON_SDK_SOURCE


@dataclass(frozen=True)
class TheRockCIRocm(ValidatedRocm):
    source: str = _THEROCK_CI_SOURCE


class SystemRocm(ValidatedRocm):
    pass


@dataclass(frozen=True)
class SystemRocmRoot:
    root: Path
    source: str


# ---------------------------------------------------------------------------
# Public installation validation
# ---------------------------------------------------------------------------

def validate_distribution(
    distribution: str, distribution_version: str | None = None
) -> ValidatedRocm:
    """Select and validate the ROCm installation for a TensileLite wheel."""
    expected = _expected_rocm_version(distribution, distribution_version)
    rocm = _select_and_validate(distribution, distribution_version, expected)
    if expected is None:
        _warn_not_checked(distribution, distribution_version, rocm)
    return rocm


def _select_and_validate(
    distribution: str, distribution_version: str | None, expected: str | None
) -> ValidatedRocm:
    if _therock_ci_selected():
        return _validate_therock_ci(
            distribution, distribution_version, expected, _therock_ci_version()
        )
    python_sdk_version = _python_sdk_version()
    if python_sdk_version is not None:
        return _validate_python_sdk(distribution, distribution_version, expected, python_sdk_version)
    return _validate_system_rocm(distribution, distribution_version, expected)


class RocmNotCheckedWarning(UserWarning):
    """A development build was used without checking it against the selected ROCm."""


_not_checked_warned = False


def _warn_not_checked(
    distribution: str, distribution_version: str | None, rocm: ValidatedRocm
) -> None:
    """Warn, once per process, that a development build makes no ROCm claim."""
    global _not_checked_warned
    if _not_checked_warned:
        return
    _not_checked_warned = True
    shown_version = distribution_version or package_version(distribution)
    warnings.warn(
        f"{distribution} {shown_version} is a development build with no ROCm identity, "
        "so it was not checked against the selected ROCm.\n"
        f"  selected ROCm: {rocm.version or 'unknown'} at {rocm.path}\n"
        f"  selected by: {rocm.source}",
        RocmNotCheckedWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

_DEV_IDENTITY_RE: re.Pattern[str] = re.compile(
    r"^(?P<release>[0-9]+(?:\.[0-9]+){2}(?:(?:a|b|rc)[0-9]+)?\.dev[0-9]+)"
    r"\.(?P<commit>[a-z0-9]+)(?P<variants>(?:\.[a-z0-9]+)*)$"
)


def comparable_identity(identity: str, *, base_only: bool = False) -> str:
    """Return the part of a canonical ROCm identity that must match.

    - ``base_only``: only ``X.Y.Z``, for installations that record no more
      (``.info/version``).
    - TheRock development identities (``X.Y.Z.devN+<commit>[.<variant>...]``):
      everything except the leading commit token of the local segment. It names
      the CI run, not the ROCm content, and stage reuse from a baseline run
      mixes runs within one build. Variant tokens (``asan``, ...) still count.
    - Release, rc and nightly identities: the whole identity.
    """
    if base_only:
        return _rocm_base_version(identity)
    match = _DEV_IDENTITY_RE.fullmatch(identity)
    if match:
        return match["release"] + match["variants"]
    return identity


def identities_match(expected: str, actual: str, *, base_only: bool = False) -> bool:
    return comparable_identity(expected, base_only=base_only) == comparable_identity(
        actual, base_only=base_only
    )


def _validate_compatibility(
    *,
    distribution: str,
    distribution_version: str | None,
    expected_version: str | None,
    actual_version: str | None,
    path: Path,
    source: str,
    base_only: bool = False,
) -> None:
    """Raise when the selected ROCm identity does not match the wheel's claim.

    A wheel without a ROCm claim (a ``+g<commit>`` development build), or an
    installation whose identity is unknown, has nothing to compare.
    """
    if expected_version is None or actual_version is None:
        return
    if identities_match(expected_version, actual_version, base_only=base_only):
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


def _expected_rocm_version(
    distribution: str, distribution_version: str | None = None
) -> str | None:
    """Extract the wheel's canonical ROCm identity from its version label.

    Returns None for a development build (``+g<commit>[.dirty]``, or no label),
    which makes no ROCm claim.
    """
    if distribution_version is None:
        try:
            distribution_version = package_version(distribution)
        except PackageNotFoundError as exc:
            raise TensileLiteRuntimeError(
                f"{distribution} must be installed as a ROCm-versioned wheel; "
                "direct source-tree imports are unsupported."
            ) from exc
    if "+" not in distribution_version:
        return None
    local = distribution_version.split("+", 1)[1]
    if re.fullmatch(r"g[0-9a-f]{40}(?:\.dirty)?", local):
        return None
    if local.startswith("devrocm"):
        return _canonical_rocm_version(local[len("devrocm") :])
    if not local.startswith("rocm"):
        raise TensileLiteRuntimeError(
            f"{distribution} {distribution_version!r} has an invalid ROCm release tag."
        )
    return _canonical_rocm_version(local[len("rocm") :])


# A canonical ROCm release has a three-part numeric base version and may
# include a lowercase publication suffix, such as 10.1.0a20260814.
_RELEASE_RE: re.Pattern[str] = re.compile(
    r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$",
    re.IGNORECASE,
)


def _canonical_rocm_version(value: str) -> str:
    """Normalize and validate a ROCm release identifier."""
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _RELEASE_RE.fullmatch(value):
        raise TensileLiteRuntimeError(f"Invalid ROCm release value: {value!r}")
    return value


def _rocm_base_version(value: str) -> str:
    """Return the three-part base release from a canonical ROCm version."""
    match = re.match(r"^([0-9]+(?:\.[0-9]+){2})", value)
    if match is None:
        raise TensileLiteRuntimeError(f"Invalid ROCm release value: {value!r}")
    return match.group(1)


# ---------------------------------------------------------------------------
# TheRock CI adapter
# ---------------------------------------------------------------------------

def _therock_ci_selected() -> bool:
    """Return whether hipBLASLt's TheRock build selected the build-time adapter."""
    value = os.environ.get(_ADAPTER_ENV)
    if value is None:
        return False
    if value != _THEROCK_CI_ADAPTER:
        raise TensileLiteRuntimeError(
            f"{_ADAPTER_ENV}={value!r} is not a TensileLite ROCm adapter. It is an "
            f"internal variable set only by hipBLASLt's TheRock build "
            f"({_ADAPTER_ENV}={_THEROCK_CI_ADAPTER}); unset it."
        )
    return True


def _therock_ci_version() -> str | None:
    """Return TheRock's package identity, or None for a development build.

    TheRock passes THEROCK_PACKAGE_VERSION only when it knows a real version;
    a missing value or local builds' "git" means a development build.
    """
    value = os.environ.get("THEROCK_PACKAGE_VERSION")
    if not value or value == "git":
        return None
    return value


def _therock_ci_executable_search_paths() -> tuple[Path, ...]:
    """Return TheRock's graph-prepared executable search locations."""
    raw_path = os.environ.get("PATH", "")
    paths = tuple(
        dict.fromkeys(Path(item).resolve() for item in raw_path.split(os.pathsep) if item)
    )
    if not paths:
        raise TensileLiteRuntimeError(
            "TheRock CI provided no executable search paths."
        )
    return paths


def _validate_therock_ci(
    distribution: str,
    distribution_version: str | None,
    expected: str,
    therock_package_version: str | None,
) -> TheRockCIRocm:
    """Validate a wheel against TheRock's graph-owned build identity."""
    version = (
        _canonical_rocm_version(therock_package_version)
        if therock_package_version is not None
        else None
    )
    executable_search_paths = _therock_ci_executable_search_paths()
    selected_path = executable_search_paths[0]
    _validate_compatibility(
        distribution=distribution,
        distribution_version=distribution_version,
        expected_version=expected,
        actual_version=version,
        path=selected_path,
        source=_THEROCK_CI_SOURCE,
    )
    return TheRockCIRocm(
        path=selected_path,
        version=version,
        executable_search_paths=executable_search_paths,
        identity_kind="full" if version is not None else None,
    )


# ---------------------------------------------------------------------------
# Python SDK adapter
# ---------------------------------------------------------------------------

def _python_sdk_version() -> str | None:
    """Return the active rocm_sdk_core version, if it is installed."""
    try:
        import rocm_sdk_core
    except ModuleNotFoundError as exc:
        if exc.name == "rocm_sdk_core":
            return None
        raise TensileLiteRuntimeError(
            "The active Python ROCm core package could not be imported.\n"
            "  selected by: active Python rocm_sdk_core\n"
            "Repair or reinstall the active rocm core package."
        ) from exc

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
    """Validate the active Python ROCm SDK against the TensileLite wheel."""
    version = _canonical_rocm_version(python_sdk_version)
    path = _python_sdk_location()
    executable_search_paths = _python_sdk_executable_search_paths()
    _validate_compatibility(
        distribution=distribution,
        distribution_version=distribution_version,
        expected_version=expected,
        actual_version=version,
        path=path,
        source=_PYTHON_SDK_SOURCE,
    )
    return PythonRocm(
        path=path,
        version=version,
        executable_search_paths=executable_search_paths,
    )


def _python_sdk_location() -> Path:
    """Return the installed rocm_sdk_core package directory."""
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


def _python_sdk_executable_search_paths() -> tuple[Path, ...]:
    """Return Python script directories that expose SDK tool trampolines."""
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
    """Validate a conventional ROCm prefix against the TensileLite wheel."""
    resolved = _resolve_system_rocm()
    actual, identity_kind = _system_rocm_identity(resolved)
    path = resolved.root
    executable_search_paths = (
        path / "bin",
        path / "lib" / "llvm" / "bin",
        path / "libexec" / "hipblaslt" / "tensilelite",
    )
    _validate_compatibility(
        distribution=distribution,
        distribution_version=distribution_version,
        expected_version=expected,
        actual_version=actual,
        path=path,
        source=resolved.source,
        base_only=identity_kind == "base",
    )
    return SystemRocm(
        path=path,
        version=actual,
        source=resolved.source,
        executable_search_paths=executable_search_paths,
        identity_kind=identity_kind,
    )


def _system_rocm_identity(resolved: SystemRocmRoot) -> tuple[str, str]:
    """Return a conventional prefix's identity and whether it is full or base.

    TheRock-built roots record the full published identity in
    share/therock/therock_manifest.json; it is used only if it is a release
    identity (local builds record "git"). Pre-TheRock roots have only
    .info/version, whose X.Y.Z base is compared base to base.
    """
    root = resolved.root
    manifest = _build_rocm.manifest_identity(root)
    info = _build_rocm.info_version(root)
    if manifest is not None and info is not None and _build_rocm.base_version(manifest) != info:
        raise TensileLiteRuntimeError(
            "Inconsistent ROCm installation: its two version records disagree.\n"
            f"  {root / _build_rocm.MANIFEST_RELPATH}: {manifest}\n"
            f"  {root / _build_rocm.INFO_VERSION_RELPATH}: {info}\n"
            f"  selected by: {resolved.source}\n"
            "Repair or reinstall the ROCm installation."
        )
    if manifest is not None:
        return _canonical_rocm_version(manifest), "full"
    if info is not None:
        return info, "base"
    raise TensileLiteRuntimeError(
        "The resolved ROCm installation has no readable release metadata.\n"
        f"  selected root: {root}\n"
        f"  selected by: {resolved.source}\n"
        f"  expected file: {root / _build_rocm.INFO_VERSION_RELPATH}"
    )


def _resolve_system_rocm() -> SystemRocmRoot:
    """Select a ROCm prefix from the environment, default location, or PATH."""
    if "ROCM_PATH" in os.environ:
        explicit = os.environ["ROCM_PATH"]
        if not explicit:
            raise TensileLiteRuntimeError(
                "ROCM_PATH is set but empty.\n"
                "Set ROCM_PATH to the matching conventional ROCm installation."
            )
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
    """Return the ROCm prefix reported by hipconfig, if it can be queried."""
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
    """Resolve root after confirming that it is an existing directory."""
    if not root.is_dir():
        raise TensileLiteRuntimeError(
            "ROCm installation not found.\n"
            f"  selected root: {root}\n"
            f"  selected by: {source}"
        )
    return SystemRocmRoot(root.resolve(), source)
