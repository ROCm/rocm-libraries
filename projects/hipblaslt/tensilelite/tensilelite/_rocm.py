# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Resolve the ROCm installation required by the TensileLite package."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path


class TensileLiteRuntimeError(ImportError):
    """The installed TensileLite wheel and ROCm runtime do not match."""


@dataclass(frozen=True)
class ValidatedRocm:
    root: Path
    version: str
    source: str


@dataclass(frozen=True)
class ResolvedRocmRoot:
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


def _validated_root(root: Path, source: str) -> ResolvedRocmRoot:
    if not root.is_dir():
        raise TensileLiteRuntimeError(
            "ROCm installation not found.\n"
            f"  selected root: {root}\n"
            f"  selected by: {source}"
        )
    return ResolvedRocmRoot(root.resolve(), source)


def _python_core_metadata() -> tuple[ResolvedRocmRoot, str] | None:
    try:
        import rocm_sdk_core
    except ModuleNotFoundError:
        return None

    try:
        root = rocm_sdk_core.get_core_root()
        version = rocm_sdk_core.__version__
    except Exception as exc:
        raise TensileLiteRuntimeError(
            "The active Python ROCm core package could not resolve its runtime metadata.\n"
            "  selected by: active Python rocm_sdk_core\n"
            "Install the matching rocm core package."
        ) from exc
    return _validated_root(root, "active Python rocm_sdk_core"), version


def _python_core_root() -> ResolvedRocmRoot | None:
    metadata = _python_core_metadata()
    return metadata[0] if metadata is not None else None


def resolve_rocm_root() -> ResolvedRocmRoot:
    python_core = _python_core_root()
    if python_core is not None:
        return python_core

    explicit = os.environ.get("ROCM_PATH")
    if explicit:
        return _validated_root(Path(explicit).expanduser(), "explicit ROCM_PATH")
    if sys.platform != "win32":
        return _validated_root(Path("/opt/rocm"), "/opt/rocm")
    raise TensileLiteRuntimeError(
        "ROCm installation not found.\n"
        "  selected by: no active Python rocm_sdk_core or explicit ROCM_PATH\n"
        "Install the matching rocm[libraries,device-...] SDK or set ROCM_PATH."
    )


def validate_distribution(
    distribution: str, distribution_version: str | None = None
) -> ValidatedRocm:
    expected = expected_rocm_version(distribution, distribution_version)
    python_core = _python_core_metadata()
    if python_core is not None:
        resolved, core_version = python_core
        actual = canonical_rocm_version(core_version)
        expected_for_comparison = expected
    else:
        resolved = resolve_rocm_root()
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
        expected_for_comparison = rocm_base_version(expected)
    if actual != expected_for_comparison:
        shown_version = distribution_version or package_version(distribution)
        raise TensileLiteRuntimeError(
            f"{distribution} and ROCm release mismatch.\n"
            f"  {distribution} version: {shown_version}\n"
            f"  expected ROCm: {expected_for_comparison}\n"
            f"  found ROCm: {actual}\n"
            f"  selected root: {resolved.root}\n"
            f"  selected by: {resolved.source}\n"
            "Install the wheel from the matching ROCm wheel index or select the matching ROCM_PATH."
        )
    return ValidatedRocm(resolved.root, actual, resolved.source)
