# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Resolve the ROCm installation required by the TensileLite package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import os
from pathlib import Path
import re
import subprocess
import sys


class TensileLiteRuntimeError(ImportError):
    """The installed TensileLite wheel and ROCm runtime do not match."""


@dataclass(frozen=True)
class ValidatedRocm:
    root: Path
    version: str


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
    if not local.startswith("rocm"):
        raise TensileLiteRuntimeError(
            f"{distribution} {distribution_version!r} has an invalid ROCm release tag."
        )
    return canonical_rocm_version(local[len("rocm") :])


def _windows_sdk_root() -> Path:
    try:
        proc = subprocess.run(
            ["rocm-sdk", "path", "--root"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TensileLiteRuntimeError(
            "ROCM_PATH is unset and the Windows ROCm SDK root could not be resolved. "
            "Set ROCM_PATH to the matching SDK installation."
        ) from exc
    return Path(proc.stdout.strip())


def resolve_rocm_root() -> Path:
    explicit = os.environ.get("ROCM_PATH")
    root = Path(explicit).expanduser() if explicit else (
        _windows_sdk_root() if sys.platform == "win32" else Path("/opt/rocm")
    )
    if not root.is_dir():
        raise TensileLiteRuntimeError(
            "ROCm installation not found.\n"
            f"  resolved root: {root}\n"
            "Set ROCM_PATH to the ROCm release matching the installed Python wheel."
        )
    return root.resolve()


def validate_distribution(
    distribution: str, distribution_version: str | None = None
) -> ValidatedRocm:
    expected = expected_rocm_version(distribution, distribution_version)
    root = resolve_rocm_root()
    version_file = root / ".info" / "version"
    try:
        actual = canonical_rocm_version(version_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TensileLiteRuntimeError(
            "The resolved ROCm installation has no readable release metadata.\n"
            f"  ROCm root: {root}\n"
            f"  expected file: {version_file}"
        ) from exc
    if actual != expected:
        shown_version = distribution_version or package_version(distribution)
        raise TensileLiteRuntimeError(
            f"{distribution} and ROCm release mismatch.\n"
            f"  {distribution} version: {shown_version}\n"
            f"  expected ROCm: {expected}\n"
            f"  found ROCm: {actual}\n"
            f"  ROCm root: {root}\n"
            "Install the wheel from the matching ROCm wheel index or select the matching ROCM_PATH."
        )
    return ValidatedRocm(root, actual)
