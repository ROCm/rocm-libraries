# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation for the ROCm-native artifacts required by TensileLite."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import sys

from ._rocm import TensileLiteRuntimeError, validate_distribution


@dataclass(frozen=True)
class RuntimeInfo:
    """Resolved, version-checked ROCm artifacts used by TensileLite."""

    rocm_root: Path
    rocm_version: str
    client: Path


def _client_path(rocm_root: Path) -> Path:
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    return rocm_root / "libexec" / "hipblaslt" / "tensilelite" / executable


def _require_rocisa() -> None:
    """Require rocisa without interpreting its version or native layout."""
    try:
        import_module("rocisa")
    except (ImportError, OSError) as exc:
        raise TensileLiteRuntimeError(
            "TensileLite requires an independently packaged, importable rocisa dependency. "
            f"The rocisa import failed: {exc}"
        ) from exc


def validate_runtime(distribution_version: str) -> RuntimeInfo:
    """Validate the ROCm release, external rocisa dependency, and native client."""

    _require_rocisa()
    validated = validate_distribution("tensilelite", distribution_version)
    client = _client_path(validated.root)
    if not client.is_file():
        raise TensileLiteRuntimeError(
            "tensilelite-client is missing from the matching ROCm installation.\n"
            f"  tensilelite version: {distribution_version}\n"
            f"  ROCm root: {validated.root}\n"
            f"  expected client: {client}\n"
            "Install the matching hipBLASLt/TensileLite ROCm runtime package."
        )
    if os.name != "nt" and not os.access(client, os.X_OK):
        raise TensileLiteRuntimeError(
            "tensilelite-client is not executable.\n"
            f"  client: {client}\n"
            "Reinstall the matching hipBLASLt/TensileLite ROCm runtime package."
        )
    return RuntimeInfo(validated.root, validated.version, client)
