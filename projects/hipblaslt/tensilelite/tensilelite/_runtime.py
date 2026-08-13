# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation for the ROCm-native artifacts required by TensileLite."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from _tensilelite_client_binding import (
    ClientBindingError,
    selected_client,
    validate_client,
)

from ._rocm import TensileLiteRuntimeError, validate_distribution


_client: Path | None = None
_root: Path | None = None
_root_source: str | None = None
_toolchain_paths: tuple[Path, ...] | None = None
_distribution_version: str | None = None


def _require_rocisa() -> None:
    """Require rocisa without interpreting its version or native layout."""
    try:
        import_module("rocisa")
    except (ImportError, OSError) as exc:
        raise TensileLiteRuntimeError(
            "TensileLite requires an independently packaged, importable rocisa dependency. "
            f"The rocisa import failed: {exc}"
        ) from exc


def initialize(distribution_version: str) -> None:
    """Validate generator prerequisites without requiring the optional client."""
    global _client, _root, _root_source, _toolchain_paths, _distribution_version

    _require_rocisa()
    validated = validate_distribution("tensilelite", distribution_version)
    _client = None
    _root = validated.root
    _root_source = validated.source
    _toolchain_paths = validated.toolchain_paths
    _distribution_version = distribution_version


def client_executable() -> Path:
    """Return the validated client selected by this installation on first use."""
    global _client

    if _client is not None:
        return _client
    if _root_source is None or _distribution_version is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")
    if _root is None:
        raise TensileLiteRuntimeError(
            "tensilelite-client is unavailable for a Python ROCm SDK installation.\n"
            "  selected by: active Python rocm_sdk_core\n"
            "The client is not yet shipped by rocm-sdk-libraries; use a conventional ROCm prefix or configure a source-development client."
        )

    try:
        selected, custom = selected_client(_root)
        validate_client(selected, _distribution_version)
    except ClientBindingError as exc:
        raise TensileLiteRuntimeError(
            f"{exc}\n"
            f"  selected root: {_root}\n"
            f"  selected by: {_root_source}"
        ) from exc
    _client = selected
    return _client

def toolchain_search_paths() -> tuple[Path, ...]:
    """Return the tool locations for the frozen ROCm installation model."""
    if _toolchain_paths is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")
    return _toolchain_paths
