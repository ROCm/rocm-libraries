# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation for the ROCm-native artifacts required by TensileLite."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import sys

from ._rocm import TensileLiteRuntimeError, validate_distribution


_CLIENT_PATH_RECORD = "tensilelite-client-path.json"
_client: Path | None = None
_custom = False


def _rocm_client_path(rocm_root: Path) -> Path:
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    return rocm_root / "libexec" / "hipblaslt" / "tensilelite" / executable


def _custom_client_path() -> Path | None:
    candidates = list(distributions(name="tensilelite"))
    if not candidates:
        raise TensileLiteRuntimeError(
            "tensilelite must be installed before it can resolve its client binding."
        )
    installed = next(
        (candidate for candidate in candidates if candidate.read_text("WHEEL") is not None),
        candidates[0],
    )
    raw = installed.read_text(_CLIENT_PATH_RECORD)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TensileLiteRuntimeError(
            f"The installed {_CLIENT_PATH_RECORD} is not valid JSON. Reinstall tensilelite."
        ) from exc
    if not isinstance(value, str) or not value:
        raise TensileLiteRuntimeError(
            f"The installed {_CLIENT_PATH_RECORD} must contain one absolute path string."
        )
    path = Path(value)
    if not path.is_absolute():
        raise TensileLiteRuntimeError(
            f"The installed {_CLIENT_PATH_RECORD} contains a non-absolute path: {value!r}."
        )
    return path


def _require_rocisa() -> None:
    """Require rocisa without interpreting its version or native layout."""
    try:
        import_module("rocisa")
    except (ImportError, OSError) as exc:
        raise TensileLiteRuntimeError(
            "TensileLite requires an independently packaged, importable rocisa dependency. "
            f"The rocisa import failed: {exc}"
        ) from exc


def _validate_client(client: Path, *, custom: bool) -> None:
    if not client.is_file():
        source = "configured source-build" if custom else "matching ROCm installation"
        raise TensileLiteRuntimeError(
            f"tensilelite-client is missing from the {source}.\n"
            f"  expected client: {client}\n"
            + (
                "Reinstall tensilelite with a valid tensilelite.client-path."
                if custom
                else "Install the matching hipBLASLt/TensileLite ROCm runtime package."
            )
        )
    if os.name != "nt" and not os.access(client, os.X_OK):
        raise TensileLiteRuntimeError(
            "tensilelite-client is not executable.\n"
            f"  client: {client}\n"
            + (
                "Reinstall tensilelite with a valid tensilelite.client-path."
                if custom
                else "Reinstall the matching hipBLASLt/TensileLite ROCm runtime package."
            )
        )


def initialize(distribution_version: str) -> None:
    """Validate and freeze the client selected by this installation."""
    global _client, _custom

    _require_rocisa()
    validated = validate_distribution("tensilelite", distribution_version)
    custom = _custom_client_path()
    selected = custom if custom is not None else _rocm_client_path(validated.root)
    _validate_client(selected, custom=custom is not None)
    _client = selected
    _custom = custom is not None


def client_executable() -> Path:
    """Return the frozen client path after revalidating its filesystem state."""
    if _client is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")
    _validate_client(_client, custom=_custom)
    return _client
