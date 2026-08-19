# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation for the ROCm-native artifacts required by TensileLite."""

from __future__ import annotations

from pathlib import Path

from _tensilelite_client_binding import (
    ClientBindingError,
    default_client_candidate,
    selected_client,
    validate_client,
)

from ._rocm import TensileLiteRuntimeError, ValidatedRocm, validate_distribution


_client: Path | None = None
_installation: ValidatedRocm | None = None


def initialize() -> None:
    """Validate generator prerequisites without requiring the optional client."""
    from tensilelite import __version__

    global _installation

    _installation = validate_distribution("tensilelite", __version__)


def client_executable() -> Path:
    """Return the validated client selected by this installation on first use."""
    global _client

    if _client is not None:
        return _client
    if _installation is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")

    from tensilelite import __version__

    candidate = None
    try:
        candidate = selected_client(
            lambda: default_client_candidate(
                _installation.executable_search_paths,
                _installation.source,
            )
        )
        validate_client(candidate.path, __version__)
    except ClientBindingError as exc:
        selected = (
            f"  selected client: {candidate.path}\n"
            f"  selected by: {candidate.source}"
            if candidate is not None
            else "  selected by: TensileLite client binding lookup"
        )
        raise TensileLiteRuntimeError(
            f"{exc}\n"
            f"{selected}"
        ) from exc
    _client = candidate.path
    return _client


def executable_search_paths() -> list[Path]:
    """Return the executable locations for the frozen ROCm installation model."""
    if _installation is None:
        initialize()
    assert _installation is not None
    return list(_installation.executable_search_paths)
