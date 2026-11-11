# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation for the ROCm-native artifacts required by TensileLite."""

from __future__ import annotations

from pathlib import Path
import warnings

from _tensilelite_client_binding import (
    ClientBindingError,
    current_installation,
    default_client_candidate,
    selected_client,
    validate_client,
)

from ._rocm import (
    TensileLiteRuntimeError,
    ValidatedRocm,
    _canonical_rocm_version,
    _expected_rocm_version,
    comparable_identity,
    validate_distribution,
)


class ClientRocmWarning(UserWarning):
    """The client's build ROCm could not be confirmed against the selected ROCm."""


_client: Path | None = None
_installation: ValidatedRocm | None = None
_warned: set[str] = set()


def initialize() -> None:
    """Validate generator prerequisites without requiring the optional client."""
    from . import __version__

    global _installation

    _installation = validate_distribution("tensilelite", __version__)


def client_executable() -> Path:
    """Return the validated client selected by this installation on first use."""
    global _client

    if _client is not None:
        return _client
    if _installation is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")

    candidate = None
    try:
        package = current_installation()
        candidate = selected_client(
            lambda: default_client_candidate(
                package,
                _installation.executable_search_paths,
                _installation.source,
            ),
            package,
        )
        identity = validate_client(candidate.path, package)
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
    # A +rocm/+devrocm package was released with a client for its ROCm, so any
    # difference is an error; a +g development build only warns.
    strict = _expected_rocm_version("tensilelite", package.version) is not None
    check_client_build_rocm(identity.build_rocm, _installation, candidate.path, strict=strict)
    _client = candidate.path
    return _client


def executable_search_paths() -> list[Path]:
    """Return the executable locations for the frozen ROCm installation model."""
    if _installation is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")
    return list(_installation.executable_search_paths)


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(message, ClientRocmWarning, stacklevel=3)


def check_client_build_rocm(
    build_rocm: str, rocm: ValidatedRocm, client: Path, *, strict: bool = False
) -> None:
    """Compare the ROCm a client was compiled against with the selected ROCm.

    Full identities are compared in full, or base to base when either side
    records only a base version. A different major version is always an error
    (HIP's soname changes across major versions). Any other difference is an
    error for a ``strict`` (``+rocm``/``+devrocm``) package and warns once per
    process otherwise. An unknown identity on either side warns once.
    """
    details = (
        f"  client: {client}\n"
        f"  client build ROCm: {build_rocm}\n"
        f"  selected ROCm: {rocm.version or 'unknown'} at {rocm.path}\n"
        f"  selected by: {rocm.source}"
    )
    parts = build_rocm.split()
    kind = parts[1] if len(parts) == 2 else "unknown"
    if kind not in ("full", "base") or rocm.version is None:
        _warn_once(
            "unknown",
            "The ROCm the TensileLite client was built against could not be checked.\n"
            + details,
        )
        return
    try:
        client_identity = _canonical_rocm_version(parts[0])
    except TensileLiteRuntimeError:
        _warn_once("unknown", f"The client reports an unreadable build ROCm.\n{details}")
        return

    base_only = kind == "base" or rocm.identity_kind == "base"
    client_value = comparable_identity(client_identity, base_only=base_only)
    selected_value = comparable_identity(rocm.version, base_only=base_only)
    if client_value == selected_value:
        return
    if client_value.split(".", 1)[0] != selected_value.split(".", 1)[0]:
        raise TensileLiteRuntimeError(
            "The TensileLite client was built for a different major ROCm version "
            "than the one selected.\n"
            + details
            + "\nUse a client built against the selected ROCm, or select the ROCm it "
            "was built against."
        )
    if strict:
        raise TensileLiteRuntimeError(
            "The TensileLite client was built against a different ROCm than the one "
            "selected, and this package was released for a specific ROCm.\n"
            + details
            + "\nUse the client shipped with the selected ROCm."
        )
    _warn_once(
        f"mismatch:{client_value}:{selected_value}",
        "The TensileLite client was built against a different ROCm than the one "
        "selected; HIP is backward compatible within a major version, so it is used.\n"
        + details,
    )
