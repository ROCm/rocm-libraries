# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This private module deliberately lives at the wheel top level so
# tensilelite_configure_client can manage an installed binding without importing
# tensilelite, whose package initialization validates the ROCm runtime.

"""Client-selection policy shared by configuration and package startup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlparse

from packaging.version import InvalidVersion, Version


_CLIENT_EXECUTABLE = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
"""Installed console-script filename for the TensileLite client."""


class ClientBindingError(RuntimeError):
    """A client binding cannot be established or used."""


@dataclass(frozen=True)
class Installation:
    package_dir: Path
    identifier: str
    version: str


@dataclass(frozen=True)
class ClientCandidate:
    path: Path
    source: str


# ---------------------------------------------------------------------------
# Public installation identity
# ---------------------------------------------------------------------------

def current_installation() -> Installation:
    """Identify the installed TensileLite package that owns this import."""
    package_dir = _package_dir()
    matches = [
        distribution
        for distribution in metadata.distributions(name="tensilelite")
        if _distribution_package_dir(distribution) == package_dir
    ]
    versions = {distribution.version for distribution in matches}
    if not matches or len(versions) != 1:
        raise ClientBindingError(
            "Could not identify one tensilelite version owning "
            f"{package_dir}; matching metadata versions: {sorted(versions)}."
        )
    identifier = hashlib.sha256(os.fsencode(str(package_dir))).hexdigest()
    return Installation(package_dir, identifier, versions.pop())


# ---------------------------------------------------------------------------
# Public binding storage
# ---------------------------------------------------------------------------

def binding_path(installation: Installation | None = None) -> Path:
    """Return the explicit-client binding path for an installation."""
    installation = installation or current_installation()
    return user_root() / "bindings" / installation.identifier / "client.json"


def read_binding(installation: Installation | None = None) -> Path | None:
    """Return an installation's explicitly configured client path, if any."""
    path = binding_path(installation)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ClientBindingError(f"Cannot read configured client binding {path}: {exc}") from exc

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClientBindingError(f"Configured client binding is not valid JSON: {path}") from exc

    if not isinstance(value, str) or not value:
        raise ClientBindingError(
            f"Configured client binding must contain one absolute path string: {path}"
        )

    client = Path(value)
    if not client.is_absolute():
        raise ClientBindingError(f"Configured client binding contains a relative path: {value!r}")
    return client


def user_root() -> Path:
    """Return the user-scoped directory that stores TensileLite bindings."""
    return Path.home() / ".tensilelite"


# ---------------------------------------------------------------------------
# Public client selection and validation
# ---------------------------------------------------------------------------


def selected_client(
    default_client: Callable[[], ClientCandidate],
    installation: Installation | None = None,
) -> ClientCandidate:
    """Select the explicit client binding or delegate to default lookup."""
    configured = read_binding(installation)
    if configured is not None:
        return ClientCandidate(configured, "explicit TensileLite client binding")
    return default_client()


def default_client_candidate(
    executable_search_paths: tuple[Path, ...],
    source: str,
) -> ClientCandidate:
    """Find tensilelite-client in the selected executable search paths."""
    for directory in executable_search_paths:
        candidate = directory / _CLIENT_EXECUTABLE
        if candidate.is_file():
            return ClientCandidate(candidate, source)
    raise ClientBindingError(
        "tensilelite-client was not found in the selected executable paths:\n"
        f"  selected by: {source}\n"
        + "\n".join(f"  {directory}" for directory in executable_search_paths)
    )


def validate_client(client: Path, expected_version: str) -> None:
    """Verify that a client executable reports the expected package version."""
    if not client.is_absolute():
        raise ClientBindingError(f"Client path must be absolute: {client}")
    if not client.is_file():
        raise ClientBindingError(f"Client path is not a regular file: {client}")
    if os.name != "nt" and not os.access(client, os.X_OK):
        raise ClientBindingError(f"Client path is not executable: {client}")

    try:
        result = subprocess.run(
            [str(client), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise ClientBindingError(f"Could not launch client {client}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClientBindingError(f"Client version query timed out after five seconds: {client}") from exc
    except OSError as exc:
        raise ClientBindingError(f"Client loader failed for {client}: {exc}") from exc

    if result.returncode < 0:
        raise ClientBindingError(
            f"Client version query was terminated by signal {-result.returncode}: {client}"
        )
    if result.returncode:
        raise ClientBindingError(
            f"Client version query exited with status {result.returncode}: {client}"
        )
    if result.stderr:
        raise ClientBindingError(f"Client version query wrote to stderr: {result.stderr.rstrip()}")
    lines = result.stdout.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ClientBindingError(
            "Client version query must print exactly one non-empty stdout line; "
            f"got {result.stdout!r}"
        )

    try:
        actual = Version(lines[0])
        expected = Version(expected_version)
    except InvalidVersion as exc:
        raise ClientBindingError(f"Client version output is not PEP 440: {lines[0]!r}") from exc

    if actual != expected:
        raise ClientBindingError(
            f"Client version mismatch: expected {expected}, found {actual} at {client}"
        )


# ---------------------------------------------------------------------------
# Private installation identity helpers
# ---------------------------------------------------------------------------

def _package_dir() -> Path:
    """Locate the installed tensilelite package without importing it."""
    spec = util.find_spec("tensilelite")
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise ClientBindingError("tensilelite must be installed before configuring its client.")
    return Path(next(iter(locations))).resolve()


def _distribution_package_dir(distribution: metadata.Distribution) -> Path | None:
    """Return the tensilelite package directory described by distribution metadata."""
    for entry in distribution.files or ():
        if str(entry).replace("\\", "/") == "tensilelite/__init__.py":
            return Path(distribution.locate_file(entry)).resolve().parent

    raw_direct_url = distribution.read_text("direct_url.json")
    if raw_direct_url:
        try:
            parsed = urlparse(json.loads(raw_direct_url)["url"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        if parsed.scheme == "file":
            return (Path(unquote(parsed.path)) / "tensilelite").resolve()
    return None
