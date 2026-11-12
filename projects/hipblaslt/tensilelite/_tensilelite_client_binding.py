# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This private module deliberately lives at the wheel top level so
# tensilelite_configure_client can manage an installed binding without importing
# tensilelite, whose package initialization validates the ROCm runtime.

"""Client-selection policy shared by configuration and package startup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
import runpy
import subprocess
import sys
from urllib.parse import unquote, urlparse
import warnings


def client_executable_name(platform: str | None = None) -> str:
    """Filename used by both CMake's client output and installed client lookup."""
    platform = sys.platform if platform is None else platform
    return "tensilelite-client.exe" if platform == "win32" else "tensilelite-client"


def built_client_path(build_dir: str | os.PathLike[str], platform: str | None = None) -> Path:
    """Absolute client output path for a TensileLite CMake build directory."""
    return Path(build_dir).resolve() / "tensilelite" / "client" / client_executable_name(platform)


class ClientBindingError(RuntimeError):
    """A client binding cannot be established or used."""


class ClientBindingWarning(UserWarning):
    """A client is usable, but may not match this installation exactly."""


@dataclass(frozen=True)
class Installation:
    package_dir: Path
    identifier: str
    version: str
    # The rocm-libraries commit this package was built from, or None. Installed
    # packages read it from their _build_info.py; editable installs compute it
    # from git when the installation is identified.
    source_revision: str | None
    editable: bool


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
    editable = any(_is_editable(distribution) for distribution in matches)
    return Installation(
        package_dir,
        identifier,
        versions.pop(),
        _installation_source_revision(package_dir, editable),
        editable,
    )


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
    installation: Installation,
) -> ClientCandidate:
    """Select the explicit client binding or delegate to default lookup."""
    configured = read_binding(installation)
    if configured is not None:
        return ClientCandidate(configured, "explicit TensileLite client binding")
    return default_client()


def default_client_candidate(
    installation: Installation,
    executable_search_paths: tuple[Path, ...],
    source: str,
) -> ClientCandidate:
    """Find tensilelite-client in the selected executable search paths."""
    for directory in executable_search_paths:
        candidate = directory / client_executable_name()
        if candidate.is_file():
            return ClientCandidate(candidate, source)
    raise ClientBindingError(
        "tensilelite-client was not found in the selected executable paths:\n"
        f"  selected by: {source}\n"
        + "\n".join(f"  {directory}" for directory in executable_search_paths)
        + "\n"
        + _rebuild_hint(installation)
    )


def validate_client(client: Path, installation: Installation) -> None:
    """Verify that a client was built from the same rocm-libraries commit as this package.

    An installed package must match the client's ``--source-revision`` exactly,
    including any dirty suffix. An editable install compares the commit only:
    a different commit fails, and uncommitted changes on either side warn.
    """
    if not client.is_absolute():
        raise ClientBindingError(f"Client path must be absolute: {client}")
    if not client.is_file():
        raise ClientBindingError(f"Client path is not a regular file: {client}")
    if os.name != "nt" and not os.access(client, os.X_OK):
        raise ClientBindingError(f"Client path is not executable: {client}")

    client_revision = _query_client(client, "--source-revision")
    build_rocm = _query_client(client, "--build-rocm")
    details = (
        f"  tensilelite: {installation.version} at {installation.package_dir}\n"
        f"  tensilelite source revision: {installation.source_revision or 'none'}"
        f"{' (editable install)' if installation.editable else ''}\n"
        f"  client: {client}\n"
        f"  client source revision: {client_revision}\n"
        f"  client build ROCm: {build_rocm}\n"
    )

    if installation.source_revision is None:
        raise ClientBindingError(
            "This tensilelite installation records no source revision, so no client "
            "can be paired with it.\n"
            + details
            + "Install tensilelite from a git checkout of rocm-libraries, or build it "
            "with TENSILELITE_SOURCE_REVISION set to the commit it came from."
        )
    if client_revision == "none":
        raise ClientBindingError(
            "The client records no source revision, so it cannot be paired.\n"
            + details
            + _rebuild_hint(installation)
        )

    revisions = _source_revision_module(installation.package_dir)
    try:
        client_commit = revisions["commit_of"](client_revision)
    except revisions["SourceRevisionError"] as exc:
        raise ClientBindingError(
            f"Client source revision is not a git commit: {client_revision!r}\n" + details
        ) from exc

    if not installation.editable:
        if client_revision != installation.source_revision:
            raise ClientBindingError(
                "Client source revision mismatch: the client was not built from the "
                "same source as this tensilelite package.\n"
                + details
                + _rebuild_hint(installation)
            )
        return

    if client_commit != revisions["commit_of"](installation.source_revision):
        raise ClientBindingError(
            "Client commit mismatch: the client was built from a different commit "
            "than this editable tensilelite checkout.\n"
            + details
            + _rebuild_hint(installation)
        )
    if client_revision != installation.source_revision:
        warnings.warn(
            "The client and this editable tensilelite checkout share a commit, but "
            "one of them was built with uncommitted changes. Rebuild the client if "
            "those changes affect it.\n" + details,
            ClientBindingWarning,
            stacklevel=2,
        )


def _query_client(client: Path, option: str) -> str:
    """Run one of the client's one-line identity queries."""
    try:
        result = subprocess.run(
            [str(client), option],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise ClientBindingError(f"Could not launch client {client}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClientBindingError(
            f"Client {option} query timed out after five seconds: {client}"
        ) from exc
    except OSError as exc:
        raise ClientBindingError(f"Client loader failed for {client}: {exc}") from exc

    if result.returncode < 0:
        raise ClientBindingError(
            f"Client {option} query was terminated by signal {-result.returncode}: {client}"
        )
    if result.returncode:
        raise ClientBindingError(
            f"Client {option} query exited with status {result.returncode}: {client}. "
            "Clients built before source-revision pairing cannot be used; rebuild it."
        )
    if result.stderr:
        raise ClientBindingError(f"Client {option} query wrote to stderr: {result.stderr.rstrip()}")
    lines = result.stdout.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ClientBindingError(
            f"Client {option} query must print exactly one non-empty stdout line; "
            f"got {result.stdout!r}"
        )
    return lines[0]


def _rebuild_hint(installation: Installation) -> str:
    if installation.editable:
        return (
            "Rebuild the client from this checkout and bind it, for example with "
            "`invoke install`, or `invoke build-client` followed by "
            "`tensilelite-configure-client --client <build>/tensilelite/client/tensilelite-client`."
        )
    return (
        "Use the tensilelite-client shipped with the same build as this package, or "
        "build one from the same rocm-libraries commit and bind it with "
        "`tensilelite-configure-client --client <path>`."
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


@functools.lru_cache(maxsize=None)
def _source_revision_module(package_dir: Path) -> dict:
    """Load tensilelite/_source_revision.py by path.

    Importing tensilelite._source_revision would run tensilelite/__init__,
    which validates the ROCm runtime; client configuration must not need that.
    """
    return runpy.run_path(str(package_dir / "_source_revision.py"))


def _installation_source_revision(package_dir: Path, editable: bool) -> str | None:
    revisions = _source_revision_module(package_dir)
    if editable:
        # Computed from the checkout each time, so it cannot go stale. The
        # build-time TENSILELITE_SOURCE_REVISION override does not apply here.
        return revisions["resolve"](package_dir.parent, {})
    build_info = package_dir / "_build_info.py"
    return revisions["read_build_info"](build_info) if build_info.is_file() else None


def _is_editable(distribution: metadata.Distribution) -> bool:
    """Return whether PEP 610 metadata marks the distribution as an editable install."""
    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        return False
    try:
        return bool(json.loads(raw_direct_url).get("dir_info", {}).get("editable", False))
    except (AttributeError, TypeError, json.JSONDecodeError):
        return False


def _distribution_package_dir(distribution: metadata.Distribution) -> Path | None:
    """Return the tensilelite package directory described by distribution metadata."""
    for entry in distribution.files or ():
        if str(entry).replace("\\", "/") == "tensilelite/__init__.py":
            return Path(distribution.locate_file(entry)).resolve().parent

    raw_direct_url = distribution.read_text("direct_url.json")
    if raw_direct_url:
        try:
            direct_url = json.loads(raw_direct_url)["url"]
            parsed = urlparse(direct_url)
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        if parsed.scheme == "file":
            return (Path(_file_url_path(direct_url)) / "tensilelite").resolve()
    return None


def _file_url_path(url: str, *, platform: str = sys.platform) -> PurePath:
    """Convert a PEP 610 file URL without losing Windows drive or UNC roots."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    authority = unquote(parsed.netloc)
    if platform == "win32":
        if authority and authority != "localhost":
            return PureWindowsPath(f"//{authority}{path}")
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return PureWindowsPath(path)
    if authority and authority != "localhost":
        return PurePosixPath(f"//{authority}{path}")
    return PurePosixPath(path)
