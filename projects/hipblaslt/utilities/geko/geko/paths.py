# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Resolution of the hipBLASLt checkout root used by GEKO entry points.

GEKO needs a built hipBLASLt checkout to locate tensilelite, the Tensile
driver, hipblaslt-bench and the generated device-library artifacts. These
are build outputs of a hipBLASLt checkout, not part of the installable geko
package, so the package's own location is NOT a reliable anchor once it has been
copied into site-packages by 'pip install .'.

Entry points (bin/geko and scripts/*.py) live in the checkout and are
run in place, so they can supply a trustworthy anchor. Library code that
needs tensilelite, the bench binary, or the device library receives
hipblaslt_path as an explicit argument. SolutionUID generation and the
gfx950 CMS loader are the exception: those entry points call
bind_hipblaslt, and a direct call uses locate_hipblaslt_root.

Resolution priority (highest first):
  1. explicit  -- e.g. the --hipblaslt CLI flag or a function argument.
  2. the GEKO_HIPBLASLT_PATH environment variable.
  3. anchor    -- a path inside the checkout (typically an entry point's
     __file__); the nearest ancestor that looks like a hipBLASLt root wins.

locate_hipblaslt_root uses the process binding from bind_hipblaslt first,
then the same environment variable and ancestor walk. It does not exit the
process when the checkout is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

HIPBLASLT_PATH_ENV_VAR = "GEKO_HIPBLASLT_PATH"

_PathLike = Union[str, "os.PathLike[str]"]
_BOUND_HIPBLASLT_PATH: Path | None = None


def looks_like_hipblaslt_root(path: Path) -> bool:
    """Return True if path looks like a hipBLASLt checkout root.

    Uses the presence of a tensilelite/ directory as the marker, since every
    GEKO workflow depends on it (Tensile driver, client build, etc.). This is a
    structural check only and is used to *locate* the root; whether the checkout
    has been compiled is a separate concern (see is_hipblaslt_built).
    """
    return (path / "tensilelite").is_dir()


def is_hipblaslt_built(path: Path) -> bool:
    """Return True if the hipBLASLt checkout appears to be built.

    All three GEKO workflows (tune, search, bench) ultimately invoke compiled
    artifacts under build/release/ (hipblaslt-bench, the device library, etc.),
    so the presence of that directory is used as the "built" marker.
    """
    return (path / "build" / "release").is_dir()


def _env_checkout() -> Path | None:
    """Return the checkout named by ``GEKO_HIPBLASLT_PATH``.

    Args:
        None.

    Returns:
        The environment path, or None when the variable is unset or empty.
        The directory is not checked for a ``tensilelite/`` subdirectory.

    Raises:
        None.
    """
    value = os.environ.get(HIPBLASLT_PATH_ENV_VAR)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _ancestor_checkout(anchor: _PathLike) -> Path | None:
    """Return the nearest ancestor of ``anchor`` that contains ``tensilelite/``.

    Args:
        anchor: A file or directory inside a hipBLASLt checkout.

    Returns:
        That checkout root, or None when no ancestor contains ``tensilelite/``.

    Raises:
        None.
    """
    anchor_path = Path(anchor).expanduser().resolve()
    for candidate in (anchor_path, *anchor_path.parents):
        if looks_like_hipblaslt_root(candidate):
            return candidate
    return None


def bind_hipblaslt(hipblaslt_path: str | Path) -> None:
    """Remember the hipBLASLt checkout for this process.

    Search, tune, prune, and the CLI call this after they resolve the
    checkout. SolutionUID generation and the gfx950 CMS loader then use
    :func:`locate_hipblaslt_root` instead of taking the path themselves.
    A later call replaces the checkout.

    Args:
        hipblaslt_path: hipBLASLt checkout root used by this process.

    Returns:
        None.

    Raises:
        None.
    """
    global _BOUND_HIPBLASLT_PATH
    _BOUND_HIPBLASLT_PATH = Path(hipblaslt_path)


def locate_hipblaslt_root(anchor: _PathLike | None = None) -> Path | None:
    """Return the hipBLASLt checkout for library code that was not passed a path.

    A checkout passed to :func:`bind_hipblaslt` wins, including a checkout
    that does not contain ``tensilelite/``. Otherwise ``GEKO_HIPBLASLT_PATH``
    wins, again without falling through. Otherwise the nearest ancestor of
    ``anchor`` that contains ``tensilelite/`` is used.

    Args:
        anchor: File or directory used only when nothing is bound and the
            environment variable is unset. ``None`` skips that walk.

    Returns:
        The selected checkout, or None when no source names one.

    Raises:
        None.
    """
    if _BOUND_HIPBLASLT_PATH is not None:
        return _BOUND_HIPBLASLT_PATH
    env_root = _env_checkout()
    if env_root is not None:
        return env_root
    if anchor is None:
        return None
    return _ancestor_checkout(anchor)


def resolve_hipblaslt_path(
    explicit: _PathLike | None = None,
    anchor: _PathLike | None = None,
    require_built: bool = False,
) -> Path:
    """Resolve the hipBLASLt checkout root using the documented priority chain.

    Args:
        explicit: Caller-supplied path (e.g. the --hipblaslt flag). Highest
            priority when truthy.
        anchor: A path located inside the checkout (typically an entry point's
            __file__). Used to auto-detect the root when neither an explicit
            path nor the environment variable is set.
        require_built: When True, also verify the resolved checkout has been
            built (see is_hipblaslt_built) and fail otherwise. Used by the
            workflows that invoke compiled artifacts (tune, search, bench).

    Returns:
        The resolved and validated hipBLASLt root.

    Raises:
        SystemExit: If an explicit/env path is supplied but invalid, if no
            source resolves to a valid hipBLASLt checkout, or if require_built
            is set and the resolved checkout has not been built.
    """
    root: Path | None = None
    if explicit:
        root = Path(explicit).expanduser().resolve()
        source = "--hipblaslt"
    else:
        root = _env_checkout()
        source = f"${HIPBLASLT_PATH_ENV_VAR}"
    if root is not None and not looks_like_hipblaslt_root(root):
        print(
            f"Error: hipBLASLt path from {source} is not a valid checkout "
            f"(no tensilelite/ directory): '{root}'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if root is None and anchor is not None:
        root = _ancestor_checkout(anchor)

    if root is None:
        print(
            "Error: Could not locate the hipBLASLt checkout root. Pass --hipblaslt PATH or "
            f"set {HIPBLASLT_PATH_ENV_VAR} to a built hipBLASLt checkout.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if require_built and not is_hipblaslt_built(root):
        print(
            f"Error: hipBLASLt checkout at '{root}' does not appear to be built "
            "(missing build/release/). Build hipBLASLt before running this workflow.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return root
