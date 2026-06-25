################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Resolution of the hipBLASLt checkout root used by GEKO entry points.

GEKO needs a built hipBLASLt checkout to locate tensilelite, the Tensile
driver, hipblaslt-bench and the generated device-library artifacts. These
are build outputs of a hipBLASLt checkout, not part of the installable geko
package, so the package's own location is NOT a reliable anchor once it has been
copied into site-packages by 'pip install .'.

Entry points (bin/geko and scripts/*.py) live in the checkout and are
run in place, so they can supply a trustworthy anchor. Library code never
infers the path; it receives hipblaslt_path as an explicit argument.

Resolution priority (highest first):
  1. explicit  -- e.g. the --hipblaslt CLI flag or a function argument.
  2. the GEKO_HIPBLASLT_PATH environment variable.
  3. anchor    -- a path inside the checkout (typically an entry point's
     __file__); the nearest ancestor that looks like a hipBLASLt root wins.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

HIPBLASLT_PATH_ENV_VAR = "GEKO_HIPBLASLT_PATH"

_PathLike = Union[str, "os.PathLike[str]"]


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

    for source, candidate in (
        ("--hipblaslt", explicit),
        (f"${HIPBLASLT_PATH_ENV_VAR}", os.environ.get(HIPBLASLT_PATH_ENV_VAR)),
    ):
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if not looks_like_hipblaslt_root(path):
                raise SystemExit(
                    f"hipBLASLt path from {source} is not a valid checkout "
                    f"(no tensilelite/ directory): '{path}'"
                )
            root = path
            break

    if root is None and anchor is not None:
        anchor_path = Path(anchor).expanduser().resolve()
        for candidate in (anchor_path, *anchor_path.parents):
            if looks_like_hipblaslt_root(candidate):
                root = candidate
                break

    if root is None:
        raise SystemExit(
            "Could not locate the hipBLASLt checkout root. Pass --hipblaslt PATH or "
            f"set {HIPBLASLT_PATH_ENV_VAR} to a built hipBLASLt checkout."
        )

    if require_built and not is_hipblaslt_built(root):
        raise SystemExit(
            f"hipBLASLt checkout at '{root}' does not appear to be built "
            "(missing build/release/). Build hipBLASLt before running this workflow."
        )

    return root
