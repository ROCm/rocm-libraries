# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``provenance.generated_from`` record a generation stamps on each KDP.

It names what produced a bundle: the engine manifest, the generator and the rocKE
library, each by the last commit that touched it (``-dirty`` when it has local edits).
It is informational. ``catalog_diff`` ignores it, so a regeneration at a later commit
still diffs clean against a shipped catalog.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: Spelled the same in every record, so a reader can tell "not in a git checkout"
#: from a missing key.
UNKNOWN = "unknown"

_GENERATOR_DIR = Path(__file__).resolve().parent.parent
_ROCKE_RELATIVE = Path("dnn-providers") / "hip-kernel-provider" / "rocke"


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def repo_root(start: Path = _GENERATOR_DIR) -> Path | None:
    top = _git(start, "rev-parse", "--show-toplevel")
    return Path(top) if top else None


def path_commit(root: Path, path: Path) -> str:
    """Last commit touching ``path`` under ``root``; ``<sha>-dirty`` if it has
    uncommitted changes; `UNKNOWN` when git cannot say."""
    if not path.exists():
        return UNKNOWN
    sha = _git(root, "log", "-1", "--format=%H", "--", str(path))
    if not sha:
        return UNKNOWN
    if _git(root, "status", "--porcelain", "--", str(path)):
        return f"{sha}-dirty"
    return sha


def generated_from(manifest: Path | None, config: Path | None = None) -> dict:
    """``{manifest?, config?, generator_commit, rocke_commit}``. Input paths are
    recorded repo-relative when they sit under the checkout, else by file name (an
    absolute path would pin one machine's scratch directory into a shipped KDP)."""
    root = repo_root()

    def label(path: Path) -> str:
        resolved = path.resolve()
        if root is not None:
            try:
                return resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                pass
        return path.name

    record: dict = {}
    if manifest is not None:
        record["manifest"] = label(manifest)
    if config is not None:
        record["config"] = label(config)
    if root is None:
        record["generator_commit"] = UNKNOWN
        record["rocke_commit"] = UNKNOWN
    else:
        record["generator_commit"] = path_commit(root, _GENERATOR_DIR)
        record["rocke_commit"] = path_commit(root, root / _ROCKE_RELATIVE)
    return record
