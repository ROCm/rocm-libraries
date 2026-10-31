# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single source of truth for TensileLite release metadata."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys


_ROCM_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$", re.IGNORECASE)
_COMPONENT_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}$")
_SOURCE_ROOT = Path(__file__).resolve().parent


def canonical_rocm_version(value: str) -> str:
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _ROCM_RELEASE_RE.fullmatch(value):
        raise RuntimeError(
            "ROCm Python package builds require a release such as '7.2.4'; "
            f"got {value!r}"
        )
    return value


def component_version() -> str:
    value = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not _COMPONENT_RELEASE_RE.fullmatch(value):
        raise RuntimeError(
            "TensileLite VERSION must contain an X.Y.Z release; " f"got {value!r}"
        )
    return value


def rocm_version(rocm_root: str | os.PathLike[str] | None = None) -> str:
    root = Path(rocm_root) if rocm_root is not None else Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    version_file = root / ".info" / "version"
    try:
        return canonical_rocm_version(version_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(
            "Point ROCM_PATH at an installation containing .info/version "
            f"(looked at {version_file})."
        ) from exc


def distribution_version(rocm_root: str | os.PathLike[str] | None = None) -> str:
    return f"{component_version()}+rocm{rocm_version(rocm_root)}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} ROCM_ROOT")
    print(distribution_version(sys.argv[1]))
