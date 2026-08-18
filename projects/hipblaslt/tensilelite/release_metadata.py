# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build-time helpers for ROCm-local Python distribution metadata."""

from __future__ import annotations

import os
from pathlib import Path
import re


_ROCM_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$", re.IGNORECASE)


def canonical_rocm_version(value: str) -> str:
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _ROCM_RELEASE_RE.fullmatch(value):
        raise RuntimeError(
            "ROCm Python package builds require a release such as '7.2.4'; "
            f"got {value!r}"
        )
    return value


def rocm_version() -> str:
    explicit = os.environ.get("ROCM_VERSION")
    if explicit:
        return canonical_rocm_version(explicit)

    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    version_file = root / ".info" / "version"
    try:
        return canonical_rocm_version(version_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(
            "Set ROCM_VERSION or point ROCM_PATH at an installation containing "
            f".info/version (looked at {version_file})."
        ) from exc


def distribution_version(component_version: str) -> str:
    return f"{component_version}+rocm{rocm_version()}"
