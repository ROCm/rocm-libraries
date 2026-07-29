# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path
import re


_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$", re.IGNORECASE)


def rocm_version() -> str:
    value = os.environ.get("ROCM_VERSION")
    if value is None:
        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        value = (root / ".info" / "version").read_text(encoding="utf-8")
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _RELEASE_RE.fullmatch(value):
        raise RuntimeError(f"Invalid ROCm release for compatibility packaging: {value!r}")
    return value
