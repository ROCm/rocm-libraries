# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build-time helpers for TensileLite's distribution version.

The build backend never looks for a ROCm installation. PEP 517 frontends
(pip, tox, build) isolate the backend from the caller's environment, so a
version read from ``ROCM_PATH`` or ``/opt/rocm`` there depends on the frontend
rather than on the build. A build with no explicit ROCm identity is a
development build and is labelled with the source commit instead.
"""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Optional

_SOURCE_ROOT = Path(__file__).resolve().parent
_source_revision = runpy.run_path(str(_SOURCE_ROOT / "tensilelite" / "_source_revision.py"))


def source_revision(root: Path = _SOURCE_ROOT) -> Optional[str]:
    return _source_revision["resolve"](root)


def render_build_info(revision: Optional[str]) -> str:
    return _source_revision["render_build_info"](revision)


def distribution_version(component_version: str, revision: Optional[str]) -> str:
    """Return ``<component>+g<sha>[.dirty]``, or the bare component without a revision."""
    return _source_revision["distribution_version"](component_version, revision)
