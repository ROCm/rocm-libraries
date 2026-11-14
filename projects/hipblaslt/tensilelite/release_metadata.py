# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single source of truth for TensileLite release metadata.

The build backend never looks for a ROCm installation. PEP 517 frontends
(pip, tox, build) isolate the backend from the caller's environment, so a
version read from ``ROCM_PATH`` or ``/opt/rocm`` there depends on the frontend
rather than on the build. A build with no explicit ROCm identity is a
development build and is labelled with the source commit instead.

CMake runs this file to give the native client the same values:

    release_metadata.py version          # the distribution version
    release_metadata.py source-revision  # the commit, or "none"
"""

from __future__ import annotations

from pathlib import Path
import re
import runpy
import sys
from typing import Optional

_COMPONENT_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}$")
_SOURCE_ROOT = Path(__file__).resolve().parent
_source_revision = runpy.run_path(str(_SOURCE_ROOT / "tensilelite" / "_source_revision.py"))


def component_version() -> str:
    value = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not _COMPONENT_RELEASE_RE.fullmatch(value):
        raise RuntimeError(
            "TensileLite VERSION must contain an X.Y.Z release; " f"got {value!r}"
        )
    return value


def source_revision(root: Path = _SOURCE_ROOT) -> Optional[str]:
    return _source_revision["resolve"](root)


def render_build_info(revision: Optional[str]) -> str:
    return _source_revision["render_build_info"](revision)


def distribution_version(revision: Optional[str]) -> str:
    """Return ``<component>+g<sha>[.dirty]``, or the bare component without a revision."""
    return _source_revision["distribution_version"](component_version(), revision)


def main(argv: list[str]) -> int:
    commands = ("version", "source-revision")
    if len(argv) != 1 or argv[0] not in commands:
        print(f"usage: {Path(sys.argv[0]).name} {{{'|'.join(commands)}}}", file=sys.stderr)
        return 2
    revision = source_revision()
    if argv[0] == "version":
        print(distribution_version(revision))
    else:
        print(revision or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
