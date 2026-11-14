# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single source of truth for TensileLite release metadata.

The build backend never looks for a ROCm installation. PEP 517 frontends
(pip, tox, build) isolate the backend from the caller's environment, so a
version read from ``ROCM_PATH`` or ``/opt/rocm`` there depends on the frontend
rather than on the build. A build with no explicit ROCm identity is a
development build and is labelled with the source commit instead.

Only TheRock's CMake build passes an identity (``TENSILELITE_ROCM_VERSION``),
and only one TheRock actually published: release, rc and nightly identities
become ``+rocm<identity>``, development identities ``+devrocm<identity>``.

CMake runs this file to give the native client the same values:

    release_metadata.py version [--rocm-identity X]  # the distribution version
    release_metadata.py source-revision              # the commit, or "none"
"""

from __future__ import annotations

from pathlib import Path
import re
import runpy
import sys
from typing import Optional

_COMPONENT_RELEASE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}$")
_ROCM_IDENTITY_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[a-z0-9.]+)?$")
_SOURCE_ROOT = Path(__file__).resolve().parent
_source_revision = runpy.run_path(str(_SOURCE_ROOT / "tensilelite" / "_source_revision.py"))


def canonical_rocm_identity(value: str) -> str:
    """Normalize a ROCm identity for a PEP 440 local version label.

    A wheel version has one ``+``, so TheRock's ``7.12.0.dev0+<sha>.asan``
    becomes ``7.12.0.dev0.<sha>.asan``.
    """
    value = re.sub(r"[-_+]+", ".", value.strip().lower()).strip(".")
    if not _ROCM_IDENTITY_RE.fullmatch(value):
        raise RuntimeError(
            "A ROCm identity must look like '7.2.4', '7.12.0rc1', '7.12.0a20261110' or "
            f"'7.12.0.dev0+<commit>'; got {value!r}"
        )
    return value


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


def distribution_version(revision: Optional[str], rocm_identity: Optional[str] = None) -> str:
    """Return the distribution version.

    With a ROCm identity: ``<component>+rocm<identity>`` or, for a development
    identity, ``<component>+devrocm<identity>``; the commit is then recorded only
    in ``_build_info.py``. Without one: ``<component>+g<sha>[.dirty]``, or the bare
    component when there is no revision either.
    """
    if rocm_identity:
        identity = canonical_rocm_identity(rocm_identity)
        tag = "devrocm" if ".dev" in identity else "rocm"
        return f"{component_version()}+{tag}{identity}"
    return _source_revision["distribution_version"](component_version(), revision)


def main(argv: list[str]) -> int:
    usage = (
        f"usage: {Path(sys.argv[0]).name} "
        "{version [--rocm-identity IDENTITY] | source-revision}"
    )
    identity = None
    if argv[:1] == ["version"] and len(argv) == 3 and argv[1] == "--rocm-identity":
        identity = argv[2] or None
        argv = argv[:1]
    if len(argv) != 1 or argv[0] not in ("version", "source-revision"):
        print(usage, file=sys.stderr)
        return 2
    revision = source_revision()
    if argv[0] == "version":
        print(distribution_version(revision, identity))
    else:
        print(revision or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
