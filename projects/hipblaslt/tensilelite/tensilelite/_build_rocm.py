# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Identify the ROCm a native TensileLite artifact is compiled against.

CMake runs this file with the build Python and the ROCm root it selected for
the client, and bakes the answer into ``tensilelite-client --build-rocm``::

    python _build_rocm.py <rocm-root>   ->   "<identity> <full|base|unknown>"

The identity comes from, in order:

1. ``rocm_sdk_core.__version__`` when the root is the build Python's own ROCm
   SDK (``rocm-sdk path --root``): the full published identity;
2. ``share/therock/therock_manifest.json`` ``rocm_package_version`` on a
   TheRock-built root, if it is a release identity (local builds record
   ``"git"``): full;
3. ``.info/version``, the only record on pre-TheRock installs: base
   (``X.Y.Z``);
4. otherwise ``unknown``.

If the manifest and ``.info/version`` disagree on the base version the
installation is inconsistent; the answer is ``unknown`` and a warning goes to
stderr. The value never feeds the ``tensilelite`` wheel's version.

This file must use only the standard library and must not import the rest of
the package: CMake loads it by path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import NamedTuple, Optional

MANIFEST_RELPATH = Path("share") / "therock" / "therock_manifest.json"
INFO_VERSION_RELPATH = Path(".info") / "version"

# TheRock's release, rc, nightly and dev identities, for example 7.12.0,
# 7.12.0rc1, 7.12.0a20261110 and 7.12.0.dev0+<sha>.asan.
_RELEASE_IDENTITY_RE = re.compile(
    r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)(?:(?:a|b|rc)[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[a-z0-9]+(?:\.[a-z0-9]+)*)?$"
)
_INFO_VERSION_RE = re.compile(r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)")


class BuildRocm(NamedTuple):
    identity: str
    kind: str  # "full", "base" or "unknown"

    def __str__(self) -> str:
        return f"{self.identity} {self.kind}"


UNKNOWN = BuildRocm("unknown", "unknown")


def base_version(identity: str) -> Optional[str]:
    """Return the ``X.Y.Z`` part of a ROCm identity, or None."""
    match = _INFO_VERSION_RE.match(identity)
    return match["base"] if match else None


def manifest_identity(root: Path) -> Optional[str]:
    """Return the root's TheRock release identity, or None if it has none."""
    try:
        data = json.loads((Path(root) / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("rocm_package_version") if isinstance(data, dict) else None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _RELEASE_IDENTITY_RE.fullmatch(value) else None


def info_version(root: Path) -> Optional[str]:
    """Return the base version from the root's ``.info/version``, or None."""
    try:
        text = (Path(root) / INFO_VERSION_RELPATH).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return base_version(text)


def _python_sdk_identity(root: Path) -> Optional[str]:
    """Return ``rocm_sdk_core.__version__`` if ``root`` is this Python's ROCm SDK."""
    try:
        import rocm_sdk_core  # noqa: PLC0415 - optional, only in pip-SDK environments
    except ImportError:
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "rocm_sdk", "path", "--root"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sdk_root = result.stdout.strip()
    if result.returncode != 0 or not sdk_root:
        return None
    if os.path.realpath(sdk_root) != os.path.realpath(root):
        return None
    version = getattr(rocm_sdk_core, "__version__", None)
    return version if isinstance(version, str) and version else None


def resolve(root: Path, warn=lambda message: print(message, file=sys.stderr)) -> BuildRocm:
    root = Path(root)
    sdk = _python_sdk_identity(root)
    if sdk is not None:
        return BuildRocm(sdk, "full")

    manifest = manifest_identity(root)
    info = info_version(root)
    if manifest is not None and info is not None and base_version(manifest) != info:
        warn(
            f"inconsistent ROCm installation at {root}: {MANIFEST_RELPATH} says "
            f"{manifest} but {INFO_VERSION_RELPATH} says {info}; recording the "
            "client's build ROCm as unknown"
        )
        return UNKNOWN
    if manifest is not None:
        return BuildRocm(manifest, "full")
    if info is not None:
        return BuildRocm(info, "base")
    return UNKNOWN


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(f"usage: {Path(sys.argv[0]).name} ROCM_ROOT", file=sys.stderr)
        return 2
    print(resolve(Path(argv[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
