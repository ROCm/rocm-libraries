# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Describe the configured interpreter's SDK site for build-tree native AUTO.

The sidecar stays in the build tree. Native installed consumers discover SDK
distribution metadata in their own prefix; no build-machine library path ships.
"""

from __future__ import annotations

import argparse
import os
import sysconfig
from importlib import metadata
from pathlib import Path


def write_location(build_root: Path) -> None:
    build_root = build_root.resolve()
    try:
        site = Path(metadata.distribution("rocm-sdk-core").locate_file(""))
    except metadata.PackageNotFoundError:
        # A subsequent SDK installation can recover without rebuilding native code.
        site = Path(sysconfig.get_paths()["purelib"])
    try:
        location = os.path.relpath(site, build_root)
        marker = "rocke-sdk-site-v1"
    except ValueError:
        # Cross-drive Windows sites require regeneration after the SDK moves.
        location = str(site)
        marker = "rocke-sdk-site-absolute-v1"
    build_root.mkdir(parents=True, exist_ok=True)
    (build_root / "rocke_sdk_site.txt").write_text(
        marker + "\n" + location + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    write_location(parser.parse_args().build_root.resolve())
