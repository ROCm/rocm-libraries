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
    try:
        site = Path(metadata.distribution("rocm-sdk-core").locate_file(""))
    except metadata.PackageNotFoundError:
        # A subsequent SDK installation can recover without rebuilding native code.
        site = Path(sysconfig.get_paths()["purelib"])
    try:
        relative = os.path.relpath(site, build_root)
    except ValueError:
        # Different Windows drives cannot be described by a relocatable path.
        relative = ""
    build_root.mkdir(parents=True, exist_ok=True)
    (build_root / "rocke_sdk_site.txt").write_text(
        "rocke-sdk-site-v1\n" + relative + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    write_location(parser.parse_args().build_root.resolve())
