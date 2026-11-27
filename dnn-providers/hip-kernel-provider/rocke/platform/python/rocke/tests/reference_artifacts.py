# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: the GPU reference variants a rocKE library tree publishes.

The library's artifact CLI (``library/tests/reference_common/artifact.py``)
lists them from the committed baseline locks, and the provider build registers
one ``rocke_<operation>_gpu_<arch>_pytest`` entry per published operation and
architecture. Tests ask the CLI rather than importing it, since the platform
does not import the library::

    for variant in published_reference_variants(rocke_dir / "library"):
        variant["operation"], variant["architecture"]

The CLI exists only in a source tree; where it is absent (the installed test
layout, a platform-only copy) the helper skips the test, as waiting for the
source-tree lane.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .lanes import SOURCE_TREE, skip_until

__all__ = ["ARTIFACT_CLI", "published_reference_variants"]

#: The artifact CLI, relative to a library directory.
ARTIFACT_CLI = Path("tests/reference_common/artifact.py")


def published_reference_variants(library: Path) -> list[dict[str, str]]:
    """The variants the artifact CLI under ``library`` lists as published,
    each with its ``operation``, ``architecture``, ``variant``, ``lock``,
    ``pointer`` and ``archive``; skips the calling test when ``library`` has
    no artifact CLI."""
    cli = library / ARTIFACT_CLI
    if not cli.is_file():
        skip_until(SOURCE_TREE, reason=f"{cli} is absent from this tree")
    result = subprocess.run(
        [sys.executable, "-I", str(cli), "published", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)
