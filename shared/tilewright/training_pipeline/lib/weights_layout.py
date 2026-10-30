# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a rocm-libraries checkout keeps the shipped tilewright models.

hipBLASLt ships the models of a Tensile library subtree with the subtree's
library logic, in its `Tilewright` directory `<arch>/<subtree>/Tilewright/`
under `LOGIC_REL` (`rel_dir`). `<arch>` is the architecture whose logic
hipBLASLt builds the subtree from: the subtree itself, as in
`gfx950/gfx950/Tilewright/`, or for a build alias the architecture it builds
(`lib.arch.BUILD_ALIASES`): the `gfx1250v0` subtree, gfx1250-strict's logic
built for gfx1250, has `gfx1250-strict/gfx1250v0/Tilewright/`. The
libraries of an architecture's other logic directories (`gfx950_128cu`, ...)
load into the same subtree, so their models are rows of that subtree's
index; hipBLASLt installs no other Tilewright directory. A directory
holds the subtree's `tilewright_index`, its models (`*.tilewright.bin`) and
their blocklist and guards sidecars (`<stem>.tilewright.blocklist`,
`<stem>.tilewright.guards`), never a `.yaml` file: TensileCreateLibrary
reads every `.yaml` file of the logic tree as library logic, and
hipBLASLt's configuration fails on one there.

hipBLASLt's build target `hipblaslt-tilewright-models` copies each directory's
index, models and sidecars to `<build>/Tensile/library/<subtree>/`, next to
the subtree's Tensile libraries, where TensileLite looks them up.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .arch import build_alias_of

LOGIC_REL = Path(
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)
DIR_NAME = "Tilewright"


def logic_arch(subtree: str) -> str:
    """The architecture directory of `LOGIC_REL` that holds the logic
    hipBLASLt builds the library subtree `subtree` from."""
    return build_alias_of(subtree) or str(subtree)


def rel_dir(subtree: str) -> Path:
    """The `Tilewright` directory of library subtree `subtree`, relative to
    the root of a rocm-libraries checkout."""
    return LOGIC_REL / logic_arch(subtree) / str(subtree) / DIR_NAME


def shipped_dirs(repo: Path) -> List[Path]:
    """The `Tilewright` directories of the checkout `repo`, sorted."""
    root = Path(repo) / LOGIC_REL
    return sorted(p for p in root.glob(f"*/*/{DIR_NAME}") if p.is_dir())
