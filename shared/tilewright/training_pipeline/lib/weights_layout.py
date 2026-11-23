# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a rocm-libraries checkout keeps the shipped tilewright models.

The models of a Tensile library subtree ship in its weights directory
`<subtree>/<subtree>/` under tilewright's weights root `WEIGHTS_REL`
(`rel_dir`): the first part is the architecture whose library logic
(`LOGIC_REL`) hipBLASLt builds the subtree from, the subtree itself, as in
`gfx950/gfx950/`. The libraries of an architecture's other logic
directories (`gfx950_id75a3`, `gfx1250_96cu`, ...) load into the same
subtree, so their models are rows of that subtree's index: one model serves
every library file of a problem type. A directory holds the subtree's
`tilewright_index`, its models (`*.tilewright.bin`, each named after the logic
file of its library) and their blocklist and guards sidecars
(`<logic name>.tilewright.blocklist`, `<logic name>.tilewright.guards`).

hipBLASLt's build target `hipblaslt-tilewright-models` copies the models and
sidecars of a subtree's directory to `<build>/Tensile/library/<subtree>/`,
next to the subtree's Tensile libraries, where TensileLite looks them up.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

LOGIC_REL = Path(
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)
WEIGHTS_REL = Path("shared/tilewright/weights")


def rel_dir(subtree: str) -> Path:
    """The weights directory of library subtree `subtree`, relative to the
    root of a rocm-libraries checkout."""
    return WEIGHTS_REL / str(subtree) / str(subtree)


def shipped_dirs(repo: Path) -> List[Path]:
    """The weights directories of the checkout `repo`, sorted."""
    root = Path(repo) / WEIGHTS_REL
    return sorted(p for p in root.glob("*/*") if p.is_dir())
