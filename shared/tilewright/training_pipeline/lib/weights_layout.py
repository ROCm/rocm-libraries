# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a rocm-libraries checkout keeps the shipped tilewright models.

The models of a Tensile library subtree ship in its weights directory
`<arch>/<subtree>/` under tilewright's weights root `WEIGHTS_REL`
(`rel_dir`). `<arch>` is the architecture whose library logic (`LOGIC_REL`)
hipBLASLt builds the subtree from: the subtree itself, as in
`gfx950/gfx950/`, or for a build alias the architecture it builds
(`lib.arch.BUILD_ALIASES`): the `gfx1250v0` subtree, gfx1250-strict's logic
built for gfx1250, has `gfx1250-strict/gfx1250v0/`. The libraries of an
architecture's other logic directories (`gfx950_128cu`, ...) load into the
same subtree, so their models are rows of that subtree's index; a CU-count
directory `<arch>_<N>cu` (`gfx1250_96cu`) can instead ship them in a weights
directory of its own, `<arch>/<arch>_<N>cu/`, whose models hipBLASLt
installs into the architecture's subtree as well (`installed_subtree`). It
installs no other weights directory. A directory holds the subtree's
`tilewright_index`, its models (`*.tilewright.bin`) and their blocklist and
guards sidecars (`<stem>.tilewright.blocklist`, `<stem>.tilewright.guards`).

hipBLASLt's build target `hipblaslt-tilewright-models` copies the models and
sidecars of the directories of a subtree to `<build>/Tensile/library/<subtree>/`,
next to the subtree's Tensile libraries, where TensileLite looks them up, with
one index that holds the rows of each directory's index; a library stem that
two of them list, or a file that two of them ship, fails the build
(`installed_with`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from .arch import build_alias_of

LOGIC_REL = Path(
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)
WEIGHTS_REL = Path("shared/tilewright/weights")

# hipBLASLt's logic directory of an architecture's devices of N CUs.
_CU_DIR_RE = re.compile(r"^(.+)_[0-9]+cu$")


def cu_dir_arch(subtree: str) -> Optional[str]:
    """The architecture of the CU-count logic directory `subtree`
    (`gfx1250_96cu` -> `gfx1250`), None for any other name."""
    m = _CU_DIR_RE.match(str(subtree))
    return m.group(1) if m else None


def logic_arch(subtree: str) -> str:
    """The architecture directory of `LOGIC_REL` that holds the logic
    hipBLASLt builds the library subtree `subtree` from, and of
    `WEIGHTS_REL` that holds the subtree's weights directory."""
    return cu_dir_arch(subtree) or build_alias_of(subtree) or str(subtree)


def rel_dir(subtree: str) -> Path:
    """The weights directory of library subtree `subtree`, relative to the
    root of a rocm-libraries checkout."""
    return WEIGHTS_REL / logic_arch(subtree) / str(subtree)


def installed_subtree(subtree: str) -> str:
    """The library subtree into whose directory of a build hipBLASLt
    installs the models of `subtree`'s weights directory: the
    architecture of a CU-count directory, whose logic builds into it, else
    `subtree`."""
    return cu_dir_arch(subtree) or str(subtree)


def shipped_dirs(repo: Path) -> List[Path]:
    """The weights directories of the checkout `repo`, sorted."""
    root = Path(repo) / WEIGHTS_REL
    return sorted(p for p in root.glob("*/*") if p.is_dir())


def installed_with(repo: Path, subtree: str) -> List[Path]:
    """The weights directories of the checkout `repo`, other than
    `subtree`'s, whose models hipBLASLt installs into the library directory
    `subtree`'s go to, sorted."""
    into = installed_subtree(subtree)
    own = Path(repo) / rel_dir(subtree)
    out = []
    for d in shipped_dirs(repo):
        sub = d.name
        installed = d == Path(repo) / rel_dir(sub)
        if installed and d != own and installed_subtree(sub) == into:
            out.append(d)
    return out
