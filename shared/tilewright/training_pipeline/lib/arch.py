# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Map an arch name to the compiler target whose ISA it executes.

The pipeline's `arch` config key names the Tensile library subtree a run
benches against and deploys to. That name can be an ASIC revision
(`gfx1250v0`) or a stepping (`gfx1250-strict`), and a device name can carry
target features (`gfx942:sramecc+:xnack-`, or `gfx942-xnack+` in directory
form). Revisions and steppings run their base target's ISA, so anything keyed
by ISA, such as the model's arch constants, must use `canonical_arch(arch)`.
"""
from __future__ import annotations

import re

# Anchored on a trailing `v<digits>` only, so targets that merely end in a
# digit (`gfx950`, `gfx1201`) are left alone.
_REVISION_SUFFIX_RE = re.compile(r"^(gfx[0-9a-f]+?)v\d+$")


def canonical_arch(arch: str) -> str:
    """`gfx1250v0` / `gfx1250-strict` / `gfx1250v0:xnack+` -> `gfx1250`;
    a bare compiler target is returned unchanged."""
    if not arch:
        return arch
    name = str(arch).strip().split(":", 1)[0]
    base, sep, _ = name.partition("-")
    if sep and base.startswith("gfx"):
        name = base
    m = _REVISION_SUFFIX_RE.match(name)
    return m.group(1) if m else name
