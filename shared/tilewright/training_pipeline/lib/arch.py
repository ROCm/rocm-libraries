# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Map an arch name to the compiler target whose ISA it executes.

The pipeline's `arch` config key names the architecture a run tunes for, as
hipBLASLt names it: the Tensile library subtree the bench device loads (the
name the runtime reports, e.g. `gfx950` or the stepping `gfx1250-strict`) and
the subtree its model deploys to by default. hipBLASLt also builds subtrees
under build aliases (`BUILD_ALIASES`): another architecture's logic built for
an existing compiler target. An alias is a deploy target, never an arch.
A device name can carry target features (`gfx942:sramecc+:xnack-`, or
`gfx942-xnack+` in directory form). Steppings and aliases run their base
target's ISA, so anything keyed by ISA, such as the model's arch constants,
must use `canonical_arch(arch)`.

Which subtree a device loads can depend on its ASIC revision and the
runtime's mode (`reported_arch`, `library_subtree`): a gfx1250 device of
revision 0 reports gfx1250-strict only with HSA_DISABLE_GFX12_STRICT=0
(`STRICT_SWITCH`) in its environment; otherwise it reports gfx1250 and
hipBLASLt loads its gfx1250v0 subtree.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional

# Anchored on a trailing `v<digits>` only, so targets that merely end in a
# digit (`gfx950`, `gfx1201`) are left alone.
_REVISION_SUFFIX_RE = re.compile(r"^(gfx[0-9a-f]+?)v\d+$")

# TensileLite's ARCH_BUILD_ALIASES: alias -> the architecture whose logic and
# capabilities it builds for the alias's base target, under that target's
# file names.
BUILD_ALIASES = {"gfx1250v0": "gfx1250-strict"}

# hipBLASLt's rocblaslt_arch_revision.hpp: compiler target -> (the name the
# runtime reports for a device of ASIC revision 0 with STRICT_SWITCH=0, the
# subtree hipBLASLt loads for such a device reporting the target itself).
STRICT_SWITCH = "HSA_DISABLE_GFX12_STRICT"
_REVISION_ZERO = {"gfx1250": ("gfx1250-strict", "gfx1250v0")}


def canonical_arch(arch: str) -> str:
    """`gfx1250-strict` / `gfx1250v0` / `gfx942:xnack+` -> its compiler
    target (`gfx1250`, `gfx942`); a bare compiler target is returned
    unchanged."""
    if not arch:
        return arch
    name = str(arch).strip().split(":", 1)[0]
    base, sep, _ = name.partition("-")
    if sep and base.startswith("gfx"):
        name = base
    m = _REVISION_SUFFIX_RE.match(name)
    return m.group(1) if m else name


def build_alias_of(arch: str) -> Optional[str]:
    """The architecture whose logic the build alias `arch` holds (any
    spelling with target features), or None when `arch` is no alias."""
    name = str(arch or "").strip().split(":", 1)[0]
    return BUILD_ALIASES.get(name)


def mode_dependent(arch: str) -> bool:
    """True when the subtree a device of `arch`'s compiler target loads
    depends on its ASIC revision and the runtime's mode."""
    return canonical_arch(arch) in _REVISION_ZERO


def reported_arch(
    target: str, revision: Optional[int], environ: Mapping[str, str]
) -> str:
    """The name the runtime reports for a device of compiler target `target`
    and ASIC revision `revision` (None: unknown) with the environment
    `environ`."""
    strict = _REVISION_ZERO.get(target, (None, None))[0]
    if strict and revision == 0 and str(environ.get(STRICT_SWITCH, "")).strip() == "0":
        return strict
    return target


def library_subtree(reported: str, revision: Optional[int]) -> str:
    """The Tensile library subtree hipBLASLt loads for a device that reports
    `reported` and has ASIC revision `revision` (None: unknown)."""
    if revision == 0 and reported in _REVISION_ZERO:
        return _REVISION_ZERO[reported][1]
    return reported


def device_requirement(arch: str) -> Optional[str]:
    """The device and mode that load the subtree `arch` names, for an arch
    whose subtree depends on them (`mode_dependent`); else None."""
    name = str(arch or "").strip().split(":", 1)[0]
    target = canonical_arch(name)
    if target not in _REVISION_ZERO:
        return None
    strict, alias = _REVISION_ZERO[target]
    if name == strict:
        return (
            f"{name} needs a {target} device of ASIC revision 0 and "
            f"{STRICT_SWITCH}=0 in the environment of the run"
        )
    return (
        f"{name} needs a {target} device of an ASIC revision other than 0 (a "
        f"revision 0 device loads {strict} with {STRICT_SWITCH}=0, else {alias})"
    )
