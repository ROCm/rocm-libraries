# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Map an arch name to the compiler target whose ISA it executes.

The pipeline's `arch` config key names the architecture a run tunes for, as
hipBLASLt names it: the Tensile library subtree the bench device loads (the
name the runtime reports, e.g. `gfx950` or the stepping `gfx1250-strict`) and
the subtree its model deploys to by default.
A device name can carry target features (`gfx942:sramecc+:xnack-`, or
`gfx942-xnack+` in directory form). Steppings run their base target's ISA,
so anything keyed by ISA, such as the model's arch constants, must use
`canonical_arch(arch)`.

Which subtree a device loads can depend on the properties its KFD node reports
and on the runtime's mode (`reported_arch`): hipBLASLt loads the subtree of the
name the runtime reports (`kernel_subtree` names the subtree selected for a
device).
"""
from __future__ import annotations

from typing import Mapping, Optional

# compiler target -> the stepping name the runtime can report (STRICT_SWITCH).
STRICT_SWITCH = "HSA_DISABLE_GFX12_STRICT"
_STRICT = {"gfx1250": "gfx1250-strict"}


def canonical_arch(arch: str) -> str:
    """`gfx1250-strict` / `gfx942:xnack+` -> its compiler target (`gfx1250`,
    `gfx942`); a bare compiler target is returned unchanged."""
    if not arch:
        return arch
    name = str(arch).strip().split(":", 1)[0]
    base, sep, _ = name.partition("-")
    if sep and base.startswith("gfx"):
        name = base
    return name


def mode_dependent(arch: str) -> bool:
    """True when the subtree a device of `arch`'s compiler target loads
    depends on the runtime's mode."""
    return canonical_arch(arch) in _STRICT


def reported_arch(
    target: str, revision: Optional[int], environ: Mapping[str, str]
) -> str:
    """The name the runtime reports for a device of compiler target `target`
    and revision field `revision` of its KFD node (None: unknown) with the
    environment `environ`."""
    strict = _STRICT.get(target)
    if strict and revision == 0 and str(environ.get(STRICT_SWITCH, "")).strip() == "0":
        return strict
    return target


def kernel_subtree(target: str, revision: Optional[int]) -> str:
    """The Tensile library subtree selected for a device of compiler target
    `target` and KFD revision field `revision` (None: unknown)."""
    return reported_arch(target, revision, {STRICT_SWITCH: "0"})


def device_requirement(arch: str) -> Optional[str]:
    """What a run of the subtree `arch` names requires of its devices, for an
    arch whose subtree depends on them (`mode_dependent`); else None."""
    name = str(arch or "").strip().split(":", 1)[0]
    target = canonical_arch(name)
    strict = _STRICT.get(target)
    if strict is None:
        return None
    if name == strict:
        return (
            f"{name} needs {target} devices whose kernels are those of "
            f"{name}, with {STRICT_SWITCH}=0 in the environment of the run"
        )
    return f"{name} needs {target} devices whose kernels are those of {name}"
