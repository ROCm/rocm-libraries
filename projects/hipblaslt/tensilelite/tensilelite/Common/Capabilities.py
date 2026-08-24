################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import rocisa

from typing import List, Dict

from .Architectures import (ARCH_CAP_OVERRIDES, baseArchName, gfxToIsa, tuningArchOf,
                           withoutPredicate)
from .Types import IsaVersion, IsaInfo


def _overrideLayers(name: str):
    """The two capability-override layers for a spec, in the order they apply.

    1. Base-name overrides (e.g. gfx1250v0's stepping deltas), keyed on the bare
       name so gfx1250v0 still picks up the right entry.
    2. Full-name overrides (e.g. gfx1250:xnack- clearing EnableXnackReplay),
       keyed on the qualified spec so the xnack suffix is visible. The predicate
       is dropped first, so gfx1250:xnack-[cu=64] finds the same entry as
       gfx1250:xnack- -- a hardware-selection predicate says nothing about caps.

    Layer 2 wins on any key present in both: the xnack mode is a stronger
    constraint than the stepping default.
    """
    base = ARCH_CAP_OVERRIDES.get(tuningArchOf(baseArchName(name)), {})
    full = ARCH_CAP_OVERRIDES.get(withoutPredicate(name), {})
    return base, full


def _effectiveOverrides(name: str) -> dict:
    """The capabilities a spec ends up with, both layers merged."""
    merged: dict = {}
    for layer in _overrideLayers(name):
        for section, deltas in layer.items():
            merged.setdefault(section, {}).update(deltas)
    return merged


def applyArchCapOverrides(isaInfoMap: Dict[IsaVersion, IsaInfo], archNames: List[str]) -> None:
    """Applies declared capability deltas for the requested architecture names.

    The capability map is keyed by ISA version, so architectures that share an
    ISA (gfx1250 and its stepping) collapse to one entry and the probe cannot
    tell them apart. ``ARCH_CAP_OVERRIDES`` declares their differences and this
    applies them in place.

    Every entry point that knows the requested names runs this immediately after
    ``makeIsaInfoMap``; the capability map is the only channel a stepping travels
    through, so skipping it builds one stepping with the other's capabilities.

    Names without declared deltas are ignored, so callers can pass the build's
    whole requested-arch list.

    This rewrites the Python-side capability map only. rocisa's own tables are
    unaffected, which is why the overridden keys must be ones nothing below the
    Python boundary reads.

    Args:
        isaInfoMap: A map of ISA versions to capabilities, modified in place.
        archNames: The gfx names requested for this build.

    Raises:
        ValueError: If two requested names share an ISA but not their deltas, or
            if a name with declared deltas has no entry to apply them to.
    """
    _rejectConflictingArchNames(archNames)
    for name in archNames:
        base_overrides, full_overrides = _overrideLayers(name)
        if not base_overrides and not full_overrides:
            continue
        isa = gfxToIsa(name)
        info = isaInfoMap.get(isa)
        if info is None:
            # Skipping would leave the map at the values probed for the shared
            # ISA, i.e. build this architecture with the other's capabilities --
            # the outcome this function exists to prevent.
            raise ValueError(
                f"Architecture {name} declares capability overrides but ISA "
                f"{tuple(isa) if isa else isa} is absent from the capability map; "
                "the requested architectures and the probed ISAs disagree."
            )
        info.asmCaps.update(base_overrides.get("asmCaps", {}))
        info.asmCaps.update(full_overrides.get("asmCaps", {}))
        info.archCaps.update(base_overrides.get("archCaps", {}))
        info.archCaps.update(full_overrides.get("archCaps", {}))


def _rejectConflictingArchNames(archNames: List[str]) -> None:
    """Rejects requesting two architectures that share an ISA but not their caps.

    The capability map is keyed by ISA version, so it can only describe one of
    them; whichever override applied last would silently win for both.

    Compared by the capabilities each spec effectively gets, not by name. Two
    specs of one architecture are only in conflict when they end up wanting
    different caps -- gfx1250 beside gfx1250:xnack- is one architecture under
    two xnack modes, and the map holds one of the two, so that pairing has to be
    refused rather than let the qualified one quietly disarm the bare one. The
    predicate is dropped first, so it can neither hide a conflict nor invent one.
    """
    namesByIsa: Dict[IsaVersion, List[str]] = {}
    for name in archNames:
        isa = gfxToIsa(name)
        spec = withoutPredicate(name)
        if isa is not None and spec not in namesByIsa.setdefault(isa, []):
            namesByIsa[isa].append(spec)

    for isa, names in namesByIsa.items():
        first = _effectiveOverrides(names[0])
        conflicting = [n for n in names[1:] if _effectiveOverrides(n) != first]
        if conflicting:
            raise ValueError(
                f"Architectures {names} share ISA {tuple(isa)} but declare different "
                "capabilities, so a single build cannot target them together; "
                "build each one separately."
            )


def makeIsaInfoMap(targetIsas: List[IsaVersion], cxxCompiler: str) -> Dict[IsaVersion, IsaInfo]:
    """Computes the supported capabilities for requested ISAs and compiler.

    Given a list of ISAs and a compiler, the ASM, Arch, Register capabilities
    and ASM bugs are computed and stored in a map.

    Capabilities that the compiler cannot be probed for, because they differ
    between architectures sharing one ISA, are layered on afterwards by
    ``applyArchCapOverrides``.

    Args:
        targetIsas: A list of requested ISA versions to inspect.
        cxxCompiler: A string path to a C++ compiler to use when computing capabilities.

    Returns:
        A map of ISA versions to capabilities.
    """
    isaInfoMap = {}
    ti = rocisa.rocIsa.getInstance()
    # Probing is hundreds of serial assembler runs per ISA; do the ISAs concurrently.
    ti.initMany(targetIsas, cxxCompiler, False)
    for v in targetIsas:
        asmCaps = ti.getIsaInfo(v).asmCaps
        archCaps = ti.getIsaInfo(v).archCaps
        regCaps = ti.getIsaInfo(v).regCaps
        asmBugs = ti.getIsaInfo(v).asmBugs
        isaInfoMap[v] = IsaInfo(asmCaps, archCaps, regCaps, asmBugs)

    return isaInfoMap
