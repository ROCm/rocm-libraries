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

from .Architectures import ARCH_CAP_OVERRIDES, baseArchName, gfxToIsa
from .Types import IsaVersion, IsaInfo


def applyArchCapOverrides(isaInfoMap: Dict[IsaVersion, IsaInfo], archNames: List[str]) -> None:
    """Applies declared capability deltas for the requested architecture names.

    Capabilities are probed by assembling at the architecture's compiler target,
    so architectures that share a target (gfx1250's two steppings) are
    indistinguishable to the probe. ``ARCH_CAP_OVERRIDES`` declares their
    differences and this applies them in place.

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
        # Keyed on the bare name: --gpu-targets forwards a requested spec verbatim,
        # predicates and all, and a lookup that missed gfx1250v0[cu=64] would build
        # v0 with the shipping stepping's capabilities without saying so.
        overrides = ARCH_CAP_OVERRIDES.get(baseArchName(name))
        if not overrides:
            continue
        isa = gfxToIsa(name)
        info = isaInfoMap.get(isa)
        if info is None:
            # Skipping would leave the map at the values probed for the shared
            # compiler target, i.e. build this architecture with the other's
            # capabilities -- the outcome this function exists to prevent.
            raise ValueError(
                f"Architecture {name} declares capability overrides but ISA "
                f"{tuple(isa) if isa else isa} is absent from the capability map; "
                "the requested architectures and the probed ISAs disagree."
            )
        info.asmCaps.update(overrides.get("asmCaps", {}))
        info.archCaps.update(overrides.get("archCaps", {}))


def _rejectConflictingArchNames(archNames: List[str]) -> None:
    """Rejects requesting two architectures that share an ISA but not their caps.

    The capability map is keyed by ISA version, so it can only describe one of
    them; whichever override applied last would silently win for both.

    Compared as bare names, so that a predicate cannot hide the conflict and two
    specs of one architecture are not mistaken for two architectures.
    """
    namesByIsa: Dict[IsaVersion, List[str]] = {}
    for name in archNames:
        isa = gfxToIsa(name)
        base = baseArchName(name)
        if isa is not None and base not in namesByIsa.setdefault(isa, []):
            namesByIsa[isa].append(base)

    for isa, names in namesByIsa.items():
        first = ARCH_CAP_OVERRIDES.get(names[0])
        conflicting = [n for n in names[1:] if ARCH_CAP_OVERRIDES.get(n) != first]
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
    between architectures sharing one target, are layered on afterwards by
    ``applyArchCapOverrides``.

    Args:
        targetIsas: A list of requested ISA versions to inspect.
        cxxCompiler: A string path to a C++ compiler to use when computing capabilities.

    Returns:
        A map of ISA versions to capabilities.
    """
    isaInfoMap = {}
    ti = rocisa.rocIsa.getInstance()
    for v in targetIsas:
        ti.init(v, cxxCompiler, False)
        asmCaps = ti.getIsaInfo(v).asmCaps
        archCaps = ti.getIsaInfo(v).archCaps
        regCaps = ti.getIsaInfo(v).regCaps
        asmBugs = ti.getIsaInfo(v).asmBugs
        isaInfoMap[v] = IsaInfo(asmCaps, archCaps, regCaps, asmBugs)

    return isaInfoMap
