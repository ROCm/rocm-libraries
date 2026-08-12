################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from typing import List, Optional

from . import Properties
from Tensile.Common.Architectures import \
    SUPPORTED_CHIP_ID_FALLBACKS, SUPPORTED_BUILD_CHIP_IDS, GFX_CHIP_IDS, \
    isaToGfx, supportsChipIdPredicate, print1

import copy
import re


def parseDeviceNameToHex(deviceName: Optional[str]) -> Optional[str]:
    """Parse 'Device 75a3' into 75a3 (hex).
    Args:
        deviceName: Of format 'Device XXXX'
    Returns:
        Hex-formatted chip ID string (without "id=" prefix), e.g. "75a3"
    Raise:
        SystemExit if format is invalid, enforces library logic at build time.
    """
    if deviceName is None:
        return None
    match = re.match(r'^Device\s+([0-9a-fA-F]+)$', deviceName.strip())
    if match:
        return match.group(1)
        #return int(match.group(1), 16)

    raise ValueError(f"Invalid device name format: '{deviceName}', expected 'Device XXXX'")


def _extractPciChipIds(pred: Optional[Properties.Predicate]) -> frozenset[int]:
    """Extract chip ID(s) from PciChipId or the 'Or' predicate.

    Returns:
        A frozenset of chip IDs. Empty set means no chip IDs were found.
    """
    if pred is None:
        return frozenset()
    if pred.tag == "PciChipId":
        return frozenset([pred.value])
    if pred.tag == "Or":
        ids = [p.value for p in pred.value if p.tag == "PciChipId"]
        return frozenset(ids)
    return frozenset()


def _buildChipIdFallbackGraph():
    """Build source->fallback-target graph using configured chip ID fallbacks."""
    graph = {}
    for sourceKey, fallbackKeys in SUPPORTED_CHIP_ID_FALLBACKS.items():
        sourceId = int(sourceKey.split("=", 1)[1], 16)
        sourceFallbacks = graph.setdefault(sourceId, set())
        for fallbackKey in fallbackKeys:
            fallbackId = int(fallbackKey.split("=", 1)[1], 16)
            sourceFallbacks.add(fallbackId)
            graph.setdefault(fallbackId, set())
    return graph


_CHIP_ID_FALLBACK_GRAPH = _buildChipIdFallbackGraph()
_CHIP_ID_TOPO_RANK_CACHE = {}


def _chipIdTopologicalRank(chipId: int, visiting=None):
    """Higher rank means farther away from fallback roots (more specific/source-like)."""
    if chipId in _CHIP_ID_TOPO_RANK_CACHE:
        return _CHIP_ID_TOPO_RANK_CACHE[chipId]

    if visiting is None:
        visiting = set()

    # Defensive cycle handling: break cycles by treating the back-edge as depth 0.
    if chipId in visiting:
        return 0

    visiting.add(chipId)
    fallbackIds = _CHIP_ID_FALLBACK_GRAPH.get(chipId, set())
    rank = 0 if not fallbackIds else 1 + max(_chipIdTopologicalRank(x, visiting) for x in fallbackIds)
    visiting.remove(chipId)

    _CHIP_ID_TOPO_RANK_CACHE[chipId] = rank
    return rank


def _chipIdSetSortKey(chipIds: frozenset):
    """Sort key for a chip-ID set using fallback-aware topological rank."""
    return tuple(
        sorted(((_chipIdTopologicalRank(chipId), chipId) for chipId in chipIds), reverse=True)
    )


class HardwarePredicate(Properties.Predicate):
    # TODO- And also FromISA() is hard to detect CU-fallback case.
    #       Perhaps we can always use FromHardware(). FromISA() is not used so far.
    @classmethod
    def FromISA(cls, isa):
        gfxArch = isaToGfx(tuple(isa))
        return cls("AMDGPU", value=cls("Processor", value=gfxArch))

    @classmethod
    def FromHardware(cls, isa, cuCount=None, deviceNames=None, logicFile=None):
        """Create a HardwarePredicate from hardware specifications.

        Args:
            isa: ISA tuple (e.g., (9, 5, 0) for gfx950)
            cuCount: Optional compute unit count
            deviceNames: Optional list of device name strings like ["Device 75a0", "Device 75a2"],
                         or a single string, or None
            logicFile: Optional source logic file path used only for diagnostics
        """
        gfxArch = isaToGfx(tuple(isa))
        props = [cls("Processor", value=gfxArch)]

        if cuCount is not None:
            props.append(cls("CUCount", value=cuCount))

        if supportsChipIdPredicate(gfxArch):
            pciChipIdPred = cls._createPciChipIdPredicate(deviceNames, gfxArch, logicFile)
            if pciChipIdPred is not None:
                props.append(pciChipIdPred)

        if len(props) == 1:
            return cls("AMDGPU", value=props[0])
        else:
            return cls("AMDGPU", value=cls.And(props))

    @classmethod
    def _createPciChipIdPredicate(cls, deviceNames, gfx: str, logicFile=None):
        """Create PciChipId predicate(s) from device names.

        Args:
            deviceNames: Can be:
                - None: returns None
                - Empty list []: returns None
                - Single string "Device XXXX": returns one PciChipId predicate
                - Single item list ["Device XXXX"]: returns one PciChipId predicate
                - Multiple item list: returns Or predicate with all PciChipId predicates
        Returns:
            A HardwarePredicate of type "PciChipId"
        """
        if isinstance(deviceNames, str):
            deviceNames = [deviceNames]

        supportedChipIds = cls._collectSupportedChipIds(deviceNames, gfx, logicFile)

        pciChipIds = []
        for chipId in supportedChipIds:
            if chipId is not None:
                # Serialize chip IDs as integers so YAML emits decimal numeric values.
                pciChipIds.append(int(chipId, 16) if isinstance(chipId, str) else int(chipId))

        if len(pciChipIds) == 0:
            return None
        if len(pciChipIds) == 1:
            return cls("PciChipId", value=pciChipIds[0])

        pciChipIdPredicates = [cls("PciChipId", value=chipId)
                               for chipId in pciChipIds]
        return cls.Or(pciChipIdPredicates)

    @classmethod
    def _collectSupportedChipIds(
        cls, deviceNames: Optional[List[str]], gfx: str, logicFile: Optional[str] = None
    ) -> List[str]:
        chipIdsRaw = [parseDeviceNameToHex(name) for name in deviceNames] if deviceNames else []
        chipIds = [chipId for chipId in chipIdsRaw if chipId is not None]
        expectedIds = GFX_CHIP_IDS.get(gfx, [])
        supportedChipIds = [chipId for chipId in chipIds if f"id={chipId.lower()}" in SUPPORTED_BUILD_CHIP_IDS]
        unsupportedChipIds = [chipId for chipId in chipIds if f"id={chipId.lower()}" not in SUPPORTED_BUILD_CHIP_IDS]

        if expectedIds and (not supportedChipIds or unsupportedChipIds):
            print1("")
            print1("********************************************************************************")
            print1("* WARNING: Logic file has invalid or unsupported chip IDs")
            print1(f"*   File: {logicFile if logicFile else '<unknown>'}")
            print1(f"*   Architecture: {gfx}")
            print1(f"*   Found chip IDs: {', '.join(chipIds) if chipIds else '<none>'}")
            print1(f"*   Supported chip IDs: {', '.join(expectedIds)}")

            if supportedChipIds:
                print1("* This logic file will still be used for the supported chip-ID variant(s)")
                print1(f"*   Using chip IDs: {', '.join(supportedChipIds)}")
                print1(f"*   Ignoring chip IDs: {', '.join(unsupportedChipIds)}")
            else:
                print1(f"* No supported chip IDs found; using this file as fallback for all {gfx} devices")
                print1(f"* For optimal kernel selection, specify IDs like: `[Device {expectedIds[0]}]`")

            print1("********************************************************************************")
            print1("")

        return supportedChipIds

    def __lt__(self, other):
        # Use superclass logic for TruePreds
        if other.tag == 'TruePred' or self.tag == 'TruePred':
            return super().__lt__(other)

        # Compute unit counts are embedded as 'And' with
        # 'Processor' and 'ComputeUnitCount' as children
        if self.value.tag == 'And':
            myAndPred = self.value
            myProcPred = next(
                iter(x for x in myAndPred.value if x.tag == "Processor"), None)
            myCUPred = next(
                iter(x for x in myAndPred.value if x.tag == "CUCount"), None)
            myPciChipIdPred = next(
                iter(x for x in myAndPred.value if x.tag in ("PciChipId", "Or")), None)
            myCUCount = myCUPred.value if myCUPred is not None else None
            myPciChipIds = _extractPciChipIds(myPciChipIdPred)
        else:
            myProcPred = self.value
            myCUCount = None
            myPciChipIds = frozenset()

        if other.value.tag == 'And':
            otherAndPred = other.value
            otherProcPred = next(
                iter(x for x in otherAndPred.value if x.tag == "Processor"), None)
            otherCUPred = next(
                iter(x for x in otherAndPred.value if x.tag == "CUCount"), None)
            otherPciChipIdPred = next(
                iter(x for x in otherAndPred.value if x.tag in ("PciChipId", "Or")), None)
            otherCUCount = otherCUPred.value if otherCUPred is not None else None
            otherPciChipIds = _extractPciChipIds(otherPciChipIdPred)
        else:
            otherProcPred = other.value
            otherCUCount = None
            otherPciChipIds = frozenset()

        # Prioritize ChipId (more specific match first)
        # A predicate with a chip ID set is more specific than one without
        if myPciChipIds and not otherPciChipIds:
            return True
        if not myPciChipIds and otherPciChipIds:
            return False
        if myPciChipIds and otherPciChipIds and myPciChipIds != otherPciChipIds:
            # Prefer source-like chip IDs (exact-capable rows) before fallback targets.
            # This ensures ordering like:
            #   Equality -> Origami/Predication -> Equality fallback -> Origami/Predication fallback.
            myChipKey = _chipIdSetSortKey(myPciChipIds)
            otherChipKey = _chipIdSetSortKey(otherPciChipIds)

            # Extract the highest rank from each set (first element of the sorted key)
            myMaxRank = myChipKey[0][0] if myChipKey else -1
            otherMaxRank = otherChipKey[0][0] if otherChipKey else -1

            # First, compare by maximum rank (higher rank = more specific = comes first)
            if myMaxRank != otherMaxRank:
                return myMaxRank > otherMaxRank

            # For same maximum rank, prefer smaller sets (exact chip matches) over multi-chip sets
            if len(myPciChipIds) != len(otherPciChipIds):
                return len(myPciChipIds) < len(otherPciChipIds)

            # For same rank and same size, compare by full chip key
            return myChipKey > otherChipKey

        # If CU properties are empty, then compare processor predicates
        if myCUCount is None and otherCUCount is None:
            # Make sure that we have valid processor preds
            assert myProcPred is not None and otherProcPred is not None, "Missing processor predicate"
            assert myProcPred.tag == otherProcPred.tag == "Processor", "Invalid processor predicate"

            # Downgrade to base class so that we don't recurse
            myProcPredCopy = copy.deepcopy(myProcPred)
            otherProcPredCopy = copy.deepcopy(otherProcPred)
            myProcPredCopy.__class__ = otherProcPredCopy.__class__ = Properties.Predicate
            return myProcPredCopy < otherProcPredCopy

        # Higher priority given to higher CU count (None treated as lowest priority)
        myCUVal = myCUCount if myCUCount is not None else 0
        otherCUVal = otherCUCount if otherCUCount is not None else 0
        return myCUVal > otherCUVal
