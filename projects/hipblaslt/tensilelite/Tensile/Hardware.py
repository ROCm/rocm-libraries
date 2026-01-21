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

from . import Properties
from Tensile.Common.Utilities import printExit
from Tensile.Common.Architectures import isaToGfx
import copy
import re
import sys


def parseDeviceNameToInt(deviceName: str):
    """Parse 'Device 75a3' into 30115 (hex to decimal).
    Args:
        deviceName: Of format 'Device 7aa7`
    Returns:
        Decimal formated integer to be serialized into solution library
    Raise:
        SystemExit if format is invalid, enforces library logic at build time.
    """
    if deviceName is None:
        return None
    match = re.match(r'^Device\s+([0-9a-fA-F]+)$', deviceName.strip())
    if match:
        return int(match.group(1), 16)

    raise ValueError(f"Invalid device name format: '{
                     deviceName}', expected 'Device XXXX'")


def _extractPciChipIDs(pred: Properties.Predicate):
    """Extract chip ID(s) from PciChipID or the 'Or' predicate.

    Returns:
        - Single int if pred is PciChipID
        - First int from list if pred is Or with PciChipID children (TODO: isn't this a problem since I should be able to match against multiple chip IDs)
        - None if pred is None or has no PciChipID values
    """
    if pred is None:
        return None
    if pred.tag == "PciChipID":
        return pred.value
    if pred.tag == "Or":
        ids = [p.value for p in pred.value if p.tag == "PciChipID"]
        return ids[0] if ids else None
    return None


class HardwarePredicate(Properties.Predicate):
    # TODO- And also FromISA() is hard to detect CU-fallback case.
    #       Perhaps we can always use FromHardware(). FromISA() is not used so far.
    @classmethod
    def FromISA(cls, isa):
        gfxArch = isaToGfx(tuple(isa))
        return cls("AMDGPU", value=cls("Processor", value=gfxArch))

    @classmethod
    def FromHardware(cls, isa, cuCount=None, deviceNames=None):
        """Create a HardwarePredicate from hardware specifications.

        Args:
            isa: ISA tuple (e.g., (9, 5, 0) for gfx950)
            cuCount: Optional compute unit count
            deviceNames: Optional list of device name strings like ["Device 75a0", "Device 75a2"],
                         or a single string, or None
        """
        gfxArch = isaToGfx(tuple(isa))
        props = [cls("Processor", value=gfxArch)]
        if cuCount is not None:
            props.append(cls("CUCount", value=cuCount))

        pciChipIdPred = cls._createPciChipIdPredicate(deviceNames)
        if pciChipIdPred is not None:
            props.append(pciChipIdPred)

        if len(props) == 1:
            return cls("AMDGPU", value=props[0])
        else:
            return cls("AMDGPU", value=cls.And(props))

    @classmethod
    def _createPciChipIdPredicate(cls, deviceNames):
        """Create PciChipID predicate(s) from device names.

        Args:
            deviceNames: Can be:
                - None: returns None
                - Empty list []: returns None
                - Single string "Device XXXX": returns one PciChipID predicate
                - Single item list ["Device XXXX"]: returns one PciChipID predicate
                - Multiple item list: returns Or predicate with all PciChipID predicates
        Returns:
            A HardwarePredicate of type "PciChipID"
        """
        if deviceNames is None:
            return None
        if isinstance(deviceNames, str):
            deviceNames = [deviceNames]

        pciChipIds = []
        for name in deviceNames:
            chipId = parseDeviceNameToInt(name)
            if chipId is not None:
                pciChipIds.append(chipId)

        if len(pciChipIds) == 0:
            return None
        if len(pciChipIds) == 1:
            return cls("PciChipID", value=pciChipIds[0])

        pciChipIdPredicates = [cls("PciChipID", value=chipId)
                               for chipId in pciChipIds]
        return cls.Or(pciChipIdPredicates)

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
            myPciChipIDPred = next(
                iter(x for x in myAndPred.value if x.tag in ("PciChipID", "Or")), None)
            myCUCount = myCUPred.value if myCUPred is not None else None
            myPciChipID = _extractPciChipIDs(myPciChipIDPred)
        else:
            myProcPred = self.value
            myCUCount = None
            myPciChipID = None

        if other.value.tag == 'And':
            otherAndPred = other.value
            otherProcPred = next(
                iter(x for x in otherAndPred.value if x.tag == "Processor"), None)
            otherCUPred = next(
                iter(x for x in otherAndPred.value if x.tag == "CUCount"), None)
            otherPciChipIDPred = next(
                iter(x for x in otherAndPred.value if x.tag in ("PciChipID", "Or")), None)
            otherCUCount = otherCUPred.value if otherCUPred is not None else None
            otherPciChipID = _extractPciChipIDs(otherPciChipIDPred)
        else:
            otherProcPred = other.value
            otherCUCount = None
            otherPciChipID = None

        # Prioritize ChipID (more specific match first)
        # A predicate with a chip ID set is more specific than one without
        if myPciChipID is not None and otherPciChipID is None:
            return True
        if myPciChipID is None and otherPciChipID is not None:
            return False
        if myPciChipID is not None and otherPciChipID is not None and myPciChipID != otherPciChipID:
            return myPciChipID < otherPciChipID

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
