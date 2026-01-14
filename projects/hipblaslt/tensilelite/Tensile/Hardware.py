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
from Tensile.Common.Architectures import isaToGfx
import copy

class HardwarePredicate(Properties.Predicate):
    # TODO- And also FromISA() is hard to detect CU-fallback case.
    #       Perhaps we can always use FromHardware(). FromISA() is not used so far.
    @classmethod
    def FromISA(cls, isa):
        gfxArch = isaToGfx(tuple(isa))
        return cls("AMDGPU", value=cls("Processor", value=gfxArch))

    @classmethod
    def FromHardware(cls, isa, cuCount=None, pciChipId=None):
        gfxArch = isaToGfx(tuple(isa))
        props = [cls("Processor", value=gfxArch)]
        if cuCount != None:
            props.append(cls("CUCount", value=cuCount))
        if pciChipId != None:
            props.append(cls("PciChipID", value=pciChipId))

        if len(props) == 1:
            return cls("AMDGPU", value=props[0])
        else:
            return cls("AMDGPU", value=cls.And(props))

    def __lt__(self, other):
        # Use superclass logic for TruePreds
        if other.tag == 'TruePred' or self.tag == 'TruePred':
            return super().__lt__(other)

        # Compute unit counts are embedded as 'And' with
        # 'Processor' and 'ComputeUnitCount' as children
        if self.value.tag == 'And':
            myAndPred = self.value
            myProcPred = next(iter(x for x in myAndPred.value if x.tag == "Processor"), None)
            myCUPred = next(iter(x for x in myAndPred.value if x.tag == "CUCount"), None)
            myPciChipIDPred = next(iter(x for x in myAndPred.value if x.tag == "PciChipID"), None)
            myCUCount = myCUPred.value if myCUPred is not None else None
            myPciChipID = myPciChipIDPred.value if myPciChipIDPred is not None else None
        else:
            myProcPred = self.value
            myCUCount = None
            myPciChipID = None

        if other.value.tag == 'And':
            otherAndPred = other.value
            otherProcPred = next(iter(x for x in otherAndPred.value if x.tag == "Processor"), None)
            otherCUPred = next(iter(x for x in otherAndPred.value if x.tag == "CUCount"), None)
            otherPciChipIDPred = next(iter(x for x in otherAndPred.value if x.tag == "PciChipID"), None)
            otherCUCount = otherCUPred.value if otherCUPred is not None else None
            otherPciChipID = otherPciChipIDPred.value if otherPciChipIDPred is not None else None
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
