################################################################################
#
# Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
from . import Common
import copy

class HardwarePredicate(Properties.Predicate):
    @classmethod
    def FromISA(cls, isa):
        gfxArch = Common.gfxName(isa)
        return cls("AMDGPU", value=cls("Processor", value=gfxArch))

    @classmethod
    def FromHardware(cls, isa, cuCount=None, isAPU=None):
        gfxArch = Common.gfxName(isa)
        if cuCount == None and isAPU == None:
            return cls("AMDGPU", value=cls("Processor", value=gfxArch))
        elif cuCount == None:
            return cls("AMDGPU", value=cls.And([cls("Processor", value=gfxArch),
                                                cls("IsAPU", value=isAPU)]))
        elif isAPU == None:
            return cls("AMDGPU", value=cls.And([cls("Processor", value=gfxArch),
                                                cls("CUCount", value=cuCount)]))
        else:
            return cls("AMDGPU", value=cls.And([cls("Processor", value=gfxArch),
                                                cls("CUCount", value=cuCount),
                                                cls("IsAPU", value=isAPU)]))

    def __lt__(self, other):
        # Use superclass logic for TruePreds
        if other.tag == 'TruePred' or self.tag == 'TruePred':
            return super().__lt__(other)

        # Compute unit counts or APU/XPU versions are embedded as 'And' with
        # 'Processor', 'CUCount', and 'IsAPU' as children
        if self.value.tag == 'And':
            myAndPred = self.value
            myProcPred = next(iter(x for x in myAndPred.value if x.tag == "Processor"), None)
            myCUPred = next(iter(x for x in myAndPred.value if x.tag == "CUCount"), None)
            myCUCount = myCUPred.value if myCUPred != None else 0
            myIsAPUPred = next(iter(x for x in myAndPred.value if x.tag == "IsAPU"), None)
            myIsAPU = myIsAPUPred.value if myIsAPUPred != None else -1
        else:
            myProcPred = self.value
            myCUCount = 0
            myIsAPU = -1

        if other.value.tag == 'And':
            otherAndPred = other.value
            otherProcPred = next(iter(x for x in otherAndPred.value if x.tag == "Processor"), None)
            otherCUPred = next(iter(x for x in otherAndPred.value if x.tag == "CUCount"), None)
            otherCUCount = otherCUPred.value if otherCUPred != None else 0
            otherIsAPUPred = next(iter(x for x in otherAndPred.value if x.tag == "IsAPU"), None)
            otherIsAPU = otherIsAPUPred.value if otherIsAPUPred != None else -1
        else:
            otherProcPred = other.value
            otherCUCount = 0
            otherIsAPU = -1

        # If APU properties are the same, then check CU count or architecture
        if myIsAPU == otherIsAPU:
            # If CU properties are empty, then compare processor predicates
            if myCUCount == otherCUCount == 0:
                # Make sure that we have valid processor preds
                assert myProcPred != None and otherProcPred != None, "Missing processor predicate"
                assert myProcPred.tag == otherProcPred.tag == "Processor", "Invalid processor predicate"

                # Downgrade to base class so that we don't recurse
                myProcPredCopy = copy.deepcopy(myProcPred)
                otherProcPredCopy = copy.deepcopy(otherProcPred)
                myProcPredCopy.__class__ = otherProcPredCopy.__class__ = Properties.Predicate
                return myProcPredCopy < otherProcPredCopy

            # Higher priority given to higher CU count
            return myCUCount > otherCUCount
        # APU sorted before XPU, and XPU sorted before generic
        return myIsAPU > otherIsAPU
