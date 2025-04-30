################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .TensileInstructions import DSStoreB8, DSStoreB8HID16, ReadWriteInstruction, \
                        DSLoadD16HIU16, DSLoadD16HIU8, \
                        DSLoadU8, DSLoadU16, DSStoreD16HIB16, \
                        DSStoreB16
from .Common import printExit

from dataclasses import dataclass, field
from typing import Type

################################################################################
# Memory Instruction
################################################################################
@dataclass
class MemoryInstruction:
    inst: Type[ReadWriteInstruction]
    numAddresses: int
    numOffsets: int
    offsetMultiplier: int
    blockWidth: float
    numBlocks: int = field(init=False)
    totalWidth: float = field(init=False)
    issueLatency: int = field(init=False)

    def __post_init__(self):
        self.numBlocks = 2 if self.numAddresses > 1 or self.numOffsets > 1 else 1
        self.totalWidth = self.blockWidth * self.numBlocks
        self.issueLatency = self.inst.issueLatency()

    def getInst(self, highBits=0):
        if highBits:
            if self.inst is DSLoadU8:
                return DSLoadD16HIU8
            elif self.inst is DSLoadU16:
                return DSLoadD16HIU16
            elif self.inst is DSStoreB16:
                return DSStoreD16HIB16
            elif self.inst is DSStoreB8:
                return DSStoreB8HID16
            else:
                printExit(str(self.inst) + " does not support high bits instructions.")

        return self.inst
