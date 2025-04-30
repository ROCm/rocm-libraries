################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

def getGlcBitName(memoryModifierFormat):
  if memoryModifierFormat == "GLC":
    return "glc"
  return "sc0"

def getSlcBitName(memoryModifierFormat):
  if memoryModifierFormat == "GLC":
    return "slc"
  return "sc1"

################################################################################
# Memory Instruction
################################################################################
class MemoryInstruction:
  def __init__(self, name, numAddresses, numOffsets, \
      offsetMultiplier, blockWidth, formatting, memoryModifierFormat, forceSC1=False):
    self.name = name
    self.formatting = formatting
    self.numAddresses = numAddresses
    self.numOffsets = numOffsets
    self.offsetMultiplier = offsetMultiplier
    self.blockWidth = blockWidth
    self.memoryModifierFormat = memoryModifierFormat
    self.numBlocks = 2 if self.numAddresses > 1 or self.numOffsets > 1 else 1
    self.totalWidth = self.blockWidth * self.numBlocks
    self.forceSC1 = forceSC1
    #in Quad-Cycle
    if (name == "_ds_load_b128"):
      self.IssueLatency = 2
    elif (name == "_ds_store_b128"):
      self.IssueLatency = 5
    elif (name == "_ds_store2_b64"):
      self.IssueLatency = 3
    elif (name == "_ds_store_b64"):
      self.IssueLatency = 3
    elif (name == "_ds_store2_b32"):
      self.IssueLatency = 3
    elif (name == "_ds_store_b32"):
      self.IssueLatency = 2
    elif (name == "_ds_store_u16") :
      self.IssueLatency = 2
    else:
      self.IssueLatency = 1
    self.endLine = "\n"

  ########################################

  # write in assembly format
  def toString(self, params, comment, nonTemporal=0, highBits=0):
    name = self.name
    if highBits:
      name += "_d16_hi"
    instStr = "%s %s" % (name, (self.formatting % params) )
    if (nonTemporal & 1) != 0 or self.forceSC1:
      instStr += " " + getGlcBitName(self.memoryModifierFormat)
    if (nonTemporal & 2) != 0 or self.forceSC1:
      instStr += " " + getSlcBitName(self.memoryModifierFormat)
    if (nonTemporal & 4) != 0:
      instStr += " nt"
    line = "%-50s // %s%s" % (instStr, comment, self.endLine)
    return line

  # Like toString, but don't add a comment or newline
  # Designed to feed into Code.Inst constructors, somewhat
  def toCodeInst(self, params, nonTemporal=0, highBits=0):
    name = self.name
    if highBits:
      name += "_d16_hi"
    instStr = "%s %s" % (name, (self.formatting % params) )
    if nonTemporal%2==1 or self.forceSC1:
      instStr += " " + getGlcBitName(self.memoryModifierFormat)
    if (nonTemporal//2)%2==1 or self.forceSC1:
      instStr += " " + getSlcBitName(self.memoryModifierFormat)
    if (nonTemporal//4)%2==1:
      instStr += " nt"
    line = "%-50s" % (instStr)
    return line


  def __str__(self):
    return self.name
