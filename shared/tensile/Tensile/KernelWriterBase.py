################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from abc  import ABC
from abc  import abstractmethod
from copy import deepcopy

class KernelWriterBase(ABC):

  def __init__(self):
    super().__init__()

    self.state = {}

    self.endLine = "\n"
    self.getGroupIdStr = "hc_get_group_id"
    self.getNumGroupsStr = "hc_get_num_groups"
    self.getLocalIdStr = "hc_get_workitem_id"
    self.getGlobalIdStr = "hc_get_workitem_absolute_id"
    self.sharedDeclStr = "__shared__ "
    self.sharedPtrStr = ""
    self.globalPtrStr = ""
    self.syncStr = "__syncthreads();"
    self.fenceStr = self.syncStr
    self.macFStr = "fmaf"
    self.macDStr = "fma"
    self.int64Str = "int64_t"
    self.uint64Str = "uint64_t"
    self.vectorComponents = ["x", "y", "z", "w"]
    self.atomicCasStr = "atomicCAS"
    self.volatileStr = ""
    self.deviceFunctionStr = "__device__ "


  def extractIndices(self, extractFrom, varPrefix, indices):
    kStr = ""
    for (i, index) in enumerate(indices):
      kStr += "  unsigned int " + varPrefix + self.indexChars[index] + " = ( " + extractFrom
      for j in reversed(list(range(i+1, len(indices)))):
        index2 = indices[j]
        kStr += " / size" + self.indexChars[index2]
      kStr += ")"

      if len(indices) > 1:
        kStr += " % size" + self.indexChars[index]
      kStr += ";" + self.endLine

    return kStr


  @abstractmethod
  def getKernelName(self):
    pass


  @abstractmethod
  def getHeaderFileString(self):
    pass


  @abstractmethod
  def getSourceFileString(self):
    pass


  ##########################
  # make class look like dict
  def keys(self):
    return list(self.state.keys())

  def __len__(self):
    return len(self.state)

  def __iter__(self):
    return iter(self.state)

  def __getitem__(self, key):
    return self.state[key]

  def __setitem__(self, key, value):
    self.state[key] = value

  def __str__(self):
    return self.getKernelName()

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return deepcopy(self.state)

  def __hash__(self):
    return hash(str(self))

  def __eq__(self, other):
    return isinstance(other, KernelWriterBase) and str(self) == str(other)

  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result
