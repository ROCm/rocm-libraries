################################################################################
#
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import tPrint, printExit, printWarning
from .Utils import roundUpToNearestMultiple
from .AsmUtils import inst, vgpr, sgpr
import traceback
from enum import Enum
from typing import Optional

################################################################################
# RegisterPool
# Debugging register performance problems:
# - Enable self.db["PrintRP"] to see messages as vgprPool state changes.
# - Search for 'overlow' to see when pool grows dynamically - typically this
#   indicates growth for temps or other cases.
# - checkIn, checkout take optional tag but this is not widely used in tensile.
# - checkout returns vgpr index that was returned - can search disasm to see where
#   this vgpr is used.
################################################################################
class RegisterPool:
  class Status(Enum):
    Unavailable = 0
    Available = 1
    InUse = 2
    AvailableForPreload = 3

  class Register:
    def __init__(self, status, tag):
      self.status = status
      self.tag = tag

    def __repr__(self) -> str:
      return f"({self.status}, {self.tag})"

  ########################################
  # Init
  # defaultPreventOverflow: control behavior of checkout and checkoutAligned when preventOverflow is not explicitly specificed.
  def __init__(self, size, type, defaultPreventOverflow, printRP=0):
    self.printRP=printRP
    self.type = type
    self.defaultPreventOverflow = defaultPreventOverflow
    self.pool = [self.Register(RegisterPool.Status.Unavailable, "init") for i in range(0,size)]

    # Map from register number -> size, deleted/overwritten when deallocated
    self.checkOutSize = {}

    # Map from tag -> size, not deleted.
    self.checkOutSizeCache = {}

    # First register that's part of the manually loaded kernargs
    self.kernargStart = None
    # Last+1 register that's part of the manually loaded kernargs
    self.kernargEnd = None

    # First register that's part of the preloaded kernargs
    self.preloadStart = None
    # Last+1 register that's part of the preloaded kernargs
    self.preloadEnd = None

    # Each of the preloaded kernargs' name and size
    self.preloadedKernargs = []
    # Each of the manually loaded kernargs' name and size
    self.selfLoadedKernargs = []


  @property
  def numKernargSGPRs(self):
    if self.kernargStart is None:
      assert self.kernargEnd is None
      return 0
    return self.kernargEnd - self.kernargStart

  @property
  def numPreloadSGPRs(self):
    if self.preloadStart is None:
      assert self.preloadEnd is None
      return 0
    return self.preloadEnd - self.preloadStart

  @property
  def kernargs(self):
    from itertools import chain
    return chain(self.preloadedKernargs, self.selfLoadedKernargs)

  ########################################
  # Adds registers to the pool so they can be used as temps
  # Convenience function that takes a range and returns it in string form
  def addRange(self, start, stop, tag=""):
    self.add(start, stop-start+1, tag)
    if (start == stop):
      return "%d"%(start)
    else:
      return "%d-%d" % (start, stop)

  ########################################
  # Adds registers to the pool so they can be used as temps
  # Add
  def add(self, start, size, tag="", newStatus=Status.Available):
    # reserve space
    if self.printRP:
      print("RP::add(%u..%u for '%s')"%(start,start+size-1,tag))
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      for i in range(0, newSize-oldSize):
        self.pool.append(self.Register(RegisterPool.Status.Unavailable,tag))
    # mark as available
    for i in range(start, start+size):
      if self.pool[i].status == RegisterPool.Status.Unavailable:
        self.pool[i].status = newStatus
        self.pool[i].tag = tag
      elif self.pool[i].status == RegisterPool.Status.Available:
        printWarning("RegisterPool::add(%u,%u) pool[%u](%s) already available" % (start, size, i, self.pool[i].tag))
      elif self.pool[i].status == RegisterPool.Status.InUse:
        printWarning("RegisterPool::add(%u,%u) pool[%u](%s) already in use" % (start, size, i, self.pool[i].tag))
      else:
        raise RuntimeError("RegisterPool::add(%u,%u) pool[%u](%s) = %s" % (start, size, i, self.pool[i].tag, self.pool[i].status))
    if self.printRP:
      print(self.state())
  ########################################
  # Remove
  # Removes registers from the pool so they cannot be subsequently allocated for tmps
  def remove(self, start, size, tag=""):
    if self.printRP:
      print("RP::remove(%u..%u) for %s"%(start,size-1,tag))
    # reserve space
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      printWarning("RegisterPool::remove(%u,%u) but poolSize=%u" % (start, size, oldSize))
    # mark as unavailable
    for i in range(start, start+size):
      if  self.pool[i].status == RegisterPool.Status.Available:
        self.pool[i].status = RegisterPool.Status.Unavailable
      elif self.pool[i].status == RegisterPool.Status.Unavailable:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) already unavailable" % (start, size, i, self.pool[i].tag))
      elif  self.pool[i].status == RegisterPool.Status.InUse:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) still in use" % (start, size, i, self.pool[i].tag))
      else:
        printExit("RegisterPool::remove(%u,%u) pool[%u](%s) = %s" % (start, size, i, self.pool[i].tag, self.pool[i].status))

  ########################################
  # Check Out
  def checkOut(self, size, tag="_untagged_", preventOverflow=-1):
    return self.checkOutAligned(size, 1, tag, preventOverflow)

  def isRangeAvailable(self, start, size, preventOverflow=-1, wantedStatus=Status.Available) -> bool:
    end = start + size
    if preventOverflow == -1:
      preventOverflow = self.defaultPreventOverflow

    if preventOverflow and end > len(self.pool):
      return False

    end = min(end, len(self.pool))

    for i in range(start, end):
      if self.pool[i].status != wantedStatus:
        return False

    return True


  def findFreeRange(
    self,
    size: int,
    alignment: int,
    preventOverflow: int=-1,
    wantedStatus: Status=Status.Available
  ) -> Optional[int]:
    """Find a free range of registers of a given size, alignment, and status from the register pool.

    Args:
        size: The required size of the register block.
        alignment: The alignment requirement for the block. Each block must start at an index
            that is a multiple of this value.
        preventOverflow: Flag to prevent overflow. If set to -1, it uses the default value.
        wantedStatus: The status of the block to be found.

    Returns:
        1. The starting index of the block if found.
        2. None if preventOverflow is set and no block is found.
        3. The start of the last available block rounded up to the alignment if preventOverflow is
            not set and no block is found.
    """
    if preventOverflow == -1:
      preventOverflow = self.defaultPreventOverflow

    left, right = 0, 0
    while right < len(self.pool):
      if self.pool[right].status == wantedStatus:
        blockSize = right - left + 1
        if blockSize >= size:
          return left
        right += 1
      else:
        left += alignment
        right = left

    return None if preventOverflow else left


  def checkOutAt(self, start, size, tag, preventOverflow, wantedStatus = Status.Available):
    if preventOverflow:
      assert start + size <= len(self.pool)

    assert self.isRangeAvailable(start, size, preventOverflow, wantedStatus=wantedStatus)

    end = start + size

    numToAdd = max(0, end - len(self.pool))
    if numToAdd > 0:
      self.add(len(self.pool), numToAdd)

    for i in range(start, end):
      assert self.pool[i].status == wantedStatus

    for i in range(start, end):
      self.pool[i].status = RegisterPool.Status.InUse
      self.pool[i].tag = tag
    self.checkOutSize[start] = size
    self.checkOutSizeCache[tag] = size

  def startOfLastAvailableBlock(self) -> int:
    """ Returns the index of the first available register in the highest-numbered free block of registers. """
    for i in range(len(self.pool)-1, 0, -1):
      if self.pool[i].status != RegisterPool.Status.Available:
        return i+1
    return len(self.pool)

  def checkOutAligned(self, size, alignment, tag="_untagged_aligned_", preventOverflow=-1, kernarg=False, preload=False):
    if preventOverflow == -1:
      preventOverflow = self.defaultPreventOverflow
    assert size > 0

    if kernarg:
      assert not preventOverflow
      entry = {"name": tag, "size": size}
      if preload:
        self.preloadedKernargs.append(entry)
      else:
        self.selfLoadedKernargs.append(entry)

    if preload:
      loc = self.findFreeRange(size, alignment, True, RegisterPool.Status.AvailableForPreload)
      if self.preloadStart is None:
        assert loc == 2, "Assume that preloaded kernargs start at s2"
        self.preloadStart = loc
      else:
        assert loc == self.preloadEnd

      self.preloadEnd = loc + size
      assert loc is not None
      self.checkOutAt(loc, size, tag, True, wantedStatus=RegisterPool.Status.AvailableForPreload)
      return loc

    if kernarg:
      assert not preventOverflow
      if self.kernargStart is None:
        loc = self.startOfLastAvailableBlock()
        loc = roundUpToNearestMultiple(loc, alignment)
        self.kernargStart = loc
      else:
        loc = roundUpToNearestMultiple(self.kernargEnd, alignment)

      self.kernargEnd = loc + size
      self.checkOutAt(loc, size, tag, preventOverflow)
      return loc

    loc = self.findFreeRange(size, alignment, preventOverflow)
    assert loc is not None
    self.checkOutAt(loc, size, tag, preventOverflow)
    return loc

  def initTmps(self, initValue, start=0, stop=-1):
    kStr = ""
    stop= len(self.pool) if stop== -1 or stop>len(self.pool) else stop+1
    for i in range(start, stop):
      #if self.type == 's':
      #  print i, self.pool[i].status
      if self.pool[i].status==RegisterPool.Status.Available:
        if self.type == 's':
          kStr += inst("s_mov_b32", sgpr(i), hex(initValue), "init tmp in pool")
        elif self.type == 'v':
          kStr += inst("v_mov_b32", vgpr(i), hex(initValue), "init tmp in pool")
        else:
          assert(0) # bad regpool type

    return kStr

  ########################################
  # Check In
  def checkIn(self, start):
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        self.pool[i].status = RegisterPool.Status.Available
      self.checkOutSize.pop(start)
      if self.printRP:
        print("RP::checkIn('%s') @ %u +%u"%(self.pool[i].tag, start,size))
    else:
      if 0:
        traceback.print_stack(None)
        import pdb; pdb.set_trace()
      printWarning("RegisterPool::checkIn('%s',%s) but it was never checked out"%(self.pool[start].tag, start))
    #traceback.print_stack(None)

  ########################################
  # Size
  def size(self):
    return len(self.pool)


  ########################################
  # Number of available registers
  def available(self):
    numAvailable = 0
    for s in self.pool:
      if s.status == RegisterPool.Status.Available:
        numAvailable += 1
    return numAvailable

  ########################################
  # Size of registers of at least specified blockSize
  def availableBlock(self, blockSize, align):
    if blockSize ==0:
      blockSize = 1
    blocksAvail = 0
    consecAvailable = 0
    #for s in self.pool:
    for i in range(0, len(self.pool)):
      s = self.pool[i]
      if s.status == RegisterPool.Status.Available:
        if not (consecAvailable == 0 and i % align != 0):
          # do not increment if the first item is not aligned
          consecAvailable += 1
      else:
        blocksAvail += consecAvailable // blockSize
        consecAvailable = 0
    blocksAvail += consecAvailable // blockSize
    #print self.state()
    #print "available()=", self.available(), "availableBlock()=",maxAvailable
    return blocksAvail * blockSize

  def availableBlockAtEnd(self):
    availCnt = 0
    for s in reversed(self.pool):
      if s.status == RegisterPool.Status.Available:
        availCnt += 1
      else:
        break

    return availCnt


  ########################################
  def checkFinalState(self):
    for si in range(0,len(self.pool)):
      if self.pool[si].status == RegisterPool.Status.InUse:
        if self.printRP:
          print(self.state())
        raise RuntimeError("RegisterPool::checkFinalState: temp (%s, '%s') was never checked in." \
            %(si, self.pool[si].tag))
    tPrint(3, "total vgpr count: %u\n"%self.size())

  ########################################
  # State
  def state(self):
    stateStr = ""
    placeValues = [1000, 100, 10, 1]
    for placeValueIdx in range(1, len(placeValues)):
      placeValue = placeValues[placeValueIdx]
      priorPlaceValue = placeValues[placeValueIdx-1]
      if len(self.pool) >= placeValue:
        pvs = "" # place value string
        for i in range(0, len(self.pool)):
          if i % placeValue==0:
            pvs += "%u"%((i%priorPlaceValue)//placeValue)
          else:
            pvs += " "
        stateStr += pvs + "\n"
    for i in range(0, len(self.pool)):
      if self.pool[i].status == RegisterPool.Status.Unavailable:
        stateStr += "." # 'removed', this indicates a fixed assignment from "remove", ie a non-tmp allocation
      elif self.pool[i].status == RegisterPool.Status.Available:
        stateStr += "|" # Can be allocated
      elif self.pool[i].status == RegisterPool.Status.InUse:
        stateStr += "#" # Checked out
    return stateStr

  def stateDetailed(self):
    for index, register in enumerate(self.vgprPool.pool):
        print("%u: %s"%(index, register.tag))
