################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from .Code import Module
from .Formatting import print2, printExit, printWarning
from .Instructions import SMovB32, VMovB32
from .Utils import vgpr, sgpr, roundUpToNearestMultiple

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import traceback

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

  class Register:
    def __init__(self, status, tag):
      self.status = status
      self.tag = tag

  class ResourceOverflowException(Exception):
    pass

  ########################################
  # Init
  # defaultPreventOverflow: control behavior of checkout and checkoutAligned when preventOverflow is not explicitly specificed.
  def __init__(self, size, type, defaultPreventOverflow, printRP=0):
    self.printRP=printRP
    self.type = type
    self.defaultPreventOverflow = defaultPreventOverflow
    self.pool = [self.Register(RegisterPool.Status.Unavailable, "init") for i in range(0,size)]
    self.checkOutSize = {}
    self.checkOutSizeTemp = {}
    self.occupancyLimitSize    = 0
    self.occupancyLimitMaxSize = 0

  #######################################
  # Set occupancy limit
  def setOccupancyLimit(self, maxSize, size):
    self.occupancyLimitSize    = size
    self.occupancyLimitMaxSize = maxSize

  def resetOccupancyLimit(self):
    self.occupancyLimitSize    = 0
    self.occupancyLimitMaxSize = 0

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
  def add(self, start, size, tag=""):
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
        self.pool[i].status = RegisterPool.Status.Available
        self.pool[i].tag = tag
      elif self.pool[i].status == RegisterPool.Status.Available:
        printWarning("RegisterPool::add(%u,%u) pool[%u](%s) already available" % (start, start+size-1, i, self.pool[i].tag))
      elif self.pool[i].status == RegisterPool.Status.InUse:
        printWarning("RegisterPool::add(%u,%u) pool[%u](%s) already in use" % (start, start+size-1, i, self.pool[i].tag))
      else:
        raise RuntimeError("RegisterPool::add(%u,%u) pool[%u](%s) = %s" % (start, start+size-1, i, self.pool[i].tag, self.pool[i].status))
    if self.printRP:
      print(self.state())

  # Adds registers to the pool so they can be used as temps
  def addFromCheckOut(self, start):
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        if self.pool[i].status != RegisterPool.Status.InUse:
          raise RuntimeError("RegisterPool::addFromCheckOut('%s',%s) is not in InUse state"%(self.pool[start].tag, start))
        self.pool[i].status = RegisterPool.Status.Available
      self.checkOutSizeTemp[start] = [size, self.pool[start].tag]
      self.checkOutSize.pop(start)
      if self.printRP:
        print("RP::addFromCheckOut('%s') @ %u +%u"%(self.pool[start].tag, start,size))
    else:
      raise RuntimeError("RegisterPool::addFromCheckOut('%s',%s) but it was never checked out"%(self.pool[start].tag, start))

  ########################################
  # Remove
  # Removes registers from the pool so they cannot be subsequently allocated for tmps
  def remove(self, start, size, tag=""):
    if self.printRP:
      print("RP::remove(%u..%u) for %s"%(start,start+size-1,tag))
    # reserve space
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      printWarning("RegisterPool::remove(%u,%u) but poolSize=%u" % (start, start+size-1, oldSize))
    # mark as unavailable
    for i in range(start, start+size):
      if  self.pool[i].status == RegisterPool.Status.Available:
        self.pool[i].status = RegisterPool.Status.Unavailable
      elif self.pool[i].status == RegisterPool.Status.Unavailable:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) already unavailable" % (start, start+size-1, i, self.pool[i].tag))
      elif  self.pool[i].status == RegisterPool.Status.InUse:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) still in use" % (start, start+size-1, i, self.pool[i].tag))
      else:
        printExit("RegisterPool::remove(%u,%u) pool[%u](%s) = %s" % (start, start+size-1, i, self.pool[i].tag, self.pool[i].status))

  # Removes registers from the pool so they cannot be subsequently allocated for tmps
  def removeFromCheckOut(self, start):
    if start in self.checkOutSizeTemp:
      size, tag = self.checkOutSizeTemp[start]
      for i in range(start, start+size):
        if self.pool[i].status != RegisterPool.Status.Available:
          raise RuntimeError("RegisterPool::addFromCheckOut('%s',%s) is not in Available state"%(self.pool[start].tag, start))
        self.pool[i].status = RegisterPool.Status.InUse
        self.pool[i].tag = tag
      self.checkOutSize[start] = size
      self.checkOutSizeTemp.pop(start)
      if self.printRP:
        print("RegisterPool::removeFromCheckOut('%s') @ %u +%u"%(self.pool[start].tag, start,size))
    else:
      raise RuntimeError("RegisterPool::removeFromCheckOut('%s',%s) but it was never checked out"%(self.pool[start].tag, start))

  ########################################
  # Check Out
  def checkOut(self, size, tag="_untagged_", preventOverflow=-1):
    return self.checkOutAligned(size, 1, tag, preventOverflow)

  def checkOutAligned(self, size, alignment, tag="_untagged_aligned_", preventOverflow=-1):
    if preventOverflow == -1:
      preventOverflow = self.defaultPreventOverflow
    assert(size > 0)
    found = -1
    for i in range(0, len(self.pool)):
      # alignment
      if i % alignment != 0:
        continue
      # enough space
      if i + size > len(self.pool):
        continue
      # all available
      allAvailable = True
      for j in range(0, size):
        if self.pool[i+j].status != RegisterPool.Status.Available:
          allAvailable = False
          i = j+1
          break
      if allAvailable:
        found = i
        break
      else:
        continue

    # success without overflowing
    if found > -1:
      #print "Found: %u" % found
      for i in range(found, found+size):
        self.pool[i].status = RegisterPool.Status.InUse
        self.pool[i].tag = tag
      self.checkOutSize[found] = size
      if self.printRP:
        print("RP::checkOut '%s' (%u,%u) @ %u avail=%u"%(tag, size,alignment, found, self.available()))
        #print self.state()
      return found
    # need overflow
    else:
      #print "RegisterPool::checkOutAligned(%u,%u) overflowing past %u" % (size, alignment, len(self.pool))
      # where does tail sequence of available registers begin
      assert (not preventOverflow)
      start = len(self.pool)
      for i in range(len(self.pool)-1, 0, -1):
        if self.pool[i].status == RegisterPool.Status.Available:
          self.pool[i].tag = tag
          start = i
          continue
        else:
          break
      #print "Start: ", start
      # move forward for alignment

      start = roundUpToNearestMultiple(start,alignment)
      #print "Aligned Start: ", start
      # new checkout can begin at start
      newSize = start + size
      oldSize = len(self.pool)
      if self.occupancyLimitSize > 0:
        if newSize > self.occupancyLimitSize and newSize <= self.occupancyLimitMaxSize:
          print("newSize", newSize, "OldSIze", oldSize, "Limit", self.occupancyLimitSize)
          assert self.occupancyLimitSize >= newSize
      overflow = newSize - oldSize
      #print "Overflow: ", overflow
      for i in range(start, len(self.pool)):
        self.pool[i].status = RegisterPool.Status.InUse
        self.pool[i].tag = tag
      for i in range(0, overflow):
        if len(self.pool) < start:
          # this is padding to meet alignment requirements
          self.pool.append(self.Register(RegisterPool.Status.Available,tag))
        else:
          self.pool.append(self.Register(RegisterPool.Status.InUse,tag))
      self.checkOutSize[start] = size
      if self.printRP:
        print(self.state())
        print("RP::checkOut' %s' (%u,%u) @ %u (overflow)"%(tag, size, alignment, start))
      return start

  def checkOutMulti(self, sizes: List[int], alignment, tags: List[str]):
      assert len(sizes) == len(tags)
      size = 0
      for s in sizes:
        size += s
      idx = self.checkOutAligned(size, alignment, tag="", preventOverflow=0)
      # Overwrite the checkOutSize in formation
      self.checkOutSize.pop(idx)
      idxVec = []
      for sIdx, s in enumerate(sizes):
        idxVec.append(idx)
        self.checkOutSize[idx] = s
        for i in range(idx, idx+s):
          self.pool[i].tag = tags[sIdx]
        idx += s
      return idxVec

  def initTmps(self, initValue, start=0, stop=-1):
    module = Module("initTmps from RegisterPool")
    stop= len(self.pool) if stop== -1 or stop>len(self.pool) else stop+1
    for i in range(start, stop):
      #if self.type == 's':
      #  print i, self.pool[i].status
      if self.pool[i].status==RegisterPool.Status.Available:
        if self.type == 's':
          module.add(SMovB32(dst=sgpr(i), src=hex(initValue), comment="init tmp in pool"))
        elif self.type == 'v':
          module.add(VMovB32(dst=vgpr(i), src=hex(initValue), comment="init tmp in pool"))
        else:
          assert(0) # bad regpool type

    return module

  ########################################
  # Check In
  def checkIn(self, start):
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        self.pool[i].status = RegisterPool.Status.Available
      self.checkOutSize.pop(start)
      if self.printRP:
        print("RP::checkIn('%s') @ %u +%u"%(self.pool[start].tag, start,size))
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

  # Size of registers of at least specified blockSize
  def availableBlockMaxVgpr(self, maxVgpr, blockSize, align):
    if blockSize ==0:
      blockSize = 1
    blocksAvail = 0
    consecAvailable = 0
    #for s in self.pool:
    for i in range(0, maxVgpr):
      if i >= len(self.pool) :
        if not (consecAvailable == 0 and i % align != 0):
          consecAvailable += 1
      else:
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
    print2("total vgpr count: %u\n"%self.size())

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
    for index, register in enumerate(self.pool):
        print("%u: %s"%(index, register.tag))
  
  def growPool(self, rangeStart: int, rangeEnd: int, checkOutSize: int, comment: str=""):
    tl = []
    for _ in range(rangeStart, rangeEnd):
      tl.append(self.checkOut(checkOutSize, comment))
    for t in tl:
      self.checkIn(t)

@dataclass
class RegisterPoolResource:
    idx: int
    size: int

@contextmanager
def allocTmpGpr(pool: RegisterPool, num: int, upperLimit: int, alignment: Optional[int]=None, tag: Optional[str]=None, overflowListener=None):
  """
  A context-manager based temporary resource allocator for given RegisterPool object.

  :param pool: `RegisterPool` Resource pool
  :param num: `int` Size to allocate
  :param alignment: `Optional[bool]` Resource to be aligned to spcified alignment. If not specified, alignment set to 2 if `num` > 1 else 1
  :param tag: `Optional[str]` Specified tag string for code generation
  :raises `RegisterPool.ResourceOverflowException` if cannot allocate resource
  :returns: `(offset, size)`, offset: `int` Start offset of allocated resource
    size: `int` Size of allocated resource
  """
  assert alignment is None or alignment > 0
  if alignment is None:
    alignment = 1 if num == 1 else 2
  if tag is None:
    tag = f"allocTmp{pool.type.upper()}gpr({num})"

  try:
    allocatedSgprIdx = pool.checkOutAligned(num, alignment, tag, False)

    if allocatedSgprIdx + num > upperLimit:
      exception = RegisterPool.ResourceOverflowException(f"{pool.type.upper()}gpr overflow")
      if overflowListener:
        overflowListener(exception)
      else:
        raise exception

    yield RegisterPoolResource(idx=allocatedSgprIdx, size=num)
  finally:
    pool.checkIn(allocatedSgprIdx) # type: ignore

@contextmanager
def allocTmpGprList(pool: RegisterPool, nums: List[int], upperLimit: int, alignments: Optional[List[int]]=None, tag: Optional[str]=None, overflowListener=None):
  """
  A context-manager based temporary resource allocator for given RegisterPool object.

  :param pool: `RegisterPool` Resource pool
  :param num: `int` Size to allocate
  :param alignment: `Optional[bool]` Resource to be aligned to spcified alignment. If not specified, alignment set to 2 if `num` > 1 else 1
  :param tag: `Optional[str]` Specified tag string for code generation
  :raises `RegisterPool.ResourceOverflowException` if cannot allocate resource
  :returns: `(offset, size)`, offset: `int` Start offset of allocated resource
    size: `int` Size of allocated resource
  """

  if alignments:
    if len(alignments) == 1:
      for num in nums:
        if num % alignments[0] != 0:
          print("Mod %% hint must == 0")
          assert 0
      alignments = [alignments[0]] * len(nums)
    else:
      assert len(nums) == len(alignments)
      for num, alignment in zip(nums, alignments):
        if num % alignment != 0:
          print("Mod %% hint must == 0")
          assert 0
  else:
    alignments = []
    for num in nums:
      alignments.append(1 if num == 1 else 2)

  try:
    allocatedSgprIdxList = []
    for num, alignment in zip(nums, alignments):
      if tag is None:
        tag = f"allocTmp{pool.type.upper()}gpr({num})"

      allocatedSgprIdx = pool.checkOutAligned(num, alignment, tag, False)

      if allocatedSgprIdx + num > upperLimit:
        exception = RegisterPool.ResourceOverflowException(f"{pool.type.upper()}gpr overflow")
        if overflowListener:
          overflowListener(exception)
        else:
          raise exception
      allocatedSgprIdxList.append([allocatedSgprIdx, num])

    registerPoolResourceList = []
    for allocatedSgprIdx, num in allocatedSgprIdxList:
      registerPoolResourceList.append(RegisterPoolResource(idx=allocatedSgprIdx, size=num))

    yield registerPoolResourceList
  finally:
    for allocatedSgprIdx, _ in allocatedSgprIdxList:
      pool.checkIn(allocatedSgprIdx) # type: ignore
