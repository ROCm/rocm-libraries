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

from rocisa.container import ContinuousRegister
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

from contextlib import contextmanager
from typing import List, Optional

def ResourceOverflowException(Exception):
  pass

@contextmanager
def allocTmpGpr(pool: RegisterPool, num: int, upperLimit: int, alignment: Optional[int]=None, tag: Optional[str]=None, overflowListener=None):
  """
  A context-manager based temporary resource allocator for given RegisterPool object.

  :param pool: `RegisterPool` Resource pool
  :param num: `int` Size to allocate
  :param alignment: `Optional[bool]` Resource to be aligned to spcified alignment. If not specified, alignment set to 2 if `num` > 1 else 1
  :param tag: `Optional[str]` Specified tag string for code generation
  :raises `ResourceOverflowException` if cannot allocate resource
  :returns: `(offset, size)`, offset: `int` Start offset of allocated resource
    size: `int` Size of allocated resource
  """
  assert alignment is None or alignment > 0
  if alignment is None:
    alignment = 1 if num == 1 else 2
  if tag is None:
    tag = f"allocTmpgpr({num})"

  try:
    allocatedSgprIdx = pool.checkOutAligned(num, alignment, tag, False)

    if allocatedSgprIdx + num > upperLimit:
      exception = ResourceOverflowException(f"gpr overflow")
      if overflowListener:
        overflowListener(exception)
      else:
        raise exception

    yield ContinuousRegister(idx=allocatedSgprIdx, size=num)
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
  :raises `ResourceOverflowException` if cannot allocate resource
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
        tag = f"allocTmpgpr({num})"

      allocatedSgprIdx = pool.checkOutAligned(num, alignment, tag, False)

      if allocatedSgprIdx + num > upperLimit:
        exception = ResourceOverflowException(f"gpr overflow")
        if overflowListener:
          overflowListener(exception)
        else:
          raise exception
      allocatedSgprIdxList.append([allocatedSgprIdx, num])

    registerPoolResourceList = []
    for allocatedSgprIdx, num in allocatedSgprIdxList:
      registerPoolResourceList.append(ContinuousRegister(idx=allocatedSgprIdx, size=num))

    yield registerPoolResourceList
  finally:
    for allocatedSgprIdx, _ in allocatedSgprIdxList:
      pool.checkIn(allocatedSgprIdx) # type: ignore
