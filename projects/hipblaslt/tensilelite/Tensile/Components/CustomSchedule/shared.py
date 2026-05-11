################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Shared utilities, dataclasses, and dtype predicates for the CustomSchedule package.

This module is the leaf of the CustomSchedule package's import graph: it must
NOT import from ``dispatch`` or any per-schedule file.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from rocisa.instruction import SBarrier, SNop, SWaitCnt

from Tensile.Utilities.Decorators.Shared import CallableGuard


# Map dtype predicate functions to human-readable names.
# Populated by `_register_dtype_name` calls below.
_DTYPE_PREDICATE_NAMES: dict[Callable, str] = {}


def _register_dtype_name(func: Callable, name: str) -> Callable:
    """Helper to register a dtype predicate name mapping."""
    _DTYPE_PREDICATE_NAMES[func] = name
    return func


@dataclass(frozen=True)
class CMSKernelInfo:
    """
    Metadata about registered CMS kernels
    Contains the minimum combination of parameters needed to use the CMS kernel.
    Important Note:
    If you are adding new parameters to this list (of params use in CMS kernels), please make sure those names match Tensile names.
    These names will be used by caller/tuning codes to set correct parameter/values.
    """
    name: str
    dtype: str
    MacroTile0: int
    MacroTile1: int
    DepthU: int
    PrefetchGlobalRead: int
    PrefetchLocalRead: int
    DirectToLds: bool
    DtlPlusLdsBuf: int
    WaveSeparateGlobalReadA: int
    WaveSeparateGlobalReadB: int
    GlobalReadVectorWidthA: int
    GlobalReadVectorWidthB: int
    LocalReadVectorWidth: int
    MatrixInstruction: list[int]
    MIWaveGroup: list[int]
    LDSTrInst: bool
    TransposeLDS: int
    TransposeA: bool
    TransposeB: bool

    def matches(self, dtype: Optional[str] = None, layout: Optional[str] = None) -> bool:
        """Check if this kernel info matches the given dtype and/or layout filter.

        Args:
            dtype:  Data type filter string (e.g. "16bit", "8bit", "TF32"), or None for any.
            layout: Layout filter string (e.g. "TN", "NT", "NN", "TT"), or None for any.

        Returns:
            True if the kernel matches all provided filters.
        """
        if dtype is not None and self.dtype.lower() != dtype.lower():
            return False
        if layout is not None:
            layout = layout.upper()
            if self.TransposeA != (layout[0] == "T") or self.TransposeB != (layout[1] == "T"):
                return False
        return True



@dataclass
class SyncSchedule:
    schedule: list[tuple[int, Union[SWaitCnt, SBarrier]]] = field(default_factory=list)

    def add(self, idx: int, dscnt: int = -1, vlcnt: int = -1, vscnt: int = -1, comment: str = "", barrier: bool = False, barrier_idx: Optional[int] = None, barrier_comment: str = ""):
        """ Add a SWaitCnt (and optionally a SBarrier) to the schedule at the given index.
        Args:
            idx:             The index at which to add the SWaitCnt.
            dscnt:           The dscnt value for the SWaitCnt.
            vlcnt:           The vlcnt value for the SWaitCnt.
            vscnt:           The vscnt value for the SWaitCnt.
            comment:         An optional comment for the SWaitCnt.
            barrier:         If True, also add a SBarrier.
            barrier_idx:     The index at which to add the SBarrier. If None, uses idx.
            barrier_comment: An optional comment for the SBarrier.

        Example:
            wait.add(2, dscnt=3)                                   adds SWaitCnt at index 2 with dscnt=3
            wait.add(5, dscnt=0, sbarrier=True)                    adds SWaitCnt at index 5 with dscnt=0 and a SBarrier at the same index
            wait.add(5, dscnt=0, sbarrier=True, barrier_idx=6)     adds SWaitCnt at index 5 with dscnt=0 and a SBarrier at index 6
        """
        self.schedule.append( (idx, SWaitCnt(dscnt=dscnt, vlcnt=vlcnt, vscnt=vscnt, comment=comment)) )
        if barrier:
            barrier_idx = barrier_idx if barrier_idx is not None else idx
            self.schedule.append( (barrier_idx, SBarrier(comment=barrier_comment)) )

    def get_indicies(self):
        return [item[0] for item in self.schedule]
    def get_code(self):
        return [item[1] for item in self.schedule]

def create_range(min_val: int, num: int, max_val: int = -1, step: int = 1, repeat: int = 2) -> list[int]:
    """
    Generate a list where each value in range(min_val, min_val+num, step) is repeated 'repeat' times.
    Value is clamped to max_val

    Args:
        min_val: Starting value (inclusive)
        num: Number of values
        step: Step between values
        max_val: Maximum value (clamp)
        repeat: Number of times to repeat each value

    Example:
        create_range(100, 5,200, 1, 2) => [100, 100, 101, 101, 102, 102, 103, 103, 104, 104]
        create_range(0, 5, 10, 2, 3) => [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8]
        create_range(0, 5, 6, 2, 3) => [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6]
    """
    if max_val == -1:
        max_val = min_val + step*num
    return [min(val, max_val) for val in range(min_val, min_val + step*num, step) for _ in range(repeat)]

def inflight(lst, index):
    """
    Return number of inflight loads in a given list of instructions at a specified index
    """
    return sum(val < (index) for val in lst)

def duplicate_list_items(input_list: list, repeat_count: int, step: int = 0) -> list:
    """
    Duplicate each item in input_list repeat_count times. Optionally duplicate with a step

    Example:
        duplicate_list_items([1, 2, 3], 3)    => [1,1,1, 2,2,2, 3,3,3]
        duplicate_list_items([1, 2, 3], 3, 1) => [1,2,3, 2,3,4, 3,4,5]
    """
    return [item + step * j for item in input_list for j in range(repeat_count)]

def count_items(input_list: list[int], sv: Optional[int] = None, ev: Optional[int] = None):
    """
    Count how many items in the list are between start value `sv` (inclusive) and end value `ev` (exclusive)

    Example:
        count_items([1,2,3,4,5], sv=2, ev=5) => 3 (2,3,4)
        count_items([1,2,3,4,5], sv=3)        => 3 (3,4,5)
        count_items([1,2,3,4,5], ev=4)        => 3 (1,2,3)
    """
    count = 0
    sv = sv if sv is not None else input_list[0]
    ev = ev if ev is not None else input_list[-1]
    for item in input_list:
        if sv <= item < ev:
            count += 1
    return count

def switch_A_B_schedule(optSchedule):
    # Swap A and B entries in the schedule
    # Only replace A/B if it's the last or second-last character
    swappedSchedule = dict()
    for key, value in optSchedule.items():
        # Check if A or B is in the last or second-last position
        if len(key) >= 1 and key[-1] in ('A', 'B'):
            # Last character is A or B
            new_key = key[:-1] + ('B' if key[-1] == 'A' else 'A')
        elif len(key) >= 2 and key[-2] in ('A', 'B'):
            # Second-last character is A or B
            new_key = key[:-2] + ('B' if key[-2] == 'A' else 'A') + key[-1]
        else:
            # No A or B in last or second-last position, keep unchanged
            new_key = key
        swappedSchedule[new_key] = value
    return swappedSchedule

class ScheduleInfo:
    def __init__(
        self,
        numCodePaths: int,
        numMfma: int,
        optSchedule: dict[str, list[list[int]]],
        syncCode: list[Union[SWaitCnt, SBarrier]],
        nglshift: int,
        nllshift: int,
        nllZeroDscnt: bool = False,
        mfmaReorder = [],
        snopCode: list[SNop] = [],
    ):
        self.numCodePaths = numCodePaths
        self.numMfma = numMfma
        self.optSchedule = optSchedule
        self.syncCode = syncCode
        self.nglshift = nglshift  # vmcnt shift for noglobalload loop
        self.nllshift = nllshift  # vmcnt shift for nolocalload loop
        self.nllZeroDscnt = nllZeroDscnt
        self.mfmaReorder = mfmaReorder
        self.snopCode = snopCode

    def pretty_print(self):
        print("{")
        keys = list(self.optSchedule.keys())
        maxKeyLen = max(len(k) for k in keys) if keys else 0
        for i, k in enumerate(keys):
            v = self.optSchedule[k]
            comma = "," if i < len(keys) - 1 else ""
            pad = " " * (maxKeyLen - len(k))
            if len(v) == 1:
                print(f"    '{k}':{pad} [{v[0]}]{comma}")
            else:
                # Align continuation rows after the opening bracket
                bracketCol = 8 + maxKeyLen
                indent = " " * (bracketCol + 1)
                print(f"    '{k}':{pad} [")
                for j, row in enumerate(v):
                    row_comma = "," if j < len(v) - 1 else ""
                    print(f"{indent}{row}{row_comma}")
                print(f"{' ' * bracketCol}]{comma}")
        print("}")

        if snops := self.optSchedule.get('SNOP', []):
            print("---- SNOP code ----")
            for idx, code in zip(snops[0], self.snopCode):
                print(f"{idx:>2}: {str(code).strip()}")

        if syncs := self.optSchedule.get('SYNC', []):
            print("---- SYNC code ----")
            for idx, code in zip(syncs[0], self.syncCode):
                print(f"{idx:>2}: {str(code).strip()}")


@CallableGuard
def isNN(kernel):
    return not kernel["ProblemType"]["TransposeA"] and not kernel["ProblemType"]["TransposeB"]

@CallableGuard
def isNT(kernel):
    return not kernel["ProblemType"]["TransposeA"] and kernel["ProblemType"]["TransposeB"]

@CallableGuard
def isTT(kernel):
    return kernel["ProblemType"]["TransposeA"] and kernel["ProblemType"]["TransposeB"]

@CallableGuard
def isTN(kernel):
    return kernel["ProblemType"]["TransposeA"] and not kernel["ProblemType"]["TransposeB"]

@CallableGuard
def is16bit(kernel):
    return kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()
_register_dtype_name(is16bit, "16bit")

@CallableGuard
def is8bit(kernel):
    return kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()
_register_dtype_name(is8bit, "8bit")

@CallableGuard
def isMixed(kernel):
    return kernel["ProblemType"]["DataTypeA"].numBytes() != kernel["ProblemType"]["DataTypeB"].numBytes()

@CallableGuard
def isTF32(kernel):
    return kernel["UseF32XEmulation"]
_register_dtype_name(isTF32, "TF32")


@dataclass(frozen=True)
class TileConfig:
    macro_tile_size_0: int
    macro_tile_size_1: int
    depth_u: int
    prefetch_global_read: int
    prefetch_local_read: int
    direct_to_lds: int
    dtl_plus_lds_buf: bool
    wave_separate_global_read_a: int
    wave_separate_global_read_b: int
    isa: tuple                              # required (no fallback to gfx950)
    wavefront_size: int = 64
