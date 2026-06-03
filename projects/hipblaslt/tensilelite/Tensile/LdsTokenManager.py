################################################################################
#
# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################

from rocisa.container import MemTokenData
from rocisa.instruction import SBarrier


class LdsTokenManager:
    """Centralized manager for LDS memory token assignment.

    Replaces the scattered manual toggle pattern for ldsTensorTokenIdx,
    ldsBarrierTokenIdx, ldsWriteTokenIdx, ldsReadTokenIdx, and
    ldsDirectToLDSTokenIdx with a single object that derives token values
    from kernel parameters and provides swap methods.

    Use createBarrier() instead of raw SBarrier() to guarantee the correct
    token is attached.
    """

    def __init__(self, kernel):
        if kernel["1LDSBuffer"]:
            self._buf0 = 0
            self._buf1 = 0
        else:
            self._buf0 = 0
            self._buf1 = 1
        self._meta = 4

        self._tensor = self._buf0
        self._barrier = self._buf0
        self._write = self._buf0
        self._read = self._buf0
        self._dtl = self._buf0
        self._lock_read_swap = False

    @property
    def buf0(self):
        return self._buf0

    @property
    def buf1(self):
        return self._buf1

    @property
    def meta(self):
        return self._meta

    @property
    def tensor(self):
        return self._tensor

    @property
    def barrier(self):
        return self._barrier

    @property
    def write(self):
        return self._write

    @property
    def read(self):
        return self._read

    @property
    def directToLds(self):
        return self._dtl

    @property
    def isReadSwapLocked(self):
        return self._lock_read_swap

    def swapTensor(self):
        self._tensor = self._buf1 if self._tensor == self._buf0 else self._buf0

    def swapBarrier(self):
        self._barrier = self._buf1 if self._barrier == self._buf0 else self._buf0

    def swapWrite(self):
        self._write = self._buf1 if self._write == self._buf0 else self._buf0

    def swapRead(self):
        if not self._lock_read_swap:
            self._read = self._buf1 if self._read == self._buf0 else self._buf0

    def swapDirectToLds(self):
        self._dtl = self._buf1 if self._dtl == self._buf0 else self._buf0

    def resetWrite(self):
        self._write = self._buf0

    def resetRead(self):
        self._read = self._buf0

    def lockReadSwap(self):
        self._lock_read_swap = True

    def createBarrier(self, comment=""):
        b = SBarrier(comment=comment)
        b.setMemToken(MemTokenData([self._barrier]))
        return b

    def createWriteMemToken(self):
        return MemTokenData([self._write])

    def createReadMemToken(self):
        return MemTokenData([self._read])
