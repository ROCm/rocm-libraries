################################################################################
#
# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
################################################################################

from rocisa.container import MemTokenData
from rocisa.instruction import SBarrier, TensorLoadToLds, \
    DSLoadInstruction, DSStoreInstruction


class LdsTokenManager:
    """Centralized manager for LDS memory token assignment.

    Two mechanisms work together:
      1. Inline assignment: swap*() methods and createBarrier()/
         createWriteMemToken() factories assign tokens during codegen.
      2. Post-codegen verification: verifyTokens() walks the built module
         and checks every tokenizable instruction for missing or
         inconsistent tokens.

    Together these catch all three bug classes:
      - Forgot to assign token  -> verifyTokens flags missing memToken
      - Wrong token value       -> verifyTokens flags out-of-range value
      - Forgot to swap          -> verifyTokens flags consecutive
                                   barriers with same token (2-buffer mode)
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

    def _swap(self, val):
        return self._buf1 if val == self._buf0 else self._buf0

    def swapTensor(self):
        self._tensor = self._swap(self._tensor)

    def swapBarrier(self):
        self._barrier = self._swap(self._barrier)

    def swapWrite(self):
        self._write = self._swap(self._write)

    def swapRead(self):
        if not self._lock_read_swap:
            self._read = self._swap(self._read)

    def swapDirectToLds(self):
        self._dtl = self._swap(self._dtl)

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

    def verifyTokens(self, module, regionName, kernelName):
        """Walk the instruction stream and verify all tokenizable
        instructions have valid tokens.

        Checks:
          1. Every non-cluster SBarrier has a memToken set.
          2. Every TensorLoadToLds has a memToken set.
          3. Every DSStore/DSLoad has a memToken set.
          4. All token values are in {buf0, buf1, meta}.
          5. No two consecutive non-cluster barriers share the same
             token in 2-buffer mode (indicates a missed swap).

        Raises RuntimeError on the first violation found.
        """
        validTokens = {self._buf0, self._buf1, self._meta}
        twoBuffer = (self._buf0 != self._buf1)
        prevBarrierTok = None

        for item in module.flatitems():
            if isinstance(item, SBarrier):
                mt = item.getMemToken()
                if mt is None or len(mt.tokens) == 0:
                    raise RuntimeError(
                        "%s: SBarrier missing memToken (kernel: %s). "
                        "Use tokenMgr.createBarrier() or setMemToken()."
                        % (regionName, kernelName))
                tok = mt.tokens[0]
                if tok not in validTokens:
                    raise RuntimeError(
                        "%s: SBarrier has invalid token %d, expected one of %s "
                        "(kernel: %s)."
                        % (regionName, tok, validTokens, kernelName))
                if twoBuffer and prevBarrierTok is not None and tok == prevBarrierTok:
                    import warnings
                    warnings.warn(
                        "%s: two consecutive barriers with same token %d "
                        "(kernel: %s). Missing swapBarrier()?"
                        % (regionName, tok, kernelName))
                prevBarrierTok = tok

            elif isinstance(item, TensorLoadToLds):
                mt = item.getMemToken()
                if mt is None or len(mt.tokens) == 0:
                    raise RuntimeError(
                        "%s: TensorLoadToLds missing memToken (kernel: %s)."
                        % (regionName, kernelName))
                tok = mt.tokens[0]
                if tok not in validTokens:
                    raise RuntimeError(
                        "%s: TensorLoadToLds has invalid token %d, "
                        "expected one of %s (kernel: %s)."
                        % (regionName, tok, validTokens, kernelName))

            elif isinstance(item, DSStoreInstruction):
                mt = item.getMemToken()
                if mt is None or len(mt.tokens) == 0:
                    raise RuntimeError(
                        "%s: DSStore missing memToken (kernel: %s)."
                        % (regionName, kernelName))
                tok = mt.tokens[0]
                if tok not in validTokens:
                    raise RuntimeError(
                        "%s: DSStore has invalid token %d, expected one of "
                        "%s (kernel: %s)."
                        % (regionName, tok, validTokens, kernelName))

            elif isinstance(item, DSLoadInstruction):
                mt = item.getMemToken()
                if mt is None or len(mt.tokens) == 0:
                    raise RuntimeError(
                        "%s: DSLoad missing memToken (kernel: %s)."
                        % (regionName, kernelName))
                tok = mt.tokens[0]
                if tok not in validTokens:
                    raise RuntimeError(
                        "%s: DSLoad has invalid token %d, expected one of "
                        "%s (kernel: %s)."
                        % (regionName, tok, validTokens, kernelName))
