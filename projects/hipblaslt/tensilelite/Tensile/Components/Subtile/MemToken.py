# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS memory-token tracking for the gfx1250 subtile path.

Mirrors the classic ``memTokenLdsBuffer0/1`` double-buffer toggling, but the
state lives entirely on the subtile emitter. Tokens are attached to LDS
producer/consumer/barrier instructions via ``MemTokenData`` so StinkyTofu can
later build producer -> barrier -> consumer dependencies (RegType::LDS
pseudo-registers keyed by token id).

``MemTokenData`` emits no assembly text, so tagging is harmless metadata when
StinkyTofu does not consume it: the emitted asm is identical whether or not the
tags are present.

Token-id scheme (matching the classic path):
  * buffer0 -> id 0, buffer1 -> id 1 (both 0 when ``1LDSBuffer`` collapses the
    double buffer to a single buffer).
  * The producer (``tensor_load_to_lds``) writes the current write buffer; its
    token id follows ``writeTokenIdx``.
  * The consumer (``ds_read``) reads the current read buffer; its token id
    follows ``readTokenIdx``.
  * A barrier separating producers from consumers carries both buffer ids it
    spans (the union of the current write and read ids).
  * A global-read LDS swap flips ``writeTokenIdx``; a local-read LDS swap flips
    ``readTokenIdx`` -- the same toggle points the runtime XORs the LDS
    addresses/offsets at.
"""

from __future__ import annotations

from rocisa.container import MemTokenData


class SubtileMemTokenTracker:
    """Tracks the current LDS read/write buffer ids for MemTokenData tagging."""

    def __init__(self, kernel):
        oneBuffer = bool(kernel.get("1LDSBuffer", False))
        self.buffer0 = 0
        self.buffer1 = 0 if oneBuffer else 1
        # Both buffers start pointing at buffer0, matching the classic init of
        # ldsTensorTokenIdx / ldsReadTokenIdx.
        self.writeTokenIdx = self.buffer0
        self.readTokenIdx = self.buffer0

    def _toggle(self, idx):
        return self.buffer1 if idx == self.buffer0 else self.buffer0

    def swapWrite(self):
        """Flip the producer (LDS write) buffer on a global-read LDS swap."""
        self.writeTokenIdx = self._toggle(self.writeTokenIdx)

    def swapRead(self):
        """Flip the consumer (LDS read) buffer on a local-read LDS swap."""
        self.readTokenIdx = self._toggle(self.readTokenIdx)

    def writeToken(self):
        """MemTokenData for an LDS producer (tensor_load_to_lds / ds_write)."""
        return MemTokenData([self.writeTokenIdx])

    def readToken(self):
        """MemTokenData for an LDS consumer (ds_read)."""
        return MemTokenData([self.readTokenIdx])

    def barrierToken(self):
        """MemTokenData for a barrier: the union of the buffers it separates."""
        return MemTokenData(sorted({self.writeTokenIdx, self.readTokenIdx}))


# Instruction classes whose instances are LDS producers / consumers and must be
# tagged consistently within a region (the rocisa MemTokenConsistencyCheck pass
# fatals on partial tagging of tensor_load / ds_read / ds_write in a block).
def isLdsProducer(inst):
    """True for LDS-producing mem-token candidates (tensor_load_to_lds / ds_write)."""
    from rocisa.instruction import TensorLoadToLds, DSStoreInstruction
    return isinstance(inst, (TensorLoadToLds, DSStoreInstruction))


def isLdsConsumer(inst):
    """True for LDS-consuming mem-token candidates (any ds_read variant)."""
    from rocisa.instruction import DSLoadInstruction
    return isinstance(inst, DSLoadInstruction)
