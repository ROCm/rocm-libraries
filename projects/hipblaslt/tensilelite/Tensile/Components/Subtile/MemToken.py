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

LDS double-buffer model (subtile):
  * Each tensor (A/B/SA/SB) owns an *independent* LDS double buffer with its own
    swap mask / LDS address (``tdmLdsSwapMask*`` for A/B,
    ``emitScaleGRLDSSwap``/``emitScaleLRLDSSwap`` for the scales). A global-read
    inc swaps only that tensor's write buffer; a local-read inc swaps only that
    tensor's read buffer. ``insert_gr_lr_inc`` attaches the inc per tensor when
    that tensor's MT iteration changes, so swaps do *not* happen in lockstep.
  * Because the buffers are distinct memory, each tensor gets a distinct token-id
    space so StinkyTofu does not create false cross-tensor LDS dependencies
    (matching the classic path, which carves out a separate id for the meta/scale
    buffer). Within a tensor: buffer0 -> base, buffer1 -> base+1 (both ``base``
    when ``1LDSBuffer`` collapses the double buffer).
  * The producer (``tensor_load_to_lds``) token follows the tensor's current
    write buffer; the consumer (``ds_read``) token follows its current read
    buffer; a barrier carries the union of every tracked tensor's current write
    and read ids (it separates all LDS producers from all consumers).

Per-body (re)initialization:
  * A single tracker instance is reused across the preloop, every mainloop unroll
    copy, the NGLL/NLL drains, and the tail. Each emitted body must start from
    the LDS buffer parity the runtime actually has at that body's entry, so the
    tracker exposes ``reset`` (back to buffer0, the kernel-entry / tail-loop
    state) and ``snapshot``/``restore`` so the caller can re-establish a body's
    entry parity instead of inheriting stale parity from an earlier schedule.
"""

from __future__ import annotations

from rocisa.container import MemTokenData


# Distinct id base per tensor: independent LDS buffers must not share token ids.
_TENSOR_TOKEN_BASE = {'A': 0, 'B': 2, 'SA': 4, 'SB': 6}


class SubtileMemTokenTracker:
    """Tracks per-tensor LDS read/write buffer ids for MemTokenData tagging."""

    def __init__(self, kernel, tensors=('A', 'B')):
        oneBuffer = bool(kernel.get("1LDSBuffer", False))
        self._tensors = tuple(tensors)
        self._buf0 = {}
        self._buf1 = {}
        self._write = {}
        self._read = {}
        for t in self._tensors:
            base = _TENSOR_TOKEN_BASE[t]
            self._buf0[t] = base
            self._buf1[t] = base if oneBuffer else base + 1
            # Every buffer starts on buffer0, matching the classic init of
            # ldsTensorTokenIdx / ldsReadTokenIdx.
            self._write[t] = base
            self._read[t] = base

    def _toggle(self, tensor, idx):
        return self._buf1[tensor] if idx == self._buf0[tensor] else self._buf0[tensor]

    def swapWrite(self, tensor):
        """Flip a tensor's producer (LDS write) buffer on a global-read swap."""
        self._write[tensor] = self._toggle(tensor, self._write[tensor])

    def swapRead(self, tensor):
        """Flip a tensor's consumer (LDS read) buffer on a local-read swap."""
        self._read[tensor] = self._toggle(tensor, self._read[tensor])

    def writeToken(self, tensor):
        """MemTokenData for a tensor's LDS producer (tensor_load_to_lds)."""
        return MemTokenData([self._write[tensor]])

    def readToken(self, tensor):
        """MemTokenData for a tensor's LDS consumer (ds_read)."""
        return MemTokenData([self._read[tensor]])

    def barrierToken(self):
        """MemTokenData for a barrier: union of every tracked buffer it separates."""
        ids = set()
        for t in self._tensors:
            ids.add(self._write[t])
            ids.add(self._read[t])
        return MemTokenData(sorted(ids))

    def reset(self):
        """Point every tensor's read/write buffer back at buffer0.

        This is the kernel-entry and tail-loop LDS state; call it at the start of
        a body that the runtime reaches with buffer0 current.
        """
        for t in self._tensors:
            self._write[t] = self._buf0[t]
            self._read[t] = self._buf0[t]

    def snapshot(self):
        """Capture the current per-tensor parity for later ``restore``."""
        return (dict(self._write), dict(self._read))

    def restore(self, snap):
        """Re-establish a parity captured by ``snapshot``."""
        write, read = snap
        self._write = dict(write)
        self._read = dict(read)


# Instruction classes whose instances are LDS producers / consumers and must be
# tagged consistently within a region (the rocisa MemTokenConsistencyCheck pass
# fatals on partial tagging of tensor_load / ds_read / ds_write in a block).
def isLdsProducer(inst):
    """True for LDS-producing mem-token candidates (tensor_load_to_lds / ds_write).

    buffer_load...lds (DTL) loads also write LDS, but are deliberately NOT
    tagged: StinkyTofu (gfx1250) classifies them as plain MUBUF loads rather
    than LDS writers, so a token on them is inert in release and asserts in
    debug. Kernels with such producers are instead kept off StinkyTofu-owned
    wait counts by the subtile guard (see StinkyTofu.subtileKernelIsWaitInsertionSafe).
    """
    from rocisa.instruction import TensorLoadToLds, DSStoreInstruction
    return isinstance(inst, (TensorLoadToLds, DSStoreInstruction))


def isLdsConsumer(inst):
    """True for LDS-consuming mem-token candidates (any ds_read variant)."""
    from rocisa.instruction import DSLoadInstruction
    return isinstance(inst, DSLoadInstruction)
