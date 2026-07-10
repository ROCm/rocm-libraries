# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MemToken helpers for gfx1250 subtile StinkyTofu waitcnt insertion."""

from __future__ import annotations

from rocisa.container import MemTokenData
from rocisa.instruction import Instruction, SBarrier


def initSubtileMemTokens(writer, kernel) -> None:
    """Ensure writer.states carries LDS mem-token indices for subtile emit.

    Mirrors KernelWriter mem-token setup (``memTokenLdsBuffer*`` / ``lds*TokenIdx``).
    No-op when tokens are already initialized (e.g. full KernelWriter codegen).
    """
    states = writer.states
    if hasattr(states, "ldsDirectToLDSTokenIdx"):
        return

    if kernel.get("1LDSBuffer", False):
        states.memTokenLdsBuffer0 = 0
        states.memTokenLdsBuffer1 = 0
    else:
        states.memTokenLdsBuffer0 = 0
        states.memTokenLdsBuffer1 = 1

    t0 = states.memTokenLdsBuffer0
    states.ldsReadTokenIdx = t0
    states.ldsTensorTokenIdx = t0
    states.ldsDirectToLDSTokenIdx = t0
    states.ldsWriteTokenIdx = t0


def _flipLdsBufferToken(writer, attr: str) -> None:
    states = writer.states
    cur = getattr(states, attr)
    b0 = states.memTokenLdsBuffer0
    b1 = states.memTokenLdsBuffer1
    setattr(states, attr, b1 if cur == b0 else b0)


def flipGrWriteTokens(writer) -> None:
    """Flip DTL write / local-write token indices after an LW buffer swap."""
    _flipLdsBufferToken(writer, "ldsDirectToLDSTokenIdx")
    _flipLdsBufferToken(writer, "ldsWriteTokenIdx")


def flipLrReadToken(writer) -> None:
    """Flip LR read token index after an LR buffer swap."""
    _flipLdsBufferToken(writer, "ldsReadTokenIdx")


def flipTensorLoadToken(writer) -> None:
    """Flip TDM tensor-load token index after a TDM LDS buffer swap."""
    _flipLdsBufferToken(writer, "ldsTensorTokenIdx")


def barrierTokens(writer, kernel) -> list[int]:
    """Tokens carried by workgroup / cluster barriers in the mainloop."""
    tokens = [writer.states.memTokenLdsBuffer0]
    if not kernel.get("1LDSBuffer", False):
        tokens.append(writer.states.memTokenLdsBuffer1)
    return sorted(set(tokens))


def tagDtlLoad(inst: Instruction, writer) -> None:
    inst.setMemToken(MemTokenData([writer.states.ldsDirectToLDSTokenIdx]))


def tagTensorLoad(inst: Instruction, writer) -> None:
    inst.setMemToken(MemTokenData([writer.states.ldsTensorTokenIdx]))


def tagDsRead(inst: Instruction, writer) -> None:
    inst.setMemToken(MemTokenData([writer.states.ldsReadTokenIdx]))


def tagBarrier(inst: Instruction, writer, kernel) -> None:
    if isinstance(inst, SBarrier):
        inst.setMemToken(MemTokenData(barrierTokens(writer, kernel)))


def tagModuleBarriers(module, writer, kernel) -> None:
    for item in module.flatitems():
        if isinstance(item, SBarrier):
            tagBarrier(item, writer, kernel)
