# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM IR text analysis.

These counters are deliberately simple and stable: they count intrinsic
declarations/calls and memory operations in the LLVM IR we emit. They are
not a replacement for ISA inspection, but they are the fastest possible
sanity check that a DSL feature actually lowered to the intended primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict


@dataclass(frozen=True)
class LlvmIrStats:
    """Instruction-family counts from AMDGPU LLVM IR text."""

    mfma_calls: int = 0
    mfma_16x16x16: int = 0
    mfma_16x16x32: int = 0
    mfma_32x32x8: int = 0
    mfma_32x32x16: int = 0
    mfma_4x4x4: int = 0
    raw_buffer_load_calls: int = 0
    raw_buffer_store_calls: int = 0
    async_buffer_load_lds_calls: int = 0
    global_vector_loads: int = 0
    global_vector_stores: int = 0
    smem_vector_loads: int = 0
    smem_vector_stores: int = 0
    barriers: int = 0
    waitcnts: int = 0
    sched_group_barriers: int = 0
    sched_barriers: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "mfma_calls": self.mfma_calls,
            "mfma_16x16x16": self.mfma_16x16x16,
            "mfma_16x16x32": self.mfma_16x16x32,
            "mfma_32x32x8": self.mfma_32x32x8,
            "mfma_32x32x16": self.mfma_32x32x16,
            "mfma_4x4x4": self.mfma_4x4x4,
            "raw_buffer_load_calls": self.raw_buffer_load_calls,
            "raw_buffer_store_calls": self.raw_buffer_store_calls,
            "async_buffer_load_lds_calls": self.async_buffer_load_lds_calls,
            "global_vector_loads": self.global_vector_loads,
            "global_vector_stores": self.global_vector_stores,
            "smem_vector_loads": self.smem_vector_loads,
            "smem_vector_stores": self.smem_vector_stores,
            "barriers": self.barriers,
            "waitcnts": self.waitcnts,
            "sched_group_barriers": self.sched_group_barriers,
            "sched_barriers": self.sched_barriers,
        }


_CALL_RE = re.compile(r"\bcall\b")


def _count_calls(text: str, needle: str) -> int:
    return sum(
        1 for line in text.splitlines() if needle in line and _CALL_RE.search(line)
    )


def _count_non_decl(text: str, needle: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if needle in line and not line.lstrip().startswith("declare ")
    )


def _count_lines(text: str, *needles: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if all(needle in line for needle in needles)
        and not line.lstrip().startswith("declare ")
    )


def analyze_llvm_ir(llvm_ir: str) -> LlvmIrStats:
    """Count key AMDGPU LLVM IR primitive families.

    This is designed for regression tests and performance debugging:

    - `async_buffer_load_lds_calls > 0` proves the async DMA path was emitted.
    - `raw_buffer_load_calls == 0` in an async-only path proves the old
      register-staged DRAM load was not accidentally left in the loop.
    - `barriers` / `waitcnts` give a quick synchronization footprint before
      going to final ISA.
    """

    mfma_16x16x16 = _count_calls(llvm_ir, "@llvm.amdgcn.mfma.f32.16x16x16f16")
    mfma_16x16x32 = _count_calls(llvm_ir, "@llvm.amdgcn.mfma.f32.16x16x32.f16")
    mfma_32x32x8 = _count_calls(llvm_ir, "@llvm.amdgcn.mfma.f32.32x32x8f16")
    mfma_32x32x16 = _count_calls(llvm_ir, "@llvm.amdgcn.mfma.f32.32x32x16.f16")
    mfma_4x4x4 = _count_calls(llvm_ir, "@llvm.amdgcn.mfma.f32.4x4x4f16")

    async_calls = _count_calls(llvm_ir, "@llvm.amdgcn.raw.ptr.buffer.load.lds")
    raw_load_calls = _count_calls(llvm_ir, "@llvm.amdgcn.raw.ptr.buffer.load")
    raw_store_calls = _count_calls(llvm_ir, "@llvm.amdgcn.raw.ptr.buffer.store")
    # The async intrinsic name contains raw.ptr.buffer.load, so subtract it.
    raw_load_calls -= async_calls

    return LlvmIrStats(
        mfma_calls=(
            mfma_16x16x16 + mfma_16x16x32 + mfma_32x32x8 + mfma_32x32x16 + mfma_4x4x4
        ),
        mfma_16x16x16=mfma_16x16x16,
        mfma_16x16x32=mfma_16x16x32,
        mfma_32x32x8=mfma_32x32x8,
        mfma_32x32x16=mfma_32x32x16,
        mfma_4x4x4=mfma_4x4x4,
        raw_buffer_load_calls=raw_load_calls,
        raw_buffer_store_calls=raw_store_calls,
        async_buffer_load_lds_calls=async_calls,
        global_vector_loads=_count_lines(llvm_ir, "load <", "addrspace(1)"),
        global_vector_stores=_count_lines(llvm_ir, "store <", "addrspace(1)"),
        smem_vector_loads=_count_lines(llvm_ir, "load <", "addrspace(3)"),
        smem_vector_stores=_count_lines(llvm_ir, "store <", "addrspace(3)"),
        barriers=_count_calls(llvm_ir, "@llvm.amdgcn.s.barrier"),
        waitcnts=_count_calls(llvm_ir, "@llvm.amdgcn.s.waitcnt"),
        sched_group_barriers=_count_calls(llvm_ir, "@llvm.amdgcn.sched.group.barrier"),
        sched_barriers=_count_calls(llvm_ir, "@llvm.amdgcn.sched.barrier"),
    )
