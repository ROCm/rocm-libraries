# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduling policies.

The AMDGPU backend still performs final instruction scheduling, but CK-style
kernels often need explicit scheduler hints around MFMA / LDS / VMEM groups.
This module centralizes those hints so instance builders do not hard-code magic
mask constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..analysis.ir import LlvmIrStats
from ..core.ir import IRBuilder


DS_READ = 0x100
DS_WRITE = 0x200
VMEM_READ = 0x020
VMEM_WRITE = 0x040
MFMA = 0x008


@dataclass(frozen=True)
class SchedulePolicy:
    """Named scheduler hint policy for an MFMA hot loop."""

    name: str = "mem"
    emit_hints: bool = False
    setprio_level: Optional[int] = None

    @classmethod
    def for_pipeline(cls, pipeline: str) -> "SchedulePolicy":
        if pipeline == "mem":
            return cls(name="mem", emit_hints=False)
        if pipeline == "compv3":
            return cls(name="compv3", emit_hints=True)
        if pipeline == "compv4":
            return cls(name="compv4", emit_hints=True, setprio_level=1)
        if pipeline == "async_dma":
            return cls(name="async_dma", emit_hints=True, setprio_level=1)
        raise ValueError(f"unknown schedule policy {pipeline!r}")

    def emit_prologue(self, b: IRBuilder) -> None:
        if self.setprio_level is not None:
            b.s_setprio(self.setprio_level)

    def emit_after_mfma_step(
        self,
        b: IRBuilder,
        *,
        ds_read_count: int,
        mfma_count: int,
    ) -> None:
        """Emit a DS_READ group followed by an MFMA group hint."""
        if not self.emit_hints:
            return
        b.sched_group_barrier(DS_READ, int(ds_read_count), 0)
        b.sched_group_barrier(MFMA, int(mfma_count), 0)

    def assert_expected_ir(self, stats: LlvmIrStats) -> None:
        """Lightweight sanity check against lowered LLVM IR stats."""
        if self.emit_hints and stats.sched_group_barriers == 0:
            raise AssertionError(
                f"schedule policy {self.name} expected sched_group_barrier ops"
            )
