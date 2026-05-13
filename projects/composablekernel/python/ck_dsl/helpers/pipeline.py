# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reusable software-pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple

from ..core.ir import IRBuilder


BufferPair = Tuple[Any, Any]
LoadFn = Callable[[int, BufferPair], None]
ComputeFn = Callable[[int, BufferPair, Any], Any]


@dataclass(frozen=True)
class SoftwarePipeline:
    """Static prologue / steady-state / epilogue pipeline.

    This helper intentionally works with Python-time iteration counts. It is
    meant for specialized kernels where the reduction tile count is known from
    the problem spec and unrolling gives the scheduler more freedom.
    """

    num_iters: int
    double_buffer: bool = True
    wait_vmcnt: bool = False
    sync_after_wait: bool = True

    def run_ping_pong(
        self,
        b: IRBuilder,
        *,
        buffers: Sequence[BufferPair],
        initial_state: Any,
        issue_load: LoadFn,
        compute: ComputeFn,
    ) -> Any:
        if self.num_iters <= 0:
            return initial_state
        if not buffers:
            raise ValueError("SoftwarePipeline needs at least one buffer pair")
        if self.double_buffer and len(buffers) < 2:
            raise ValueError("double-buffered pipeline needs two buffer pairs")

        issue_load(0, buffers[0])
        state = initial_state
        for it in range(self.num_iters):
            cur = buffers[it & 1] if self.double_buffer else buffers[0]
            if self.double_buffer and it + 1 < self.num_iters:
                nxt = buffers[(it + 1) & 1]
                issue_load(it + 1, nxt)
            if self.wait_vmcnt:
                b.s_waitcnt(vmcnt=0)
            if self.sync_after_wait:
                b.sync()
            state = compute(it, cur, state)
            if not self.double_buffer:
                b.sync()
        return state
