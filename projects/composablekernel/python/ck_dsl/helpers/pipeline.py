# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reusable software-pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from ..core.ir import IRBuilder
from .schedule import SchedulePolicy


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
    sync_before_issue: bool = True
    overlap_vmcnt: bool = False

    def run_ping_pong(
        self,
        b: IRBuilder,
        *,
        buffers: Sequence[BufferPair],
        initial_state: Any,
        issue_load: LoadFn,
        compute: ComputeFn,
        schedule: Optional[SchedulePolicy] = None,
    ) -> Any:
        """Run a double-buffered ping-pong pipeline over `num_iters`.

        Per-iter schedule (for `double_buffer=True`):
            1. (it > 0 and `sync_before_issue`) workgroup barrier so all
               waves have finished reading from the buffer that the
               next async load is about to overwrite. Without this
               barrier the ds_reads from the previous iter's `compute`
               can race with the LDS writes of the next `issue_load`
               against the *same* LDS buffer (the two-iter ABA pong
               hazard) — producing silent data corruption that scales
               with workgroup-count and only shows up at large grids.
            2. `issue_load(it+1, buffers[(it+1) & 1])` if it+1 < num_iters.
            3a. If `overlap_vmcnt`: emit `s_waitcnt(vmcnt=1)` (drain
                everything *except* the just-issued load, so it stays
                in flight while `compute(it)` runs). The very last
                iter, which doesn't issue a next load, drops to
                `vmcnt=0`.
            3b. Else: `s_waitcnt(vmcnt=0)` drains all VMEM (no overlap
                with compute; matches the conservative pre-fix path).
            4. `b.sync()` so all waves agree the iter-(it) LDS write is
               visible before any wave starts reading it.
            5. `compute(it, buffers[it & 1], state)`.

        For the single-buffer path the same barrier is replaced by the
        existing post-compute `b.sync()` (which serializes the buffer).
        """
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
            has_next = self.double_buffer and it + 1 < self.num_iters
            if has_next:
                nxt = buffers[(it + 1) & 1]
                if it > 0 and self.sync_before_issue:
                    if self.overlap_vmcnt:
                        b.sync_lds_only()
                    else:
                        b.sync()
                issue_load(it + 1, nxt)
            if self.wait_vmcnt:
                if self.overlap_vmcnt and has_next:
                    b.s_waitcnt(vmcnt=1)
                else:
                    b.s_waitcnt(vmcnt=0)
            if self.sync_after_wait:
                if self.overlap_vmcnt and has_next:
                    b.sync_lds_only()
                else:
                    b.sync()
            if schedule is not None:
                schedule.emit_compute_prologue(b)
            state = compute(it, cur, state)
            if schedule is not None:
                schedule.emit_compute_epilogue(b)
            if not self.double_buffer:
                b.sync()
        return state
