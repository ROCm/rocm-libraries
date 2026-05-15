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

    Attributes
    ----------
    num_iters
        Iteration count (Python int).
    double_buffer
        Legacy boolean: ``True`` enables 2-buffer rotation;
        equivalent to ``num_buffers=2``.
    num_buffers
        Number of LDS buffers in the rotation (1 = single buffer,
        2 = classic double buffer / ping-pong, 4 = quad buffer with
        2x prefetch depth — the canonical pattern for keeping more
        VMEM loads outstanding to hide DRAM latency). When >1, each
        iter rotates ``cur = buffers[it % num_buffers]`` and
        prefetches ``buffers[(it + num_buffers - 1) % num_buffers]``.
    wait_vmcnt
        Insert ``s_waitcnt(vmcnt=...)`` before each compute step.
    sync_after_wait
        Insert workgroup barrier after the VMEM wait.
    sync_before_issue
        Insert workgroup barrier before the next ``issue_load`` to close
        the iter-N compute → iter-N+2 issue ABA hazard window.
    overlap_vmcnt
        Use ``s_waitcnt(vmcnt=num_buffers-1)`` (partial drain) instead of
        ``vmcnt(0)`` so prefetched loads stay in flight across compute.
        Pairs with ``sync_lds_only()`` barriers that don't drain VMEM.
    """

    num_iters: int
    double_buffer: bool = True
    wait_vmcnt: bool = False
    sync_after_wait: bool = True
    sync_before_issue: bool = True
    overlap_vmcnt: bool = False
    num_buffers: int = 0  # 0 = derive from double_buffer (legacy)

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

        # Derive buffer count: prefer explicit `num_buffers`, fall back
        # to the legacy `double_buffer` boolean.
        if self.num_buffers > 0:
            nb = self.num_buffers
        elif self.double_buffer:
            nb = 2
        else:
            nb = 1

        if nb > len(buffers):
            raise ValueError(
                f"SoftwarePipeline: num_buffers={nb} but only "
                f"{len(buffers)} buffer pair(s) supplied"
            )
        rotating = nb > 1
        prefetch_depth = nb - 1  # iters in flight ahead of current

        # Prologue: issue the first `prefetch_depth` loads so the
        # steady-state can immediately overlap them with compute.
        for p in range(min(prefetch_depth, self.num_iters)):
            issue_load(p, buffers[p % nb])

        state = initial_state
        for it in range(self.num_iters):
            cur = buffers[it % nb] if rotating else buffers[0]
            # Next load goes into the buffer slot that will be re-used
            # `prefetch_depth` iters from now (so iter `it+prefetch_depth`
            # lives in the slot freed by iter `it`).
            issue_idx = it + prefetch_depth
            has_next = rotating and issue_idx < self.num_iters
            if has_next:
                nxt = buffers[issue_idx % nb]
                if it > 0 and self.sync_before_issue:
                    # Close the N-step ABA window: barrier before the
                    # next async-load overwrites a buffer that may
                    # still be in flight from iter (it - prefetch_depth).
                    if self.overlap_vmcnt:
                        b.sync_lds_only()
                    else:
                        b.sync()
                issue_load(issue_idx, nxt)
            if self.wait_vmcnt:
                if self.overlap_vmcnt and has_next:
                    # Drain everything except the most-recent
                    # `prefetch_depth` outstanding async loads so they
                    # keep streaming while compute proceeds.
                    b.s_waitcnt(vmcnt=prefetch_depth)
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
            if not rotating:
                b.sync()
        return state
