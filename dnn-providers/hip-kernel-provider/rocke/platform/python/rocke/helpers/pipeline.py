# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reusable software-pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from ..core.ir import IRBuilder
from .schedule import SchedulePolicy


BufferPair = Tuple[Any, Any]
LoadFn = Callable[[int, BufferPair], None]
ComputeFn = Callable[[int, BufferPair, Any], Any]


# Byte size per element for the dtype names rocke uses elsewhere
# (matches the map in core/lower_llvm.py plus the 8-bit fp variants).
_DTYPE_BYTES = {
    "i8": 1,
    "fp8e4m3": 1,
    "bf8e5m2": 1,
    "f8": 1,
    "bf8": 1,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "f32": 4,
    "i64": 8,
    "f64": 8,
}


def _dtype_bytes(dtype: Union[str, int]) -> int:
    """Element byte size from a dtype name (``"f16"``) or an explicit int."""
    if isinstance(dtype, int):
        if dtype <= 0:
            raise ValueError(f"dtype byte size must be positive, got {dtype}")
        return dtype
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown dtype {dtype!r}; pass one of {sorted(_DTYPE_BYTES)} "
            f"or an explicit byte-size int"
        ) from exc


def recommend_prefetch_stages(
    tile_m: int,
    tile_n: int,
    kper: int,
    dtype_a: Union[str, int],
    dtype_b: Union[str, int],
    block_size: int,
    *,
    wave_size: int = 64,
    lds_budget: int = 32768,
) -> int:
    """Bandwidth-derived prefetch depth, clamped to ``[2, 8]``.

    Closed-form port of CK's classic-XDLOPS auto-tuner
    (``blockwise_gemm_pipeline_xdlops_v2.hpp:147-154``)::

        WgpPerCU = max(4 * WaveSize / BlockSize, 1)
        FullMemBandPrefetchStages =
            ceil( (32768 / WgpPerCU)
                  / ((MPerBlock*sizeof(A) + NPerBlock*sizeof(B)) * KPerBlock) )
        PrefetchStages = clamp(FullMemBandPrefetchStages, 2, 8)

    All inner divisions follow CK's C++ semantics exactly: ``4*WaveSize/BlockSize``
    and ``32768/WgpPerCU`` are integer (floor) divisions, and the outer division is
    ``math::integer_divide_ceil`` (``(x + y - 1) // y``). The result keeps the full
    HBM->LDS bandwidth in flight given the per-CU LDS budget (32 KB) and the
    per-tile byte footprint.

    Parameters
    ----------
    tile_m, tile_n
        Block tile extents (``MPerBlock`` / ``NPerBlock``).
    kper
        ``KPerBlock`` reduction tile.
    dtype_a, dtype_b
        Operand dtypes, as a rocke name (``"f16"``, ``"f8"``, ...) or an explicit
        element byte size.
    block_size
        Threads per block.
    wave_size
        Hardware wave size (64 for CDNA / gfx9xx; defaulted so existing callers
        need not supply it).
    lds_budget
        Per-CU LDS budget in bytes (32768 on CDNA; exposed for completeness).

    Returns
    -------
    int
        Prefetch depth in ``[2, 8]``, usable as ``SoftwarePipeline.num_buffers``.
    """
    if min(tile_m, tile_n, kper, block_size, wave_size, lds_budget) <= 0:
        raise ValueError("all sizes must be positive")
    bytes_a = _dtype_bytes(dtype_a)
    bytes_b = _dtype_bytes(dtype_b)

    wgp_per_cu = (4 * wave_size) // block_size
    if wgp_per_cu < 1:
        wgp_per_cu = 1

    numer = lds_budget // wgp_per_cu  # integer (floor) division, as in CK
    denom = (tile_m * bytes_a + tile_n * bytes_b) * kper
    full_mem_band = (numer + denom - 1) // denom  # math::integer_divide_ceil

    if full_mem_band < 2:
        return 2
    if full_mem_band > 8:
        return 8
    return full_mem_band


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

    def _ping_pong_phase(
        self,
        b: IRBuilder,
        *,
        k_cur: Any,
        k_next: Any,
        cur_buf: BufferPair,
        nxt_buf: BufferPair,
        state: Sequence[Any],
        issue_load_fn: Callable[[Any, BufferPair], None],
        compute_fn: Callable[[Any, BufferPair, Any], Any],
        schedule: Optional[SchedulePolicy],
    ) -> list:
        """One prefetch+compute phase of the dynamic ping-pong.

        Emits the same barrier / ``s_waitcnt`` sequence as one steady-state
        iteration of :meth:`run_ping_pong` with ``num_buffers=2``: issue the
        next tile into the buffer the current compute is *not* reading, wait
        for everything except that just-issued load, then compute.
        """
        if self.sync_before_issue:
            # Close the ABA window: every wave must be done reading
            # ``nxt_buf`` (two phases ago) before we overwrite it.
            if self.overlap_vmcnt:
                b.sync_lds_only()
            else:
                b.sync()
        issue_load_fn(k_next, nxt_buf)
        if self.wait_vmcnt:
            # prefetch_depth == 1: leave the just-issued load in flight.
            b.s_waitcnt(vmcnt=1 if self.overlap_vmcnt else 0)
        if self.sync_after_wait:
            if self.overlap_vmcnt:
                b.sync_lds_only()
            else:
                b.sync()
        if schedule is not None:
            schedule.emit_compute_prologue(b)
        state = compute_fn(k_cur, cur_buf, list(state))
        if schedule is not None:
            schedule.emit_compute_epilogue(b)
        return list(state)

    def run_ping_pong_dynamic(
        self,
        b: IRBuilder,
        *,
        k_extent: Any,
        block_k: int,
        k_lo: Any = None,
        k_zero_fill: Any = None,
        buffers: Sequence[BufferPair],
        iter_args: Sequence[Tuple[str, Any]],
        issue_load_fn: Callable[[Any, BufferPair], None],
        compute_fn: Callable[[Any, BufferPair, Any], Any],
        schedule: Optional[SchedulePolicy] = None,
    ) -> list:
        """Double-buffered ping-pong over a **runtime** reduction extent.

        Unlike :meth:`run_ping_pong`, the trip count is not known at build
        time: ``k_extent`` is an i32 SSA ``Value`` holding the reduction
        extent in *elements* (e.g. the ``p_K_gemm`` kernel arg), and
        ``block_k`` is the compile-time tile width.

        Why the body is unrolled 2x
        ---------------------------
        An LDS allocation is a build-time SSA value, so ``buffers[it % 2]``
        cannot be evaluated against a runtime ``it``. Unrolling the body
        twice and stepping by ``2 * block_k`` binds each phase to a
        Python-time buffer constant while still alternating them, which is
        what makes it a real ping-pong rather than a single-buffer loop
        wearing one.

        Odd tile counts
        ---------------
        When the tile count is odd the second phase of the final body
        iteration addresses ``k >= k_extent``. At the real end of the tensor
        those coords fall outside the descriptor's padded bounds, so the
        buffer resource returns zero and a zero tile contributes nothing to
        the accumulator — the same zero-fill the single-buffer path already
        relies on for a partial last tile. Issuing it unconditionally keeps
        the loop free of divergent control flow around barriers.

        That breaks when ``k_extent`` is a split-K slice end inside the
        tensor: the tile at ``k_extent`` is the *next* slice's first tile and
        reads as real data. Pass ``k_zero_fill`` -- an offset whose tile is
        known to read as zero, i.e. the global reduction extent -- and the
        phase-A prefetch is redirected there when it lands at or past
        ``k_extent``. That is one scalar select per iteration, and the barrier
        sequence stays uniform.

        ``k_lo`` is the first tile offset; it defaults to 0.  Split-K passes
        the slice base here so the loop walks ``[k_lo, k_extent)`` -- the
        reduction is sliced by offsetting both ends, not by rebasing the
        descriptors.

        ``iter_args`` is a sequence of ``(name, init_value)`` pairs, exactly
        as :meth:`~rocke.core.ir.IRBuilder.scf_for_iter` expects.
        ``issue_load_fn(k_offset, buf_pair)`` and
        ``compute_fn(k_offset, buf_pair, state)`` both receive i32 SSA
        values for the tile offset.

        Only the 2-buffer rotation is supported; for single-buffer or
        4-buffer modes use the compile-time variant.
        """
        if len(buffers) < 2:
            raise ValueError(
                f"run_ping_pong_dynamic needs 2 buffer pairs, got {len(buffers)}"
            )
        if block_k <= 0:
            raise ValueError(f"block_k must be positive, got {block_k}")

        buf0, buf1 = buffers[0], buffers[1]
        c_bk = b.const_i32(block_k)
        c_2bk = b.const_i32(2 * block_k)
        k_first = b.const_i32(0) if k_lo is None else k_lo

        # Prologue: stage the first tile into buf0 so the first phase's
        # compute has something to read. Every later tile is staged by the
        # phase before the one that consumes it.
        issue_load_fn(k_first, buf0)

        for_op = b.scf_for_iter(
            k_first, k_extent, c_2bk, list(iter_args), iv_name="k_pipe"
        )
        with for_op as entered:
            # scf_for_iter hands back the bare induction variable when there
            # is no loop-carried state, and (iv, vars) otherwise.
            k, loop_vars = entered if iter_args else (entered, ())
            state = list(loop_vars)
            k1 = b.add(k, c_bk)
            k2 = b.add(k, c_2bk)
            k1_load = (
                k1
                if k_zero_fill is None
                else b.select(b.cmp_lt(k1, k_extent), k1, k_zero_fill)
            )
            # Phase A: compute tile k out of buf0 while tile k+1 streams
            # into buf1.
            state = self._ping_pong_phase(
                b,
                k_cur=k,
                k_next=k1_load,
                cur_buf=buf0,
                nxt_buf=buf1,
                state=state,
                issue_load_fn=issue_load_fn,
                compute_fn=compute_fn,
                schedule=schedule,
            )
            # Phase B: the buffers swap roles.
            state = self._ping_pong_phase(
                b,
                k_cur=k1,
                k_next=k2,
                cur_buf=buf1,
                nxt_buf=buf0,
                state=state,
                issue_load_fn=issue_load_fn,
                compute_fn=compute_fn,
                schedule=schedule,
            )
            b.scf_yield(*state)

        # The final phase left a prefetch in flight and (with overlap_vmcnt)
        # only an LDS-scoped barrier behind it. Drain both before the
        # epilogue, which stages its own data through the same LDS.
        b.s_waitcnt(vmcnt=0)
        b.sync()
        return list(for_op.results)
