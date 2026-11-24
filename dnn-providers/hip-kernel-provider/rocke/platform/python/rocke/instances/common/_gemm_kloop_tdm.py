# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The universal GEMM's Tensor Data Mover K-loops.

:func:`~rocke.instances.common.gemm_universal.build_universal_gemm` hands every
``tdm`` spec to :meth:`TdmKLoop.emit`, which picks one of four loop shapes by
``tdm_depth`` and the K-loop traits:

========================  ==========================================
``tdm_depth`` 1           :meth:`TdmKLoop.unpipelined`: issue, drain, compute
``tdm_depth`` 2           :meth:`TdmKLoop.ping_pong`: two LDS halves
``tdm_depth`` 2 + early   :meth:`TdmKLoop.early_refill`
refill
``tdm_depth`` 3 and 4     :meth:`TdmKLoop.ring`: a modular LDS ring
========================  ==========================================

Depths 1 and 2 keep their own loop shapes rather than being special cases of
the ring, so the ring cannot reshape the depth-2 form the tuned shapes ship.

The C++ engine mirrors this module in ``cpp/instances/common/
gemm_build_kloop_tdm.cpp``. Every builder call allocates an SSA name, so both
engines emit the same calls in the same order; reordering any of them breaks
byte identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

from ...core.ir import IRBuilder, Value

# Workgroup ids of the split barrier: signal / wait the whole workgroup.
_WG_SIGNAL, _WG_WAIT = 0xFFFFFFFF, 0xFFFF


@dataclass(frozen=True)
class TdmKLoop:
    """What the TDM K-loops need from the GEMM builder.

    ``load``, ``wmma`` and ``mma`` are the builder's ``emit_load_phase``,
    ``_emit_wmma_phase`` and ``emit_mfma_phase``; ``accs`` the accumulator
    loop arguments; ``waves`` and ``wave_id`` the launched wave count and this
    wave's index (the TDM issuers).
    """

    b: IRBuilder
    trait: Any
    a_smem: Value
    b_smem: Value
    load: Callable[..., None]
    wmma: Callable[..., List[Value]]
    mma: Callable[..., List[Value]]
    accs: Sequence[Any]
    k_lo: Value
    k_upper: Value
    block_k: int
    c_block_k: Value
    c0: Value
    waves: int
    wave_id: Value

    def emit(self) -> Sequence[Value]:
        """Emit the K-loop for ``trait.tdm_depth``; return the accumulators."""
        if self.trait.tdm_depth >= 3:
            return self.ring()
        if self.trait.tdm_depth == 2:
            if self.trait.tdm_early_refill:
                return self.early_refill()
            return self.ping_pong()
        return self.unpipelined()

    # --- shared pieces --------------------------------------------------------

    def _publish(self, allowed: int) -> None:
        """Publish TDM fills: wait until at most ``allowed`` are outstanding,
        then signal the split barrier."""
        self.b.s_wait_tensorcnt(allowed)
        self.b.s_barrier_signal(_WG_SIGNAL)

    def _signal(self, allowed: int) -> None:
        """A K tile's split-barrier signal: drain this wave's fragment reads,
        then publish."""
        self.b.s_waitcnt(lgkmcnt=0)
        self._publish(allowed)

    def _wait(self) -> None:
        self.b.s_barrier_wait(_WG_WAIT)

    def _drain(self) -> None:
        """Drain every TDM fill, then rendezvous on LDS. TENSORcnt ticks once
        per descriptor, so a full drain is 0 regardless of tile shape."""
        self.b.s_wait_tensorcnt(0)
        self.b.sync_lds_only()

    def _k_tile_count(self) -> Value:
        """ceil((K - k_lo) / block_k): the K tiles of this K range. The last
        one is partial when K is not a multiple of block_k; the mover clips it
        to K."""
        b = self.b
        return b.div(
            b.add(b.sub(self.k_upper, self.k_lo), b.const_i32(self.block_k - 1)),
            self.c_block_k,
        )

    # --- the four loop shapes -------------------------------------------------

    def unpipelined(self) -> Sequence[Value]:
        """``tdm_depth`` 1: issue the tile, drain it, compute, barrier.
        Unlike ASYNCcnt, which counts instructions, TENSORcnt drains to 0 for
        any tile."""
        b = self.b
        for_op = b.scf_for_iter(
            self.k_lo, self.k_upper, self.c_block_k, self.accs, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            self.load(self.a_smem, self.b_smem, k0)
            self._drain()
            new_accs = self.mma(self.a_smem, self.b_smem, iter_vars)
            b.sync()
            b.scf_yield(*new_accs)
        return for_op.results

    def ping_pong(self) -> Sequence[Value]:
        """``tdm_depth`` 2: two LDS halves, the next tile's fill overlapping
        the current tile's WMMAs.

        The prologue loads tile 0 into half 0 and the loop runs the first
        K - 1 tiles, each waiting for its own tile, issuing the next one into
        the other half and computing; the last tile is computed after the
        loop. With the split barrier, every wait (each iteration's top, and
        the last tile's) pairs with the signal raised one phase earlier -- the
        prologue's for the first iteration, then the one inside each
        iteration's WMMA phase. A signal follows the wave's own drains, so
        arriving at the wait means every wave has finished reading the half
        about to be refilled and the fill of the tile about to be read has
        landed.
        """
        b = self.b
        split = self.trait.split_barrier

        # Prologue: load tile 0 into half 0 (the slice base in split-K mode).
        self.load(self.a_smem, self.b_smem, self.k_lo, lds_parity=0)
        if split:
            self._publish(0)

        k_minus_one_tile = b.sub(self.k_upper, self.c_block_k)
        c1_i32 = b.const_i32(1)
        loop_args = [("par", self.c0)] + list(self.accs)
        for_op = b.scf_for_iter(
            self.k_lo, k_minus_one_tile, self.c_block_k, loop_args, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            parity = iter_vars[0]
            acc_iter = iter_vars[1:]
            next_parity = b.sub(c1_i32, parity)  # 1 - parity (0/1 only)
            k_next = b.add(k0, self.c_block_k)
            # Only the current tile's descriptors are outstanding here -- the
            # next tile is issued after the drain -- so a full TENSORcnt drain
            # still leaves the copy overlapped with the previous WMMAs.
            if split:
                self._wait()
            else:
                self._drain()
            self.load(self.a_smem, self.b_smem, k_next, lds_parity=next_parity)
            if split:
                new_accs = self.wmma(
                    self.a_smem,
                    self.b_smem,
                    acc_iter,
                    lds_parity=parity,
                    after_last_reads=lambda: self._signal(0),
                )
            else:
                new_accs = self.mma(
                    self.a_smem, self.b_smem, acc_iter, lds_parity=parity
                )
            b.scf_yield(next_parity, *new_accs)

        # The last tile: already loaded into the final parity.
        final_parity = for_op.results[0]
        if split:
            self._wait()
        else:
            self._drain()
        return self.mma(
            self.a_smem, self.b_smem, for_op.results[1:], lds_parity=final_parity
        )

    def early_refill(self) -> Sequence[Value]:
        """``tdm_depth`` 2 with ``tdm_early_refill``.

        Waves ``2p``/``2p+1`` own half ``p`` (alternating issuers). For K-tile
        ``i`` in half ``p``, after the wave's split-barrier signal and the
        first row of the last k-atom's WMMAs, the pair owning half ``p`` waits
        on that barrier (every wave has finished reading half ``p``) and
        refills it with tile ``i + 2``; the wait is issued right before the TDM
        load so the descriptor arithmetic overlaps it. Every other wave
        consumes the same barrier phase at the top of tile ``i + 1``, so each
        wave waits exactly once per phase. The pair's signal for tile ``i + 1``
        waits for the tile it refilled two steps earlier, which is what makes
        tile ``i + 2`` visible to the waves that read it.

        The loop runs two tiles per iteration with static parity, so the LDS
        reads use immediate offsets, over an even tile count (see below).
        """
        b = self.b
        c1 = b.const_i32(1)
        c2 = b.const_i32(2)
        bk2 = b.const_i32(2 * self.block_k)
        pair = b.div(self.wave_id, c2)

        def _parity(p):
            return p if isinstance(p, Value) else b.const_i32(p)

        def _other(p):
            return b.sub(c1, p) if isinstance(p, Value) else b.const_i32(1 - p)

        def _top_wait(p) -> None:
            # The pair owning the other half already consumed this phase when
            # it refilled that half.
            with b.scf_if(b.cmp_ne(pair, _other(p))):
                self._wait()

        def _refill(p, k_cur, refill: bool) -> None:
            with b.scf_if(b.cmp_eq(pair, _parity(p))):
                if refill:
                    self.load(
                        self.a_smem,
                        self.b_smem,
                        b.add(k_cur, bk2),
                        lds_parity=p,
                        before_issue=self._wait,
                    )
                else:
                    self._wait()

        def _tile(p, k_cur, acc, refill: bool) -> List[Value]:
            _top_wait(p)
            return self.wmma(
                self.a_smem,
                self.b_smem,
                acc,
                lds_parity=p,
                after_last_reads=lambda: self._signal(0),
                after_first_row=lambda: _refill(p, k_cur, refill),
            )

        # Prologue: tiles 0 and 1 into halves 0 and 1. Pair 1 consumes the
        # prologue phase here, as if it had refilled half 1 after tile -1.
        self.load(self.a_smem, self.b_smem, self.k_lo, lds_parity=0)
        self.load(
            self.a_smem, self.b_smem, b.add(self.k_lo, self.c_block_k), lds_parity=1
        )
        self._publish(0)
        with b.scf_if(b.cmp_eq(pair, c1)):
            self._wait()

        # The loop runs an even number of tiles so every parity is static (a
        # runtime-parity tail spills the steady state at the 256-VGPR budget).
        # The tile count is rounded up first, so a partial last tile (K not a
        # multiple of block_k) is loaded; the mover clips it to K. With an odd
        # count the extra tile lies past K: its descriptor's remaining-K extent
        # is zero, the mover fills it with zeros, and its WMMAs add nothing.
        n_tiles = self._k_tile_count()
        n_even = b.add(n_tiles, b.mod(n_tiles, c2))
        n_pairs = b.div(b.sub(n_even, c2), c2)
        k_pairs_end = b.add(self.k_lo, b.mul(n_pairs, bk2))
        pairs = b.scf_for_iter(
            self.k_lo, k_pairs_end, bk2, list(self.accs), iv_name="k0"
        )
        with pairs as (k0, acc_iter):
            acc = _tile(0, k0, list(acc_iter), refill=True)
            acc = _tile(1, b.add(k0, self.c_block_k), acc, refill=True)
            b.scf_yield(*acc)

        # Tail: tile T-2 (half 0) has nothing left to refill; tile T-1 (half 1)
        # raises no signal.
        acc = _tile(0, k_pairs_end, list(pairs.results), refill=False)
        _top_wait(1)
        return self.wmma(self.a_smem, self.b_smem, acc, lds_parity=1)

    def ring(self) -> Sequence[Value]:
        """``tdm_depth`` D >= 3: a D-buffer LDS ring with a partial wait.

        Body order and barrier count are chosen so that at any given point
        ``D - 2`` descriptor fills are in flight, one buffer is being read by
        the WMMAs, and one is being written by the fill that just finished.
        The partial wait count ``(D-2)*per_tile`` is a constant because
        TENSORcnt retires in order. The prologue fills are clamped so every
        fill is real, and a single ``s_wait_tensorcnt(0)`` after the loop
        drains the look-ahead fills.
        """
        b = self.b
        D = self.trait.tdm_depth
        # Descriptor sets per tile per issuing wave.
        per_tile = 1 if self.waves > 1 else 2
        allowed = (D - 2) * per_tile
        ahead_k = (D - 1) * self.block_k
        # Last in-bounds tile origin, block_k-aligned from k_lo. Never below
        # k_lo: with no K tile at all (K = 0) every prologue fill starts at k_lo
        # with an empty K extent instead of one tile before A and B.
        last_origin = b.smax(
            b.add(
                self.k_lo,
                b.mul(b.sub(self._k_tile_count(), b.const_i32(1)), self.c_block_k),
            ),
            self.k_lo,
        )

        # Prologue: fill ring slots 0 .. D-2 (clamped so every fill is real).
        for j in range(D - 1):
            self.load(
                self.a_smem,
                self.b_smem,
                b.smin(b.add(self.k_lo, b.const_i32(j * self.block_k)), last_origin),
                lds_parity=j,
            )

        # Split barrier: the joint barrier at each iteration's top becomes a
        # wait there, paired with a signal raised once the previous iteration's
        # last fragment reads have drained and its tile + 1 has landed -- the
        # same ``allowed`` count, since the outstanding fills (tiles k+1 ..
        # k+D-1) are the same set at both points. The prologue raises the
        # first signal and a wait after the loop consumes the last one.
        split = self.trait.split_barrier
        if split:
            self._publish(allowed)

        c_ring = b.const_i32(D)
        c_ahead = b.const_i32(D - 1)
        c_one = b.const_i32(1)
        loop_args = [("ring", self.c0)] + list(self.accs)
        for_op = b.scf_for_iter(
            self.k_lo, self.k_upper, self.c_block_k, loop_args, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            ring = iter_vars[0]
            acc_iter = iter_vars[1:]
            if split:
                self._wait()
            else:
                b.s_wait_tensorcnt(allowed)
                b.sync_lds_only()
            self.load(
                self.a_smem,
                self.b_smem,
                b.smin(b.add(k0, b.const_i32(ahead_k)), last_origin),
                lds_parity=b.mod(b.add(ring, c_ahead), c_ring),
            )
            if split:
                new_accs = self.wmma(
                    self.a_smem,
                    self.b_smem,
                    acc_iter,
                    lds_parity=ring,
                    after_last_reads=lambda: self._signal(allowed),
                )
            else:
                new_accs = self.mma(self.a_smem, self.b_smem, acc_iter, lds_parity=ring)
            b.scf_yield(b.mod(b.add(ring, c_one), c_ring), *new_accs)

        if split:
            self._wait()
        # Drain the in-flight look-ahead fills before the epilogue reuses LDS.
        b.s_wait_tensorcnt(0)
        return for_op.results[1:]
