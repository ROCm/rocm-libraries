# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""The wide-head_dim kernel body: no dual-wave pipeline, D staged and sharded.

Serves head_dim 384 and 512. A separate file from
`flash_attn_func_gfx950.py` because it is a **different algorithm**, not a
wider setting of the same one -- only the ABI and the helpers are shared.

--- Why the dual-wave pipeline does not come along ---------------------------

The production schedule keeps two KV tiles in flight across eight clusters and
splits the softmax between them. It is worth its cost up to head_dim 256 and
cannot be paid for above it, for two independent reasons:

- **LDS.** Two tiles in flight means two resident K+V buffers, and LDS is
  `BLOCK_N * head_dim * ~8.3 B`. head_dim 384 needs 204288 B and 512 needs
  272384 B against a 163840 B cap. The prefetch distance the schedule is built
  on is simply not purchasable.
- **Registers.** Two tiles in flight also means two of everything the loop
  carries -- `v_s_0/v_s_1`, `v_p_0/v_p_1`, two K sets, two V sets.

So this body runs one tile at a time. It does *not* give up prefetching --
`D_STAGES` buys back a 2-deep pipeline over stages, alternating the same two
LDS buffers the dual-wave path uses for two tiles. ATT is what forced that:
the first cut drained after every DMA, and
`s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` came to 26% of runtime at 749
cycles a hit, against 10.8% for every MFMA combined. Pipelining the stages
was worth +27% at 384 and +21% at 512, and prefetching *across* the two
phases -- V's first stage during QK, the next tile's first K stage during PV --
a further +9%. Only the loop prologue now exposes a DMA.

--- The two D-axis cuts, and why both are needed -----------------------------

Removing the dual-wave duplication is *not sufficient*, and measurement is what
settled that. With it already gone, head_dim 512 still spilled 286 registers
and issued 525 `v_accvgpr_write` against 128 MFMAs -- the AGPR file being used
as a spill area rather than as the accumulator. The residue is per-wave data,
not pipelining:

    Q  = ROWS_PER_WAVE * head_dim / 64 / 2 = 128 VGPR   (arch, loop-invariant)
    O  = ROWS_PER_WAVE * head_dim / 64     = 256 VGPR   (the whole AGPR file)

`ROWS_PER_WAVE` is pinned at 32 by the MFMA's M extent, so neither term can be
reduced by tiling differently. The head dim itself has to be split:

- **`D_STAGES`** splits it *in time*. One stage of K, then the next, into the
  same LDS. Answers the LDS wall and bounds the live K/V window.
- **`VO_SHARDS`** splits it *across waves*. Wave *s* accumulates only
  `O[:, slice_s]`, which is what brings the 256 down.

`VO_SHARDS` rather than the stronger `QK_SHARDS` (which would also slice Q) is
a deliberate first cut: sharding only the *output* axis means the shards write
disjoint columns and **never have to agree on anything**, so there is no
cross-wave reduction, no extra barrier, and no summation order to get wrong.
The price is that every shard recomputes the whole S. Since O is twice Q, the
cheap cut is the one that pays -- see `sdpa-close-gap-gfx950.md`.

--- The shard split is stage-major, and that is not arbitrary ----------------

Within stage `st`, shard `s` owns

    dc_global = st * D_CHUNKS_PER_STAGE + s * D_CHUNKS_PER_STAGE_SHARD + i

so each shard has work in *every* stage. Giving each shard a contiguous half of
the head dim instead would be tidier to address, and would idle half the waves
for half the stages: stage 0 stages exactly shard 0's columns into LDS, leaving
the shard-1 waves nothing to do. The cost of the interleave is that a wave's
output columns are no longer contiguous, so the store walks one run per stage.
"""

import contextlib
from dataclasses import replace

from kernels.attention_bwd_cdna.fmha_common_gfx1201 import MaskedAxis
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvLdsToVgprLoader,
    ParitySoftmaxHelper,
    ParityStoreHelper,
)
from kernels.attention_bwd_cdna.gfx950_standalone import dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

__all__ = [
    "WideGemmHelper",
    "WideKernelContext",
    "WideKvLdsToVgprLoader",
    "WideSoftmaxHelper",
    "WideStoreHelper",
    "make_wide_body",
]

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier
_waitcnt_vm_n = dualwave._waitcnt_vm_n


class WideKernelContext(ParityKernelContext):
    """Parity context whose Q rows follow the *Q tile*, not the wave.

    With `VO_SHARDS` waves per Q tile, `wave_id` no longer identifies a row
    block -- `wave_id // VO_SHARDS` does, and the remainder picks which output
    columns the wave owns. The production context sets `wave_q_offset` inline
    in `__init__`, so this recomputes it and re-runs `init_q_row`, which reads
    it and sets nothing else.

    At `VO_SHARDS == 1` every value below is what the base class already
    computed, so the unsharded path is untouched rather than merely equivalent.
    """

    def init_thread_mapping(self):
        super().init_thread_mapping()
        traits = self.traits
        self.q_tile_id = self.wave_id // traits.VO_SHARDS
        self.vo_shard_id = self.wave_id % traits.VO_SHARDS
        if const_expr(traits.VO_SHARDS > 1):
            self.wave_q_offset = self.q_tile_id * traits.ROWS_PER_WAVE

    def init_q_row(self):
        super().init_q_row()
        traits = self.traits
        if const_expr(traits.VO_SHARDS > 1):
            # `q_start_pos_i32` is derived from the *uniform* wave id, which
            # the base class also reads as a row block. Same correction, on the
            # scalar path -- missing it would leave the causal mask keyed to a
            # different row than the one being computed.
            self.q_start_pos_i32 = fx.Int32(
                self.q_start
                + (self.wave_id_uni // traits.VO_SHARDS) * traits.ROWS_PER_WAVE
            )


class WideKvLdsToVgprLoader(ParityKvLdsToVgprLoader):
    """V reads restricted to the wave's own output columns.

    The shard offset is folded into `urv_base` rather than added per read.
    `_swizzled_v_dc_off(dc) = (dc // 2) * PAIR + (dc % 2) * IN_PAIR` splits as
    `_swizzled_v_dc_off(a + i) = (a // 2) * PAIR + _swizzled_v_dc_off(i)` for
    even `a`, so a shard starting on an even chunk contributes a *constant*
    that the base absorbs. `make_traits` enforces that evenness rather than
    leaving it as a silent precondition.
    """

    def load_v_shard(self, buf_id, shard_slot):
        traits = self.traits
        per = traits.D_CHUNKS_PER_STAGE_SHARD
        urv = self.v_lds_read_base_per_lane
        if const_expr(traits.VO_SHARDS > 1):
            # `shard_slot` is a runtime value (`wave_id % VO_SHARDS`), so this
            # is a runtime multiply by a constant, not a folded literal.
            urv = urv + shard_slot * (per // 2) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE
        full = self.traits
        self.traits = replace(full, D_CHUNKS=per, K_STEPS_QK=full.K_STEPS_PER_STAGE)
        try:
            return self.read_v_packs(buf_id, urv)
        finally:
            self.traits = full


class WideSoftmaxHelper(ParitySoftmaxHelper):
    """Softmax whose O-wide operations size themselves to the wave's shard.

    `scale_o` and `_anchor_v_o` walk `D_CHUNKS` accumulators, which is right
    everywhere except here: a sharded wave holds `D_CHUNKS // VO_SHARDS` of
    them, and the full count runs straight off the end of the list. Scoped
    rather than reimplemented, for the same reason as
    `WideKvLdsToVgprLoader` -- the anchor in particular builds an inline-asm
    constraint string from the count, and a second copy of that is not
    something to maintain.
    """

    @contextlib.contextmanager
    def _owned(self):
        """Scope `D_CHUNKS` to what this wave holds. **Idempotent, and it must be.**

        `rescale_o_serial` calls `scale_o`, so the two scopes nest. Written as
        `D_CHUNKS // VO_SHARDS` that composes -- 8 becomes 4 becomes 2 -- and
        the inner call then rescales only half the accumulators. The other half
        keeps a stale magnitude, which is not a crash and not a NaN: it is an
        output that is merely too large, and it cost a bisect to find.

        So the target is stated absolutely rather than as a division.
        `D_CHUNKS_PER_STAGE_SHARD` and `D_STAGES` both survive `replace`, so
        this lands on the same value however many times it is applied.
        """
        full = self.traits
        owned = full.D_CHUNKS_PER_STAGE_SHARD * full.D_STAGES
        if const_expr(full.VO_SHARDS == 1) or full.D_CHUNKS == owned:
            yield
            return
        self.traits = replace(full, D_CHUNKS=owned)
        try:
            yield
        finally:
            self.traits = full

    def scale_o(self, v_o, scale_scalar):
        with self._owned():
            return super().scale_o(v_o, scale_scalar)

    def rescale_o_serial(self, v_o, m_row, l_row, m_tile_max):
        with self._owned():
            return super().rescale_o_serial(v_o, m_row, l_row, m_tile_max)


class WideGemmHelper(ParityGemmHelper):
    """PV that accumulates into the wave's slice of O, indexed locally."""

    def pv_shard(self, v_p, v_v, v_o, local_base):
        traits = self.traits
        per = traits.D_CHUNKS_PER_STAGE_SHARD
        for step in range_constexpr(4):
            v_p_lo, v_p_hi = v_p
            v_pk = v_v[step]
            p_pk = v_p_lo[step] if const_expr(step < 2) else v_p_hi[step - 2]
            for i in range_constexpr(per):
                out = local_base + i
                v_o[out] = dualwave._mfma_acc(
                    v_pk[i], p_pk, v_o[out], self.mma_atom, self.mfma_acc_vec_type
                )
        return v_o


class WideStoreHelper(ParityStoreHelper):
    """Store this wave's output columns, one contiguous run per stage.

    `_final_o_global` derives the column from `dc` alone, so a run is stored by
    shifting `o_base` by the run's global start and handing the inner store a
    *slice* of the accumulator list. Nothing below this needs to know about
    shards.
    """

    # The global chunk index of the run being stored. Read by
    # `_final_o_global`; see there for why a *local* index is not enough.
    _wide_dc_base = 0

    def _final_o_global(self, o_base, dc, g):
        """Address from the local index, padded-head suppression from the global one.

        The two disagree on this path and the base class assumes they cannot.
        It derives both the address *and* the `hdim_vo` bound check from `dc`,
        which holds while `dc` is the absolute output chunk. Here a run hands
        down a slice, so `dc` restarts at 0 while the columns it addresses do
        not -- `o_base` already carries the run's offset.

        Left uncorrected the bound check sees a column far below the real one,
        so chunks past `hdim_vo` are *not* suppressed. They are not harmlessly
        dropped either: the descriptor spans the whole tensor rather than one
        row, so a store past the row pitch lands in the next row and silently
        corrupts it. That reads as a plain wrong answer, with no NaN and no
        fault -- head_dim 300 came out at 0.58 absolute error.
        """
        off = dualwave.DualwaveStoreHelper._final_o_global(self, o_base, dc, g)
        if const_expr(not self.PADDED_HEAD):
            return off
        col_base = (
            (self._wide_dc_base + dc) * self.traits.D_CHUNK
            + 2 * g * 8
            + self.lane_div_32 * fx.Index(8)
        )
        in_range = MaskedAxis(fx.Index(self.hdim_vo)).valid(fx.Index(col_base))
        return fx.Index(in_range.select(fx.Index(off), self.o_oob_off))

    def store_final_o_runs(self, v_o, q_row, runs, shard_slot, m_row=None, l_row=None):
        traits = self.traits
        # Runtime, because the shard comes from `wave_id`. Added to the base
        # once per run rather than folded into the constexpr run offsets.
        shard_chunks = shard_slot * traits.D_CHUNKS_PER_STAGE_SHARD
        o_base = self._final_o_base(q_row) + shard_chunks * traits.D_CHUNK
        for local0, global0, count in runs:
            base = o_base + global0 * traits.D_CHUNK
            tail = v_o[local0:]
            self._wide_dc_base = global0 + shard_chunks
            for i in range_constexpr(count):
                for g in range_constexpr(2):
                    self._store_final_o_128(tail, i, g, base)
        self._wide_dc_base = 0
        if const_expr(traits.RETURN_LSE):
            # Every shard of a Q tile ran the same softmax over the same rows,
            # so the LSE is replicated `VO_SHARDS` times. Shard 0 writes it;
            # letting all of them would be a benign but pointless race.
            self._store_lse_row_if_shard0(m_row, l_row, q_row, shard_slot)

    def _store_lse_row_if_shard0(self, m_row, l_row, q_row, shard_slot):
        if const_expr(self.traits.VO_SHARDS == 1):
            self._store_lse_row(m_row, l_row, q_row)
            return
        store = self._store_lse_row

        @flyc.jit
        def _lse_if_shard0():
            if shard_slot == fx.Index(0):
                store(m_row, l_row, q_row)

        _lse_if_shard0()


def make_wide_body(
    ctx,
    traits,
    *,
    q_loader,
    kv_gmem_to_lds,
    kv_lds_to_regs,
    gemm_helper,
    softmax_helper,
    output_store,
):
    """Return the traced body: one KV tile per iteration, D staged and sharded.

    A factory returning a `@flyc.jit` closure rather than a plain function the
    kernel calls. The loop below uses the `for ... init=[...]` / `yield`
    protocol, which only exists after the AST rewrite -- a module-level
    function called from the kernel is compiled by Python as written, and
    `range(..., init=...)` is then just a `TypeError`. Closing over the helpers
    keeps them out of the traced signature, where only values belong.
    """

    @flyc.jit
    def _wide_body():
        """One KV tile per iteration, D staged through LDS and sharded across waves."""
        # **Not scaled.** `qk_scale` lands on the f32 scores below, once the D
        # axis is complete; `ParityQLoader.scale_all` records what folding it
        # into Q costs and now refuses.
        q_all_bf16 = q_loader.load_all()

        runs = tuple(
            (
                st * traits.D_CHUNKS_PER_STAGE_SHARD,
                st * traits.D_CHUNKS_PER_STAGE,
                traits.D_CHUNKS_PER_STAGE_SHARD,
            )
            for st in range(traits.D_STAGES)
        )
        owned = traits.D_CHUNKS // traits.VO_SHARDS

        # `split_tile(0)` unconditionally: split-K used to be the only thing
        # that moved this workgroup's tile base, and a window moves it too --
        # `_skip_dead_leading_tiles` starts the walk at the window's left edge.
        # Spelling the non-split arm as a literal 0 does not make a window
        # *wrong* here, because this body masks every tile it visits and a dead
        # one contributes nothing; it just walks the dead ones anyway and
        # throws the whole saving away. Identical wherever the base is zero.
        loop_lb = ctx.split_tile(0)
        init_args = [softmax_helper.c_neg_floor, ctx.c_zero_f]
        for _ in range_constexpr(owned):
            init_args.append(ctx.c_zero_v16f32)
        loop_results = init_args

        # Prime the pipeline. From here every tile's first K stage is issued by
        # the *previous* tile's PV phase, so this is the only DMA in the kernel
        # whose latency is not covered by compute.
        kv_gmem_to_lds.load_k_tile(loop_lb, 0, stage=0)

        for j, loop_args in range(
            loop_lb, ctx.split_t_end, fx.Index(1), init=init_args
        ):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(owned)]

            # -- GEMM1. D is the reduction axis, so every stage feeds the same S
            #    and the softmax cannot start until the last one has landed.
            v_s = (ctx.c_zero_v16f32, ctx.c_zero_v16f32)
            for st in range_constexpr(traits.D_STAGES):
                cur = st % 2
                more = st + 1 < traits.D_STAGES
                if const_expr(more or st == 0):
                    # Gates every buffer written below: each was last read a
                    # stage (or a phase) ago, and this closes those readers.
                    _s_barrier()
                if const_expr(more):
                    kv_gmem_to_lds.load_k_tile(j, (st + 1) % 2, stage=st + 1)
                if const_expr(st == 0):
                    # Cross-phase: V's first stage rides the whole QK phase.
                    kv_gmem_to_lds.load_v_tile(j, 0, stage=0)
                # `vmcnt` retires in issue order, so leaving exactly the
                # newer groups outstanding retires stage `st` and nothing
                # else. This is the dual-wave body's `NUM_DMA_K + NUM_DMA_V`
                # idiom, at stage granularity rather than tile.
                _waitcnt_vm_n((ctx.NUM_DMA_K if more else 0) + ctx.NUM_DMA_V)
                _sched_barrier(0)
                _s_barrier()  # every wave's stage-st DMA has landed, not just mine
                v_k = kv_lds_to_regs.load_k(cur, stage=st)
                v_s = gemm_helper.qk_stage(v_k, q_all_bf16, v_s, st)

            # Every stage has landed, so S is complete and this is the one
            # point where `qk_scale` can be applied exactly once. Before the
            # masks, so the multiply never meets a `-inf`; see `scale_scores`.
            v_s = gemm_helper.scale_scores(v_s)

            if const_expr(traits.CAUSAL):
                v_s = softmax_helper.causal_mask_prologue_if_needed(
                    v_s, j, kv_end_tile=j + fx.Index(1)
                )
            else:
                v_s = softmax_helper.seq_pad_mask_if_needed(v_s, j)
                v_s = softmax_helper.v_s_vec_to_lists(v_s)
            m_tile_max = softmax_helper.reduce_max(v_s)
            v_o, m_row, l_row = softmax_helper.rescale_o_serial(
                v_o, m_row, l_row, m_tile_max
            )
            v_s = softmax_helper.sub_m(v_s, m_row)
            # `exp2` takes an explicit (start, length) because the dual-wave
            # schedule splits the 32 scores across two clusters. Here they are
            # simply adjacent.
            v_p = softmax_helper.exp2(v_s, 0, 16)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = softmax_helper.cast_p(v_p, j)

            # -- GEMM2. D is an output axis here: each (stage, shard) owns a
            #    disjoint run of O, so nothing is combined afterwards.
            for st in range_constexpr(traits.D_STAGES):
                cur = st % 2
                more = st + 1 < traits.D_STAGES
                if const_expr(more or st == 0):
                    _s_barrier()
                if const_expr(more):
                    kv_gmem_to_lds.load_v_tile(j, (st + 1) % 2, stage=st + 1)
                if const_expr(st == 0):
                    # Cross-phase, across the tile boundary: the next tile's
                    # first K stage rides this whole PV phase, so the KV loop
                    # never exposes a DMA after its prologue. Reading past the
                    # last tile is harmless -- the buffer descriptor bounds it
                    # and that staging is never read. The dual-wave body
                    # prefetches `j_idx + 1` the same way.
                    kv_gmem_to_lds.load_k_tile(j + fx.Index(1), 0, stage=0)
                _waitcnt_vm_n((ctx.NUM_DMA_V if more else 0) + ctx.NUM_DMA_K)
                _sched_barrier(0)
                _s_barrier()
                v_v = kv_lds_to_regs.load_v_shard(cur, ctx.vo_shard_id)
                v_o = gemm_helper.pv_shard(
                    v_p, v_v, v_o, st * traits.D_CHUNKS_PER_STAGE_SHARD
                )

            loop_results = yield [m_row, l_row] + v_o

        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(owned)]

        l_inv = softmax_helper.safe_l_inv(l_row)
        softmax_helper.scale_o(v_o, l_inv)
        _s_barrier()

        output_store.store_final_o_runs(
            v_o, ctx.q_row, runs, ctx.vo_shard_id, m_row, l_row
        )

    return _wide_body
