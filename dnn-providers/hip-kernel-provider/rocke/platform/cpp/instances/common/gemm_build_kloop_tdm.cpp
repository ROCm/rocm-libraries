// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gemm_build_kloop_tdm.cpp -- the universal GEMM's Tensor Data Mover K-loops.
 * Mirrors rocke/instances/common/_gemm_kloop_tdm.py (TdmKLoop).
 *
 * rocke_gemm_emit_compute_and_epilogue hands every tdm spec to
 * rocke_gemm_emit_kloop_tdm, which picks the loop shape:
 *   tdm_depth 1              unpipelined: issue, drain, compute
 *   tdm_depth 2              ping-pong over two LDS halves
 *   tdm_depth 2 + early      early refill (tdm_early_refill)
 *   tdm_depth 3 and 4        a modular LDS ring
 * Depths 1 and 2 keep their own loop shapes rather than being special cases of
 * the ring, so the ring cannot reshape the depth-2 form the tuned shapes ship.
 *
 * Every builder call allocates an SSA name, so the call sequence here matches
 * the Python engine's one for one; reordering any of it breaks byte identity.
 */

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_gemm_internal.h"

/* Drain every TDM fill, then rendezvous on LDS. TENSORcnt ticks once per
 * descriptor, so a full drain is 0 regardless of tile shape. Shared by the
 * K-loops whose non-TDM form waits and barriers instead. Mirrors Python
 * TdmKLoop._drain. */
static void rocke_gemm_tdm_drain(rocke_ir_builder_t* b)
{
    rocke_b_s_wait_tensorcnt(b, 0);
    rocke_b_sync_lds_only(b);
}

/* Workgroup barrier ids of the split barrier: signal / wait the whole WG. */
#define ROCKE_GEMM_WG_SIGNAL 0xFFFFFFFFu
#define ROCKE_GEMM_WG_WAIT 0xFFFF

/* Publish TDM fills: wait until at most `allowed` are outstanding, then signal
 * the split barrier. Shared by the TDM K-loops. Mirrors Python TdmKLoop._publish. */
static void rocke_gemm_tdm_publish(rocke_ir_builder_t* b, int allowed)
{
    rocke_b_s_wait_tensorcnt(b, allowed);
    rocke_b_s_barrier_signal(b, ROCKE_GEMM_WG_SIGNAL);
}

/* A K tile's split-barrier signal: drain this wave's fragment reads, then
 * publish with at most `*(int*)user` fills outstanding (the tile about to be
 * read has landed). Mirrors Python TdmKLoop._signal. */
static void rocke_gemm_split_signal(rocke_gemm_build_ctx_t* ctx, void* user)
{
    rocke_b_s_waitcnt(ctx->b, -1, 0, -1);
    rocke_gemm_tdm_publish(ctx->b, *(const int*)user);
}

/* ceil((K - k_lo) / block_k): the K tiles of this K range. The last one is
 * partial when K is not a multiple of block_k; the mover clips it to K. Shared
 * by the TDM K-loops that count tiles. Mirrors Python TdmKLoop._k_tile_count; one value
 * per statement, in the Python engine's order. */
static rocke_value_t* rocke_gemm_k_tile_count(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* k_span = rocke_b_sub(b, ctx->k_upper, ctx->k_lo);
    rocke_value_t* c_bk_m1 = rocke_b_const_i32(b, ctx->block_k - 1);
    return rocke_b_div(b, rocke_b_add(b, k_span, c_bk_m1), ctx->c_block_k);
}

/* ===================================================================== *
 *  tdm_depth 1: issue the tile, drain it, compute, barrier. Unlike ASYNCcnt,
 *  which counts instructions, TENSORcnt drains to 0 for any tile. Mirrors
 *  Python TdmKLoop.unpipelined.
 * ===================================================================== */
static void rocke_gemm_emit_kloop_tdm_unpipelined(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_iter_arg_t loop_args[ROCKE_GEMM_MAX_ACCS];
    rocke_for_t for_op;
    int i;

    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[i].name = ctx->acc_names[i];
        loop_args[i].init = ctx->acc_inits[i];
    }

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  ctx->k_upper,
                                  ctx->c_block_k,
                                  loop_args,
                                  ctx->num_accs,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];

        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, for_op.iv, 0, NULL);
        rocke_gemm_tdm_drain(b);
        rocke_gemm_emit_mfma_phase(ctx,
                                   ctx->A_smem,
                                   ctx->B_smem,
                                   for_op.iter_vars,
                                   for_op.num_iter_vars,
                                   0,
                                   NULL,
                                   new_accs);
        rocke_b_sync(b);
        rocke_b_scf_yield(b, new_accs, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    for(i = 0; i < for_op.op->num_results; ++i)
        ctx->for_results[i] = for_op.op->results[i];
    ctx->num_for_results = for_op.op->num_results;
}

/* ===================================================================== *
 *  tdm_depth 2: two LDS halves, the next tile's fill overlapping the current
 *  tile's WMMAs. Mirrors Python TdmKLoop.ping_pong.
 * ===================================================================== */
static void rocke_gemm_emit_kloop_tdm_ping_pong(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const bool split = ctx->spec->trait.split_barrier;
    static const int tile_done_allowed = 0;
    rocke_gemm_emit_hook_t signal_tile_done = {rocke_gemm_split_signal, (void*)&tile_done_allowed};
    rocke_gemm_emit_hook_t none = {NULL, NULL};
    rocke_value_t* K_minus_one_tile;
    rocke_value_t* c1_i32;
    rocke_iter_arg_t loop_args[1 + ROCKE_GEMM_MAX_ACCS];
    int num_loop_args;
    rocke_for_t for_op;
    int i;

    /* Prologue: load tile 0 into half 0 (slice base k_lo in split-K). */
    rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, ctx->k_lo, 0, NULL);

    /* Split barrier: every wait (each iteration's top, and the epilogue's)
     * pairs with the signal raised one phase earlier -- the prologue's for the
     * first iteration, then the one inside each iteration's WMMA phase. */
    if(split)
    {
        rocke_gemm_tdm_publish(b, 0);
    }

    K_minus_one_tile = rocke_b_sub(b, ctx->k_upper, ctx->c_block_k);
    c1_i32 = rocke_b_const_i32(b, 1);

    loop_args[0].name = "par";
    loop_args[0].init = ctx->c0;
    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[1 + i].name = ctx->acc_names[i];
        loop_args[1 + i].init = ctx->acc_inits[i];
    }
    num_loop_args = 1 + ctx->num_accs;

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  K_minus_one_tile,
                                  ctx->c_block_k,
                                  loop_args,
                                  num_loop_args,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* parity = for_op.iter_vars[0];
        rocke_value_t* const* acc_iter = &for_op.iter_vars[1];
        int num_acc_iter = for_op.num_iter_vars - 1;
        rocke_value_t* next_parity = rocke_b_sub(b, c1_i32, parity);
        rocke_value_t* k_next = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* yield_vals[1 + ROCKE_GEMM_MAX_ACCS];

        /* Only the current tile's descriptors are outstanding here -- the next
         * tile is issued after the drain -- so a full TENSORcnt drain still
         * leaves the copy overlapped with the previous WMMAs. */
        if(split)
            rocke_b_s_barrier_wait(b, ROCKE_GEMM_WG_WAIT);
        else
            rocke_gemm_tdm_drain(b);
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_next, 0, next_parity);
        if(split)
            rocke_gemm_emit_wmma_phase_ex(ctx,
                                          ctx->A_smem,
                                          ctx->B_smem,
                                          acc_iter,
                                          num_acc_iter,
                                          0,
                                          parity,
                                          new_accs,
                                          signal_tile_done,
                                          none);
        else
            rocke_gemm_emit_mfma_phase(
                ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc_iter, 0, parity, new_accs);

        yield_vals[0] = next_parity;
        for(i = 0; i < num_acc_iter; ++i)
            yield_vals[1 + i] = new_accs[i];
        rocke_b_scf_yield(b, yield_vals, 1 + num_acc_iter);
    }
    rocke_b_region_leave(b);

    /* The last tile: already loaded into the final parity. */
    {
        rocke_value_t* final_parity = for_op.op->results[0];
        rocke_value_t* const* tail_accs = &for_op.op->results[1];
        int num_tail = for_op.op->num_results - 1;
        rocke_value_t* epi_accs[ROCKE_GEMM_MAX_ACCS];

        if(split)
            rocke_b_s_barrier_wait(b, ROCKE_GEMM_WG_WAIT);
        else
            rocke_gemm_tdm_drain(b);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, tail_accs, num_tail, 0, final_parity, epi_accs);

        for(i = 0; i < num_tail; ++i)
            ctx->for_results[i] = epi_accs[i];
        ctx->num_for_results = num_tail;
    }
}

/* ===================================================================== *
 *  tdm_depth 2 with tdm_early_refill. Mirrors Python TdmKLoop.early_refill.
 *
 *  Waves 2p / 2p+1 own half p (alternating issuers). For K-tile i in half p,
 *  after the wave's split-barrier signal and the first row of the last
 *  k-atom's WMMAs, the pair owning half p waits on that barrier (every wave
 *  has finished reading half p) and refills it with tile i + 2; the wait is
 *  issued right before the TDM load so the descriptor arithmetic overlaps it.
 *  Every other wave consumes the same barrier phase at the top of tile i + 1.
 *  The loop runs two tiles per iteration with static parity over an even tile
 *  count; with an odd count the last tile lies past K and is zero-filled.
 * ===================================================================== */
typedef struct rocke_gemm_early_refill
{
    rocke_value_t* pair; /* tdm_wave_id / 2 */
    rocke_value_t* bk2; /* const_i32(2 * block_k) */
    int p; /* LDS half of the tile */
    rocke_value_t* k_cur; /* K origin of the tile */
    bool refill; /* refill the half with tile + 2 */
} rocke_gemm_early_refill_t;

static void rocke_gemm_er_wait(rocke_gemm_build_ctx_t* ctx, void* user)
{
    (void)user;
    rocke_b_s_barrier_wait(ctx->b, ROCKE_GEMM_WG_WAIT);
}

/* The pair owning the other half already consumed this phase when it
 * refilled that half. */
static void rocke_gemm_er_top_wait(rocke_gemm_build_ctx_t* ctx, rocke_value_t* pair, int p)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* other = rocke_b_const_i32(b, 1 - p);
    rocke_if_t gate = rocke_b_scf_if(b, rocke_b_cmp_ne(b, pair, other));
    rocke_b_region_enter(b, gate.then_region);
    rocke_gemm_er_wait(ctx, NULL);
    rocke_b_region_leave(b);
}

static void rocke_gemm_er_refill(rocke_gemm_build_ctx_t* ctx, void* user)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_early_refill_t* r = (const rocke_gemm_early_refill_t*)user;
    rocke_value_t* half = rocke_b_const_i32(b, r->p);
    rocke_if_t gate = rocke_b_scf_if(b, rocke_b_cmp_eq(b, r->pair, half));
    rocke_b_region_enter(b, gate.then_region);
    if(r->refill)
    {
        rocke_gemm_emit_hook_t wait = {rocke_gemm_er_wait, NULL};
        rocke_value_t* k_next = rocke_b_add(b, r->k_cur, r->bk2);
        rocke_gemm_emit_load_phase_ex(ctx, ctx->A_smem, ctx->B_smem, k_next, r->p, NULL, wait);
    }
    else
    {
        rocke_gemm_er_wait(ctx, NULL);
    }
    rocke_b_region_leave(b);
}

static void rocke_gemm_er_tile(rocke_gemm_build_ctx_t* ctx,
                               rocke_value_t* pair,
                               rocke_value_t* bk2,
                               int p,
                               rocke_value_t* k_cur,
                               rocke_value_t* const* acc,
                               int num_acc,
                               bool refill,
                               rocke_value_t** out_accs)
{
    static const int all_landed = 0;
    rocke_gemm_early_refill_t r;
    rocke_gemm_emit_hook_t signal = {rocke_gemm_split_signal, (void*)&all_landed};
    rocke_gemm_emit_hook_t refill_hook = {rocke_gemm_er_refill, &r};
    r.pair = pair;
    r.bk2 = bk2;
    r.p = p;
    r.k_cur = k_cur;
    r.refill = refill;
    rocke_gemm_er_top_wait(ctx, pair, p);
    rocke_gemm_emit_wmma_phase_ex(
        ctx, ctx->A_smem, ctx->B_smem, acc, num_acc, p, NULL, out_accs, signal, refill_hook);
}

static void rocke_gemm_emit_kloop_tdm_early_refill(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* bk2 = rocke_b_const_i32(b, 2 * ctx->block_k);
    rocke_value_t* pair = rocke_b_div(b, ctx->tdm_wave_id, c2);
    rocke_iter_arg_t loop_args[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* acc0[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* acc1[ROCKE_GEMM_MAX_ACCS];
    rocke_for_t for_op;
    int i;

    /* Prologue: tiles 0 and 1 into halves 0 and 1. Pair 1 consumes the
     * prologue phase here, as if it had refilled half 1 after tile -1. */
    rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, ctx->k_lo, 0, NULL);
    {
        rocke_value_t* k1 = rocke_b_add(b, ctx->k_lo, ctx->c_block_k);
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k1, 1, NULL);
    }
    rocke_gemm_tdm_publish(b, 0);
    {
        rocke_if_t gate = rocke_b_scf_if(b, rocke_b_cmp_eq(b, pair, c1));
        rocke_b_region_enter(b, gate.then_region);
        rocke_gemm_er_wait(ctx, NULL);
        rocke_b_region_leave(b);
    }

    /* The loop runs an even number of tiles so every parity is static (a
     * runtime-parity tail spills the steady state at the 256-VGPR budget).
     * The tile count is rounded up first, so a partial last tile (K not a
     * multiple of block_k) is loaded; the mover clips it to K. With an odd
     * count the extra tile lies past K: its descriptor's remaining-K extent
     * is zero, the mover fills it with zeros, and its WMMAs add nothing. */
    rocke_value_t* n_tiles = rocke_gemm_k_tile_count(ctx);
    rocke_value_t* n_even = rocke_b_add(b, n_tiles, rocke_b_mod(b, n_tiles, c2));
    rocke_value_t* n_pairs = rocke_b_div(b, rocke_b_sub(b, n_even, c2), c2);
    rocke_value_t* k_pairs_end = rocke_b_add(b, ctx->k_lo, rocke_b_mul(b, n_pairs, bk2));

    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[i].name = ctx->acc_names[i];
        loop_args[i].init = ctx->acc_inits[i];
    }
    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  k_pairs_end,
                                  bk2,
                                  loop_args,
                                  ctx->num_accs,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k1;
        rocke_gemm_er_tile(
            ctx, pair, bk2, 0, for_op.iv, for_op.iter_vars, for_op.num_iter_vars, true, acc0);
        k1 = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_gemm_er_tile(ctx, pair, bk2, 1, k1, acc0, for_op.num_iter_vars, true, acc1);
        rocke_b_scf_yield(b, acc1, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    /* Tail: tile T-2 (half 0) has nothing left to refill; tile T-1 (half 1)
     * raises no signal. */
    rocke_gemm_er_tile(
        ctx, pair, bk2, 0, k_pairs_end, for_op.op->results, for_op.op->num_results, false, acc0);
    rocke_gemm_er_top_wait(ctx, pair, 1);
    rocke_gemm_emit_wmma_phase(
        ctx, ctx->A_smem, ctx->B_smem, acc0, for_op.op->num_results, 1, NULL, acc1);

    for(i = 0; i < for_op.op->num_results; ++i)
        ctx->for_results[i] = acc1[i];
    ctx->num_for_results = for_op.op->num_results;
}

/* ===================================================================== *
 *  tdm_depth D >= 3: a D-buffer LDS ring with a partial wait. Mirrors Python
 *  TdmKLoop.ring.
 * ===================================================================== */
static void rocke_gemm_emit_kloop_tdm_ring(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int D = ctx->spec->trait.tdm_depth;
    const int per_tile = (ctx->tdm_waves > 1) ? 1 : 2;
    const int allowed = (D - 2) * per_tile;
    const int ahead_k = (D - 1) * ctx->block_k;
    const bool split = ctx->spec->trait.split_barrier;
    rocke_gemm_emit_hook_t signal_next_tile_ready = {rocke_gemm_split_signal, (void*)&allowed};
    rocke_gemm_emit_hook_t none = {NULL, NULL};
    rocke_value_t* n_tiles = rocke_gemm_k_tile_count(ctx);
    rocke_value_t* c_one_k = rocke_b_const_i32(b, 1);
    rocke_value_t* last_tile = rocke_b_sub(b, n_tiles, c_one_k);
    /* Never below k_lo: with no K tile at all (K = 0) every prologue fill starts
     * at k_lo with an empty K extent instead of one tile before A and B. */
    rocke_value_t* last_k_tile
        = rocke_b_add(b, ctx->k_lo, rocke_b_mul(b, last_tile, ctx->c_block_k));
    rocke_value_t* last_origin = rocke_b_smax(b, last_k_tile, ctx->k_lo);
    rocke_value_t* c_ring;
    rocke_value_t* c_ahead;
    rocke_value_t* c_one;
    rocke_iter_arg_t loop_args[1 + ROCKE_GEMM_MAX_ACCS];
    rocke_for_t for_op;
    int i;

    /* Prologue: fill ring slots 0 .. D-2. Emitted BEFORE the three ring
     * constants below, matching Python's statement order: every builder call
     * allocates an SSA name, so hoisting the constants would shift every
     * later number and break byte-identity. */
    for(int j = 0; j < D - 1; ++j)
    {
        rocke_value_t* k_off = rocke_b_smin(
            b, rocke_b_add(b, ctx->k_lo, rocke_b_const_i32(b, j * ctx->block_k)), last_origin);
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_off, j, NULL);
    }

    /* Split barrier: the joint barrier at each iteration's top becomes a wait
     * there, paired with a signal raised once the previous iteration's last
     * fragment reads have drained and its tile + 1 has landed -- the same
     * `allowed` count, since the outstanding fills are the same set at both
     * points. The prologue raises the first signal and a wait after the loop
     * consumes the last one. */
    if(split)
    {
        rocke_gemm_tdm_publish(b, allowed);
    }

    c_ring = rocke_b_const_i32(b, D);
    c_ahead = rocke_b_const_i32(b, D - 1);
    c_one = rocke_b_const_i32(b, 1);
    loop_args[0].name = "ring";
    loop_args[0].init = ctx->c0;
    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[1 + i].name = ctx->acc_names[i];
        loop_args[1 + i].init = ctx->acc_inits[i];
    }

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  ctx->k_upper,
                                  ctx->c_block_k,
                                  loop_args,
                                  1 + ctx->num_accs,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* ring = for_op.iter_vars[0];
        rocke_value_t* const* acc_iter = &for_op.iter_vars[1];
        int num_acc_iter = for_op.num_iter_vars - 1;
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* yield_vals[1 + ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* k_ahead;
        rocke_value_t* slot_ahead;

        if(split)
        {
            rocke_b_s_barrier_wait(b, ROCKE_GEMM_WG_WAIT);
        }
        else
        {
            rocke_b_s_wait_tensorcnt(b, allowed);
            rocke_b_sync_lds_only(b);
        }
        k_ahead = rocke_b_smin(
            b, rocke_b_add(b, for_op.iv, rocke_b_const_i32(b, ahead_k)), last_origin);
        slot_ahead = rocke_b_mod(b, rocke_b_add(b, ring, c_ahead), c_ring);
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_ahead, 0, slot_ahead);
        if(split)
            rocke_gemm_emit_wmma_phase_ex(ctx,
                                          ctx->A_smem,
                                          ctx->B_smem,
                                          acc_iter,
                                          num_acc_iter,
                                          0,
                                          ring,
                                          new_accs,
                                          signal_next_tile_ready,
                                          none);
        else
            rocke_gemm_emit_mfma_phase(
                ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc_iter, 0, ring, new_accs);

        yield_vals[0] = rocke_b_mod(b, rocke_b_add(b, ring, c_one), c_ring);
        for(i = 0; i < num_acc_iter; ++i)
            yield_vals[1 + i] = new_accs[i];
        rocke_b_scf_yield(b, yield_vals, 1 + num_acc_iter);
    }
    rocke_b_region_leave(b);

    if(split)
        rocke_b_s_barrier_wait(b, ROCKE_GEMM_WG_WAIT);
    /* Drain the in-flight fills so downstream code (cshuffle epilogue, persistent
     * prologue) can safely reuse LDS. */
    rocke_b_s_wait_tensorcnt(b, 0);

    for(i = 0; i < for_op.op->num_results - 1; ++i)
        ctx->for_results[i] = for_op.op->results[1 + i];
    ctx->num_for_results = for_op.op->num_results - 1;
}

/* ===================================================================== *
 *  The K-loop of a tdm spec, by tdm_depth. Mirrors Python TdmKLoop.emit.
 * ===================================================================== */
void rocke_gemm_emit_kloop_tdm(rocke_gemm_build_ctx_t* ctx)
{
    const int depth = ctx->spec->trait.tdm_depth;
    if(depth >= 3)
        rocke_gemm_emit_kloop_tdm_ring(ctx);
    else if(depth == 2 && ctx->spec->trait.tdm_early_refill)
        rocke_gemm_emit_kloop_tdm_early_refill(ctx);
    else if(depth == 2)
        rocke_gemm_emit_kloop_tdm_ping_pong(ctx);
    else
        rocke_gemm_emit_kloop_tdm_unpipelined(ctx);
}
