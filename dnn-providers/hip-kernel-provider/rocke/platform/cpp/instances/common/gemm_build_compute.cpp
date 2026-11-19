// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_build_compute.c -- one chunk of the C99 port of
 * build_universal_gemm (rocke/instances/common/gemm_universal.py).
 *
 * SCOPE (this TU):
 *   - emit_mfma_phase            (rocke_gemm_emit_mfma_phase)
 *   - emit_compute_and_epilogue  (rocke_gemm_emit_compute_and_epilogue)
 *   - _emit_kloop_db             (rocke_gemm_emit_kloop_db)
 *   - _emit_kloop_simple         (rocke_gemm_emit_kloop_simple)
 *   - _emit_kloop_prefetch       (rocke_gemm_emit_kloop_prefetch)
 *
 * The TDM K-loops are in gemm_build_kloop_tdm.cpp (Python _gemm_kloop_tdm.py).
 *
 * These map to the Python closures of the same names (gemm_universal.py
 * ~1466-1782). Every other closure/helper is a peer TU reached via
 * rocke/instance_gemm_internal.h. The emission order + builder-call sequence is a
 * faithful 1:1 port so the lowered op stream is byte-identical to Python.
 */

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.tiling.mma.h"
#include "rocke/instance_gemm_internal.h"

/* Bind the current LDS slot and lane geometry for the staged TileMma readers. */
struct rocke_gemm_mfma_operand
{
    rocke_value_t* src;
    rocke_value_t* warp_off;
    rocke_value_t* lane_off;
    rocke_value_t* slot_off;
    int static_slot_off;
    int atom_rows;
    int frag_len;
};

struct rocke_gemm_mfma_tile
{
    rocke_gemm_build_ctx_t* ctx;
    rocke_value_t* k_base;
    rocke_gemm_mfma_operand a;
    rocke_gemm_mfma_operand b;
};

static rocke_value_t* rocke_gemm_read_mfma_operand(rocke_ir_builder_t* b,
                                                   const rocke_gemm_mfma_tile& tile,
                                                   const rocke_gemm_mfma_operand& operand,
                                                   int ki,
                                                   int atom)
{
    auto* ctx = tile.ctx;
    auto* col = rocke_b_add(b, tile.k_base, rocke_b_const_i32(b, ki * ctx->spec->tile.warp_tile_k));
    auto* row = rocke_b_add(
        b,
        operand.warp_off,
        rocke_b_add(b,
                    rocke_b_const_i32(b, atom * operand.atom_rows + operand.static_slot_off),
                    operand.lane_off));
    if(operand.slot_off != nullptr)
        row = rocke_b_add(b, row, operand.slot_off);
    return rocke_gemm_emit_smem_load(b,
                                     operand.src,
                                     row,
                                     rocke_gemm_swz_col(ctx, col, row),
                                     operand.frag_len,
                                     ctx->storage_dtype);
}

static rocke_value_t* rocke_gemm_read_mfma_a(rocke_ir_builder_t* b, int ki, int mi, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_mfma_tile*>(user);
    return rocke_gemm_read_mfma_operand(b, tile, tile.a, ki, mi);
}

static rocke_value_t* rocke_gemm_read_mfma_b(rocke_ir_builder_t* b, int ki, int ni, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_mfma_tile*>(user);
    return rocke_gemm_read_mfma_operand(b, tile, tile.b, ki, ni);
}

static rocke_value_t* rocke_gemm_issue_mfma(
    rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb, rocke_value_t* acc, void* user)
{
    const auto& tile = *static_cast<rocke_gemm_mfma_tile*>(user);
    return rocke_gemm_emit_mma(b, tile.ctx->op, a, bb, acc);
}

static void rocke_gemm_start_mfma_row(rocke_ir_builder_t* b, int, int, void*)
{
    rocke_b_s_setprio(b, 1);
}

static void rocke_gemm_finish_mfma_row(rocke_ir_builder_t* b, int, int, void*)
{
    rocke_b_s_setprio(b, 0);
    rocke_b_sched_barrier(b, 0);
}

/* ===================================================================== *
 *  emit_mfma_phase
 *
 *  One K-tile worth of MMAs across all per-warp atom positions and every K
 *  atom step inside this K-tile. On WMMA (RDNA) this delegates to the fully
 *  contract-driven _emit_wmma_phase. On MFMA (CDNA) it keeps the byte-identical
 *  arch-formula fragment load; the matmul + result fragment length are
 *  contract-driven (rocke_gemm_emit_mma / op->dst.frag_len).
 *
 *  lds_parity is split (per the internal-header convention) into a compile-time
 *  int (parity_imm) OR a runtime Value (parity_v != NULL == Python isinstance
 *  Value). Only the prefetch/db paths pass a non-zero parity.
 * ===================================================================== */
void rocke_gemm_emit_mfma_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                int parity_imm,
                                rocke_value_t* parity_v,
                                rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;

    int mfmas_m = ctx->mfmas_m;
    int mfmas_n = ctx->mfmas_n;
    int k_atoms = ctx->k_atoms;

    /* parity carry: a runtime Value form (a_par_row_v / b_par_row_v) when the
     * parity is itself a Value; a compile-time additive offset otherwise. */
    rocke_value_t* a_par_row_v = NULL;
    rocke_value_t* b_par_row_v = NULL;
    int a_row_parity_off = 0;
    int b_row_parity_off = 0;

    rocke_value_t* m_in_atom;
    rocke_value_t* k_blk;
    rocke_value_t* n_in_atom;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    rocke_value_t* k_blk_kbase;

    if(ctx->is_wmma)
    {
        rocke_gemm_emit_wmma_phase(
            ctx, A_src, B_src, iter_vars, num_iter_vars, parity_imm, parity_v, out_accs);
        return;
    }

    if(ctx->prefetch || ctx->db)
    {
        if(parity_v != NULL)
        {
            a_par_row_v = rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->block_m));
            b_par_row_v = rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->block_n));
            a_row_parity_off = 0; /* carried as a Value addition below */
            b_row_parity_off = 0;
        }
        else
        {
            a_par_row_v = NULL;
            b_par_row_v = NULL;
            a_row_parity_off = parity_imm * ctx->block_m;
            b_row_parity_off = parity_imm * ctx->block_n;
        }
    }
    else
    {
        a_par_row_v = NULL;
        b_par_row_v = NULL;
        a_row_parity_off = 0;
        b_row_parity_off = 0;
    }

    /* Lane mapping into LDS. M-in-atom = lane % warp_tile_m, K-sub-block =
     * lane / warp_tile_m, N-in-atom = lane % warp_tile_n. */
    m_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    k_blk = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    n_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_n));

    warp_m_off = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    warp_n_off = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));

    /* Lane-local K base inside the current K-tile, hoisted once. */
    k_blk_kbase = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, ctx->a_per_lane));

    rocke_gemm_mfma_tile tile{ctx,
                              k_blk_kbase,
                              {A_src,
                               warp_m_off,
                               m_in_atom,
                               a_par_row_v,
                               a_row_parity_off,
                               t->warp_tile_m,
                               ctx->a_per_lane},
                              {B_src,
                               warp_n_off,
                               n_in_atom,
                               b_par_row_v,
                               b_row_parity_off,
                               t->warp_tile_n,
                               ctx->b_per_lane}};
    const bool prioritize = ctx->swz && ctx->sched_hints;
    const rocke_tile_mma_readers_t readers{
        .read_a = rocke_gemm_read_mfma_a,
        .read_b = rocke_gemm_read_mfma_b,
        .mma = rocke_gemm_issue_mfma,
        .after_reads = nullptr,
        .before_row = prioritize ? rocke_gemm_start_mfma_row : nullptr,
        .after_row = prioritize ? rocke_gemm_finish_mfma_row : nullptr,
        .user = &tile};
    rocke_tile_mma_accumulate_staged(b,
                                     {mfmas_m, mfmas_n, k_atoms},
                                     iter_vars,
                                     static_cast<size_t>(num_iter_vars),
                                     &readers,
                                     out_accs);

    /* Quantitative two-stage HotLoop schedule, issued ONCE per K-tile.
     * Suppressed when _sched_hints is off (gfx950 default). */
    if(ctx->sched_hints && ctx->spec->trait.pipeline
       && (strcmp(ctx->spec->trait.pipeline, "compv3") == 0
           || strcmp(ctx->spec->trait.pipeline, "compv4") == 0))
    {
        rocke_gemm_emit_hotloop_schedule(b, ctx->spec, ctx->load_vec);
    }
}

/* ===================================================================== *
 *  emit_compute_and_epilogue: prefetch/db/simple K-loop selector + epilogue.
 * ===================================================================== */
void rocke_gemm_emit_compute_and_epilogue(rocke_gemm_build_ctx_t* ctx)
{
    if(ctx->spec->trait.tdm)
        rocke_gemm_emit_kloop_tdm(ctx);
    else if(ctx->prefetch)
        rocke_gemm_emit_kloop_prefetch(ctx);
    else if(ctx->db)
        rocke_gemm_emit_kloop_db(ctx);
    else
        rocke_gemm_emit_kloop_simple(ctx);
    rocke_gemm_emit_epilogue(ctx);
}

/* GemmKLoop.ping_pong: publish the current slot and retire the previous
 * readers before allowing the next issue to overwrite their slot. */
static void rocke_gemm_publish_and_retire(rocke_ir_builder_t* b, bool direct_to_lds)
{
    if(direct_to_lds)
    {
        rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
        rocke_b_s_barrier_bare(b);
    }
    else
        rocke_b_sync(b);
}

static void rocke_gemm_emit_kloop_ping_pong(rocke_gemm_build_ctx_t* ctx, bool direct_to_lds)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c1_i32;
    rocke_value_t* K_minus_one_tile;
    rocke_iter_arg_t loop_args[1 + ROCKE_GEMM_MAX_ACCS];
    int num_loop_args;
    rocke_for_t for_op;
    int i;

    /* Preserve the original SSA construction order for both policies. */
    if(direct_to_lds)
    {
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, ctx->k_lo, 0, NULL);
        K_minus_one_tile = rocke_b_sub(b, ctx->k_upper, ctx->c_block_k);
        c1_i32 = rocke_b_const_i32(b, 1);
    }
    else
    {
        c1_i32 = rocke_b_const_i32(b, 1);
        K_minus_one_tile = rocke_b_sub(b, ctx->k_upper, ctx->c_block_k);
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, ctx->k_lo, 0, NULL);
    }

    /* loop_args = [("par", c0)] + list(accs) */
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

        rocke_gemm_publish_and_retire(b, direct_to_lds);
        /* Issue the next K tile into the retired slot, then consume the current slot. */
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_next, 0, next_parity);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc_iter, 0, parity, new_accs);

        yield_vals[0] = next_parity;
        for(i = 0; i < num_acc_iter; ++i)
            yield_vals[1 + i] = new_accs[i];
        rocke_b_scf_yield(b, yield_vals, 1 + num_acc_iter);
    }
    rocke_b_region_leave(b);

    /* Epilogue: drain the last in-loop load, barrier, MFMA that tile. */
    {
        rocke_value_t* final_parity = for_op.op->results[0];
        rocke_value_t* const* tail_accs = &for_op.op->results[1];
        int num_tail = for_op.op->num_results - 1;
        rocke_value_t* epi_accs[ROCKE_GEMM_MAX_ACCS];

        rocke_gemm_publish_and_retire(b, direct_to_lds);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, tail_accs, num_tail, 0, final_parity, epi_accs);

        for(i = 0; i < num_tail; ++i)
            ctx->for_results[i] = epi_accs[i];
        ctx->num_for_results = num_tail;
    }
}

void rocke_gemm_emit_kloop_db(rocke_gemm_build_ctx_t* ctx)
{
    rocke_gemm_emit_kloop_ping_pong(ctx, /*direct_to_lds=*/false);
}

/* ===================================================================== *
 *  _emit_kloop_simple: single-buffer load-then-compute K-loop.
 * ===================================================================== */
void rocke_gemm_emit_kloop_simple(rocke_gemm_build_ctx_t* ctx)
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

        /* Single-buffer load-then-compute pipeline. parity is compile-time 0
         * (Python passes no lds_parity -> default 0). */
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, for_op.iv, 0, NULL);
        rocke_b_sync(b);

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
 *  _emit_kloop_prefetch: DTLA ping-pong software-pipelined K-loop. Falls
 *  back to _simple when loads_per_tile > 63 (vmcnt is 6 bits on gfx950).
 * ===================================================================== */
void rocke_gemm_emit_kloop_prefetch(rocke_gemm_build_ctx_t* ctx)
{
    /* Preserve the existing cutoff for the non-prefetched fallback. */
    if(ctx->dtl_a_passes + ctx->dtl_b_passes > 63)
        rocke_gemm_emit_kloop_simple(ctx);
    else
        rocke_gemm_emit_kloop_ping_pong(ctx, /*direct_to_lds=*/true);
}
