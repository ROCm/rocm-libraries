// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_kloop_drivers.c -- C99 port of the three
 * K-loop drivers of build_implicit_gemm_conv
 * (rocke/instances/common/conv_implicit_gemm.py, lines 1276-1347).
 *
 * Each driver sequences the per-K-tile load + compute phases and writes the
 * final-tile accumulators into ctx->final_accs / ctx->num_final_accs, which the
 * epilogue phase then reads. Exactly one driver is called per build, chosen by
 * the public driver as Python does:
 *   spec.unroll_k          -> rocke_conv_emit_kloop_unroll
 *   else not async_dma     -> rocke_conv_emit_kloop_simple
 *   else (async_dma)       -> rocke_conv_emit_kloop_async
 *
 * The builder-call sequence here is byte-identical to the Python source span.
 * Phase functions (emit_load_phase / emit_mfma_phase) are peers reached through
 * the internal header; this TU touches only ctx + the builder it carries.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h" /* rocke_arena_strdup */
#include "rocke/helper_helpers.asm.h" /* rocke_lds_barrier_keep_vmem */
#include "rocke/helper_rocke.helpers.pipeline.h"
#include "rocke/instance_conv_implicit_gemm_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ----- shared small helper: copy a working acc array into ctx->final_accs ----
 * Python sets `final_accs = current_accs` (or `for_op.results`); in C the
 * drivers write the ctx slot the epilogue reads. */
/* Every driver now walks a runtime [k_lo, k_hi) range with an scf.for, so the
 * old build-time tile-offset helper (const_i32(it*block_k), plus the slice base
 * for wgrad) has no callers left: the loop induction variable *is* the offset.
 */

static void
    rocke_conv_set_final_accs(rocke_conv_build_ctx_t* ctx, rocke_value_t* const* accs, int num_accs)
{
    int i;
    ctx->num_final_accs = num_accs;
    for(i = 0; i < num_accs; ++i)
        ctx->final_accs[i] = accs[i];
}

/* Mirrors the Python ping-pong's odd-tail guard: with ctx->kloop_k_zero_fill
 * set, a prefetch offset at or past k_hi is redirected to it (a tile that
 * reads as zero), so a split-K slice with an odd tile count cannot pull in the
 * next slice's first tile. Without it the offset passes through unchanged and
 * nothing is emitted. */
static rocke_value_t* rocke_conv_kloop_guard_prefetch(rocke_conv_build_ctx_t* ctx,
                                                      rocke_value_t* k,
                                                      rocke_value_t* k_hi)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* in_slice;
    if(ctx->kloop_k_zero_fill == NULL)
        return k;
    in_slice = rocke_b_cmp_lt(b, k, k_hi);
    return rocke_b_select(b, in_slice, k, ctx->kloop_k_zero_fill);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_unroll   (Python lines 1276-1310)
 *
 * Double-buffered Python-unrolled K-loop software pipeline (ping-pong
 * A_smem/A_smem2). Stage tile it+1 into the alternate LDS buffer while the MFMA
 * for tile it reads the current buffer. One barrier per iteration publishes the
 * prefetched tile and orders the current tile's ds_reads ahead of the it+2
 * prefetch that reuses the same buffer two iterations later.
 * ===================================================================== */
void rocke_conv_emit_kloop_unroll(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int block_k = ctx->block_k;
    int num_accs = ctx->num_accs;
    int i;

    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_for_t for_op;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_a[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_b[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* c_2block_k;
    rocke_value_t* k_lo;
    rocke_value_t* k_hi;

    k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    c_2block_k = rocke_b_const_i32(b, 2 * block_k);

    /* Prologue: stage tile 0 into buf0 and publish it. */
    rocke_conv_emit_load_phase(ctx, k_lo, ctx->A_smem, ctx->B_smem);
    rocke_b_sync(b);

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }

    /* AOT: the trip count is runtime, so bufs[it % 2] cannot be evaluated --
     * an LDS allocation is a build-time value. The body is unrolled twice and
     * steps by 2*block_k instead, which binds each phase to a build-time
     * buffer while still alternating them. An odd tile count needs no guard:
     * the trailing phase addresses k >= K, whose coords fall outside the
     * descriptor's padded bounds, so the buffer resource returns zero. */
    for_op = rocke_b_scf_for_iter(b,
                                  k_lo,
                                  k_hi,
                                  c_2block_k,
                                  iter_args,
                                  num_accs,
                                  "k_unroll",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k_odd = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_value_t* k_nxt_pair = rocke_b_add(b, for_op.iv, c_2block_k);
        rocke_value_t* k_odd_load = rocke_conv_kloop_guard_prefetch(ctx, k_odd, k_hi);

        /* Phase A: prefetch tile k+1 into buf1, MFMA tile k out of buf0. */
        rocke_conv_emit_load_phase(ctx, k_odd_load, ctx->A_smem2, ctx->B_smem2);
        ctx->k_off_capture = for_op.iv;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, accs_a);
        /* Publishes buf1 and drains buf0's ds_reads. */
        rocke_b_sync(b);

        /* Phase B: the buffers swap roles. */
        rocke_conv_emit_load_phase(ctx, k_nxt_pair, ctx->A_smem, ctx->B_smem);
        ctx->k_off_capture = k_odd;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem2, ctx->B_smem2, accs_a, for_op.num_iter_vars, accs_b);
        rocke_b_sync(b);

        rocke_b_scf_yield(b, accs_b, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_async_global_lds   (wgrad ``elif _wmma_async:``)
 *
 * gfx1250 async ping-pong with one barrier per tile. Each phase waits for its
 * own tile's global_load_async_to_lds copies (asynccnt 0), the LDS barrier
 * publishes that tile and retires the ds_reads of the other buffer, and only
 * then is the next tile issued into the other buffer, streaming while this
 * tile's WMMAs run. Unrolled twice so each phase binds a build-time buffer.
 * ===================================================================== */
void rocke_conv_emit_kloop_async_global_lds(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int num_accs = ctx->num_accs;
    int i;

    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_a[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_b[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    rocke_value_t* k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    rocke_value_t* c_2block_k = rocke_b_const_i32(b, 2 * ctx->block_k);
    rocke_for_t for_op;

    rocke_conv_emit_load_phase(ctx, k_lo, ctx->A_smem, ctx->B_smem);

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }
    for_op = rocke_b_scf_for_iter(b,
                                  k_lo,
                                  k_hi,
                                  c_2block_k,
                                  iter_args,
                                  num_accs,
                                  "k_async",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k_odd = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_value_t* k_nxt_pair = rocke_b_add(b, for_op.iv, c_2block_k);
        rocke_value_t* k_load;

        /* Phase A: compute tile k out of buf0 while tile k+1 streams into buf1. */
        rocke_b_s_wait_asynccnt(b, 0);
        rocke_b_sync_lds_only(b);
        k_load = rocke_conv_kloop_guard_prefetch(ctx, k_odd, k_hi);
        rocke_conv_emit_load_phase(ctx, k_load, ctx->A_smem2, ctx->B_smem2);
        ctx->k_off_capture = for_op.iv;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, accs_a);

        /* Phase B: the buffers swap roles. */
        rocke_b_s_wait_asynccnt(b, 0);
        rocke_b_sync_lds_only(b);
        k_load = rocke_conv_kloop_guard_prefetch(ctx, k_nxt_pair, k_hi);
        rocke_conv_emit_load_phase(ctx, k_load, ctx->A_smem, ctx->B_smem);
        ctx->k_off_capture = k_odd;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem2, ctx->B_smem2, accs_a, for_op.num_iter_vars, accs_b);

        rocke_b_scf_yield(b, accs_b, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    /* The last phase left one tile in flight; drain it and the zero-fill
     * stores before the epilogue reuses LDS. */
    rocke_b_s_wait_asynccnt(b, 0);
    rocke_b_sync(b);
    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_simple   (Python lines 1311-1319)
 *
 * Single scf.for_iter load + sync + mfma + sync. The not-async, not-unroll
 * branch.
 * ===================================================================== */
void rocke_conv_emit_kloop_simple(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int num_accs = ctx->num_accs;
    int i;

    /* for_op = b.scf_for_iter(c0, c_K_gemm, c_block_k, accs, iv_name="k0") */
    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_for_t for_op;
    rocke_value_t* k0;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }

    /* Start at the slice base, not at a bare zero: wgrad hands the drivers a
     * runtime k_lo and only the forward conv leaves it NULL. Reading ctx->c0
     * here worked while wgrad aliased the two, and silently restarted every
     * split-K slice from 0 once they came apart. The forward split-K binary
     * also ends at its slice: kloop_k_hi (wgrad sets it to c_K_gemm). */
    for_op = rocke_b_scf_for_iter(b,
                                  (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0,
                                  (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm,
                                  ctx->c_block_k,
                                  iter_args,
                                  num_accs,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    /* with for_op as (k0, iter_vars): */
    k0 = for_op.iv;
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        /* emit_load_phase(k0, A_smem, B_smem) */
        rocke_conv_emit_load_phase(ctx, k0, ctx->A_smem, ctx->B_smem);
        rocke_b_sync(b);
        /* new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars) */
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, new_accs);
        rocke_b_sync(b);
        /* b.scf_yield(*new_accs) */
        rocke_b_scf_yield(b, new_accs, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    /* final_accs = for_op.results */
    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_uniform_decode   (Python ``elif spec.uniform_k_decode:``)
 *
 * Uniform per-tile reduction decode (ImplicitGemmConvSpec.uniform_k_decode).
 * Loop 1 runs when cpg is a multiple of tile_k and carries the tile's
 * (c0, x, y) as scalar iteration arguments; loop 2 is the plain loop and runs
 * otherwise. Exactly one of the two has a non-empty range, chosen per problem
 * from kernargs. Every builder call is its own statement in Python's
 * left-to-right order.
 * ===================================================================== */

/* The tile's uniform terms (Python uk_cap[0] = _uk_tile(ct, xd, yd)) plus the
 * build ctx, threaded to the descriptors as the loaders' `user`. */
typedef struct rocke_conv_ukd_env
{
    rocke_conv_build_ctx_t* ctx;
    rocke_value_t* h_lo;
    rocke_value_t* h_hi;
    rocke_value_t* w_lo;
    rocke_value_t* w_hi;
    rocke_value_t* y_ok;
    rocke_value_t* a_off;
    rocke_value_t* b_off;
    /* Prefetch mask (Python _uk_masked): NULL when the loads are unmasked. */
    rocke_value_t* tile_ok;
} rocke_conv_ukd_env_t;

/* Python _uk_tile(ct, xd, yd). */
static void
    ukd_tile(rocke_conv_ukd_env_t* env, rocke_value_t* ct, rocke_value_t* xd, rocke_value_t* yd)
{
    rocke_conv_build_ctx_t* ctx = env->ctx;
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* hd = rocke_b_mul(b, yd, ctx->p_dH);
    rocke_value_t* wd = rocke_b_mul(b, xd, ctx->p_dW);
    rocke_value_t* a_y = rocke_b_mul(b, yd, ctx->split_ystep);
    rocke_value_t* a_x = rocke_b_mul(b, xd, ctx->split_xstep);
    rocke_value_t* a_yx = rocke_b_add(b, a_y, a_x);
    env->a_off = rocke_b_add(b, a_yx, ct);
    rocke_value_t* b_y = rocke_b_mul(b, yd, ctx->p_B_stride_y);
    rocke_value_t* b_x = rocke_b_mul(b, xd, ctx->p_B_stride_x);
    rocke_value_t* b_yx = rocke_b_add(b, b_y, b_x);
    env->b_off = rocke_b_add(b, b_yx, ct);
    env->h_lo = rocke_b_sub(b, ctx->c0, hd);
    env->h_hi = rocke_b_sub(b, ctx->p_Hi, hd);
    env->w_lo = rocke_b_sub(b, ctx->c0, wd);
    env->w_hi = rocke_b_sub(b, ctx->p_Wi, wd);
    env->y_ok = rocke_b_cmp_lt(b, yd, ctx->p_Y);
}

/* Python _a_uk(b_, row, col) [wrapped by _uk_masked when env->tile_ok]. */
static rocke_value_t* ukd_a_descriptor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* user)
{
    rocke_conv_ukd_env_t* env = (rocke_conv_ukd_env_t*)user;
    rocke_conv_build_ctx_t* ctx = env->ctx;
    rocke_value_t *q_wo, *wo, *n, *ho;
    rocke_value_t* m_val = rocke_b_add(b, ctx->block_m_off_v, row);
    rocke_conv_magic_divmod(
        b, m_val, ctx->p_magic_m_Wo_mult, ctx->p_magic_m_Wo_shift, ctx->p_Wo, false, &q_wo, &wo);
    rocke_conv_magic_divmod(
        b, q_wo, ctx->p_magic_m_Ho_mult, ctx->p_magic_m_Ho_shift, ctx->p_Ho, false, &n, &ho);
    rocke_value_t* ho_s = rocke_b_mul(b, ho, ctx->p_sH);
    rocke_value_t* hi0 = rocke_b_add(b, ho_s, ctx->split_neg_pH);
    rocke_value_t* wo_s = rocke_b_mul(b, wo, ctx->p_sW);
    rocke_value_t* wi0 = rocke_b_add(b, wo_s, ctx->split_neg_pW);
    rocke_value_t* n_off = rocke_b_mul(b, n, ctx->p_A_stride_n);
    rocke_value_t* hi_off = rocke_b_mul(b, hi0, ctx->p_A_stride_hi);
    rocke_value_t* base = rocke_b_add(b, n_off, hi_off);
    rocke_value_t* wi_off = rocke_b_mul(b, wi0, ctx->p_A_stride_wi);
    base = rocke_b_add(b, base, wi_off);
    if(ctx->split_group_base != NULL)
        base = rocke_b_add(b, base, ctx->split_group_base);
    rocke_value_t* h_ge = rocke_b_cmp_ge(b, hi0, env->h_lo);
    rocke_value_t* h_lt = rocke_b_cmp_lt(b, hi0, env->h_hi);
    rocke_value_t* h_ok = rocke_b_land(b, h_ge, h_lt);
    rocke_value_t* w_ge = rocke_b_cmp_ge(b, wi0, env->w_lo);
    rocke_value_t* w_lt = rocke_b_cmp_lt(b, wi0, env->w_hi);
    rocke_value_t* w_ok = rocke_b_land(b, w_ge, w_lt);
    rocke_value_t* ok = rocke_b_land(b, h_ok, w_ok);
    rocke_value_t* off = rocke_b_add(b, base, env->a_off);
    off = rocke_b_add(b, off, col);
    ok = rocke_b_land(b, ok, env->y_ok);
    if(env->tile_ok != NULL)
        ok = rocke_b_land(b, ok, env->tile_ok);
    *out_valid = ok;
    return off;
}

/* Python _b_uk(b_, row, col) [wrapped by _uk_masked when env->tile_ok]. */
static rocke_value_t* ukd_b_descriptor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* user)
{
    rocke_conv_ukd_env_t* env = (rocke_conv_ukd_env_t*)user;
    rocke_conv_build_ctx_t* ctx = env->ctx;
    rocke_value_t* n_val = rocke_b_add(b, ctx->block_n_off_v, row);
    rocke_value_t* k_out
        = (ctx->k_out_group_base != NULL) ? rocke_b_add(b, ctx->k_out_group_base, n_val) : n_val;
    rocke_value_t* k_base = rocke_b_mul(b, k_out, ctx->p_B_stride_k);
    rocke_value_t* off = rocke_b_add(b, k_base, env->b_off);
    off = rocke_b_add(b, off, col);
    rocke_value_t* ok = env->y_ok;
    if(env->tile_ok != NULL)
        ok = rocke_b_land(b, ok, env->tile_ok);
    *out_valid = ok;
    return off;
}

/* Python _uk_next(ct, xd, yd): the next tile's (c0, x, y); c0 wraps into the
 * next filter tap. */
static void ukd_next(rocke_conv_build_ctx_t* ctx,
                     rocke_value_t* c1,
                     rocke_value_t* ct,
                     rocke_value_t* xd,
                     rocke_value_t* yd,
                     rocke_value_t** out)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* nc = rocke_b_add(b, ct, ctx->c_block_k);
    rocke_value_t* wrap_c = rocke_b_cmp_ge(b, nc, ctx->p_cpg);
    nc = rocke_b_select(b, wrap_c, ctx->c0, nc);
    rocke_value_t* x_step = rocke_b_select(b, wrap_c, c1, ctx->c0);
    rocke_value_t* nx = rocke_b_add(b, xd, x_step);
    rocke_value_t* wrap_x = rocke_b_cmp_ge(b, nx, ctx->p_X);
    nx = rocke_b_select(b, wrap_x, ctx->c0, nx);
    rocke_value_t* y_step = rocke_b_select(b, wrap_x, c1, ctx->c0);
    rocke_value_t* ny = rocke_b_add(b, yd, y_step);
    out[0] = nc;
    out[1] = nx;
    out[2] = ny;
}

/* Python _uk_fetch(k_tile, ct, xd, yd): the tile's uniform terms, then its
 * A/B loads masked by k_tile < uk_end (_uk_masked). */
static void ukd_fetch(rocke_conv_build_ctx_t* ctx,
                      rocke_conv_ukd_env_t* env,
                      rocke_value_t* k_tile,
                      rocke_value_t* uk_end,
                      rocke_value_t* ct,
                      rocke_value_t* xd,
                      rocke_value_t* yd,
                      rocke_ctl_staged_t* a_staged,
                      rocke_ctl_staged_t* b_staged)
{
    rocke_ir_builder_t* b = ctx->b;
    ukd_tile(env, ct, xd, yd);
    ctx->k_off_capture = k_tile;
    env->tile_ok = rocke_b_cmp_lt(b, k_tile, uk_end);
    rocke_coalesced_tile_loader_load_global(
        b, &ctx->a_sync_loader, ctx->tid, ukd_a_descriptor, env, ctx->a_rsrc, NULL, a_staged);
    rocke_coalesced_tile_loader_load_global(
        b, &ctx->b_sync_loader, ctx->tid, ukd_b_descriptor, env, ctx->b_rsrc, NULL, b_staged);
}

/* Python _uk_store(staged) / _store(staged): the prefetched tile into LDS. */
static void ukd_store(rocke_conv_build_ctx_t* ctx,
                      const rocke_ctl_staged_t* a_staged,
                      const rocke_ctl_staged_t* b_staged)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_coalesced_tile_loader_store_lds(b, &ctx->a_sync_loader, ctx->A_smem, a_staged);
    rocke_coalesced_tile_loader_store_lds(b, &ctx->b_sync_loader, ctx->B_smem, b_staged);
}

void rocke_conv_emit_kloop_uniform_decode(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_conv_build_overrides_t* ov = ctx->ov;
    const int n_acc = ctx->num_accs;
    int i;

    if((ov != NULL && (ov->a_load_override != NULL || ov->m_index_fn != NULL)) || !ctx->split_ab)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "uniform_k_decode takes the split A/B addressing without loader "
                        "overrides");
        return;
    }
    rocke_value_t* uk_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    rocke_value_t* uk_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* cpg_rem = rocke_b_mod(b, ctx->p_cpg, ctx->c_block_k);
    rocke_value_t* uk_ok = rocke_b_cmp_eq(b, cpg_rem, ctx->c0);
    rocke_value_t* uk_q = rocke_b_div(b, uk_lo, ctx->p_cpg);
    rocke_value_t* uk_c0 = rocke_b_mod(b, uk_lo, ctx->p_cpg);
    rocke_value_t* uk_x0 = rocke_b_mod(b, uk_q, ctx->p_X);
    rocke_value_t* uk_y0 = rocke_b_div(b, uk_q, ctx->p_X);
    rocke_value_t* uk_end = rocke_b_select(b, uk_ok, uk_hi, uk_lo);

    rocke_conv_ukd_env_t env;
    memset(&env, 0, sizeof(env));
    env.ctx = ctx;

    rocke_iter_arg_t uk_iter[ROCKE_CONV_MAX_ACCS + 3];
    for(i = 0; i < n_acc; ++i)
    {
        uk_iter[i].name = ctx->acc_names[i];
        uk_iter[i].init = ctx->acc_inits[i];
    }
    uk_iter[n_acc + 0].name = "uk_c";
    uk_iter[n_acc + 0].init = uk_c0;
    uk_iter[n_acc + 1].name = "uk_x";
    uk_iter[n_acc + 1].init = uk_x0;
    uk_iter[n_acc + 2].name = "uk_y";
    uk_iter[n_acc + 2].init = uk_y0;

    rocke_for_t for_uk;
    if(ctx->spec->regt_prefetch)
    {
        /* Mode-1 register prefetch over the uniform decode (regt_prefetch):
         * the next tile's (c0, x, y) is advanced and its loads issued before
         * the MMAs; a tile at or past uk_end is masked to the out-of-range
         * sentinel and reads zero. */
        rocke_ctl_staged_t a_staged;
        rocke_ctl_staged_t b_staged;
        ukd_fetch(ctx, &env, uk_lo, uk_end, uk_c0, uk_x0, uk_y0, &a_staged, &b_staged);
        ukd_store(ctx, &a_staged, &b_staged);
        rocke_b_sync(b);
        for_uk = rocke_b_scf_for_iter(b,
                                      uk_lo,
                                      uk_end,
                                      ctx->c_block_k,
                                      uk_iter,
                                      n_acc + 3,
                                      "k_uk",
                                      /*unroll=*/false,
                                      /*elide_trailing_barrier=*/false);
        rocke_b_region_enter(b, for_uk.body);
        {
            rocke_value_t* nxt[3];
            rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
            rocke_value_t* yields[ROCKE_CONV_MAX_ACCS + 3];
            ukd_next(ctx,
                     c1,
                     for_uk.iter_vars[n_acc + 0],
                     for_uk.iter_vars[n_acc + 1],
                     for_uk.iter_vars[n_acc + 2],
                     nxt);
            rocke_value_t* k_next = rocke_b_add(b, for_uk.iv, ctx->c_block_k);
            ukd_fetch(ctx, &env, k_next, uk_end, nxt[0], nxt[1], nxt[2], &a_staged, &b_staged);
            rocke_b_sched_barrier(b, 0);
            ctx->k_off_capture = for_uk.iv;
            rocke_conv_emit_mfma_phase(
                ctx, ctx->A_smem, ctx->B_smem, for_uk.iter_vars, n_acc, new_accs);
            rocke_lds_barrier_keep_vmem(ctx->b, ctx->arch);
            ukd_store(ctx, &a_staged, &b_staged);
            rocke_lds_barrier_keep_vmem(ctx->b, ctx->arch);
            for(i = 0; i < n_acc; ++i)
                yields[i] = new_accs[i];
            yields[n_acc + 0] = nxt[0];
            yields[n_acc + 1] = nxt[1];
            yields[n_acc + 2] = nxt[2];
            rocke_b_scf_yield(b, yields, n_acc + 3);
        }
        rocke_b_region_leave(b);
    }
    else
    {
        for_uk = rocke_b_scf_for_iter(b,
                                      uk_lo,
                                      uk_end,
                                      ctx->c_block_k,
                                      uk_iter,
                                      n_acc + 3,
                                      "k_uk",
                                      /*unroll=*/false,
                                      /*elide_trailing_barrier=*/true);
        rocke_b_region_enter(b, for_uk.body);
        {
            rocke_value_t* ct = for_uk.iter_vars[n_acc + 0];
            rocke_value_t* xd = for_uk.iter_vars[n_acc + 1];
            rocke_value_t* yd = for_uk.iter_vars[n_acc + 2];
            rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
            rocke_value_t* yields[ROCKE_CONV_MAX_ACCS + 3];
            rocke_ctl_staged_t a_staged;
            rocke_ctl_staged_t b_staged;

            ukd_tile(&env, ct, xd, yd);
            ctx->k_off_capture = for_uk.iv;
            rocke_coalesced_tile_loader_load_global(b,
                                                    &ctx->a_sync_loader,
                                                    ctx->tid,
                                                    ukd_a_descriptor,
                                                    &env,
                                                    ctx->a_rsrc,
                                                    NULL,
                                                    &a_staged);
            rocke_coalesced_tile_loader_load_global(b,
                                                    &ctx->b_sync_loader,
                                                    ctx->tid,
                                                    ukd_b_descriptor,
                                                    &env,
                                                    ctx->b_rsrc,
                                                    NULL,
                                                    &b_staged);
            rocke_coalesced_tile_loader_store_lds(b, &ctx->a_sync_loader, ctx->A_smem, &a_staged);
            rocke_coalesced_tile_loader_store_lds(b, &ctx->b_sync_loader, ctx->B_smem, &b_staged);
            rocke_b_sync(b);
            rocke_conv_emit_mfma_phase(
                ctx, ctx->A_smem, ctx->B_smem, for_uk.iter_vars, n_acc, new_accs);
            rocke_b_sync(b);
            for(i = 0; i < n_acc; ++i)
                yields[i] = new_accs[i];
            ukd_next(ctx, c1, ct, xd, yd, &yields[n_acc]);
            rocke_b_scf_yield(b, yields, n_acc + 3);
        }
        rocke_b_region_leave(b);
    }

    /* Loop 2: the plain loop over [uk_end, uk_hi), its accs named fb_<acc>. */
    rocke_iter_arg_t fb_iter[ROCKE_CONV_MAX_ACCS];
    for(i = 0; i < n_acc; ++i)
    {
        char tmp[48];
        snprintf(tmp, sizeof(tmp), "fb_%s", ctx->acc_names[i]);
        fb_iter[i].name = rocke_arena_strdup(&b->arena, tmp);
        fb_iter[i].init = for_uk.op->results[i];
    }
    rocke_for_t for_fb = rocke_b_scf_for_iter(b,
                                              uk_end,
                                              uk_hi,
                                              ctx->c_block_k,
                                              fb_iter,
                                              n_acc,
                                              "k0",
                                              /*unroll=*/false,
                                              /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, for_fb.body);
    {
        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
        rocke_conv_emit_load_phase(ctx, for_fb.iv, ctx->A_smem, ctx->B_smem);
        rocke_b_sync(b);
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, for_fb.iter_vars, n_acc, new_accs);
        rocke_b_sync(b);
        rocke_b_scf_yield(b, new_accs, n_acc);
    }
    rocke_b_region_leave(b);
    rocke_conv_set_final_accs(ctx, for_fb.op->results, for_fb.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_regt_prefetch   (Python ``elif spec.regt_prefetch:``)
 *
 * Register-prefetch K loop (ImplicitGemmConvSpec.regt_prefetch = 1):
 *   prologue: tile k_lo -> buf; barrier
 *   per tile: regs <- tile k+1; MMA(buf); barrier; buf <- regs; barrier
 * Every tile at or past the reduction (split-K slice) end is masked to the
 * out-of-range sentinel, so it reads zero and adds nothing.
 * ===================================================================== */

/* Python _masked(fn, tile_ok): the plain descriptor ANDed with tile_ok (or
 * tile_ok alone where the descriptor has no predicate). */
typedef struct rocke_conv_rpf_env
{
    rocke_conv_build_ctx_t* ctx;
    rocke_loads_descriptor_fn fn;
    rocke_value_t* tile_ok;
} rocke_conv_rpf_env_t;

static rocke_value_t* rpf_masked_descriptor(rocke_ir_builder_t* b,
                                            rocke_value_t* row,
                                            rocke_value_t* col,
                                            rocke_value_t** out_valid,
                                            void* user)
{
    rocke_conv_rpf_env_t* env = (rocke_conv_rpf_env_t*)user;
    rocke_value_t* ok = NULL;
    rocke_value_t* off = env->fn(b, row, col, &ok, env->ctx);
    *out_valid = (ok == NULL) ? env->tile_ok : rocke_b_land(b, ok, env->tile_ok);
    return off;
}

/* Python _fetch(k_tile). */
static void rpf_fetch(rocke_conv_build_ctx_t* ctx,
                      rocke_conv_rpf_env_t* a_env,
                      rocke_conv_rpf_env_t* b_env,
                      rocke_value_t* k_tile,
                      rocke_value_t* k_upper,
                      rocke_ctl_staged_t* a_staged,
                      rocke_ctl_staged_t* b_staged)
{
    rocke_ir_builder_t* b = ctx->b;
    ctx->k_off_capture = k_tile;
    rocke_value_t* tile_ok = rocke_b_cmp_lt(b, k_tile, k_upper);
    a_env->tile_ok = tile_ok;
    b_env->tile_ok = tile_ok;
    rocke_coalesced_tile_loader_load_global(b,
                                            &ctx->a_sync_loader,
                                            ctx->tid,
                                            rpf_masked_descriptor,
                                            a_env,
                                            ctx->a_rsrc,
                                            NULL,
                                            a_staged);
    rocke_coalesced_tile_loader_load_global(b,
                                            &ctx->b_sync_loader,
                                            ctx->tid,
                                            rpf_masked_descriptor,
                                            b_env,
                                            ctx->b_rsrc,
                                            NULL,
                                            b_staged);
}

void rocke_conv_emit_kloop_regt_prefetch(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int n_acc = ctx->num_accs;
    int i;

    if(ctx->ov != NULL && ctx->ov->a_load_override != NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "regt_prefetch does not support an a_load_override");
        return;
    }
    rocke_value_t* k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    rocke_value_t* k_upper = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;

    rocke_conv_rpf_env_t a_env;
    rocke_conv_rpf_env_t b_env;
    a_env.ctx = ctx;
    a_env.fn = rocke_conv_a_descriptor;
    a_env.tile_ok = NULL;
    b_env.ctx = ctx;
    b_env.fn = (ctx->b_descriptor_fn != NULL) ? ctx->b_descriptor_fn : rocke_conv_b_descriptor;
    b_env.tile_ok = NULL;

    rocke_ctl_staged_t a_staged;
    rocke_ctl_staged_t b_staged;
    rpf_fetch(ctx, &a_env, &b_env, k_lo, k_upper, &a_staged, &b_staged);
    ukd_store(ctx, &a_staged, &b_staged);
    rocke_b_sync(b);

    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    for(i = 0; i < n_acc; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }
    rocke_for_t for_op = rocke_b_scf_for_iter(b,
                                              k_lo,
                                              k_upper,
                                              ctx->c_block_k,
                                              iter_args,
                                              n_acc,
                                              "k_rpf",
                                              /*unroll=*/false,
                                              /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
        rocke_value_t* k_next = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rpf_fetch(ctx, &a_env, &b_env, k_next, k_upper, &a_staged, &b_staged);
        /* Pin the prefetch ahead of the MMAs: the machine scheduler otherwise
         * sinks the loads to the end of the WMMA block. */
        rocke_b_sched_barrier(b, 0);
        ctx->k_off_capture = for_op.iv;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, for_op.iter_vars, n_acc, new_accs);
        rocke_lds_barrier_keep_vmem(ctx->b, ctx->arch);
        ukd_store(ctx, &a_staged, &b_staged);
        rocke_lds_barrier_keep_vmem(ctx->b, ctx->arch);
        rocke_b_scf_yield(b, new_accs, n_acc);
    }
    rocke_b_region_leave(b);
    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_async
 *
 * async_dma path: SoftwarePipeline.run_ping_pong_dynamic over the
 * AsyncTileLoader path, through the helper port
 * (helper_rocke.helpers.pipeline.h) with the policy the conv builders
 * construct:
 *
 *   SoftwarePipeline(wait_vmcnt=True, sync_after_wait=True,
 *                    sync_before_issue=True, overlap_vmcnt=True)
 *   issue_load(k, buf)      = emit_load_phase(k, buf[0], buf[1])
 *   compute(k, buf, state)  = emit_mfma_phase(buf[0], buf[1], state)
 *   buffers = [(A_smem,B_smem),(A_smem2,B_smem2)]
 *   schedule = ctx->schedule
 *   k_zero_fill = ctx->kloop_k_zero_fill (wgrad: wg_K under split-K)
 * ===================================================================== */
static void kloop_async_issue_load(rocke_ir_builder_t* b,
                                   rocke_value_t* k_offset,
                                   const rocke_buffer_pair_t* buf,
                                   void* user)
{
    (void)b;
    rocke_conv_emit_load_phase((rocke_conv_build_ctx_t*)user, k_offset, buf->a, buf->b);
}

static void kloop_async_compute(rocke_ir_builder_t* b,
                                rocke_value_t* k_offset,
                                const rocke_buffer_pair_t* buf,
                                rocke_value_t* const* state_in,
                                int num_state,
                                rocke_value_t** state_out,
                                void* user)
{
    rocke_conv_build_ctx_t* ctx = (rocke_conv_build_ctx_t*)user;
    (void)b;
    ctx->k_off_capture = k_offset;
    rocke_conv_emit_mfma_phase(ctx, buf->a, buf->b, state_in, num_state, state_out);
}

void rocke_conv_emit_kloop_async(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int num_accs = ctx->num_accs;
    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* results[ROCKE_CONV_MAX_ACCS];
    rocke_software_pipeline_t pipe;
    rocke_buffer_pair_t buffers[2];

    for(int i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }
    pipe.wait_vmcnt = true;
    /* The accumulators only ever see a zero tile past the extent. */
    pipe.mask_tail_state = false;
    pipe.sync_after_wait = true;
    pipe.sync_before_issue = true;
    pipe.overlap_vmcnt = true;
    buffers[0].a = ctx->A_smem;
    buffers[0].b = ctx->B_smem;
    buffers[1].a = ctx->A_smem2;
    buffers[1].b = ctx->B_smem2;

    if(!rocke_software_pipeline_run_ping_pong_dynamic(
           &pipe,
           b,
           (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm,
           ctx->block_k,
           (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0,
           ctx->kloop_k_zero_fill,
           buffers,
           iter_args,
           num_accs,
           kloop_async_issue_load,
           kloop_async_compute,
           ctx,
           &ctx->schedule,
           results))
    {
        return;
    }
    rocke_conv_set_final_accs(ctx, results, num_accs);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_wavelet   (Python lines 1539-1797)
 *
 * Dedicated load-wave / math-wave split pipeline (CK Tile #8009).
 *
 * The workgroup is split into two roles identified by warp_id:
 *   math waves  [0,         n_math_warps)  -- LDS reads + WMMA/MFMA + epilogue
 *   load waves  [n_math_warps, total_warps) -- DRAM→register fetch + LDS write
 *
 * Two sub-paths:
 *
 *   WMMA path (ctx->is_wmma == true, e.g. gfx1250):
 *     Uses scf_if_else (br i1) with a shared join block.  Both branches emit
 *     s_barrier calls; the shared join prevents simplifycfg from removing them.
 *
 *   MFMA path (ctx->is_wmma == false, e.g. gfx942):
 *     Uses exec-mask instructions (exec_and_saveexec / exec_xor /
 *     exec_or_saveexec / exec_or) to flat-sequentially encode the split.
 *     LLVM divergent branches would deadlock the barriers; exec-mask avoids
 *     that by keeping all threads in one basic block while suppressing VGPR
 *     writes for the inactive side.
 *
 * Barrier protocol (must be bit-identical in both branches):
 *
 *   MATH branch:
 *     barrier_0
 *     for i in 0..K_iters-2:
 *       MFMA(LDS)
 *       barrier_A   <- math done reading LDS
 *       barrier_B   <- wait for load to write next tile
 *     MFMA(LDS)     <- tail, no barriers
 *     [epilogue -- emits epi_barriers barriers]
 *
 *   LOAD branch:
 *     fetch tile 0 -> regs
 *     store regs -> LDS
 *     barrier_0
 *     for i in 0..K_iters-2:
 *       fetch tile i+1 -> regs    <- overlaps math MFMA
 *       barrier_A                 <- wait for math to release LDS
 *       store regs -> LDS
 *       barrier_B                 <- signal LDS ready
 *     [epilogue stub -- epi_barriers bare barriers, no stores]
 * ===================================================================== */

/* wavelet_fetch: issue buffer_load_vN for A and B into VGPR staging using
 * load_tid (load-wave-relative thread index). Sets k_off_capture.
 * Mirrors Python wavelet_fetch = a_wavelet_loader.fetch(b, k_off, load_tid). */
static void wavelet_fetch(rocke_conv_build_ctx_t* ctx,
                          rocke_value_t* k_off,
                          rocke_ctl_staged_t* a_staged,
                          rocke_ctl_staged_t* b_staged)
{
    rocke_ir_builder_t* b = ctx->b;
    ctx->k_off_capture = k_off;
    rocke_coalesced_tile_loader_load_global(b,
                                            &ctx->a_wavelet_loader,
                                            ctx->wavelet_load_tid,
                                            rocke_conv_a_descriptor,
                                            ctx,
                                            ctx->a_rsrc,
                                            NULL,
                                            a_staged);
    /* The merged B descriptor under group_merge, as the sync path: Python
     * swaps the closure once, so every consumer, the wavelet fetch included,
     * loads through it. */
    rocke_coalesced_tile_loader_load_global(
        b,
        &ctx->b_wavelet_loader,
        ctx->wavelet_load_tid,
        (ctx->b_descriptor_fn != NULL) ? ctx->b_descriptor_fn : rocke_conv_b_descriptor,
        ctx,
        ctx->b_rsrc,
        NULL,
        b_staged);
}

/* wavelet_store: write staged VGPRs to LDS.
 * Mirrors Python wavelet_store = a_wavelet_loader.store_fetched(b, a_regs, A_smem). */
static void wavelet_store(rocke_conv_build_ctx_t* ctx,
                          const rocke_ctl_staged_t* a_staged,
                          const rocke_ctl_staged_t* b_staged,
                          rocke_value_t* A_smem,
                          rocke_value_t* B_smem)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_coalesced_tile_loader_store_lds(b, &ctx->a_wavelet_loader, A_smem, a_staged);
    rocke_coalesced_tile_loader_store_lds(b, &ctx->b_wavelet_loader, B_smem, b_staged);
}

void rocke_conv_emit_kloop_wavelet(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int num_accs = ctx->num_accs;
    const int epi_barriers = ctx->wavelet_epi_barriers;
    int i;

    /* The wavelet driver emits the epilogue inline (inside the math branch).
     * Signal the build driver to skip its epilogue call. */
    ctx->epilogue_already_emitted = true;

    rocke_value_t* A_smem = ctx->A_smem;
    rocke_value_t* B_smem = ctx->B_smem;
    rocke_value_t* k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    rocke_value_t* k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    rocke_value_t* c_block_k;
    rocke_value_t* c_nmath;
    rocke_value_t* warp_id_s;

    rocke_ctl_staged_t a_staged;
    rocke_ctl_staged_t b_staged;

    /* Python emits this prologue at the top of emit_wavelet_kloop_dynamic, so
     * it lands after the D descriptor. c_block_k is a fresh constant there,
     * not the one the build prologue already made. */
    c_block_k = rocke_b_const_i32(b, ctx->block_k);
    c_nmath = rocke_b_const_i32(b, ctx->wavelet_n_math_warps);
    /* warp_id is tid/wave_size -- a VGPR. Materialise it as a scalar via
     * readfirstlane so the branch lowers to s_cmp + s_cbranch (uniform) rather
     * than v_cmpx (exec-masked), which would make barrier placement inside the
     * branch accidentally legal. */
    warp_id_s = rocke_b_readfirstlane(b, ctx->warp_id);
    ctx->wavelet_is_math = rocke_b_cmp_lt(b, warp_id_s, c_nmath);
    ctx->wavelet_load_tid
        = rocke_b_sub(b, ctx->tid, rocke_b_const_i32(b, ctx->wavelet_math_block_size));

    if(ctx->is_wmma)
    {
        /* ----------------------------------------------------------------
         * WMMA path: scf_if_else (gfx1250).
         *
         * gfx1250 has separate VMEM and WMMA issue slots, so load and math
         * waves truly overlap without exec-mask interleaving. The LLVM br i1
         * divergent branch is fine here — the hardware concurrency comes
         * from the distinct issue queues, not from hardware scheduling at
         * s_barrier.
         * ---------------------------------------------------------------- */
        rocke_if_else_t ife = rocke_b_scf_if_else(b, ctx->wavelet_is_math);

        /* ---- MATH WAVE branch ---- */
        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];

        rocke_b_region_enter(b, ife.then_region);
        {
            /* AOT: the reduction extent is a kernarg, so the compile-time
             * peel of the final tile is gone. Every iteration now emits the
             * full barrier pair, and the load branch below runs the same trip
             * count -- guarding the barriers instead would make the two wave
             * groups emit different counts and hang the workgroup. */
            rocke_iter_arg_t m_args[ROCKE_CONV_MAX_ACCS];
            rocke_for_t for_m;
            rocke_value_t* m_vars[ROCKE_CONV_MAX_ACCS];

            rocke_b_sync(b); /* barrier_0 */

            for(i = 0; i < num_accs; ++i)
            {
                m_args[i].name = ctx->acc_names[i];
                m_args[i].init = ctx->acc_inits[i];
            }
            for_m = rocke_b_scf_for_iter(b,
                                         k_lo,
                                         k_hi,
                                         c_block_k,
                                         m_args,
                                         num_accs,
                                         "k_math",
                                         /*unroll=*/false,
                                         /*elide_trailing_barrier=*/true);
            for(i = 0; i < for_m.num_iter_vars; ++i)
                m_vars[i] = for_m.iter_vars[i];

            rocke_b_region_enter(b, for_m.body);
            {
                ctx->k_off_capture = for_m.iv;
                rocke_conv_emit_mfma_phase(
                    ctx, A_smem, B_smem, m_vars, for_m.num_iter_vars, new_accs);
                rocke_b_sync(b); /* barrier_A */
                rocke_b_sync(b); /* barrier_B */
                rocke_b_scf_yield(b, new_accs, for_m.num_iter_vars);
            }
            rocke_b_region_leave(b);

            rocke_conv_set_final_accs(ctx, for_m.op->results, for_m.op->num_results);
            rocke_conv_emit_epilogue(ctx);
        }
        rocke_b_region_leave(b);

        /* ---- LOAD WAVE branch ---- */
        rocke_b_region_enter(b, ife.else_region);
        {
            /* fetch tile k_lo -> regs, store -> LDS, barrier_0 */
            rocke_for_t for_l;

            wavelet_fetch(ctx, k_lo, &a_staged, &b_staged);
            rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
            wavelet_store(ctx, &a_staged, &b_staged, A_smem, B_smem);
            rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
            rocke_b_sync(b); /* barrier_0 */

            /* Same trip count as the math branch -- that is what keeps the two
             * wave groups' barrier counts equal. The trailing iteration
             * prefetches a tile past the extent; its coords fall outside the
             * descriptor bounds, so the buffer resource returns zero. */
            for_l = rocke_b_scf_for(b, k_lo, k_hi, c_block_k, "k_load");
            rocke_b_region_enter(b, for_l.body);
            {
                wavelet_fetch(ctx, rocke_b_add(b, for_l.iv, c_block_k), &a_staged, &b_staged);
                rocke_b_sync(b); /* barrier_A */
                rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
                wavelet_store(ctx, &a_staged, &b_staged, A_smem, B_smem);
                rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
                rocke_b_sync(b); /* barrier_B */
            }
            rocke_b_region_leave(b);

            /* epilogue stub: N_epi bare barriers matching the math branch */
            for(i = 0; i < epi_barriers; ++i)
                rocke_b_sync(b);
        }
        rocke_b_region_leave(b);
    }
    else
    {
        /* The exec-mask MFMA wavelet path has a data race: the math section reads
         * a single-buffer LDS that the load section overwrites every K iteration.
         * True concurrent execution (different SIMD units) is required, but this
         * driver emits flat sequential code with no cross-section barriers, so
         * math reads garbage after iteration 0.
         * is_valid_spec blocks wavelet on MFMA targets, so this branch is
         * unreachable in normal use. Emit an error and bail. */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "pipeline='wavelet' is not supported for MFMA/CDNA targets: "
                        "the exec-mask split path has a data race (single LDS buffer "
                        "overwritten each K iteration with no cross-section barriers).");
    }
}
