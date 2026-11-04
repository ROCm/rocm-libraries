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
#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.pipeline.h"
#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/instance_conv_implicit_gemm_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include <stdio.h>
#include <string.h>

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

/* ===================================================================== *
 * rocke_conv_emit_fwd_kloops -- Python _emit_fwd_kloops: the reg_prefetch /
 * uniform_k_decode K loops (run_loop), the uniform-tap loop and its plain
 * fallback. A prefetch past the end of the reduction addresses an
 * out-of-range tap / k and loads zeros without touching memory.
 * ===================================================================== */
typedef struct
{
    rocke_ctl_staged_t a;
    rocke_ctl_staged_t b;
} conv_rpf_staged_t;

#define CONV_UK_MAX_STATE 4

/* The uniform-tap loop's build-time state (Python closures of the uk section). */
typedef struct
{
    rocke_conv_build_ctx_t* ctx;
    bool is3;
    rocke_value_t* a_ystep;
    rocke_value_t* a_xstep;
    rocke_value_t* a_zstep;
    rocke_value_t* a_gbase;
    /* uk_cap[0] = tile_uniforms(st) */
    rocke_value_t* hsh;
    rocke_value_t* wsh;
    rocke_value_t* dsh;
    rocke_value_t* a_off;
    rocke_value_t* b_off;
    rocke_value_t* t_ok;
} conv_uk_t;

/* Python a_uk(b_, row, col). */
static rocke_value_t* conv_uk_a_desc(rocke_ir_builder_t* b,
                                     rocke_value_t* row,
                                     rocke_value_t* col,
                                     rocke_value_t** out_valid,
                                     void* user)
{
    conv_uk_t* u = (conv_uk_t*)user;
    rocke_conv_build_ctx_t* ctx = u->ctx;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    rocke_value_t* m_val = rocke_b_add(b, ctx->block_m_off_v, row);
    rocke_value_t *q_wo, *wo, *q_ho, *ho, *n, *do_ = NULL;
    rocke_conv_magic_divmod(
        b, m_val, ctx->p_magic_m_Wo_mult, ctx->p_magic_m_Wo_shift, ctx->p_Wo, false, &q_wo, &wo);
    rocke_conv_magic_divmod(
        b, q_wo, ctx->p_magic_m_Ho_mult, ctx->p_magic_m_Ho_shift, ctx->p_Ho, false, &q_ho, &ho);
    if(u->is3)
        rocke_conv_magic_divmod(
            b, q_ho, ctx->p_magic_m_Do_mult, ctx->p_magic_m_Do_shift, ctx->p_Do, false, &n, &do_);
    else
        n = q_ho;
    rocke_value_t* hi0 = rocke_b_sub(b, rocke_b_mul(b, ho, ctx->p_sH), ctx->p_pH);
    rocke_value_t* wi0 = rocke_b_sub(b, rocke_b_mul(b, wo, ctx->p_sW), ctx->p_pW);
    rocke_value_t* n_off = rocke_b_mul(b, n, ctx->p_A_stride_n);
    rocke_value_t* h_off = rocke_b_mul(b, hi0, ctx->p_A_stride_hi);
    rocke_value_t* base = rocke_b_add(b, n_off, h_off);
    base = rocke_b_add(b, base, rocke_b_mul(b, wi0, ctx->p_A_stride_wi));
    if(spec->uk_nopad)
    {
        /* Unpadded: every tap of an output pixel is in range; only the
         * tap-range check of the prefetch past the end stays. */
        if(u->is3)
        {
            rocke_value_t* di0 = rocke_b_sub(b, rocke_b_mul(b, do_, ctx->p_sD), ctx->p_pD);
            base = rocke_b_add(b, base, rocke_b_mul(b, di0, ctx->p_A_stride_di));
        }
        if(u->a_gbase != NULL)
            base = rocke_b_add(b, base, u->a_gbase);
        *out_valid = u->t_ok;
        return rocke_b_add(b, rocke_b_add(b, base, col), u->a_off);
    }
    /* (u32)(origin + shift) < extent  <=>  0 <= origin + shift < extent */
    rocke_value_t* h_ok = rocke_b_cmp_ult(b, rocke_b_add(b, hi0, u->hsh), ctx->p_Hi);
    rocke_value_t* w_ok = rocke_b_cmp_ult(b, rocke_b_add(b, wi0, u->wsh), ctx->p_Wi);
    rocke_value_t* ok = rocke_b_land(b, h_ok, w_ok);
    if(u->is3)
    {
        rocke_value_t* di0 = rocke_b_sub(b, rocke_b_mul(b, do_, ctx->p_sD), ctx->p_pD);
        base = rocke_b_add(b, base, rocke_b_mul(b, di0, ctx->p_A_stride_di));
        rocke_value_t* d_ok = rocke_b_cmp_ult(b, rocke_b_add(b, di0, u->dsh), ctx->p_Di);
        ok = rocke_b_land(b, ok, d_ok);
    }
    if(u->a_gbase != NULL)
        base = rocke_b_add(b, base, u->a_gbase);
    ok = rocke_b_land(b, ok, u->t_ok);
    *out_valid = ok;
    return rocke_b_add(b, rocke_b_add(b, base, col), u->a_off);
}

/* Python _eqx_chan (epi_quad_xpose 2/3/4): the output channel (tile-relative)
 * whose weights the B LDS row r holds; identity otherwise. */
rocke_value_t* rocke_conv_eqx_chan(rocke_conv_build_ctx_t* ctx, rocke_value_t* r)
{
    rocke_ir_builder_t* b = ctx->b;
    const int eqx = ctx->spec->epi_quad_xpose;
    if(eqx != 2 && eqx != 3 && eqx != 4)
        return r;
    rocke_value_t* p32 = rocke_b_land(b, r, rocke_b_const_i32(b, ~31));
    rocke_value_t* ni_lo = rocke_b_land(b, r, rocke_b_const_i32(b, 16));
    rocke_value_t* j = rocke_b_land(b, r, rocke_b_const_i32(b, 15));
    rocke_value_t* jj;
    if(eqx == 2)
    {
        rocke_value_t* jl = rocke_b_land(b, j, rocke_b_const_i32(b, 12));
        rocke_value_t* a = rocke_b_mul(b, jl, rocke_b_const_i32(b, 2));
        jj = rocke_b_add(b, a, rocke_b_land(b, j, rocke_b_const_i32(b, 3)));
    }
    else
    {
        rocke_value_t* jl = rocke_b_land(b, j, rocke_b_const_i32(b, 3));
        rocke_value_t* a = rocke_b_mul(b, jl, rocke_b_const_i32(b, 8));
        jj = rocke_b_add(b, a, rocke_b_div(b, j, rocke_b_const_i32(b, 4)));
    }
    rocke_value_t* q = rocke_b_div(b, ni_lo, rocke_b_const_i32(b, 4));
    return rocke_b_add(b, rocke_b_add(b, p32, q), jj);
}

/* Python b_uk(b_, row, col) (epi_quad_xpose's channel permutation: identity). */
static rocke_value_t* conv_uk_b_desc(rocke_ir_builder_t* b,
                                     rocke_value_t* row,
                                     rocke_value_t* col,
                                     rocke_value_t** out_valid,
                                     void* user)
{
    conv_uk_t* u = (conv_uk_t*)user;
    rocke_conv_build_ctx_t* ctx = u->ctx;
    rocke_value_t* k_out = rocke_b_add(b, ctx->block_n_off_v, rocke_conv_eqx_chan(ctx, row));
    if(ctx->k_out_group_base != NULL)
        k_out = rocke_b_add(b, ctx->k_out_group_base, k_out);
    rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, k_out, ctx->p_B_stride_k), col);
    off = rocke_b_add(b, off, u->b_off);
    *out_valid = u->t_ok;
    return off;
}

/* Python tile_uniforms(st), st = (c, x, y[, z]). */
static void conv_uk_tile_uniforms(conv_uk_t* u, rocke_value_t* const* st)
{
    rocke_conv_build_ctx_t* ctx = u->ctx;
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c_t = st[0];
    rocke_value_t* x_t = st[1];
    rocke_value_t* y_t = st[2];
    u->hsh = rocke_b_mul(b, y_t, ctx->p_dH);
    u->wsh = rocke_b_mul(b, x_t, ctx->p_dW);
    rocke_value_t* ay = rocke_b_mul(b, y_t, u->a_ystep);
    rocke_value_t* ax = rocke_b_mul(b, x_t, u->a_xstep);
    u->a_off = rocke_b_add(b, rocke_b_add(b, ay, ax), c_t);
    rocke_value_t* by = rocke_b_mul(b, y_t, ctx->p_B_stride_y);
    rocke_value_t* bx = rocke_b_mul(b, x_t, ctx->p_B_stride_x);
    u->b_off = rocke_b_add(b, rocke_b_add(b, by, bx), c_t);
    if(u->is3)
    {
        rocke_value_t* z_t = st[3];
        u->dsh = rocke_b_mul(b, z_t, ctx->p_dD);
        u->a_off = rocke_b_add(b, u->a_off, rocke_b_mul(b, z_t, u->a_zstep));
        u->b_off = rocke_b_add(b, u->b_off, rocke_b_mul(b, z_t, ctx->p_B_stride_z));
        u->t_ok = rocke_b_cmp_lt(b, z_t, ctx->p_Z);
    }
    else
    {
        u->dsh = NULL;
        u->t_ok = rocke_b_cmp_lt(b, y_t, ctx->p_Y);
    }
}

/* Python uk_adv(st): the next tile's (c, x, y[, z]). */
static void conv_uk_adv(conv_uk_t* u, rocke_value_t* const* st, rocke_value_t** out)
{
    rocke_conv_build_ctx_t* ctx = u->ctx;
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* c_t = st[0];
    rocke_value_t* x_t = st[1];
    rocke_value_t* y_t = st[2];
    rocke_value_t* z_t = u->is3 ? st[3] : NULL;
    if(spec->uk_tap_fast && spec->uk_chunk_group)
    {
        /* Chunks fastest inside a group of uk_chunk_group chunks, then the
         * tap (x, y, z), then the next group. */
        rocke_value_t* c_gspan = rocke_b_const_i32(b, spec->uk_chunk_group * ctx->block_k);
        rocke_value_t* c_gmask = rocke_b_const_i32(b, spec->uk_chunk_group * ctx->block_k - 1);
        rocke_value_t* nc = rocke_b_add(b, c_t, ctx->c_block_k);
        rocke_value_t* e1 = rocke_b_cmp_eq(b, rocke_b_land(b, nc, c_gmask), c0);
        rocke_value_t* w_c = rocke_b_lor(b, e1, rocke_b_cmp_ge(b, nc, ctx->p_cpg));
        nc = rocke_b_select(b, w_c, rocke_b_sub(b, c_t, rocke_b_land(b, c_t, c_gmask)), nc);
        rocke_value_t* nx = rocke_b_add(b, x_t, rocke_b_select(b, w_c, ctx->uk_c1, c0));
        rocke_value_t* w_x = rocke_b_cmp_ge(b, nx, ctx->p_X);
        nx = rocke_b_select(b, w_x, c0, nx);
        rocke_value_t* ny = rocke_b_add(b, y_t, rocke_b_select(b, w_x, ctx->uk_c1, c0));
        rocke_value_t* w_y = rocke_b_cmp_ge(b, ny, ctx->p_Y);
        ny = rocke_b_select(b, w_y, c0, ny);
        rocke_value_t* nz = rocke_b_add(b, z_t, rocke_b_select(b, w_y, ctx->uk_c1, c0));
        rocke_value_t* w_z = rocke_b_cmp_ge(b, nz, ctx->p_Z);
        nz = rocke_b_select(b, w_z, c0, nz);
        nc = rocke_b_add(b, nc, rocke_b_select(b, w_z, c_gspan, c0));
        nz = rocke_b_select(b, rocke_b_cmp_ge(b, nc, ctx->p_cpg), ctx->p_Z, nz);
        out[0] = nc;
        out[1] = nx;
        out[2] = ny;
        out[3] = nz;
        return;
    }
    if(spec->uk_tap_fast)
    {
        /* Filter taps fastest, channel chunk outermost. */
        rocke_value_t* nx = rocke_b_add(b, x_t, ctx->uk_c1);
        rocke_value_t* w_x = rocke_b_cmp_ge(b, nx, ctx->p_X);
        nx = rocke_b_select(b, w_x, c0, nx);
        rocke_value_t* ny = rocke_b_add(b, y_t, rocke_b_select(b, w_x, ctx->uk_c1, c0));
        rocke_value_t* w_y = rocke_b_cmp_ge(b, ny, ctx->p_Y);
        ny = rocke_b_select(b, w_y, c0, ny);
        rocke_value_t* nz = rocke_b_add(b, z_t, rocke_b_select(b, w_y, ctx->uk_c1, c0));
        rocke_value_t* w_z = rocke_b_cmp_ge(b, nz, ctx->p_Z);
        nz = rocke_b_select(b, w_z, c0, nz);
        rocke_value_t* nc = rocke_b_add(b, c_t, rocke_b_select(b, w_z, ctx->c_block_k, c0));
        /* Past the end: park on an out-of-range tap (z = Z). */
        nz = rocke_b_select(b, rocke_b_cmp_ge(b, nc, ctx->p_cpg), ctx->p_Z, nz);
        out[0] = nc;
        out[1] = nx;
        out[2] = ny;
        out[3] = nz;
        return;
    }
    rocke_value_t* nc = rocke_b_add(b, c_t, ctx->c_block_k);
    rocke_value_t* w_c = rocke_b_cmp_ge(b, nc, ctx->p_cpg);
    nc = rocke_b_select(b, w_c, c0, nc);
    rocke_value_t* nx = rocke_b_add(b, x_t, rocke_b_select(b, w_c, ctx->uk_c1, c0));
    rocke_value_t* w_x = rocke_b_cmp_ge(b, nx, ctx->p_X);
    nx = rocke_b_select(b, w_x, c0, nx);
    rocke_value_t* ny = rocke_b_add(b, y_t, rocke_b_select(b, w_x, ctx->uk_c1, c0));
    out[0] = nc;
    out[1] = nx;
    out[2] = ny;
    if(!u->is3)
        return;
    rocke_value_t* w_y = rocke_b_cmp_ge(b, ny, ctx->p_Y);
    ny = rocke_b_select(b, w_y, c0, ny);
    rocke_value_t* nz = rocke_b_add(b, z_t, rocke_b_select(b, w_y, ctx->uk_c1, c0));
    out[2] = ny;
    out[3] = nz;
}

/* The run_loop "fetch_tile" / "advance" pair: plain (u == NULL) or uniform-tap. */
static void conv_kl_fetch(rocke_conv_build_ctx_t* ctx,
                          conv_uk_t* u,
                          rocke_value_t* k,
                          rocke_value_t* const* state,
                          conv_rpf_staged_t* st)
{
    rocke_ir_builder_t* b = ctx->b;
    /* fetch(): _fetch_part (pp_split_fetch) selects the operands read. */
    const bool fa = !ctx->fetch_skip_a;
    const bool fb = !ctx->fetch_skip_b;
    st->a.count = 0;
    st->b.count = 0;
    if(u == NULL)
    {
        /* plain_fetch: k_off_capture = k; fetch(a_descriptor, b_descriptor) */
        ctx->k_off_capture = k;
        if(fa)
            rocke_coalesced_tile_loader_load_global(b,
                                                    &ctx->a_sync_loader,
                                                    ctx->tid,
                                                    rocke_conv_a_descriptor,
                                                    ctx,
                                                    ctx->a_rsrc,
                                                    NULL,
                                                    &st->a);
        if(fb)
            rocke_coalesced_tile_loader_load_global(b,
                                                    &ctx->b_sync_loader,
                                                    ctx->tid,
                                                    rocke_conv_b_descriptor,
                                                    ctx,
                                                    ctx->b_rsrc,
                                                    NULL,
                                                    &st->b);
        return;
    }
    /* uk_fetch: uk_cap[0] = tile_uniforms(st); fetch(a_uk, b_uk) */
    conv_uk_tile_uniforms(u, state);
    if(ctx->fetch_hi != NULL)
    {
        /* pp_end_guard: no reads for a tile at or past the loop's end. */
        rocke_value_t* in_range = rocke_b_cmp_lt(b, k, ctx->fetch_hi);
        u->t_ok = rocke_b_land(b, u->t_ok, in_range);
    }
    if(fa)
        rocke_coalesced_tile_loader_load_global(
            b, &ctx->a_sync_loader, ctx->tid, conv_uk_a_desc, u, ctx->a_rsrc, NULL, &st->a);
    if(fb)
        rocke_coalesced_tile_loader_load_global(
            b, &ctx->b_sync_loader, ctx->tid, conv_uk_b_desc, u, ctx->b_rsrc, NULL, &st->b);
}

/* part(which): _fetch_part.update(a=which in (None, 0), b=which in (None, 1));
 * which = -1 for None. */
static void conv_part(rocke_conv_build_ctx_t* ctx, int which)
{
    ctx->fetch_skip_a = !(which == -1 || which == 0);
    ctx->fetch_skip_b = !(which == -1 || which == 1);
}

static void conv_kl_adv(conv_uk_t* u, rocke_value_t* const* st, int n_state, rocke_value_t** out)
{
    if(u == NULL)
    {
        for(int i = 0; i < n_state; ++i)
            out[i] = st[i];
        return;
    }
    conv_uk_adv(u, st, out);
}

/* store(staged, A_dst, B_dst): the loaders' LDS stores, or the lds_xor
 * swizzled ones (smem_store_vN at [row, lds_swz(row, col)]). */
static void conv_rpf_store(rocke_conv_build_ctx_t* ctx,
                           const conv_rpf_staged_t* st,
                           rocke_value_t* A_dst,
                           rocke_value_t* B_dst)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->spec->lds_xor)
    {
        rocke_coalesced_tile_loader_store_lds(b, &ctx->a_sync_loader, A_dst, &st->a);
        rocke_coalesced_tile_loader_store_lds(b, &ctx->b_sync_loader, B_dst, &st->b);
        return;
    }
    const rocke_ctl_staged_t* parts[2] = {&st->a, &st->b};
    rocke_value_t* dsts[2] = {A_dst, B_dst};
    int vecs[2] = {ctx->a_sync_loader.load_vec, ctx->b_sync_loader.load_vec};
    for(int o = 0; o < 2; ++o)
    {
        for(int e = 0; e < parts[o]->count; ++e)
        {
            rocke_value_t* row = parts[o]->vecs[e].row;
            rocke_value_t* col = rocke_conv_lds_swz(ctx, row, parts[o]->vecs[e].col);
            rocke_value_t* idx[2] = {row, col};
            rocke_b_smem_store_vN(b, dsts[o], idx, 2, parts[o]->vecs[e].v, vecs[o]);
        }
    }
}

/* Python run_pingpong(lo, hi, it1, n_acc, fetch_tile, advance, tag): the
 * rpf_pingpong loop, entered with tile `lo` in the LDS buffer, published.
 * The C port carries pp 1-5, pp_pre_mfma, pp_b_double, vmem_interleave,
 * rpf_setprio and pp_split_fetch 1-4 / 10 (no pp_fetch_deep /
 * x-halo / a_direct, which its validator rejects). */
typedef struct
{
    rocke_conv_build_ctx_t* ctx;
    conv_uk_t* u;
    int n_acc;
    int n_st;
    int n_rg;
    int n_a;
    int pre;
    int h_mf;
    bool late;
    bool in_comp;
    int psf;
    int sf_m; /* -1 => None */
    int sf_c;
    bool psf_ahead;
    bool psf_memb;
    bool bdb;
    int half_kk;
    rocke_value_t* c_2bk;
    rocke_value_t* c_fbk;
    conv_rpf_staged_t* staged0;
    conv_rpf_staged_t* tmp;
    conv_rpf_staged_t* staged_m;
    conv_rpf_staged_t* nxt_c;
    conv_rpf_staged_t* nxt;
    rocke_conv_frag_t* frags;
    rocke_conv_frag_t* bf;
} conv_pp_t;

static void conv_pp_pin(rocke_ir_builder_t* b)
{
    rocke_b_sched_barrier(b, 0);
}

static void conv_pp_prio(conv_pp_t* p, bool on)
{
    const rocke_implicit_gemm_conv_spec_t* spec = p->ctx->spec;
    if(spec->rpf_setprio)
        rocke_b_s_setprio(p->ctx->b, on ? spec->rpf_setprio : 0);
}

/* mfmas(frags, accs, i0, i1) = emit_mfma_phase(A_smem, B_smem, accs, pp_use=...) */
static void conv_pp_mfmas(conv_pp_t* p, rocke_value_t** accs, int i0, int i1)
{
    rocke_conv_build_ctx_t* ctx = p->ctx;
    rocke_value_t* out[ROCKE_CONV_MAX_ACCS];
    ctx->pp_use = p->frags;
    ctx->pp_i0 = i0;
    ctx->pp_i1 = i1;
    rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, accs, p->n_acc, out);
    ctx->pp_use = NULL;
    for(int i = 0; i < p->n_acc; ++i)
        accs[i] = out[i];
}

static void conv_pp_sched_loads(conv_pp_t* p, int n_ld, int n_m)
{
    rocke_ir_builder_t* b = p->ctx->b;
    const int vi = p->ctx->spec->vmem_interleave;
    for(int i = 0; i < n_ld; ++i)
    {
        const int take = vi < n_m ? vi : n_m;
        if(take)
        {
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, take, 0);
            n_m -= take;
        }
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_VMEM_READ, 1, 0);
    }
    if(n_m)
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, n_m, 0);
}

static void conv_pp_bar_mem_a(conv_pp_t* p)
{
    rocke_ir_builder_t* b = p->ctx->b;
    const int pp = p->ctx->spec->rpf_pingpong;
    if(pp == 1 || pp == 4 || pp == 5)
        rocke_b_sync_lds_only_fence(b, "compiler");
    else if(pp == 2)
        rocke_b_s_barrier_bare_fence(b, "compiler");
    else
        rocke_b_s_barrier_bare_fence(b, "none");
}

/* open_compute: the first `pre` MFMAs of the phase, then its barrier. */
static void conv_pp_open(conv_pp_t* p, rocke_value_t** accs, int i0, bool mem_a)
{
    rocke_ir_builder_t* b = p->ctx->b;
    conv_pp_prio(p, true);
    if(p->pre)
        conv_pp_mfmas(p, accs, i0, i0 + p->pre);
    conv_pp_pin(b);
    if(mem_a)
        conv_pp_bar_mem_a(p);
    else
        rocke_b_sync_lds_only_fence(b, "compiler");
    conv_pp_pin(b);
}

static void conv_pp_close(conv_pp_t* p, rocke_value_t** accs, int i0, int i1)
{
    rocke_ir_builder_t* b = p->ctx->b;
    conv_pp_mfmas(p, accs, i0, i1);
    conv_pp_pin(b);
    conv_pp_prio(p, false);
    rocke_b_s_barrier_bare_fence(b, "compiler");
    conv_pp_pin(b);
}

/* tuple(staged_m[i] if i == sf_m else x[i] for i in (0, 1)) */
static void conv_pp_merge(const conv_rpf_staged_t* m, int sf_m, conv_rpf_staged_t* x)
{
    if(sf_m == 0)
        x->a = m->a;
    else if(sf_m == 1)
        x->b = m->b;
}

/* body(kv, iv, B_cur, B_nxt) -> the values the loop carries on (out). */
static int conv_pp_body(conv_pp_t* p,
                        rocke_value_t* kv,
                        rocke_value_t* const* iv,
                        rocke_value_t* B_cur,
                        rocke_value_t* B_nxt,
                        rocke_value_t** out)
{
    rocke_conv_build_ctx_t* ctx = p->ctx;
    rocke_ir_builder_t* b = ctx->b;
    const int n_acc = p->n_acc;
    const int n_st = p->n_st;
    rocke_value_t* accs[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* st[CONV_UK_MAX_STATE];
    rocke_value_t* st_ahead[CONV_UK_MAX_STATE];
    rocke_value_t* st_n[CONV_UK_MAX_STATE];
    for(int i = 0; i < n_acc; ++i)
        accs[i] = iv[i];
    for(int i = 0; i < n_st; ++i)
        st[i] = iv[n_acc + i];

    /* [mem A] every fragment of tile t; pp_b_double: B of the second half later. */
    ctx->k_off_capture = kv;
    if(p->bdb)
        ctx->pp_b_from_p1 = p->half_kk + 1;
    ctx->pp_read = p->frags;
    ctx->pp_read_n = 0;
    {
        rocke_value_t* dummy[ROCKE_CONV_MAX_ACCS];
        rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, B_cur, accs, n_acc, dummy);
    }
    ctx->pp_read = NULL;
    ctx->pp_b_from_p1 = 0;
    conv_pp_pin(b);
    if(p->psf == 10)
    {
        conv_kl_adv(p->u, st, n_st, st_ahead);
    }
    else if(p->psf_ahead)
    {
        /* pp_split_fetch 3/4: tile t+2's [mem A] operand. */
        conv_kl_adv(p->u, st, n_st, st_ahead);
        conv_part(ctx, p->sf_m);
        conv_kl_fetch(ctx, p->u, rocke_b_add(b, kv, p->c_2bk), st_ahead, p->staged_m);
        conv_part(ctx, -1);
        conv_pp_pin(b);
    }
    else if(p->psf)
    {
        /* pp_split_fetch 1/2: tile t+1's other operand (st is its state). */
        conv_part(ctx, p->sf_m);
        conv_kl_fetch(ctx, p->u, rocke_b_add(b, kv, ctx->c_block_k), st, p->staged_m);
        conv_part(ctx, -1);
        conv_pp_pin(b);
    }
    conv_rpf_staged_t* staged = p->tmp;
    if(p->late)
    {
        for(int i = 0; i < n_st; ++i)
            st_n[i] = st[i];
    }
    else
    {
        conv_kl_adv(p->u, st, n_st, st_n);
        conv_kl_fetch(ctx, p->u, rocke_b_add(b, kv, ctx->c_block_k), st_n, staged);
        conv_pp_pin(b);
    }
    conv_pp_open(p, accs, 0, true);
    /* [compute A] */
    conv_pp_close(p, accs, p->pre, p->h_mf);
    /* [mem B] tile t+1 into the buffer. */
    if(p->late)
    {
        rocke_value_t* const* regs = iv + n_acc + n_st;
        *staged = *p->staged0;
        for(int e = 0; e < staged->a.count; ++e)
            staged->a.vecs[e].v = regs[e];
        for(int e = 0; e < staged->b.count; ++e)
            staged->b.vecs[e].v = regs[p->n_a + e];
        if(p->psf && !p->psf_ahead)
            conv_pp_merge(p->staged_m, p->sf_m, staged);
    }
    conv_rpf_store(ctx, staged, ctx->A_smem, B_nxt);
    conv_pp_pin(b);
    if(p->psf_memb)
    {
        conv_part(ctx, p->psf == 10 ? -1 : p->sf_c);
        conv_kl_fetch(ctx, p->u, rocke_b_add(b, kv, p->c_2bk), st_ahead, p->nxt_c);
        conv_part(ctx, -1);
        conv_pp_pin(b);
    }
    if(p->bdb)
    {
        /* The second half's B columns of tile t. */
        ctx->pp_b_only = true;
        ctx->pp_kk_from = p->half_kk;
        ctx->pp_read = p->bf;
        ctx->pp_read_n = 0;
        rocke_value_t* dummy[ROCKE_CONV_MAX_ACCS];
        rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, B_cur, accs, n_acc, dummy);
        const int nbf = ctx->pp_read_n;
        ctx->pp_read = NULL;
        ctx->pp_b_only = false;
        ctx->pp_kk_from = 0;
        for(int i = 0; i < nbf; ++i)
        {
            rocke_conv_frag_t* f = &p->frags[p->half_kk + i];
            for(int j = 0; j < p->bf[i].nb; ++j)
                f->b[j] = p->bf[i].b[j];
            f->nb = p->bf[i].nb;
        }
        conv_pp_pin(b);
    }
    conv_rpf_staged_t* nxt = p->nxt;
    if(p->late && !p->in_comp)
    {
        conv_kl_adv(p->u, st, n_st, st_n);
        conv_kl_fetch(ctx, p->u, rocke_b_add(b, kv, p->c_fbk), st_n, nxt);
        conv_pp_pin(b);
    }
    conv_pp_open(p, accs, p->h_mf, false);
    /* [compute B] */
    if(p->in_comp)
    {
        conv_pp_mfmas(p, accs, p->h_mf + p->pre, 2 * p->h_mf);
        if(p->psf_ahead)
            for(int i = 0; i < n_st; ++i)
                st_n[i] = st_ahead[i];
        else
            conv_kl_adv(p->u, st, n_st, st_n);
        int n_ld;
        if(p->psf_memb)
        {
            *nxt = *p->nxt_c;
            n_ld = 0;
        }
        else
        {
            conv_part(ctx, p->sf_c);
            conv_kl_fetch(ctx, p->u, rocke_b_add(b, kv, p->c_fbk), st_n, nxt);
            conv_part(ctx, -1);
            n_ld = nxt->a.count + nxt->b.count;
        }
        if(p->psf_ahead)
            conv_pp_merge(p->staged_m, p->sf_m, nxt);
        conv_pp_sched_loads(p, n_ld, p->h_mf - p->pre);
        conv_pp_pin(b);
        conv_pp_prio(p, false);
        rocke_b_s_barrier_bare_fence(b, "compiler");
        conv_pp_pin(b);
    }
    else
    {
        conv_pp_close(p, accs, p->h_mf + p->pre, 2 * p->h_mf);
    }
    int n = 0;
    for(int i = 0; i < n_acc; ++i)
        out[n++] = accs[i];
    for(int i = 0; i < n_st; ++i)
        out[n++] = st_n[i];
    if(p->late)
    {
        for(int e = 0; e < nxt->a.count; ++e)
            out[n++] = nxt->a.vecs[e].v;
        for(int e = 0; e < nxt->b.count; ++e)
            out[n++] = nxt->b.vecs[e].v;
    }
    return n;
}

#define CONV_PP_MAX_IT (ROCKE_CONV_MAX_ACCS + CONV_UK_MAX_STATE + 2 * ROCKE_CTL_MAX_VECS_PER_THREAD)

static void conv_run_pingpong(rocke_conv_build_ctx_t* ctx,
                              rocke_value_t* lo,
                              rocke_value_t* hi,
                              const rocke_iter_arg_t* it1,
                              int n_it1,
                              conv_uk_t* u,
                              const char* tag,
                              rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const int n_acc = ctx->num_accs;
    const int block_k = ctx->block_k;
    conv_pp_t* p = (conv_pp_t*)rocke_arena_alloc(&b->arena, sizeof(conv_pp_t));
    conv_rpf_staged_t* bufs
        = (conv_rpf_staged_t*)rocke_arena_alloc(&b->arena, 5 * sizeof(conv_rpf_staged_t));
    rocke_conv_frag_t* frags = (rocke_conv_frag_t*)rocke_arena_alloc(
        &b->arena, 2 * ctx->k_atoms * sizeof(rocke_conv_frag_t));
    if(p == NULL || bufs == NULL || frags == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "conv run_pingpong: arena alloc failed");
        return;
    }
    memset(p, 0, sizeof(*p));
    memset(bufs, 0, 5 * sizeof(conv_rpf_staged_t));
    p->ctx = ctx;
    p->u = u;
    p->n_acc = n_acc;
    p->staged0 = &bufs[0];
    p->tmp = &bufs[1];
    p->staged_m = &bufs[2];
    p->nxt_c = &bufs[3];
    p->nxt = &bufs[4];
    p->frags = frags;
    p->bf = frags + ctx->k_atoms;

    const int nw = spec->warp_m * spec->warp_n;
    rocke_value_t* wid = rocke_b_readfirstlane(b, ctx->warp_id);
    int sh = 0;
    for(int v = nw / 2; v > 0; v >>= 1)
        ++sh; /* (nw // 2).bit_length() */
    rocke_value_t* hi_half = rocke_b_lshr(b, wid, rocke_b_const_i32(b, sh - 1));
    rocke_value_t* lo_half = rocke_b_sub(b, ctx->uk_c1, hi_half);
    p->h_mf = ctx->k_atoms * ctx->mfmas_m * ctx->mfmas_n / 2;
    p->pre = spec->pp_pre_mfma;
    p->late = spec->rpf_pingpong == 4 || spec->rpf_pingpong == 5;
    p->in_comp = spec->rpf_pingpong == 5;

    if(spec->pp_end_guard)
        ctx->fetch_hi = hi;
    /* The high half runs one barrier ahead. */
    rocke_b_sync_half_block(b, hi_half);
    /* Retire every scalar load before the loop. */
    rocke_b_s_waitcnt(b, -1, 0, -1);
    rocke_iter_arg_t it[CONV_PP_MAX_IT];
    int n_it = n_it1;
    for(int i = 0; i < n_it1; ++i)
        it[i] = it1[i];
    p->n_st = n_it1 - n_acc;
    p->psf = (p->late && p->in_comp) ? spec->pp_split_fetch : 0;
    static const int sfm[11] = {-1, 1, 0, 1, 0, 1, 1, 1, 1, 1, -1};
    static const int sfc[11] = {-1, 0, 1, 0, 1, 0, 0, 0, 0, 0, -1};
    p->sf_m = sfm[p->psf];
    p->sf_c = sfc[p->psf];
    p->psf_ahead = p->psf >= 3;
    p->psf_memb = p->psf == 5 || p->psf == 10;
    char nbuf[96];

    if(p->late)
    {
        /* pp4: the global reads of tile t+2 issue in [mem B] of tile t; the
         * staged registers are loop-carried with the prologue's coordinates. */
        rocke_value_t* st0[CONV_UK_MAX_STATE];
        rocke_value_t* st1[CONV_UK_MAX_STATE];
        for(int i = 0; i < p->n_st; ++i)
            st0[i] = it1[n_acc + i].init;
        conv_kl_adv(u, st0, p->n_st, st1);
        conv_part(ctx, p->psf_ahead ? -1 : p->sf_c);
        conv_kl_fetch(ctx, u, rocke_b_add(b, lo, ctx->c_block_k), st1, p->staged0);
        conv_part(ctx, -1);
        p->n_a = p->staged0->a.count;
        p->n_rg = p->staged0->a.count + p->staged0->b.count;
        n_it = n_acc;
        for(int i = 0; i < p->n_st; ++i)
        {
            snprintf(nbuf, sizeof(nbuf), "%s_ps%d", tag, i);
            it[n_it].name = rocke_arena_strdup(&b->arena, nbuf);
            it[n_it++].init = st1[i];
        }
        for(int i = 0; i < p->n_rg; ++i)
        {
            snprintf(nbuf, sizeof(nbuf), "%s_pr%d", tag, i);
            it[n_it].name = rocke_arena_strdup(&b->arena, nbuf);
            it[n_it++].init
                = i < p->n_a ? p->staged0->a.vecs[i].v : p->staged0->b.vecs[i - p->n_a].v;
        }
    }
    p->c_2bk = rocke_b_const_i32(b, 2 * block_k);
    p->c_fbk = p->c_2bk;
    p->bdb = spec->pp_b_double;
    p->half_kk = ctx->k_atoms / 2;

    rocke_value_t* iv[CONV_PP_MAX_IT];
    rocke_value_t* y1[CONV_PP_MAX_IT];
    rocke_value_t* y2[CONV_PP_MAX_IT];
    char ivn[32];
    snprintf(ivn, sizeof(ivn), "%s_k", tag);
    if(!p->bdb)
    {
        rocke_for_t f = rocke_b_scf_for_iter(
            b, lo, hi, ctx->c_block_k, it, n_it, rocke_arena_strdup(&b->arena, ivn), false, true);
        for(int i = 0; i < f.num_iter_vars; ++i)
            iv[i] = f.iter_vars[i];
        rocke_b_region_enter(b, f.body);
        const int n = conv_pp_body(p, f.iv, iv, ctx->B_smem, ctx->B_smem, y1);
        rocke_b_scf_yield(b, y1, n);
        rocke_b_region_leave(b);
        rocke_b_sync_half_block(b, lo_half);
        for(int i = 0; i < n_acc; ++i)
            out_accs[i] = f.op->results[i];
        return;
    }
    /* pp_b_double: B alternates between two buffers, so the loop runs whole
     * pairs of tiles and an odd last tile in a 0/1-trip tail. */
    rocke_value_t* span = rocke_b_sub(b, hi, lo);
    rocke_value_t* span_r = rocke_b_add(b, span, rocke_b_const_i32(b, block_k - 1));
    rocke_value_t* n_tiles = rocke_b_div(b, span_r, ctx->c_block_k);
    rocke_value_t* n_pairs = rocke_b_div(b, n_tiles, rocke_b_const_i32(b, 2));
    rocke_value_t* pair_hi = rocke_b_add(b, lo, rocke_b_mul(b, n_pairs, p->c_2bk));
    rocke_value_t* tail_hi = rocke_b_add(b, lo, rocke_b_mul(b, n_tiles, ctx->c_block_k));
    rocke_for_t f = rocke_b_scf_for_iter(
        b, lo, pair_hi, p->c_2bk, it, n_it, rocke_arena_strdup(&b->arena, ivn), false, true);
    for(int i = 0; i < f.num_iter_vars; ++i)
        iv[i] = f.iter_vars[i];
    rocke_b_region_enter(b, f.body);
    {
        conv_pp_body(p, f.iv, iv, ctx->B_smem, ctx->B_smem2, y1);
        const int n = conv_pp_body(
            p, rocke_b_add(b, f.iv, ctx->c_block_k), y1, ctx->B_smem2, ctx->B_smem, y2);
        rocke_b_scf_yield(b, y2, n);
    }
    rocke_b_region_leave(b);
    rocke_iter_arg_t it_t[CONV_PP_MAX_IT];
    for(int i = 0; i < n_it; ++i)
    {
        snprintf(nbuf, sizeof(nbuf), "%s_t", it[i].name);
        it_t[i].name = rocke_arena_strdup(&b->arena, nbuf);
        it_t[i].init = f.op->results[i];
    }
    snprintf(ivn, sizeof(ivn), "%s_kt", tag);
    rocke_for_t t = rocke_b_scf_for_iter(b,
                                         pair_hi,
                                         tail_hi,
                                         ctx->c_block_k,
                                         it_t,
                                         n_it,
                                         rocke_arena_strdup(&b->arena, ivn),
                                         false,
                                         true);
    for(int i = 0; i < t.num_iter_vars; ++i)
        iv[i] = t.iter_vars[i];
    rocke_b_region_enter(b, t.body);
    {
        const int n = conv_pp_body(p, t.iv, iv, ctx->B_smem, ctx->B_smem2, y1);
        rocke_b_scf_yield(b, y1, n);
    }
    rocke_b_region_leave(b);
    rocke_b_sync_half_block(b, lo_half);
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = t.op->results[i];
}

/* Python run_loop(lo, hi, accs_in, state_in, fetch_tile, advance, tag). */
static void conv_run_loop(rocke_conv_build_ctx_t* ctx,
                          rocke_value_t* lo,
                          rocke_value_t* hi,
                          rocke_value_t* const* accs_in,
                          const char* const* state_names,
                          rocke_value_t* const* state_in,
                          int n_state,
                          conv_uk_t* u,
                          const char* tag,
                          rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const int n_acc = ctx->num_accs;
    const int block_k = ctx->block_k;
    conv_rpf_staged_t* st
        = (conv_rpf_staged_t*)rocke_arena_alloc(&b->arena, sizeof(conv_rpf_staged_t));
    if(st == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "conv run_loop: arena alloc failed");
        return;
    }
    rocke_iter_arg_t it[ROCKE_CONV_MAX_ACCS + CONV_UK_MAX_STATE];
    rocke_value_t* iv[ROCKE_CONV_MAX_ACCS + CONV_UK_MAX_STATE];
    rocke_value_t* acc_a[ROCKE_CONV_MAX_ACCS + CONV_UK_MAX_STATE];
    rocke_value_t* acc_b[ROCKE_CONV_MAX_ACCS + CONV_UK_MAX_STATE];
    rocke_value_t* st1[CONV_UK_MAX_STATE];
    rocke_value_t* st2[CONV_UK_MAX_STATE];
    char nbuf[96];
    char ivn[32];

    for(int i = 0; i < n_acc; ++i)
    {
        snprintf(nbuf, sizeof(nbuf), "%s_%s", tag, ctx->acc_names[i]);
        it[i].name = rocke_arena_strdup(&b->arena, nbuf);
        it[i].init = accs_in[i];
    }
    for(int i = 0; i < n_state; ++i)
    {
        snprintf(nbuf, sizeof(nbuf), "%s_%s", tag, state_names[i]);
        it[n_acc + i].name = rocke_arena_strdup(&b->arena, nbuf);
        it[n_acc + i].init = state_in[i];
    }
    snprintf(ivn, sizeof(ivn), "%s_k", tag);

    if(!spec->reg_prefetch)
    {
        rocke_for_t f = rocke_b_scf_for_iter(b,
                                             lo,
                                             hi,
                                             ctx->c_block_k,
                                             it,
                                             n_acc + n_state,
                                             rocke_arena_strdup(&b->arena, ivn),
                                             false,
                                             true);
        for(int i = 0; i < f.num_iter_vars; ++i)
            iv[i] = f.iter_vars[i];
        rocke_b_region_enter(b, f.body);
        {
            conv_kl_fetch(ctx, u, f.iv, iv + n_acc, st);
            conv_rpf_store(ctx, st, ctx->A_smem, ctx->B_smem);
            rocke_b_sync(b);
            rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, iv, n_acc, acc_a);
            rocke_b_sync(b);
            conv_kl_adv(u, iv + n_acc, n_state, acc_a + n_acc);
            rocke_b_scf_yield(b, acc_a, n_acc + n_state);
        }
        rocke_b_region_leave(b);
        for(int i = 0; i < n_acc; ++i)
            out_accs[i] = f.op->results[i];
        return;
    }

    /* Prologue: tile `lo` into buffer 0, published. The fallback loop of a
     * fb_skip_empty kernel only runs it when its range is not empty. */
    if(spec->uk_xhalo || (spec->fb_skip_empty && strcmp(tag, "fb") == 0))
    {
        rocke_if_t if_op = rocke_b_scf_if(b, rocke_b_cmp_lt(b, lo, hi));
        rocke_b_region_enter(b, if_op.then_region);
        conv_kl_fetch(ctx, u, lo, state_in, st);
        conv_rpf_store(ctx, st, ctx->A_smem, ctx->B_smem);
        rocke_b_sync(b);
        rocke_b_region_leave(b);
    }
    else
    {
        conv_kl_fetch(ctx, u, lo, state_in, st);
        conv_rpf_store(ctx, st, ctx->A_smem, ctx->B_smem);
        rocke_b_sync(b);
    }

    if(spec->reg_prefetch == 1 && spec->rpf_pingpong && strcmp(tag, "uk") == 0)
    {
        conv_run_pingpong(ctx, lo, hi, it, n_acc + n_state, u, tag, out_accs);
        return;
    }
    if(spec->reg_prefetch == 1)
    {
        /* One buffer: read tile t+1 into registers, MFMA tile t, retire every
         * wave's ds_reads, then overwrite the buffer and publish. */
        rocke_for_t f = rocke_b_scf_for_iter(b,
                                             lo,
                                             hi,
                                             ctx->c_block_k,
                                             it,
                                             n_acc + n_state,
                                             rocke_arena_strdup(&b->arena, ivn),
                                             false,
                                             true);
        for(int i = 0; i < f.num_iter_vars; ++i)
            iv[i] = f.iter_vars[i];
        rocke_b_region_enter(b, f.body);
        {
            conv_kl_adv(u, iv + n_acc, n_state, st1);
            /* rpf_fetch_late: the next tile's reads go behind the first LDS
             * fragment reads (the MFMA phase calls back). */
            struct late_fetch
            {
                rocke_conv_build_ctx_t* ctx;
                conv_uk_t* u;
                rocke_value_t* kv;
                rocke_value_t* const* st1;
                conv_rpf_staged_t* st;
            } lf = {ctx, u, f.iv, st1, st};
            if(spec->rpf_fetch_late)
            {
                ctx->after_first_reads_user = &lf;
                ctx->after_first_reads = [](void* user) {
                    late_fetch* l = (late_fetch*)user;
                    rocke_value_t* k1 = rocke_b_add(l->ctx->b, l->kv, l->ctx->c_block_k);
                    conv_kl_fetch(l->ctx, l->u, k1, l->st1, l->st);
                };
            }
            else
                conv_kl_fetch(ctx, u, rocke_b_add(b, f.iv, ctx->c_block_k), st1, st);
            if(spec->vmem_interleave)
                ctx->vmi_pending = st->a.count + st->b.count; /* the MFMA phase places them */
            else
                rocke_b_sched_barrier(b, 0);
            ctx->k_off_capture = f.iv;
            if(spec->defer_mfma)
            {
                rocke_conv_frag_t* dfr = (rocke_conv_frag_t*)rocke_arena_alloc(
                    &b->arena, (size_t)spec->defer_mfma * sizeof(rocke_conv_frag_t));
                ctx->defer_on = true;
                ctx->defer_n = spec->defer_mfma;
                ctx->defer_frags = dfr;
                ctx->defer_cnt = 0;
                /* (An x-halo kernel's fallback loop keeps its priority.) */
                if(spec->rpf_setprio && !spec->uk_xhalo)
                    rocke_b_s_setprio(b, spec->rpf_setprio);
                rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, iv, n_acc, acc_a);
                ctx->after_first_reads = NULL;
                ctx->defer_on = false;
                rocke_b_sched_barrier(b, 0);
                if(spec->rpf_setprio && !spec->uk_xhalo)
                    rocke_b_s_setprio(b, 0);
                rocke_b_sync_lds_only(b);
                conv_rpf_store(ctx, st, ctx->A_smem, ctx->B_smem);
                for(int d = 0; d < ctx->defer_cnt; ++d)
                {
                    int flat = 0;
                    for(int mi = 0; mi < ctx->mfmas_m; ++mi)
                        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
                        {
                            acc_a[flat] = rocke_conv_emit_mfma(
                                b, ctx->atom, dfr[d].a[mi], dfr[d].b[ni], acc_a[flat]);
                            ++flat;
                        }
                }
                ctx->defer_cnt = 0;
                if(spec->defer_mfma_per_write)
                {
                    for(int w = 0; w < st->a.count + st->b.count; ++w)
                    {
                        rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_WRITE, 1, 0);
                        rocke_b_sched_group_barrier(
                            b, ROCKE_SCHED_MFMA, spec->defer_mfma_per_write, 0);
                    }
                }
                else
                {
                    /* 1 ds_write : 2 MFMAs */
                    for(int w = 0; w < 8; ++w)
                    {
                        rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_WRITE, 1, 0);
                        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 2, 0);
                    }
                }
                rocke_b_sched_barrier(b, 0);
                rocke_b_sync_lds_only(b);
                for(int i = 0; i < n_state; ++i)
                    acc_a[n_acc + i] = st1[i];
                rocke_b_scf_yield(b, acc_a, n_acc + n_state);
                rocke_b_region_leave(b);
                for(int i = 0; i < n_acc; ++i)
                    out_accs[i] = f.op->results[i];
                return;
            }
            if(spec->rpf_setprio)
                rocke_b_s_setprio(b, spec->rpf_setprio);
            rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, iv, n_acc, acc_a);
            ctx->after_first_reads = NULL;
            rocke_b_sched_barrier(b, 0);
            if(spec->rpf_setprio)
                rocke_b_s_setprio(b, 0);
            if(spec->rpf_lds_barrier)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
            conv_rpf_store(ctx, st, ctx->A_smem, ctx->B_smem);
            if(spec->rpf_lds_barrier)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
            for(int i = 0; i < n_state; ++i)
                acc_a[n_acc + i] = st1[i];
            rocke_b_scf_yield(b, acc_a, n_acc + n_state);
        }
        rocke_b_region_leave(b);
        for(int i = 0; i < n_acc; ++i)
            out_accs[i] = f.op->results[i];
        return;
    }

    /* reg_prefetch == 2: whole pairs of tiles in the unrolled loop; an odd last
     * tile was prefetched into buffer 0 by the last pair and runs in a 0/1-trip
     * tail loop. */
    rocke_value_t* c_2bk = rocke_b_const_i32(b, 2 * block_k);
    rocke_value_t* span = rocke_b_sub(b, hi, lo);
    rocke_value_t* span_r = rocke_b_add(b, span, rocke_b_const_i32(b, block_k - 1));
    rocke_value_t* n_tiles = rocke_b_div(b, span_r, ctx->c_block_k);
    rocke_value_t* n_pairs = rocke_b_div(b, n_tiles, rocke_b_const_i32(b, 2));
    rocke_value_t* pair_hi = rocke_b_add(b, lo, rocke_b_mul(b, n_pairs, c_2bk));
    rocke_value_t* tail_hi = rocke_b_add(b, lo, rocke_b_mul(b, n_tiles, ctx->c_block_k));

    rocke_for_t f = rocke_b_scf_for_iter(b,
                                         lo,
                                         pair_hi,
                                         c_2bk,
                                         it,
                                         n_acc + n_state,
                                         rocke_arena_strdup(&b->arena, ivn),
                                         false,
                                         true);
    for(int i = 0; i < f.num_iter_vars; ++i)
        iv[i] = f.iter_vars[i];
    rocke_b_region_enter(b, f.body);
    {
        rocke_value_t* k1 = rocke_b_add(b, f.iv, ctx->c_block_k);
        conv_kl_adv(u, iv + n_acc, n_state, st1);
        if(spec->rpf_setprio)
            rocke_b_s_setprio(b, spec->rpf_setprio);
        conv_kl_fetch(ctx, u, k1, st1, st);
        if(spec->vmem_interleave)
            ctx->vmi_pending = st->a.count + st->b.count;
        else
            rocke_b_sched_barrier(b, 0);
        ctx->k_off_capture = f.iv;
        rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, iv, n_acc, acc_a);
        rocke_b_sched_barrier(b, 0);
        if(spec->rpf_setprio)
            rocke_b_s_setprio(b, 0);
        conv_rpf_store(ctx, st, ctx->A_smem2, ctx->B_smem2);
        rocke_b_sync(b);
        rocke_value_t* k2 = rocke_b_add(b, f.iv, c_2bk);
        conv_kl_adv(u, st1, n_state, st2);
        if(spec->rpf_setprio)
            rocke_b_s_setprio(b, spec->rpf_setprio);
        conv_kl_fetch(ctx, u, k2, st2, st);
        if(spec->vmem_interleave)
            ctx->vmi_pending = st->a.count + st->b.count;
        else
            rocke_b_sched_barrier(b, 0);
        ctx->k_off_capture = k1;
        rocke_conv_emit_mfma_phase(ctx, ctx->A_smem2, ctx->B_smem2, acc_a, n_acc, acc_b);
        rocke_b_sched_barrier(b, 0);
        if(spec->rpf_setprio)
            rocke_b_s_setprio(b, 0);
        conv_rpf_store(ctx, st, ctx->A_smem, ctx->B_smem);
        rocke_b_sync(b);
        for(int i = 0; i < n_state; ++i)
            acc_b[n_acc + i] = st2[i];
        rocke_b_scf_yield(b, acc_b, n_acc + n_state);
    }
    rocke_b_region_leave(b);

    rocke_iter_arg_t tit[ROCKE_CONV_MAX_ACCS];
    for(int i = 0; i < n_acc; ++i)
    {
        snprintf(nbuf, sizeof(nbuf), "%s_t_%s", tag, ctx->acc_names[i]);
        tit[i].name = rocke_arena_strdup(&b->arena, nbuf);
        tit[i].init = f.op->results[i];
    }
    snprintf(ivn, sizeof(ivn), "%s_kt", tag);
    rocke_for_t t = rocke_b_scf_for_iter(b,
                                         pair_hi,
                                         tail_hi,
                                         ctx->c_block_k,
                                         tit,
                                         n_acc,
                                         rocke_arena_strdup(&b->arena, ivn),
                                         false,
                                         true);
    for(int i = 0; i < t.num_iter_vars; ++i)
        iv[i] = t.iter_vars[i];
    rocke_b_region_enter(b, t.body);
    {
        ctx->k_off_capture = t.iv;
        rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, iv, n_acc, acc_a);
        rocke_b_sync(b);
        rocke_b_scf_yield(b, acc_a, n_acc);
    }
    rocke_b_region_leave(b);
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = t.op->results[i];
}

/* ===================================================================== *
 * The uk_xhalo loop (Python _emit_xhalo_loop) with the wave-private DMA
 * staging of _emit_xhalo_dma (fxh_dma >= 1000; xh_stride_m, lds_kpack_read,
 * lds_xor, tile_k 32). Returns xh_ok; the accumulators after it in out.
 * ===================================================================== */
#define CONV_XH_TAPS 3

typedef struct
{
    rocke_conv_build_ctx_t* ctx;
    conv_uk_t* u;
    rocke_value_t* m0;
} conv_xh_t;

/* lin_of(m): input pixel index of output pixel m at tap 0 (pad 0, sW 1). */
static rocke_value_t* conv_xh_lin_of(rocke_conv_build_ctx_t* ctx, rocke_value_t* m_val)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t *q_wo, *wo, *q_ho, *ho, *n_, *do_;
    rocke_conv_magic_divmod(
        b, m_val, ctx->p_magic_m_Wo_mult, ctx->p_magic_m_Wo_shift, ctx->p_Wo, false, &q_wo, &wo);
    rocke_conv_magic_divmod(
        b, q_wo, ctx->p_magic_m_Ho_mult, ctx->p_magic_m_Ho_shift, ctx->p_Ho, false, &q_ho, &ho);
    rocke_conv_magic_divmod(
        b, q_ho, ctx->p_magic_m_Do_mult, ctx->p_magic_m_Do_shift, ctx->p_Do, false, &n_, &do_);
    rocke_value_t* nd = rocke_b_mul(b, n_, ctx->p_Di);
    rocke_value_t* ds = rocke_b_mul(b, do_, ctx->p_sD);
    rocke_value_t* d_ = rocke_b_add(b, nd, ds);
    rocke_value_t* dh = rocke_b_mul(b, d_, ctx->p_Hi);
    rocke_value_t* hs = rocke_b_mul(b, ho, ctx->p_sH);
    rocke_value_t* h_ = rocke_b_add(b, dh, hs);
    return rocke_b_add(b, rocke_b_mul(b, h_, ctx->p_Wi), wo);
}

/* xh_uniforms(st): (a_off, b_off, t_ok) of the (c, y, z) step. */
static void conv_xh_uniforms(conv_xh_t* x,
                             rocke_value_t* const* st,
                             rocke_value_t** a_off,
                             rocke_value_t** b_off,
                             rocke_value_t** t_ok)
{
    rocke_conv_build_ctx_t* ctx = x->ctx;
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* ay = rocke_b_mul(b, st[1], x->u->a_ystep);
    rocke_value_t* az = rocke_b_mul(b, st[2], x->u->a_zstep);
    *a_off = rocke_b_add(b, rocke_b_add(b, ay, az), st[0]);
    rocke_value_t* by = rocke_b_mul(b, st[1], ctx->p_B_stride_y);
    rocke_value_t* bz = rocke_b_mul(b, st[2], ctx->p_B_stride_z);
    *b_off = rocke_b_add(b, rocke_b_add(b, by, bz), st[0]);
    *t_ok = rocke_b_cmp_lt(b, st[2], ctx->p_Z);
}

/* xh_adv(st): the next (c, y, z) step. */
static void conv_xh_adv(conv_xh_t* x, rocke_value_t* const* st, rocke_value_t** out)
{
    rocke_conv_build_ctx_t* ctx = x->ctx;
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* c1 = ctx->uk_c1;
    rocke_value_t *c_t = st[0], *y_t = st[1], *z_t = st[2];
    if(spec->uk_tap_fast)
    {
        const int g = spec->uk_chunk_group;
        rocke_value_t *c_gspan, *nc, *ny;
        if(g)
        {
            c_gspan = rocke_b_const_i32(b, g * ctx->block_k);
            rocke_value_t* c_gmask = rocke_b_const_i32(b, g * ctx->block_k - 1);
            nc = rocke_b_add(b, c_t, ctx->c_block_k);
            rocke_value_t* e1 = rocke_b_cmp_eq(b, rocke_b_land(b, nc, c_gmask), c0);
            rocke_value_t* w_c = rocke_b_lor(b, e1, rocke_b_cmp_ge(b, nc, ctx->p_cpg));
            nc = rocke_b_select(b, w_c, rocke_b_sub(b, c_t, rocke_b_land(b, c_t, c_gmask)), nc);
            ny = rocke_b_add(b, y_t, rocke_b_select(b, w_c, c1, c0));
        }
        else
        {
            c_gspan = ctx->c_block_k;
            nc = c_t;
            ny = rocke_b_add(b, y_t, c1);
        }
        rocke_value_t* w_y = rocke_b_cmp_ge(b, ny, ctx->p_Y);
        ny = rocke_b_select(b, w_y, c0, ny);
        rocke_value_t* nz = rocke_b_add(b, z_t, rocke_b_select(b, w_y, c1, c0));
        rocke_value_t* w_z = rocke_b_cmp_ge(b, nz, ctx->p_Z);
        nz = rocke_b_select(b, w_z, c0, nz);
        nc = rocke_b_add(b, nc, rocke_b_select(b, w_z, c_gspan, c0));
        nz = rocke_b_select(b, rocke_b_cmp_ge(b, nc, ctx->p_cpg), ctx->p_Z, nz);
        out[0] = nc;
        out[1] = ny;
        out[2] = nz;
        return;
    }
    rocke_value_t* nc = rocke_b_add(b, c_t, ctx->c_block_k);
    rocke_value_t* w_c = rocke_b_cmp_ge(b, nc, ctx->p_cpg);
    nc = rocke_b_select(b, w_c, c0, nc);
    rocke_value_t* ny = rocke_b_add(b, y_t, rocke_b_select(b, w_c, c1, c0));
    rocke_value_t* w_y = rocke_b_cmp_ge(b, ny, ctx->p_Y);
    ny = rocke_b_select(b, w_y, c0, ny);
    rocke_value_t* nz = rocke_b_add(b, z_t, rocke_b_select(b, w_y, c1, c0));
    out[0] = nc;
    out[1] = ny;
    out[2] = nz;
}

/* xh_pad (Python _emit_xhalo_dma _xp helpers): the padded-input masking of
 * the wave-private DMA loop. */
typedef struct
{
    bool on;
    rocke_value_t* a_ptr_b; /* A moved down by the pad offset */
    rocke_value_t* a_nb; /* A_bytes + the pad offset */
} conv_xpd_t;

/* bit t set <=> 0 <= org + t < extent (t < 31; dilation 1) */
static rocke_value_t*
    conv_xpd_tap_mask(rocke_conv_build_ctx_t* ctx, rocke_value_t* org, rocke_value_t* extent)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* c1 = ctx->uk_c1;
    rocke_value_t* c31 = rocke_b_const_i32(b, 31);
    rocke_value_t* l0 = rocke_b_sub(b, c0, org);
    rocke_value_t* l1 = rocke_b_smax(b, l0, c0);
    rocke_value_t* lo = rocke_b_smin(b, l1, c31);
    rocke_value_t* h0 = rocke_b_sub(b, extent, org);
    rocke_value_t* h1 = rocke_b_smax(b, h0, c0);
    rocke_value_t* hi = rocke_b_smin(b, h1, c31);
    rocke_value_t* sh = rocke_b_shl(b, c1, hi);
    rocke_value_t* mh = rocke_b_sub(b, sh, c1);
    rocke_value_t* sl = rocke_b_shl(b, c1, lo);
    rocke_value_t* ml = rocke_b_sub(b, sl, c1);
    rocke_value_t* neg = rocke_b_const_i32(b, -1);
    rocke_value_t* nl = rocke_b_xor(b, ml, neg);
    return rocke_b_to_sgpr_u32(b, rocke_b_land(b, mh, nl));
}

/* (z tap mask, y tap mask, input w) of the input row of output pixel m_val. */
static void conv_xpd_pcoords(rocke_conv_build_ctx_t* ctx, rocke_value_t* m_val, rocke_value_t** pc)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t *q_wo, *wo, *q_ho, *ho, *n_, *do_;
    rocke_conv_magic_divmod(
        b, m_val, ctx->p_magic_m_Wo_mult, ctx->p_magic_m_Wo_shift, ctx->p_Wo, false, &q_wo, &wo);
    rocke_conv_magic_divmod(
        b, q_wo, ctx->p_magic_m_Ho_mult, ctx->p_magic_m_Ho_shift, ctx->p_Ho, false, &q_ho, &ho);
    rocke_conv_magic_divmod(
        b, q_ho, ctx->p_magic_m_Do_mult, ctx->p_magic_m_Do_shift, ctx->p_Do, false, &n_, &do_);
    rocke_value_t* ds = rocke_b_mul(b, do_, ctx->p_sD);
    rocke_value_t* dd = rocke_b_sub(b, ds, ctx->p_pD);
    pc[0] = conv_xpd_tap_mask(ctx, dd, ctx->p_Di);
    rocke_value_t* hs = rocke_b_mul(b, ho, ctx->p_sH);
    rocke_value_t* hh = rocke_b_sub(b, hs, ctx->p_pH);
    pc[1] = conv_xpd_tap_mask(ctx, hh, ctx->p_Hi);
    pc[2] = rocke_b_to_sgpr_u32(b, rocke_b_sub(b, wo, ctx->p_pW));
}

/* A rsrc of a DMA list whose rows have tap masks (zm, ym) at step st:
 * zero-size when that tap's input plane / row is outside. */
static rocke_value_t* conv_xpd_rsrc(rocke_conv_build_ctx_t* ctx,
                                    const conv_xpd_t* xp,
                                    rocke_value_t* zm,
                                    rocke_value_t* ym,
                                    rocke_value_t* const* st)
{
    if(!xp->on)
        return ctx->a_rsrc;
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* zs = rocke_b_lshr(b, zm, st[2]);
    rocke_value_t* ys = rocke_b_lshr(b, ym, st[1]);
    rocke_value_t* a = rocke_b_land(b, zs, ys);
    rocke_value_t* a1 = rocke_b_land(b, a, ctx->uk_c1);
    rocke_value_t* ok = rocke_b_cmp_ne(b, a1, c0);
    rocke_value_t* nb = rocke_b_to_sgpr_u32(b, rocke_b_select(b, ok, xp->a_nb, c0));
    return rocke_b_buffer_rsrc(b, xp->a_ptr_b, nb);
}

/* lane of range row rel: input column w + rel outside -> out of range */
static rocke_value_t* conv_xpd_wmask(rocke_conv_build_ctx_t* ctx,
                                     rocke_value_t* w,
                                     rocke_value_t* rel,
                                     rocke_value_t* voff,
                                     rocke_value_t* c_oob)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* wr = rocke_b_add(b, w, rel);
    return rocke_b_select(b, rocke_b_cmp_ult(b, wr, ctx->p_Wi), voff, c_oob);
}

static void conv_xh_hint_interleave(
    rocke_ir_builder_t* b, int n_first, int n_other, int other, int n_mf, int per)
{
    if(n_first)
        rocke_b_sched_group_barrier(b, other, n_first, 0);
    int n_o = n_other;
    while(n_mf > 0)
    {
        const int take = per < n_mf ? per : n_mf;
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, take, 0);
        n_mf -= take;
        if(n_o)
        {
            rocke_b_sched_group_barrier(b, other, 1, 0);
            n_o -= 1;
        }
    }
    if(n_o)
        rocke_b_sched_group_barrier(b, other, n_o, 0);
}

typedef struct conv_dma_s
{
    rocke_value_t* lds;
    rocke_value_t* voff;
} conv_dma_t;

typedef struct
{
    conv_xh_t* x;
    rocke_value_t* xh_ok;
    rocke_value_t* xh_hi;
    rocke_value_t* wave_;
    rocke_value_t* l16;
    rocke_value_t* lch;
    rocke_value_t* ldw;
    rocke_value_t* c_oob;
    rocke_value_t* c2;
    rocke_value_t* A_addr;
    rocke_value_t* B_addr;
    struct conv_dma_s* b_dma;
    int n_bdma;
    bool bcheap;
    rocke_value_t** base_rows;
    conv_xpd_t xp;
} conv_xy_t;

/* An x-halo DMA destination: a fixed LDS address, or (kind, constant segment,
 * validity) resolved per issue() call (Python's ("own"|"w"|"u", cst, inval)). */
typedef struct
{
    char kind; /* 0: fixed lds, 'o' own, 'w', 'u' */
    int cst;
    rocke_value_t* inval;
    rocke_value_t* lds;
    rocke_value_t* voff;
} conv_xdst_t;

static rocke_value_t*
    conv_xh_yring(conv_xy_t* y, rocke_value_t* const* accs_in, rocke_value_t** out_accs);

static rocke_value_t* conv_xh_emit(conv_xh_t* x,
                                   rocke_value_t* uk_ok_,
                                   rocke_value_t* const* accs_in,
                                   rocke_value_t** out_accs)
{
    rocke_conv_build_ctx_t* ctx = x->ctx;
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    conv_uk_t* u = x->u;
    const int XT = CONV_XH_TAPS;
    const int block_m = ctx->block_m, block_n = ctx->block_n, block_k = ctx->block_k;
    const int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n, k_atoms = ctx->k_atoms;
    const int n_acc = ctx->num_accs;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* c1 = ctx->uk_c1;
    const int R = block_m + spec->uk_xhalo;
    rocke_value_t* c_R = rocke_b_const_i32(b, R);
    rocke_value_t* c_XT = rocke_b_const_i32(b, XT);
    rocke_value_t* m0 = ctx->block_m_off_v;
    x->m0 = m0;
    rocke_value_t* s_n = ctx->p_A_stride_n;
    rocke_value_t* s_hi = ctx->p_A_stride_hi;
    rocke_value_t* s_wi = ctx->p_A_stride_wi;
    rocke_value_t* s_di = ctx->p_A_stride_di;

    rocke_value_t* c_bm = rocke_b_const_i32(b, block_m);
    rocke_value_t* m_end = rocke_b_add(b, m0, c_bm);
    rocke_value_t* m_lt = rocke_b_cmp_lt(b, m_end, ctx->p_M);
    rocke_value_t* m_last = rocke_b_sub(b, rocke_b_select(b, m_lt, m_end, ctx->p_M), c1);
    rocke_value_t* lin0 = conv_xh_lin_of(ctx, m0);
    rocke_value_t* lin_l = conv_xh_lin_of(ctx, m_last);
    rocke_value_t* span = rocke_b_add(b, rocke_b_sub(b, lin_l, lin0), c_XT);
    rocke_value_t* conds[24];
    int nc = 0;
    conds[nc++] = uk_ok_;
    conds[nc++] = rocke_b_cmp_le(b, span, c_R);
    if(!spec->xh_pad)
    {
        /* xh_pad: the DMA loop masks the padding itself */
        conds[nc++] = rocke_b_cmp_eq(b, ctx->p_pD, c0);
        conds[nc++] = rocke_b_cmp_eq(b, ctx->p_pH, c0);
        conds[nc++] = rocke_b_cmp_eq(b, ctx->p_pW, c0);
    }
    conds[nc++] = rocke_b_cmp_eq(b, ctx->p_sW, c1);
    conds[nc++] = rocke_b_cmp_eq(b, ctx->p_dW, c1);
    conds[nc++] = rocke_b_cmp_eq(b, ctx->p_X, c_XT);
    conds[nc++] = rocke_b_cmp_eq(b, s_hi, rocke_b_mul(b, ctx->p_Wi, s_wi));
    conds[nc++] = rocke_b_cmp_eq(b, s_di, rocke_b_mul(b, ctx->p_Hi, s_hi));
    conds[nc++] = rocke_b_cmp_eq(b, s_n, rocke_b_mul(b, ctx->p_Di, s_di));
    if(spec->xh_pad)
    {
        /* the DMA loop's tap masks: dilation 1 on D / H, < 32 taps */
        rocke_value_t* c32 = rocke_b_const_i32(b, 32);
        conds[nc++] = rocke_b_cmp_eq(b, ctx->p_dH, c1);
        conds[nc++] = rocke_b_cmp_eq(b, ctx->p_dD, c1);
        conds[nc++] = rocke_b_cmp_lt(b, ctx->p_Y, c32);
        conds[nc++] = rocke_b_cmp_lt(b, ctx->p_Z, c32);
    }
    if(spec->xh_stride_m)
    {
        rocke_value_t* md = rocke_b_mod(b, ctx->p_Wo, rocke_b_const_i32(b, mfmas_m));
        conds[nc++] = rocke_b_cmp_eq(b, md, c0);
    }
    if(spec->fxh_dma >= 1000)
    {
        rocke_value_t* md
            = rocke_b_mod(b, ctx->p_Wo, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
        conds[nc++] = rocke_b_cmp_eq(b, md, c0);
    }
    if(spec->xh_yring)
    {
        /* whole output rows per tile, whole tiles per plane, Y == 3 */
        rocke_value_t* wr_ = rocke_b_mul(b, ctx->p_Wo, rocke_b_const_i32(b, spec->xh_yring % 10));
        conds[nc++] = rocke_b_cmp_eq(b, wr_, rocke_b_const_i32(b, block_m));
        rocke_value_t* hw = rocke_b_mul(b, ctx->p_Ho, ctx->p_Wo);
        rocke_value_t* hm = rocke_b_mod(b, hw, rocke_b_const_i32(b, block_m));
        conds[nc++] = rocke_b_cmp_eq(b, hm, c0);
        conds[nc++] = rocke_b_cmp_eq(b, ctx->p_Y, c_XT);
        rocke_value_t* cm = rocke_b_mod(b, ctx->p_cpg, rocke_b_const_i32(b, block_k));
        conds[nc++] = rocke_b_cmp_eq(b, cm, c0);
    }
    rocke_value_t* xh_ok = conds[0];
    for(int i = 1; i < nc; ++i)
        xh_ok = rocke_b_land(b, xh_ok, conds[i]);
    ctx->xsm_ok = xh_ok;
    /* (chunk, z, y) tiles: Z*Y*cpg / tile_k of them. */
    rocke_value_t* zy = rocke_b_mul(b, ctx->p_Z, ctx->p_Y);
    rocke_value_t* xh_hi = rocke_b_select(b, xh_ok, rocke_b_mul(b, zy, ctx->p_cpg), c0);
    rocke_value_t* a_base0 = rocke_b_mul(b, lin0, s_wi);
    if(u->a_gbase != NULL)
        a_base0 = rocke_b_add(b, a_base0, u->a_gbase);
    rocke_value_t* c_oob_a = rocke_b_const_i32(b, 0x7FFFFFFF);
    rocke_value_t* c_eb_a
        = rocke_b_const_i32(b, (spec->dtype_a && strcmp(spec->dtype_a, "fp32") == 0) ? 4 : 2);
    /* chunks(): the plain loop's (row, col, live) slots of A and B. */
    enum
    {
        XH_MAX_SLOTS = 64
    };
    rocke_value_t* sl_row[2][XH_MAX_SLOTS];
    rocke_value_t* sl_col[2][XH_MAX_SLOTS];
    rocke_value_t* sl_live[2][XH_MAX_SLOTS];
    int sl_n[2] = {0, 0};

    /* chunks(rows, vec, cap_row): the plain x-halo loader slots (emitted by
     * Python ahead of the loop dispatch even though the DMA loop never reads
     * them). */
    const int threads = ctx->threads;
    const int lva = ctx->a_sync_loader.load_vec, lvb = ctx->b_sync_loader.load_vec;
    const int rows_l[2] = {R, XT * block_n};
    const int vecs_l[2] = {lva, lvb};
    for(int w = 0; w < 2; ++w)
    {
        const int rows = rows_l[w], vec = vecs_l[w], cap_row = rows_l[w];
        const int cpv = block_k / vec;
        const int total = rows * cpv;
        const int n_e = (total + threads - 1) / threads;
        for(int e = 0; e < n_e; ++e)
        {
            rocke_value_t* v = rocke_b_add(b, rocke_b_const_i32(b, e * threads), ctx->tid);
            const bool surplus = (e + 1) * threads > total;
            rocke_value_t* row = rocke_b_div(b, v, rocke_b_const_i32(b, cpv));
            if(spec->xh_stride_m && cap_row == R && cpv == 4)
            {
                rocke_value_t* hs = rocke_b_lshr(b, row, rocke_b_const_i32(b, 2));
                row = rocke_b_xor(b, row, rocke_b_land(b, hs, c1));
            }
            rocke_value_t* cm = rocke_b_mod(b, v, rocke_b_const_i32(b, cpv));
            rocke_value_t* col = rocke_b_mul(b, cm, rocke_b_const_i32(b, vec));
            rocke_value_t* live = NULL;
            if(surplus)
            {
                live = rocke_b_cmp_lt(b, v, rocke_b_const_i32(b, total));
                row = rocke_b_select(b, live, row, rocke_b_const_i32(b, cap_row));
            }
            if(sl_n[w] < XH_MAX_SLOTS)
            {
                sl_row[w][sl_n[w]] = row;
                sl_col[w][sl_n[w]] = col;
                sl_live[w][sl_n[w]] = live;
                sl_n[w]++;
            }
        }
    }

    /* The MFMA A rows: lin(m0 + row) - lin0, then + x per tap. */
    rocke_lane_decode_t dec = rocke_decode_mfma_lanes(b, ctx->atom, ctx->lane);
    rocke_value_t* wm_off = rocke_warp_grid_warp_m_off(b, &ctx->grid);
    const int WR = rocke_conv_fxd_wave_rows(spec);
    rocke_value_t* base_rows[ROCKE_CONV_MAX_ACCS];
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        rocke_value_t* a_row;
        if(spec->xh_stride_m)
        {
            rocke_value_t* mm = rocke_b_mul(b, dec.m_in_atom, rocke_b_const_i32(b, mfmas_m));
            a_row = rocke_b_add(b, wm_off, rocke_b_add(b, mm, rocke_b_const_i32(b, mi)));
        }
        else
        {
            rocke_value_t* mc = rocke_b_const_i32(b, mi * spec->warp_tile_m);
            a_row = rocke_b_add(b, wm_off, rocke_b_add(b, mc, dec.m_in_atom));
        }
        if(spec->fxh_dma >= 1000)
        {
            /* fxh_dma >= 1000: the wave's private range. */
            rocke_value_t* wr = rocke_b_mul(b, ctx->grid.warp_m_idx, rocke_b_const_i32(b, WR));
            rocke_value_t* la = conv_xh_lin_of(ctx, rocke_b_add(b, m0, a_row));
            rocke_value_t* lw = conv_xh_lin_of(ctx, rocke_b_add(b, m0, wm_off));
            base_rows[mi] = rocke_b_add(b, wr, rocke_b_sub(b, la, lw));
        }
        else
        {
            rocke_value_t* la = conv_xh_lin_of(ctx, rocke_b_add(b, m0, a_row));
            base_rows[mi] = rocke_b_sub(b, la, lin0);
        }
    }

    if(spec->fxh_dma == 0)
    {
        /* ---- the plain x-halo loop: R input rows and the X taps' B tiles
         * staged through registers (one buffer). ---- */
        const rocke_type_t* dt_a
            = (spec->dtype_a && strcmp(spec->dtype_a, "bf16") == 0)   ? rocke_bf16()
              : (spec->dtype_a && strcmp(spec->dtype_a, "fp32") == 0) ? rocke_f32()
                                                                      : rocke_f16();
        const rocke_type_t* dt_b
            = (spec->dtype_b && strcmp(spec->dtype_b, "bf16") == 0)   ? rocke_bf16()
              : (spec->dtype_b && strcmp(spec->dtype_b, "fp32") == 0) ? rocke_f32()
                                                                      : rocke_f16();
        struct xh_st_t
        {
            rocke_value_t* va[XH_MAX_SLOTS];
            rocke_value_t* vb[XH_MAX_SLOTS];
        };
        auto xh_fetch = [&](rocke_value_t* const* st_, xh_st_t* out) {
            rocke_value_t *a_off, *b_off, *t_ok;
            conv_xh_uniforms(x, st_, &a_off, &b_off, &t_ok);
            rocke_value_t* a_base = rocke_b_add(b, a_base0, a_off);
            for(int i = 0; i < sl_n[0]; ++i)
            {
                rocke_value_t* rw = rocke_b_mul(b, sl_row[0][i], s_wi);
                rocke_value_t* off = rocke_b_add(b, rocke_b_add(b, a_base, rw), sl_col[0][i]);
                rocke_value_t* ok = sl_live[0][i] ? rocke_b_land(b, t_ok, sl_live[0][i]) : t_ok;
                rocke_value_t* safe = rocke_b_select(b, ok, rocke_b_mul(b, off, c_eb_a), c_oob_a);
                out->va[i] = rocke_b_buffer_load_vN(b, ctx->a_rsrc, safe, c0, dt_a, lva);
            }
            rocke_value_t* c_tn = rocke_b_const_i32(b, block_n);
            for(int i = 0; i < sl_n[1]; ++i)
            {
                rocke_value_t* row = sl_row[1][i];
                rocke_value_t* x_ = rocke_b_div(b, row, c_tn);
                rocke_value_t* rm = rocke_b_mod(b, row, c_tn);
                rocke_value_t* k_out
                    = rocke_b_add(b, ctx->block_n_off_v, rocke_conv_eqx_chan(ctx, rm));
                if(ctx->k_out_group_base != NULL)
                    k_out = rocke_b_add(b, ctx->k_out_group_base, k_out);
                rocke_value_t* o1 = rocke_b_mul(b, k_out, ctx->p_B_stride_k);
                rocke_value_t* o2 = rocke_b_mul(b, x_, ctx->p_B_stride_x);
                rocke_value_t* o12 = rocke_b_add(b, o1, o2);
                rocke_value_t* o3 = rocke_b_add(b, b_off, sl_col[1][i]);
                rocke_value_t* off = rocke_b_add(b, o12, o3);
                rocke_value_t* ok = sl_live[1][i] ? rocke_b_land(b, t_ok, sl_live[1][i]) : t_ok;
                rocke_value_t* safe = rocke_b_select(b, ok, rocke_b_mul(b, off, c_eb_a), c_oob_a);
                out->vb[i] = rocke_b_buffer_load_vN(b, ctx->b_rsrc, safe, c0, dt_b, lvb);
            }
        };
        auto xh_store = [&](const xh_st_t* stg) {
            for(int w = 0; w < 2; ++w)
            {
                rocke_value_t* dst = w == 0 ? ctx->A_smem : ctx->B_smem;
                const int vec = w == 0 ? lva : lvb;
                for(int i = 0; i < sl_n[w]; ++i)
                {
                    rocke_value_t* idx[2];
                    if(spec->xh_stride_m && w == 0)
                        rocke_conv_xa(ctx, sl_row[w][i], sl_col[w][i], &idx[0], &idx[1]);
                    else
                    {
                        rocke_value_t* sc = rocke_conv_lds_swz(ctx, sl_row[w][i], sl_col[w][i]);
                        idx[0] = sl_row[w][i];
                        idx[1] = sc;
                    }
                    rocke_b_smem_store_vN(b, dst, idx, 2, w == 0 ? stg->va[i] : stg->vb[i], vec);
                }
            }
        };
        /* xh_mfma: the X taps' MFMA phases (defer_mfma: the last tap's last
         * kk steps run after the barrier). */
        rocke_conv_frag_t* dfr = NULL;
        auto xh_mfma = [&](rocke_value_t* const* accs_t, bool defer_last, rocke_value_t** out) {
            rocke_value_t* cur[ROCKE_CONV_MAX_ACCS];
            for(int i = 0; i < n_acc; ++i)
                cur[i] = accs_t[i];
            /* xh_stride_m: each tap records its A fragments (a_rec) and the
             * next reads them shifted (a_shift_prev). */
            typedef rocke_value_t* xrec_t[ROCKE_CONV_MAX_ACCS];
            xrec_t* xrecs = NULL;
            if(spec->xh_stride_m)
                xrecs
                    = (xrec_t*)rocke_arena_alloc(&b->arena, (size_t)XT * k_atoms * sizeof(xrec_t));
            for(int xx = 0; xx < XT; ++xx)
            {
                if(xrecs != NULL)
                {
                    ctx->xh_prev = xx == 0 ? NULL : xrecs + (size_t)(xx - 1) * k_atoms;
                    ctx->xh_rec = xrecs + (size_t)xx * k_atoms;
                }
                if(defer_last && xx == XT - 1)
                {
                    dfr = (rocke_conv_frag_t*)rocke_arena_alloc(
                        &b->arena, (size_t)spec->defer_mfma * sizeof(rocke_conv_frag_t));
                    ctx->defer_on = true;
                    ctx->defer_n = spec->defer_mfma;
                    ctx->defer_frags = dfr;
                    ctx->defer_cnt = 0;
                }
                rocke_value_t* rows[ROCKE_CONV_MAX_ACCS];
                for(int mi = 0; mi < mfmas_m; ++mi)
                    rows[mi] = xx == 0 ? base_rows[mi]
                                       : rocke_b_add(b, base_rows[mi], rocke_b_const_i32(b, xx));
                ctx->xh_a_rows = rows;
                ctx->xh_b_row_off = xx * block_n;
                rocke_value_t* nx[ROCKE_CONV_MAX_ACCS];
                rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, cur, n_acc, nx);
                ctx->xh_a_rows = NULL;
                ctx->xh_b_row_off = 0;
                ctx->xh_prev = NULL;
                ctx->xh_rec = NULL;
                for(int i = 0; i < n_acc; ++i)
                    cur[i] = nx[i];
            }
            ctx->defer_on = false;
            for(int i = 0; i < n_acc; ++i)
                out[i] = cur[i];
        };
        rocke_value_t* st0[3] = {c0, c0, c0};
        {
            rocke_if_t ifo = rocke_b_scf_if(b, xh_ok);
            rocke_b_region_enter(b, ifo.then_region);
            xh_st_t s0;
            xh_fetch(st0, &s0);
            xh_store(&s0);
            rocke_b_sync(b);
            rocke_b_region_leave(b);
        }
        rocke_iter_arg_t pit[ROCKE_CONV_MAX_ACCS + 3];
        char pnb[64];
        for(int i = 0; i < n_acc; ++i)
        {
            snprintf(pnb, sizeof(pnb), "xh_%s", ctx->acc_names[i]);
            pit[i].name = rocke_arena_strdup(&b->arena, pnb);
            pit[i].init = accs_in[i];
        }
        pit[n_acc].name = "xh_c";
        pit[n_acc].init = c0;
        pit[n_acc + 1].name = "xh_y";
        pit[n_acc + 1].init = c0;
        pit[n_acc + 2].name = "xh_z";
        pit[n_acc + 2].init = c0;
        if(spec->xh_wait_smem)
            rocke_b_s_waitcnt(b, -1, 0, -1);
        rocke_for_t pf = rocke_b_scf_for_iter(
            b, c0, xh_hi, ctx->c_block_k, pit, n_acc + 3, "xh_k", false, false);
        rocke_b_region_enter(b, pf.body);
        {
            rocke_value_t* iv[ROCKE_CONV_MAX_ACCS + 3];
            for(int i = 0; i < pf.num_iter_vars; ++i)
                iv[i] = pf.iter_vars[i];
            rocke_value_t* st[3] = {iv[n_acc], iv[n_acc + 1], iv[n_acc + 2]};
            rocke_value_t* st1[3];
            conv_xh_adv(x, st, st1);
            if(spec->rpf_setprio)
                rocke_b_s_setprio(b, spec->rpf_setprio);
            xh_st_t staged;
            xh_fetch(st1, &staged);
            if(spec->vmem_interleave)
                ctx->vmi_pending = sl_n[0] + sl_n[1];
            else
                rocke_b_sched_barrier(b, 0);
            rocke_value_t* acc_n[ROCKE_CONV_MAX_ACCS + 3];
            xh_mfma(iv, spec->defer_mfma != 0, acc_n);
            rocke_b_sched_barrier(b, 0);
            if(spec->rpf_setprio)
                rocke_b_s_setprio(b, 0);
            if(spec->defer_mfma)
            {
                rocke_b_sync_lds_only(b);
                xh_store(&staged);
                for(int d = 0; d < ctx->defer_cnt; ++d)
                {
                    int flat = 0;
                    for(int mi = 0; mi < mfmas_m; ++mi)
                        for(int ni = 0; ni < mfmas_n; ++ni)
                        {
                            acc_n[flat] = rocke_conv_emit_mfma(
                                b, ctx->atom, dfr[d].a[mi], dfr[d].b[ni], acc_n[flat]);
                            ++flat;
                        }
                }
                ctx->defer_cnt = 0;
                const int n_w = sl_n[0] + sl_n[1];
                const int n_m = spec->defer_mfma * mfmas_m * mfmas_n;
                int per_w = n_m / (n_w > 1 ? n_w : 1);
                per_w = per_w < 1 ? 1 : per_w;
                per_w = per_w > 2 ? 2 : per_w;
                const int nn = n_w < n_m ? n_w : n_m;
                for(int w = 0; w < nn; ++w)
                {
                    rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_WRITE, 1, 0);
                    rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, per_w, 0);
                }
                rocke_b_sched_barrier(b, 0);
                rocke_b_sync_lds_only(b);
            }
            else
            {
                if(spec->rpf_lds_barrier)
                    rocke_b_sync_lds_only(b);
                else
                    rocke_b_sync(b);
                xh_store(&staged);
                if(spec->rpf_lds_barrier)
                    rocke_b_sync_lds_only(b);
                else
                    rocke_b_sync(b);
            }
            for(int i = 0; i < 3; ++i)
                acc_n[n_acc + i] = st1[i];
            rocke_b_scf_yield(b, acc_n, n_acc + 3);
        }
        rocke_b_region_leave(b);
        for(int i = 0; i < n_acc; ++i)
            out_accs[i] = pf.op->results[i];
        return xh_ok;
    }

    /* ---- _emit_xhalo_dma (wave-private A) ---- */
    rocke_value_t* c64 = rocke_b_const_i32(b, 64);
    const int n_waves = spec->warp_m * spec->warp_n;
    rocke_value_t* lane_ = rocke_b_mod(b, ctx->tid, c64);
    rocke_value_t* wave_ = rocke_b_to_sgpr_u32(b, rocke_b_div(b, ctx->tid, c64));
    rocke_value_t* l16 = rocke_b_div(b, lane_, rocke_b_const_i32(b, 16));
    rocke_value_t* lm16 = rocke_b_mod(b, lane_, rocke_b_const_i32(b, 16));
    rocke_value_t* lch = rocke_b_div(b, lm16, rocke_b_const_i32(b, 4));
    rocke_value_t* ldw = rocke_b_mod(b, lane_, rocke_b_const_i32(b, 4));
    rocke_value_t* c_oob = rocke_b_const_i32(b, 0x7FFFFFFF);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* c_eb = c2;
    const int row_bytes = block_k * 2;
    rocke_value_t* A_addr = rocke_b_smem_addr_of(b, ctx->A_smem);
    rocke_value_t* B_addr = rocke_b_smem_addr_of(b, ctx->B_smem);
    const rocke_type_t* i64 = rocke_i64();
    conv_xpd_t xp = {spec->xh_pad, NULL, NULL};
    if(xp.on)
    {
        /* A buffer base moved down by the pad offset, so the wave ranges' lane
         * offsets (from lin_of, no pad term) and the tap offsets stay >= 0. */
        rocke_value_t* bd = rocke_b_mul(b, ctx->p_pD, s_di);
        rocke_value_t* bh = rocke_b_mul(b, ctx->p_pH, s_hi);
        rocke_value_t* bdh = rocke_b_add(b, bd, bh);
        rocke_value_t* bw = rocke_b_mul(b, ctx->p_pW, s_wi);
        rocke_value_t* bias = rocke_b_mul(b, rocke_b_add(b, bdh, bw), c_eb);
        rocke_value_t* nbias = rocke_b_sext(b, rocke_b_sub(b, c0, bias), i64);
        xp.a_ptr_b = rocke_b_global_ptr_offset(b, ctx->A, nbias);
        xp.a_nb = rocke_b_add(b, ctx->A_bytes, bias);
    }

    conv_dma_t a_dma[64];
    int n_adma = 0;
    rocke_value_t* pc_w[3] = {NULL, NULL, NULL};
    {
        rocke_value_t* wq
            = rocke_b_mul(b, wave_, rocke_b_const_i32(b, spec->tile_m / spec->warp_m));
        rocke_value_t* w_m0 = rocke_b_add(b, m0, wq);
        if(xp.on)
            conv_xpd_pcoords(ctx, w_m0, pc_w);
        rocke_value_t* a_base_w = rocke_b_mul(b, conv_xh_lin_of(ctx, w_m0), s_wi);
        if(u->a_gbase != NULL)
            a_base_w = rocke_b_add(b, a_base_w, u->a_gbase);
        const int hp = spec->tile_m / spec->warp_m + XT - 1;
        rocke_value_t* w_seg = rocke_b_mul(b, wave_, rocke_b_const_i32(b, WR / 4));
        rocke_value_t* p_rel[4];
        rocke_value_t* p_o[4];
        for(int k = 0; k < 4; ++k)
        {
            rocke_value_t* g = rocke_b_add(b, w_seg, rocke_b_const_i32(b, k));
            rocke_value_t* rel0 = rocke_b_xor(b, l16, rocke_b_land(b, g, c1));
            rocke_value_t* chunk = rocke_b_xor(b, lch, rocke_b_land(b, g, rocke_b_const_i32(b, 3)));
            rocke_value_t* rw = rocke_b_mul(b, rel0, s_wi);
            rocke_value_t* ch8 = rocke_b_mul(b, chunk, rocke_b_const_i32(b, 8));
            rocke_value_t* inner = rocke_b_add(b, ch8, rocke_b_mul(b, ldw, c2));
            rocke_value_t* o = rocke_b_add(b, rw, inner);
            p_rel[k] = rel0;
            p_o[k] = rocke_b_mul(b, o, c_eb);
        }
        for(int j = 0; j < WR / 4; ++j)
        {
            rocke_value_t* sg = rocke_b_add(b, w_seg, rocke_b_const_i32(b, j));
            rocke_value_t* rel0 = p_rel[j % 4];
            rocke_value_t* o = p_o[j % 4];
            rocke_value_t* r4 = rocke_b_mul(b, rocke_b_const_i32(b, 4 * j), s_wi);
            rocke_value_t* ubase
                = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_add(b, a_base_w, r4), c_eb));
            rocke_value_t* voff = rocke_b_add(b, o, ubase);
            if(4 * j + 3 >= hp)
            {
                rocke_value_t* rr = rocke_b_add(b, rel0, rocke_b_const_i32(b, 4 * j));
                rocke_value_t* okr = rocke_b_cmp_lt(b, rr, rocke_b_const_i32(b, hp));
                voff = rocke_b_select(b, okr, voff, c_oob);
            }
            if(xp.on)
            {
                rocke_value_t* rr = rocke_b_add(b, rel0, rocke_b_const_i32(b, 4 * j));
                voff = conv_xpd_wmask(ctx, pc_w[2], rr, voff, c_oob);
            }
            rocke_value_t* sb = rocke_b_mul(b, sg, rocke_b_const_i32(b, 4 * row_bytes));
            rocke_value_t* lds = rocke_b_smem_ptr_add(b, A_addr, rocke_b_zext(b, sb, i64));
            a_dma[n_adma].lds = lds;
            a_dma[n_adma++].voff = voff;
        }
    }
    /* B: rows (x, n) of the lds_xor layout; chunk = pchunk ^ (key(row) & 3). */
    const int _sh = spec->lds_xor_rowkey ? 0 : 1;
    conv_dma_t b_dma[256];
    int n_bdma = 0;
    rocke_value_t* c_tn = rocke_b_const_i32(b, block_n);
    const bool bcheap = n_waves == 4 && block_n == 64;
    if(bcheap)
    {
        rocke_value_t* key;
        if(_sh)
        {
            rocke_value_t* w4 = rocke_b_add(b, rocke_b_mul(b, wave_, rocke_b_const_i32(b, 4)), l16);
            key = rocke_b_lshr(b, w4, rocke_b_const_i32(b, _sh));
        }
        else
            key = l16;
        rocke_value_t* chunk = rocke_b_xor(b, lch, rocke_b_land(b, key, rocke_b_const_i32(b, 3)));
        rocke_value_t* ch8 = rocke_b_mul(b, chunk, rocke_b_const_i32(b, 8));
        rocke_value_t* col = rocke_b_add(b, ch8, rocke_b_mul(b, ldw, c2));
        rocke_value_t* bpats[4];
        for(int k = 0; k < 4; ++k)
        {
            rocke_value_t* c16 = rocke_b_const_i32(b, 16 * k);
            rocke_value_t* w4 = rocke_b_add(b, rocke_b_mul(b, wave_, rocke_b_const_i32(b, 4)), l16);
            rocke_value_t* nloc = rocke_b_add(b, c16, w4);
            rocke_value_t* k_out
                = rocke_b_add(b, ctx->block_n_off_v, rocke_conv_eqx_chan(ctx, nloc));
            if(ctx->k_out_group_base != NULL)
                k_out = rocke_b_add(b, ctx->k_out_group_base, k_out);
            rocke_value_t* ks = rocke_b_mul(b, k_out, ctx->p_B_stride_k);
            bpats[k] = rocke_b_mul(b, rocke_b_add(b, ks, col), c_eb);
        }
        for(int j = 0; j < XT * block_n / 16; ++j)
        {
            rocke_value_t* sg = rocke_b_add(b, rocke_b_const_i32(b, 4 * j), wave_);
            rocke_value_t* jx = rocke_b_mul(b, rocke_b_const_i32(b, j / 4), ctx->p_B_stride_x);
            rocke_value_t* ub = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, jx, c_eb));
            rocke_value_t* sb = rocke_b_mul(b, sg, rocke_b_const_i32(b, 4 * row_bytes));
            rocke_value_t* lds = rocke_b_smem_ptr_add(b, B_addr, rocke_b_zext(b, sb, i64));
            b_dma[n_bdma].lds = lds;
            b_dma[n_bdma++].voff = rocke_b_add(b, bpats[j % 4], ub);
        }
    }
    else
    {
        const int n_seg = XT * block_n / 4;
        for(int jj = 0; jj < (n_seg + n_waves - 1) / n_waves; ++jj)
        {
            rocke_value_t* sg = rocke_b_add(b, rocke_b_const_i32(b, jj * n_waves), wave_);
            if((jj + 1) * n_waves > n_seg)
                sg = rocke_b_smin(b, sg, rocke_b_const_i32(b, n_seg - 1));
            rocke_value_t* prow = rocke_b_add(b, rocke_b_mul(b, sg, rocke_b_const_i32(b, 4)), l16);
            rocke_value_t* key = _sh ? rocke_b_lshr(b, prow, rocke_b_const_i32(b, _sh)) : prow;
            rocke_value_t* chunk
                = rocke_b_xor(b, lch, rocke_b_land(b, key, rocke_b_const_i32(b, 3)));
            rocke_value_t* ch8 = rocke_b_mul(b, chunk, rocke_b_const_i32(b, 8));
            rocke_value_t* col = rocke_b_add(b, ch8, rocke_b_mul(b, ldw, c2));
            rocke_value_t* x_ = rocke_b_div(b, prow, c_tn);
            rocke_value_t* pm = rocke_b_mod(b, prow, c_tn);
            rocke_value_t* k_out = rocke_b_add(b, ctx->block_n_off_v, rocke_conv_eqx_chan(ctx, pm));
            if(ctx->k_out_group_base != NULL)
                k_out = rocke_b_add(b, ctx->k_out_group_base, k_out);
            rocke_value_t* ks = rocke_b_mul(b, k_out, ctx->p_B_stride_k);
            rocke_value_t* xs = rocke_b_mul(b, x_, ctx->p_B_stride_x);
            rocke_value_t* off = rocke_b_add(b, rocke_b_add(b, ks, xs), col);
            rocke_value_t* sb = rocke_b_mul(b, sg, rocke_b_const_i32(b, 4 * row_bytes));
            rocke_value_t* lds = rocke_b_smem_ptr_add(b, B_addr, rocke_b_zext(b, sb, i64));
            b_dma[n_bdma].lds = lds;
            b_dma[n_bdma++].voff = rocke_b_mul(b, off, c_eb);
        }
    }

    rocke_value_t* st0[3] = {c0, c0, c0};
    if(spec->xh_yring)
    {
        conv_xy_t y;
        y.x = x;
        y.xh_ok = xh_ok;
        y.xh_hi = xh_hi;
        y.wave_ = wave_;
        y.l16 = l16;
        y.lch = lch;
        y.ldw = ldw;
        y.c_oob = c_oob;
        y.c2 = c2;
        y.A_addr = A_addr;
        y.B_addr = B_addr;
        y.b_dma = b_dma;
        y.n_bdma = n_bdma;
        y.bcheap = bcheap;
        y.base_rows = base_rows;
        y.xp = xp;
        return conv_xh_yring(&y, accs_in, out_accs);
    }
    /* prologue: dma(st0) under xh_ok */
    {
        rocke_if_t g = rocke_b_scf_if(b, xh_ok);
        rocke_b_region_enter(b, g.then_region);
        rocke_value_t *ao, *bo, *to;
        conv_xh_uniforms(x, st0, &ao, &bo, &to);
        rocke_value_t* sa = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ao, c_eb));
        rocke_value_t* sb = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, bo, c_eb));
        rocke_value_t* a_rs = xp.on ? conv_xpd_rsrc(ctx, &xp, pc_w[0], pc_w[1], st0) : ctx->a_rsrc;
        for(int i = 0; i < n_adma; ++i)
            rocke_b_async_buffer_load_lds_addr(b, a_rs, a_dma[i].lds, a_dma[i].voff, sa, 1, 0);
        for(int i = 0; i < n_bdma; ++i)
            rocke_b_async_buffer_load_lds_addr(
                b, ctx->b_rsrc, b_dma[i].lds, b_dma[i].voff, sb, 1, 0);
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    const int n_mf = k_atoms * mfmas_m * mfmas_n;
    const int n_pre_mf = spec->fxh_dma % 1000;
    int na_pre = n_adma;
    if(spec->fxh_dma >= 2000)
        na_pre = (n_pre_mf / 2) < n_adma ? n_pre_mf / 2 : n_adma;
    const int n_rb = XT * mfmas_n;

    rocke_iter_arg_t it[ROCKE_CONV_MAX_ACCS + 3];
    char nbuf[64];
    for(int i = 0; i < n_acc; ++i)
    {
        snprintf(nbuf, sizeof(nbuf), "xd_%s", ctx->acc_names[i]);
        it[i].name = rocke_arena_strdup(&b->arena, nbuf);
        it[i].init = accs_in[i];
    }
    it[n_acc].name = "xd_c";
    it[n_acc].init = c0;
    it[n_acc + 1].name = "xd_y";
    it[n_acc + 1].init = c0;
    it[n_acc + 2].name = "xd_z";
    it[n_acc + 2].init = c0;
    rocke_for_t f
        = rocke_b_scf_for_iter(b, c0, xh_hi, ctx->c_block_k, it, n_acc + 3, "xd_k", false, false);
    rocke_b_region_enter(b, f.body);
    {
        rocke_value_t* iv[ROCKE_CONV_MAX_ACCS + 3];
        for(int i = 0; i < f.num_iter_vars; ++i)
            iv[i] = f.iter_vars[i];
        rocke_value_t* st[3] = {iv[n_acc], iv[n_acc + 1], iv[n_acc + 2]};
        rocke_value_t* st1[3];
        conv_xh_adv(x, st, st1);
        if(spec->rpf_setprio)
            rocke_b_s_setprio(b, spec->rpf_setprio);
        /* reads_part(True) / reads_part(False): the 3 taps' A then B fragments. */
        rocke_conv_frag_t* fa = (rocke_conv_frag_t*)rocke_arena_alloc(
            &b->arena, (size_t)XT * k_atoms * sizeof(rocke_conv_frag_t));
        rocke_conv_frag_t* fb = (rocke_conv_frag_t*)rocke_arena_alloc(
            &b->arena, (size_t)XT * k_atoms * sizeof(rocke_conv_frag_t));
        rocke_conv_frag_t* frs = (rocke_conv_frag_t*)rocke_arena_alloc(
            &b->arena, (size_t)XT * k_atoms * sizeof(rocke_conv_frag_t));
        typedef rocke_value_t* recrow_t[ROCKE_CONV_MAX_ACCS];
        recrow_t* recs
            = (recrow_t*)rocke_arena_alloc(&b->arena, (size_t)XT * k_atoms * sizeof(recrow_t));
        rocke_value_t* dummy[ROCKE_CONV_MAX_ACCS];
        for(int part = 1; part >= 0; --part)
        {
            for(int xx = 0; xx < XT; ++xx)
            {
                rocke_value_t* rows[ROCKE_CONV_MAX_ACCS];
                for(int mi = 0; mi < mfmas_m; ++mi)
                    rows[mi] = xx == 0 ? base_rows[mi]
                                       : rocke_b_add(b, base_rows[mi], rocke_b_const_i32(b, xx));
                ctx->xh_a_rows = rows;
                ctx->xh_b_row_off = xx * block_n;
                if(part)
                {
                    ctx->pp_b_from_p1 = 1; /* b_from = 0 */
                    ctx->xh_prev = xx == 0 ? NULL : recs + (size_t)(xx - 1) * k_atoms;
                    ctx->xh_rec = recs + (size_t)xx * k_atoms;
                }
                else
                {
                    ctx->pp_b_only = true;
                    ctx->xh_prev = NULL;
                    ctx->xh_rec = NULL;
                }
                ctx->pp_read = (part ? fa : fb) + (size_t)xx * k_atoms;
                ctx->pp_read_n = 0;
                rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, accs_in, n_acc, dummy);
                ctx->pp_read = NULL;
                ctx->pp_b_from_p1 = 0;
                ctx->pp_b_only = false;
                ctx->xh_prev = NULL;
                ctx->xh_rec = NULL;
                ctx->xh_a_rows = NULL;
                ctx->xh_b_row_off = 0;
            }
            if(part)
                rocke_b_sched_barrier(b, 0);
        }
        for(int xx = 0; xx < XT; ++xx)
            for(int kk = 0; kk < k_atoms; ++kk)
            {
                rocke_conv_frag_t* d = &frs[xx * k_atoms + kk];
                const rocke_conv_frag_t* sa_ = &fa[xx * k_atoms + kk];
                const rocke_conv_frag_t* sb_ = &fb[xx * k_atoms + kk];
                for(int i = 0; i < sa_->na; ++i)
                    d->a[i] = sa_->a[i];
                d->na = sa_->na;
                for(int i = 0; i < sb_->nb; ++i)
                    d->b[i] = sb_->b[i];
                d->nb = sb_->nb;
            }
        rocke_b_s_waitcnt(b, -1, n_rb, -1);
        rocke_value_t* nk = rocke_b_add(b, f.iv, ctx->c_block_k);
        rocke_value_t* has_next = rocke_b_cmp_lt(b, nk, xh_hi);
        rocke_value_t* sel[3];
        for(int i = 0; i < 3; ++i)
            sel[i] = rocke_b_select(b, has_next, st1[i], st[i]);
        rocke_value_t *ao, *bo, *to;
        conv_xh_uniforms(x, sel, &ao, &bo, &to);
        rocke_value_t* sa = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ao, c_eb));
        rocke_value_t* sb = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, bo, c_eb));
        rocke_value_t* a_rs = conv_xpd_rsrc(ctx, &xp, pc_w[0], pc_w[1], sel);
        for(int i = 0; i < na_pre; ++i)
            rocke_b_async_buffer_load_lds_addr(b, a_rs, a_dma[i].lds, a_dma[i].voff, sa, 1, 0);
        rocke_value_t* acc[ROCKE_CONV_MAX_ACCS];
        for(int i = 0; i < n_acc; ++i)
            acc[i] = iv[i];
        auto run_mf = [&](int lo, int hi) {
            for(int xx = 0; xx < XT; ++xx)
            {
                const int a_ = lo > xx * n_mf ? lo : xx * n_mf;
                const int b_ = hi < (xx + 1) * n_mf ? hi : (xx + 1) * n_mf;
                if(a_ < b_)
                {
                    rocke_value_t* o[ROCKE_CONV_MAX_ACCS];
                    ctx->pp_use = frs + (size_t)xx * k_atoms;
                    ctx->pp_i0 = a_ - xx * n_mf;
                    ctx->pp_i1 = b_ - xx * n_mf;
                    rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, acc, n_acc, o);
                    ctx->pp_use = NULL;
                    for(int i = 0; i < n_acc; ++i)
                        acc[i] = o[i];
                }
            }
        };
        run_mf(0, n_pre_mf);
        {
            const int per = n_pre_mf / (na_pre > 1 ? na_pre : 1);
            conv_xh_hint_interleave(
                b, 0, na_pre, ROCKE_SCHED_VMEM_READ, n_pre_mf, per > 1 ? per : 1);
        }
        rocke_b_sync_lds_only_fence(b, "compiler");
        for(int i = na_pre; i < n_adma; ++i)
            rocke_b_async_buffer_load_lds_addr(b, a_rs, a_dma[i].lds, a_dma[i].voff, sa, 1, 0);
        for(int i = 0; i < n_bdma; ++i)
            rocke_b_async_buffer_load_lds_addr(
                b, ctx->b_rsrc, b_dma[i].lds, b_dma[i].voff, sb, 1, 0);
        run_mf(n_pre_mf, XT * n_mf);
        {
            const int n_post = n_adma - na_pre + n_bdma;
            const int rest = XT * n_mf - n_pre_mf;
            int per = rest / (n_post > 1 ? n_post : 1) - 1;
            conv_xh_hint_interleave(b, 0, n_post, ROCKE_SCHED_VMEM_READ, rest, per > 1 ? per : 1);
        }
        if(spec->rpf_setprio)
            rocke_b_s_setprio(b, 0);
        rocke_b_sync_fence(b, "compiler");
        rocke_value_t* y[ROCKE_CONV_MAX_ACCS + 3];
        for(int i = 0; i < n_acc; ++i)
            y[i] = acc[i];
        for(int i = 0; i < 3; ++i)
            y[n_acc + i] = st1[i];
        rocke_b_scf_yield(b, y, n_acc + 3);
    }
    rocke_b_region_leave(b);
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = f.op->results[i];
    return xh_ok;
}

/* Python _xhalo_dma_yring: the wave-private fxd loop unrolled over the 3 y
 * taps of a (chunk, z), with the input-row ring (xh_yring = RR + 10*mode). */
static rocke_value_t*
    conv_xh_yring(conv_xy_t* y, rocke_value_t* const* accs_in, rocke_value_t** out_accs)
{
    conv_xh_t* x = y->x;
    rocke_conv_build_ctx_t* ctx = x->ctx;
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    conv_uk_t* u = x->u;
    const int XT = CONV_XH_TAPS;
    const int block_k = ctx->block_k;
    const int mfmas_m = ctx->mfmas_m, mfmas_n = ctx->mfmas_n, k_atoms = ctx->k_atoms;
    const int n_acc = ctx->num_accs;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* c1 = ctx->uk_c1;
    rocke_value_t* m0 = x->m0;
    rocke_value_t* s_wi = ctx->p_A_stride_wi;
    rocke_value_t* wave_ = y->wave_;
    rocke_value_t* c_eb = y->c2;
    const rocke_type_t* i64 = rocke_i64();
    const int row_bytes = block_k * 2;
    const int RR = spec->xh_yring % 10;
    const bool early = spec->xh_yring / 10 == 1 || spec->xh_yring / 10 == 2;
    const bool early_ring = spec->xh_yring / 10 == 2;
    const int QQ = spec->warp_m / RR;
    const int WR = rocke_conv_fxd_wave_rows(spec);
    const int NSEG = WR / 4;
    const int WEXT = spec->tile_m / spec->warp_m;
    const int hp = WEXT + XT - 1;
    if(early_ring)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "xh_yring early ring rows (+20) are not ported");
        return y->xh_ok;
    }
    rocke_value_t* c_hp = rocke_b_const_i32(b, hp);
    rocke_value_t* c_seg = rocke_b_const_i32(b, 4 * row_bytes);
    rocke_value_t* lp_rel[4];
    rocke_value_t* lp_o[4];
    for(int k = 0; k < 4; ++k)
    {
        rocke_value_t* rel0 = rocke_b_xor(b, y->l16, rocke_b_const_i32(b, k & 1));
        rocke_value_t* chunk = rocke_b_xor(b, y->lch, rocke_b_const_i32(b, k));
        rocke_value_t* rw = rocke_b_mul(b, rel0, s_wi);
        rocke_value_t* ch8 = rocke_b_mul(b, chunk, rocke_b_const_i32(b, 8));
        rocke_value_t* inner = rocke_b_add(b, ch8, rocke_b_mul(b, y->ldw, y->c2));
        rocke_value_t* o = rocke_b_add(b, rw, inner);
        lp_rel[k] = rel0;
        lp_o[k] = rocke_b_mul(b, o, c_eb);
    }
    auto a_base_of = [&](rocke_value_t* m_rel) -> rocke_value_t* {
        rocke_value_t* v = rocke_b_mul(b, conv_xh_lin_of(ctx, rocke_b_add(b, m0, m_rel)), s_wi);
        return u->a_gbase != NULL ? rocke_b_add(b, v, u->a_gbase) : v;
    };
    auto lds_at = [&](rocke_value_t* seg) -> rocke_value_t* {
        return rocke_b_smem_ptr_add(b, y->A_addr, rocke_b_zext(b, rocke_b_mul(b, seg, c_seg), i64));
    };
    rocke_value_t* a_base_w = a_base_of(rocke_b_mul(b, wave_, rocke_b_const_i32(b, WEXT)));
    /* xh_pad: tap masks / input w of each list's rows (own, ring row R - 1); the
     * z mask is the tile's (whole tiles per plane), shared by every list. */
    const conv_xpd_t* xp = &y->xp;
    rocke_value_t* pc_own[3] = {NULL, NULL, NULL};
    rocke_value_t* pc_ring[8][3];
    if(xp->on)
    {
        rocke_value_t* wo_ = rocke_b_mul(b, wave_, rocke_b_const_i32(b, WEXT));
        conv_xpd_pcoords(ctx, rocke_b_add(b, m0, wo_), pc_own);
        for(int cw = 0; cw < QQ; ++cw)
            conv_xpd_pcoords(
                ctx, rocke_b_add(b, m0, rocke_b_const_i32(b, ((RR - 1) * QQ + cw) * WEXT)), pc_ring[cw]);
    }
    conv_xdst_t own[64];
    int n_own = 0;
    for(int j = 0; j < NSEG; ++j)
    {
        rocke_value_t* r4 = rocke_b_mul(b, rocke_b_const_i32(b, 4 * j), s_wi);
        rocke_value_t* ub
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_add(b, a_base_w, r4), c_eb));
        rocke_value_t* voff = rocke_b_add(b, lp_o[j % 4], ub);
        if(4 * j + 3 >= hp)
        {
            rocke_value_t* rr = rocke_b_add(b, lp_rel[j % 4], rocke_b_const_i32(b, 4 * j));
            voff = rocke_b_select(b, rocke_b_cmp_lt(b, rr, c_hp), voff, y->c_oob);
        }
        if(xp->on)
        {
            rocke_value_t* rr = rocke_b_add(b, lp_rel[j % 4], rocke_b_const_i32(b, 4 * j));
            voff = conv_xpd_wmask(ctx, pc_own[2], rr, voff, y->c_oob);
        }
        own[n_own++] = {'o', j, NULL, NULL, voff};
    }
    rocke_value_t* vq = rocke_b_land(b, wave_, rocke_b_const_i32(b, 3));
    rocke_value_t* rel_v = lp_rel[3];
    rocke_value_t* o_v = lp_o[3];
    for(int k = 2; k >= 0; --k)
    {
        rocke_value_t* is_k = rocke_b_cmp_eq(b, vq, rocke_b_const_i32(b, k));
        rel_v = rocke_b_select(b, is_k, lp_rel[k], rel_v);
        o_v = rocke_b_select(b, is_k, lp_o[k], o_v);
    }
    const int n_slot = (NSEG + 3) / 4;
    rocke_value_t* scratch = rocke_b_const_i32(b, spec->warp_m * NSEG);
    rocke_value_t* ring_v[64];
    int ring_cw[64], ring_i[64];
    rocke_value_t* ring_js[64];
    int n_ring = 0;
    for(int cw = 0; cw < QQ; ++cw)
    {
        rocke_value_t* src = a_base_of(rocke_b_const_i32(b, ((RR - 1) * QQ + cw) * WEXT));
        for(int i = 0; i < n_slot; ++i)
        {
            rocke_value_t* j_s = rocke_b_add(b, wave_, rocke_b_const_i32(b, 4 * i));
            rocke_value_t* j4 = rocke_b_mul(b, j_s, rocke_b_const_i32(b, 4));
            rocke_value_t* jw = rocke_b_mul(b, j4, s_wi);
            rocke_value_t* ub
                = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_add(b, src, jw), c_eb));
            rocke_value_t* voff = rocke_b_add(b, o_v, ub);
            if(4 * (4 * i + 3) + 3 >= hp)
            {
                rocke_value_t* l1 = rocke_b_cmp_lt(b, j_s, rocke_b_const_i32(b, NSEG));
                rocke_value_t* rj
                    = rocke_b_add(b, rel_v, rocke_b_mul(b, j_s, rocke_b_const_i32(b, 4)));
                rocke_value_t* ok = rocke_b_land(b, l1, rocke_b_cmp_lt(b, rj, c_hp));
                voff = rocke_b_select(b, ok, voff, y->c_oob);
            }
            if(xp->on)
            {
                rocke_value_t* j4b = rocke_b_mul(b, j_s, rocke_b_const_i32(b, 4));
                rocke_value_t* rj = rocke_b_add(b, rel_v, j4b);
                voff = conv_xpd_wmask(ctx, pc_ring[cw][2], rj, voff, y->c_oob);
            }
            ring_v[n_ring] = voff;
            ring_cw[n_ring] = cw;
            ring_i[n_ring] = i;
            ring_js[n_ring++] = j_s;
        }
    }
    conv_xdst_t b_lazy[256];
    int n_blazy = 0;
    for(int j = 0; j < y->n_bdma; ++j)
    {
        if(y->bcheap)
            b_lazy[n_blazy++] = {'w', 4 * j, NULL, NULL, y->b_dma[j].voff};
        else
            b_lazy[n_blazy++] = {0, 0, NULL, y->b_dma[j].lds, y->b_dma[j].voff};
    }
    conv_xdst_t ring[CONV_XH_TAPS - 1][64];
    int n_rl[CONV_XH_TAPS - 1];
    for(int yy = 0; yy < XT - 1; ++yy)
    {
        n_rl[yy] = 0;
        for(int r = 0; r < n_ring; ++r)
        {
            const int d = (yy % RR) * QQ + ring_cw[r];
            rocke_value_t* inval = NULL;
            if(4 * (4 * ring_i[r] + 3) >= 4 * NSEG)
            {
                rocke_value_t* lt = rocke_b_cmp_lt(b, ring_js[r], rocke_b_const_i32(b, NSEG));
                inval = rocke_b_to_sgpr_u32(b, rocke_b_select(b, lt, c1, c0));
            }
            ring[yy][n_rl[yy]++] = {'w', d * NSEG + 4 * ring_i[r], inval, NULL, ring_v[r]};
        }
    }
    conv_xdst_t own2[64];
    int n_own2 = 0;
    rocke_value_t* uu = NULL;
    rocke_value_t* pc_u[3] = {NULL, NULL, NULL};
    if(early)
    {
        int lg0 = 0;
        for(int v = QQ - 1; v > 0; v >>= 1)
            ++lg0;
        rocke_value_t* rw0 = lg0 ? rocke_b_lshr(b, wave_, rocke_b_const_i32(b, lg0)) : wave_;
        rocke_value_t* a2 = rocke_b_add(b, rw0, rocke_b_const_i32(b, 2));
        rocke_value_t* m1 = rocke_b_land(b, a2, rocke_b_const_i32(b, RR - 1));
        rocke_value_t* mq = rocke_b_mul(b, m1, rocke_b_const_i32(b, QQ));
        rocke_value_t* wq = rocke_b_land(b, wave_, rocke_b_const_i32(b, QQ - 1));
        uu = rocke_b_to_sgpr_u32(b, rocke_b_add(b, mq, wq));
        rocke_value_t* a_base_u = a_base_of(rocke_b_mul(b, uu, rocke_b_const_i32(b, WEXT)));
        if(xp->on)
        {
            rocke_value_t* uo = rocke_b_mul(b, uu, rocke_b_const_i32(b, WEXT));
            conv_xpd_pcoords(ctx, rocke_b_add(b, m0, uo), pc_u);
        }
        for(int j = 0; j < NSEG; ++j)
        {
            rocke_value_t* r4 = rocke_b_mul(b, rocke_b_const_i32(b, 4 * j), s_wi);
            rocke_value_t* ub
                = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_add(b, a_base_u, r4), c_eb));
            rocke_value_t* voff = rocke_b_add(b, lp_o[j % 4], ub);
            if(4 * j + 3 >= hp)
            {
                rocke_value_t* rr = rocke_b_add(b, lp_rel[j % 4], rocke_b_const_i32(b, 4 * j));
                voff = rocke_b_select(b, rocke_b_cmp_lt(b, rr, c_hp), voff, y->c_oob);
            }
            if(xp->on)
            {
                rocke_value_t* rr = rocke_b_add(b, lp_rel[j % 4], rocke_b_const_i32(b, 4 * j));
                voff = conv_xpd_wmask(ctx, pc_u[2], rr, voff, y->c_oob);
            }
            own2[n_own2++] = {'u', j, NULL, NULL, voff};
        }
    }
    /* reads: region-local rows, + the region's base row per y */
    rocke_value_t* loc[ROCKE_CONV_MAX_ACCS];
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        rocke_value_t* wr = rocke_b_mul(b, ctx->grid.warp_m_idx, rocke_b_const_i32(b, WR));
        loc[mi] = rocke_b_sub(b, y->base_rows[mi], wr);
    }
    int lg = 0;
    for(int v = QQ - 1; v > 0; v >>= 1)
        ++lg;
    rocke_value_t* rw = lg ? rocke_b_lshr(b, wave_, rocke_b_const_i32(b, lg)) : wave_;
    rocke_value_t* cwv = rocke_b_land(b, wave_, rocke_b_const_i32(b, QQ - 1));
    rocke_value_t* roff[CONV_XH_TAPS];
    for(int yy = 0; yy < XT; ++yy)
    {
        rocke_value_t* a1 = rocke_b_add(b, rw, rocke_b_const_i32(b, yy));
        rocke_value_t* m1 = rocke_b_land(b, a1, rocke_b_const_i32(b, RR - 1));
        rocke_value_t* mq = rocke_b_mul(b, m1, rocke_b_const_i32(b, QQ));
        rocke_value_t* ad = rocke_b_add(b, mq, cwv);
        roff[yy] = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ad, rocke_b_const_i32(b, WR)));
    }
    const int n_mf = k_atoms * mfmas_m * mfmas_n;
    const int n_pre_mf = spec->fxh_dma % 1000;
    rocke_value_t* c_nseg = rocke_b_const_i32(b, NSEG);

    auto issue = [&](conv_xdst_t* lst,
                     int n,
                     rocke_value_t* rsrc,
                     rocke_value_t* base_addr,
                     rocke_value_t* soff,
                     rocke_value_t* z0) {
        rocke_value_t* bases[3] = {NULL, NULL, NULL}; /* own, w, u */
        for(int i = 0; i < n; ++i)
        {
            if(lst[i].kind == 0)
            {
                rocke_b_async_buffer_load_lds_addr(b, rsrc, lst[i].lds, lst[i].voff, soff, 1, 0);
                continue;
            }
            const int ki = lst[i].kind == 'o' ? 0 : lst[i].kind == 'w' ? 1 : 2;
            if(bases[ki] == NULL)
            {
                /* {"own": mul(wave_, c_nseg), "w": wave_}.get(kind) builds the dict. */
                rocke_value_t* own_b = rocke_b_mul(b, wave_, c_nseg);
                rocke_value_t* w_ = ki == 0 ? own_b : ki == 1 ? wave_ : rocke_b_mul(b, uu, c_nseg);
                bases[ki] = rocke_b_add(b, w_, z0);
            }
            rocke_value_t* seg = rocke_b_add(b, bases[ki], rocke_b_const_i32(b, lst[i].cst));
            if(lst[i].inval != NULL)
                seg = rocke_b_select(b, rocke_b_cmp_ne(b, lst[i].inval, c0), seg, scratch);
            rocke_value_t* dst
                = base_addr == NULL
                      ? lds_at(seg)
                      : rocke_b_smem_ptr_add(
                            b, base_addr, rocke_b_zext(b, rocke_b_mul(b, seg, c_seg), i64));
            rocke_b_async_buffer_load_lds_addr(b, rsrc, dst, lst[i].voff, soff, 1, 0);
        }
    };

    /* prologue */
    {
        rocke_if_t g = rocke_b_scf_if(b, y->xh_ok);
        rocke_b_region_enter(b, g.then_region);
        rocke_value_t* st0[3] = {c0, c0, c0};
        rocke_value_t *ao, *bo, *to;
        conv_xh_uniforms(x, st0, &ao, &bo, &to);
        rocke_value_t* sa0 = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ao, c_eb));
        rocke_value_t* sb0 = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, bo, c_eb));
        issue(own,
              n_own,
              conv_xpd_rsrc(ctx, xp, xp->on ? pc_ring[0][0] : NULL, pc_own[1], st0),
              NULL,
              sa0,
              c0);
        issue(b_lazy, n_blazy, ctx->b_rsrc, y->B_addr, sb0, c0);
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    rocke_iter_arg_t it[ROCKE_CONV_MAX_ACCS + 3];
    char nbuf[64];
    for(int i = 0; i < n_acc; ++i)
    {
        snprintf(nbuf, sizeof(nbuf), "xr_%s", ctx->acc_names[i]);
        it[i].name = rocke_arena_strdup(&b->arena, nbuf);
        it[i].init = accs_in[i];
    }
    it[n_acc] = {"xr_c", c0};
    it[n_acc + 1] = {"xr_y", c0};
    it[n_acc + 2] = {"xr_z", c0};
    rocke_value_t* c3bk = rocke_b_const_i32(b, XT * block_k);
    rocke_for_t f
        = rocke_b_scf_for_iter(b, c0, y->xh_hi, c3bk, it, n_acc + 3, "xr_k", false, false);
    rocke_b_region_enter(b, f.body);
    {
        rocke_value_t* vals[ROCKE_CONV_MAX_ACCS + 3];
        for(int i = 0; i < f.num_iter_vars; ++i)
            vals[i] = f.iter_vars[i];
        rocke_conv_frag_t* fa = (rocke_conv_frag_t*)rocke_arena_alloc(
            &b->arena, (size_t)XT * k_atoms * sizeof(rocke_conv_frag_t));
        rocke_conv_frag_t* fb = (rocke_conv_frag_t*)rocke_arena_alloc(
            &b->arena, (size_t)XT * k_atoms * sizeof(rocke_conv_frag_t));
        rocke_conv_frag_t* frs = (rocke_conv_frag_t*)rocke_arena_alloc(
            &b->arena, (size_t)XT * k_atoms * sizeof(rocke_conv_frag_t));
        typedef rocke_value_t* recrow_t[ROCKE_CONV_MAX_ACCS];
        recrow_t* recs
            = (recrow_t*)rocke_arena_alloc(&b->arena, (size_t)XT * k_atoms * sizeof(recrow_t));
        for(int yy = 0; yy < XT; ++yy)
        {
            rocke_value_t* kv_b
                = yy ? rocke_b_add(b, f.iv, rocke_b_const_i32(b, yy * block_k)) : f.iv;
            rocke_value_t* st[3] = {vals[n_acc], vals[n_acc + 1], vals[n_acc + 2]};
            rocke_value_t* st1[3];
            conv_xh_adv(x, st, st1);
            if(spec->rpf_setprio)
                rocke_b_s_setprio(b, spec->rpf_setprio);
            rocke_value_t* dummy[ROCKE_CONV_MAX_ACCS];
            for(int part = 1; part >= 0; --part)
            {
                ctx->xa_ring_off = part ? roff[yy] : NULL;
                for(int xx = 0; xx < XT; ++xx)
                {
                    rocke_value_t* rows[ROCKE_CONV_MAX_ACCS];
                    for(int mi = 0; mi < mfmas_m; ++mi)
                        rows[mi]
                            = xx == 0 ? loc[mi] : rocke_b_add(b, loc[mi], rocke_b_const_i32(b, xx));
                    ctx->xh_a_rows = rows;
                    ctx->xh_b_row_off = xx * ctx->block_n;
                    if(part)
                    {
                        ctx->pp_b_from_p1 = 1;
                        ctx->xh_prev = xx == 0 ? NULL : recs + (size_t)(xx - 1) * k_atoms;
                        ctx->xh_rec = recs + (size_t)xx * k_atoms;
                    }
                    else
                        ctx->pp_b_only = true;
                    ctx->pp_read = (part ? fa : fb) + (size_t)xx * k_atoms;
                    ctx->pp_read_n = 0;
                    rocke_conv_emit_mfma_phase(
                        ctx, ctx->A_smem, ctx->B_smem, accs_in, n_acc, dummy);
                    ctx->pp_read = NULL;
                    ctx->pp_b_from_p1 = 0;
                    ctx->pp_b_only = false;
                    ctx->xh_prev = NULL;
                    ctx->xh_rec = NULL;
                    ctx->xh_a_rows = NULL;
                    ctx->xh_b_row_off = 0;
                }
                ctx->xa_ring_off = NULL;
                if(part)
                    rocke_b_sched_barrier(b, 0);
            }
            for(int xx = 0; xx < XT; ++xx)
                for(int kk = 0; kk < k_atoms; ++kk)
                {
                    rocke_conv_frag_t* d = &frs[xx * k_atoms + kk];
                    const rocke_conv_frag_t* sa_ = &fa[xx * k_atoms + kk];
                    const rocke_conv_frag_t* sb_ = &fb[xx * k_atoms + kk];
                    for(int i = 0; i < sa_->na; ++i)
                        d->a[i] = sa_->a[i];
                    d->na = sa_->na;
                    for(int i = 0; i < sb_->nb; ++i)
                        d->b[i] = sb_->b[i];
                    d->nb = sb_->nb;
                }
            rocke_value_t* sel[3];
            rocke_value_t* a_rs;
            conv_xdst_t* a_list;
            int n_alist;
            if(yy == XT - 1)
            {
                rocke_value_t* nk = rocke_b_add(b, kv_b, ctx->c_block_k);
                rocke_value_t* hn = rocke_b_cmp_lt(b, nk, y->xh_hi);
                for(int i = 0; i < 3; ++i)
                    sel[i] = rocke_b_select(b, hn, st1[i], st[i]);
                a_list = own;
                n_alist = n_own;
                a_rs = conv_xpd_rsrc(ctx, xp, xp->on ? pc_ring[0][0] : NULL, pc_own[1], sel);
            }
            else
            {
                for(int i = 0; i < 3; ++i)
                    sel[i] = st1[i];
                a_list = ring[yy];
                n_alist = n_rl[yy];
                a_rs = conv_xpd_rsrc(
                    ctx, xp, xp->on ? pc_ring[0][0] : NULL, xp->on ? pc_ring[0][1] : NULL, sel);
            }
            rocke_value_t *ao, *bo, *to;
            conv_xh_uniforms(x, sel, &ao, &bo, &to);
            rocke_value_t* sa = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ao, c_eb));
            rocke_value_t* sb = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, bo, c_eb));
            int n_pre = 0;
            rocke_value_t* z0
                = rocke_b_to_sgpr_u32(b, rocke_b_lshr(b, kv_b, rocke_b_const_i32(b, 31)));
            if(early && yy == XT - 1)
            {
                rocke_b_s_waitcnt(b, -1, XT * mfmas_n, -1);
                n_pre = n_own2;
                n_alist = 0;
                issue(own2,
                      n_own2,
                      conv_xpd_rsrc(ctx, xp, xp->on ? pc_ring[0][0] : NULL, pc_u[1], sel),
                      NULL,
                      sa,
                      z0);
            }
            rocke_value_t* acc[ROCKE_CONV_MAX_ACCS];
            for(int i = 0; i < n_acc; ++i)
                acc[i] = vals[i];
            auto run_mf = [&](int lo, int hi) {
                for(int xx = 0; xx < XT; ++xx)
                {
                    const int a_ = lo > xx * n_mf ? lo : xx * n_mf;
                    const int b_ = hi < (xx + 1) * n_mf ? hi : (xx + 1) * n_mf;
                    if(a_ < b_)
                    {
                        rocke_value_t* o[ROCKE_CONV_MAX_ACCS];
                        ctx->pp_use = frs + (size_t)xx * k_atoms;
                        ctx->pp_i0 = a_ - xx * n_mf;
                        ctx->pp_i1 = b_ - xx * n_mf;
                        rocke_conv_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, acc, n_acc, o);
                        ctx->pp_use = NULL;
                        for(int i = 0; i < n_acc; ++i)
                            acc[i] = o[i];
                    }
                }
            };
            run_mf(0, n_pre_mf);
            {
                const int per = n_pre_mf / (n_pre > 1 ? n_pre : 1);
                conv_xh_hint_interleave(
                    b, 0, n_pre, ROCKE_SCHED_VMEM_READ, n_pre_mf, per > 1 ? per : 1);
            }
            rocke_b_sync_lds_only_fence(b, "compiler");
            issue(a_list, n_alist, a_rs, NULL, sa, z0);
            issue(b_lazy, n_blazy, ctx->b_rsrc, y->B_addr, sb, z0);
            run_mf(n_pre_mf, XT * n_mf);
            {
                const int n_post = n_alist + n_blazy;
                const int rest = XT * n_mf - n_pre_mf;
                const int per = rest / (n_post > 1 ? n_post : 1) - 1;
                conv_xh_hint_interleave(
                    b, 0, n_post, ROCKE_SCHED_VMEM_READ, rest, per > 1 ? per : 1);
            }
            if(spec->rpf_setprio)
                rocke_b_s_setprio(b, 0);
            rocke_b_sync_fence(b, "compiler");
            for(int i = 0; i < n_acc; ++i)
                vals[i] = acc[i];
            for(int i = 0; i < 3; ++i)
                vals[n_acc + i] = st1[i];
        }
        rocke_b_scf_yield(b, vals, n_acc + 3);
    }
    rocke_b_region_leave(b);
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = f.op->results[i];
    return y->xh_ok;
}

void rocke_conv_emit_fwd_kloops(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_implicit_gemm_conv_spec_t* spec = ctx->spec;
    const int n_acc = ctx->num_accs;
    rocke_value_t* c0 = ctx->c0;
    rocke_value_t* k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    rocke_value_t* k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    rocke_value_t* accs0[ROCKE_CONV_MAX_ACCS] = {};
    rocke_value_t* res[ROCKE_CONV_MAX_ACCS] = {};
    for(int i = 0; i < n_acc; ++i)
        accs0[i] = ctx->acc_inits[i];

    ctx->uk_c1 = rocke_b_const_i32(b, 1); /* c1 = b.const_i32(1) */
    const bool is3 = ctx->spec->problem.is_3d;

    if(!spec->uniform_k_decode)
    {
        conv_run_loop(ctx, k_lo, k_hi, accs0, NULL, NULL, 0, NULL, "k", res);
        rocke_conv_set_final_accs(ctx, res, n_acc);
        return;
    }

    /* ---- uniform-tap loop ---- */
    conv_uk_t* u = (conv_uk_t*)rocke_arena_alloc(&b->arena, sizeof(conv_uk_t));
    if(u == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "conv uk: arena alloc failed");
        return;
    }
    memset(u, 0, sizeof(*u));
    u->ctx = ctx;
    u->is3 = is3;
    rocke_value_t* c_bk_mask = rocke_b_const_i32(b, ctx->block_k - 1);
    rocke_value_t* uk_ok = rocke_b_cmp_eq(b, rocke_b_land(b, ctx->p_cpg, c_bk_mask), c0);
    if(ctx->fwd_asplit && spec->uk_chunk_group)
    {
        /* The split start decode below assumes whole channel groups. */
        rocke_value_t* gm = rocke_b_const_i32(b, spec->uk_chunk_group * ctx->block_k - 1);
        rocke_value_t* g_ok = rocke_b_cmp_eq(b, rocke_b_land(b, ctx->p_cpg, gm), c0);
        uk_ok = rocke_b_land(b, uk_ok, g_ok);
    }
    rocke_value_t* uk_hi = rocke_b_select(b, uk_ok, ctx->c_K_gemm, c0);
    u->a_ystep = rocke_b_mul(b, ctx->p_dH, ctx->p_A_stride_hi);
    u->a_xstep = rocke_b_mul(b, ctx->p_dW, ctx->p_A_stride_wi);
    if(is3)
        u->a_zstep = rocke_b_mul(b, ctx->p_dD, ctx->p_A_stride_di);
    u->a_gbase = (ctx->group_idx != NULL) ? rocke_b_mul(b, ctx->group_idx, ctx->p_cpg) : NULL;

    static const char* const uk_names[4] = {"uc", "ux", "uy", "uz"};
    const int n_state = is3 ? 4 : 3;
    rocke_value_t* state[4];
    rocke_value_t* uk_accs[ROCKE_CONV_MAX_ACCS] = {};

    if(!ctx->fwd_asplit)
    {
        for(int i = 0; i < n_state; ++i)
            state[i] = c0;
        rocke_value_t* accs_in0[ROCKE_CONV_MAX_ACCS] = {};
        for(int i = 0; i < n_acc; ++i)
            accs_in0[i] = accs0[i];
        if(spec->uk_xhalo)
        {
            conv_xh_t xx;
            xx.ctx = ctx;
            xx.u = u;
            xx.m0 = NULL;
            rocke_value_t* xo[ROCKE_CONV_MAX_ACCS] = {};
            rocke_value_t* xh_ok = conv_xh_emit(&xx, uk_ok, accs_in0, xo);
            for(int i = 0; i < n_acc; ++i)
                accs_in0[i] = xo[i];
            rocke_value_t* nx = rocke_b_land(b, uk_ok, rocke_b_lnot(b, xh_ok));
            uk_hi = rocke_b_select(b, nx, ctx->c_K_gemm, c0);
        }
        conv_run_loop(ctx, c0, uk_hi, accs_in0, uk_names, state, n_state, u, "uk", uk_accs);
        rocke_value_t* fb_lo = rocke_b_select(b, uk_ok, ctx->c_K_gemm, c0);
        conv_run_loop(ctx, fb_lo, ctx->c_K_gemm, uk_accs, NULL, NULL, 0, NULL, "fb", res);
        rocke_conv_set_final_accs(ctx, res, n_acc);
        return;
    }
    /* split_k: this slice runs K tiles [k_lo, k_hi) / block_k, so the uniform
     * tap state starts at tile k_lo / block_k in the loop's own tile order. A
     * slice past the end starts on an out-of-range tap and loads zeros. */
    rocke_value_t* t_lo = rocke_b_div(b, k_lo, ctx->c_block_k);
    rocke_value_t* n_taps_xy = rocke_b_mul(b, ctx->p_X, ctx->p_Y);
    rocke_value_t *c_s, *x_s, *y_s, *z_s = NULL;
    if(spec->uk_tap_fast && spec->uk_chunk_group)
    {
        /* Grouped tap-fast order: uk_chunk_group chunks fastest, then the tap,
         * then the next group (cpg a multiple of the group span). */
        rocke_value_t* c_g = rocke_b_const_i32(b, spec->uk_chunk_group);
        rocke_value_t* n_taps = rocke_b_mul(b, n_taps_xy, ctx->p_Z);
        rocke_value_t* g_tiles = rocke_b_mul(b, n_taps, c_g);
        rocke_value_t* grp = rocke_b_div(b, t_lo, g_tiles);
        rocke_value_t* g_r = rocke_b_mod(b, t_lo, g_tiles);
        rocke_value_t* tap = rocke_b_div(b, g_r, c_g);
        rocke_value_t* gg = rocke_b_mul(b, grp, c_g);
        rocke_value_t* gsum = rocke_b_add(b, gg, rocke_b_mod(b, g_r, c_g));
        c_s = rocke_b_mul(b, gsum, ctx->c_block_k);
        x_s = rocke_b_mod(b, tap, ctx->p_X);
        y_s = rocke_b_mod(b, rocke_b_div(b, tap, ctx->p_X), ctx->p_Y);
        z_s = rocke_b_div(b, tap, n_taps_xy);
        z_s = rocke_b_select(b, rocke_b_cmp_ge(b, c_s, ctx->p_cpg), ctx->p_Z, z_s);
    }
    else if(spec->uk_tap_fast)
    {
        rocke_value_t* n_taps = rocke_b_mul(b, n_taps_xy, ctx->p_Z);
        rocke_value_t* tap = rocke_b_mod(b, t_lo, n_taps);
        c_s = rocke_b_mul(b, rocke_b_div(b, t_lo, n_taps), ctx->c_block_k);
        x_s = rocke_b_mod(b, tap, ctx->p_X);
        y_s = rocke_b_mod(b, rocke_b_div(b, tap, ctx->p_X), ctx->p_Y);
        z_s = rocke_b_div(b, tap, n_taps_xy);
        z_s = rocke_b_select(b, rocke_b_cmp_ge(b, c_s, ctx->p_cpg), ctx->p_Z, z_s);
    }
    else
    {
        rocke_value_t* tap = rocke_b_div(b, k_lo, ctx->p_cpg);
        c_s = rocke_b_sub(b, k_lo, rocke_b_mul(b, tap, ctx->p_cpg));
        x_s = rocke_b_mod(b, tap, ctx->p_X);
        y_s = is3 ? rocke_b_mod(b, rocke_b_div(b, tap, ctx->p_X), ctx->p_Y)
                  : rocke_b_div(b, tap, ctx->p_X);
        if(is3)
            z_s = rocke_b_div(b, tap, n_taps_xy);
    }
    state[0] = c_s;
    state[1] = x_s;
    state[2] = y_s;
    state[3] = z_s;
    rocke_value_t* uk_hi_s = rocke_b_select(b, uk_ok, k_hi, k_lo);
    conv_run_loop(ctx, k_lo, uk_hi_s, accs0, uk_names, state, n_state, u, "uk", uk_accs);
    conv_run_loop(ctx, uk_hi_s, k_hi, uk_accs, NULL, NULL, 0, NULL, "fb", res);
    rocke_conv_set_final_accs(ctx, res, n_acc);
}
