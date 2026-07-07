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
#include "rocke/instance_conv_implicit_gemm_internal.h"

/* ----- shared small helper: copy a working acc array into ctx->final_accs ----
 * Python sets `final_accs = current_accs` (or `for_op.results`); in C the
 * drivers write the ctx slot the epilogue reads. */
static void
    rocke_conv_set_final_accs(rocke_conv_build_ctx_t* ctx, rocke_value_t* const* accs, int num_accs)
{
    int i;
    ctx->num_final_accs = num_accs;
    for(i = 0; i < num_accs; ++i)
        ctx->final_accs[i] = accs[i];
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
    const rocke_conv_problem_t* p = ctx->p;
    int block_k = ctx->block_k;
    int K_iters = (rocke_conv_problem_k_gemm(p) + block_k - 1) / block_k;
    int num_accs = ctx->num_accs;
    int it, i;

    /* current_accs = [v for _, v in accs] */
    rocke_value_t* current_accs[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];

    /* bufs = [(A_smem, B_smem), (A_smem2, B_smem2)] */
    rocke_value_t* buf_a[2];
    rocke_value_t* buf_b[2];
    buf_a[0] = ctx->A_smem;
    buf_b[0] = ctx->B_smem;
    buf_a[1] = ctx->A_smem2;
    buf_b[1] = ctx->B_smem2;

    for(i = 0; i < num_accs; ++i)
        current_accs[i] = ctx->acc_inits[i];

    /* Prologue: stage tile 0 into buffer 0 and publish it.
     *   emit_load_phase(b.const_i32(0), bufs[0][0], bufs[0][1]) */
    rocke_conv_emit_load_phase(ctx, rocke_b_const_i32(b, 0), buf_a[0], buf_b[0]);
    rocke_b_sync(b);

    for(it = 0; it < K_iters; ++it)
    {
        int cur = it % 2;
        if(it + 1 < K_iters)
        {
            int nxt = (it + 1) % 2;
            /* emit_load_phase(b.const_i32((it + 1) * block_k), nxt[0], nxt[1]) */
            rocke_conv_emit_load_phase(
                ctx, rocke_b_const_i32(b, (int64_t)(it + 1) * block_k), buf_a[nxt], buf_b[nxt]);
        }
        /* The prefetch above clobbered k_off_capture with tile it+1's offset.
         * Restore tile it's offset so an a_operand_override (if any) addresses
         * the tile actually consumed by this MFMA.
         *   k_off_capture[0] = b.const_i32(it * block_k) */
        ctx->k_off_capture = rocke_b_const_i32(b, (int64_t)it * block_k);
        /* current_accs = emit_mfma_phase(cur[0], cur[1], current_accs) */
        rocke_conv_emit_mfma_phase(ctx, buf_a[cur], buf_b[cur], current_accs, num_accs, new_accs);
        for(i = 0; i < num_accs; ++i)
            current_accs[i] = new_accs[i];
        rocke_b_sync(b);
    }

    /* final_accs = current_accs */
    rocke_conv_set_final_accs(ctx, current_accs, num_accs);
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

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->c0,
                                  ctx->c_K_gemm,
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
 * rocke_conv_emit_kloop_async   (Python lines 1320-1347)
 *
 * async_dma path: SoftwarePipeline.run_ping_pong over the AsyncTileLoader path.
 * SoftwarePipeline is a peer port; until it lands, this driver reproduces the
 * run_ping_pong sequencing inline (helpers/pipeline.py, run_ping_pong) for the
 * exact policy the conv builder constructs:
 *
 *   SoftwarePipeline(num_iters=K_iters, double_buffer=double_buffer,
 *                    wait_vmcnt=True, sync_after_wait=True,
 *                    sync_before_issue=True, overlap_vmcnt=True)
 *   issue_load(it, buf)      = emit_load_phase(const_i32(it*block_k), buf[0], buf[1])
 *   compute(it, buf, state)  = emit_mfma_phase(buf[0], buf[1], state)
 *   buffers = [(A_smem,B_smem),(A_smem2,B_smem2)]
 *   schedule = ctx->schedule
 *
 * Constructed flags: wait_vmcnt=True, sync_after_wait=True,
 * sync_before_issue=True, overlap_vmcnt=True; num_buffers unset (0) =>
 * nb derived from double_buffer. The two buffer pairs supplied bound nb<=2.
 * ===================================================================== */
void rocke_conv_emit_kloop_async(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_conv_problem_t* p = ctx->p;
    int block_k = ctx->block_k;
    int num_iters = (rocke_conv_problem_k_gemm(p) + block_k - 1) / block_k;
    int num_accs = ctx->num_accs;
    int it, i, pp;

    /* Pipeline flags (the conv builder's SoftwarePipeline(...) ctor). */
    const bool wait_vmcnt = true;
    const bool sync_after_wait = true;
    const bool sync_before_issue = true;
    const bool overlap_vmcnt = true;
    const int num_buffers = 0; /* 0 => derive from double_buffer */
    const bool double_buffer = ctx->double_buffer;

    /* buffers = [(A_smem,B_smem),(A_smem2,B_smem2)] */
    rocke_value_t* buf_a[2];
    rocke_value_t* buf_b[2];
    int n_bufs = 2;

    /* state = initial_state = [v for _, v in accs] */
    rocke_value_t* state[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* new_state[ROCKE_CONV_MAX_ACCS];

    int nb; /* derived buffer count                     */
    bool rotating; /* nb > 1                                    */
    int prefetch_depth;

    buf_a[0] = ctx->A_smem;
    buf_b[0] = ctx->B_smem;
    buf_a[1] = ctx->A_smem2;
    buf_b[1] = ctx->B_smem2;

    for(i = 0; i < num_accs; ++i)
        state[i] = ctx->acc_inits[i];

    /* if self.num_iters <= 0: return initial_state */
    if(num_iters <= 0)
    {
        rocke_conv_set_final_accs(ctx, state, num_accs);
        return;
    }
    /* `buffers` is always non-empty here (two pairs supplied). */

    /* Derive buffer count: prefer explicit num_buffers, fall back to the legacy
     * double_buffer boolean. */
    if(num_buffers > 0)
        nb = num_buffers;
    else if(double_buffer)
        nb = 2;
    else
        nb = 1;

    /* if nb > len(buffers): raise. len(buffers) == 2 here. */
    /* (Reproduced as a clamp guard; the conv builder always supplies 2 pairs.) */
    if(nb > n_bufs)
        nb = n_bufs;

    rotating = nb > 1;
    prefetch_depth = nb - 1;

    /* Prologue: issue the first prefetch_depth loads.
     *   for p in range(min(prefetch_depth, num_iters)):
     *       issue_load(p, buffers[p % nb]) */
    {
        int prologue_n = prefetch_depth < num_iters ? prefetch_depth : num_iters;
        for(pp = 0; pp < prologue_n; ++pp)
        {
            int slot = pp % nb;
            rocke_conv_emit_load_phase(
                ctx, rocke_b_const_i32(b, (int64_t)pp * block_k), buf_a[slot], buf_b[slot]);
        }
    }

    for(it = 0; it < num_iters; ++it)
    {
        int cur = rotating ? (it % nb) : 0;
        int issue_idx = it + prefetch_depth;
        bool has_next = rotating && (issue_idx < num_iters);

        if(has_next)
        {
            int nxt = issue_idx % nb;
            if(it > 0 && sync_before_issue)
            {
                if(overlap_vmcnt)
                    rocke_b_sync_lds_only(b);
                else
                    rocke_b_sync(b);
            }
            /* issue_load(issue_idx, nxt) */
            rocke_conv_emit_load_phase(
                ctx, rocke_b_const_i32(b, (int64_t)issue_idx * block_k), buf_a[nxt], buf_b[nxt]);
        }

        if(wait_vmcnt)
        {
            if(overlap_vmcnt && has_next)
                rocke_b_s_waitcnt(b, /*vmcnt=*/prefetch_depth, /*lgkmcnt=*/-1, /*expcnt=*/-1);
            else
                rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/-1, /*expcnt=*/-1);
        }

        if(sync_after_wait)
        {
            if(overlap_vmcnt && has_next)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
        }

        /* schedule is not None for the conv path. */
        rocke_schedule_policy_emit_compute_prologue(&ctx->schedule, b);

        /* state = compute(it, cur, state) = emit_mfma_phase(cur[0], cur[1], state) */
        rocke_conv_emit_mfma_phase(ctx, buf_a[cur], buf_b[cur], state, num_accs, new_state);
        for(i = 0; i < num_accs; ++i)
            state[i] = new_state[i];

        rocke_schedule_policy_emit_compute_epilogue(&ctx->schedule, b);

        if(!rotating)
            rocke_b_sync(b);
    }

    /* return state */
    rocke_conv_set_final_accs(ctx, state, num_accs);
}
