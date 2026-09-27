// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_body_stages.c -- C99 port of the `_emit_body` closure
 * of build_moe_fused_mega_gemm (rocke/instances/common/moe_fused_mega.py, lines
 * 636-738), split into the STAGE phase functions over the build context
 * (rocke/instance_moe_fused_mega_internal.h).
 *
 * SCOPE OF THIS PART-FILE (the STAGE phase functions only):
 *   rocke_moe_mega_emit_stage1_gate_up        (Python STAGE 1, lines 637-652)
 *   rocke_moe_mega_emit_stage2_silu_to_lds    (Python STAGE 2, lines 654-673)
 *   rocke_moe_mega_emit_stage3_reshape        (Python STAGE 3, lines 675-684; no-op)
 *   rocke_moe_mega_emit_stage45_down_reduce   (Python STAGE 4+5, lines 686-734)
 *   rocke_moe_mega_emit_body                   (the closure walk, lines 636-738)
 *
 * The two file-level STAGE-4 helpers (_emit_moe_down_mfma_phase_lds_a /
 * _emit_moe_down_kloop_lds_a -> rocke_moe_mega_emit_down_mfma_phase_lds_a /
 * rocke_moe_mega_emit_down_kloop_lds_a) live in a SEPARATE part-file; this file
 * calls rocke_moe_mega_emit_down_kloop_lds_a through the internal header.
 *
 * Byte-identical builder-call sequence: every emitted SSA value and IR op is
 * produced in the exact lexical order of the Python `_emit_body`.
 */
#include <stdio.h>

#include "rocke/instance_fused_moe_internal.h" /* rocke_moe_silu_mul_f32 prototype */
#include "rocke/instance_moe_fused_mega_internal.h"

/* ===================================================================== *
 *  STAGE 1: gate + up GEMM -> f32 register groups (Python lines 637-652).
 *
 *      gate_res, up_res = _emit_moe_prefetch_kloop(
 *          plan, a_view, a_lds_view, A_smem, a_mn_origin, operands,
 *          b_mn_origin, [gate_accs, up_accs], K,
 *          warp_m_idx, warp_n_idx, lane,
 *          sched_groups=2 * mfmas_m * mfmas_n)
 *
 *  The two named accumulator-init groups (gate_accs / up_accs) are each
 *  length mfmas_m*mfmas_n = ctx->num_gate_up_accs, every element == acc_init
 *  (Python comprehensions, lines 557-566). The reused C k-loop takes the inits
 *  flat in operand-then-flat order; group_sizes is one entry per operand.
 *  out_gate_res / out_up_res receive the two final flat groups.
 * ===================================================================== */
void rocke_moe_mega_emit_stage1_gate_up(rocke_moe_mega_build_ctx_t* ctx,
                                        rocke_value_t** out_gate_res,
                                        rocke_value_t** out_up_res)
{
    const int n = ctx->num_gate_up_accs; /* mfmas_m * mfmas_n */
    const int group_sizes[2] = {n, n};

    /* Flat acc-inits in operand-then-flat order: [gate_accs..., up_accs...].
     * The Python list comprehensions fill every slot with the SAME acc_init
     * Value (one _emit_zero_acc result reused). */
    rocke_value_t* acc_inits_flat[2 * ROCKE_MOE_MEGA_MAX_ACCS];
    for(int j = 0; j < n; ++j)
    {
        acc_inits_flat[j] = ctx->acc_init; /* gate group */
        acc_inits_flat[n + j] = ctx->acc_init; /* up   group */
    }

    /* Loop-carried accumulator SSA names, byte-identical to the Python
     * comprehensions (moe_fused_mega.py lines 557-566):
     *   gate_accs = [("gate_acc_m{mi}_n{ni}", acc_init) ...]
     *   up_accs   = [("up_acc_m{mi}_n{ni}",   acc_init) ...]
     * flat slot within each group = mi * mfmas_n + ni; operand-then-flat order
     * is [gate group..., up group...]. */
    /* 40 fits "gate_acc_m"/"up_acc_m" + two worst-case %d (11 chars each) + NUL,
     * so the emitted SSA names can never be truncated. */
    char gu_name_buf[2 * ROCKE_MOE_MEGA_MAX_ACCS][40];
    const char* acc_names_flat[2 * ROCKE_MOE_MEGA_MAX_ACCS];
    for(int mi = 0; mi < ctx->mfmas_m; ++mi)
    {
        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            const int flat = mi * ctx->mfmas_n + ni;
            snprintf(gu_name_buf[flat], sizeof(gu_name_buf[0]), "gate_acc_m%d_n%d", mi, ni);
            acc_names_flat[flat] = gu_name_buf[flat];
            snprintf(gu_name_buf[n + flat], sizeof(gu_name_buf[0]), "up_acc_m%d_n%d", mi, ni);
            acc_names_flat[n + flat] = gu_name_buf[n + flat];
        }
    }

    /* a_mn_origin / b_mn_origin are 2-elem (batch_off, block_mn_off) arrays. */
    rocke_value_t* a_mn_origin[2] = {ctx->a_mn_origin[0], ctx->a_mn_origin[1]};
    rocke_value_t* b_mn_origin[2] = {ctx->b_mn_origin[0], ctx->b_mn_origin[1]};

    const int sched_groups = 2 * ctx->mfmas_m * ctx->mfmas_n;

    /* The reused k-loop returns the final groups flat (sum(group_sizes)). */
    rocke_value_t* out_flat[2 * ROCKE_MOE_MEGA_MAX_ACCS];
    rocke_moe_emit_prefetch_kloop(&ctx->plan,
                                  ctx->a_view,
                                  ctx->a_lds_view,
                                  ctx->A_smem,
                                  a_mn_origin,
                                  ctx->operands,
                                  2,
                                  b_mn_origin,
                                  acc_inits_flat,
                                  acc_names_flat,
                                  group_sizes,
                                  ctx->K,
                                  ctx->warp_m_idx,
                                  ctx->warp_n_idx,
                                  ctx->lane,
                                  sched_groups,
                                  out_flat);

    /* Split the flat result back into gate_res (group 0) / up_res (group 1). */
    for(int j = 0; j < n; ++j)
    {
        out_gate_res[j] = out_flat[j];
        out_up_res[j] = out_flat[n + j];
    }
}

/* ===================================================================== *
 *  STAGE 2: SiLU(gate)*up -> PERSISTENT LDS Hidden_smem (Python lines 654-673).
 *
 *      warp_m_off = b.mul(warp_m_idx, const_i32(mfmas_m * t.warp_tile_m))
 *      warp_n_off = b.mul(warp_n_idx, const_i32(mfmas_n * t.warp_tile_n))
 *      cdec = _CWarpDecode(b, u_gu, warp_m_off, warp_n_off, lane)
 *      def _silu_cell(mi, ni, i):
 *          flat = mi * mfmas_n + ni
 *          g  = b.vec_extract(gate_res[flat], i)
 *          up = b.vec_extract(up_res[flat], i)
 *          return b.cast_f32_to(
 *              _silu_mul_f32(b, g, up, one_f32=one_f32, c_neg_log2e=c_neg_log2e),
 *              storage_dtype)
 *      _emit_cshuffle_stage(b, u_gu, cdec, Hidden_smem, storage_dtype,
 *                           c_per_lane, _silu_cell)
 *      b.sync()
 *
 *  The Python `_silu_cell` closure captures b/mfmas_n/gate_res/up_res/one_f32/
 *  c_neg_log2e/storage_dtype. In C the cell is a free function + a user-context
 *  struct carrying those captures.
 * ===================================================================== */

/* Capture context for the per-(mi,ni,i) SiLU cell (Python `_silu_cell`). */
typedef struct rocke_moe_mega_silu_cell_ctx
{
    rocke_ir_builder_t* b;
    int mfmas_n;
    rocke_value_t* const* gate_res;
    rocke_value_t* const* up_res;
    rocke_value_t* one_f32;
    rocke_value_t* c_neg_log2e;
    const rocke_type_t* storage_dtype;
} rocke_moe_mega_silu_cell_ctx_t;

static rocke_value_t* rocke_moe_mega_silu_cell(int mi, int ni, int i, void* user)
{
    rocke_moe_mega_silu_cell_ctx_t* cc = (rocke_moe_mega_silu_cell_ctx_t*)user;
    const int flat = mi * cc->mfmas_n + ni;
    rocke_value_t* g = rocke_b_vec_extract(cc->b, cc->gate_res[flat], i);
    rocke_value_t* up = rocke_b_vec_extract(cc->b, cc->up_res[flat], i);
    rocke_value_t* silu = rocke_moe_silu_mul_f32(cc->b, g, up, cc->one_f32, cc->c_neg_log2e);
    return rocke_b_cast_f32_to(cc->b, silu, cc->storage_dtype);
}

void rocke_moe_mega_emit_stage2_silu_to_lds(rocke_moe_mega_build_ctx_t* ctx,
                                            rocke_value_t* const* gate_res,
                                            rocke_value_t* const* up_res)
{
    rocke_ir_builder_t* b = ctx->b;

    rocke_value_t* warp_m_off = rocke_b_mul(
        b, ctx->warp_m_idx, rocke_b_const_i32(b, (int64_t)ctx->mfmas_m * ctx->t.warp_tile_m));
    rocke_value_t* warp_n_off = rocke_b_mul(
        b, ctx->warp_n_idx, rocke_b_const_i32(b, (int64_t)ctx->mfmas_n * ctx->t.warp_tile_n));

    rocke_moe_cwarp_decode_t cdec;
    if(!rocke_moe_cwarp_decode_init(&cdec, b, ctx->u_gu, warp_m_off, warp_n_off, ctx->lane))
    {
        return; /* sticky error already set on b */
    }

    rocke_moe_mega_silu_cell_ctx_t cell_ctx;
    cell_ctx.b = b;
    cell_ctx.mfmas_n = ctx->mfmas_n;
    cell_ctx.gate_res = gate_res;
    cell_ctx.up_res = up_res;
    cell_ctx.one_f32 = ctx->one_f32;
    cell_ctx.c_neg_log2e = ctx->c_neg_log2e;
    cell_ctx.storage_dtype = ctx->storage_dtype;

    rocke_moe_emit_cshuffle_stage(b,
                                  ctx->u_gu,
                                  &cdec,
                                  ctx->Hidden_smem,
                                  ctx->storage_dtype,
                                  ctx->c_per_lane,
                                  rocke_moe_mega_silu_cell,
                                  &cell_ctx);
    rocke_b_sync(b);
}

/* ===================================================================== *
 *  STAGE 3: RESHAPE Hidden_smem -> down-GEMM A layout (Python lines 675-684).
 *
 *  OPTION A is an IDENTITY: _emit_cshuffle_stage stored Hidden_smem in logical
 *  (row=m, col=inter) packed layout, and the down k-loop reads A via the SAME
 *  logical (m, i) indexing, so no explicit LDS transpose IR is emitted. This is
 *  a documented no-op phase; it exists so the body walks all five comment stages
 *  and so the parity fallback (Option B / load_tile_transpose) has a named hook.
 * ===================================================================== */
void rocke_moe_mega_emit_stage3_reshape(rocke_moe_mega_build_ctx_t* ctx)
{
    (void)ctx; /* Option A: no IR emitted. */
}

/* ===================================================================== *
 *  STAGE 4 + 5: DOWN GEMM (Hidden_LDS @ Wdown^T) -> weighted atomic reduce into
 *  Y (Python lines 686-734).
 *
 *      down_for = b.scf_for_iter(c0, H_out, c_block_n_down, [], iv_name="ho")
 *      with down_for as ho:
 *          down_accs = [(f"down_acc_m{mi}_n{ni}", down_acc_init)
 *                       for mi in range(down_mfmas_m)
 *                       for ni in range(down_mfmas_n)]
 *          down_res = _emit_moe_down_kloop_lds_a(
 *              plan_down, Hidden_smem, down_operand, down_accs,
 *              (batch_off_b, ho), gu_n_off, c_down_k,
 *              warp_m_idx, warp_n_idx, lane,
 *              sched_groups=down_mfmas_m * down_mfmas_n)
 *          b.sync()
 *          _emit_down_reduce_epilogue_atomic(
 *              b, u_down, tuple(down_res), warp_m_idx, warp_n_idx, lane,
 *              block_m_off, ho, M, H_out, SortedTokenIds, SortedWeights, Y,
 *              c_per_lane, batch_bucket_off=c0, tokens=tokens)
 *          b.scf_yield()
 *
 *  The loop carries NO iter-args (empty list); the induction variable `ho` is
 *  the down-output column base. Each iteration re-builds the per-tile down
 *  acc-init group (every slot == ctx->down_acc_init).
 * ===================================================================== */
void rocke_moe_mega_emit_stage45_down_reduce(rocke_moe_mega_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* scf_for_iter(c0, H_out, c_block_n_down, [], iv_name="ho"). No carried
     * iter-args (Python passes []). */
    rocke_for_t down_for = rocke_b_scf_for_iter(b,
                                                ctx->c0,
                                                ctx->H_out,
                                                ctx->c_block_n_down,
                                                /*iter_args=*/NULL,
                                                /*num=*/0,
                                                "ho",
                                                /*unroll=*/false,
                                                /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, down_for.body);
    {
        rocke_value_t* ho = down_for.iv;

        /* Per-tile down acc-init group: down_mfmas_m * down_mfmas_n slots, each
         * == down_acc_init (Python comprehension, lines 694-698). */
        const int nd = ctx->num_down_accs; /* down_mfmas_m * down_mfmas_n */
        rocke_value_t* down_accs[ROCKE_MOE_MEGA_MAX_ACCS];
        for(int j = 0; j < nd; ++j)
        {
            down_accs[j] = ctx->down_acc_init;
        }

        /* W_down output-row base = ho; contraction base = this TG's inter slice
         * (gu_n_off). b_mn_origin = (batch_off_b, ho). */
        rocke_value_t* b_mn_origin[2] = {ctx->batch_off_b, ho};

        const int down_sched_groups = ctx->down_mfmas_m * ctx->down_mfmas_n;

        rocke_value_t* down_res[ROCKE_MOE_MEGA_MAX_ACCS];
        rocke_moe_mega_emit_down_kloop_lds_a(&ctx->plan_down,
                                             ctx->Hidden_smem,
                                             &ctx->down_operand,
                                             down_accs,
                                             nd,
                                             b_mn_origin,
                                             ctx->gu_n_off,
                                             ctx->c_down_k,
                                             ctx->warp_m_idx,
                                             ctx->warp_n_idx,
                                             ctx->lane,
                                             down_sched_groups,
                                             down_res);

        /* Barrier before reusing Bd_smem in the next output tile. */
        rocke_b_sync(b);

        rocke_moe_emit_down_reduce_epilogue_atomic(b,
                                                   ctx->u_down,
                                                   down_res,
                                                   ctx->warp_m_idx,
                                                   ctx->warp_n_idx,
                                                   ctx->lane,
                                                   ctx->block_m_off,
                                                   ho, /* block_n_off */
                                                   ctx->M,
                                                   ctx->H_out, /* N for down out */
                                                   ctx->SortedTokenIds,
                                                   ctx->SortedWeights,
                                                   ctx->Y,
                                                   ctx->c_per_lane,
                                                   ctx->c0, /* batch_bucket_off */
                                                   ctx->tokens);

        rocke_b_scf_yield(b, NULL, 0);
    }
    rocke_b_region_leave(b);
}

/* ===================================================================== *
 *  `_emit_body`: walk STAGE 1..5 in Python order (lines 636-738).
 *
 *  STAGE 1 fills the gate/up f32 register groups consumed by STAGE 2; this
 *  entry owns those temporaries. STAGE 3 is the Option-A identity no-op. The
 *  public driver wraps this call in scf_if(expert_idx >= 0).
 * ===================================================================== */
void rocke_moe_mega_emit_body(rocke_moe_mega_build_ctx_t* ctx)
{
    /* STAGE 1: gate + up GEMM -> f32 register groups. */
    rocke_value_t* gate_res[ROCKE_MOE_MEGA_MAX_ACCS];
    rocke_value_t* up_res[ROCKE_MOE_MEGA_MAX_ACCS];
    rocke_moe_mega_emit_stage1_gate_up(ctx, gate_res, up_res);

    /* STAGE 2: SiLU(gate)*up -> PERSISTENT LDS Hidden_smem. */
    rocke_moe_mega_emit_stage2_silu_to_lds(ctx, gate_res, up_res);

    /* STAGE 3: reshape note -- Option A identity (no IR). */
    rocke_moe_mega_emit_stage3_reshape(ctx);

    /* STAGE 4 + 5: DOWN GEMM -> weighted atomic reduce into Y. */
    rocke_moe_mega_emit_stage45_down_reduce(ctx);
}
