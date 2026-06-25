// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_down_kloop_lds_a.c -- C99 port of the two STAGE-4
 * file-level down-GEMM-with-LDS-A helpers of
 * ck_dsl/instances/common/moe_fused_mega.py (lines 242-426):
 *
 *   _emit_moe_down_mfma_phase_lds_a  -> ckc_moe_mega_emit_down_mfma_phase_lds_a
 *   _emit_moe_down_kloop_lds_a       -> ckc_moe_mega_emit_down_kloop_lds_a
 *
 * Both are module-level (ctx-free) functions taking their own per-call args, so
 * they bind only to the internal header's prototypes + the reused moe_gemm_fused
 * plan/operand helpers and the gemm_universal _emit_smem_load / _emit_mfma.
 *
 * The MFMA phase is identical to _emit_moe_mfma_phase (single B operand) EXCEPT
 * the A fragment is read from the persistent Hidden_smem at column
 * `a_col_base + k_blk*a_per_lane + kk*warp_tile_k` (the K-tile origin k0 added to
 * the per-tile-local A offset); the B (W_down) fragment read is the per-tile-
 * local single-buffered LDS column unchanged.
 *
 * The k-loop is a trimmed copy of _emit_moe_prefetch_kloop with the A global-load
 * and A LDS-store removed (A already resident in LDS at full contraction width):
 * only the W_down B operand is software-prefetched.
 *
 * Byte-faithful builder-call order against ckc/ir.h + the sibling helper headers.
 */
#include "ckc/instance_moe_fused_mega_internal.h"

#include <stdio.h>
#include <string.h>

#include "ckc/instance_gemm_internal.h" /* ckc_gemm_emit_mfma / _emit_smem_load*/
#include "ckc/ir_internal.h" /* ckc_i_set_err                       */

/* ------------------------------------------------------------------ guards */

#define CKC_MOE_MEGA_DK_MAX_MFMAS 256
#define CKC_MOE_MEGA_DK_MAX_VECS 256
/* iter-args: accumulators (<= MAX_MFMAS) + B prefetch regs (<= MAX_VECS). */
#define CKC_MOE_MEGA_DK_MAX_ITER_ARGS (CKC_MOE_MEGA_DK_MAX_MFMAS + CKC_MOE_MEGA_DK_MAX_VECS)

/* ====================================================================== *
 *  _rowcol: per-thread (row, col) decode for B vec-load element `e`.
 *  Mirrors _MoeKloopPlan._rowcol (-> _vec_rowcol). The plan struct does not
 *  expose a _rowcol method, so we call the shared ckc_moe_vec_rowcol the same
 *  way the moe_gemm_fused k-loop core does.
 * ====================================================================== */
static void ckc_moe_mega_dk_rowcol(const ckc_moe_kloop_plan_t* plan,
                                   int e,
                                   ckc_value_t** out_row,
                                   ckc_value_t** out_col)
{
    ckc_moe_vec_rowcol(plan->b,
                       e,
                       plan->tid,
                       plan->c_threads,
                       plan->block_k_div_vec,
                       plan->c_load_vec,
                       plan->load_vec,
                       out_row,
                       out_col);
}

/* ====================================================================== *
 *  _emit_moe_down_mfma_phase_lds_a
 * ====================================================================== */

void ckc_moe_mega_emit_down_mfma_phase_lds_a(const ckc_moe_kloop_plan_t* plan,
                                             ckc_value_t* a_smem,
                                             ckc_value_t* a_col_base,
                                             const ckc_moe_operand_t* operand,
                                             ckc_value_t* const* accs,
                                             int num_accs,
                                             ckc_value_t* warp_m_idx,
                                             ckc_value_t* warp_n_idx,
                                             ckc_value_t* lane,
                                             int sched_groups,
                                             ckc_value_t** out_accs)
{
    ckc_ir_builder_t* b = plan->b;
    const ckc_gemm_tile_spec_t* t = &plan->u->tile;

    ckc_value_t* m_in_atom = ckc_b_mod(b, lane, ckc_b_const_i32(b, t->warp_tile_m));
    ckc_value_t* k_blk = ckc_b_div(b, lane, ckc_b_const_i32(b, t->warp_tile_m));
    ckc_value_t* n_in_atom = ckc_b_mod(b, lane, ckc_b_const_i32(b, t->warp_tile_n));
    ckc_value_t* warp_m_off
        = ckc_b_mul(b, warp_m_idx, ckc_b_const_i32(b, plan->mfmas_m * t->warp_tile_m));
    ckc_value_t* warp_n_off
        = ckc_b_mul(b, warp_n_idx, ckc_b_const_i32(b, plan->mfmas_n * t->warp_tile_n));

    /* new_accs = list(accs). */
    for(int i = 0; i < num_accs; ++i)
    {
        out_accs[i] = accs[i];
    }

    for(int kk = 0; kk < plan->k_atoms; ++kk)
    {
        /* B (single-buffered LDS) column base: per-tile-local. Python emits the
         * operands strictly left-to-right:
         *   b.add(b.mul(k_blk, const(b_per_lane)), const(kk*warp_tile_k))
         * C arg-evaluation order is unspecified, so sequence the nested calls so
         * the SSA value counter stays byte-identical. */
        ckc_value_t* b_col_mul = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, plan->b_per_lane));
        ckc_value_t* b_col_base = ckc_b_add(b, b_col_mul, ckc_b_const_i32(b, kk * t->warp_tile_k));
        /* A (persistent Hidden_smem) column base: add the K-tile origin k0.
         * Python: b.add(a_col_base,
         *                b.add(b.mul(k_blk, const(a_per_lane)),
         *                      const(kk*warp_tile_k))) */
        ckc_value_t* a_col_mul = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, plan->a_per_lane));
        ckc_value_t* a_col_inner = ckc_b_add(b, a_col_mul, ckc_b_const_i32(b, kk * t->warp_tile_k));
        ckc_value_t* a_col = ckc_b_add(b, a_col_base, a_col_inner);

        ckc_value_t* a_rows[CKC_MOE_MEGA_DK_MAX_MFMAS];
        for(int mi = 0; mi < plan->mfmas_m; ++mi)
        {
            ckc_value_t* a_row = ckc_b_add(
                b, warp_m_off, ckc_b_add(b, ckc_b_const_i32(b, mi * t->warp_tile_m), m_in_atom));
            a_rows[mi] = ckc_gemm_emit_smem_load(
                b, a_smem, a_row, a_col, plan->a_per_lane, plan->storage_dtype);
        }
        ckc_value_t* b_cols[CKC_MOE_MEGA_DK_MAX_MFMAS];
        for(int ni = 0; ni < plan->mfmas_n; ++ni)
        {
            ckc_value_t* b_row = ckc_b_add(
                b, warp_n_off, ckc_b_add(b, ckc_b_const_i32(b, ni * t->warp_tile_n), n_in_atom));
            b_cols[ni] = ckc_gemm_emit_smem_load(
                b, operand->smem, b_row, b_col_base, plan->b_per_lane, plan->storage_dtype);
        }
        int flat = 0;
        for(int mi = 0; mi < plan->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < plan->mfmas_n; ++ni)
            {
                out_accs[flat]
                    = ckc_gemm_emit_mfma(b, plan->u, a_rows[mi], b_cols[ni], out_accs[flat]);
                flat++;
            }
        }
        if(sched_groups
           && (strcmp(plan->u->trait.pipeline, "compv3") == 0
               || strcmp(plan->u->trait.pipeline, "compv4") == 0))
        {
            ckc_b_sched_group_barrier(b, 0x100, 1, 0);
            ckc_b_sched_group_barrier(b, 0x008, sched_groups, 0);
        }
    }
}

/* ====================================================================== *
 *  _emit_moe_down_kloop_lds_a
 *
 *  Software-prefetched down-GEMM k-loop reading A from persistent LDS. Only the
 *  B operand (W_down) is prefetched (global -> single LDS buffer). The inner
 *  closures _load_b_tile / _store_b_tile are inlined as static helpers below.
 * ====================================================================== */

/* _load_b_tile(k_off): coalesced global -> register load of one W_down K-tile.
 * `k_off` is the LDS-local contraction origin; the global read offsets it by
 * `b_k_base`. Writes plan->b_vecs_per_thread regs into `out_regs`. */
static void ckc_moe_mega_dk_load_b_tile(const ckc_moe_kloop_plan_t* plan,
                                        const ckc_moe_operand_t* operand,
                                        ckc_value_t* const b_mn_origin[2],
                                        ckc_value_t* b_k_base,
                                        ckc_value_t* k_off,
                                        ckc_value_t** out_regs)
{
    ckc_ir_builder_t* b = plan->b;
    ckc_value_t* b_origin[3] = {b_mn_origin[0], b_mn_origin[1], ckc_b_add(b, b_k_base, k_off)};
    int b_lengths[3] = {1, plan->block_n, plan->block_k};

    ckc_tile_window_t b_global;
    if(ckc_make_tile_window(&b_global, operand->global_view, b_lengths, b_origin, 3) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "_emit_moe_down_kloop_lds_a: B window");
        return;
    }
    for(int e = 0; e < plan->b_vecs_per_thread; ++e)
    {
        ckc_value_t* row = NULL;
        ckc_value_t* col = NULL;
        ckc_moe_mega_dk_rowcol(plan, e, &row, &col);
        /* Python builds the batch const inline at the load call site after
         * _rowcol; emit it here in the same order (no hoisted constant). */
        ckc_value_t* idx[3] = {ckc_b_const_i32(b, 0), row, col};
        if(plan->load_vec == 1)
        {
            out_regs[e] = ckc_tile_window_load_scalar(b, &b_global, idx, 3);
        }
        else
        {
            out_regs[e] = ckc_tile_window_load_vec(b, &b_global, idx, 3, plan->load_vec);
        }
    }
}

/* _store_b_tile(regs): store one prefetched W_down K-tile into the single LDS
 * buffer. */
static void ckc_moe_mega_dk_store_b_tile(const ckc_moe_kloop_plan_t* plan,
                                         const ckc_moe_operand_t* operand,
                                         ckc_value_t* const* regs)
{
    ckc_ir_builder_t* b = plan->b;
    ckc_value_t* z[2] = {ckc_b_const_i32(b, 0), ckc_b_const_i32(b, 0)};
    int b_lengths[2] = {plan->block_n, plan->block_k};

    ckc_tile_window_t b_lds;
    if(ckc_make_tile_window(&b_lds, operand->lds_view, b_lengths, z, 2) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "_emit_moe_down_kloop_lds_a: B lds window");
        return;
    }
    for(int e = 0; e < plan->b_vecs_per_thread; ++e)
    {
        ckc_value_t* row = NULL;
        ckc_value_t* col = NULL;
        ckc_moe_mega_dk_rowcol(plan, e, &row, &col);
        ckc_value_t* idx[2] = {row, col};
        if(plan->load_vec == 1 && operand->store_scalar_ok)
        {
            ckc_tile_window_store_scalar(b, &b_lds, idx, 2, (ckc_value_t*)regs[e], 0);
        }
        else
        {
            ckc_tile_window_store_vec(b, &b_lds, idx, 2, (ckc_value_t*)regs[e], plan->load_vec);
        }
    }
}

int ckc_moe_mega_emit_down_kloop_lds_a(const ckc_moe_kloop_plan_t* plan,
                                       ckc_value_t* a_smem_persistent,
                                       const ckc_moe_operand_t* operand,
                                       ckc_value_t* const* acc_inits,
                                       int num_accs,
                                       ckc_value_t* const b_mn_origin[2],
                                       ckc_value_t* b_k_base,
                                       ckc_value_t* K,
                                       ckc_value_t* warp_m_idx,
                                       ckc_value_t* warp_n_idx,
                                       ckc_value_t* lane,
                                       int sched_groups,
                                       ckc_value_t** out_accs)
{
    ckc_ir_builder_t* b = plan->b;
    ckc_value_t* c0 = ckc_b_const_i32(b, 0);
    ckc_value_t* c_block_k = ckc_b_const_i32(b, plan->block_k);

    int n_acc = num_accs;
    int n_b = plan->b_vecs_per_thread;

    /* Prefetch B tile 0 (A is already resident -> no A prefetch). */
    ckc_value_t* b_pre0[CKC_MOE_MEGA_DK_MAX_VECS];
    ckc_moe_mega_dk_load_b_tile(plan, operand, b_mn_origin, b_k_base, c0, b_pre0);

    /* carried = accs + b_pre0. */
    ckc_iter_arg_t iter_args[CKC_MOE_MEGA_DK_MAX_ITER_ARGS];
    /* 40 fits "down_acc_m" + two worst-case %d (11 chars each) + NUL, so the
     * emitted SSA names can never be truncated. */
    char names[CKC_MOE_MEGA_DK_MAX_ITER_ARGS][40];
    int n_ia = 0;
    /* Python carries the caller's down accumulator SSA names through
     * _emit_moe_down_kloop_lds_a (carried += [(name, v) for name, v in accs]).
     * The caller (STAGE 4/5) builds them as
     *   down_accs = [("down_acc_m{mi}_n{ni}", down_acc_init) ...]
     * with flat slot = mi * down_mfmas_n + ni. Reconstruct the same names from
     * the plan's MFMA geometry so the loop-carried phi names are byte-identical. */
    for(int mi = 0; mi < plan->mfmas_m; ++mi)
    {
        for(int ni = 0; ni < plan->mfmas_n; ++ni)
        {
            const int flat = mi * plan->mfmas_n + ni;
            if(flat >= n_acc)
            {
                continue;
            }
            snprintf(names[flat], sizeof(names[0]), "down_acc_m%d_n%d", mi, ni);
            iter_args[flat].name = names[flat];
            iter_args[flat].init = acc_inits[flat];
        }
    }
    n_ia = n_acc;
    for(int i = 0; i < n_b; ++i)
    {
        snprintf(names[n_ia], sizeof(names[0]), "bd_pre%d", i);
        iter_args[n_ia].name = names[n_ia];
        iter_args[n_ia].init = b_pre0[i];
        n_ia++;
    }

    ckc_for_t for_op = ckc_b_scf_for_iter(b, c0, K, c_block_k, iter_args, n_ia, "dk0", false, true);
    ckc_b_region_enter(b, for_op.body);
    {
        ckc_value_t* k0 = for_op.iv;
        ckc_value_t** iv = for_op.iter_vars;

        ckc_value_t* cur_accs[CKC_MOE_MEGA_DK_MAX_MFMAS];
        for(int j = 0; j < n_acc; ++j)
        {
            cur_accs[j] = iv[j];
        }
        ckc_value_t* b_regs[CKC_MOE_MEGA_DK_MAX_VECS];
        for(int i = 0; i < n_b; ++i)
        {
            b_regs[i] = iv[n_acc + i];
        }

        /* Store the prefetched B tile into the single LDS buffer (no A store). */
        ckc_moe_mega_dk_store_b_tile(plan, operand, b_regs);
        ckc_b_sync(b);
        /* Issue the NEXT B tile's clamped global loads (overlap the MFMAs). */
        ckc_value_t* k_next = ckc_b_add(b, k0, c_block_k);
        ckc_value_t* k_clamped = ckc_b_select(b, ckc_b_cmp_lt(b, k_next, K), k_next, k0);
        ckc_value_t* b_next[CKC_MOE_MEGA_DK_MAX_VECS];
        ckc_moe_mega_dk_load_b_tile(plan, operand, b_mn_origin, b_k_base, k_clamped, b_next);

        /* MFMA: A from persistent Hidden_smem at column k0; B from LDS tile. */
        ckc_value_t* new_accs[CKC_MOE_MEGA_DK_MAX_MFMAS];
        ckc_moe_mega_emit_down_mfma_phase_lds_a(plan,
                                                a_smem_persistent,
                                                k0,
                                                operand,
                                                cur_accs,
                                                n_acc,
                                                warp_m_idx,
                                                warp_n_idx,
                                                lane,
                                                sched_groups,
                                                new_accs);
        ckc_b_sync(b);

        ckc_value_t* yielded[CKC_MOE_MEGA_DK_MAX_ITER_ARGS];
        int ny = 0;
        for(int j = 0; j < n_acc; ++j)
        {
            yielded[ny++] = new_accs[j];
        }
        for(int i = 0; i < n_b; ++i)
        {
            yielded[ny++] = b_next[i];
        }
        ckc_b_scf_yield(b, yielded, ny);
    }
    ckc_b_region_leave(b);

    /* results[0:n_acc] are the final accumulators. */
    for(int j = 0; j < n_acc; ++j)
    {
        out_accs[j] = (for_op.op != NULL) ? for_op.op->results[j] : NULL;
    }
    return 1;
}
