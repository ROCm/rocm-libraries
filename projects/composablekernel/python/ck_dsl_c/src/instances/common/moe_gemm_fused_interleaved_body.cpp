// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_gemm_fused_interleaved-body.c -- C99 port of the INTERLEAVED
 * GATE+UP+SILU (single-B) builder body from
 * ck_dsl/instances/common/moe_gemm_fused.py
 * (build_moe_interleaved_gate_up_silu_gemm, lines 1144-1391 + _load_wgateup,
 * lines 1308-1322 + the compute closure emit_compute_and_epilogue,
 * lines 1351-1384).
 *
 * Scope (this TU): the three interleaved-family phase functions declared in
 * ckc/instance_moe_gemm_fused_internal.h:
 *   ckc_moe_interleaved_build_ctx_init   (prologue 1156-1349)
 *   ckc_moe_interleaved_load_wgateup     (preshuffle B-load closure 1308-1322)
 *   ckc_moe_interleaved_emit_compute     (compute+epilogue 1351-1384)
 *
 * Peers (epilogue staging, k-loop core, leaf helpers, spec/validate) are called
 * through the internal + helper + gemm-internal headers and may be unresolved at
 * -fsyntax-only time. Byte-faithful builder-call order against the Python span.
 */
#include "ckc/instance_moe_gemm_fused_internal.h"

#include <stdio.h>
#include <string.h>

#include "ckc/ir_internal.h"            /* ckc_i_set_err                       */
#include "ckc/instance_gemm_internal.h" /* ckc_gemm_emit_zero_acc              */

/* ------------------------------------------------------------------ guards */
/* Per-thread temporaries; CKC_MOE_MAX_ACCS already covers the acc group. */

/* ---------------------------------------------------------------- file helpers
 *
 * _storage_dtype(u) and _mfma_atom_widths(u)[2]. The value-type helper TU keeps
 * these static, so the body re-derives them locally (identical computation). */

static const ckc_type_t* ckc_moei_storage_dtype(const ckc_gemm_universal_spec_t* u)
{
    const char* d = u->data.dtype_a;
    if(d == NULL)
    {
        return ckc_f16();
    }
    if(strcmp(d, "f16") == 0 || strcmp(d, "fp16") == 0)
    {
        return ckc_f16();
    }
    if(strcmp(d, "bf16") == 0)
    {
        return ckc_bf16();
    }
    return ckc_scalar_by_name(d);
}

static int ckc_moei_c_per_lane(const ckc_gemm_universal_spec_t* u)
{
    const ckc_gemm_tile_spec_t* t = &u->tile;
    int wm                        = t->warp_tile_m;
    int wn                        = t->warp_tile_n;
    int wave                      = u->wave_size;
    return (wm * wn) / wave;
}

/* ====================================================================== *
 *  ckc_moe_interleaved_load_wgateup -- _load_wgateup (lines 1308-1322)
 * ====================================================================== *
 *
 * The preshuffle per-element B load override. `user` is the build ctx; matches
 * ckc_moe_load_b_fn. row/col are unused (the preshuffle path computes its own
 * contiguous per-tile offset). */
ckc_value_t* ckc_moe_interleaved_load_wgateup(
    ckc_ir_builder_t* bb, int e, ckc_value_t* k_off, ckc_value_t* row, ckc_value_t* col, void* user)
{
    ckc_moe_interleaved_build_ctx_t* ctx = (ckc_moe_interleaved_build_ctx_t*)user;
    (void)row;
    (void)col;

    int block_n  = ctx->block_n;
    int block_k  = ctx->block_k;
    int load_vec = ctx->plan.load_vec;

    /* n_tile_idx = block_n_off / c_block_n */
    ckc_value_t* n_tile_idx = ckc_b_div(bb, ctx->block_n_off, ctx->c_block_n);
    /* k_tile_idx = k_off / c_block_k */
    ckc_value_t* k_tile_idx = ckc_b_div(bb, k_off, ctx->c_block_k);
    /* two_n = N * 2 */
    ckc_value_t* two_n = ckc_b_mul(bb, ctx->N, ckc_b_const_i32(bb, 2));
    /* n_tile_count = two_n / c_block_n */
    ckc_value_t* n_tile_count = ckc_b_div(bb, two_n, ctx->c_block_n);
    /* tile_offset_elements = (k_tile_idx*n_tile_count + n_tile_idx)*(block_n*block_k) */
    ckc_value_t* tile_offset_elements =
        ckc_b_mul(bb,
                  ckc_b_add(bb, ckc_b_mul(bb, k_tile_idx, n_tile_count), n_tile_idx),
                  ckc_b_const_i32(bb, block_n * block_k));
    /* base_off = batch_off_b + tile_offset_elements */
    ckc_value_t* base_off = ckc_b_add(bb, ctx->batch_off_b, tile_offset_elements);
    /* vec_idx = e*c_threads + tid */
    ckc_value_t* vec_idx =
        ckc_b_add(bb, ckc_b_mul(bb, ckc_b_const_i32(bb, e), ctx->plan.c_threads), ctx->tid);
    /* glob_off = base_off + vec_idx*c_load_vec */
    ckc_value_t* glob_off = ckc_b_add(bb, base_off, ckc_b_mul(bb, vec_idx, ctx->plan.c_load_vec));
    if(load_vec == 1)
    {
        return ckc_b_global_load(bb, ctx->WGateUp, glob_off, ctx->storage_dtype, 0);
    }
    return ckc_b_global_load_vN(bb, ctx->WGateUp, glob_off, ctx->storage_dtype, load_vec, 0);
}

/* ====================================================================== *
 *  ckc_moe_interleaved_build_ctx_init -- prologue (lines 1156-1349)
 * ====================================================================== */
bool ckc_moe_interleaved_build_ctx_init(ckc_moe_interleaved_build_ctx_t* ctx,
                                        ckc_ir_builder_t* b,
                                        const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
                                        const char* arch)
{
    if(ctx == NULL || b == NULL || spec == NULL)
    {
        if(b != NULL)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved ctx_init: null arg");
        }
        return false;
    }
    memset(ctx, 0, sizeof(*ctx));
    ctx->b    = b;
    ctx->spec = spec;
    ctx->arch = (arch != NULL) ? arch : "gfx950";

    /* u = spec.to_universal_spec() */
    ctx->u = ckc_moe_interleaved_gate_up_silu_gemm_spec_to_universal(spec);
    const ckc_gemm_universal_spec_t* u = &ctx->u;

    /* ok, why = is_valid_gemm_spec(u, arch); raise on reject */
    {
        char why[256];
        if(!ckc_gemm_universal_is_valid_spec(u, ctx->arch, why, sizeof(why)))
        {
            char msg[320];
            snprintf(msg, sizeof(msg), "invalid interleaved gate/up GEMM spec: %s", why);
            ckc_i_set_err(b, CKC_ERR_VALUE, msg);
            return false;
        }
    }

    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    ctx->storage_dtype = ckc_moei_storage_dtype(u);
    {
        ckc_kernel_def_t* k = ckc_ir_builder_kernel(b);
        if(k != NULL)
        {
            ckc_attr_set_int(b, &k->attrs, "max_workgroup_size", spec->block_size);
        }
    }

    const ckc_type_t* sd = ctx->storage_dtype;

    /* ---- params ---- */
    const ckc_type_t* p_ptr_global = ckc_ptr_type(b, sd, "global");

    ckc_param_opts_t op;

    /* A (noalias readonly align16) */
    memset(&op, 0, sizeof(op));
    op.noalias      = true;
    op.noalias_set  = true;
    op.readonly     = true;
    op.readonly_set = true;
    op.align        = 16;
    op.align_set    = true;
    ctx->A          = ckc_b_param(b, "A", p_ptr_global, &op);

    /* WGateUp (noalias readonly align16) */
    ctx->WGateUp = ckc_b_param(b, "WGateUp", p_ptr_global, &op);

    /* Hidden (noalias writeonly align16) */
    memset(&op, 0, sizeof(op));
    op.noalias       = true;
    op.noalias_set   = true;
    op.writeonly     = true;
    op.writeonly_set = true;
    op.align         = 16;
    op.align_set     = true;
    ctx->Hidden      = ckc_b_param(b, "Hidden", p_ptr_global, &op);

    /* M, N, K, stride_a, stride_b, stride_c (I32) */
    ctx->M        = ckc_b_param(b, "M", ckc_i32(), NULL);
    ctx->N        = ckc_b_param(b, "N", ckc_i32(), NULL);
    ctx->K        = ckc_b_param(b, "K", ckc_i32(), NULL);
    ctx->stride_a = ckc_b_param(b, "stride_a", ckc_i32(), NULL);
    ctx->stride_b = ckc_b_param(b, "stride_b", ckc_i32(), NULL);
    ctx->stride_c = ckc_b_param(b, "stride_c", ckc_i32(), NULL);

    /* grouped = bool(getattr(spec, "grouped", False)) */
    ctx->grouped = spec->grouped ? true : false;
    if(ctx->grouped)
    {
        /* BlockExpertIds (PtrType(I32, "global") noalias readonly align4) */
        const ckc_type_t* p_i32_global = ckc_ptr_type(b, ckc_i32(), "global");
        memset(&op, 0, sizeof(op));
        op.noalias            = true;
        op.noalias_set        = true;
        op.readonly           = true;
        op.readonly_set       = true;
        op.align              = 4;
        op.align_set          = true;
        ctx->block_expert_ids = ckc_b_param(b, "BlockExpertIds", p_i32_global, &op);
    }

    /* active_tile_skip = u.trait.active_tile_skip && !grouped */
    ctx->active_tile_skip = (u->trait.active_tile_skip && !ctx->grouped) ? true : false;
    if(ctx->active_tile_skip)
    {
        const ckc_type_t* p_i32_global = ckc_ptr_type(b, ckc_i32(), "global");
        memset(&op, 0, sizeof(op));
        op.noalias            = true;
        op.noalias_set        = true;
        op.readonly           = true;
        op.readonly_set       = true;
        op.align              = 4;
        op.align_set          = true;
        ctx->sorted_token_ids = ckc_b_param(b, "SortedTokenIds", p_i32_global, &op);
        ctx->slot_size_p      = ckc_b_param(b, "slot_size", ckc_i32(), NULL);
    }

    ctx->preshuffle_b = u->trait.preshuffle_b ? true : false;

    /* ---- geometry ---- */
    const ckc_gemm_tile_spec_t* t = &u->tile;
    ctx->c_per_lane               = ckc_moei_c_per_lane(u);
    ctx->block_m                  = t->tile_m;
    ctx->block_n                  = t->tile_n;
    ctx->block_k                  = t->tile_k;

    /* if block_n % 2: raise ValueError("interleaved gate/up requires even tile_n") */
    if(ctx->block_n % 2)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved gate/up requires even tile_n");
        return false;
    }

    /* ---- SSA constants ---- */
    ctx->c0        = ckc_b_const_i32(b, 0);
    ctx->c_wave    = ckc_b_const_i32(b, spec->wave_size);
    ctx->c_warps_n = ckc_b_const_i32(b, t->warp_n);
    ctx->c_block_m = ckc_b_const_i32(b, ctx->block_m);
    ctx->c_block_n = ckc_b_const_i32(b, ctx->block_n);
    ctx->c_block_k = ckc_b_const_i32(b, ctx->block_k);

    /* ---- decomposition ---- */
    ctx->tid        = ckc_b_thread_id_x(b);
    ctx->warp_id    = ckc_b_div(b, ctx->tid, ctx->c_wave);
    ctx->warp_m_idx = ckc_b_div(b, ctx->warp_id, ctx->c_warps_n);
    ctx->warp_n_idx = ckc_b_mod(b, ctx->warp_id, ctx->c_warps_n);
    ctx->lane       = ckc_b_mod(b, ctx->tid, ctx->c_wave);

    /* ---- batched-vs-grouped dispatch ---- */
    if(ctx->grouped)
    {
        ckc_value_t* m_block_idx = ckc_b_block_id_y(b);
        ctx->expert_idx          = ckc_b_global_load_i32(b, ctx->block_expert_ids, m_block_idx, 0);
        ctx->batch_off_a         = ctx->c0;
        /* b_base_bytes = expert_i64 * stride_b_i64 * elem_bytes_b(=2)
         *
         * Python: b.mul(b.sext(stride_b,I64), ...) order -- the source is
         *   stride_b_i64 = b.sext(stride_b, I64)
         *   expert_i64   = b.sext(expert_idx, I64)
         *   b_base_bytes = b.mul(b.mul(expert_i64, stride_b_i64), elem_bytes_b)
         * so sext(stride_b) is emitted FIRST, then sext(expert_idx). Keep that
         * exact statement order here (it is the interleaved Python span, which
         * differs from the gate-up/down span that sexts expert first). */
        ckc_value_t* elem_bytes_b = ckc_b_const_i64(b, 2);
        ckc_value_t* stride_b_i64 = ckc_b_sext(b, ctx->stride_b, ckc_i64());
        ckc_value_t* expert_i64   = ckc_b_sext(b, ctx->expert_idx, ckc_i64());
        ckc_value_t* b_base_bytes =
            ckc_b_mul(b, ckc_b_mul(b, expert_i64, stride_b_i64), elem_bytes_b);
        ctx->WGateUp     = ckc_b_global_ptr_add(b, ctx->WGateUp, b_base_bytes);
        ctx->batch_off_b = ctx->c0;
        ctx->batch_off_c = ctx->c0;
        ctx->block_m_off = ckc_b_mul(b, m_block_idx, ctx->c_block_m);
    }
    else
    {
        ckc_value_t* batch_idx = ckc_b_block_id_z(b);
        ctx->batch_off_a       = ckc_b_mul(b, batch_idx, ctx->stride_a);
        ctx->batch_off_b       = ckc_b_mul(b, batch_idx, ctx->stride_b);
        ctx->batch_off_c       = ckc_b_mul(b, batch_idx, ctx->stride_c);
        ctx->block_m_off       = ckc_b_mul(b, ckc_b_block_id_y(b), ctx->c_block_m);
    }
    ctx->block_n_off = ckc_b_mul(b, ckc_b_block_id_x(b), ctx->c_block_n);

    /* ---- smem ---- */
    {
        int a_shape[2] = {ctx->block_m, ctx->block_k};
        int b_shape[2] = {ctx->block_n, ctx->block_k};
        int c_shape[2] = {ctx->block_m, ctx->block_n};
        ctx->A_smem    = ckc_b_smem_alloc(b, sd, a_shape, 2, "A_smem");
        ctx->B_smem    = ckc_b_smem_alloc(b, sd, b_shape, 2, "B_smem");
        ctx->C_smem    = ckc_b_smem_alloc(b, sd, c_shape, 2, "GateUp_smem");
    }

    /* ---- accumulators: single zero-acc group ----
     * accs = [(f"gu_acc_m{mi}_n{ni}", acc_init) for mi in mfmas_m for ni in mfmas_n] */
    ctx->mfmas_m  = ckc_gemm_tile_mfmas_per_warp_m(t);
    ctx->mfmas_n  = ckc_gemm_tile_mfmas_per_warp_n(t);
    ctx->acc_init = ckc_gemm_emit_zero_acc(b, u);
    ctx->num_accs = ctx->mfmas_m * ctx->mfmas_n;
    if(ctx->num_accs > CKC_MOE_MAX_ACCS)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved ctx_init: acc group exceeds CKC_MOE_MAX_ACCS");
        return false;
    }
    {
        int idx = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                /* names mirror Python f"gu_acc_m{mi}_n{ni}"; stored for the
                 * iter-arg labels the k-loop core consumes. */
                char nm[32];
                snprintf(nm, sizeof(nm), "gu_acc_m%d_n%d", mi, ni);
                ctx->acc_names[idx] = ckc_arena_strdup(&b->arena, nm);
                ctx->acc_inits[idx] = ctx->acc_init;
                ++idx;
            }
        }
    }

    /* ---- global + LDS views ----
     * a_view = make_global_view(A, shape=(1,1,1), dtype, strides=(1,K,1))
     * b_view = make_global_view(WGateUp, shape=(1,1,1), dtype, strides=(1,K,1)) */
    {
        int shape3[3] = {1, 1, 1};
        ckc_stride_t st[3];
        st[0] = ckc_stride_imm(1);
        st[1] = ckc_stride_value(ctx->K);
        st[2] = ckc_stride_imm(1);
        if(ckc_make_global_view(&ctx->a_view, ctx->A, shape3, 3, sd, st) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved ctx_init: a_view");
            return false;
        }
        if(ckc_make_global_view(&ctx->b_view, ctx->WGateUp, shape3, 3, sd, st) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved ctx_init: b_view");
            return false;
        }
    }

    /* a_lds_view = TensorView(base=A_smem, desc=packed((block_m, block_k)), lds)
     * b_lds_view = TensorView(base=B_smem, desc=packed((block_n, block_k)), lds) */
    {
        int a_lds_shape[2]         = {ctx->block_m, ctx->block_k};
        int b_lds_shape[2]         = {ctx->block_n, ctx->block_k};
        ctx->a_lds_view.base       = ctx->A_smem;
        ctx->a_lds_view.addr_space = CKC_ADDR_LDS;
        if(ckc_tensor_descriptor_packed(&ctx->a_lds_view.desc, a_lds_shape, 2, sd) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved ctx_init: a_lds_view");
            return false;
        }
        ctx->b_lds_view.base       = ctx->B_smem;
        ctx->b_lds_view.addr_space = CKC_ADDR_LDS;
        if(ckc_tensor_descriptor_packed(&ctx->b_lds_view.desc, b_lds_shape, 2, sd) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "interleaved ctx_init: b_lds_view");
            return false;
        }
    }

    /* plan = _MoeKloopPlan(b, u, tid) */
    if(!ckc_moe_kloop_plan_init(&ctx->plan, b, u, ctx->tid))
    {
        return false;
    }

    /* ---- single-B operand: preshuffle override OR canonical ---- */
    memset(&ctx->operand, 0, sizeof(ctx->operand));
    if(ctx->preshuffle_b)
    {
        ctx->operand.global_view     = &ctx->b_view;
        ctx->operand.lds_view        = &ctx->b_lds_view;
        ctx->operand.smem            = ctx->B_smem;
        ctx->operand.load_b          = ckc_moe_interleaved_load_wgateup;
        ctx->operand.load_b_user     = ctx;
        ctx->operand.store_scalar_ok = false;
    }
    else
    {
        ctx->operand.global_view     = &ctx->b_view;
        ctx->operand.lds_view        = &ctx->b_lds_view;
        ctx->operand.smem            = ctx->B_smem;
        ctx->operand.load_b          = NULL;
        ctx->operand.load_b_user     = NULL;
        ctx->operand.store_scalar_ok = false;
    }

    /* a_mn_origin = (batch_off_a, block_m_off); b_mn_origin = (batch_off_b, block_n_off) */
    ctx->a_mn_origin[0] = ctx->batch_off_a;
    ctx->a_mn_origin[1] = ctx->block_m_off;
    ctx->b_mn_origin[0] = ctx->batch_off_b;
    ctx->b_mn_origin[1] = ctx->block_n_off;

    /* ---- active-tile gate -> do_work_cond ---- */
    ctx->do_work_cond = NULL;
    if(ctx->grouped)
    {
        /* expert_idx >= 0 */
        ctx->do_work_cond = ckc_b_cmp_ge(b, ctx->expert_idx, ctx->c0);
    }
    else if(ctx->active_tile_skip)
    {
        ckc_value_t* bucket_head =
            ckc_b_add(b, ckc_b_mul(b, ckc_b_block_id_z(b), ctx->slot_size_p), ctx->block_m_off);
        ckc_value_t* first_token = ckc_b_global_load_i32(b, ctx->sorted_token_ids, bucket_head, 0);
        ctx->do_work_cond        = ckc_b_cmp_ge(b, first_token, ctx->c0);
    }

    return true;
}

/* ====================================================================== *
 *  ckc_moe_interleaved_emit_compute -- emit_compute_and_epilogue (1351-1384)
 * ====================================================================== *
 *
 * The single-B software-prefetched MFMA k-loop into acc_res, then the
 * interleaved silu epilogue. The Python wraps the call in the optional
 * `with b.scf_if(do_work_cond)` gate; that gating is the caller's
 * responsibility (do_work_cond is exposed on the ctx) -- this matches the
 * other family compute closures which run unconditionally. */
void ckc_moe_interleaved_emit_compute(ckc_moe_interleaved_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* (acc_res,) = _emit_moe_prefetch_kloop(plan, a_view, a_lds_view, A_smem,
     *     a_mn_origin, [operand], b_mn_origin, [accs], K, warp_m_idx,
     *     warp_n_idx, lane, sched_groups=0) */
    int group_sizes[1] = {ctx->num_accs};
    if(!ckc_moe_emit_prefetch_kloop(&ctx->plan,
                                    &ctx->a_view,
                                    &ctx->a_lds_view,
                                    ctx->A_smem,
                                    ctx->a_mn_origin,
                                    &ctx->operand,
                                    1,
                                    ctx->b_mn_origin,
                                    ctx->acc_inits,
                                    (const char* const*)ctx->acc_names, /* "gu_acc_m*_n*" */
                                    group_sizes,
                                    ctx->K,
                                    ctx->warp_m_idx,
                                    ctx->warp_n_idx,
                                    ctx->lane,
                                    0,
                                    ctx->acc_res))
    {
        return;
    }

    /* _emit_interleaved_silu_epilogue(b, u, acc_res, C_smem, warp_m_idx,
     *     warp_n_idx, lane, block_m_off, block_n_off, M, N, Hidden,
     *     c_per_lane, batch_off_c=batch_off_c) */
    ckc_moe_emit_interleaved_silu_epilogue(b,
                                           &ctx->u,
                                           ctx->acc_res,
                                           ctx->num_accs,
                                           ctx->C_smem,
                                           ctx->warp_m_idx,
                                           ctx->warp_n_idx,
                                           ctx->lane,
                                           ctx->block_m_off,
                                           ctx->block_n_off,
                                           ctx->M,
                                           ctx->N,
                                           ctx->Hidden,
                                           ctx->c_per_lane,
                                           ctx->batch_off_c);
}
