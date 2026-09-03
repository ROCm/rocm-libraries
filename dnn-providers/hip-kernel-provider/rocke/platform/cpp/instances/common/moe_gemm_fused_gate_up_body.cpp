// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_gemm_fused_gate-up-body.c -- C99 port of the GATE+UP+SILU
 * (dual-B) builder body of rocke/instances/common/moe_gemm_fused.py.
 *
 * SCOPE (this translation unit only):
 *   rocke_moe_gate_up_build_ctx_init   <- build_moe_gate_up_silu_gemm prologue
 *                                       (Python lines 723-866)
 *   rocke_moe_gate_up_emit_compute     <- the _emit_gate_up_compute closure
 *                                       (Python lines 868-900)
 *
 * Both operate over rocke_moe_gate_up_build_ctx_t (the per-builder shared state
 * defined in instance_moe_gemm_fused_internal.h). Builder-call order is
 * byte-faithful to the Python prologue / closure.
 *
 * Peer phases (the silu epilogue rocke_moe_emit_gate_up_silu_epilogue_default,
 * the driver rocke_build_moe_gate_up_silu_gemm, the interleaved/down families)
 * live in sibling .c files and are reached only through the internal header.
 */
#include "rocke/instance_moe_gemm_fused_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom                 */
#include "rocke/helper_rocke.instances.common.moe_gemm_fused.h"
#include "rocke/instance_gemm_internal.h" /* rocke_gemm_emit_zero_acc            */
#include "rocke/ir_internal.h" /* rocke_i_set_err                     */
/* rocke_moe_storage_dtype / rocke_moe_mfma_atom_widths */

/* Build a 2D packed LDS TensorView over `smem` of (d0, d1) elements. Mirrors
 * the Python TensorView(base=smem, desc=TensorDescriptor.packed((d0,d1),
 * dtype), addr_space="lds"). No IR is emitted. Returns false on a descriptor
 * rank error (sticky set on `b`). */
static bool rocke_moe_make_lds_view2(rocke_ir_builder_t* b,
                                     rocke_tensor_view_t* out,
                                     rocke_value_t* smem,
                                     int d0,
                                     int d1,
                                     const rocke_type_t* dtype)
{
    int shape[2];
    shape[0] = d0;
    shape[1] = d1;
    if(rocke_tensor_descriptor_packed(&out->desc, shape, 2, dtype) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_moe_gate_up_silu_gemm: LDS view");
        return false;
    }
    out->base = smem;
    out->addr_space = ROCKE_ADDR_LDS;
    return true;
}

/* ====================================================================== *
 *  rocke_moe_gate_up_build_ctx_init  (Python lines 723-866)
 * ====================================================================== */
bool rocke_moe_gate_up_build_ctx_init(rocke_moe_gate_up_build_ctx_t* ctx,
                                      rocke_ir_builder_t* b,
                                      const rocke_moe_gate_up_silu_gemm_spec_t* spec,
                                      const char* arch)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->b = b;
    ctx->spec = spec;
    ctx->arch = (arch != NULL) ? arch : "gfx950";

    /* u = spec.to_universal_spec(); ok, why = is_valid_gemm_spec(u, arch=arch) */
    ctx->u = rocke_moe_gate_up_silu_gemm_spec_to_universal(spec);
    {
        char why[ROCKE_ERR_MSG_CAP];
        why[0] = '\0';
        if(!rocke_gemm_universal_is_valid_spec(&ctx->u, ctx->arch, why, sizeof(why)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fused gate+up GEMM spec: %s", why);
            return false;
        }
    }

    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size
     * if spec.trait.waves_per_eu is not None:
     *     b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    if(spec->trait.waves_per_eu_set)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->trait.waves_per_eu);
    }

    ctx->storage_dtype = rocke_moe_storage_dtype(&ctx->u);
    if(ctx->storage_dtype == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_moe_gate_up_silu_gemm: storage dtype");
        return false;
    }

    /* ---- kernel params (Values) ---- */
    {
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_storage = rocke_ptr_type(b, ctx->storage_dtype, "global");

        /* A / WGate / WUp: noalias, readonly, align=16 */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->A = rocke_b_param(b, "A", ptr_storage, &opts);
        ctx->WGate = rocke_b_param(b, "WGate", ptr_storage, &opts);
        ctx->WUp = rocke_b_param(b, "WUp", ptr_storage, &opts);

        /* Hidden: noalias, writeonly, align=16 */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->Hidden = rocke_b_param(b, "Hidden", ptr_storage, &opts);

        ctx->M = rocke_b_param(b, "M", rocke_i32(), NULL);
        ctx->N = rocke_b_param(b, "N", rocke_i32(), NULL);
        ctx->K = rocke_b_param(b, "K", rocke_i32(), NULL);
        ctx->stride_a = rocke_b_param(b, "stride_a", rocke_i32(), NULL);
        ctx->stride_b = rocke_b_param(b, "stride_b", rocke_i32(), NULL);
        ctx->stride_c = rocke_b_param(b, "stride_c", rocke_i32(), NULL);

        ctx->grouped = spec->grouped;
        if(ctx->grouped)
        {
            rocke_param_opts_t gopts;
            memset(&gopts, 0, sizeof(gopts));
            gopts.noalias = true;
            gopts.noalias_set = true;
            gopts.readonly = true;
            gopts.readonly_set = true;
            gopts.align = 4;
            gopts.align_set = true;
            ctx->block_expert_ids = rocke_b_param(
                b, "BlockExpertIds", rocke_ptr_type(b, rocke_i32(), "global"), &gopts);
        }
    }

    /* t = spec.tile; _, _, c_per_lane = _mfma_atom_widths(u) */
    {
        const rocke_gemm_tile_spec_t* t = &ctx->u.tile;
        int a_per = 0, b_per = 0, c_per = 0;
        rocke_moe_mfma_atom_widths(&ctx->u, &a_per, &b_per, &c_per);
        ctx->c_per_lane = c_per;

        ctx->block_m = t->tile_m;
        ctx->block_n = t->tile_n;
        ctx->block_k = t->tile_k;

        ctx->mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
        ctx->mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);

        /* c_wave/c_warps_n/c_block_m/c_block_n/c0 */
        ctx->c_wave = rocke_b_const_i32(b, spec->wave_size);
        ctx->c_warps_n = rocke_b_const_i32(b, t->warp_n);
        ctx->c_block_m = rocke_b_const_i32(b, ctx->block_m);
        ctx->c_block_n = rocke_b_const_i32(b, ctx->block_n);
        ctx->c0 = rocke_b_const_i32(b, 0);
    }

    /* tid / warp / lane decomposition */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->warp_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->warp_m_idx = rocke_b_div(b, ctx->warp_id, ctx->c_warps_n);
    ctx->warp_n_idx = rocke_b_mod(b, ctx->warp_id, ctx->c_warps_n);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);

    /* batched-vs-grouped tile origins */
    if(ctx->grouped)
    {
        rocke_value_t* m_block_idx = rocke_b_block_id_y(b);
        ctx->expert_idx = rocke_b_global_load_i32(b, ctx->block_expert_ids, m_block_idx, 0);

        /* b_base_bytes = (sext(expert_idx,i64) * sext(stride_b,i64)) * 2 (f16/bf16)
         *
         * Python evaluates the mul operands left-to-right, so sext(expert_idx)
         * is emitted BEFORE sext(stride_b). C function-argument evaluation order
         * is unspecified, so the two sexts MUST be sequenced into locals (expert
         * first) to keep the emitted SSA order byte-identical to Python. */
        rocke_value_t* elem_bytes_b = rocke_b_const_i64(b, 2);
        rocke_value_t* expert_i64 = rocke_b_sext(b, ctx->expert_idx, rocke_i64());
        rocke_value_t* stride_b_i64 = rocke_b_sext(b, ctx->stride_b, rocke_i64());
        rocke_value_t* b_base_bytes
            = rocke_b_mul(b, rocke_b_mul(b, expert_i64, stride_b_i64), elem_bytes_b);
        ctx->WGate = rocke_b_global_ptr_add(b, ctx->WGate, b_base_bytes);
        ctx->WUp = rocke_b_global_ptr_add(b, ctx->WUp, b_base_bytes);

        ctx->batch_off_a = ctx->c0;
        ctx->batch_off_b = ctx->c0;
        ctx->batch_off_c = ctx->c0;
        ctx->block_m_off = rocke_b_mul(b, m_block_idx, ctx->c_block_m);
    }
    else
    {
        rocke_value_t* batch_idx = rocke_b_block_id_z(b);
        ctx->batch_off_a = rocke_b_mul(b, batch_idx, ctx->stride_a);
        ctx->batch_off_b = rocke_b_mul(b, batch_idx, ctx->stride_b);
        ctx->batch_off_c = rocke_b_mul(b, batch_idx, ctx->stride_c);
        ctx->block_m_off = rocke_b_mul(b, rocke_b_block_id_y(b), ctx->c_block_m);
    }
    ctx->block_n_off = rocke_b_mul(b, rocke_b_block_id_x(b), ctx->c_block_n);

    /* smem allocations: A_smem [block_m,block_k], Bg/Bu_smem [block_n,block_k] */
    {
        int a_shape[2];
        int b_shape[2];
        a_shape[0] = ctx->block_m;
        a_shape[1] = ctx->block_k;
        b_shape[0] = ctx->block_n;
        b_shape[1] = ctx->block_k;
        ctx->A_smem = rocke_b_smem_alloc(b, ctx->storage_dtype, a_shape, 2, "A_smem");
        ctx->Bg_smem = rocke_b_smem_alloc(b, ctx->storage_dtype, b_shape, 2, "Bg_smem");
        ctx->Bu_smem = rocke_b_smem_alloc(b, ctx->storage_dtype, b_shape, 2, "Bu_smem");
    }

    /* acc_init = _emit_zero_acc(b, u); gate_accs / up_accs iter-arg lists */
    ctx->acc_init = rocke_gemm_emit_zero_acc(b, &ctx->u);
    {
        int idx = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                if(idx >= ROCKE_MOE_MAX_ACCS)
                {
                    rocke_i_set_err(
                        b, ROCKE_ERR_VALUE, "build_moe_gate_up_silu_gemm: too many accumulators");
                    return false;
                }
                char nm[48];
                snprintf(nm, sizeof(nm), "gate_acc_m%d_n%d", mi, ni);
                ctx->gate_acc_names[idx] = rocke_arena_strdup(&b->arena, nm);
                ctx->gate_acc_inits[idx] = ctx->acc_init;
                snprintf(nm, sizeof(nm), "up_acc_m%d_n%d", mi, ni);
                ctx->up_acc_names[idx] = rocke_arena_strdup(&b->arena, nm);
                ctx->up_acc_inits[idx] = ctx->acc_init;
                ++idx;
            }
        }
        ctx->num_accs = idx; /* mfmas_m * mfmas_n */
    }

    /* 3D global views: make_global_view(P, shape=(1,1,1), dtype, strides=(1,K,1)) */
    {
        int gshape[3];
        rocke_stride_t gstr[3];
        gshape[0] = 1;
        gshape[1] = 1;
        gshape[2] = 1;
        gstr[0] = rocke_stride_imm(1);
        gstr[1] = rocke_stride_value(ctx->K);
        gstr[2] = rocke_stride_imm(1);
        if(rocke_make_global_view(&ctx->a_view, ctx->A, gshape, 3, ctx->storage_dtype, gstr)
               != ROCKE_OK
           || rocke_make_global_view(&ctx->wg_view, ctx->WGate, gshape, 3, ctx->storage_dtype, gstr)
                  != ROCKE_OK
           || rocke_make_global_view(&ctx->wu_view, ctx->WUp, gshape, 3, ctx->storage_dtype, gstr)
                  != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_moe_gate_up_silu_gemm: global view");
            return false;
        }
    }

    /* 2D packed LDS views over A_smem / Bg_smem / Bu_smem */
    if(!rocke_moe_make_lds_view2(
           b, &ctx->a_lds_view, ctx->A_smem, ctx->block_m, ctx->block_k, ctx->storage_dtype)
       || !rocke_moe_make_lds_view2(
           b, &ctx->bg_lds_view, ctx->Bg_smem, ctx->block_n, ctx->block_k, ctx->storage_dtype)
       || !rocke_moe_make_lds_view2(
           b, &ctx->bu_lds_view, ctx->Bu_smem, ctx->block_n, ctx->block_k, ctx->storage_dtype))
    {
        return false;
    }

    /* plan = _MoeKloopPlan(b, u, tid) */
    if(!rocke_moe_kloop_plan_init(&ctx->plan, b, &ctx->u, ctx->tid))
    {
        return false;
    }

    /* operands = [gate, up] */
    memset(&ctx->operands[0], 0, sizeof(ctx->operands[0]));
    ctx->operands[0].global_view = &ctx->wg_view;
    ctx->operands[0].lds_view = &ctx->bg_lds_view;
    ctx->operands[0].smem = ctx->Bg_smem;
    memset(&ctx->operands[1], 0, sizeof(ctx->operands[1]));
    ctx->operands[1].global_view = &ctx->wu_view;
    ctx->operands[1].lds_view = &ctx->bu_lds_view;
    ctx->operands[1].smem = ctx->Bu_smem;

    /* a_mn_origin = (batch_off_a, block_m_off); b_mn_origin = (batch_off_b, block_n_off) */
    ctx->a_mn_origin[0] = ctx->batch_off_a;
    ctx->a_mn_origin[1] = ctx->block_m_off;
    ctx->b_mn_origin[0] = ctx->batch_off_b;
    ctx->b_mn_origin[1] = ctx->block_n_off;

    return rocke_ir_builder_ok(b);
}

/* ====================================================================== *
 *  rocke_moe_gate_up_emit_compute  (Python _emit_gate_up_compute, lines 868-900)
 * ====================================================================== */
void rocke_moe_gate_up_emit_compute(rocke_moe_gate_up_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int n = ctx->num_accs; /* per-group accumulator count = mfmas_m*mfmas_n */

    /* gate_res, up_res = _emit_moe_prefetch_kloop(plan, a_view, a_lds_view,
     *     A_smem, a_mn_origin, operands, b_mn_origin, [gate_accs, up_accs], K,
     *     warp_m_idx, warp_n_idx, lane, sched_groups=2*mfmas_m*mfmas_n) */
    int group_sizes[2];
    group_sizes[0] = n;
    group_sizes[1] = n;

    /* acc_inits + acc_names flat, operand-then-flat order: gate group then up
     * group. The names carry the Python loop-carried phi labels
     * ("gate_acc_m*_n*" then "up_acc_m*_n*") so the emitted iter-arg SSA names
     * are byte-identical to Python. */
    rocke_value_t* acc_inits_flat[2 * ROCKE_MOE_MAX_ACCS];
    const char* acc_names_flat[2 * ROCKE_MOE_MAX_ACCS];
    rocke_value_t* out_flat[2 * ROCKE_MOE_MAX_ACCS];
    for(int i = 0; i < n; ++i)
    {
        acc_inits_flat[i] = ctx->gate_acc_inits[i];
        acc_inits_flat[n + i] = ctx->up_acc_inits[i];
        acc_names_flat[i] = ctx->gate_acc_names[i];
        acc_names_flat[n + i] = ctx->up_acc_names[i];
    }

    int sched_groups = 2 * ctx->mfmas_m * ctx->mfmas_n;

    if(!rocke_moe_emit_prefetch_kloop(&ctx->plan,
                                      &ctx->a_view,
                                      &ctx->a_lds_view,
                                      ctx->A_smem,
                                      ctx->a_mn_origin,
                                      ctx->operands,
                                      2,
                                      ctx->b_mn_origin,
                                      acc_inits_flat,
                                      acc_names_flat,
                                      group_sizes,
                                      ctx->K,
                                      ctx->warp_m_idx,
                                      ctx->warp_n_idx,
                                      ctx->lane,
                                      sched_groups,
                                      out_flat))
    {
        return;
    }

    for(int i = 0; i < n; ++i)
    {
        ctx->gate_res[i] = out_flat[i];
        ctx->up_res[i] = out_flat[n + i];
    }

    /* _emit_gate_up_silu_epilogue_default(b, u, gate_res, up_res, warp_m_idx,
     *     warp_n_idx, lane, block_m_off, block_n_off, M, N, Hidden,
     *     c_per_lane, batch_off_c=batch_off_c) */
    rocke_moe_emit_gate_up_silu_epilogue_default(b,
                                                 &ctx->u,
                                                 ctx->gate_res,
                                                 ctx->up_res,
                                                 n,
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
