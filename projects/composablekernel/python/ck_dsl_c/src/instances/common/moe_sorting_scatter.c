/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_moe_sorting_instance_moe_sorting_scatter.c.c -- chunked port of the
 * SCATTER kernel phase of build_moe_sort_scatter (ck_dsl/instances/common/
 * moe_sorting.py lines 447-540).
 *
 * Implements the two scatter phase functions declared in
 * ckc/instance_moe_sorting_internal.h:
 *   ckc_moe_sort_scatter_prologue  (lines 481-525)
 *   ckc_moe_sort_scatter_body      (lines 527-540)
 *
 * Peer phase functions / module helpers (decode_pair_token_topk,
 * decode_expert_load, is_valid_spec_impl, kernel_name, ...) live in their own
 * TUs and are reached via the internal header; this TU implements ONLY the
 * scatter scope.
 */

#include <stdio.h>
#include <string.h>

#include "ckc/instance_moe_sorting_internal.h"
#include "ckc/instance_moe_sorting.h"
#include "ckc/ir.h"

/* ===================================================================== *
 *  SCATTER PROLOGUE  (Python build_moe_sort_scatter, lines 481-525).
 *
 *  Python:
 *    ok, why = is_valid_spec(spec, arch)
 *    if not ok: raise ValueError(...)
 *    b = IRBuilder(spec.kernel_name("scatter"))           # builder pre-init'd
 *    b.kernel.attrs["max_workgroup_size"] = spec.block_size
 *    TopkIds        = b.param("TopkIds",  ptr<i32,global>, noalias,readonly,align=4)
 *    TopkWeights    = b.param("TopkWeights", ptr<f32,global>, noalias,readonly,align=4)
 *    Offsets        = b.param("Offsets", ptr<i32,global>, noalias,readonly,align=4)
 *    Counter        = b.param("Counter", ptr<i32,global>, align=4)
 *    SortedTokenIds = b.param("SortedTokenIds", ptr<i32,global>, writeonly,align=4)
 *    SortedTopkIds  = b.param("SortedTopkIds",  ptr<i32,global>, writeonly,align=4)
 *    SortedWeights  = b.param("SortedWeights",  ptr<f32,global>, writeonly,align=4)
 *    tokens         = b.param("tokens", I32)
 *    topk           = b.param("topk", I32)
 *    num_experts    = b.param("num_experts", I32)
 *    tid = b.thread_id_x(); bid = b.block_id_x()
 *    pair_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)
 *    t_idx, k_idx = _decode_pair_token_topk(b, pair_idx, spec.topk)
 *    num_pairs = b.mul(tokens, topk)
 *    in_bounds = b.cmp_lt(pair_idx, num_pairs)
 * ===================================================================== */
bool ckc_moe_sort_scatter_prologue(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    const ckc_moe_sorting_spec_t* spec;
    char reason[CKC_ERR_MSG_CAP];
    ckc_param_opts_t opts;

    if (ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
        return false;

    b = ctx->b;
    spec = ctx->spec;

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if (!ckc_moe_sort_is_valid_spec_impl(spec, ctx->arch, reason, sizeof(reason),
                                         NULL))
    {
        /* raise ValueError(f"invalid moe_sorting spec: {why}") */
        if (b->status == CKC_OK)
        {
            b->status = CKC_ERR_VALUE;
            CKC_ERR_SNPRINTF(b->err, CKC_ERR_MSG_CAP, "invalid moe_sorting spec: %s",
                     reason);
        }
        return false;
    }

    /* geometry scalars (Python: BS = spec.block_size; topk = spec.topk; E unused
     * by scatter body but mirrored for ctx consistency). */
    ctx->BS = spec->block_size;
    ctx->E = spec->experts;
    ctx->topk = spec->topk;

    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* ---- 10-entry ABI param block, in Python declaration order. ---- */

    /* TopkIds: ptr<i32,global>, noalias=True, readonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->TopkIds = ckc_b_param(b, "TopkIds", ckc_ptr_type(b, ckc_i32(), "global"),
                               &opts);

    /* TopkWeights: ptr<f32,global>, noalias=True, readonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->TopkWeights = ckc_b_param(
        b, "TopkWeights", ckc_ptr_type(b, ckc_f32(), "global"), &opts);

    /* Offsets: ptr<i32,global>, noalias=True, readonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->Offsets = ckc_b_param(b, "Offsets", ckc_ptr_type(b, ckc_i32(), "global"),
                               &opts);

    /* Counter: ptr<i32,global>, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.align = 4;
    opts.align_set = true;
    ctx->Counter = ckc_b_param(b, "Counter", ckc_ptr_type(b, ckc_i32(), "global"),
                               &opts);

    /* SortedTokenIds: ptr<i32,global>, writeonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->SortedTokenIds = ckc_b_param(
        b, "SortedTokenIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);

    /* SortedTopkIds: ptr<i32,global>, writeonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->SortedTopkIds = ckc_b_param(
        b, "SortedTopkIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);

    /* SortedWeights: ptr<f32,global>, writeonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->SortedWeights = ckc_b_param(
        b, "SortedWeights", ckc_ptr_type(b, ckc_f32(), "global"), &opts);

    /* tokens = b.param("tokens", I32)  # noqa: F841 - ABI */
    ctx->tokens = ckc_b_param(b, "tokens", ckc_i32(), NULL);
    /* topk = b.param("topk", I32) */
    ctx->topk_param = ckc_b_param(b, "topk", ckc_i32(), NULL);
    /* num_experts = b.param("num_experts", I32) */
    ctx->num_experts = ckc_b_param(b, "num_experts", ckc_i32(), NULL);

    /* tid = b.thread_id_x() */
    ctx->tid = ckc_b_thread_id_x(b);
    /* bid = b.block_id_x() */
    ctx->bid = ckc_b_block_id_x(b);
    /* pair_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid) */
    ctx->pair_idx = ckc_b_add(
        b, ckc_b_mul(b, ctx->bid, ckc_b_const_i32(b, spec->block_size)), ctx->tid);

    /* t_idx, k_idx = _decode_pair_token_topk(b, pair_idx, spec.topk) */
    ckc_moe_sort_decode_pair_token_topk(b, ctx->pair_idx, spec->topk,
                                        &ctx->t_idx, &ctx->k_idx);

    /* num_pairs = b.mul(tokens, topk) */
    ctx->num_pairs = ckc_b_mul(b, ctx->tokens, ctx->topk_param);
    /* in_bounds = b.cmp_lt(pair_idx, num_pairs) */
    ctx->in_bounds = ckc_b_cmp_lt(b, ctx->pair_idx, ctx->num_pairs);

    return b->status == CKC_OK;
}

/* ===================================================================== *
 *  SCATTER BODY + RETURN  (Python build_moe_sort_scatter, lines 527-540).
 *
 *  Python:
 *    with b.scf_if(in_bounds):
 *        eid, valid_e = _decode_expert_load(b, TopkIds, pair_idx, num_experts)
 *        with b.scf_if(valid_e):
 *            local_off  = b.global_atomic_add(Counter, eid, b.const_i32(1))
 *            base       = b.global_load_i32(Offsets, eid)
 *            global_off = b.add(base, local_off)
 *            w          = b.global_load_f32(TopkWeights, pair_idx)
 *            b.global_store(SortedTokenIds, global_off, t_idx, align=4)
 *            b.global_store(SortedTopkIds,  global_off, k_idx, align=4)
 *            b.global_store(SortedWeights,  global_off, w,     align=4)
 *    return b.kernel
 * ===================================================================== */
ckc_kernel_def_t* ckc_moe_sort_scatter_body(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b;
    ckc_if_t outer;

    if (ctx == NULL || ctx->b == NULL)
        return NULL;

    b = ctx->b;

    /* with b.scf_if(in_bounds): */
    outer = ckc_b_scf_if(b, ctx->in_bounds);
    ckc_b_region_enter(b, outer.then_region);
    {
        ckc_if_t inner;

        /* eid, valid_e = _decode_expert_load(b, TopkIds, pair_idx, num_experts) */
        ckc_moe_sort_decode_expert_load(b, ctx->TopkIds, ctx->pair_idx,
                                        ctx->num_experts, &ctx->eid,
                                        &ctx->valid_e);

        /* with b.scf_if(valid_e): */
        inner = ckc_b_scf_if(b, ctx->valid_e);
        ckc_b_region_enter(b, inner.then_region);
        {
            ckc_value_t* local_off;
            ckc_value_t* base;
            ckc_value_t* global_off;
            ckc_value_t* w;

            /* local_off = b.global_atomic_add(Counter, eid, b.const_i32(1)) */
            local_off = ckc_b_global_atomic_add(b, ctx->Counter, ctx->eid,
                                                ckc_b_const_i32(b, 1), NULL);
            /* base = b.global_load_i32(Offsets, eid)  (Python default align=4) */
            base = ckc_b_global_load_i32(b, ctx->Offsets, ctx->eid, 4);
            /* global_off = b.add(base, local_off) */
            global_off = ckc_b_add(b, base, local_off);

            /* w = b.global_load_f32(TopkWeights, pair_idx)  (default align=4) */
            w = ckc_b_global_load_f32(b, ctx->TopkWeights, ctx->pair_idx, 4);

            /* b.global_store(SortedTokenIds, global_off, t_idx, align=4) */
            ckc_b_global_store(b, ctx->SortedTokenIds, global_off, ctx->t_idx, 4);
            /* b.global_store(SortedTopkIds, global_off, k_idx, align=4) */
            ckc_b_global_store(b, ctx->SortedTopkIds, global_off, ctx->k_idx, 4);
            /* b.global_store(SortedWeights, global_off, w, align=4) */
            ckc_b_global_store(b, ctx->SortedWeights, global_off, w, 4);
        }
        ckc_b_region_leave(b); /* leave inner valid_e then-region */
    }
    ckc_b_region_leave(b); /* leave outer in_bounds then-region */

    /* return b.kernel */
    return b->kernel;
}
