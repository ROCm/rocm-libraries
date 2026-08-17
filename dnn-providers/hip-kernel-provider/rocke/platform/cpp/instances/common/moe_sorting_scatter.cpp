// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_scatter.c.c -- chunked port of the
 * SCATTER kernel phase of build_moe_sort_scatter (rocke/instances/common/
 * moe_sorting.py lines 447-540).
 *
 * Implements the two scatter phase functions declared in
 * rocke/instance_moe_sorting_internal.h:
 *   rocke_moe_sort_scatter_prologue  (lines 481-525)
 *   rocke_moe_sort_scatter_body      (lines 527-540)
 *
 * Peer phase functions / module helpers (decode_pair_token_topk,
 * decode_expert_load, is_valid_spec_impl, kernel_name, ...) live in their own
 * TUs and are reached via the internal header; this TU implements ONLY the
 * scatter scope.
 */

#include <stdio.h>
#include <string.h>

#include "rocke/instance_moe_sorting.h"
#include "rocke/instance_moe_sorting_internal.h"
#include "rocke/ir.h"

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
bool rocke_moe_sort_scatter_prologue(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b;
    const rocke_moe_sorting_spec_t* spec;
    char reason[ROCKE_ERR_MSG_CAP];
    rocke_param_opts_t opts;

    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
        return false;

    b = ctx->b;
    spec = ctx->spec;

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if(!rocke_moe_sort_is_valid_spec_impl(spec, ctx->arch, reason, sizeof(reason), NULL))
    {
        /* raise ValueError(f"invalid moe_sorting spec: {why}") */
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
            ROCKE_ERR_SNPRINTF(b->err, ROCKE_ERR_MSG_CAP, "invalid moe_sorting spec: %s", reason);
        }
        return false;
    }

    /* geometry scalars (Python: BS = spec.block_size; topk = spec.topk; E unused
     * by scatter body but mirrored for ctx consistency). */
    ctx->BS = spec->block_size;
    ctx->E = spec->experts;
    ctx->topk = spec->topk;

    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* ---- 10-entry ABI param block, in Python declaration order. ---- */

    /* TopkIds: ptr<i32,global>, noalias=True, readonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->TopkIds = rocke_b_param(b, "TopkIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* TopkWeights: ptr<f32,global>, noalias=True, readonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->TopkWeights
        = rocke_b_param(b, "TopkWeights", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    /* Offsets: ptr<i32,global>, noalias=True, readonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->Offsets = rocke_b_param(b, "Offsets", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* Counter: ptr<i32,global>, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.align = 4;
    opts.align_set = true;
    ctx->Counter = rocke_b_param(b, "Counter", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* SortedTokenIds: ptr<i32,global>, writeonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->SortedTokenIds
        = rocke_b_param(b, "SortedTokenIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* SortedTopkIds: ptr<i32,global>, writeonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->SortedTopkIds
        = rocke_b_param(b, "SortedTopkIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    /* SortedWeights: ptr<f32,global>, writeonly=True, align=4 */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    ctx->SortedWeights
        = rocke_b_param(b, "SortedWeights", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    /* tokens = b.param("tokens", I32)  # noqa: F841 - ABI */
    ctx->tokens = rocke_b_param(b, "tokens", rocke_i32(), NULL);
    /* topk = b.param("topk", I32) */
    ctx->topk_param = rocke_b_param(b, "topk", rocke_i32(), NULL);
    /* num_experts = b.param("num_experts", I32) */
    ctx->num_experts = rocke_b_param(b, "num_experts", rocke_i32(), NULL);

    /* tid = b.thread_id_x() */
    ctx->tid = rocke_b_thread_id_x(b);
    /* bid = b.block_id_x() */
    ctx->bid = rocke_b_block_id_x(b);
    /* pair_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid) */
    ctx->pair_idx = rocke_b_add(
        b, rocke_b_mul(b, ctx->bid, rocke_b_const_i32(b, spec->block_size)), ctx->tid);

    /* t_idx, k_idx = _decode_pair_token_topk(b, pair_idx, spec.topk) */
    rocke_moe_sort_decode_pair_token_topk(b, ctx->pair_idx, spec->topk, &ctx->t_idx, &ctx->k_idx);

    /* num_pairs = b.mul(tokens, topk) */
    ctx->num_pairs = rocke_b_mul(b, ctx->tokens, ctx->topk_param);
    /* in_bounds = b.cmp_lt(pair_idx, num_pairs) */
    ctx->in_bounds = rocke_b_cmp_lt(b, ctx->pair_idx, ctx->num_pairs);

    return b->status == ROCKE_OK;
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
rocke_kernel_def_t* rocke_moe_sort_scatter_body(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b;
    rocke_if_t outer;

    if(ctx == NULL || ctx->b == NULL)
        return NULL;

    b = ctx->b;

    /* with b.scf_if(in_bounds): */
    outer = rocke_b_scf_if(b, ctx->in_bounds);
    rocke_b_region_enter(b, outer.then_region);
    {
        rocke_if_t inner;

        /* eid, valid_e = _decode_expert_load(b, TopkIds, pair_idx, num_experts) */
        rocke_moe_sort_decode_expert_load(
            b, ctx->TopkIds, ctx->pair_idx, ctx->num_experts, &ctx->eid, &ctx->valid_e);

        /* with b.scf_if(valid_e): */
        inner = rocke_b_scf_if(b, ctx->valid_e);
        rocke_b_region_enter(b, inner.then_region);
        {
            rocke_value_t* local_off;
            rocke_value_t* base;
            rocke_value_t* global_off;
            rocke_value_t* w;

            /* local_off = b.global_atomic_add(Counter, eid, b.const_i32(1)) */
            local_off = rocke_b_global_atomic_add(
                b, ctx->Counter, ctx->eid, rocke_b_const_i32(b, 1), NULL);
            /* base = b.global_load_i32(Offsets, eid)  (Python default align=4) */
            base = rocke_b_global_load_i32(b, ctx->Offsets, ctx->eid, 4);
            /* global_off = b.add(base, local_off) */
            global_off = rocke_b_add(b, base, local_off);

            /* w = b.global_load_f32(TopkWeights, pair_idx)  (default align=4) */
            w = rocke_b_global_load_f32(b, ctx->TopkWeights, ctx->pair_idx, 4);

            /* b.global_store(SortedTokenIds, global_off, t_idx, align=4) */
            rocke_b_global_store(b, ctx->SortedTokenIds, global_off, ctx->t_idx, 4);
            /* b.global_store(SortedTopkIds, global_off, k_idx, align=4) */
            rocke_b_global_store(b, ctx->SortedTopkIds, global_off, ctx->k_idx, 4);
            /* b.global_store(SortedWeights, global_off, w, align=4) */
            rocke_b_global_store(b, ctx->SortedWeights, global_off, w, 4);
        }
        rocke_b_region_leave(b); /* leave inner valid_e then-region */
    }
    rocke_b_region_leave(b); /* leave outer in_bounds then-region */

    /* return b.kernel */
    return b->kernel;
}
