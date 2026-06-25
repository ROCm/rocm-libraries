// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_fused_moe_scatter_reduce.c -- C99 port of the two
 * atomic/indirect MoE kernel builders in
 * ck_dsl/instances/common/fused_moe.py:
 *   build_moe_static_scatter_gather   (lines 731-849)
 *   build_moe_topk_weighted_reduce    (lines 880-997)
 *
 * Scope of THIS translation unit:
 *   ckc_moe_ssg_prologue  -- SSG prologue (lines 759-803): spec gate, geometry
 *                            consts, kernel name + max_workgroup_size, the 12
 *                            params in ABI order with attrs, bid/tid,
 *                            out_row_slot smem_alloc, num_pairs, in_bounds,
 *                            c_vec, chunks. Fills a ckc_moe_ssg_ctx_t.
 *   ckc_moe_ssg_claim     -- slot-claim phase (lines 805-819): opens
 *                            scf_if(in_bounds) -> eid -> valid_e ->
 *                            scf_if(valid_e) -> t_idx, is_lead ->
 *                            scf_if(is_lead){ atomic_add slot claim + lead-lane
 *                            writes }. Leaves only the is_lead scope; the
 *                            in_bounds / valid_e scopes stay open for the copy
 *                            phase.
 *   ckc_moe_ssg_copy      -- broadcast + copy phase (lines 820-847): sync(),
 *                            LDS broadcast of out_row, src/dst row bases, the
 *                            interleaved-chunk vec row copy; then closes the
 *                            valid_e + in_bounds scopes and returns b->kernel.
 *   ckc_moe_reduce_prologue -- reduce prologue (lines 911-980): spec gate,
 *                            geometry consts, kernel name + max_workgroup_size,
 *                            the 7 params, bid/tid, pinned token_id, weight,
 *                            valid_token, bucket_base, y_row_base, chunks, and
 *                            the BLOCK-PARTITIONED lane_chunk_base=tid*EPT.
 *   ckc_moe_reduce_body   -- reduce body (lines 981-995): scf_if(valid_token)
 *                            with the per-lane vec load + scalar f32 atomic_add
 *                            accumulate into Y.
 *
 * The builder-call sequence is byte-identical to the Python source so a lowered
 * .ll diffs clean. Peers (ckc_moe_effective_vec, ckc_fused_moe_is_valid_spec,
 * ckc_fused_moe_spec_kernel_name, ckc_fused_moe_spec_elems_per_thread_hidden,
 * ckc_io_ir_type, ckc_b_load_sorted_token_id, ckc_b_load_sorted_topk_weight,
 * ckc_b_load_scalar_as_f32) are resolved via their headers.
 */

#include <stddef.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/instance_fused_moe.h"
#include "ckc/instance_fused_moe_internal.h"
#include "ckc/helper_ck_dsl.helpers.io.h"             /* ckc_io_ir_type, ckc_b_load_scalar_as_f32 */
#include "ckc/helper_ck_dsl.helpers.gather_scatter.h" /* ckc_b_load_sorted_token_id/topk_weight   */
#include "ckc/ir_internal.h" /* ckc_i_set_err                             */

/* ===================================================================== *
 *  STATIC_SCATTER_GATHER PROLOGUE  (lines 759-803)
 * ===================================================================== */
bool ckc_moe_ssg_prologue(ckc_moe_ssg_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return false;
    }

    ckc_ir_builder_t* b              = ctx->b;
    const ckc_fused_moe_spec_t* spec = ctx->spec;

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    char why[CKC_ERR_MSG_CAP];
    if(!ckc_fused_moe_is_valid_spec(spec, why, sizeof(why)))
    {
        /* raise ValueError(f"invalid fused_moe spec: {why}") -- this prologue
         * runs inside a ckc::guard_builder boundary, so the throwing
         * ckc_i_set_err records the exact Python message text + status; the bare
         * return below is dead after the throw. */
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fused_moe spec: %s", why);
        return false;
    }

    /* H = spec.hidden; BS = spec.block_size; EPT = spec.elems_per_thread_hidden;
     * VEC = _effective_vec(spec.vec, BS, H); dtype = spec.dtype */
    ctx->H     = spec->hidden;
    ctx->BS    = spec->block_size;
    ctx->EPT   = ckc_fused_moe_spec_elems_per_thread_hidden(spec);
    ctx->VEC   = ckc_moe_effective_vec(spec->vec, ctx->BS, ctx->H);
    ctx->dtype = spec->dtype;

    /* b.kernel.attrs["max_workgroup_size"] = BS (name seeded with
     * spec.kernel_name("static_scatter_gather") by the *_new entry). */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    /* ty = io_ir_type(dtype) */
    ctx->ty = ckc_io_ir_type(ctx->dtype);

    /* TopkIds = b.param("TopkIds", PtrType(I32,"global"),
     *                   noalias=True, readonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias          = true;
        opts.noalias_set      = true;
        opts.readonly         = true;
        opts.readonly_set     = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->TopkIds = ckc_b_param(b, "TopkIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* TopkWeights = b.param("TopkWeights", PtrType(F32,"global"),
     *                       noalias=True, readonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias          = true;
        opts.noalias_set      = true;
        opts.readonly         = true;
        opts.readonly_set     = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->TopkWeights =
            ckc_b_param(b, "TopkWeights", ckc_ptr_type(b, ckc_f32(), "global"), &opts);
    }

    /* Counter = b.param("Counter", PtrType(I32,"global"), align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.align            = 4;
        opts.align_set        = true;
        ctx->Counter = ckc_b_param(b, "Counter", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* X = b.param("X", PtrType(ty,"global"), noalias=True, readonly=True, align=16) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias          = true;
        opts.noalias_set      = true;
        opts.readonly         = true;
        opts.readonly_set     = true;
        opts.align            = 16;
        opts.align_set        = true;
        ctx->X                = ckc_b_param(b, "X", ckc_ptr_type(b, ctx->ty, "global"), &opts);
    }

    /* SortedTokenIds = b.param("SortedTokenIds", PtrType(I32,"global"),
     *                          writeonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.writeonly        = true;
        opts.writeonly_set    = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->SortedTokenIds =
            ckc_b_param(b, "SortedTokenIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* SortedWeights = b.param("SortedWeights", PtrType(F32,"global"),
     *                         writeonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.writeonly        = true;
        opts.writeonly_set    = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->SortedWeights =
            ckc_b_param(b, "SortedWeights", ckc_ptr_type(b, ckc_f32(), "global"), &opts);
    }

    /* GroupedInput = b.param("GroupedInput", PtrType(ty,"global"),
     *                        writeonly=True, align=16) */
    {
        ckc_param_opts_t opts = {0};
        opts.writeonly        = true;
        opts.writeonly_set    = true;
        opts.align            = 16;
        opts.align_set        = true;
        ctx->GroupedInput =
            ckc_b_param(b, "GroupedInput", ckc_ptr_type(b, ctx->ty, "global"), &opts);
    }

    /* tokens = b.param("tokens", I32) */
    ctx->tokens = ckc_b_param(b, "tokens", ckc_i32(), NULL);
    /* topk = b.param("topk", I32) */
    ctx->topk = ckc_b_param(b, "topk", ckc_i32(), NULL);
    /* num_experts = b.param("num_experts", I32) */
    ctx->num_experts = ckc_b_param(b, "num_experts", ckc_i32(), NULL);
    /* _hidden = b.param("hidden", I32)  # ABI compatibility */
    ctx->p_hidden = ckc_b_param(b, "hidden", ckc_i32(), NULL);
    /* slot_size = b.param("slot_size", I32) */
    ctx->slot_size = ckc_b_param(b, "slot_size", ckc_i32(), NULL);

    /* bid = b.block_id_x(); tid = b.thread_id_x() */
    ctx->bid = ckc_b_block_id_x(b);
    ctx->tid = ckc_b_thread_id_x(b);

    /* out_row_slot = b.smem_alloc(I32, [1], name_hint="sg_out_row") */
    {
        int shape[1]      = {1};
        ctx->out_row_slot = ckc_b_smem_alloc(b, ckc_i32(), shape, 1, "sg_out_row");
    }

    /* num_pairs = b.mul(tokens, topk) */
    ctx->num_pairs = ckc_b_mul(b, ctx->tokens, ctx->topk);
    /* in_bounds = b.cmp_lt(bid, num_pairs) */
    ctx->in_bounds = ckc_b_cmp_lt(b, ctx->bid, ctx->num_pairs);

    /* c_vec = b.const_i32(VEC); chunks = EPT // VEC */
    ctx->c_vec  = ckc_b_const_i32(b, ctx->VEC);
    ctx->chunks = ctx->EPT / ctx->VEC;

    return ckc_ir_builder_ok(b);
}

/* ===================================================================== *
 *  STATIC_SCATTER_GATHER SLOT-CLAIM PHASE  (lines 805-819)
 *
 *  Opens scf_if(in_bounds) -> scf_if(valid_e) -> scf_if(is_lead){ ... }, then
 *  leaves ONLY the is_lead scope. The in_bounds + valid_e scopes stay open so
 *  the copy phase emits into them (mirrors the single Python `with` nest).
 * ===================================================================== */
void ckc_moe_ssg_claim(ckc_moe_ssg_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return;
    }

    ckc_ir_builder_t* b = ctx->b;

    /* with b.scf_if(in_bounds): */
    ckc_if_t if_bounds = ckc_b_scf_if(b, ctx->in_bounds);
    ckc_b_region_enter(b, if_bounds.then_region);

    /* eid = b.global_load_i32(TopkIds, bid) */
    ctx->eid = ckc_b_global_load_i32(b, ctx->TopkIds, ctx->bid, 0);
    /* valid_e = b.land(b.cmp_ge(eid, b.const_i32(0)),
     *                  b.cmp_lt(eid, num_experts))
     * Python evaluates args left-to-right: cmp_ge first (lower SSA id), then
     * cmp_lt. C argument-evaluation order is unspecified (right-to-left here),
     * which would swap the ids. Bind in source order to match Python. */
    ckc_value_t* valid_e_ge = ckc_b_cmp_ge(b, ctx->eid, ckc_b_const_i32(b, 0));
    ckc_value_t* valid_e_lt = ckc_b_cmp_lt(b, ctx->eid, ctx->num_experts);
    ctx->valid_e            = ckc_b_land(b, valid_e_ge, valid_e_lt);

    /* with b.scf_if(valid_e): */
    ckc_if_t if_valid = ckc_b_scf_if(b, ctx->valid_e);
    ckc_b_region_enter(b, if_valid.then_region);

    /* t_idx = b.div(bid, topk) */
    ctx->t_idx = ckc_b_div(b, ctx->bid, ctx->topk);
    /* is_lead = b.cmp_eq(tid, b.const_i32(0)) */
    ctx->is_lead = ckc_b_cmp_eq(b, ctx->tid, ckc_b_const_i32(b, 0));

    /* with b.scf_if(is_lead): */
    ckc_if_t if_lead = ckc_b_scf_if(b, ctx->is_lead);
    ckc_b_region_enter(b, if_lead.then_region);

    /* local = b.global_atomic_add(Counter, eid, b.const_i32(1)) */
    ctx->local = ckc_b_global_atomic_add(b, ctx->Counter, ctx->eid, ckc_b_const_i32(b, 1), NULL);
    /* base = b.mul(eid, slot_size) */
    ctx->base = ckc_b_mul(b, ctx->eid, ctx->slot_size);
    /* out_row_lead = b.add(base, local) */
    ctx->out_row_lead = ckc_b_add(b, ctx->base, ctx->local);
    /* b.smem_store_vN(out_row_slot, [b.const_i32(0)], out_row_lead, n=1) */
    {
        ckc_value_t* idx[1] = {ckc_b_const_i32(b, 0)};
        ckc_b_smem_store_vN(b, ctx->out_row_slot, idx, 1, ctx->out_row_lead, 1);
    }

    /* w = b.global_load_f32(TopkWeights, bid) */
    ctx->w = ckc_b_global_load_f32(b, ctx->TopkWeights, ctx->bid, 0);
    /* b.global_store(SortedTokenIds, out_row_lead, t_idx, align=4) */
    ckc_b_global_store(b, ctx->SortedTokenIds, ctx->out_row_lead, ctx->t_idx, 4);
    /* b.global_store(SortedWeights, out_row_lead, w, align=4) */
    ckc_b_global_store(b, ctx->SortedWeights, ctx->out_row_lead, ctx->w, 4);

    /* leave the is_lead scope; in_bounds + valid_e stay open for copy phase. */
    ckc_b_region_leave(b); /* is_lead */
}

/* ===================================================================== *
 *  STATIC_SCATTER_GATHER BROADCAST + COPY PHASE  (lines 820-847)
 * ===================================================================== */
ckc_kernel_def_t* ckc_moe_ssg_copy(ckc_moe_ssg_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    ckc_ir_builder_t* b = ctx->b;
    const int BS        = ctx->BS;
    const int VEC       = ctx->VEC;
    const int H         = ctx->H;
    const char* dtype   = ctx->dtype;

    /* b.sync()  (still inside the valid_e scope) */
    ckc_b_sync(b);

    /* out_row = b.to_sgpr_u32(
     *     b.vec_extract(
     *         b.smem_load_vN(out_row_slot, b.const_i32(0), dtype=I32, n=1), 0)) */
    {
        ckc_value_t* idx[1] = {ckc_b_const_i32(b, 0)};
        ckc_value_t* loaded = ckc_b_smem_load_vN(b, ctx->out_row_slot, idx, 1, ckc_i32(), 1);
        ctx->out_row        = ckc_b_to_sgpr_u32(b, ckc_b_vec_extract(b, loaded, 0));
    }

    /* src_row_base = b.mul(t_idx, b.const_i32(H)) */
    ctx->src_row_base = ckc_b_mul(b, ctx->t_idx, ckc_b_const_i32(b, H));
    /* dst_row_base = b.mul(out_row, b.const_i32(H)) */
    ctx->dst_row_base = ckc_b_mul(b, ctx->out_row, ckc_b_const_i32(b, H));

    /* Copy X[t_idx, :] -> GroupedInput[out_row, :] in interleaved-chunk vec
     * loads (matches build_moe_gather). */
    for(int k = 0; k < ctx->chunks; ++k)
    {
        /* h_col = b.add(b.const_i32(k * BS * VEC), b.mul(tid, c_vec))
         * Bind operands in source order (const then mul) so the SSA ids match
         * Python's left-to-right arg evaluation; C arg-eval order is unspecified
         * (right-to-left here) and would otherwise swap the ids. */
        ckc_value_t* h_col_const = ckc_b_const_i32(b, (int64_t)k * BS * VEC);
        ckc_value_t* h_col_mul   = ckc_b_mul(b, ctx->tid, ctx->c_vec);
        ckc_value_t* h_col       = ckc_b_add(b, h_col_const, h_col_mul);
        /* src = b.add(src_row_base, h_col) */
        ckc_value_t* src = ckc_b_add(b, ctx->src_row_base, h_col);
        /* dst = b.add(dst_row_base, h_col) */
        ckc_value_t* dst = ckc_b_add(b, ctx->dst_row_base, h_col);

        if(VEC == 1)
        {
            ckc_value_t* v;
            if(dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
            {
                /* v = b.global_load_f16(X, src) */
                v = ckc_b_global_load_f16(b, ctx->X, src, 0);
            }
            else
            {
                /* v = b.global_load_bf16(X, src) */
                v = ckc_b_global_load_bf16(b, ctx->X, src, 0);
            }
            /* b.global_store(GroupedInput, dst, v) */
            ckc_b_global_store(b, ctx->GroupedInput, dst, v, 0);
        }
        else
        {
            /* v = b.global_load_vN(X, src, ty, VEC) */
            ckc_value_t* v = ckc_b_global_load_vN(b, ctx->X, src, ctx->ty, VEC, 0);
            /* b.global_store_vN(GroupedInput, dst, v, VEC) */
            ckc_b_global_store_vN(b, ctx->GroupedInput, dst, v, VEC, 0);
        }
    }

    /* close the valid_e + in_bounds scopes opened by the claim phase. */
    ckc_b_region_leave(b); /* valid_e   */
    ckc_b_region_leave(b); /* in_bounds */

    /* return b.kernel */
    return ckc_ir_builder_ok(b) ? b->kernel : NULL;
}

/* ===================================================================== *
 *  TOPK_WEIGHTED_REDUCE PROLOGUE  (lines 911-980)
 * ===================================================================== */
bool ckc_moe_reduce_prologue(ckc_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return false;
    }

    ckc_ir_builder_t* b              = ctx->b;
    const ckc_fused_moe_spec_t* spec = ctx->spec;

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    char why[CKC_ERR_MSG_CAP];
    if(!ckc_fused_moe_is_valid_spec(spec, why, sizeof(why)))
    {
        /* raise ValueError(f"invalid fused_moe spec: {why}") -- this prologue
         * runs inside a ckc::guard_builder boundary, so the throwing
         * ckc_i_set_err records the exact Python message text + status; the bare
         * return below is dead after the throw. */
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fused_moe spec: %s", why);
        return false;
    }

    /* H = spec.hidden; BS = spec.block_size; EPT = spec.elems_per_thread_hidden;
     * VEC = _effective_vec(spec.vec, BS, H); dtype = spec.dtype */
    ctx->kind  = CKC_MOE_STREAM_REDUCE;
    ctx->N     = spec->hidden; /* H              */
    ctx->BS    = spec->block_size;
    ctx->EPT   = ckc_fused_moe_spec_elems_per_thread_hidden(spec);
    ctx->VEC   = ckc_moe_effective_vec(spec->vec, ctx->BS, ctx->N);
    ctx->dtype = spec->dtype;

    /* b.kernel.attrs["max_workgroup_size"] = BS (name seeded with
     * spec.kernel_name("reduce") by the *_new entry). */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    /* ty = io_ir_type(dtype) */
    ctx->ty = ckc_io_ir_type(ctx->dtype);

    /* DownOut = b.param("DownOut", PtrType(ty,"global"),
     *                   noalias=True, readonly=True, align=16) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias          = true;
        opts.noalias_set      = true;
        opts.readonly         = true;
        opts.readonly_set     = true;
        opts.align            = 16;
        opts.align_set        = true;
        ctx->DownOut = ckc_b_param(b, "DownOut", ckc_ptr_type(b, ctx->ty, "global"), &opts);
    }

    /* SortedTokenIds = b.param("SortedTokenIds", PtrType(I32,"global"),
     *                          noalias=True, readonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias          = true;
        opts.noalias_set      = true;
        opts.readonly         = true;
        opts.readonly_set     = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->SortedTokenIds =
            ckc_b_param(b, "SortedTokenIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* SortedWeights = b.param("SortedWeights", PtrType(F32,"global"),
     *                         noalias=True, readonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias          = true;
        opts.noalias_set      = true;
        opts.readonly         = true;
        opts.readonly_set     = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->SortedWeights =
            ckc_b_param(b, "SortedWeights", ckc_ptr_type(b, ckc_f32(), "global"), &opts);
    }

    /* Y = b.param("Y", PtrType(F32,"global"), align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.align            = 4;
        opts.align_set        = true;
        ctx->Y                = ckc_b_param(b, "Y", ckc_ptr_type(b, ckc_f32(), "global"), &opts);
    }

    /* _total_pairs = b.param("total_pairs", I32) */
    ctx->p_total_pairs = ckc_b_param(b, "total_pairs", ckc_i32(), NULL);
    /* _hidden = b.param("hidden", I32) */
    ctx->p_hidden = ckc_b_param(b, "hidden", ckc_i32(), NULL);
    /* _tokens = b.param("tokens", I32) */
    ctx->p_tokens = ckc_b_param(b, "tokens", ckc_i32(), NULL);

    /* bid = b.block_id_x(); tid = b.thread_id_x() */
    ctx->bid = ckc_b_block_id_x(b);
    ctx->tid = ckc_b_thread_id_x(b);

    /* token_id = b.to_sgpr_u32(load_sorted_token_id(b, SortedTokenIds, bid)) */
    ctx->token_id =
        ckc_b_to_sgpr_u32(b, ckc_b_load_sorted_token_id(b, ctx->SortedTokenIds, ctx->bid));
    /* weight = load_sorted_topk_weight(b, SortedWeights, bid) */
    ctx->weight = ckc_b_load_sorted_topk_weight(b, ctx->SortedWeights, ctx->bid);
    /* valid_token = b.cmp_ge(token_id, b.const_i32(0)) */
    ctx->valid_token = ckc_b_cmp_ge(b, ctx->token_id, ckc_b_const_i32(b, 0));
    /* bucket_base = b.mul(bid, b.const_i32(H)) */
    ctx->bucket_base = ckc_b_mul(b, ctx->bid, ckc_b_const_i32(b, ctx->N));
    /* y_row_base = b.mul(token_id, b.const_i32(H)) */
    ctx->y_row_base = ckc_b_mul(b, ctx->token_id, ckc_b_const_i32(b, ctx->N));

    /* chunks = EPT // VEC */
    ctx->chunks = ctx->EPT / ctx->VEC;
    /* lane_chunk_base = b.mul(tid, b.const_i32(EPT))  (BLOCK-PARTITIONED) */
    ctx->lane_chunk_base = ckc_b_mul(b, ctx->tid, ckc_b_const_i32(b, ctx->EPT));

    return ckc_ir_builder_ok(b);
}

/* ===================================================================== *
 *  TOPK_WEIGHTED_REDUCE BODY  (lines 981-995)
 * ===================================================================== */
ckc_kernel_def_t* ckc_moe_reduce_body(ckc_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    ckc_ir_builder_t* b = ctx->b;
    const int VEC       = ctx->VEC;
    const char* dtype   = ctx->dtype;

    /* with b.scf_if(valid_token): */
    ckc_if_t iff = ckc_b_scf_if(b, ctx->valid_token);
    ckc_b_region_enter(b, iff.then_region);

    for(int k = 0; k < ctx->chunks; ++k)
    {
        /* h_col = b.add(lane_chunk_base, b.const_i32(k * VEC)) */
        ckc_value_t* h_col =
            ckc_b_add(b, ctx->lane_chunk_base, ckc_b_const_i32(b, (int64_t)k * VEC));
        /* src_off = b.add(bucket_base, h_col) */
        ckc_value_t* src_off = ckc_b_add(b, ctx->bucket_base, h_col);
        /* dst_off = b.add(y_row_base, h_col) */
        ckc_value_t* dst_off = ckc_b_add(b, ctx->y_row_base, h_col);

        if(VEC == 1)
        {
            /* v = load_scalar_as_f32(b, DownOut, src_off, dtype=dtype) */
            ckc_value_t* v = ckc_b_load_scalar_as_f32(b, ctx->DownOut, src_off, dtype);
            /* contrib = b.fmul(weight, v) */
            ckc_value_t* contrib = ckc_b_fmul(b, ctx->weight, v);
            /* b.global_atomic_add(Y, dst_off, contrib) */
            ckc_b_global_atomic_add(b, ctx->Y, dst_off, contrib, NULL);
        }
        else
        {
            /* v_vec = b.global_load_vN(DownOut, src_off, ty, VEC) */
            ckc_value_t* v_vec = ckc_b_global_load_vN(b, ctx->DownOut, src_off, ctx->ty, VEC, 0);
            for(int i = 0; i < VEC; ++i)
            {
                /* v = b.cast_to_f32(b.vec_extract(v_vec, i)) */
                ckc_value_t* v = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, v_vec, i));
                /* contrib = b.fmul(weight, v) */
                ckc_value_t* contrib = ckc_b_fmul(b, ctx->weight, v);
                /* b.global_atomic_add(Y, b.add(dst_off, b.const_i32(i)), contrib) */
                ckc_b_global_atomic_add(
                    b, ctx->Y, ckc_b_add(b, dst_off, ckc_b_const_i32(b, i)), contrib, NULL);
            }
        }
    }

    ckc_b_region_leave(b); /* valid_token */

    /* return b.kernel */
    return ckc_ir_builder_ok(b) ? b->kernel : NULL;
}
