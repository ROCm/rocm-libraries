// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_fused_moe_gather.c -- C99 port of the build_moe_gather
 * phase functions (rocke/instances/common/fused_moe.py, lines 326-441).
 *
 * Scope of THIS translation unit:
 *   rocke_moe_gather_prologue  -- prologue (lines 368-415): spec gate, geometry
 *                               consts, kernel name + max_workgroup_size, the 5
 *                               params in ABI order, bid/tid, pinned token_id,
 *                               valid_token, bucket_base, src_row_base, c_vec,
 *                               chunks. Fills a rocke_moe_stream_ctx_t (kind
 *                               GATHER).
 *   rocke_moe_gather_body      -- body (lines 416-439): scf_if(valid_token) with
 *                               the interleaved-chunk vec copy and the VEC==1
 *                               native-dtype fallback.
 *
 * The builder-call sequence is byte-identical to the Python source so a lowered
 * .ll diffs clean. Peers (rocke_moe_effective_vec, rocke_fused_moe_is_valid_spec,
 * rocke_fused_moe_spec_kernel_name, rocke_fused_moe_spec_elems_per_thread_hidden,
 * rocke_io_ir_type, rocke_b_load_sorted_token_id) are resolved via their headers.
 */

#include <stddef.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.gather_scatter.h" /* rocke_b_load_sorted_token_id*/
#include "rocke/helper_rocke.helpers.io.h" /* rocke_io_ir_type           */
#include "rocke/instance_fused_moe.h"
#include "rocke/instance_fused_moe_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err            */

/* ===================================================================== *
 *  PROLOGUE  (build_moe_gather, lines 368-415)
 * ===================================================================== */
bool rocke_moe_gather_prologue(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL || ctx->spec == NULL)
    {
        return false;
    }

    rocke_ir_builder_t* b = ctx->b;
    const rocke_fused_moe_spec_t* spec = ctx->spec;

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    char why[ROCKE_ERR_MSG_CAP];
    if(!rocke_fused_moe_is_valid_spec(spec, why, sizeof(why)))
    {
        /* Mirror: raise ValueError(f"invalid fused_moe spec: {why}"). The build
         * entry calls this prologue inside a ckc::guard_builder boundary, so the
         * throwing rocke_i_set_err records the exact Python message text + status
         * on the builder; the bare return below is dead after the throw. */
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fused_moe spec: %s", why);
        return false;
    }

    /* H = spec.hidden; BS = spec.block_size; EPT = spec.elems_per_thread_hidden;
     * VEC = _effective_vec(spec.vec, BS, H); dtype = spec.dtype */
    ctx->kind = ROCKE_MOE_STREAM_GATHER;
    ctx->N = spec->hidden; /* H              */
    ctx->BS = spec->block_size; /* BS             */
    ctx->EPT = rocke_fused_moe_spec_elems_per_thread_hidden(spec);
    ctx->VEC = rocke_moe_effective_vec(spec->vec, ctx->BS, ctx->N);
    ctx->dtype = spec->dtype;

    /* b.kernel.attrs["max_workgroup_size"] = BS  (the IRBuilder name was seeded
     * with spec.kernel_name("gather") by the convenience *_new entry). */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    /* ty = io_ir_type(dtype) */
    ctx->ty = rocke_io_ir_type(ctx->dtype);

    /* X = b.param("X", PtrType(ty, "global"), noalias=True, readonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->X = rocke_b_param(b, "X", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }

    /* SortedTokenIds = b.param("SortedTokenIds", PtrType(I32,"global"),
     *                          noalias=True, readonly=True, align=4) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 4;
        opts.align_set = true;
        ctx->SortedTokenIds
            = rocke_b_param(b, "SortedTokenIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);
    }

    /* GroupedInput = b.param("GroupedInput", PtrType(ty,"global"),
     *                        noalias=True, writeonly=True, align=16) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx->GroupedInput
            = rocke_b_param(b, "GroupedInput", rocke_ptr_type(b, ctx->ty, "global"), &opts);
    }

    /* _tokens = b.param("tokens", I32)   # ABI matches CK Tile */
    ctx->p_tokens = rocke_b_param(b, "tokens", rocke_i32(), NULL);
    /* _hidden = b.param("hidden", I32) */
    ctx->p_hidden = rocke_b_param(b, "hidden", rocke_i32(), NULL);

    /* bid = b.block_id_x(); tid = b.thread_id_x() */
    ctx->bid = rocke_b_block_id_x(b);
    ctx->tid = rocke_b_thread_id_x(b);

    /* token_id = b.to_sgpr_u32(load_sorted_token_id(b, SortedTokenIds, bid)) */
    ctx->token_id
        = rocke_b_to_sgpr_u32(b, rocke_b_load_sorted_token_id(b, ctx->SortedTokenIds, ctx->bid));
    /* valid_token = b.cmp_ge(token_id, b.const_i32(0)) */
    ctx->valid_token = rocke_b_cmp_ge(b, ctx->token_id, rocke_b_const_i32(b, 0));
    /* bucket_base = b.mul(bid, b.const_i32(H)) */
    ctx->bucket_base = rocke_b_mul(b, ctx->bid, rocke_b_const_i32(b, ctx->N));
    /* src_row_base = b.mul(token_id, b.const_i32(H)) */
    ctx->src_row_base = rocke_b_mul(b, ctx->token_id, rocke_b_const_i32(b, ctx->N));

    /* chunks = EPT // VEC; c_vec = b.const_i32(VEC) */
    ctx->chunks = ctx->EPT / ctx->VEC;
    ctx->c_vec = rocke_b_const_i32(b, ctx->VEC);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  BODY  (build_moe_gather, lines 416-439)
 * ===================================================================== */
rocke_kernel_def_t* rocke_moe_gather_body(rocke_moe_stream_ctx_t* ctx)
{
    if(ctx == NULL || ctx->b == NULL)
    {
        return NULL;
    }

    rocke_ir_builder_t* b = ctx->b;
    const int BS = ctx->BS;
    const int VEC = ctx->VEC;
    const char* dtype = ctx->dtype;

    /* with b.scf_if(valid_token): */
    rocke_if_t iff = rocke_b_scf_if(b, ctx->valid_token);
    rocke_b_region_enter(b, iff.then_region);

    for(int k = 0; k < ctx->chunks; ++k)
    {
        /* h_col = b.add(b.const_i32(k * BS * VEC), b.mul(tid, c_vec))
         * Python evaluates the args left-to-right: const_i32 first (lower SSA
         * id), then mul. C function-argument evaluation order is unspecified
         * (right-to-left on this toolchain), which would swap the ids. Bind the
         * operands to temporaries in source order to match Python byte-for-byte. */
        rocke_value_t* h_col_const = rocke_b_const_i32(b, (int64_t)k * BS * VEC);
        rocke_value_t* h_col_mul = rocke_b_mul(b, ctx->tid, ctx->c_vec);
        rocke_value_t* h_col = rocke_b_add(b, h_col_const, h_col_mul);
        /* src_off = b.add(src_row_base, h_col) */
        rocke_value_t* src_off = rocke_b_add(b, ctx->src_row_base, h_col);
        /* dst_off = b.add(bucket_base, h_col) */
        rocke_value_t* dst_off = rocke_b_add(b, ctx->bucket_base, h_col);

        if(VEC == 1)
        {
            /* VEC==1 native-dtype fallback. */
            rocke_value_t* v;
            if(dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
            {
                /* v = b.global_load_f16(X, src_off) */
                v = rocke_b_global_load_f16(b, ctx->X, src_off, 0);
            }
            else
            {
                /* v = b.global_load_bf16(X, src_off) */
                v = rocke_b_global_load_bf16(b, ctx->X, src_off, 0);
            }
            /* b.global_store(GroupedInput, dst_off, v) */
            rocke_b_global_store(b, ctx->GroupedInput, dst_off, v, 0);
        }
        else
        {
            /* v = b.global_load_vN(X, src_off, ty, VEC) */
            rocke_value_t* v = rocke_b_global_load_vN(b, ctx->X, src_off, ctx->ty, VEC, 0);
            /* b.global_store_vN(GroupedInput, dst_off, v, VEC) */
            rocke_b_global_store_vN(b, ctx->GroupedInput, dst_off, v, VEC, 0);
        }
    }

    rocke_b_region_leave(b);

    /* return b.kernel */
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}
