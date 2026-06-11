/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_moe_sorting_instance_moe_sorting_histogram.c.c -- C99 port of the
 * HISTOGRAM kernel phase of ck_dsl/instances/common/moe_sorting.py
 * (build_moe_sort_histogram, Python lines 194-272).
 *
 * Implements ONLY the three histogram phase functions declared in
 * ckc/instance_moe_sorting_internal.h:
 *   ckc_moe_sort_hist_prologue        (lines 224-243)
 *   ckc_moe_sort_hist_block_histogram (lines 245-258)
 *   ckc_moe_sort_hist_merge_to_global (lines 260-272)
 *
 * The shared module helpers (ckc_moe_sort_is_valid_spec_impl,
 * ckc_moe_sort_decode_expert_load) and the spec accessors / lds_zero_i32 helper
 * are implemented by peer TUs and the ported helper libraries; we call them
 * through the internal header / helper headers only.
 */

#include "ckc/instance_moe_sorting_internal.h"
#include "ckc/instance_moe_sorting.h"
#include "ckc/helper_ck_dsl.helpers.scan.h" /* ckc_lds_zero_i32 */
#include "ckc/ir.h"

/* ===================================================================== *
 *  Prologue (Python lines 224-243).
 *
 *    ok, why = is_valid_spec(spec, arch)
 *    if not ok: raise ValueError(...)
 *    BS = spec.block_size
 *    E  = spec.experts
 *    b.kernel.attrs["max_workgroup_size"] = BS
 *    TopkIds = b.param("TopkIds", PtrType(I32,"global"),
 *                      noalias=True, readonly=True, align=4)
 *    Hist    = b.param("Hist", PtrType(I32,"global"), align=4)
 *    num_pairs   = b.param("num_pairs", I32)
 *    num_experts = b.param("num_experts", I32)
 *    tid = b.thread_id_x()
 *    bid = b.block_id_x()
 *    pair_idx = b.add(b.mul(bid, b.const_i32(BS)), tid)
 * ===================================================================== */
bool ckc_moe_sort_hist_prologue(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    char reason[CKC_ERR_MSG_CAP];
    if (!ckc_moe_sort_is_valid_spec_impl(ctx->spec, ctx->arch, reason, sizeof(reason), NULL))
    {
        /* raise ValueError(f"invalid moe_sorting spec: {why}") -- route through
         * the sticky-error builder so callers observe the rejection. */
        b->status = CKC_ERR_VALUE;
        /* TODO(port): byte-identical "invalid moe_sorting spec: {why}"
         * formatting once the shared error-string convention is finalised. */
        return false;
    }

    /* BS = spec.block_size ; E = spec.experts */
    ctx->BS = ctx->spec->block_size;
    ctx->E  = ctx->spec->experts;

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)ctx->BS);

    /* TopkIds = b.param("TopkIds", PtrType(I32,"global"),
     *                   noalias=True, readonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.noalias      = true;
        opts.noalias_set  = true;
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 4;
        opts.align_set    = true;
        ctx->TopkIds = ckc_b_param(b, "TopkIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* Hist = b.param("Hist", PtrType(I32,"global"), align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.align     = 4;
        opts.align_set = true;
        ctx->Hist = ckc_b_param(b, "Hist", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* num_pairs = b.param("num_pairs", I32) */
    ctx->num_pairs = ckc_b_param(b, "num_pairs", ckc_i32(), NULL);
    /* num_experts = b.param("num_experts", I32) */
    ctx->num_experts = ckc_b_param(b, "num_experts", ckc_i32(), NULL);

    /* tid = b.thread_id_x() ; bid = b.block_id_x() */
    ctx->tid = ckc_b_thread_id_x(b);
    ctx->bid = ckc_b_block_id_x(b);

    /* pair_idx = b.add(b.mul(bid, b.const_i32(BS)), tid) */
    ctx->pair_idx =
        ckc_b_add(b, ckc_b_mul(b, ctx->bid, ckc_b_const_i32(b, (int64_t)ctx->BS)), ctx->tid);

    return ckc_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Stage 1: per-block LDS histogram (Python lines 245-258).
 *
 *    lds_hist = b.smem_alloc(I32, [E], name_hint="lds_hist")
 *    lds_zero_i32(b, lds_hist, tid=tid, block_size=BS, length=E)
 *    in_bounds = b.cmp_lt(pair_idx, num_pairs)
 *    with b.scf_if(in_bounds):
 *        eid, valid_e = _decode_expert_load(b, TopkIds, pair_idx, num_experts)
 *        with b.scf_if(valid_e):
 *            b.lds_atomic_add(lds_hist, [eid], b.const_i32(1))
 *    b.sync()
 * ===================================================================== */
void ckc_moe_sort_hist_block_histogram(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* lds_hist = b.smem_alloc(I32, [E], name_hint="lds_hist") */
    {
        int shape[1] = {ctx->E};
        ctx->lds_hist = ckc_b_smem_alloc(b, ckc_i32(), shape, 1, "lds_hist");
    }

    /* lds_zero_i32(b, lds_hist, tid=tid, block_size=BS, length=E) */
    ckc_lds_zero_i32(b, ctx->lds_hist, ctx->tid, ctx->BS, ctx->E);

    /* in_bounds = b.cmp_lt(pair_idx, num_pairs) */
    ctx->in_bounds = ckc_b_cmp_lt(b, ctx->pair_idx, ctx->num_pairs);

    /* with b.scf_if(in_bounds): */
    {
        ckc_if_t gate = ckc_b_scf_if(b, ctx->in_bounds);
        ckc_b_region_enter(b, gate.then_region);

        /* eid, valid_e = _decode_expert_load(b, TopkIds, pair_idx, num_experts) */
        ckc_moe_sort_decode_expert_load(
            b, ctx->TopkIds, ctx->pair_idx, ctx->num_experts, &ctx->eid, &ctx->valid_e);

        /* with b.scf_if(valid_e): */
        {
            ckc_if_t vgate = ckc_b_scf_if(b, ctx->valid_e);
            ckc_b_region_enter(b, vgate.then_region);

            /* b.lds_atomic_add(lds_hist, [eid], b.const_i32(1)) */
            {
                ckc_value_t* indices[1] = {ctx->eid};
                ckc_b_lds_atomic_add(
                    b, ctx->lds_hist, indices, 1, ckc_b_const_i32(b, 1), NULL);
            }

            ckc_b_region_leave(b);
        }

        ckc_b_region_leave(b);
    }

    /* b.sync() */
    ckc_b_sync(b);
}

/* ===================================================================== *
 *  Stage 2 + return (Python lines 260-272).
 *
 *    c_E = b.const_i32(E)
 *    in_bin = b.cmp_lt(tid, c_E)
 *    with b.scf_if(in_bin):
 *        cnt = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0)
 *        with b.scf_if(b.cmp_gt(cnt, b.const_i32(0))):
 *            b.global_atomic_add(Hist, tid, cnt)
 *    return b.kernel
 * ===================================================================== */
ckc_kernel_def_t* ckc_moe_sort_hist_merge_to_global(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* c_E = b.const_i32(E) */
    ctx->c_E = ckc_b_const_i32(b, (int64_t)ctx->E);

    /* in_bin = b.cmp_lt(tid, c_E) */
    ckc_value_t* in_bin = ckc_b_cmp_lt(b, ctx->tid, ctx->c_E);

    /* with b.scf_if(in_bin): */
    {
        ckc_if_t gate = ckc_b_scf_if(b, in_bin);
        ckc_b_region_enter(b, gate.then_region);

        /* cnt = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0) */
        ckc_value_t* tid_idx[1] = {ctx->tid};
        ckc_value_t* loaded     = ckc_b_smem_load_vN(b, ctx->lds_hist, tid_idx, 1, ckc_i32(), 1);
        ckc_value_t* cnt        = ckc_b_vec_extract(b, loaded, 0);

        /* with b.scf_if(b.cmp_gt(cnt, b.const_i32(0))): */
        {
            ckc_if_t cgate = ckc_b_scf_if(b, ckc_b_cmp_gt(b, cnt, ckc_b_const_i32(b, 0)));
            ckc_b_region_enter(b, cgate.then_region);

            /* b.global_atomic_add(Hist, tid, cnt) */
            ckc_b_global_atomic_add(b, ctx->Hist, ctx->tid, cnt, NULL);

            ckc_b_region_leave(b);
        }

        ckc_b_region_leave(b);
    }

    /* return b.kernel */
    return b->kernel;
}
