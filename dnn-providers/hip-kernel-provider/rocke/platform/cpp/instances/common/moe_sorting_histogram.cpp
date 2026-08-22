// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_histogram.c.c -- C99 port of the
 * HISTOGRAM kernel phase of rocke/instances/common/moe_sorting.py
 * (build_moe_sort_histogram, Python lines 194-272).
 *
 * Implements ONLY the three histogram phase functions declared in
 * rocke/instance_moe_sorting_internal.h:
 *   rocke_moe_sort_hist_prologue        (lines 224-243)
 *   rocke_moe_sort_hist_block_histogram (lines 245-258)
 *   rocke_moe_sort_hist_merge_to_global (lines 260-272)
 *
 * The shared module helpers (rocke_moe_sort_is_valid_spec_impl,
 * rocke_moe_sort_decode_expert_load) and the spec accessors / lds_zero_i32 helper
 * are implemented by peer TUs and the ported helper libraries; we call them
 * through the internal header / helper headers only.
 */

#include "rocke/helper_rocke.helpers.scan.h" /* rocke_lds_zero_i32 */
#include "rocke/instance_moe_sorting.h"
#include "rocke/instance_moe_sorting_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

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
bool rocke_moe_sort_hist_prologue(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_moe_sort_is_valid_spec_impl(ctx->spec, ctx->arch, reason, sizeof(reason), NULL))
    {
        /* raise ValueError(f"invalid moe_sorting spec: {why}") -- the build
         * entries call this prologue inside a ckc::guard_builder boundary, so
         * the throwing rocke_i_set_err records the exact Python message text and
         * status on the builder; the bare return below is dead after the throw,
         * mirroring the other instance build paths. */
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid moe_sorting spec: %s", reason);
        return false;
    }

    /* BS = spec.block_size ; E = spec.experts */
    ctx->BS = ctx->spec->block_size;
    ctx->E = ctx->spec->experts;

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)ctx->BS);

    /* TopkIds = b.param("TopkIds", PtrType(I32,"global"),
     *                   noalias=True, readonly=True, align=4) */
    {
        rocke_param_opts_t opts = {0};
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 4;
        opts.align_set = true;
        ctx->TopkIds = rocke_b_param(b, "TopkIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);
    }

    /* Hist = b.param("Hist", PtrType(I32,"global"), align=4) */
    {
        rocke_param_opts_t opts = {0};
        opts.align = 4;
        opts.align_set = true;
        ctx->Hist = rocke_b_param(b, "Hist", rocke_ptr_type(b, rocke_i32(), "global"), &opts);
    }

    /* num_pairs = b.param("num_pairs", I32) */
    ctx->num_pairs = rocke_b_param(b, "num_pairs", rocke_i32(), NULL);
    /* num_experts = b.param("num_experts", I32) */
    ctx->num_experts = rocke_b_param(b, "num_experts", rocke_i32(), NULL);

    /* tid = b.thread_id_x() ; bid = b.block_id_x() */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->bid = rocke_b_block_id_x(b);

    /* pair_idx = b.add(b.mul(bid, b.const_i32(BS)), tid) */
    ctx->pair_idx = rocke_b_add(
        b, rocke_b_mul(b, ctx->bid, rocke_b_const_i32(b, (int64_t)ctx->BS)), ctx->tid);

    return rocke_ir_builder_ok(b);
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
void rocke_moe_sort_hist_block_histogram(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* lds_hist = b.smem_alloc(I32, [E], name_hint="lds_hist") */
    {
        int shape[1] = {ctx->E};
        ctx->lds_hist = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "lds_hist");
    }

    /* lds_zero_i32(b, lds_hist, tid=tid, block_size=BS, length=E) */
    rocke_lds_zero_i32(b, ctx->lds_hist, ctx->tid, ctx->BS, ctx->E);

    /* in_bounds = b.cmp_lt(pair_idx, num_pairs) */
    ctx->in_bounds = rocke_b_cmp_lt(b, ctx->pair_idx, ctx->num_pairs);

    /* with b.scf_if(in_bounds): */
    {
        rocke_if_t gate = rocke_b_scf_if(b, ctx->in_bounds);
        rocke_b_region_enter(b, gate.then_region);

        /* eid, valid_e = _decode_expert_load(b, TopkIds, pair_idx, num_experts) */
        rocke_moe_sort_decode_expert_load(
            b, ctx->TopkIds, ctx->pair_idx, ctx->num_experts, &ctx->eid, &ctx->valid_e);

        /* with b.scf_if(valid_e): */
        {
            rocke_if_t vgate = rocke_b_scf_if(b, ctx->valid_e);
            rocke_b_region_enter(b, vgate.then_region);

            /* b.lds_atomic_add(lds_hist, [eid], b.const_i32(1)) */
            {
                rocke_value_t* indices[1] = {ctx->eid};
                rocke_b_lds_atomic_add(b, ctx->lds_hist, indices, 1, rocke_b_const_i32(b, 1), NULL);
            }

            rocke_b_region_leave(b);
        }

        rocke_b_region_leave(b);
    }

    /* b.sync() */
    rocke_b_sync(b);
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
rocke_kernel_def_t* rocke_moe_sort_hist_merge_to_global(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* c_E = b.const_i32(E) */
    ctx->c_E = rocke_b_const_i32(b, (int64_t)ctx->E);

    /* in_bin = b.cmp_lt(tid, c_E) */
    rocke_value_t* in_bin = rocke_b_cmp_lt(b, ctx->tid, ctx->c_E);

    /* with b.scf_if(in_bin): */
    {
        rocke_if_t gate = rocke_b_scf_if(b, in_bin);
        rocke_b_region_enter(b, gate.then_region);

        /* cnt = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0) */
        rocke_value_t* tid_idx[1] = {ctx->tid};
        rocke_value_t* loaded = rocke_b_smem_load_vN(b, ctx->lds_hist, tid_idx, 1, rocke_i32(), 1);
        rocke_value_t* cnt = rocke_b_vec_extract(b, loaded, 0);

        /* with b.scf_if(b.cmp_gt(cnt, b.const_i32(0))): */
        {
            rocke_if_t cgate = rocke_b_scf_if(b, rocke_b_cmp_gt(b, cnt, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, cgate.then_region);

            /* b.global_atomic_add(Hist, tid, cnt) */
            rocke_b_global_atomic_add(b, ctx->Hist, ctx->tid, cnt, NULL);

            rocke_b_region_leave(b);
        }

        rocke_b_region_leave(b);
    }

    /* return b.kernel */
    return b->kernel;
}
