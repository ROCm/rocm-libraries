// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_persistent.c.c -- C99 port of the
 * PERSISTENT single-CTA fused kernel phase of
 * ck_dsl/instances/common/moe_sorting.py
 * (build_moe_sort_persistent, Python lines 613-741).
 *
 * Implements ONLY the four persistent phase functions declared in
 * ckc/instance_moe_sorting_internal.h:
 *   ckc_moe_sort_persistent_prologue   (lines 639-682)
 *   ckc_moe_sort_persistent_histogram  (lines 684-704)
 *   ckc_moe_sort_persistent_scan       (lines 706-713)
 *   ckc_moe_sort_persistent_scatter    (lines 715-741)
 *
 * The shared module helpers (ckc_moe_sort_is_valid_spec_impl,
 * ckc_moe_sort_decode_expert_load, ckc_moe_sort_decode_pair_token_topk), the
 * spec accessors, and the scan helpers (ckc_lds_zero_i32,
 * ckc_block_exclusive_scan_i32) are implemented by peer TUs and the ported
 * helper libraries; we call them through the internal header / helper headers
 * only.
 */

#include "ckc/instance_moe_sorting_internal.h"
#include "ckc/instance_moe_sorting.h"
#include "ckc/helper_ck_dsl.helpers.scan.h" /* ckc_lds_zero_i32, ckc_block_exclusive_scan_i32 */
#include "ckc/ir.h"

/* ===================================================================== *
 *  Prologue (Python lines 639-682).
 *
 *    ok, why = is_valid_spec(spec, arch)
 *    if not ok: raise ValueError(...)
 *    BS = spec.block_size
 *    E  = spec.experts
 *    NP = spec.total_pairs
 *    b = IRBuilder(spec.kernel_name("persistent"))   (already inited)
 *    b.kernel.attrs["max_workgroup_size"] = BS
 *    TopkIds        = b.param("TopkIds",  PtrType(I32,"global"),
 *                             noalias=True, readonly=True, align=4)
 *    TopkWeights    = b.param("TopkWeights", PtrType(F32,"global"),
 *                             noalias=True, readonly=True, align=4)
 *    Offsets        = b.param("Offsets", PtrType(I32,"global"), align=4)
 *    Counts         = b.param("Counts",  PtrType(I32,"global"),
 *                             writeonly=True, align=4)
 *    SortedTokenIds = b.param("SortedTokenIds", PtrType(I32,"global"),
 *                             writeonly=True, align=4)
 *    SortedTopkIds  = b.param("SortedTopkIds",  PtrType(I32,"global"),
 *                             writeonly=True, align=4)
 *    SortedWeights  = b.param("SortedWeights",  PtrType(F32,"global"),
 *                             writeonly=True, align=4)
 *    _tokens     = b.param("tokens", I32)
 *    _topk       = b.param("topk", I32)
 *    num_experts = b.param("num_experts", I32)
 *    tid   = b.thread_id_x()
 *    c_one = b.const_i32(1)
 *    c_zero= b.const_i32(0)
 *    c_E   = b.const_i32(E)
 *    c_BS  = b.const_i32(BS)
 *    c_NP  = b.const_i32(NP)
 *    pairs_per_thread = (NP + BS - 1) // BS   (stashed as n_pairs_per_thread)
 * ===================================================================== */
bool ckc_moe_sort_persistent_prologue(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    char reason[CKC_ERR_MSG_CAP];
    if(!ckc_moe_sort_is_valid_spec_impl(ctx->spec, ctx->arch, reason, sizeof(reason), NULL))
    {
        /* raise ValueError(f"invalid moe_sorting spec: {why}") -- route through
         * the sticky-error builder so callers observe the rejection. */
        b->status = CKC_ERR_VALUE;
        /* TODO(port): byte-identical "invalid moe_sorting spec: {why}"
         * formatting once the shared error-string convention is finalised. */
        return false;
    }

    /* BS = spec.block_size ; E = spec.experts ; NP = spec.total_pairs */
    ctx->BS   = ctx->spec->block_size;
    ctx->E    = ctx->spec->experts;
    ctx->NP   = ckc_moe_sorting_spec_total_pairs(ctx->spec);
    ctx->topk = ctx->spec->topk;

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)ctx->BS);

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

    /* Offsets = b.param("Offsets", PtrType(I32,"global"), align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.align            = 4;
        opts.align_set        = true;
        ctx->Offsets = ckc_b_param(b, "Offsets", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
    }

    /* Counts = b.param("Counts", PtrType(I32,"global"), writeonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.writeonly        = true;
        opts.writeonly_set    = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->Counts = ckc_b_param(b, "Counts", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
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

    /* SortedTopkIds = b.param("SortedTopkIds", PtrType(I32,"global"),
     *                         writeonly=True, align=4) */
    {
        ckc_param_opts_t opts = {0};
        opts.writeonly        = true;
        opts.writeonly_set    = true;
        opts.align            = 4;
        opts.align_set        = true;
        ctx->SortedTopkIds =
            ckc_b_param(b, "SortedTopkIds", ckc_ptr_type(b, ckc_i32(), "global"), &opts);
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

    /* _tokens = b.param("tokens", I32)  # noqa: F841 -- ABI */
    ctx->tokens = ckc_b_param(b, "tokens", ckc_i32(), NULL);
    /* _topk = b.param("topk", I32)  # noqa: F841 -- ABI; split uses spec.topk */
    ctx->topk_param = ckc_b_param(b, "topk", ckc_i32(), NULL);
    /* num_experts = b.param("num_experts", I32) */
    ctx->num_experts = ckc_b_param(b, "num_experts", ckc_i32(), NULL);

    /* tid = b.thread_id_x() */
    ctx->tid = ckc_b_thread_id_x(b);

    /* c_one = b.const_i32(1) */
    ctx->c_one = ckc_b_const_i32(b, 1);
    /* c_zero = b.const_i32(0) */
    ctx->c_zero = ckc_b_const_i32(b, 0);
    /* c_E = b.const_i32(E) */
    ctx->c_E = ckc_b_const_i32(b, (int64_t)ctx->E);
    /* c_BS = b.const_i32(BS) */
    ctx->c_BS = ckc_b_const_i32(b, (int64_t)ctx->BS);
    /* c_NP = b.const_i32(NP) */
    ctx->c_NP = ckc_b_const_i32(b, (int64_t)ctx->NP);

    /* pairs_per_thread = (NP + BS - 1) // BS */
    ctx->n_pairs_per_thread = (ctx->NP + ctx->BS - 1) / ctx->BS;

    return ckc_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Phase 1: LDS histogram + Counts store (Python lines 684-704).
 *
 *    lds_hist = b.smem_alloc(I32, [E], name_hint="lds_hist")
 *    lds_zero_i32(b, lds_hist, tid=tid, block_size=BS, length=E)
 *    pairs_per_thread = (NP + BS - 1) // BS
 *    for i in range(pairs_per_thread):
 *        pair_idx  = b.add(b.mul(b.const_i32(i), c_BS), tid)
 *        in_bounds = b.cmp_lt(pair_idx, c_NP)
 *        with b.scf_if(in_bounds):
 *            eid     = b.global_load_i32(TopkIds, pair_idx)
 *            valid_e = b.land(b.cmp_ge(eid, c_zero),
 *                             b.cmp_lt(eid, num_experts))
 *            with b.scf_if(valid_e):
 *                b.lds_atomic_add(lds_hist, [eid], c_one)
 *    b.sync()
 *    with b.scf_if(b.cmp_lt(tid, c_E)):
 *        cnt = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0)
 *        b.global_store(Counts, tid, cnt, align=4)
 * ===================================================================== */
void ckc_moe_sort_persistent_histogram(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    int i;

    /* lds_hist = b.smem_alloc(I32, [E], name_hint="lds_hist") */
    {
        int shape[1]  = {ctx->E};
        ctx->lds_hist = ckc_b_smem_alloc(b, ckc_i32(), shape, 1, "lds_hist");
    }

    /* lds_zero_i32(b, lds_hist, tid=tid, block_size=BS, length=E) */
    ckc_lds_zero_i32(b, ctx->lds_hist, ctx->tid, ctx->BS, ctx->E);

    /* for i in range(pairs_per_thread): ... */
    for(i = 0; i < ctx->n_pairs_per_thread; ++i)
    {
        /* pair_idx = b.add(b.mul(b.const_i32(i), c_BS), tid) */
        ctx->pair_idx =
            ckc_b_add(b, ckc_b_mul(b, ckc_b_const_i32(b, (int64_t)i), ctx->c_BS), ctx->tid);

        /* in_bounds = b.cmp_lt(pair_idx, c_NP) */
        ctx->in_bounds = ckc_b_cmp_lt(b, ctx->pair_idx, ctx->c_NP);

        /* with b.scf_if(in_bounds): */
        {
            ckc_if_t gate = ckc_b_scf_if(b, ctx->in_bounds);
            ckc_b_region_enter(b, gate.then_region);

            /* eid = b.global_load_i32(TopkIds, pair_idx) */
            ctx->eid = ckc_b_global_load_i32(b, ctx->TopkIds, ctx->pair_idx, 0);

            /* valid_e = b.land(b.cmp_ge(eid, c_zero),
             *                  b.cmp_lt(eid, num_experts)) */
            ctx->valid_e = ckc_b_land(b,
                                      ckc_b_cmp_ge(b, ctx->eid, ctx->c_zero),
                                      ckc_b_cmp_lt(b, ctx->eid, ctx->num_experts));

            /* with b.scf_if(valid_e): */
            {
                ckc_if_t vgate = ckc_b_scf_if(b, ctx->valid_e);
                ckc_b_region_enter(b, vgate.then_region);

                /* b.lds_atomic_add(lds_hist, [eid], c_one) */
                {
                    ckc_value_t* indices[1] = {ctx->eid};
                    ckc_b_lds_atomic_add(b, ctx->lds_hist, indices, 1, ctx->c_one, NULL);
                }

                ckc_b_region_leave(b);
            }

            ckc_b_region_leave(b);
        }
    }

    /* b.sync() */
    ckc_b_sync(b);

    /* Write Counts to global before the scan overwrites lds_hist.
     * with b.scf_if(b.cmp_lt(tid, c_E)): */
    {
        ckc_if_t gate = ckc_b_scf_if(b, ckc_b_cmp_lt(b, ctx->tid, ctx->c_E));
        ckc_b_region_enter(b, gate.then_region);

        /* cnt = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0) */
        {
            ckc_value_t* tid_idx[1] = {ctx->tid};
            ckc_value_t* loaded = ckc_b_smem_load_vN(b, ctx->lds_hist, tid_idx, 1, ckc_i32(), 1);
            ckc_value_t* cnt    = ckc_b_vec_extract(b, loaded, 0);

            /* b.global_store(Counts, tid, cnt, align=4) */
            ckc_b_global_store(b, ctx->Counts, ctx->tid, cnt, 4);
        }

        ckc_b_region_leave(b);
    }
}

/* ===================================================================== *
 *  Phase 2: in-place exclusive scan + Offsets store (Python lines 706-713).
 *
 *    block_exclusive_scan_i32(b, lds_hist, tid=tid, block_size=BS, length=E)
 *    # block_exclusive_scan_i32 finishes with a b.sync() already.
 *    with b.scf_if(b.cmp_lt(tid, c_E)):
 *        off = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0)
 *        b.global_store(Offsets, tid, off, align=4)
 * ===================================================================== */
void ckc_moe_sort_persistent_scan(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* block_exclusive_scan_i32(b, lds_hist, tid=tid, block_size=BS, length=E) */
    ckc_block_exclusive_scan_i32(b, ctx->lds_hist, ctx->tid, ctx->BS, ctx->E);

    /* Write Offsets to global.
     * with b.scf_if(b.cmp_lt(tid, c_E)): */
    {
        ckc_if_t gate = ckc_b_scf_if(b, ckc_b_cmp_lt(b, ctx->tid, ctx->c_E));
        ckc_b_region_enter(b, gate.then_region);

        /* off = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0) */
        {
            ckc_value_t* tid_idx[1] = {ctx->tid};
            ckc_value_t* loaded = ckc_b_smem_load_vN(b, ctx->lds_hist, tid_idx, 1, ckc_i32(), 1);
            ckc_value_t* off    = ckc_b_vec_extract(b, loaded, 0);

            /* b.global_store(Offsets, tid, off, align=4) */
            ckc_b_global_store(b, ctx->Offsets, ctx->tid, off, 4);
        }

        ckc_b_region_leave(b);
    }
}

/* ===================================================================== *
 *  Phase 3: LDS scatter + return (Python lines 715-741).
 *
 *    lds_counter = b.smem_alloc(I32, [E], name_hint="lds_counter")
 *    lds_zero_i32(b, lds_counter, tid=tid, block_size=BS, length=E)
 *    for i in range(pairs_per_thread):
 *        pair_idx  = b.add(b.mul(b.const_i32(i), c_BS), tid)
 *        in_bounds = b.cmp_lt(pair_idx, c_NP)
 *        with b.scf_if(in_bounds):
 *            eid     = b.global_load_i32(TopkIds, pair_idx)
 *            valid_e = b.land(b.cmp_ge(eid, c_zero),
 *                             b.cmp_lt(eid, num_experts))
 *            with b.scf_if(valid_e):
 *                local_off  = b.lds_atomic_add(lds_counter, [eid], c_one)
 *                base       = b.vec_extract(
 *                                 b.smem_load_vN(lds_hist, eid, dtype=I32, n=1), 0)
 *                global_off = b.add(base, local_off)
 *                t_idx, k_idx = _decode_pair_token_topk(b, pair_idx, spec.topk)
 *                w = b.global_load_f32(TopkWeights, pair_idx)
 *                b.global_store(SortedTokenIds, global_off, t_idx, align=4)
 *                b.global_store(SortedTopkIds,  global_off, k_idx, align=4)
 *                b.global_store(SortedWeights,  global_off, w,     align=4)
 *    return b.kernel
 * ===================================================================== */
ckc_kernel_def_t* ckc_moe_sort_persistent_scatter(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    int i;

    /* lds_counter = b.smem_alloc(I32, [E], name_hint="lds_counter") */
    {
        int shape[1]     = {ctx->E};
        ctx->lds_counter = ckc_b_smem_alloc(b, ckc_i32(), shape, 1, "lds_counter");
    }

    /* lds_zero_i32(b, lds_counter, tid=tid, block_size=BS, length=E) */
    ckc_lds_zero_i32(b, ctx->lds_counter, ctx->tid, ctx->BS, ctx->E);

    /* for i in range(pairs_per_thread): ... */
    for(i = 0; i < ctx->n_pairs_per_thread; ++i)
    {
        /* pair_idx = b.add(b.mul(b.const_i32(i), c_BS), tid) */
        ctx->pair_idx =
            ckc_b_add(b, ckc_b_mul(b, ckc_b_const_i32(b, (int64_t)i), ctx->c_BS), ctx->tid);

        /* in_bounds = b.cmp_lt(pair_idx, c_NP) */
        ctx->in_bounds = ckc_b_cmp_lt(b, ctx->pair_idx, ctx->c_NP);

        /* with b.scf_if(in_bounds): */
        {
            ckc_if_t gate = ckc_b_scf_if(b, ctx->in_bounds);
            ckc_b_region_enter(b, gate.then_region);

            /* eid = b.global_load_i32(TopkIds, pair_idx) */
            ctx->eid = ckc_b_global_load_i32(b, ctx->TopkIds, ctx->pair_idx, 0);

            /* valid_e = b.land(b.cmp_ge(eid, c_zero),
             *                  b.cmp_lt(eid, num_experts)) */
            ctx->valid_e = ckc_b_land(b,
                                      ckc_b_cmp_ge(b, ctx->eid, ctx->c_zero),
                                      ckc_b_cmp_lt(b, ctx->eid, ctx->num_experts));

            /* with b.scf_if(valid_e): */
            {
                ckc_if_t vgate = ckc_b_scf_if(b, ctx->valid_e);
                ckc_b_region_enter(b, vgate.then_region);

                /* local_off = b.lds_atomic_add(lds_counter, [eid], c_one) */
                ckc_value_t* local_off;
                {
                    ckc_value_t* indices[1] = {ctx->eid};
                    local_off =
                        ckc_b_lds_atomic_add(b, ctx->lds_counter, indices, 1, ctx->c_one, NULL);
                }

                /* base = b.vec_extract(
                 *            b.smem_load_vN(lds_hist, eid, dtype=I32, n=1), 0) */
                ckc_value_t* base;
                {
                    ckc_value_t* eid_idx[1] = {ctx->eid};
                    ckc_value_t* loaded =
                        ckc_b_smem_load_vN(b, ctx->lds_hist, eid_idx, 1, ckc_i32(), 1);
                    base = ckc_b_vec_extract(b, loaded, 0);
                }

                /* global_off = b.add(base, local_off) */
                ckc_value_t* global_off = ckc_b_add(b, base, local_off);

                /* t_idx, k_idx = _decode_pair_token_topk(b, pair_idx, spec.topk) */
                ckc_moe_sort_decode_pair_token_topk(
                    b, ctx->pair_idx, ctx->topk, &ctx->t_idx, &ctx->k_idx);

                /* w = b.global_load_f32(TopkWeights, pair_idx) */
                ckc_value_t* w = ckc_b_global_load_f32(b, ctx->TopkWeights, ctx->pair_idx, 0);

                /* b.global_store(SortedTokenIds, global_off, t_idx, align=4) */
                ckc_b_global_store(b, ctx->SortedTokenIds, global_off, ctx->t_idx, 4);
                /* b.global_store(SortedTopkIds, global_off, k_idx, align=4) */
                ckc_b_global_store(b, ctx->SortedTopkIds, global_off, ctx->k_idx, 4);
                /* b.global_store(SortedWeights, global_off, w, align=4) */
                ckc_b_global_store(b, ctx->SortedWeights, global_off, w, 4);

                ckc_b_region_leave(b);
            }

            ckc_b_region_leave(b);
        }
    }

    /* return b.kernel */
    return b->kernel;
}
