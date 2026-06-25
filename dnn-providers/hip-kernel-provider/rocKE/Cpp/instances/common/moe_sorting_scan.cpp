// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_scan.c.c -- C99 port of the SCAN
 * kernel phase functions of build_moe_sort_scan
 * (ck_dsl/instances/common/moe_sorting.py, lines 325-439).
 *
 * Implements the three scan phase functions declared in
 * ckc/instance_moe_sorting_internal.h:
 *   ckc_moe_sort_scan_prologue   (lines 363-384)
 *   ckc_moe_sort_scan_wave_path  (lines 386-418, E <= wave_size)
 *   ckc_moe_sort_scan_lds_path   (lines 419-439, E > wave_size)
 *
 * Each function reproduces its Python builder-call sequence byte-faithfully and
 * mutates only the shared ckc_moe_sort_ctx_t; peers (the spec gate, the wave
 * Kogge-Stone helper) are reached through the internal header. Bound to
 * ckc/ir.h's public ckc_b_* surface and the ported scan helper.
 */
#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.helpers.scan.h"
#include "ckc/instance_moe_sorting.h"
#include "ckc/instance_moe_sorting_internal.h"
#include "ckc/ir.h"

/* ---------------------------------------------------------------------
 * Prologue (Python lines 363-384).
 *
 *   ok, why = is_valid_spec(spec, arch)
 *   if not ok: raise ValueError(...)
 *   wave_size = ArchTarget.from_gfx(arch).wave_size
 *   BS = spec.block_size; E = spec.experts
 *   b = IRBuilder(spec.kernel_name("scan"))   # already init'd by the driver
 *   b.kernel.attrs["max_workgroup_size"] = BS
 *   Hist    = b.param("Hist",    PtrType(I32,"global"), noalias=True, readonly=True, align=4)
 *   Offsets = b.param("Offsets", PtrType(I32,"global"), writeonly=True, align=4)
 *   Counts  = b.param("Counts",  PtrType(I32,"global"), writeonly=True, align=4)
 *   _       = b.param("num_experts", I32)
 *   tid       = b.thread_id_x()
 *   c_E       = b.const_i32(E)
 *   in_bounds = b.cmp_lt(tid, c_E)
 * --------------------------------------------------------------------- */
bool ckc_moe_sort_scan_prologue(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* is_valid_spec(spec, arch) gate; resolves wave_size on accept. */
    char reason[CKC_ERR_MSG_CAP];
    int wave_size = 0;
    if(!ckc_moe_sort_is_valid_spec_impl(ctx->spec, ctx->arch, reason, sizeof(reason), &wave_size))
    {
        /* Python: raise ValueError(f"invalid moe_sorting spec: {why}"). Route
         * through the builder sticky error so callers observe the failure. */
        ckc_attr_set_str(b, &b->kernel->attrs, "_moe_sort_invalid_spec", reason);
        b->status = CKC_ERR_VALUE;
        return false;
    }
    ctx->wave_size = wave_size;

    ctx->BS = ctx->spec->block_size;
    ctx->E = ctx->spec->experts;

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    const ckc_type_t* ptr_i32 = ckc_ptr_type(b, ckc_i32(), "global");

    /* Hist = b.param("Hist", PtrType(I32,"global"), noalias=True, readonly=True, align=4) */
    ckc_param_opts_t hist_opts = {0};
    hist_opts.noalias = true;
    hist_opts.noalias_set = true;
    hist_opts.readonly = true;
    hist_opts.readonly_set = true;
    hist_opts.align = 4;
    hist_opts.align_set = true;
    ctx->Hist = ckc_b_param(b, "Hist", ptr_i32, &hist_opts);

    /* Offsets = b.param("Offsets", PtrType(I32,"global"), writeonly=True, align=4) */
    ckc_param_opts_t off_opts = {0};
    off_opts.writeonly = true;
    off_opts.writeonly_set = true;
    off_opts.align = 4;
    off_opts.align_set = true;
    ctx->Offsets = ckc_b_param(b, "Offsets", ptr_i32, &off_opts);

    /* Counts = b.param("Counts", PtrType(I32,"global"), writeonly=True, align=4) */
    ckc_param_opts_t cnt_opts = {0};
    cnt_opts.writeonly = true;
    cnt_opts.writeonly_set = true;
    cnt_opts.align = 4;
    cnt_opts.align_set = true;
    ctx->Counts = ckc_b_param(b, "Counts", ptr_i32, &cnt_opts);

    /* _ = b.param("num_experts", I32)  -- declared for ABI, retained in ctx. */
    ctx->num_experts = ckc_b_param(b, "num_experts", ckc_i32(), NULL);

    /* tid = b.thread_id_x() */
    ctx->tid = ckc_b_thread_id_x(b);
    /* c_E = b.const_i32(E) */
    ctx->c_E = ckc_b_const_i32(b, ctx->E);
    /* in_bounds = b.cmp_lt(tid, c_E) */
    ctx->in_bounds = ckc_b_cmp_lt(b, ctx->tid, ctx->c_E);

    return ckc_ir_builder_ok(b);
}

/* ---------------------------------------------------------------------
 * Wave path (Python lines 386-418, taken when E <= wave_size).
 *
 *   safe_idx = b.select(in_bounds, tid, b.const_i32(0))
 *   v = b.global_load_i32(Hist, safe_idx)
 *   v = b.select(in_bounds, v, b.const_i32(0))
 *   with b.scf_if(in_bounds):
 *       b.global_store(Counts, tid, v, align=4)
 *   inclusive = _wave_kogge_stone_scan_i32(b, v, length=E, lane_id=tid)
 *   prev_lane = b.select(b.cmp_gt(tid, b.const_i32(0)),
 *                        b.sub(tid, b.const_i32(1)), b.const_i32(0))
 *   addr = b.shl(prev_lane, b.const_i32(2))
 *   shifted = b.ds_bpermute(addr, inclusive)
 *   excl = b.select(b.cmp_gt(tid, b.const_i32(0)), shifted, b.const_i32(0))
 *   with b.scf_if(in_bounds):
 *       b.global_store(Offsets, tid, excl, align=4)
 *   return b.kernel
 * --------------------------------------------------------------------- */
ckc_kernel_def_t* ckc_moe_sort_scan_wave_path(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* 1) Per-lane load of the histogram. OOB lanes carry 0. */
    /* safe_idx = b.select(in_bounds, tid, b.const_i32(0)) */
    ckc_value_t* safe_idx = ckc_b_select(b, ctx->in_bounds, ctx->tid, ckc_b_const_i32(b, 0));
    /* v = b.global_load_i32(Hist, safe_idx) */
    ckc_value_t* v = ckc_b_global_load_i32(b, ctx->Hist, safe_idx, 0);
    /* v = b.select(in_bounds, v, b.const_i32(0)) */
    v = ckc_b_select(b, ctx->in_bounds, v, ckc_b_const_i32(b, 0));

    /* 2) Counts mirror unchanged. */
    /* with b.scf_if(in_bounds): b.global_store(Counts, tid, v, align=4) */
    {
        ckc_if_t if_ib = ckc_b_scf_if(b, ctx->in_bounds);
        ckc_b_region_enter(b, if_ib.then_region);
        ckc_b_global_store(b, ctx->Counts, ctx->tid, v, 4);
        ckc_b_region_leave(b);
    }

    /* 3) Inclusive Kogge-Stone over up to wave_size lanes. */
    /* inclusive = _wave_kogge_stone_scan_i32(b, v, length=E, lane_id=tid) */
    ckc_value_t* inclusive = ckc_moe_sort_wave_kogge_stone_scan_i32(b, v, ctx->E, ctx->tid);

    /* 4) Inclusive -> exclusive: one ds_bpermute right-shift, set lane 0 to 0. */
    /* prev_lane = b.select(b.cmp_gt(tid, b.const_i32(0)),
     *                      b.sub(tid, b.const_i32(1)), b.const_i32(0))
     * Python evaluates select() args left-to-right: the cmp_gt condition emits
     * its SSA temp BEFORE the b.sub. C leaves argument evaluation order
     * unspecified, so hoist both into statements in Python order to keep SSA
     * value ids byte-identical (otherwise cmp_gt/sub swap, e.g. %gt40/%sub42
     * -> %sub41/%gt43). */
    ckc_value_t* prev_gt = ckc_b_cmp_gt(b, ctx->tid, ckc_b_const_i32(b, 0));
    ckc_value_t* prev_sub = ckc_b_sub(b, ctx->tid, ckc_b_const_i32(b, 1));
    ckc_value_t* prev_lane = ckc_b_select(b, prev_gt, prev_sub, ckc_b_const_i32(b, 0));
    /* addr = b.shl(prev_lane, b.const_i32(2)) */
    ckc_value_t* addr = ckc_b_shl(b, prev_lane, ckc_b_const_i32(b, 2));
    /* shifted = b.ds_bpermute(addr, inclusive) */
    ckc_value_t* shifted = ckc_b_ds_bpermute(b, addr, inclusive);
    /* excl = b.select(b.cmp_gt(tid, b.const_i32(0)), shifted, b.const_i32(0))
     * Python evaluates the cmp_gt (and its inner const_i32(0)) BEFORE the
     * trailing const_i32(0) select arg. C leaves argument evaluation order
     * unspecified, so hoist the cmp_gt to pin const->cmp_gt->const and keep
     * the cmp_gt SSA id byte-identical (otherwise it drifts +1, e.g.
     * %gt49 -> %gt50). */
    ckc_value_t* excl_gt = ckc_b_cmp_gt(b, ctx->tid, ckc_b_const_i32(b, 0));
    ckc_value_t* excl = ckc_b_select(b, excl_gt, shifted, ckc_b_const_i32(b, 0));

    /* with b.scf_if(in_bounds): b.global_store(Offsets, tid, excl, align=4) */
    {
        ckc_if_t if_ib = ckc_b_scf_if(b, ctx->in_bounds);
        ckc_b_region_enter(b, if_ib.then_region);
        ckc_b_global_store(b, ctx->Offsets, ctx->tid, excl, 4);
        ckc_b_region_leave(b);
    }

    /* return b.kernel */
    return b->kernel;
}

/* ---------------------------------------------------------------------
 * LDS fallback path (Python lines 419-439, taken when E > wave_size).
 *
 *   lds = b.smem_alloc(I32, [E], name_hint="lds_scan")
 *   with b.scf_if(in_bounds):
 *       v = b.global_load_i32(Hist, tid)
 *       b.smem_store_vN(lds, [tid], v, 1)
 *       b.global_store(Counts, tid, v, align=4)
 *   b.sync()
 *   block_exclusive_scan_i32(b, lds, tid=tid, block_size=BS, length=E)
 *   with b.scf_if(in_bounds):
 *       v = b.vec_extract(b.smem_load_vN(lds, tid, dtype=I32, n=1), 0)
 *       b.global_store(Offsets, tid, v, align=4)
 *   return b.kernel
 * --------------------------------------------------------------------- */
ckc_kernel_def_t* ckc_moe_sort_scan_lds_path(ckc_moe_sort_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;

    /* lds = b.smem_alloc(I32, [E], name_hint="lds_scan") */
    int shape[1] = {ctx->E};
    ctx->lds_scan = ckc_b_smem_alloc(b, ckc_i32(), shape, 1, "lds_scan");

    /* 1) Copy Hist -> LDS (and into Counts unchanged).
     * with b.scf_if(in_bounds):
     *     v = b.global_load_i32(Hist, tid)
     *     b.smem_store_vN(lds, [tid], v, 1)
     *     b.global_store(Counts, tid, v, align=4) */
    {
        ckc_if_t if_ib = ckc_b_scf_if(b, ctx->in_bounds);
        ckc_b_region_enter(b, if_ib.then_region);
        ckc_value_t* v = ckc_b_global_load_i32(b, ctx->Hist, ctx->tid, 0);
        ckc_value_t* idx[1] = {ctx->tid};
        ckc_b_smem_store_vN(b, ctx->lds_scan, idx, 1, v, 1);
        ckc_b_global_store(b, ctx->Counts, ctx->tid, v, 4);
        ckc_b_region_leave(b);
    }
    /* b.sync() */
    ckc_b_sync(b);

    /* 2) In-place exclusive scan in LDS.
     * block_exclusive_scan_i32(b, lds, tid=tid, block_size=BS, length=E) */
    ckc_block_exclusive_scan_i32(b, ctx->lds_scan, ctx->tid, ctx->BS, ctx->E);

    /* 3) Copy LDS -> Offsets.
     * with b.scf_if(in_bounds):
     *     v = b.vec_extract(b.smem_load_vN(lds, tid, dtype=I32, n=1), 0)
     *     b.global_store(Offsets, tid, v, align=4) */
    {
        ckc_if_t if_ib = ckc_b_scf_if(b, ctx->in_bounds);
        ckc_b_region_enter(b, if_ib.then_region);
        ckc_value_t* idx[1] = {ctx->tid};
        ckc_value_t* loaded = ckc_b_smem_load_vN(b, ctx->lds_scan, idx, 1, ckc_i32(), 1);
        ckc_value_t* v = ckc_b_vec_extract(b, loaded, 0);
        ckc_b_global_store(b, ctx->Offsets, ctx->tid, v, 4);
        ckc_b_region_leave(b);
    }

    /* return b.kernel */
    return b->kernel;
}
