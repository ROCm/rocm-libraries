// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_scan.c.c -- C99 port of the SCAN
 * kernel phase functions of build_moe_sort_scan
 * (rocke/instances/common/moe_sorting.py, lines 325-439).
 *
 * Implements the three scan phase functions declared in
 * rocke/instance_moe_sorting_internal.h:
 *   rocke_moe_sort_scan_prologue   (lines 363-384)
 *   rocke_moe_sort_scan_wave_path  (lines 386-418, E <= wave_size)
 *   rocke_moe_sort_scan_lds_path   (lines 419-439, E > wave_size)
 *
 * Each function reproduces its Python builder-call sequence byte-faithfully and
 * mutates only the shared rocke_moe_sort_ctx_t; peers (the spec gate, the wave
 * Kogge-Stone helper) are reached through the internal header. Bound to
 * rocke/ir.h's public rocke_b_* surface and the ported scan helper.
 */
#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.scan.h"
#include "rocke/instance_moe_sorting.h"
#include "rocke/instance_moe_sorting_internal.h"
#include "rocke/ir.h"

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
bool rocke_moe_sort_scan_prologue(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* is_valid_spec(spec, arch) gate; resolves wave_size on accept. */
    char reason[ROCKE_ERR_MSG_CAP];
    int wave_size = 0;
    if(!rocke_moe_sort_is_valid_spec_impl(ctx->spec, ctx->arch, reason, sizeof(reason), &wave_size))
    {
        /* Python: raise ValueError(f"invalid moe_sorting spec: {why}"). Route
         * through the builder sticky error so callers observe the failure. */
        rocke_attr_set_str(b, &b->kernel->attrs, "_moe_sort_invalid_spec", reason);
        b->status = ROCKE_ERR_VALUE;
        return false;
    }
    ctx->wave_size = wave_size;

    ctx->BS = ctx->spec->block_size;
    ctx->E = ctx->spec->experts;

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->BS);

    const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");

    /* Hist = b.param("Hist", PtrType(I32,"global"), noalias=True, readonly=True, align=4) */
    rocke_param_opts_t hist_opts = {0};
    hist_opts.noalias = true;
    hist_opts.noalias_set = true;
    hist_opts.readonly = true;
    hist_opts.readonly_set = true;
    hist_opts.align = 4;
    hist_opts.align_set = true;
    ctx->Hist = rocke_b_param(b, "Hist", ptr_i32, &hist_opts);

    /* Offsets = b.param("Offsets", PtrType(I32,"global"), writeonly=True, align=4) */
    rocke_param_opts_t off_opts = {0};
    off_opts.writeonly = true;
    off_opts.writeonly_set = true;
    off_opts.align = 4;
    off_opts.align_set = true;
    ctx->Offsets = rocke_b_param(b, "Offsets", ptr_i32, &off_opts);

    /* Counts = b.param("Counts", PtrType(I32,"global"), writeonly=True, align=4) */
    rocke_param_opts_t cnt_opts = {0};
    cnt_opts.writeonly = true;
    cnt_opts.writeonly_set = true;
    cnt_opts.align = 4;
    cnt_opts.align_set = true;
    ctx->Counts = rocke_b_param(b, "Counts", ptr_i32, &cnt_opts);

    /* _ = b.param("num_experts", I32)  -- declared for ABI, retained in ctx. */
    ctx->num_experts = rocke_b_param(b, "num_experts", rocke_i32(), NULL);

    /* tid = b.thread_id_x() */
    ctx->tid = rocke_b_thread_id_x(b);
    /* c_E = b.const_i32(E) */
    ctx->c_E = rocke_b_const_i32(b, ctx->E);
    /* in_bounds = b.cmp_lt(tid, c_E) */
    ctx->in_bounds = rocke_b_cmp_lt(b, ctx->tid, ctx->c_E);

    return rocke_ir_builder_ok(b);
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
rocke_kernel_def_t* rocke_moe_sort_scan_wave_path(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* 1) Per-lane load of the histogram. OOB lanes carry 0. */
    /* safe_idx = b.select(in_bounds, tid, b.const_i32(0)) */
    rocke_value_t* safe_idx = rocke_b_select(b, ctx->in_bounds, ctx->tid, rocke_b_const_i32(b, 0));
    /* v = b.global_load_i32(Hist, safe_idx) */
    rocke_value_t* v = rocke_b_global_load_i32(b, ctx->Hist, safe_idx, 0);
    /* v = b.select(in_bounds, v, b.const_i32(0)) */
    v = rocke_b_select(b, ctx->in_bounds, v, rocke_b_const_i32(b, 0));

    /* 2) Counts mirror unchanged. */
    /* with b.scf_if(in_bounds): b.global_store(Counts, tid, v, align=4) */
    {
        rocke_if_t if_ib = rocke_b_scf_if(b, ctx->in_bounds);
        rocke_b_region_enter(b, if_ib.then_region);
        rocke_b_global_store(b, ctx->Counts, ctx->tid, v, 4);
        rocke_b_region_leave(b);
    }

    /* 3) Inclusive Kogge-Stone over up to wave_size lanes. */
    /* inclusive = _wave_kogge_stone_scan_i32(b, v, length=E, lane_id=tid) */
    rocke_value_t* inclusive = rocke_moe_sort_wave_kogge_stone_scan_i32(b, v, ctx->E, ctx->tid);

    /* 4) Inclusive -> exclusive: one ds_bpermute right-shift, set lane 0 to 0. */
    /* prev_lane = b.select(b.cmp_gt(tid, b.const_i32(0)),
     *                      b.sub(tid, b.const_i32(1)), b.const_i32(0))
     * Python evaluates select() args left-to-right: the cmp_gt condition emits
     * its SSA temp BEFORE the b.sub. C leaves argument evaluation order
     * unspecified, so hoist both into statements in Python order to keep SSA
     * value ids byte-identical (otherwise cmp_gt/sub swap, e.g. %gt40/%sub42
     * -> %sub41/%gt43). */
    rocke_value_t* prev_gt = rocke_b_cmp_gt(b, ctx->tid, rocke_b_const_i32(b, 0));
    rocke_value_t* prev_sub = rocke_b_sub(b, ctx->tid, rocke_b_const_i32(b, 1));
    rocke_value_t* prev_lane = rocke_b_select(b, prev_gt, prev_sub, rocke_b_const_i32(b, 0));
    /* addr = b.shl(prev_lane, b.const_i32(2)) */
    rocke_value_t* addr = rocke_b_shl(b, prev_lane, rocke_b_const_i32(b, 2));
    /* shifted = b.ds_bpermute(addr, inclusive) */
    rocke_value_t* shifted = rocke_b_ds_bpermute(b, addr, inclusive);
    /* excl = b.select(b.cmp_gt(tid, b.const_i32(0)), shifted, b.const_i32(0))
     * Python evaluates the cmp_gt (and its inner const_i32(0)) BEFORE the
     * trailing const_i32(0) select arg. C leaves argument evaluation order
     * unspecified, so hoist the cmp_gt to pin const->cmp_gt->const and keep
     * the cmp_gt SSA id byte-identical (otherwise it drifts +1, e.g.
     * %gt49 -> %gt50). */
    rocke_value_t* excl_gt = rocke_b_cmp_gt(b, ctx->tid, rocke_b_const_i32(b, 0));
    rocke_value_t* excl = rocke_b_select(b, excl_gt, shifted, rocke_b_const_i32(b, 0));

    /* with b.scf_if(in_bounds): b.global_store(Offsets, tid, excl, align=4) */
    {
        rocke_if_t if_ib = rocke_b_scf_if(b, ctx->in_bounds);
        rocke_b_region_enter(b, if_ib.then_region);
        rocke_b_global_store(b, ctx->Offsets, ctx->tid, excl, 4);
        rocke_b_region_leave(b);
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
rocke_kernel_def_t* rocke_moe_sort_scan_lds_path(rocke_moe_sort_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* lds = b.smem_alloc(I32, [E], name_hint="lds_scan") */
    int shape[1] = {ctx->E};
    ctx->lds_scan = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "lds_scan");

    /* 1) Copy Hist -> LDS (and into Counts unchanged).
     * with b.scf_if(in_bounds):
     *     v = b.global_load_i32(Hist, tid)
     *     b.smem_store_vN(lds, [tid], v, 1)
     *     b.global_store(Counts, tid, v, align=4) */
    {
        rocke_if_t if_ib = rocke_b_scf_if(b, ctx->in_bounds);
        rocke_b_region_enter(b, if_ib.then_region);
        rocke_value_t* v = rocke_b_global_load_i32(b, ctx->Hist, ctx->tid, 0);
        rocke_value_t* idx[1] = {ctx->tid};
        rocke_b_smem_store_vN(b, ctx->lds_scan, idx, 1, v, 1);
        rocke_b_global_store(b, ctx->Counts, ctx->tid, v, 4);
        rocke_b_region_leave(b);
    }
    /* b.sync() */
    rocke_b_sync(b);

    /* 2) In-place exclusive scan in LDS.
     * block_exclusive_scan_i32(b, lds, tid=tid, block_size=BS, length=E) */
    rocke_block_exclusive_scan_i32(b, ctx->lds_scan, ctx->tid, ctx->BS, ctx->E);

    /* 3) Copy LDS -> Offsets.
     * with b.scf_if(in_bounds):
     *     v = b.vec_extract(b.smem_load_vN(lds, tid, dtype=I32, n=1), 0)
     *     b.global_store(Offsets, tid, v, align=4) */
    {
        rocke_if_t if_ib = rocke_b_scf_if(b, ctx->in_bounds);
        rocke_b_region_enter(b, if_ib.then_region);
        rocke_value_t* idx[1] = {ctx->tid};
        rocke_value_t* loaded = rocke_b_smem_load_vN(b, ctx->lds_scan, idx, 1, rocke_i32(), 1);
        rocke_value_t* v = rocke_b_vec_extract(b, loaded, 0);
        rocke_b_global_store(b, ctx->Offsets, ctx->tid, v, 4);
        rocke_b_region_leave(b);
    }

    /* return b.kernel */
    return b->kernel;
}
