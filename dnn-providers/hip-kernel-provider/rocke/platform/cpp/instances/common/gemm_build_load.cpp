// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_build_load.c -- one part-file of the chunked C99 port of
 * build_universal_gemm (rocke/instances/common/gemm_universal.py).
 *
 * SCOPE (this TU only):
 *   * rocke_gemm_build_populate_ctx -- the build prologue that POPULATES the
 *     shared rocke_gemm_build_ctx_t. Mirrors the Python enclosing-function body
 *     from `t = spec.tile` (line ~824) through the DirectToLDS plumbing
 *     (line ~1110), in the SAME order Python computes the locals, so the
 *     emitted SSA constants (c0, c_wave, ..., the DTL const_i32s) appear in the
 *     byte-identical order. The driver part-file fills the input/param fields
 *     (b, spec, arch, target, op, storage_dtype, is_wmma, A/Bp/C/M/N/K and the
 *     batched stride / active-tile params) BEFORE calling this; everything from
 *     the per-lane fragment widths onward is populated here.
 *   * rocke_gemm_emit_load_phase    (Python closure emit_load_phase, ~1112)
 *   * rocke_gemm_emit_frag_smem_load(Python closure _emit_frag_smem_load, ~1372)
 *   * rocke_gemm_emit_wmma_phase    (Python closure _emit_wmma_phase, ~1409)
 *
 * Peers (the module-level helpers + the other phase closures) are CALLED via
 * rocke/instance_gemm_internal.h and are defined in sibling part-files; this TU
 * never re-defines them.
 *
 * Faithfulness: every builder call below is in the same order, with the same
 * operands/attrs, as the corresponding Python line. Where the Python relies on
 * host-side helper objects (TensorView / TileWindow), this TU uses the C ports
 * in rocke/helper_rocke.helpers.tensor_view.h, which themselves reproduce the
 * Python builder-call sequence.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.grid.h" /* rocke_chiplet_aware_super_tile_dynamic */
#include "rocke/instance_gemm_internal.h"

/* The driver populates the param + environment fields then calls this; declared
 * here (not in the shared header, to keep that surface frozen) and referenced by
 * the driver part-file via an extern prototype of its own. */
void rocke_gemm_build_populate_ctx(rocke_gemm_build_ctx_t* ctx);

/* ===================================================================== *
 *  build prologue -- POPULATE rocke_gemm_build_ctx_t
 *
 *  Python span: gemm_universal.py lines ~824 .. ~1110.
 * ===================================================================== */
void rocke_gemm_build_populate_ctx(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    const rocke_gemm_tile_spec_t* t = &spec->tile;

    /* t = spec.tile
     * a_per_lane, b_per_lane, c_per_lane = _atom_frag_lengths(op) */
    rocke_gemm_atom_frag_lengths(ctx->op, &ctx->a_per_lane, &ctx->b_per_lane, &ctx->c_per_lane);

    /* block_m / block_n / block_k = t.tile_m / t.tile_n / t.tile_k */
    ctx->block_m = t->tile_m;
    ctx->block_n = t->tile_n;
    ctx->block_k = t->tile_k;

    /* Common geometry (SSA constants, in source order). */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c_wave = rocke_b_const_i32(b, spec->wave_size);
    ctx->c_warps_n = rocke_b_const_i32(b, t->warp_n);
    ctx->c_block_m = rocke_b_const_i32(b, ctx->block_m);
    ctx->c_block_n = rocke_b_const_i32(b, ctx->block_n);
    ctx->c_block_k = rocke_b_const_i32(b, ctx->block_k);

    /* tid / warp_id / warp_m_idx / warp_n_idx / lane */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->warp_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->warp_m_idx = rocke_b_div(b, ctx->warp_id, ctx->c_warps_n);
    ctx->warp_n_idx = rocke_b_mod(b, ctx->warp_id, ctx->c_warps_n);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);

    /* LDS XOR swizzle. _SWZ = spec.trait.lds_swizzle. */
    ctx->swz = spec->trait.lds_swizzle;
    {
        /* _ilog2 / _swz_elem / _swz_slots / _auto_l / _auto_w geometry. */
        int swz_elem = (ctx->a_per_lane == ctx->b_per_lane) ? ctx->a_per_lane : 0;
        int swz_slots
            = (swz_elem && (ctx->block_k % swz_elem == 0)) ? (ctx->block_k / swz_elem) : 0;
        int auto_l = -1; /* -1 represents Python None */
        int auto_w = -1;
        if(swz_elem && (swz_elem & (swz_elem - 1)) == 0)
        {
            int v = swz_elem, lg = 0;
            while(v > 1)
            {
                v >>= 1;
                ++lg;
            }
            auto_l = lg;
        }
        if(swz_slots && (swz_slots & (swz_slots - 1)) == 0)
        {
            int v = swz_slots, lg = 0;
            while(v > 1)
            {
                v >>= 1;
                ++lg;
            }
            auto_w = lg;
        }
        int def_r, def_w, def_l;
        if(auto_l >= 0 && auto_w >= 0 && auto_w >= 1)
        {
            def_r = 0;
            def_w = auto_w;
            def_l = auto_l;
        }
        else
        {
            def_r = 3;
            def_w = 1;
            def_l = 4;
        }
        /* os.environ.get(CK_SWZ_{R,W,L}, default). The faithful port honours the
         * same env overrides as Python so sweep parity holds. */
        const char* er = getenv("CK_SWZ_R");
        const char* ew = getenv("CK_SWZ_W");
        const char* el = getenv("CK_SWZ_L");
        ctx->swz_r = er ? (int)strtol(er, NULL, 10) : def_r;
        ctx->swz_w = ew ? (int)strtol(ew, NULL, 10) : def_w;
        ctx->swz_l = el ? (int)strtol(el, NULL, 10) : def_l;
    }
    /* _c_swr, _c_swmod, _c_swl = const_i32(swz_r), const_i32(1<<swz_w), const_i32(swz_l) */
    ctx->c_swr = rocke_b_const_i32(b, ctx->swz_r);
    ctx->c_swmod = rocke_b_const_i32(b, (int64_t)1 << ctx->swz_w);
    ctx->c_swl = rocke_b_const_i32(b, ctx->swz_l);

    /* Batch-axis pointer offsets. */
    if(spec->batched)
    {
        ctx->batch_idx = rocke_b_to_sgpr_u32(b, rocke_b_block_id_z(b));
        ctx->batch_off_a = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ctx->batch_idx, ctx->stride_a));
        ctx->batch_off_b = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ctx->batch_idx, ctx->stride_b));
        ctx->batch_off_c = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ctx->batch_idx, ctx->stride_c));
    }
    else
    {
        ctx->batch_idx = NULL;
        ctx->batch_off_a = ctx->c0;
        ctx->batch_off_b = ctx->c0;
        ctx->batch_off_c = ctx->c0;
    }

    /* Per-CTA tile origins (chiplet-swizzle aware). */
    if(spec->trait.chiplet_swizzle)
    {
        rocke_value_t* n_pid_m = rocke_b_div(
            b, rocke_b_add(b, ctx->M, rocke_b_const_i32(b, ctx->block_m - 1)), ctx->c_block_m);
        rocke_value_t* n_pid_n = rocke_b_div(
            b, rocke_b_add(b, ctx->N, rocke_b_const_i32(b, ctx->block_n - 1)), ctx->c_block_n);
        rocke_value_t* wgid_flat
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_block_id_y(b), n_pid_n), rocke_b_block_id_x(b));
        rocke_super_tile_swizzle_result_t swz
            = rocke_chiplet_aware_super_tile_dynamic(b,
                                                     wgid_flat,
                                                     n_pid_m,
                                                     n_pid_n,
                                                     spec->trait.chiplet_wgm,
                                                     spec->trait.chiplet_num_xcds,
                                                     spec->trait.chiplet_chunk_size);
        ctx->block_m_off = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, swz.row, ctx->c_block_m));
        ctx->block_n_off = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, swz.col, ctx->c_block_n));
    }
    else
    {
        ctx->block_m_off
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_y(b), ctx->c_block_m));
        ctx->block_n_off
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_x(b), ctx->c_block_n));
    }

    /* AB LDS double-buffer plan. _prefetch = bool(trait.dtl_prefetch). */
    ctx->prefetch = spec->trait.dtl_prefetch;
    if(ctx->prefetch && !spec->trait.direct_to_lds)
    {
        /* raise ValueError("dtl_prefetch requires direct_to_lds=True") */
        ctx->b->status = ROCKE_ERR_VALUE;
        snprintf(ctx->b->err, ROCKE_ERR_MSG_CAP, "dtl_prefetch requires direct_to_lds=True");
        return;
    }
    /* _, _db, _two_buf = _ab_lds_plan(spec, arch). The db/two_buf decision is
     * pure host arithmetic; ported inline (mirrors _ab_lds_plan). */
    {
        long ab_single = ((long)t->tile_m * t->tile_k + (long)t->tile_n * t->tile_k) * 2;
        bool db_fits_2wg = false;
        if(ctx->target)
        {
            /* (2*ab_single)*2 <= lds_capacity_bytes */
            db_fits_2wg = rocke_archtarget_fits_lds(ctx->target, (2 * ab_single) * 2);
        }
        bool db = (strcmp(spec->trait.pipeline, "compv4") == 0)
                  && (strcmp(spec->trait.epilogue, "cshuffle") != 0) && !spec->trait.direct_to_lds
                  && db_fits_2wg;
        ctx->db = db;
        ctx->two_buf = spec->trait.dtl_prefetch || db;
    }
    ctx->A_LDS_M = ctx->two_buf ? 2 * ctx->block_m : ctx->block_m;
    ctx->B_LDS_N = ctx->two_buf ? 2 * ctx->block_n : ctx->block_n;
    /* _lds_pad = trait.lds_k_pad if not direct_to_lds else 0; _lds_k = block_k + pad */
    ctx->lds_pad = spec->trait.direct_to_lds ? 0 : spec->trait.lds_k_pad;
    ctx->lds_k = ctx->block_k + ctx->lds_pad;
    {
        int a_shape[2] = {ctx->A_LDS_M, ctx->lds_k};
        int b_shape[2] = {ctx->B_LDS_N, ctx->lds_k};
        ctx->A_smem = rocke_b_smem_alloc(b, ctx->storage_dtype, a_shape, 2, "A_smem");
        ctx->B_smem = rocke_b_smem_alloc(b, ctx->storage_dtype, b_shape, 2, "B_smem");
    }

    /* Per-warp MFMA tile counts. */
    ctx->mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    ctx->mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    ctx->k_atoms = rocke_gemm_tile_k_atoms_per_tile_k(t);

    /* Accumulators. acc_init = _emit_zero_acc_op(b, op). accs list = one zero-acc
     * iter-arg per (mi, ni) warp MFMA tile, named "acc_m{mi}_n{ni}". */
    ctx->acc_init = rocke_gemm_emit_zero_acc_op(b, ctx->op);
    ctx->num_accs = ctx->mfmas_m * ctx->mfmas_n;
    {
        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                /* acc_m{mi}_n{ni} name into arena-stable storage. The builder
                 * arena owns iter-arg names; format into a fresh buffer it
                 * copies. Names are short; cap is generous. */
                char namebuf[32];
                snprintf(namebuf, sizeof(namebuf), "acc_m%d_n%d", mi, ni);
                /* Persist the name: rocke_b_fresh duplicates into the arena and
                 * returns an owned copy without emitting any op. We strip the
                 * leading '%' it prepends so the iter-arg name matches Python
                 * (the scf_for_iter builder re-adds '%'). */
                const char* owned = rocke_b_fresh(b, namebuf);
                if(owned && owned[0] == '%')
                    ++owned;
                ctx->acc_names[flat] = owned;
                ctx->acc_inits[flat] = ctx->acc_init;
                ++flat;
            }
        }
    }

    /* Global -> LDS coalesced copy plan. */
    ctx->threads = spec->block_size;
    ctx->load_vec = rocke_gemm_choose_load_vec(spec);
    ctx->a_total = ctx->block_m * ctx->block_k;
    ctx->b_total = ctx->block_n * ctx->block_k;
    ctx->a_vec_total = ctx->a_total / ctx->load_vec;
    ctx->b_vec_total = ctx->b_total / ctx->load_vec;
    ctx->a_vecs_per_thread = ctx->a_vec_total / ctx->threads;
    ctx->b_vecs_per_thread = ctx->b_vec_total / ctx->threads;
    ctx->c_threads = rocke_b_const_i32(b, ctx->threads);
    ctx->c_load_vec = rocke_b_const_i32(b, ctx->load_vec);
    ctx->c_block_k_div_vec = rocke_b_const_i32(b, ctx->block_k / ctx->load_vec);

    /* CK Tile-style data views. a_view = make_global_view(A, (1,1,1), strides=(1,K,1)).
     * (Python passes shape=(1,1,1); the descriptor only uses strides for the
     * offset formula.) */
    {
        int view_shape[3] = {1, 1, 1};
        rocke_stride_t a_strides[3]
            = {rocke_stride_imm(1), rocke_stride_value(ctx->K), rocke_stride_imm(1)};
        rocke_stride_t b_strides[3]
            = {rocke_stride_imm(1), rocke_stride_value(ctx->K), rocke_stride_imm(1)};
        rocke_make_global_view(&ctx->a_view, ctx->A, view_shape, 3, ctx->storage_dtype, a_strides);
        rocke_make_global_view(&ctx->b_view, ctx->Bp, view_shape, 3, ctx->storage_dtype, b_strides);
    }

    /* LDS views: 2D (block_*, block_k); padded => with_strides((_lds_k,1)). */
    {
        int a_shape[2] = {ctx->block_m, ctx->block_k};
        int b_shape[2] = {ctx->block_n, ctx->block_k};
        if(ctx->lds_pad)
        {
            rocke_stride_t a_strides[2] = {rocke_stride_imm(ctx->lds_k), rocke_stride_imm(1)};
            rocke_stride_t b_strides[2] = {rocke_stride_imm(ctx->lds_k), rocke_stride_imm(1)};
            rocke_tensor_descriptor_with_strides(
                &ctx->a_lds_desc, a_shape, a_strides, 2, ctx->storage_dtype);
            rocke_tensor_descriptor_with_strides(
                &ctx->b_lds_desc, b_shape, b_strides, 2, ctx->storage_dtype);
        }
        else
        {
            rocke_tensor_descriptor_packed(&ctx->a_lds_desc, a_shape, 2, ctx->storage_dtype);
            rocke_tensor_descriptor_packed(&ctx->b_lds_desc, b_shape, 2, ctx->storage_dtype);
        }
        ctx->a_lds_view.base = ctx->A_smem;
        ctx->a_lds_view.desc = ctx->a_lds_desc;
        ctx->a_lds_view.addr_space = ROCKE_ADDR_LDS;
        ctx->b_lds_view.base = ctx->B_smem;
        ctx->b_lds_view.desc = ctx->b_lds_desc;
        ctx->b_lds_view.addr_space = ROCKE_ADDR_LDS;
    }

    /* DirectToLDS (DTLA/DTLB) plumbing. */
    ctx->dtl = spec->trait.direct_to_lds;
    if(ctx->dtl)
    {
        ctx->dtl_dwords = 4; /* _DTL_DWORDS         */
        ctx->dtl_halves = ctx->dtl_dwords * 2; /* _DTL_HALVES         */
        ctx->dtl_bytes_per_lane = ctx->dtl_dwords * 4; /* _DTL_BYTES_PER_LANE */
        if((ctx->block_k % ctx->dtl_halves) != 0)
        {
            ctx->b->status = ROCKE_ERR_VALUE;
            snprintf(ctx->b->err,
                     ROCKE_ERR_MSG_CAP,
                     "direct_to_lds requires block_k %% %d == 0 (got %d)",
                     ctx->dtl_halves,
                     ctx->block_k);
            return;
        }
        ctx->dtl_a_chunks = (ctx->block_m * ctx->block_k) / ctx->dtl_halves;
        ctx->dtl_b_chunks = (ctx->block_n * ctx->block_k) / ctx->dtl_halves;
        ctx->dtl_a_passes = (ctx->dtl_a_chunks + spec->block_size - 1) / spec->block_size;
        ctx->dtl_b_passes = (ctx->dtl_b_chunks + spec->block_size - 1) / spec->block_size;
        ctx->dtl_pass_bytes = spec->block_size * ctx->dtl_bytes_per_lane;

        ctx->dtl_big_bytes = rocke_b_const_i32(b, 0x7FFF0000);
        ctx->dtl_a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->dtl_big_bytes);
        ctx->dtl_b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->dtl_big_bytes);
        ctx->dtl_a_lds_base = rocke_b_smem_addr_of(b, ctx->A_smem);
        ctx->dtl_b_lds_base = rocke_b_smem_addr_of(b, ctx->B_smem);
        ctx->dtl_zero_soff = rocke_b_const_i32(b, 0);
        ctx->dtl_chunks_per_row = ctx->block_k / ctx->dtl_halves;
        ctx->dtl_c_chunks_per_row = rocke_b_const_i32(b, ctx->dtl_chunks_per_row);
        ctx->dtl_c_halves_per_chunk = rocke_b_const_i32(b, ctx->dtl_halves);
        ctx->dtl_c_block_size = rocke_b_const_i32(b, spec->block_size);
    }

    /* do_work_cond / for_results are set by the K-loop part-files; default off. */
    ctx->do_work_cond = NULL;
    ctx->num_for_results = 0;
}

/* ===================================================================== *
 *  _swz_col -- the LDS XOR-swizzle column rewrite (Python inner closure).
 *
 *  Python span: gemm_universal.py _swz_col (lines ~895 .. ~898):
 *      def _swz_col(col, row):
 *          if not _SWZ:
 *              return col
 *          return b.xor(col, b.shl(b.mod(b.lshr(row, _c_swr), _c_swmod), _c_swl))
 *
 *  When swizzle is off (`ctx->swz` false) `col` is returned unchanged, exactly
 *  as the Python closure short-circuits. Applied identically to the LDS store
 *  column and the MFMA ds_read column so the physical address agrees on both
 *  sides (correctness preserved). The constants c_swr / c_swmod / c_swl are
 *  pre-materialised in rocke_gemm_build_populate_ctx above.
 * ===================================================================== */
rocke_value_t*
    rocke_gemm_swz_col(rocke_gemm_build_ctx_t* ctx, rocke_value_t* col, rocke_value_t* row)
{
    rocke_ir_builder_t* b = ctx->b;
    if(!ctx->swz)
    {
        return col;
    }
    return rocke_b_xor(
        b,
        col,
        rocke_b_shl(b, rocke_b_mod(b, rocke_b_lshr(b, row, ctx->c_swr), ctx->c_swmod), ctx->c_swl));
}

/* ===================================================================== *
 *  emit_load_phase -- one K-tile's coalesced global->LDS copy.
 *
 *  Python span: gemm_universal.py emit_load_phase (lines ~1112 .. ~1370).
 *  lds_parity is int|Value: parity_v != NULL => runtime Value form; else
 *  parity_imm carries the compile-time 0/1 parity.
 * ===================================================================== */
void rocke_gemm_emit_load_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_dst,
                                rocke_value_t* B_dst,
                                rocke_value_t* k_off,
                                int parity_imm,
                                rocke_value_t* parity_v)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_universal_spec_t* spec = ctx->spec;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* I64 = rocke_i64();
    (void)A_dst; /* Python emit_load_phase's A_dst/B_dst args are unused in body. */
    (void)B_dst;

    /* ----------------------- DirectToLDS path ----------------------- */
    if(spec->trait.direct_to_lds)
    {
        bool parity_is_value = (parity_v != NULL);
        int a_half_bytes = ctx->block_m * ctx->block_k * 2;
        int b_half_bytes = ctx->block_n * ctx->block_k * 2;
        rocke_value_t* a_lds_par_base;
        rocke_value_t* b_lds_par_base;
        if(ctx->prefetch && parity_is_value)
        {
            rocke_value_t* a_par_v = rocke_b_zext(
                b, rocke_b_mul(b, parity_v, rocke_b_const_i32(b, a_half_bytes)), I64);
            rocke_value_t* b_par_v = rocke_b_zext(
                b, rocke_b_mul(b, parity_v, rocke_b_const_i32(b, b_half_bytes)), I64);
            a_lds_par_base = rocke_b_smem_ptr_add(b, ctx->dtl_a_lds_base, a_par_v);
            b_lds_par_base = rocke_b_smem_ptr_add(b, ctx->dtl_b_lds_base, b_par_v);
        }
        else
        {
            a_lds_par_base = ctx->dtl_a_lds_base;
            b_lds_par_base = ctx->dtl_b_lds_base;
        }
        int a_parity_bytes_static
            = (ctx->prefetch && !parity_is_value) ? parity_imm * a_half_bytes : 0;
        int b_parity_bytes_static
            = (ctx->prefetch && !parity_is_value) ? parity_imm * b_half_bytes : 0;
        rocke_value_t* c2 = rocke_b_const_i32(b, 2);

        /* Per-wave LDS offset (multi-wave WG). */
        int wave_bytes = spec->wave_size * ctx->dtl_bytes_per_lane;
        rocke_value_t* a_lds_wave_base;
        rocke_value_t* b_lds_wave_base;
        if(t->warp_m * t->warp_n * t->warp_k > 1)
        {
            rocke_value_t* warp_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
            rocke_value_t* wave_par_off
                = rocke_b_zext(b, rocke_b_mul(b, warp_id, rocke_b_const_i32(b, wave_bytes)), I64);
            a_lds_wave_base = rocke_b_smem_ptr_add(b, a_lds_par_base, wave_par_off);
            b_lds_wave_base = rocke_b_smem_ptr_add(b, b_lds_par_base, wave_par_off);
        }
        else
        {
            a_lds_wave_base = a_lds_par_base;
            b_lds_wave_base = b_lds_par_base;
        }

        for(int p = 0; p < ctx->dtl_a_passes; ++p)
        {
            int pass_off_bytes = p * ctx->dtl_pass_bytes + a_parity_bytes_static;
            rocke_value_t* pass_lds_a
                = (pass_off_bytes > 0)
                      ? rocke_b_smem_ptr_add(
                            b,
                            a_lds_wave_base,
                            rocke_b_zext(b, rocke_b_const_i32(b, pass_off_bytes), I64))
                      : a_lds_wave_base;
            rocke_value_t* chunk_idx
                = rocke_b_add(b, ctx->tid, rocke_b_const_i32(b, p * spec->block_size));
            rocke_value_t* row = rocke_b_div(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col_v = rocke_b_mod(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col = rocke_b_mul(b, col_v, ctx->dtl_c_halves_per_chunk);
            /* Python: b.add(batch_off_a, b.add(b.mul(b.add(block_m_off, row), K),
             *                                   b.add(k_off, _swz_col(col, row))))
             * Python evaluates the inner add's first arg (the mul subtree) before
             * its second (the k_off add). C arg eval order is unspecified -- pin
             * with temporaries so the SSA order matches. */
            rocke_value_t* a_row_term
                = rocke_b_mul(b, rocke_b_add(b, ctx->block_m_off, row), ctx->K);
            rocke_value_t* a_k_term = rocke_b_add(b, k_off, rocke_gemm_swz_col(ctx, col, row));
            rocke_value_t* off_elems
                = rocke_b_add(b, ctx->batch_off_a, rocke_b_add(b, a_row_term, a_k_term));
            rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c2);
            rocke_b_async_buffer_load_lds_addr(b,
                                               ctx->dtl_a_rsrc,
                                               pass_lds_a,
                                               off_bytes,
                                               ctx->dtl_zero_soff,
                                               ctx->dtl_dwords,
                                               spec->trait.dtl_cache_a);
        }
        for(int p = 0; p < ctx->dtl_b_passes; ++p)
        {
            int pass_off_bytes = p * ctx->dtl_pass_bytes + b_parity_bytes_static;
            rocke_value_t* pass_lds_b
                = (pass_off_bytes > 0)
                      ? rocke_b_smem_ptr_add(
                            b,
                            b_lds_wave_base,
                            rocke_b_zext(b, rocke_b_const_i32(b, pass_off_bytes), I64))
                      : b_lds_wave_base;
            rocke_value_t* chunk_idx
                = rocke_b_add(b, ctx->tid, rocke_b_const_i32(b, p * spec->block_size));
            rocke_value_t* row = rocke_b_div(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col_v = rocke_b_mod(b, chunk_idx, ctx->dtl_c_chunks_per_row);
            rocke_value_t* col = rocke_b_mul(b, col_v, ctx->dtl_c_halves_per_chunk);
            /* Same operand-order pinning as the A path above. */
            rocke_value_t* b_row_term
                = rocke_b_mul(b, rocke_b_add(b, ctx->block_n_off, row), ctx->K);
            rocke_value_t* b_k_term = rocke_b_add(b, k_off, rocke_gemm_swz_col(ctx, col, row));
            rocke_value_t* off_elems
                = rocke_b_add(b, ctx->batch_off_b, rocke_b_add(b, b_row_term, b_k_term));
            rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c2);
            rocke_b_async_buffer_load_lds_addr(b,
                                               ctx->dtl_b_rsrc,
                                               pass_lds_b,
                                               off_bytes,
                                               ctx->dtl_zero_soff,
                                               ctx->dtl_dwords,
                                               spec->trait.dtl_cache_b);
        }
        return;
    }

    /* ----------------------- canonical / preshuffle path ----------------------- */
    int load_vec = ctx->load_vec;
    bool db = ctx->db;

    /* a_global_tile / b_global_tile / a_lds_tile / b_lds_tile. */
    rocke_tile_window_t a_global_tile, b_global_tile, a_lds_tile, b_lds_tile;
    {
        int a_glen[3] = {1, ctx->block_m, ctx->block_k};
        int b_glen[3] = {1, ctx->block_n, ctx->block_k};
        rocke_value_t* a_gorigin[3] = {ctx->batch_off_a, ctx->block_m_off, k_off};
        rocke_value_t* b_gorigin[3] = {ctx->batch_off_b, ctx->block_n_off, k_off};
        rocke_make_tile_window(&a_global_tile, &ctx->a_view, a_glen, a_gorigin, 3);
        rocke_make_tile_window(&b_global_tile, &ctx->b_view, b_glen, b_gorigin, 3);

        int a_llen[2] = {ctx->block_m, ctx->block_k};
        int b_llen[2] = {ctx->block_n, ctx->block_k};
        rocke_value_t* zero_a = rocke_b_const_i32(b, 0);
        rocke_value_t* zero_b = rocke_b_const_i32(b, 0);
        rocke_value_t* a_lorigin[2] = {zero_a, zero_b};
        rocke_value_t* zero_c = rocke_b_const_i32(b, 0);
        rocke_value_t* zero_d = rocke_b_const_i32(b, 0);
        rocke_value_t* b_lorigin[2] = {zero_c, zero_d};
        rocke_make_tile_window(&a_lds_tile, &ctx->a_lds_view, a_llen, a_lorigin, 2);
        rocke_make_tile_window(&b_lds_tile, &ctx->b_lds_view, b_llen, b_lorigin, 2);
    }

    /* _ld_a_row / _ld_b_row : double-buffer parity row fold (compv4 non-DTL).
     * Implemented as inline lambdas via a helper macro-free local routine. */
#define LD_ROW(rvar, blk)                                                                        \
    (!db ? (rvar)                                                                                \
         : (parity_v != NULL                                                                     \
                ? rocke_b_add(b, (rvar), rocke_b_mul(b, parity_v, rocke_b_const_i32(b, (blk))))  \
                : (parity_imm ? rocke_b_add(b, (rvar), rocke_b_const_i32(b, parity_imm * (blk))) \
                              : (rvar))))

#define PADK_VALID(elem_col) rocke_b_cmp_lt(b, rocke_b_add(b, k_off, (elem_col)), ctx->K)
#define STORAGE_ZERO() rocke_b_cast_f32_to(b, rocke_b_const_f32(b, 0.0), ctx->storage_dtype)
#define MASK_STORAGE(value, valid) rocke_b_select(b, (valid), (value), STORAGE_ZERO())

    /* A-load loop. */
    for(int e = 0; e < ctx->a_vecs_per_thread; ++e)
    {
        rocke_value_t* vec_idx
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
        /* _vec_rc(vec_idx). */
        rocke_value_t* row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
        rocke_value_t* col_v = rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec);
        rocke_value_t* col = (load_vec > 1) ? rocke_b_mul(b, col_v, ctx->c_load_vec) : col_v;
        if(spec->trait.pad_k)
        {
            rocke_value_t* comps[16];
            rocke_value_t* a_val;
            for(int i = 0; i < load_vec; ++i)
            {
                rocke_value_t* elem_col = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                rocke_value_t* valid = PADK_VALID(elem_col);
                rocke_value_t* safe_col
                    = rocke_b_select(b, valid, elem_col, rocke_b_const_i32(b, 0));
                rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, safe_col};
                rocke_value_t* raw = rocke_tile_window_load_scalar(b, &a_global_tile, gidx, 3);
                comps[i] = MASK_STORAGE(raw, valid);
            }
            a_val = (load_vec == 1) ? comps[0]
                                    : rocke_b_vec_pack(b, comps, load_vec, ctx->storage_dtype);
            rocke_value_t* lidx[2] = {LD_ROW(row, ctx->block_m), rocke_gemm_swz_col(ctx, col, row)};
            if(load_vec == 1)
                rocke_tile_window_store_scalar(b, &a_lds_tile, lidx, 2, a_val, 0);
            else
                rocke_tile_window_store_vec(b, &a_lds_tile, lidx, 2, a_val, load_vec);
        }
        else if(load_vec == 1)
        {
            rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
            rocke_value_t* a_val = rocke_tile_window_load_scalar(b, &a_global_tile, gidx, 3);
            rocke_value_t* lidx[2] = {LD_ROW(row, ctx->block_m), rocke_gemm_swz_col(ctx, col, row)};
            rocke_tile_window_store_scalar(b, &a_lds_tile, lidx, 2, a_val, 0);
        }
        else
        {
            rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
            rocke_value_t* a_val = rocke_tile_window_load_vec(b, &a_global_tile, gidx, 3, load_vec);
            rocke_value_t* lidx[2] = {LD_ROW(row, ctx->block_m), rocke_gemm_swz_col(ctx, col, row)};
            rocke_tile_window_store_vec(b, &a_lds_tile, lidx, 2, a_val, load_vec);
        }
    }

    /* B-load branch. */
    if(spec->trait.preshuffle_b)
    {
        rocke_value_t* n_tile_idx = rocke_b_div(b, ctx->block_n_off, ctx->c_block_n);
        rocke_value_t* k_tile_idx = rocke_b_div(b, k_off, ctx->c_block_k);
        rocke_value_t* n_tile_count = rocke_b_div(b, ctx->N, ctx->c_block_n);
        /* Python evaluates rocke_b_mul args left-to-right: the inner add/mul are
         * built BEFORE the const. C evaluates call args right-to-left, so bind
         * each sub-expression to a temp in Python order to match SSA ids. */
        rocke_value_t* tile_off_inner
            = rocke_b_add(b, rocke_b_mul(b, k_tile_idx, n_tile_count), n_tile_idx);
        rocke_value_t* tile_off_const = rocke_b_const_i32(b, ctx->block_n * ctx->block_k);
        rocke_value_t* tile_offset_elements = rocke_b_mul(b, tile_off_inner, tile_off_const);
        rocke_value_t* base_off = rocke_b_add(b, ctx->batch_off_b, tile_offset_elements);
        for(int e = 0; e < ctx->b_vecs_per_thread; ++e)
        {
            rocke_value_t* vec_idx
                = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
            rocke_value_t* glob_off
                = rocke_b_add(b, base_off, rocke_b_mul(b, vec_idx, ctx->c_load_vec));
            rocke_value_t* row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col_v = rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col = (load_vec > 1) ? rocke_b_mul(b, col_v, ctx->c_load_vec) : col_v;
            if(spec->trait.pad_k)
            {
                rocke_value_t* comps[16];
                rocke_value_t* b_val;
                for(int i = 0; i < load_vec; ++i)
                {
                    rocke_value_t* elem_col
                        = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                    rocke_value_t* valid = PADK_VALID(elem_col);
                    rocke_value_t* raw_off = rocke_b_add(
                        b,
                        base_off,
                        rocke_b_add(
                            b, rocke_b_mul(b, vec_idx, ctx->c_load_vec), rocke_b_const_i32(b, i)));
                    rocke_value_t* safe_off = rocke_b_select(b, valid, raw_off, base_off);
                    rocke_value_t* raw
                        = rocke_b_global_load(b, ctx->Bp, safe_off, ctx->storage_dtype, 0);
                    comps[i] = MASK_STORAGE(raw, valid);
                }
                b_val = (load_vec == 1) ? comps[0]
                                        : rocke_b_vec_pack(b, comps, load_vec, ctx->storage_dtype);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                if(load_vec == 1)
                    rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
                else
                    rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
            else if(load_vec == 1)
            {
                rocke_value_t* b_val
                    = rocke_b_global_load(b, ctx->Bp, glob_off, ctx->storage_dtype, 0);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
            }
            else
            {
                rocke_value_t* b_val
                    = rocke_b_global_load_vN(b, ctx->Bp, glob_off, ctx->storage_dtype, load_vec, 0);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
        }
    }
    else
    {
        for(int e = 0; e < ctx->b_vecs_per_thread; ++e)
        {
            rocke_value_t* vec_idx
                = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), ctx->c_threads), ctx->tid);
            rocke_value_t* row = rocke_b_div(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col_v = rocke_b_mod(b, vec_idx, ctx->c_block_k_div_vec);
            rocke_value_t* col = (load_vec > 1) ? rocke_b_mul(b, col_v, ctx->c_load_vec) : col_v;
            if(spec->trait.pad_k)
            {
                rocke_value_t* comps[16];
                rocke_value_t* b_val;
                for(int i = 0; i < load_vec; ++i)
                {
                    rocke_value_t* elem_col
                        = i ? rocke_b_add(b, col, rocke_b_const_i32(b, i)) : col;
                    rocke_value_t* valid = PADK_VALID(elem_col);
                    rocke_value_t* safe_col
                        = rocke_b_select(b, valid, elem_col, rocke_b_const_i32(b, 0));
                    rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, safe_col};
                    rocke_value_t* raw = rocke_tile_window_load_scalar(b, &b_global_tile, gidx, 3);
                    comps[i] = MASK_STORAGE(raw, valid);
                }
                b_val = (load_vec == 1) ? comps[0]
                                        : rocke_b_vec_pack(b, comps, load_vec, ctx->storage_dtype);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                if(load_vec == 1)
                    rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
                else
                    rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
            else if(load_vec == 1)
            {
                rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
                rocke_value_t* b_val = rocke_tile_window_load_scalar(b, &b_global_tile, gidx, 3);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_scalar(b, &b_lds_tile, lidx, 2, b_val, 0);
            }
            else
            {
                rocke_value_t* gidx[3] = {rocke_b_const_i32(b, 0), row, col};
                rocke_value_t* b_val
                    = rocke_tile_window_load_vec(b, &b_global_tile, gidx, 3, load_vec);
                rocke_value_t* lidx[2]
                    = {LD_ROW(row, ctx->block_n), rocke_gemm_swz_col(ctx, col, row)};
                rocke_tile_window_store_vec(b, &b_lds_tile, lidx, 2, b_val, load_vec);
            }
        }
    }
#undef MASK_STORAGE
#undef STORAGE_ZERO
#undef PADK_VALID
#undef LD_ROW
}

/* ===================================================================== *
 *  _emit_frag_smem_load -- one frag_len-wide operand fragment from a
 *  row-major LDS tile (8-wide chunked + vec_concat for wide WMMA frags).
 *
 *  Python span: gemm_universal.py _emit_frag_smem_load (lines ~1372 .. ~1407).
 * ===================================================================== */
rocke_value_t* rocke_gemm_emit_frag_smem_load(rocke_gemm_build_ctx_t* ctx,
                                              rocke_value_t* src,
                                              rocke_value_t* mn_in_atom,
                                              rocke_value_t* k_in_atom,
                                              rocke_value_t* atom_mn_base,
                                              rocke_value_t* k_tile_base,
                                              int frag_len)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* lds_row = rocke_b_add(b, atom_mn_base, mn_in_atom);
    rocke_value_t* lds_col = rocke_b_add(b, k_tile_base, k_in_atom);
    /* max_vec = 8 if storage_dtype in (F16, BF16) else frag_len */
    bool is_half = rocke_type_eq(ctx->storage_dtype, rocke_f16())
                   || rocke_type_eq(ctx->storage_dtype, rocke_bf16());
    int max_vec = is_half ? 8 : frag_len;
    if(frag_len <= max_vec)
    {
        return rocke_gemm_emit_smem_load(b, src, lds_row, lds_col, frag_len, ctx->storage_dtype);
    }
    rocke_value_t* frag = NULL;
    for(int off = 0; off < frag_len; off += max_vec)
    {
        rocke_value_t* chunk
            = rocke_gemm_emit_smem_load(b,
                                        src,
                                        lds_row,
                                        rocke_b_add(b, lds_col, rocke_b_const_i32(b, off)),
                                        max_vec,
                                        ctx->storage_dtype);
        frag = (frag == NULL) ? chunk : rocke_b_vec_concat(b, frag, chunk);
    }
    return frag;
}

/* ===================================================================== *
 *  _emit_wmma_phase -- one K-tile of WMMA atoms, fully contract-driven.
 *
 *  Python span: gemm_universal.py _emit_wmma_phase (lines ~1409 .. ~1464).
 *  Reads iter_vars (length ctx->num_accs), writes the new accs into out_accs.
 * ===================================================================== */
void rocke_gemm_emit_wmma_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;
    (void)num_iter_vars;

    const rocke_arch_layout_map_t* a_map = rocke_mmaop_a_layout(ctx->op, b);
    const rocke_arch_layout_map_t* b_map = rocke_mmaop_b_layout(ctx->op, b);

    /* a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
     * b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0) */
    rocke_value_t* a_row_in_atom = NULL;
    rocke_value_t* a_k_in_atom = NULL;
    rocke_value_t* b_k_in_atom = NULL;
    rocke_value_t* b_col_in_atom = NULL;
    rocke_arch_layout_map_coord(a_map, b, ctx->lane, 0, &a_row_in_atom, &a_k_in_atom);
    rocke_arch_layout_map_coord(b_map, b, ctx->lane, 0, &b_k_in_atom, &b_col_in_atom);

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, ctx->mfmas_m * t->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, ctx->mfmas_n * t->warp_tile_n));

    /* new_accs = list(iter_vars) */
    for(int i = 0; i < ctx->num_accs; ++i)
        out_accs[i] = iter_vars[i];

    /* Bounded scratch for per-K-step fragment vectors. */
    rocke_value_t* a_rows[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* b_cols[ROCKE_GEMM_MAX_ACCS];

    for(int kk = 0; kk < ctx->k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * t->warp_tile_k);
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * t->warp_tile_m));
            a_rows[mi] = rocke_gemm_emit_frag_smem_load(
                ctx, A_src, a_row_in_atom, a_k_in_atom, atom_row, k_tile_base, ctx->a_per_lane);
        }
        for(int ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * t->warp_tile_n));
            b_cols[ni] = rocke_gemm_emit_frag_smem_load(
                ctx, B_src, b_col_in_atom, b_k_in_atom, atom_row, k_tile_base, ctx->b_per_lane);
        }
        int flat = 0;
        for(int mi = 0; mi < ctx->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < ctx->mfmas_n; ++ni)
            {
                out_accs[flat]
                    = rocke_gemm_emit_mma(b, ctx->op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        }
    }
}
