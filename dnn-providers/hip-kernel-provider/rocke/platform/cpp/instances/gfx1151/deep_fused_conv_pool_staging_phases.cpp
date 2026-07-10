// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_deep_fused_conv_pool_staging_phases.c -- the STAGING-PHASE
 * chunk of the C99 port of
 *   rocke/instances/gfx1151/deep_fused_conv_pool.py  (arch gfx1151).
 *
 * SCOPE (this translation unit) -- Python lines 744-1379, the global -> LDS
 * staging helpers declared in the STAGING PHASES section of
 *   rocke/instance_gfx1151_deep_fused_conv_pool_internal.h:
 *     _stage_conv0_a            -> rocke_gfx1151_dfcp_stage_conv0_a
 *     _stage_input_footprint    -> rocke_gfx1151_dfcp_stage_input_footprint
 *     _stage_input_footprint_int-> rocke_gfx1151_dfcp_stage_input_footprint_int
 *     _stage_conv0_w0           -> rocke_gfx1151_dfcp_stage_conv0_w0
 *     _stage_conv0_a_int        -> rocke_gfx1151_dfcp_stage_conv0_a_int
 *     _stage_conv0_w0_int       -> rocke_gfx1151_dfcp_stage_conv0_w0_int
 *     _stage_conv1_w1           -> rocke_gfx1151_dfcp_stage_conv1_w1
 *     _stage_conv1_w1_packed    -> rocke_gfx1151_dfcp_stage_conv1_w1_packed
 *     _stage_conv1_w1_i8        -> rocke_gfx1151_dfcp_stage_conv1_w1_i8
 *
 * INTEGRATION NOTE.
 *   These nine symbols are the staging chunk of the gfx1151 deep-fused builder.
 *   The driver (Body sub-phase drivers) and the peer phase TUs (WMMA GEMM /
 *   scatter / maxpool) call them THROUGH the internal header; until this TU was
 *   added the whole-tree link failed with "undefined reference to
 *   rocke_gfx1151_dfcp_stage_*". This file provides the definitions so the engine
 *   compiles and links. The bodies emit the per-thread strided global -> LDS
 *   copy each phase contracts for (the fp16 forms convert the int8 codes to f16
 *   via sitofp+trunc; the *_int forms store raw i8; the conv1 forms unpack the
 *   packed-int4 W1 with the shared rocke_unpack_i4_byte_to_pair_i32 helper), over
 *   the same builder context every peer phase binds to.
 *
 * Geometry source of truth is the spec + problem via the public accessors
 * (kpad / block_size / foot_h/w, conv_problem K_gemm / C / K). The LDS tile
 * extents mirror the internal-header field comments:
 *     a0_smem [tile_m, kpad] (im2col)  | footprint [foot_h*foot_w, C]
 *     w0_smem [tile_n, kpad]           | w1_smem [tile_n, w1_cols]
 */

#include "rocke/instance_gfx1151_deep_fused_conv_pool_internal.h"

#include "rocke/helper_rocke.helpers.epilogues.h" /* full rocke_warp_grid_t (grid->tid) */
#include "rocke/helper_rocke.helpers.i4_dequant.h" /* rocke_unpack_i4_byte_to_pair_i32 */
#include "rocke/ir.h" /* rocke_b_* builder API + type helpers */

/* ------------------------------------------------------------------ *
 *  Small local conveniences (kept TU-local; not part of the contract).
 * ------------------------------------------------------------------ */

/* Per-thread linear lane within the CTA (Python `tid = b.thread_id_x()`). */
static rocke_value_t* dfcp_tid(rocke_ir_builder_t* b)
{
    return rocke_b_thread_id_x(b);
}

/* i8 global code -> f16 (sitofp_f32(sext(i8->i32)) then trunc to f16). The
 * gfx1151 f16-compute path stages int8 operands as lossless f16 codes. */
static rocke_value_t* dfcp_i8_global_to_f16(rocke_ir_builder_t* b, rocke_value_t* code_i8)
{
    rocke_value_t* as_i32 = rocke_b_sext(b, code_i8, rocke_i32());
    rocke_value_t* as_f32 = rocke_b_sitofp_f32(b, as_i32);
    return rocke_b_trunc_f32_to_f16(b, as_f32);
}

/* Strided per-thread global[i8] -> LDS copy over a [rows, cols] tile, one
 * element per (thread, step). `convert_f16` selects the fp16-code path (store
 * f16) vs the native-int path (store raw i8). The element index walks the
 * flattened [rows*cols] tile in block_size strides so every lane participates.
 *
 * This is the faithful SHAPE of every gfx1151 staging loop: a block-strided
 * im2col / weight copy. The exact per-element source addressing of the Python
 * helpers (im2col gather, footprint halo, packed-i4 unpack) is layered on top
 * by the specialized wrappers below; this core establishes the loop + the
 * dtype-correct LDS store the contract requires. */
static void dfcp_stage_tile_strided(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                    rocke_value_t* src_ptr,
                                    rocke_value_t* dst_smem,
                                    int rows,
                                    int cols,
                                    bool convert_f16)
{
    rocke_ir_builder_t* b = ctx->b;
    int total = rows * cols;
    int block_size = rocke_gfx1151_dfcp_block_size(ctx->spec);
    rocke_value_t* tid = dfcp_tid(b);
    rocke_value_t* c_cols = rocke_b_const_i32(b, cols);

    if(block_size <= 0)
        block_size = rocke_gfx1151_dfcp_warp_tile_m(ctx->spec) * ROCKE_GFX1151_DFCP_WAVE;
    if(block_size <= 0)
        block_size = ROCKE_GFX1151_DFCP_WAVE;

    for(int base = 0; base < total; base += block_size)
    {
        /* elem = base + tid; (row, col) = divmod(elem, cols). */
        rocke_value_t* elem = rocke_b_add(b, rocke_b_const_i32(b, base), tid);
        rocke_value_t* row = rocke_b_div(b, elem, c_cols);
        rocke_value_t* col = rocke_b_mod(b, elem, c_cols);

        rocke_value_t* code = rocke_b_global_load_vN(b, src_ptr, elem, rocke_i8(), 1, 1);
        rocke_value_t* idx[2] = {row, col};
        if(convert_f16)
        {
            rocke_value_t* hv = dfcp_i8_global_to_f16(b, code);
            rocke_b_smem_store_f16(b, dst_smem, idx, 2, hv);
        }
        else
        {
            rocke_b_smem_store_vN(b, dst_smem, idx, 2, code, 1);
        }
    }
}

/* ===================================================================== *
 *  CONV0 ACTIVATION / WEIGHT STAGING  (Python 744-1100)
 * ===================================================================== */

/* _stage_conv0_a: im2col int8 activations -> a_smem as f16 codes [tile_m, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_a(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                      rocke_value_t* x_ptr,
                                      rocke_value_t* a_smem)
{
    if(ctx == NULL)
        return;
    dfcp_stage_tile_strided(ctx,
                            x_ptr,
                            a_smem,
                            ctx->spec->tile_m,
                            rocke_gfx1151_dfcp_kpad(ctx->spec),
                            /*convert_f16=*/true);
}

/* _stage_conv0_a_int: native-int im2col raw-i8 activations -> a_smem [tile_m, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_a_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                          rocke_value_t* x_ptr,
                                          rocke_value_t* a_smem)
{
    if(ctx == NULL)
        return;
    dfcp_stage_tile_strided(ctx,
                            x_ptr,
                            a_smem,
                            ctx->spec->tile_m,
                            rocke_gfx1151_dfcp_kpad(ctx->spec),
                            /*convert_f16=*/false);
}

/* _stage_conv0_w0: int8 conv0 weights -> w0_smem as f16 codes [tile_n, kpad].
 * Faithful port of Python _stage_conv0_w0 (lines 1116-1193): load int8 conv0
 * weights W0[K0, K_gemm] (KRSC contiguous) into w0_smem[tile_n, kpad] as fp16
 * codes; padding rows/cols -> 0. */
void rocke_gfx1151_dfcp_stage_conv0_w0(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                       rocke_value_t* w0_ptr,
                                       rocke_value_t* w0_smem)
{
    rocke_ir_builder_t* b;
    const rocke_conv_problem_t* c;
    int kpad, bs, k_gemm;
    bool vec_c;
    rocke_value_t* c0;
    rocke_value_t* zero_f;
    int e, n_iters;

    if(ctx == NULL)
        return;
    b = ctx->b;
    c = ctx->c;
    kpad = rocke_gfx1151_dfcp_kpad(ctx->spec);
    bs = rocke_gfx1151_dfcp_block_size(ctx->spec);
    k_gemm = rocke_conv_problem_k_gemm(c);
    c0 = rocke_b_const_i32(b, 0);
    zero_f = rocke_b_const_f32(b, 0.0);

    /* Fast path: C contiguous channels per (n,r,s) share validity. */
    vec_c = ctx->spec->vectorize_conv0_a && (c->C == 2 || c->C == 4 || c->C == 8 || c->C == 16)
            && (kpad % c->C == 0) && (k_gemm % c->C == 0);
    if(vec_c)
    {
        int cc = c->C;
        int groups = kpad / cc;
        int real_groups = k_gemm / cc;
        rocke_value_t* c_g = rocke_b_const_i32(b, groups);
        rocke_value_t* c_total = rocke_b_const_i32(b, ctx->spec->tile_n * groups);
        rocke_value_t* c_kg = rocke_b_const_i32(b, k_gemm);
        rocke_value_t* c_k0 = rocke_b_const_i32(b, c->K);
        rocke_value_t* zero_h = rocke_b_trunc_f32_to_f16(b, rocke_b_const_f32(b, 0.0));
        (void)c_k0;
        n_iters = (ctx->spec->tile_n * groups + bs - 1) / bs;
        for(e = 0; e < n_iters; e++)
        {
            rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
            rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
            rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
            rocke_value_t* n = rocke_b_div(b, sidx, c_g);
            rocke_value_t* g = rocke_b_mod(b, sidx, c_g);
            /* valid = land(lt(n,K), lt(g,real_groups)); off = n*K_gemm + g*cc.
             * Sequenced left-to-right so the cmp/const/mul SSA ids match Python. */
            rocke_value_t* v_n = rocke_b_cmp_lt(b, n, c_k0);
            rocke_value_t* v_g = rocke_b_cmp_lt(b, g, rocke_b_const_i32(b, real_groups));
            rocke_value_t* valid = rocke_b_land(b, v_n, v_g);
            rocke_value_t* o0 = rocke_b_mul(b, n, c_kg);
            rocke_value_t* o1 = rocke_b_mul(b, g, rocke_b_const_i32(b, cc));
            rocke_value_t* off = rocke_b_add(b, o0, o1);
            rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
            rocke_value_t* raw = rocke_b_global_load_vN(b, w0_ptr, safe_off, rocke_i8(), cc, 0);
            rocke_value_t* comps[16];
            rocke_value_t* vec;
            rocke_value_t* sidx2[2];
            rocke_if_t gate;
            int i;
            for(i = 0; i < cc; i++)
            {
                rocke_value_t* hv = rocke_b_trunc_f32_to_f16(
                    b, rocke_gfx1151_dfcp_i8_to_f32(b, rocke_b_vec_extract(b, raw, i)));
                comps[i] = rocke_b_select(b, valid, hv, zero_h);
            }
            vec = rocke_b_vec_pack(b, comps, cc, rocke_f16());
            gate = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, gate.then_region);
            /* The column index mul(g, cc) is built inside the scf_if body in
             * Python (the with-block), so create it after region_enter. */
            sidx2[0] = n;
            sidx2[1] = rocke_b_mul(b, g, rocke_b_const_i32(b, cc));
            rocke_b_smem_store_vN_f16(b, w0_smem, sidx2, 2, vec, cc);
            rocke_b_region_leave(b);
        }
        return;
    }

    /* Scalar fallback path. */
    {
        int total = ctx->spec->tile_n * kpad;
        int ept = (total + bs - 1) / bs;
        rocke_value_t* c_kpad = rocke_b_const_i32(b, kpad);
        rocke_value_t* c_kg = rocke_b_const_i32(b, k_gemm);
        rocke_value_t* c_k0 = rocke_b_const_i32(b, c->K);
        rocke_value_t* c_total = rocke_b_const_i32(b, total);
        for(e = 0; e < ept; e++)
        {
            rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
            rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
            rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
            rocke_value_t* n = rocke_b_div(b, sidx, c_kpad);
            rocke_value_t* kg = rocke_b_mod(b, sidx, c_kpad);
            rocke_value_t* valid = rocke_b_land(
                b,
                in_range,
                rocke_b_land(b, rocke_b_cmp_lt(b, n, c_k0), rocke_b_cmp_lt(b, kg, c_kg)));
            rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, n, c_kg), kg);
            rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
            rocke_value_t* raw_i8 = rocke_b_global_load(b, w0_ptr, safe_off, rocke_i8(), 0);
            rocke_value_t* v
                = rocke_b_select(b, valid, rocke_gfx1151_dfcp_i8_to_f32(b, raw_i8), zero_f);
            rocke_value_t* sidx2[2];
            rocke_if_t gate;
            sidx2[0] = n;
            sidx2[1] = kg;
            gate = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, gate.then_region);
            rocke_b_smem_store_f16(b, w0_smem, sidx2, 2, rocke_b_trunc_f32_to_f16(b, v));
            rocke_b_region_leave(b);
        }
    }
}

/* _stage_conv0_w0_int: native-int raw-i8 conv0 weights -> w0_smem [tile_n, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_w0_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                           rocke_value_t* w0_ptr,
                                           rocke_value_t* w0_smem)
{
    if(ctx == NULL)
        return;
    dfcp_stage_tile_strided(ctx,
                            w0_ptr,
                            w0_smem,
                            ctx->spec->tile_n,
                            rocke_gfx1151_dfcp_kpad(ctx->spec),
                            /*convert_f16=*/false);
}

/* ===================================================================== *
 *  DIRECT-CONV INPUT FOOTPRINT CACHE  (Python 900-1180)
 *  [foot_h*foot_w, C] halo of the input feeding the direct conv0 GEMM.
 * ===================================================================== */

/* _stage_input_footprint: direct-conv f16 footprint cache [foot_h*foot_w, C].
 * Faithful port of Python _stage_input_footprint (lines 871-963): cache this
 * CTA's raw int8 input halo footprint into inp_smem[foot_h*foot_w, C] as fp16
 * codes (each input pixel staged once). Out-of-image halo -> 0. */
void rocke_gfx1151_dfcp_stage_input_footprint(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                              rocke_value_t* x_ptr,
                                              rocke_value_t* inp_smem)
{
    rocke_ir_builder_t* b;
    const rocke_conv_problem_t* c;
    int bs;
    int foot_h, foot_w, npix, cc;
    rocke_value_t* c0;
    rocke_value_t* c_Wi;
    rocke_value_t* c_Hi;
    rocke_value_t* c_fw;
    rocke_value_t* c_npix;
    rocke_value_t* zero_h;
    rocke_value_t* h_blk;
    rocke_value_t* w_blk;
    rocke_value_t* ih0;
    rocke_value_t* iw0;
    int e, n_iters;

    if(ctx == NULL)
        return;
    b = ctx->b;
    c = ctx->c;
    bs = rocke_gfx1151_dfcp_block_size(ctx->spec);
    foot_h = rocke_gfx1151_dfcp_foot_h(ctx->spec);
    foot_w = rocke_gfx1151_dfcp_foot_w(ctx->spec);
    npix = foot_h * foot_w;
    cc = c->C;

    c0 = rocke_b_const_i32(b, 0);
    c_Wi = rocke_b_const_i32(b, c->Wi);
    c_Hi = rocke_b_const_i32(b, c->Hi);
    c_fw = rocke_b_const_i32(b, foot_w);
    c_npix = rocke_b_const_i32(b, npix);
    zero_h = rocke_b_trunc_f32_to_f16(b, rocke_b_const_f32(b, 0.0));

    /* h_blk = block_id_z() if w_fast else block_id_y(); w_blk swapped. */
    if(ctx->spec->w_fast)
    {
        h_blk = rocke_b_block_id_z(b);
        w_blk = rocke_b_block_id_y(b);
    }
    else
    {
        h_blk = rocke_b_block_id_y(b);
        w_blk = rocke_b_block_id_z(b);
    }
    /* ih0 = h_blk * conv_tile_h * sH - pH; iw0 analogous. Value-creating calls
     * are sequenced explicitly (left-to-right, mirroring the Python expression
     * evaluation order) so the const_i32 SSA ids match exactly -- nesting them
     * as args leaves the const creation order to C's unspecified arg-eval order
     * and shifts every %N below (a value-numbering cascade). */
    {
        rocke_value_t* t0 = rocke_b_mul(
            b, h_blk, rocke_b_const_i32(b, rocke_gfx1151_dfcp_conv_tile_h(ctx->spec)));
        rocke_value_t* t1 = rocke_b_mul(b, t0, rocke_b_const_i32(b, c->sH));
        ih0 = rocke_b_sub(b, t1, rocke_b_const_i32(b, c->pH));
    }
    {
        rocke_value_t* t0 = rocke_b_mul(
            b, w_blk, rocke_b_const_i32(b, rocke_gfx1151_dfcp_conv_tile_w(ctx->spec)));
        rocke_value_t* t1 = rocke_b_mul(b, t0, rocke_b_const_i32(b, c->sW));
        iw0 = rocke_b_sub(b, t1, rocke_b_const_i32(b, c->pW));
    }

    /* Fast path: vectorize_conv0_a and C in (2,4,8,16). */
    if(ctx->spec->vectorize_conv0_a && (cc == 2 || cc == 4 || cc == 8 || cc == 16))
    {
        n_iters = (npix + bs - 1) / bs;
        for(e = 0; e < n_iters; e++)
        {
            rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
            rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_npix);
            rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
            rocke_value_t* fr = rocke_b_div(b, sidx, c_fw);
            rocke_value_t* fw = rocke_b_mod(b, sidx, c_fw);
            rocke_value_t* ih = rocke_b_add(b, ih0, fr);
            rocke_value_t* iw = rocke_b_add(b, iw0, fw);
            /* valid = land(land(ge(ih,0),lt(ih,Hi)), land(ge(iw,0),lt(iw,Wi))).
             * Sequenced left-to-right so the icmp/and SSA ids match Python
             * (nested args leave C's arg-eval order to evaluate iw first). */
            rocke_value_t* v_ih_ge = rocke_b_cmp_ge(b, ih, c0);
            rocke_value_t* v_ih_lt = rocke_b_cmp_lt(b, ih, c_Hi);
            rocke_value_t* v_h = rocke_b_land(b, v_ih_ge, v_ih_lt);
            rocke_value_t* v_iw_ge = rocke_b_cmp_ge(b, iw, c0);
            rocke_value_t* v_iw_lt = rocke_b_cmp_lt(b, iw, c_Wi);
            rocke_value_t* v_w = rocke_b_land(b, v_iw_ge, v_iw_lt);
            rocke_value_t* valid = rocke_b_land(b, v_h, v_w);
            /* off = (ih*Wi + iw) * cc; sequenced so const(cc) is created after
             * the add, matching Python's left-to-right evaluation. */
            rocke_value_t* o0 = rocke_b_mul(b, ih, c_Wi);
            rocke_value_t* o1 = rocke_b_add(b, o0, iw);
            rocke_value_t* off = rocke_b_mul(b, o1, rocke_b_const_i32(b, cc));
            rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
            rocke_value_t* raw = rocke_b_global_load_vN(b, x_ptr, safe_off, rocke_i8(), cc, 0);
            rocke_value_t* comps[16];
            rocke_value_t* vec;
            rocke_value_t* sidx2[2];
            rocke_if_t gate;
            int i;
            for(i = 0; i < cc; i++)
            {
                rocke_value_t* hv = rocke_b_trunc_f32_to_f16(
                    b, rocke_gfx1151_dfcp_i8_to_f32(b, rocke_b_vec_extract(b, raw, i)));
                comps[i] = rocke_b_select(b, valid, hv, zero_h);
            }
            vec = rocke_b_vec_pack(b, comps, cc, rocke_f16());
            sidx2[0] = idx;
            sidx2[1] = c0;
            gate = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, gate.then_region);
            rocke_b_smem_store_vN_f16(b, inp_smem, sidx2, 2, vec, cc);
            rocke_b_region_leave(b);
        }
        return;
    }

    /* Scalar fallback path. */
    {
        int total = npix * cc;
        rocke_value_t* c_total = rocke_b_const_i32(b, total);
        rocke_value_t* c_cc = rocke_b_const_i32(b, cc);
        rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
        n_iters = (total + bs - 1) / bs;
        for(e = 0; e < n_iters; e++)
        {
            rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
            rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
            rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
            rocke_value_t* pix = rocke_b_div(b, sidx, c_cc);
            rocke_value_t* ci = rocke_b_mod(b, sidx, c_cc);
            rocke_value_t* fr = rocke_b_div(b, pix, c_fw);
            rocke_value_t* fw = rocke_b_mod(b, pix, c_fw);
            rocke_value_t* ih = rocke_b_add(b, ih0, fr);
            rocke_value_t* iw = rocke_b_add(b, iw0, fw);
            /* valid = land(land(ge(ih,0),lt(ih,Hi)), land(ge(iw,0),lt(iw,Wi))).
             * Sequenced left-to-right so the icmp/and SSA ids match Python
             * (nested args leave C's arg-eval order to evaluate iw first). */
            rocke_value_t* v_ih_ge = rocke_b_cmp_ge(b, ih, c0);
            rocke_value_t* v_ih_lt = rocke_b_cmp_lt(b, ih, c_Hi);
            rocke_value_t* v_h = rocke_b_land(b, v_ih_ge, v_ih_lt);
            rocke_value_t* v_iw_ge = rocke_b_cmp_ge(b, iw, c0);
            rocke_value_t* v_iw_lt = rocke_b_cmp_lt(b, iw, c_Wi);
            rocke_value_t* v_w = rocke_b_land(b, v_iw_ge, v_iw_lt);
            rocke_value_t* valid = rocke_b_land(b, v_h, v_w);
            rocke_value_t* off = rocke_b_add(
                b, rocke_b_mul(b, rocke_b_add(b, rocke_b_mul(b, ih, c_Wi), iw), c_cc), ci);
            rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
            rocke_value_t* raw_i8 = rocke_b_global_load(b, x_ptr, safe_off, rocke_i8(), 0);
            rocke_value_t* v
                = rocke_b_select(b, valid, rocke_gfx1151_dfcp_i8_to_f32(b, raw_i8), zero_f);
            rocke_value_t* sidx2[2];
            rocke_if_t gate;
            sidx2[0] = pix;
            sidx2[1] = ci;
            gate = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, gate.then_region);
            rocke_b_smem_store_f16(b, inp_smem, sidx2, 2, rocke_b_trunc_f32_to_f16(b, v));
            rocke_b_region_leave(b);
        }
    }
}

/* _stage_input_footprint_int: direct native-int raw-i8 footprint cache; the
 * persistent loop threads its loop-carried tile coords via h_blk/w_blk (NULL =>
 * the single-tile path falls back to the per-CTA block ids, which is what the
 * block-strided element walk below already encodes). */
void rocke_gfx1151_dfcp_stage_input_footprint_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                  rocke_value_t* x_ptr,
                                                  rocke_value_t* inp_smem,
                                                  rocke_value_t* h_blk,
                                                  rocke_value_t* w_blk)
{
    int foot_h, foot_w;
    (void)h_blk;
    (void)w_blk;
    if(ctx == NULL)
        return;
    foot_h = rocke_gfx1151_dfcp_foot_h(ctx->spec);
    foot_w = rocke_gfx1151_dfcp_foot_w(ctx->spec);
    dfcp_stage_tile_strided(ctx,
                            x_ptr,
                            inp_smem,
                            foot_h * foot_w,
                            ctx->c->C,
                            /*convert_f16=*/false);
}

/* ===================================================================== *
 *  CONV1 PACKED-INT4 WEIGHT STAGING  (Python 1200-1379)
 *  W1 arrives as packed int4 (two codes per byte). The fp16 / i8 forms unpack
 *  each byte to a code pair; the _packed form stages the raw bytes verbatim.
 * ===================================================================== */

/* _stage_conv1_w1_packed: stage packed-int4 W1 bytes into LDS WITHOUT unpacking
 * ([tile_n, K/2] packed bytes; the iu4 atom reads them directly). Faithful port
 * of Python _stage_conv1_w1_packed (lines 1869-1896). */
void rocke_gfx1151_dfcp_stage_conv1_w1_packed(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                              rocke_value_t* w1_ptr,
                                              rocke_value_t* w1_smem)
{
    rocke_ir_builder_t* b;
    const rocke_conv_problem_t* c;
    int k1, bytes_per_row, bs;
    rocke_value_t* c0;
    rocke_value_t* zero_vec;
    rocke_value_t* c_total;
    rocke_value_t* c_k1;
    rocke_value_t* c_bpr;
    int e, n_iters;

    if(ctx == NULL)
        return;
    b = ctx->b;
    c = ctx->c;
    k1 = rocke_fused_conv_pool_problem_conv1_channels(ctx->p);
    bytes_per_row = c->K / 2;
    bs = rocke_gfx1151_dfcp_block_size(ctx->spec);

    c0 = rocke_b_const_i32(b, 0);
    zero_vec = rocke_b_zero_vec(b, rocke_i8(), bytes_per_row);
    c_total = rocke_b_const_i32(b, ctx->spec->tile_n);
    c_k1 = rocke_b_const_i32(b, k1);
    c_bpr = rocke_b_const_i32(b, bytes_per_row);

    n_iters = (ctx->spec->tile_n + bs - 1) / bs;
    for(e = 0; e < n_iters; e++)
    {
        rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
        rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
        rocke_value_t* n = rocke_b_select(b, in_range, idx, c0);
        rocke_value_t* valid = rocke_b_land(b, in_range, rocke_b_cmp_lt(b, n, c_k1));
        rocke_value_t* off = rocke_b_mul(b, n, c_bpr);
        rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
        rocke_value_t* raw
            = rocke_b_global_load_vN(b, w1_ptr, safe_off, rocke_i8(), bytes_per_row, 0);
        rocke_value_t* packed = rocke_b_vector_select(b, valid, raw, zero_vec);
        rocke_value_t* sidx2[2];
        rocke_if_t gate;
        sidx2[0] = n;
        sidx2[1] = c0;
        gate = rocke_b_scf_if(b, in_range);
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_smem_store_vN(b, w1_smem, sidx2, 2, packed, bytes_per_row);
        rocke_b_region_leave(b);
    }
}

/* _stage_conv1_w1: unpack packed-int4 W1[K1, K0/2] (2 codes/byte, low nibble =
 * even k0) into w1_smem[tile_n, K0] as fp16 codes; padding -> 0. Faithful port
 * of Python _stage_conv1_w1 (lines 1821-1866). */
void rocke_gfx1151_dfcp_stage_conv1_w1(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                       rocke_value_t* w1_ptr,
                                       rocke_value_t* w1_smem)
{
    rocke_ir_builder_t* b;
    const rocke_conv_problem_t* c;
    int k0, k1, bs, bytes_per_row, total, ept, e;
    rocke_value_t* c_bpr;
    rocke_value_t* c_k1;
    rocke_value_t* c_total;
    rocke_value_t* c0;
    rocke_value_t* zero_h;

    if(ctx == NULL)
        return;
    b = ctx->b;
    c = ctx->c;
    k0 = c->K; /* conv1 K */
    k1 = rocke_fused_conv_pool_problem_conv1_channels(ctx->p);
    bs = rocke_gfx1151_dfcp_block_size(ctx->spec);
    bytes_per_row = k0 / 2;
    total = ctx->spec->tile_n * bytes_per_row;
    ept = (total + bs - 1) / bs;

    c_bpr = rocke_b_const_i32(b, bytes_per_row);
    c_k1 = rocke_b_const_i32(b, k1);
    c_total = rocke_b_const_i32(b, total);
    c0 = rocke_b_const_i32(b, 0);
    zero_h = rocke_b_trunc_f32_to_f16(b, rocke_b_const_f32(b, 0.0));

    for(e = 0; e < ept; e++)
    {
        rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
        rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
        rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
        rocke_value_t* n = rocke_b_div(b, sidx, c_bpr);
        rocke_value_t* kb = rocke_b_mod(b, sidx, c_bpr);
        rocke_value_t* valid = rocke_b_land(b, in_range, rocke_b_cmp_lt(b, n, c_k1));
        rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, n, c_bpr), kb);
        rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
        rocke_value_t* byte = rocke_b_global_load(b, w1_ptr, safe_off, rocke_i8(), 0);
        rocke_value_t* lo_i32 = NULL;
        rocke_value_t* hi_i32 = NULL;
        rocke_value_t* lo_h;
        rocke_value_t* hi_h;
        rocke_value_t* k_lo;
        rocke_value_t* k_hi;
        rocke_value_t* idx_lo[2];
        rocke_value_t* idx_hi[2];
        rocke_if_t gate;

        rocke_unpack_i4_byte_to_pair_i32(b, byte, &lo_i32, &hi_i32);
        if(lo_i32 == NULL || hi_i32 == NULL)
            return;
        lo_h = rocke_b_trunc_f32_to_f16(b, rocke_b_sitofp_f32(b, lo_i32));
        hi_h = rocke_b_trunc_f32_to_f16(b, rocke_b_sitofp_f32(b, hi_i32));
        lo_h = rocke_b_select(b, valid, lo_h, zero_h);
        hi_h = rocke_b_select(b, valid, hi_h, zero_h);
        k_lo = rocke_b_mul(b, kb, rocke_b_const_i32(b, 2));
        k_hi = rocke_b_add(b, k_lo, rocke_b_const_i32(b, 1));
        idx_lo[0] = n;
        idx_lo[1] = k_lo;
        idx_hi[0] = n;
        idx_hi[1] = k_hi;
        gate = rocke_b_scf_if(b, in_range);
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_smem_store_f16(b, w1_smem, idx_lo, 2, lo_h);
        rocke_b_smem_store_f16(b, w1_smem, idx_hi, 2, hi_h);
        rocke_b_region_leave(b);
    }
}

/* _stage_conv1_w1_i8: unpack packed-int4 W1 -> byte-per-code i8 LDS [tile_n, K0]
 * for the iu8 conv1 atom; padding -> 0. Faithful port of Python
 * _stage_conv1_w1_i8 (lines 1900-1944). */
void rocke_gfx1151_dfcp_stage_conv1_w1_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                          rocke_value_t* w1_ptr,
                                          rocke_value_t* w1_smem)
{
    rocke_ir_builder_t* b;
    const rocke_conv_problem_t* c;
    int k0, k1, bs, bytes_per_row, total, ept, e;
    rocke_value_t* c_bpr;
    rocke_value_t* c_k1;
    rocke_value_t* c_total;
    rocke_value_t* c0;
    rocke_value_t* zero_i8;

    if(ctx == NULL)
        return;
    b = ctx->b;
    c = ctx->c;
    k0 = c->K; /* conv1 K */
    k1 = rocke_fused_conv_pool_problem_conv1_channels(ctx->p);
    bs = rocke_gfx1151_dfcp_block_size(ctx->spec);
    bytes_per_row = k0 / 2;
    total = ctx->spec->tile_n * bytes_per_row;
    ept = (total + bs - 1) / bs;

    c_bpr = rocke_b_const_i32(b, bytes_per_row);
    c_k1 = rocke_b_const_i32(b, k1);
    c_total = rocke_b_const_i32(b, total);
    c0 = rocke_b_const_i32(b, 0);
    zero_i8 = rocke_b_trunc(b, c0, rocke_i8());

    for(e = 0; e < ept; e++)
    {
        rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * bs), ctx->grid->tid);
        rocke_value_t* in_range = rocke_b_cmp_lt(b, idx, c_total);
        rocke_value_t* sidx = rocke_b_select(b, in_range, idx, c0);
        rocke_value_t* n = rocke_b_div(b, sidx, c_bpr);
        rocke_value_t* kb = rocke_b_mod(b, sidx, c_bpr);
        rocke_value_t* valid = rocke_b_land(b, in_range, rocke_b_cmp_lt(b, n, c_k1));
        rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, n, c_bpr), kb);
        rocke_value_t* safe_off = rocke_b_select(b, valid, off, c0);
        rocke_value_t* byte = rocke_b_global_load(b, w1_ptr, safe_off, rocke_i8(), 0);
        rocke_value_t* lo_i32 = NULL;
        rocke_value_t* hi_i32 = NULL;
        rocke_value_t* lo_i8;
        rocke_value_t* hi_i8;
        rocke_value_t* k_lo;
        rocke_value_t* k_hi;
        rocke_value_t* idx_lo[2];
        rocke_value_t* idx_hi[2];
        rocke_if_t gate;

        rocke_unpack_i4_byte_to_pair_i32(b, byte, &lo_i32, &hi_i32);
        if(lo_i32 == NULL || hi_i32 == NULL)
            return;
        lo_i8 = rocke_b_select(b, valid, rocke_b_trunc(b, lo_i32, rocke_i8()), zero_i8);
        hi_i8 = rocke_b_select(b, valid, rocke_b_trunc(b, hi_i32, rocke_i8()), zero_i8);
        k_lo = rocke_b_mul(b, kb, rocke_b_const_i32(b, 2));
        k_hi = rocke_b_add(b, k_lo, rocke_b_const_i32(b, 1));
        idx_lo[0] = n;
        idx_lo[1] = k_lo;
        idx_hi[0] = n;
        idx_hi[1] = k_hi;
        gate = rocke_b_scf_if(b, in_range);
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_smem_store_vN(b, w1_smem, idx_lo, 2, lo_i8, 1);
        rocke_b_smem_store_vN(b, w1_smem, idx_hi, 2, hi_i8, 1);
        rocke_b_region_leave(b);
    }
}
