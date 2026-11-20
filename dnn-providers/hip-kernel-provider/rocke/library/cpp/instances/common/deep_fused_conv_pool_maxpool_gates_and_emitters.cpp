// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_deep_fused_conv_pool_maxpool_gates_and_emitters.c -- chunked C99 port
 * of the maxpool register-residency gates + the three output-writeback emitters
 * from rocke/instances/common/deep_fused_conv_pool.py:
 *
 *   _maxpool_is_intra_lane(spec, grid)            (py 1013-1043)
 *     -> rocke_dfcp_maxpool_is_intra_lane
 *   _maxpool_is_intra_lane_wmma(spec, grid, op)   (py 1046-1080)
 *     -> rocke_dfcp_maxpool_is_intra_lane_wmma
 *   _emit_inline_maxpool_from_cshuffle(...)       (py 913-1011)
 *     -> rocke_dfcp_emit_inline_maxpool_from_cshuffle  (generic cshuffle-LDS gather)
 *   _emit_wmma_maxpool_from_registers(...)        (py 1083-1146)
 *     -> rocke_dfcp_emit_wmma_maxpool_from_registers   (RDNA4 register-resident)
 *   _emit_inline_maxpool_from_registers(...)      (py 1159-1209)
 *     -> rocke_dfcp_emit_inline_maxpool_from_registers (MFMA-32x32 vec<16> intra-lane)
 *
 * These are the three write-back paths the epilogue routes among, plus the two
 * pure predicates that select between them.
 *
 * The builder-call sequence is byte-identical to the Python. The grid / op
 * accessors used here are the now-landed geometry/arch ports: WarpGrid is the
 * struct in rocke/helper_rocke.helpers.epilogues.h (compile-time geometry fields +
 * bound SSA Values) with the @property analogues
 * rocke_warp_grid_mfmas_per_warp_{m,n}; MmaOp is the struct in rocke/arch_target.h.
 * The opaque forward-decls in the deep_fused_conv_pool helper header alias these
 * same struct tags, so including the real headers here is ABI-compatible.
 */
#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
#include "rocke/instance_deep_fused_conv_pool_internal.h"

#include "rocke/arch_target.h" /* full rocke_mma_op_t struct */
#include "rocke/helper_rocke.helpers.epilogues.h" /* full rocke_warp_grid_t struct + property fns */

/* ===================================================================== *
 *  maxpool register-residency gates (pure)
 * ===================================================================== */

/* _maxpool_is_intra_lane(spec, grid) (py 1013-1043). MFMA-32x32 fast path:
 * one 32x32 atom per warp + warp_n==1 => each lane owns a vec<16> accumulator
 * tiling a 4x4 conv block for one channel; a 2x2 stride-2 pool reduces purely
 * intra-lane. The exact 32x32 geometry check naturally returns false for WMMA. */
bool rocke_dfcp_maxpool_is_intra_lane(const rocke_deep_fused_conv_pool_spec_t* spec,
                                      const rocke_warp_grid_t* grid)
{
    const rocke_fused_conv_pool_problem_t* p;
    int conv_tile_h;
    int conv_tile_w;

    if(spec == NULL || grid == NULL)
    {
        return false;
    }
    p = &spec->problem;
    conv_tile_h = spec->pool_tile_h * p->pool_stride_h;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;

    return grid->warp_tile_m == 32 && grid->warp_tile_n == 32
           && rocke_warp_grid_mfmas_per_warp_m(NULL, grid) == 1
           && rocke_warp_grid_mfmas_per_warp_n(NULL, grid) == 1 && grid->warp_n == 1
           && grid->warp_m == 2 && p->pool_stride_h == 2 && p->pool_stride_w == 2
           && conv_tile_h == 8 && conv_tile_w == 8 && spec->tile_m == 64
           && rocke_fused_conv_pool_problem_conv1_channels(p) <= 32;
}

/* _maxpool_is_intra_lane_wmma(spec, grid, op) (py 1046-1080). WMMA analogue:
 * wave32 16x16x16; a lane owns 8 consecutive conv-M rows at one channel, so a
 * 2x2 stride-2 pool with warp_m==pool_tile_h and mfmas_per_warp_m==2 keeps all
 * four window corners in the same lane across two adjacent m-tile accs. */
bool rocke_dfcp_maxpool_is_intra_lane_wmma(const rocke_deep_fused_conv_pool_spec_t* spec,
                                           const rocke_warp_grid_t* grid,
                                           const rocke_mma_op_t* op)
{
    const rocke_fused_conv_pool_problem_t* p;
    int conv_tile_w;

    if(spec == NULL || grid == NULL || op == NULL)
    {
        return false;
    }
    p = &spec->problem;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;

    return grid->wave_size == 32 && op->m == 16 && op->n == 16 && grid->warp_tile_m == 16
           && grid->warp_tile_n == 16 && rocke_warp_grid_mfmas_per_warp_m(NULL, grid) == 2
           && grid->warp_n == 1 && p->pool_stride_h == 2 && p->pool_stride_w == 2
           && conv_tile_w == 16 && grid->warp_m == spec->pool_tile_h
           && rocke_fused_conv_pool_problem_conv1_channels(p) <= 32;
}

/* ===================================================================== *
 *  _emit_inline_maxpool_from_cshuffle (py 913-1011)
 *
 *  Reduce the staged conv tile into final pooled NHWK output. When `epilogue`
 *  is non-NULL it is applied to each pooled fp32 result before the fp16 store
 *  (the deferred conv1 epilogue).
 * ===================================================================== */
void rocke_dfcp_emit_inline_maxpool_from_cshuffle(rocke_ir_builder_t* b,
                                                  const rocke_deep_fused_conv_pool_spec_t* spec,
                                                  rocke_value_t* c_smem,
                                                  rocke_value_t* y_rsrc,
                                                  const rocke_warp_grid_t* grid,
                                                  const rocke_conv_acc_epilogue_t* epilogue)
{
    const rocke_fused_conv_pool_problem_t* p;
    int out_k;
    int conv_tile_w;
    int kvec;
    int block_size;
    int kblocks;
    int total_vec;
    int elems_per_thread;
    int pool_wo;
    int e;

    rocke_value_t* c_total_vec;
    rocke_value_t* c_kblocks;
    rocke_value_t* c_kvec;
    rocke_value_t* c_pool_tile_w;
    rocke_value_t* c_conv_tile_w;
    rocke_value_t* c_out_k;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* neg_inf;
    rocke_value_t* block_pool_h;
    rocke_value_t* block_pool_w;

    static const int kvec_cands[3] = {8, 4, 2};
    int ci;

    if(b == NULL || spec == NULL)
    {
        return;
    }
    p = &spec->problem;
    out_k = rocke_fused_conv_pool_problem_conv1_channels(p);
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    pool_wo = rocke_fused_conv_pool_problem_pool_wo(p);
    block_size = rocke_deep_fused_conv_pool_spec_block_size(spec);

    /* Pick the largest valid kvec width that divides out_k while keeping >= half
     * the block's threads active. */
    kvec = 1;
    for(ci = 0; ci < 3; ++ci)
    {
        int cand = kvec_cands[ci];
        if(out_k % cand == 0
           && (spec->pool_tile_h * spec->pool_tile_w * (out_k / cand)) >= block_size / 2)
        {
            kvec = cand;
            break;
        }
    }
    kblocks = out_k / kvec;
    total_vec = spec->pool_tile_h * spec->pool_tile_w * kblocks;
    elems_per_thread = (total_vec + block_size - 1) / block_size;

    c_total_vec = rocke_b_const_i32(b, total_vec);
    c_kblocks = rocke_b_const_i32(b, kblocks);
    c_kvec = rocke_b_const_i32(b, kvec);
    c_pool_tile_w = rocke_b_const_i32(b, spec->pool_tile_w);
    c_conv_tile_w = rocke_b_const_i32(b, conv_tile_w);
    c_out_k = rocke_b_const_i32(b, out_k);
    c_half_bytes = rocke_b_const_i32(b, 2);
    oob_sentinel = rocke_b_const_i32(b, (int64_t)2147483647);
    neg_inf = rocke_b_const_f32(b, -3.4028234663852886e38);
    /* block_pool_h/w = b.mul(b.block_id_*(), b.const_i32(...)). Python evaluates
     * mul args left-to-right (block_id before const); C call-arg order is
     * unspecified (GCC: right-to-left). Hoist block_id into temps to pin
     * Python source-order so later SSA names line up. */
    {
        rocke_value_t* bid_y = rocke_b_block_id_y(b);
        block_pool_h = rocke_b_mul(b, bid_y, rocke_b_const_i32(b, spec->pool_tile_h));
    }
    {
        rocke_value_t* bid_z = rocke_b_block_id_z(b);
        block_pool_w = rocke_b_mul(b, bid_z, rocke_b_const_i32(b, spec->pool_tile_w));
    }

    for(e = 0; e < elems_per_thread; ++e)
    {
        rocke_value_t* vec_idx;
        rocke_value_t* in_range;
        rocke_value_t* safe_vec_idx;
        rocke_value_t* kb;
        rocke_value_t* k0;
        rocke_value_t* t0;
        rocke_value_t* local_pwo;
        rocke_value_t* local_pho;
        rocke_value_t* global_pho;
        rocke_value_t* global_pwo;
        rocke_value_t* accs[8]; /* kvec in {1,2,4,8} */
        int yy;
        int j;
        rocke_value_t* y_base_elems;
        rocke_value_t* halves[8];
        rocke_value_t* base_off_bytes;
        rocke_value_t* safe_base;

        {
            /* hoist mul operands in Python's left-to-right order */
            rocke_value_t* e_c = rocke_b_const_i32(b, e);
            rocke_value_t* bs_c = rocke_b_const_i32(b, block_size);
            vec_idx = rocke_b_add(b, rocke_b_mul(b, e_c, bs_c), grid->tid);
        }
        in_range = rocke_b_cmp_lt(b, vec_idx, c_total_vec);
        safe_vec_idx = rocke_b_select(b, in_range, vec_idx, rocke_b_const_i32(b, 0));

        kb = rocke_b_mod(b, safe_vec_idx, c_kblocks);
        k0 = rocke_b_mul(b, kb, c_kvec);
        t0 = rocke_b_div(b, safe_vec_idx, c_kblocks);
        local_pwo = rocke_b_mod(b, t0, c_pool_tile_w);
        local_pho = rocke_b_div(b, t0, c_pool_tile_w);
        global_pho = rocke_b_add(b, block_pool_h, local_pho);
        global_pwo = rocke_b_add(b, block_pool_w, local_pwo);

        for(j = 0; j < kvec; ++j)
        {
            accs[j] = neg_inf;
        }
        for(yy = 0; yy < 2; ++yy)
        {
            /* local_conv_h = b.add(b.mul(local_pho, b.const_i32(2)),
             *                      b.const_i32(yy))
             * Python evaluates the add's args left-to-right (the mul, with its
             * const 2, emits BEFORE const(yy)); for yy==0 const(0) is still
             * emitted and DCE'd. C call-arg order is unspecified (GCC:
             * right-to-left); hoist the mul into a temp to pin source-order. */
            rocke_value_t* lch_mul = rocke_b_mul(b, local_pho, rocke_b_const_i32(b, 2));
            rocke_value_t* local_conv_h = rocke_b_add(b, lch_mul, rocke_b_const_i32(b, yy));
            int xx;
            for(xx = 0; xx < 2; ++xx)
            {
                rocke_value_t* lcw_mul = rocke_b_mul(b, local_pwo, rocke_b_const_i32(b, 2));
                rocke_value_t* local_conv_w = rocke_b_add(b, lcw_mul, rocke_b_const_i32(b, xx));
                rocke_value_t* cml_mul = rocke_b_mul(b, local_conv_h, c_conv_tile_w);
                rocke_value_t* conv_m_local = rocke_b_add(b, cml_mul, local_conv_w);
                rocke_value_t* idx[2];
                rocke_value_t* v_vec;
                idx[0] = conv_m_local;
                idx[1] = k0;
                v_vec = rocke_b_smem_load_vN_f16(b, c_smem, idx, 2, kvec);
                for(j = 0; j < kvec; ++j)
                {
                    accs[j] = rocke_b_fmax(
                        b, accs[j], rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, v_vec, j)));
                }
            }
        }

        y_base_elems = rocke_b_add(
            b,
            rocke_b_mul(b,
                        rocke_b_add(b,
                                    rocke_b_mul(b, global_pho, rocke_b_const_i32(b, pool_wo)),
                                    global_pwo),
                        c_out_k),
            k0);
        for(j = 0; j < kvec; ++j)
        {
            rocke_value_t* acc = accs[j];
            if(epilogue != NULL)
            {
                acc = rocke_dfcp_apply_epilogue_scalar(b, epilogue, acc);
            }
            halves[j] = rocke_b_trunc_f32_to_f16(b, acc);
        }
        base_off_bytes = rocke_b_mul(b, y_base_elems, c_half_bytes);
        safe_base = rocke_b_select(b, in_range, base_off_bytes, oob_sentinel);
        if(kvec >= 2)
        {
            rocke_value_t* y_vec = rocke_b_vec_pack(b, halves, kvec, rocke_f16());
            rocke_b_buffer_store_vN_f16(
                b, y_rsrc, safe_base, rocke_b_const_i32(b, 0), y_vec, kvec / 2);
        }
        else
        {
            rocke_b_buffer_store_f16(b, y_rsrc, safe_base, rocke_b_const_i32(b, 0), halves[0]);
        }
    }
}

/* ===================================================================== *
 *  _emit_wmma_maxpool_from_registers (py 1083-1146)
 *
 *  RDNA4 register-resident maxpool (no conv1->maxpool LDS handoff). Gated by
 *  rocke_dfcp_maxpool_is_intra_lane_wmma. Each lane reduces the four corners of
 *  every pool window it owns straight from its conv1 WMMA accumulators.
 * ===================================================================== */
void rocke_dfcp_emit_wmma_maxpool_from_registers(rocke_ir_builder_t* b,
                                                 const rocke_deep_fused_conv_pool_spec_t* spec,
                                                 rocke_value_t* const* conv1_accs,
                                                 size_t num_accs,
                                                 rocke_value_t* y_rsrc,
                                                 const rocke_warp_grid_t* grid,
                                                 const rocke_mma_op_t* op,
                                                 const rocke_conv_acc_epilogue_t* epilogue)
{
    const rocke_fused_conv_pool_problem_t* p;
    int out_k;
    int mfmas_n;
    int pool_wo;
    int w_local;

    (void)op; /* part of the emitter signature family; not referenced here */

    rocke_value_t* col;
    rocke_value_t* half;
    rocke_value_t* block_pool_h;
    rocke_value_t* block_pool_w;
    rocke_value_t* gpho;
    rocke_value_t* pwo_base;
    rocke_value_t* oob_sentinel;
    rocke_value_t* c_pool_wo;
    rocke_value_t* c_out_k;
    rocke_value_t* c_half_bytes;
    rocke_value_t* row_off;

    (void)num_accs;

    if(b == NULL || spec == NULL)
    {
        return;
    }
    p = &spec->problem;
    out_k = rocke_fused_conv_pool_problem_conv1_channels(p);
    mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    pool_wo = rocke_fused_conv_pool_problem_pool_wo(p);

    col = rocke_b_mod(b, grid->lane, rocke_b_const_i32(b, 16)); /* channel within an n-atom */
    half = rocke_b_div(
        b, grid->lane, rocke_b_const_i32(b, 16)); /* which 8-col half of the conv row */
    /* block_pool_h = b.mul(b.block_id_y(), b.const_i32(pool_tile_h)); _w likewise.
     * Python evaluates the mul args left-to-right (block_id before const). C
     * call-arg order is unspecified (GCC: right-to-left); hoist block_id into
     * temps to pin Python source-order so later SSA names line up. */
    {
        rocke_value_t* bid_y = rocke_b_block_id_y(b);
        block_pool_h = rocke_b_mul(b, bid_y, rocke_b_const_i32(b, spec->pool_tile_h));
    }
    {
        rocke_value_t* bid_z = rocke_b_block_id_z(b);
        block_pool_w = rocke_b_mul(b, bid_z, rocke_b_const_i32(b, spec->pool_tile_w));
    }
    gpho = rocke_b_add(b, block_pool_h, grid->warp_m_idx); /* pho == warp_m_idx */
    /* pwo = half*4 + w_local; conv cols 2*pwo,2*pwo+1 -> slots 2*w_local,2*w_local+1. */
    pwo_base = rocke_b_add(b, block_pool_w, rocke_b_mul(b, half, rocke_b_const_i32(b, 4)));

    oob_sentinel = rocke_b_const_i32(b, (int64_t)2147483647);
    c_pool_wo = rocke_b_const_i32(b, pool_wo);
    c_out_k = rocke_b_const_i32(b, out_k);
    c_half_bytes = rocke_b_const_i32(b, 2);
    row_off = rocke_b_mul(b, gpho, c_pool_wo); /* gpho*pool_wo, shared across windows */

    for(w_local = 0; w_local < 4; ++w_local)
    {
        rocke_value_t* gpwo = rocke_b_add(b, pwo_base, rocke_b_const_i32(b, w_local));
        int s0 = 2 * w_local;
        int s1 = 2 * w_local + 1;
        rocke_value_t* pix_off = rocke_b_mul(
            b, rocke_b_add(b, row_off, gpwo), c_out_k); /* (gpho*pool_wo+gpwo)*out_k */
        int ni;
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc_top = conv1_accs[0 * mfmas_n + ni]; /* mi=0 -> conv row 2*pho */
            rocke_value_t* acc_bot = conv1_accs[1 * mfmas_n + ni]; /* mi=1 -> conv row 2*pho+1 */
            rocke_value_t* top0 = rocke_b_vec_extract(b, acc_top, s0);
            rocke_value_t* top1 = rocke_b_vec_extract(b, acc_top, s1);
            rocke_value_t* bot0 = rocke_b_vec_extract(b, acc_bot, s0);
            rocke_value_t* bot1 = rocke_b_vec_extract(b, acc_bot, s1);
            rocke_value_t* pool3 = rocke_b_fmax3(b, top0, top1, bot0);
            rocke_value_t* acc;
            rocke_value_t* y_h;
            rocke_value_t* channel;
            rocke_value_t* in_range;
            rocke_value_t* y_off_bytes;
            rocke_value_t* safe_off;

            if(epilogue != NULL && rocke_dfcp_epilogue_is_relu_only(epilogue))
            {
                acc = rocke_b_fmax3(b, pool3, bot1, rocke_b_const_f32(b, 0.0));
            }
            else
            {
                acc = rocke_b_fmax(b, pool3, bot1);
            }
            if(epilogue != NULL && !rocke_dfcp_epilogue_is_relu_only(epilogue))
            {
                acc = rocke_dfcp_apply_epilogue_scalar(b, epilogue, acc);
            }
            y_h = rocke_b_trunc_f32_to_f16(b, acc);
            channel = rocke_b_add(b, rocke_b_const_i32(b, ni * 16), col);
            in_range = rocke_b_cmp_lt(b, channel, c_out_k);
            y_off_bytes = rocke_b_mul(b, rocke_b_add(b, pix_off, channel), c_half_bytes);
            safe_off = rocke_b_select(b, in_range, y_off_bytes, oob_sentinel);
            rocke_b_buffer_store_f16(b, y_rsrc, safe_off, rocke_b_const_i32(b, 0), y_h);
        }
    }
}

/* Pool window (pho_l, pwo_l) -> the four accumulator slots holding its corners
 * (py 1148-1156 _INTRA_LANE_WINDOW_SLOTS). Derived from the 32x32 C-fragment
 * layout: slot = (i//4)*4 + (i%4) with i//4 = pho_l*2 + yy, i%4 = pwo_l*2 + xx
 * over the 2x2 window (yy, xx in 0..1). Indexed [pho_l][pwo_l][corner]. */
static const int dfcp_intra_lane_window_slots[2][2][4] = {
    {{0, 1, 4, 5}, {2, 3, 6, 7}},
    {{8, 9, 12, 13}, {10, 11, 14, 15}},
};

/* ===================================================================== *
 *  _emit_inline_maxpool_from_registers (py 1159-1209)
 *
 *  Reduce the conv1 accumulators directly into pooled NHWK output (MFMA-only).
 *  Eliminates the conv1->maxpool cshuffle handoff: each lane reduces its own
 *  vec<16> accumulator. Gated by rocke_dfcp_maxpool_is_intra_lane. When `epilogue`
 *  is non-NULL it is applied once per pooled fp32 result.
 * ===================================================================== */
void rocke_dfcp_emit_inline_maxpool_from_registers(rocke_ir_builder_t* b,
                                                   const rocke_deep_fused_conv_pool_spec_t* spec,
                                                   rocke_value_t* const* conv1_accs,
                                                   size_t num_accs,
                                                   rocke_value_t* y_rsrc,
                                                   const rocke_warp_grid_t* grid,
                                                   const rocke_conv_acc_epilogue_t* epilogue)
{
    const rocke_fused_conv_pool_problem_t* p;
    int out_k;
    int pool_wo;
    rocke_value_t* acc_vec;

    rocke_value_t* channel;
    rocke_value_t* m_blk;
    rocke_value_t* block_pool_h;
    rocke_value_t* block_pool_w;
    rocke_value_t* pho_base;
    rocke_value_t* pwo_base;
    rocke_value_t* in_range;
    rocke_value_t* oob_sentinel;
    rocke_value_t* c_pool_wo;
    rocke_value_t* c_out_k;
    rocke_value_t* c_half_bytes;
    int pho_l;

    (void)num_accs;

    if(b == NULL || spec == NULL)
    {
        return;
    }
    p = &spec->problem;
    out_k = rocke_fused_conv_pool_problem_conv1_channels(p);
    pool_wo = rocke_fused_conv_pool_problem_pool_wo(p);
    acc_vec = conv1_accs[0];

    channel = rocke_b_mod(b, grid->lane, rocke_b_const_i32(b, 32));
    m_blk = rocke_b_div(b, grid->lane, rocke_b_const_i32(b, 32));
    /* block_pool_h/w = b.mul(b.block_id_*(), b.const_i32(...)). Python evaluates
     * mul args left-to-right (block_id before const); C call-arg order is
     * unspecified (GCC: right-to-left). Hoist block_id into temps to pin
     * Python source-order so later SSA names line up. */
    {
        rocke_value_t* bid_y = rocke_b_block_id_y(b);
        block_pool_h = rocke_b_mul(b, bid_y, rocke_b_const_i32(b, spec->pool_tile_h));
    }
    {
        rocke_value_t* bid_z = rocke_b_block_id_z(b);
        block_pool_w = rocke_b_mul(b, bid_z, rocke_b_const_i32(b, spec->pool_tile_w));
    }
    pho_base
        = rocke_b_add(b, block_pool_h, rocke_b_mul(b, grid->warp_m_idx, rocke_b_const_i32(b, 2)));
    pwo_base = rocke_b_add(b, block_pool_w, rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 2)));

    in_range = rocke_b_cmp_lt(b, channel, rocke_b_const_i32(b, out_k));
    oob_sentinel = rocke_b_const_i32(b, (int64_t)2147483647);
    c_pool_wo = rocke_b_const_i32(b, pool_wo);
    c_out_k = rocke_b_const_i32(b, out_k);
    c_half_bytes = rocke_b_const_i32(b, 2);

    for(pho_l = 0; pho_l < 2; ++pho_l)
    {
        rocke_value_t* gpho = rocke_b_add(b, pho_base, rocke_b_const_i32(b, pho_l));
        int pwo_l;
        for(pwo_l = 0; pwo_l < 2; ++pwo_l)
        {
            rocke_value_t* gpwo = rocke_b_add(b, pwo_base, rocke_b_const_i32(b, pwo_l));
            const int* slots = dfcp_intra_lane_window_slots[pho_l][pwo_l];
            int s0 = slots[0];
            int s1 = slots[1];
            int s2 = slots[2];
            int s3 = slots[3];
            rocke_value_t* acc = rocke_b_fmax(
                b,
                rocke_b_fmax(
                    b, rocke_b_vec_extract(b, acc_vec, s0), rocke_b_vec_extract(b, acc_vec, s1)),
                rocke_b_fmax(
                    b, rocke_b_vec_extract(b, acc_vec, s2), rocke_b_vec_extract(b, acc_vec, s3)));
            rocke_value_t* y_h;
            rocke_value_t* y_off_elems;
            rocke_value_t* y_off_bytes;
            rocke_value_t* safe_off;

            if(epilogue != NULL)
            {
                acc = rocke_dfcp_apply_epilogue_scalar(b, epilogue, acc);
            }
            y_h = rocke_b_trunc_f32_to_f16(b, acc);
            y_off_elems = rocke_b_add(
                b,
                rocke_b_mul(b, rocke_b_add(b, rocke_b_mul(b, gpho, c_pool_wo), gpwo), c_out_k),
                channel);
            y_off_bytes = rocke_b_mul(b, y_off_elems, c_half_bytes);
            safe_off = rocke_b_select(b, in_range, y_off_bytes, oob_sentinel);
            rocke_b_buffer_store_f16(b, y_rsrc, safe_off, rocke_b_const_i32(b, 0), y_h);
        }
    }
}
