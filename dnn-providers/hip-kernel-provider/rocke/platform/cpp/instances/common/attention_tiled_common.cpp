// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_tiled_common.cpp -- emission helpers shared by the gfx942 and
 * gfx950 forward attention kernels. Mirrors
 * library/kernels/common/_attention_tiled_shared.py.
 *
 * The two 32x32 C helpers reproduce the Python builder-call sequence
 * byte-faithfully: same div/mod constants, same calculate_x ys/ps wiring, same
 * trailing add for the N-tile base. C32_DIST (built once at Python module
 * import from make_static_tile_distribution(make_c_warp_dstr_encoding(
 * MfmaAtom.f16_32x32x16()))) is reproduced as a lazily-built cached
 * distribution off rocke_mfma_atom("f16", 32, 32, 16).
 *
 * Lifetime: every IR node is arena-owned (rocke_ir_builder_t.arena). The cached
 * C32_DIST is built on the first call's builder arena. Because the C
 * builder/arena is freed at the end of each build, the cache is per-build:
 * rocke_attn2d_c32_dist_reset() clears it at every build entry so build N+1
 * never reads build N's freed arena.
 */

#include <stdio.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom, rocke_make_c_warp_dstr_encoding */
#include "rocke/helper_rocke.helpers.distribution.h" /* rocke_tile_distribution_t, calculate_x */
#include "rocke/instance_attention_tiled_common_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ------------------------------------------------------ _C32_DIST (cached) */
/* make_static_tile_distribution(make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16()))
 * -- a host-side distribution the Python caches at module scope. Built lazily on
 * the first 32x32-C-helper call of a build.
 *
 * Re-entrancy: the cached rocke_tile_distribution_t (and all its inner nodes) is
 * arena-allocated off the *current build's* rocke_ir_builder. When that builder is
 * freed at the end of a build, this pointer dangles; reusing it on the next
 * build feeds freed memory into calculate_x. So the cache is per-build, not
 * process-lifetime: rocke_attn2d_c32_dist_reset() clears it at each build entry
 * (see the gfx942/gfx950 public entries). */
static const rocke_tile_distribution_t* g_c32_dist = NULL;

/* Re-entrancy reset: drop the dangling per-build cache before a new build. */
void rocke_attn2d_c32_dist_reset(void)
{
    g_c32_dist = NULL;
}

static const rocke_tile_distribution_t* rocke_attn2d_c32_dist(rocke_ir_builder_t* b)
{
    const rocke_mfma_atom_t* atom;
    const rocke_tile_distribution_encoding_t* enc;
    const rocke_tile_distribution_t* dist;

    if(g_c32_dist != NULL)
    {
        return g_c32_dist;
    }
    if(b == NULL)
    {
        return NULL;
    }

    atom = rocke_mfma_atom("f16", 32, 32, 16);
    if(atom == NULL)
    {
        ckc::raise_status(ROCKE_ERR_VALUE, "_C32_DIST: no f16 32x32x16 MFMA atom");
        return NULL;
    }
    enc = rocke_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
    {
        return NULL;
    }
    dist = rocke_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return NULL;
    }
    g_c32_dist = dist;
    return g_c32_dist;
}

/* ------------------------------------------------ _mfma_32x32_c_row / _col */

rocke_value_t* rocke__mfma_32x32_c_row(rocke_ir_builder_t* b, rocke_value_t* lane, int elem_idx)
{
    const rocke_tile_distribution_t* dist;
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2];

    if(b == NULL)
    {
        return NULL;
    }
    /* if not (0 <= elem_idx < 16): raise ValueError */
    if(!(elem_idx >= 0 && elem_idx < 16))
    {
        /* Raised as a ckc::Error; the public entry boundary records it on b. */
        ckc::raise_status(
            ROCKE_ERR_VALUE,
            ckc::format_error("mfma_32x32x16 elem_idx must be 0..15, got %d", elem_idx).c_str());
    }

    dist = rocke_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));

    /* y0 = const(elem_idx // 4); y1 = const(elem_idx % 4) */
    ys[0] = rocke_b_const_i32(b, (int64_t)(elem_idx / 4));
    ys[1] = rocke_b_const_i32(b, (int64_t)(elem_idx % 4));

    /* ps=[[m_blk, n]] */
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    /* row, _col = _C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]]) */
    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row */
}

rocke_value_t* rocke__mfma_32x32_c_col(rocke_ir_builder_t* b, rocke_value_t* lane, int n_tile32)
{
    const rocke_tile_distribution_t* dist;
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2];
    rocke_value_t* col;

    if(b == NULL)
    {
        return NULL;
    }

    dist = rocke_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));

    /* ys=[const(0), const(0)] */
    ys[0] = rocke_b_const_i32(b, 0);
    ys[1] = rocke_b_const_i32(b, 0);

    /* ps=[[m_blk, n]] */
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    /* _row, col = _C32_DIST.calculate_x(b, ys=[0, 0], ps=[[m_blk, n]]) */
    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    col = out_x[1];

    /* if n_tile32 == 0: return col; else return add(const(n_tile32*32), col) */
    if(n_tile32 == 0)
    {
        return col;
    }
    return rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n_tile32 * 32)), col);
}

/* ------------------------------------------- tiled 2D lane map (TiledLaneMap) */

/* lane_rg / lane_col: the prologue-cached SSA value, or a fresh op if a caller
 * runs before the prologue emits it. */
static rocke_value_t* lm_lane_rg(const rocke_attn_tiled_lane_map_t* lm)
{
    if(lm->lane_rg != NULL)
        return lm->lane_rg;
    return rocke_b_div(lm->b, lm->lane, rocke_b_const_i32(lm->b, 16));
}

static rocke_value_t* lm_lane_col(const rocke_attn_tiled_lane_map_t* lm)
{
    if(lm->lane_col != NULL)
        return lm->lane_col;
    return rocke_b_mod(lm->b, lm->lane, rocke_b_const_i32(lm->b, 16));
}

rocke_value_t* rocke_attn_tiled_in_warp_row(const rocke_attn_tiled_lane_map_t* lm, int r)
{
    /* atom_idx = r // 4 ; in_atom = r % 4
     * row = lane_rg*4 + (atom_idx*16 + in_atom) */
    int atom_idx = r / 4;
    int in_atom = r % 4;
    /* Python evaluates b.mul(lane_rg, const(4)) BEFORE the trailing const
     * (left-to-right arg order). Bind the mul to a temp so C's unspecified
     * arg-eval order does not allocate the trailing const ahead of the mul. */
    rocke_value_t* rg4 = rocke_b_mul(lm->b, lm_lane_rg(lm), rocke_b_const_i32(lm->b, 4));
    return rocke_b_add(lm->b, rg4, rocke_b_const_i32(lm->b, atom_idx * 16 + in_atom));
}

rocke_value_t* rocke_attn_tiled_state_row(const rocke_attn_tiled_lane_map_t* lm, int r)
{
    if(lm->use_mfma_32x32 && !lm->transposed_qk_32x32)
        return rocke__mfma_32x32_c_row(lm->b, lm->lane, r);
    return rocke_attn_tiled_in_warp_row(lm, r);
}

rocke_value_t* rocke_attn_tiled_bit2(rocke_ir_builder_t* b, rocke_value_t* v, int bit)
{
    /* Python: b.land(b.lshr(v, const(bit)), b.const_i32(1)) creates the lshr
     * (and its bit const) BEFORE the trailing const(1) (left-to-right). Bind the
     * lshr to a temp so C's arg-eval order does not allocate const(1) ahead of
     * the lshr and shift the SSA ids. */
    rocke_value_t* sh = rocke_b_lshr(b, v, rocke_b_const_i32(b, bit));
    return rocke_b_land(b, sh, rocke_b_const_i32(b, 1));
}

rocke_value_t* rocke_attn_tiled_select_lane_rg(const rocke_attn_tiled_lane_map_t* lm,
                                               rocke_value_t* v0,
                                               rocke_value_t* v1,
                                               rocke_value_t* v2,
                                               rocke_value_t* v3)
{
    /* select(lane_rg_is0, v0, select(lane_rg_is1, v1, select(lane_rg_is2, v2, v3))).
     * Reuse the cached lane_rg_is* SSA values when present, else recompute. */
    rocke_ir_builder_t* b = lm->b;
    rocke_value_t* is0 = lm->lane_rg_is0;
    rocke_value_t* is1 = lm->lane_rg_is1;
    rocke_value_t* is2 = lm->lane_rg_is2;
    if(is0 == NULL || is1 == NULL || is2 == NULL)
    {
        rocke_value_t* lane_rg = lm_lane_rg(lm);
        is0 = rocke_b_cmp_eq(b, lane_rg, rocke_b_const_i32(b, 0));
        is1 = rocke_b_cmp_eq(b, lane_rg, rocke_b_const_i32(b, 1));
        is2 = rocke_b_cmp_eq(b, lane_rg, rocke_b_const_i32(b, 2));
    }
    return rocke_b_select(b, is0, v0, rocke_b_select(b, is1, v1, rocke_b_select(b, is2, v2, v3)));
}

void rocke_attn_tiled_permute_p_c_to_a16(const rocke_attn_tiled_lane_map_t* lm,
                                         rocke_value_t* const* p_regs_f32,
                                         int n,
                                         rocke_value_t** out,
                                         int* out_n)
{
    rocke_ir_builder_t* b = lm->b;
    /* Python uses the prologue-cached lane_col_mod4 / lane_col_div4; reuse the
     * same SSA values instead of recomputing the mod/div here. */
    rocke_value_t* lane_col_mod4 = (lm->lane_col_mod4 != NULL)
                                       ? lm->lane_col_mod4
                                       : rocke_b_mod(b, lm_lane_col(lm), rocke_b_const_i32(b, 4));
    rocke_value_t* lane_col_div4 = (lm->lane_col_div4 != NULL)
                                       ? lm->lane_col_div4
                                       : rocke_b_div(b, lm_lane_col(lm), rocke_b_const_i32(b, 4));
    rocke_value_t* lane_rg = lm_lane_rg(lm); /* lane / 16 */

    /* The transform is defined for a 4-element register quad. */
    rocke_value_t* vals[4];
    int i;
    for(i = 0; i < 4; ++i)
        vals[i] = (i < n) ? p_regs_f32[i] : NULL;

    /* Two bit-level transposes: swap C (lane_col_mod4) with R (reg index). */
    for(int bit = 0; bit < 2; ++bit)
    {
        rocke_value_t* lane_bit = rocke_attn_tiled_bit2(b, lane_col_mod4, bit);
        int reg_bit = 1 << bit; /* 1 or 2: intra-quad -> dpp (quad_perm), no ds_swizzle */
        rocke_value_t* old[4];
        for(i = 0; i < 4; ++i)
            old[i] = vals[i];
        for(int reg = 0; reg < 4; ++reg)
        {
            rocke_value_t* partner = rocke_b_warp_shuffle_xor_dpp(b, old[reg ^ reg_bit], reg_bit);
            rocke_value_t* same_bit
                = rocke_b_cmp_eq(b, lane_bit, rocke_b_const_i32(b, (reg >> bit) & 1));
            vals[reg] = rocke_b_select(b, same_bit, old[reg], partner);
        }
    }

    /* Lane-field swaps: exchange A (lane_rg) and B (lane_col_div4). */
    {
        const int bits[2] = {0, 1};
        const int lane_xors[2] = {20, 40};
        for(int j = 0; j < 2; ++j)
        {
            int bit = bits[j];
            int lane_xor = lane_xors[j];
            rocke_value_t* a_bit = rocke_attn_tiled_bit2(b, lane_rg, bit);
            rocke_value_t* b_bit = rocke_attn_tiled_bit2(b, lane_col_div4, bit);
            rocke_value_t* swap = rocke_b_cmp_ne(b, a_bit, b_bit);
            for(i = 0; i < 4; ++i)
                vals[i] = rocke_b_select(
                    b, swap, rocke_b_warp_shuffle_xor(b, vals[i], lane_xor), vals[i]);
        }
    }

    for(i = 0; i < 4; ++i)
        out[i] = vals[i];
    if(out_n != NULL)
        *out_n = 4;
}

rocke_value_t* rocke_attn_tiled_pack_p_a16(const rocke_attn_tiled_lane_map_t* lm,
                                           rocke_value_t* const* p_regs_f32,
                                           int n)
{
    rocke_ir_builder_t* b = lm->b;
    rocke_value_t* permuted[4];
    rocke_value_t* casted[4];
    int out_n = 0;
    rocke_attn_tiled_permute_p_c_to_a16(lm, p_regs_f32, n, permuted, &out_n);
    for(int i = 0; i < 4; ++i)
        casted[i] = rocke_b_cast_f32_to(b, permuted[i], lm->dtype);
    return rocke_b_vec_pack(b, casted, 4, lm->dtype);
}

rocke_value_t* rocke_attn_tiled_pack_p_a32(const rocke_attn_tiled_lane_map_t* lm,
                                           rocke_value_t* const* p_regs0_f32,
                                           rocke_value_t* const* p_regs1_f32,
                                           int n)
{
    rocke_ir_builder_t* b = lm->b;
    rocke_value_t* vals0[4];
    rocke_value_t* vals1[4];
    rocke_value_t* lohi[8];
    int dummy = 0;
    rocke_attn_tiled_permute_p_c_to_a16(lm, p_regs0_f32, n, vals0, &dummy);
    rocke_attn_tiled_permute_p_c_to_a16(lm, p_regs1_f32, n, vals1, &dummy);
    for(int j = 0; j < 4; ++j)
    {
        rocke_value_t* v0 = vals0[j];
        rocke_value_t* v1 = vals1[j];
        rocke_value_t* v0_x16 = rocke_b_warp_shuffle_xor(b, v0, 16);
        rocke_value_t* v0_x32 = rocke_b_warp_shuffle_xor(b, v0, 32);
        rocke_value_t* v0_x48 = rocke_b_warp_shuffle_xor(b, v0, 48);
        rocke_value_t* v1_x16 = rocke_b_warp_shuffle_xor(b, v1, 16);
        rocke_value_t* v1_x32 = rocke_b_warp_shuffle_xor(b, v1, 32);
        rocke_value_t* v1_x48 = rocke_b_warp_shuffle_xor(b, v1, 48);
        /* lo[j] = select_lane_rg(v0, v0_x48, v1_x32, v1_x16) */
        lohi[j] = rocke_attn_tiled_select_lane_rg(lm, v0, v0_x48, v1_x32, v1_x16);
        /* hi[j] = select_lane_rg(v0_x16, v0_x32, v1_x48, v1) */
        lohi[4 + j] = rocke_attn_tiled_select_lane_rg(lm, v0_x16, v0_x32, v1_x48, v1);
    }
    {
        rocke_value_t* casted[8];
        for(int i = 0; i < 8; ++i)
            casted[i] = rocke_b_cast_f32_to(b, lohi[i], lm->dtype);
        return rocke_b_vec_pack(b, casted, 8, lm->dtype);
    }
}

/* ------------------------------------- fp8 K/V sync loaders (Fp8KvSyncLoader) */

/* Paged K/V byte offset for (tile_idx, linear_half, kv_head) through the
 * resolved descriptor. Only the i32 offset path is ported; a NULL descriptor
 * or offset falls back to const 0. */
static rocke_value_t* fp8_kv_voff(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                  rocke_value_t* tile_idx,
                                  rocke_value_t* linear_half)
{
    const char* names[3] = {"tile_idx", "linear_half", "kv_head"};
    rocke_value_t* vals[3] = {tile_idx, linear_half, ld->kv_head_idx};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;

    if(ld->kv_desc != NULL)
    {
        (void)rocke_transforms_descriptor_offset(ld->b, ld->kv_desc, names, vals, 3, &off, &valid);
    }
    if(off == NULL)
    {
        off = rocke_b_const_i32(ld->b, 0);
    }
    return off;
}

void rocke_attn_tiled_fp8_issue_dequant_loads(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx,
                                              bool is_v)
{
    rocke_ir_builder_t* b = ld->b;
    const int CHUNK = 8; /* fp8_elems_per_chunk */
    int total_chunks;
    int chunks_per_thread;
    int cols_per_row;
    rocke_value_t* scale;
    rocke_value_t* lds;
    rocke_value_t* src;
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }

    total_chunks = (ld->T * ld->HD) / CHUNK;
    if(ld->THREADS <= 0 || (total_chunks % ld->THREADS) != 0)
    {
        return;
    }
    chunks_per_thread = total_chunks / ld->THREADS;
    cols_per_row = ld->HD / CHUNK; /* HD // fp8_elems_per_chunk */

    /* lds_token selector: "V" -> value/v_scale/V_lds. */
    scale = is_v ? ld->v_scale : ld->k_scale;
    lds = is_v ? ld->v_lds : ld->k_lds;
    src = is_v ? ld->value : ld->key;

    for(call = 0; call < chunks_per_thread; ++call)
    {
        rocke_value_t* chunk_call = rocke_b_const_i32(b, call);
        rocke_value_t* chunk_thr = rocke_b_const_i32(b, ld->THREADS);
        rocke_value_t* chunk_id = rocke_b_add(b, rocke_b_mul(b, chunk_call, chunk_thr), ld->tid);
        rocke_value_t* row = rocke_b_div(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        /* Python: col = mul(mod(chunk_id, const(cols_per_row)), const(CHUNK));
         * the mod (and its const) are created before const(CHUNK). Sequence
         * explicitly so C arg-eval order does not reorder value creation. */
        rocke_value_t* col_mod = rocke_b_mod(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        rocke_value_t* col = rocke_b_mul(b, col_mod, rocke_b_const_i32(b, CHUNK));
        rocke_value_t* linear_half_first
            = rocke_b_add(b, rocke_b_mul(b, row, rocke_b_const_i32(b, ld->HD)), col);
        rocke_value_t* voff = fp8_kv_voff(ld, kv_tile_idx, linear_half_first);
        rocke_value_t* fp8_vec
            = rocke_b_global_load_vN(b, src, voff, rocke_fp8e4m3(), CHUNK, CHUNK);

        rocke_value_t* lo_comp[4];
        rocke_value_t* hi_comp[4];
        rocke_value_t* lo_quad;
        rocke_value_t* hi_quad;
        rocke_value_t* lo_f32x4;
        rocke_value_t* hi_f32x4;
        rocke_value_t* lo_scaled[4];
        rocke_value_t* hi_scaled[4];
        rocke_value_t* dequanted[8];
        rocke_value_t* packed;
        rocke_value_t* idx_store[3];
        int i;

        /* Python emits all 4 lo extracts (then pack), then all 4 hi extracts
         * (then pack) -- two separate list comprehensions, NOT interleaved. */
        for(i = 0; i < 4; ++i)
            lo_comp[i] = rocke_b_vec_extract(b, fp8_vec, i);
        lo_quad = rocke_b_vec_pack(b, lo_comp, 4, rocke_fp8e4m3());
        for(i = 0; i < 4; ++i)
            hi_comp[i] = rocke_b_vec_extract(b, fp8_vec, i + 4);
        hi_quad = rocke_b_vec_pack(b, hi_comp, 4, rocke_fp8e4m3());

        /* CORRECTNESS: unfused cvt + *scale (non-pow2 scale safety). */
        lo_f32x4 = rocke_b_cvt_pk_f32_fp8x4(b, lo_quad);
        hi_f32x4 = rocke_b_cvt_pk_f32_fp8x4(b, hi_quad);
        for(i = 0; i < 4; ++i)
        {
            lo_scaled[i] = rocke_b_fmul(b, rocke_b_vec_extract(b, lo_f32x4, i), scale);
            hi_scaled[i] = rocke_b_fmul(b, rocke_b_vec_extract(b, hi_f32x4, i), scale);
        }
        lo_f32x4 = rocke_b_vec_pack(b, lo_scaled, 4, rocke_f32());
        hi_f32x4 = rocke_b_vec_pack(b, hi_scaled, 4, rocke_f32());
        for(i = 0; i < 4; ++i)
        {
            dequanted[i] = rocke_b_cast_f32_to(b, rocke_b_vec_extract(b, lo_f32x4, i), ld->dtype);
        }
        for(i = 0; i < 4; ++i)
        {
            dequanted[i + 4]
                = rocke_b_cast_f32_to(b, rocke_b_vec_extract(b, hi_f32x4, i), ld->dtype);
        }
        packed = rocke_b_vec_pack(b, dequanted, CHUNK, ld->dtype);
        idx_store[0] = buf_idx;
        idx_store[1] = row;
        idx_store[2] = col;
        rocke_b_smem_store_vN(b, lds, idx_store, 3, packed, CHUNK);
    }
}

void rocke_attn_tiled_fp8_issue_v_mfma_stripe(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                              rocke_value_t* kv_tile_idx)
{
    rocke_ir_builder_t* b = ld->b;
    const int CHUNK = 8; /* fp8_elems_per_chunk */
    int total_chunks;
    int chunks_per_thread;
    int cols_per_row;
    int call;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }

    total_chunks = (ld->T * ld->HD) / CHUNK;
    if(ld->THREADS <= 0 || (total_chunks % ld->THREADS) != 0)
    {
        return;
    }
    chunks_per_thread = total_chunks / ld->THREADS;
    cols_per_row = ld->HD / CHUNK; /* HD // fp8_elems_per_chunk */

    for(call = 0; call < chunks_per_thread; ++call)
    {
        rocke_value_t* chunk_call = rocke_b_const_i32(b, call);
        rocke_value_t* chunk_thr = rocke_b_const_i32(b, ld->THREADS);
        rocke_value_t* chunk_id = rocke_b_add(b, rocke_b_mul(b, chunk_call, chunk_thr), ld->tid);
        rocke_value_t* token = rocke_b_div(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        /* Python: col = mul(mod(chunk_id, const(cols_per_row)), const(CHUNK))
         * creates the mod BEFORE const(CHUNK) (left-to-right); bind the mod so
         * C's arg-eval order does not allocate const(CHUNK) first. */
        rocke_value_t* col_mod = rocke_b_mod(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        rocke_value_t* col = rocke_b_mul(b, col_mod, rocke_b_const_i32(b, CHUNK));
        rocke_value_t* linear_first
            = rocke_b_add(b, rocke_b_mul(b, token, rocke_b_const_i32(b, ld->HD)), col);
        rocke_value_t* voff = fp8_kv_voff(ld, kv_tile_idx, linear_first);
        rocke_value_t* fp8_vec
            = rocke_b_global_load_vN(b, ld->value, voff, rocke_fp8e4m3(), CHUNK, CHUNK);
        rocke_value_t* stripe_idx = rocke_b_div(b, col, rocke_b_const_i32(b, 16));
        rocke_value_t* col_in_stripe = rocke_b_mod(b, col, rocke_b_const_i32(b, 16));
        rocke_value_t* idx_store[4];

        idx_store[0] = rocke_b_const_i32(b, 0);
        idx_store[1] = stripe_idx;
        idx_store[2] = token;
        idx_store[3] = col_in_stripe;
        rocke_b_smem_store_vN(b, ld->v_lds, idx_store, 4, fp8_vec, CHUNK);
    }
}

/* --------------------------------------- tiled 3D K/V LDS swizzle (KvLdsSwizzle) */

rocke_value_t* rocke_attn_tiled_kv_swz_col(const rocke_attn_tiled_kv_swizzle_t* s,
                                           rocke_value_t* row,
                                           rocke_value_t* col)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* col_chunk;
    rocke_value_t* row_ph;
    rocke_value_t* chunk;
    rocke_value_t* chunk_col;
    rocke_value_t* in_chunk;
    if(!s->swizzle)
    {
        return col;
    }
    col_chunk = rocke_b_div(b, col, rocke_b_const_i32(b, 8));
    row_ph = rocke_b_mod(b, row, rocke_b_const_i32(b, s->chunks_per_row));
    chunk = rocke_b_xor(b, col_chunk, row_ph);
    chunk_col = rocke_b_mul(b, chunk, rocke_b_const_i32(b, 8));
    in_chunk = rocke_b_mod(b, col, rocke_b_const_i32(b, 8));
    return rocke_b_add(b, chunk_col, in_chunk);
}

rocke_value_t* rocke_attn_tiled_kv_swz_src_linear(const rocke_attn_tiled_kv_swizzle_t* s,
                                                  rocke_value_t* linear_half)
{
    rocke_ir_builder_t* b = s->b;
    rocke_value_t* row;
    rocke_value_t* row_base;
    rocke_value_t* col;
    if(!s->swizzle)
    {
        return linear_half;
    }
    row = rocke_b_div(b, linear_half, rocke_b_const_i32(b, s->HD));
    row_base = rocke_b_mul(b, row, rocke_b_const_i32(b, s->HD));
    col = rocke_b_mod(b, linear_half, rocke_b_const_i32(b, s->HD));
    return rocke_b_add(b, row_base, rocke_attn_tiled_kv_swz_col(s, row, col));
}

/* ------------------------------------------------- 16x16 C row (C16_DIST) */

const rocke_tile_distribution_t* rocke_attn_tiled_make_c16_dist(rocke_ir_builder_t* b)
{
    const rocke_mfma_atom_t* c_atom;
    const rocke_tile_distribution_encoding_t* enc;

    c_atom = rocke_mfma_atom("f16", 16, 16, 16);
    if(c_atom == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "attn_tiled_3d: no f16 16x16x16 MFMA atom");
        return NULL;
    }
    enc = rocke_make_c_warp_dstr_encoding(b, c_atom);
    if(enc == NULL)
    {
        return NULL;
    }
    return rocke_make_static_tile_distribution(b, enc);
}

rocke_value_t* rocke_attn_tiled_mfma_16x16_c_row(rocke_ir_builder_t* b,
                                                 const rocke_tile_distribution_t* c16_dist,
                                                 rocke_value_t* lane,
                                                 int reg)
{
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2] = {NULL, NULL};

    if(!(0 <= reg && reg < 4))
    {
        /* Raised as a ckc::Error; the public entry boundary records it on b. */
        ckc::raise_status(ROCKE_ERR_VALUE,
                          ckc::format_error("mfma_16x16 reg must be 0..3, got %d", reg).c_str());
    }

    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));

    ys[0] = rocke_b_const_i32(b, 0);
    ys[1] = rocke_b_const_i32(b, reg);
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    if(!rocke_tile_distribution_calculate_x(b, c16_dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row (col discarded) */
}

/* ------------------------------------------------- paged block-table loads */

void rocke_attn_tiled_fast_paged_kv_blocks(rocke_ir_builder_t* b,
                                           rocke_value_t* kv_tile_idx,
                                           rocke_value_t* block_tables,
                                           rocke_value_t* seq_base,
                                           rocke_value_t* block_table_max_idx,
                                           rocke_value_t** out_block0,
                                           rocke_value_t** out_block1)
{
    rocke_value_t* logical_block0 = rocke_b_mul(b, kv_tile_idx, rocke_b_const_i32(b, 2));
    rocke_value_t* logical_block1 = rocke_b_add(b, logical_block0, rocke_b_const_i32(b, 1));
    rocke_value_t* idx0 = rocke_b_add(b, seq_base, logical_block0);
    rocke_value_t* idx1 = rocke_b_add(b, seq_base, logical_block1);
    /* Python evaluates masked_global_load args left-to-right: the cmp_lt mask is
     * created BEFORE the const(0) default. C arg-eval order is unspecified, so
     * bind the mask + default in source order to keep the value numbering. */
    rocke_value_t* mask0 = rocke_b_cmp_lt(b, idx0, block_table_max_idx);
    rocke_value_t* def0 = rocke_b_const_i32(b, 0);
    rocke_value_t* block0
        = rocke_b_masked_global_load(b, block_tables, idx0, mask0, def0, rocke_i32(), 4);
    rocke_value_t* mask1 = rocke_b_cmp_lt(b, idx1, block_table_max_idx);
    rocke_value_t* def1 = rocke_b_const_i32(b, 0);
    rocke_value_t* block1
        = rocke_b_masked_global_load(b, block_tables, idx1, mask1, def1, rocke_i32(), 4);
    if(out_block0)
        *out_block0 = rocke_b_to_sgpr_u32(b, block0);
    if(out_block1)
        *out_block1 = rocke_b_to_sgpr_u32(b, block1);
}

rocke_value_t* rocke_attn_tiled_load_page_id_window(rocke_ir_builder_t* b,
                                                    rocke_value_t* rel_page,
                                                    rocke_value_t* tid,
                                                    rocke_value_t* block_tables,
                                                    rocke_value_t* seq_base,
                                                    rocke_value_t* seq_pages,
                                                    rocke_value_t* seg_first_page)
{
    rocke_value_t* first = rocke_b_add(b, seg_first_page, rel_page);
    rocke_value_t* pg = rocke_b_add(b, first, tid);
    rocke_value_t* idx = rocke_b_add(b, seq_base, pg);
    rocke_value_t* live = rocke_b_cmp_lt(b, pg, seq_pages);
    rocke_value_t* other = rocke_b_const_i32(b, 0);
    return rocke_b_masked_global_load(b, block_tables, idx, live, other, rocke_i32(), 4);
}

/* ------------------------------------------------- fp8 LDS->LDS dequant */

void rocke_attn_tiled_fp8_dequant_lds_to_bf16(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                              rocke_value_t* src,
                                              rocke_value_t* dst,
                                              rocke_value_t* scale,
                                              rocke_value_t* buf_idx)
{
    rocke_ir_builder_t* b = ld->b;
    const int CHUNK = 8; /* fp8_dequant_elems_per_chunk */
    int total_chunks;
    int chunks_per_thread;
    int cols_per_row;
    rocke_value_t* bf16_buf;
    int c;

    if(b == NULL || b->status != ROCKE_OK)
    {
        return;
    }

    total_chunks = (ld->T * ld->HD) / CHUNK;
    if(ld->THREADS <= 0 || (total_chunks % ld->THREADS) != 0)
    {
        return; /* prologue assert: divisible by THREADS */
    }
    chunks_per_thread = total_chunks / ld->THREADS;
    cols_per_row = ld->HD / CHUNK;
    /* The Python bf16_buf destination selector is always 0 on this path (the
     * FP8 path single-buffers the working-dtype K_lds). */
    bf16_buf = rocke_b_const_i32(b, 0);

    for(c = 0; c < chunks_per_thread; ++c)
    {
        rocke_value_t* chunk_c = rocke_b_const_i32(b, c);
        rocke_value_t* chunk_thr = rocke_b_const_i32(b, ld->THREADS);
        rocke_value_t* chunk_id = rocke_b_add(b, rocke_b_mul(b, chunk_c, chunk_thr), ld->tid);
        rocke_value_t* row = rocke_b_div(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        /* Python: col = mul(mod(chunk_id, const(cols_per_row)), const(CHUNK));
         * the mod (and its const) are created before const(CHUNK). Sequence
         * explicitly so C arg-eval order does not reorder value creation. */
        rocke_value_t* col_mod = rocke_b_mod(b, chunk_id, rocke_b_const_i32(b, cols_per_row));
        rocke_value_t* col = rocke_b_mul(b, col_mod, rocke_b_const_i32(b, CHUNK));
        rocke_value_t* idx_load[3] = {buf_idx, row, col};
        rocke_value_t* fp8_vec = rocke_b_smem_load_vN(b, src, idx_load, 3, rocke_fp8e4m3(), CHUNK);
        rocke_value_t* dequanted[8];
        rocke_value_t* packed;
        rocke_value_t* idx_store[3];
        int i;

        for(i = 0; i < CHUNK; ++i)
        {
            rocke_value_t* fp8_v = rocke_b_vec_extract(b, fp8_vec, i);
            rocke_value_t* f32_v = rocke_b_fmul(b, rocke_b_cvt_fp8_to_f32(b, fp8_v), scale);
            dequanted[i] = rocke_b_cast_f32_to(b, f32_v, ld->dtype);
        }
        packed = rocke_b_vec_pack(b, dequanted, CHUNK, ld->dtype);
        idx_store[0] = bf16_buf;
        idx_store[1] = row;
        idx_store[2] = col;
        rocke_b_smem_store_vN(b, dst, idx_store, 3, packed, CHUNK);
    }
}
