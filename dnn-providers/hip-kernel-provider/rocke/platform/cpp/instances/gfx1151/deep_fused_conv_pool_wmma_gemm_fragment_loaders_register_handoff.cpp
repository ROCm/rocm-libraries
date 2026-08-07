// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_deep_fused_conv_pool_WMMA GEMM + fragment loaders + register
 * handoff.c -- one chunk of the C99 port of
 *   rocke/instances/gfx1151/deep_fused_conv_pool.py  (arch gfx1151).
 *
 * SCOPE (this translation unit):
 *   Fragment loaders   (Python lines 1382-1459, 2011-2149):
 *     _load_frag_iu8_from_lds / _pack_i4_codes_to_i32 /
 *     _load_frag_iu4_codes_from_lds / _load_frag_iu4_packed_from_lds /
 *     _load_conv0_a_frag_from_footprint{,_iu8,_iu8_static}.
 *   WMMA GEMMs         (Python lines 1462-2276):
 *     _emit_wmma_k_sched + _wmma_gemm_from_lds{,_int} +
 *     _wmma_gemm_conv0_direct{,_int} + _wmma_gemm_conv1_i4_from_lds +
 *     _wmma_gemm_conv1_i4_packed_from_lds + _wmma_gemm_conv1_i4_from_regs +
 *     _wmma_gemm_conv1_i8_from_regs.
 *   Register handoff   (Python lines 1644-1707):
 *     _fuse_c0_to_conv1_a_regs (permlanex16 transpose; int8 path).
 *
 * All accumulator / a-fragment producers write into the caller-supplied
 * out-arrays (out_accs[ROCKE_GFX1151_DFCP_MAX_ACCS], *out_count) per the internal
 * contract; the driver stores into ctx->accs0 / ctx->accs1 / ctx->a_frags.
 *
 * Byte-identical to the Python builder-call sequence: the C body emits the same
 * ops in the same order over ctx->b. Peer phases (staging / scatter / maxpool /
 * driver / quant primitives) live in sibling TUs and are reached only through
 * the internal header.
 */

#include "rocke/instance_gfx1151_deep_fused_conv_pool_internal.h"

#include "rocke/arch_target.h" /* rocke_mma_op_t fields + rocke_mma_op_*_layout */
#include "rocke/helper_rocke.helpers.epilogues.h" /* rocke_warp_grid_t + warp_*_off / mfmas_per_warp_* */
#include "rocke/helper_rocke.helpers.layouts.h" /* rocke_layout_map_coord (via arch_target.h) */
#include "rocke/helper_rocke.helpers.schedule.h" /* rocke_schedule_policy_t + ROCKE_SCHED_DS_READ/MFMA */
#include "rocke/instance_conv_implicit_gemm_internal.h" /* rocke_conv_emit_frag_smem_load (peer _emit_frag_smem_load) */

/* ------------------------------------------------------------------ *
 * Module-pinned geometry constants (Python module-level names).
 * ------------------------------------------------------------------ */
#define DFCP_WMMA ROCKE_GFX1151_DFCP_WMMA /* _WMMA      = 16 */
#define DFCP_K_PER_I32 ROCKE_GFX1151_DFCP_K_PER_I32 /* _K_PER_I32 = 4 */
#define DFCP_I4_PER_I32 ROCKE_GFX1151_DFCP_I4_PER_I32 /* _I4_PER_I32 = 8 */

/* ------------------------------------------------------------------ *
 * Small local conveniences (1:1 with `b.<op>(...)` shorthands).
 * ------------------------------------------------------------------ */

/* b.mma(op, a, b, c) -> target-neutral MMA over op.op_id, no scale operands. */
static rocke_value_t* dfcp_mma(rocke_ir_builder_t* b,
                               const rocke_mma_op_t* op,
                               rocke_value_t* a,
                               rocke_value_t* bb,
                               rocke_value_t* c)
{
    return rocke_b_mma(b, op->op_id, a, bb, c, NULL, 0);
}

/* a_map.coord(b, lane, 0) -> (row, k). Mirrors LayoutMap.coord with slot 0.
 * Returns the first coordinate (row/col) and optionally the second (k). */
static void dfcp_layout_coord0(rocke_ir_builder_t* b,
                               const rocke_layout_map_t* m,
                               rocke_value_t* lane,
                               rocke_value_t** out0,
                               rocke_value_t** out1)
{
    rocke_value_t* c0 = NULL;
    rocke_value_t* c1 = NULL;
    rocke_layout_map_coord(m, b, lane, 0, &c0, &c1);
    if(out0)
        *out0 = c0;
    if(out1)
        *out1 = c1;
}

/* ===================================================================== *
 *  _emit_wmma_k_sched  (Python lines 1949-1956)
 * ===================================================================== */
void rocke_gfx1151_dfcp_emit_wmma_k_sched(rocke_ir_builder_t* b,
                                          const void* policy,
                                          int n_ds,
                                          int n_mma)
{
    const rocke_schedule_policy_t* p = (const rocke_schedule_policy_t*)policy;
    /* if policy is None or not policy.emit_hints: return */
    if(p == NULL || !p->emit_hints)
        return;
    /* b.sched_group_barrier(DS_READ, int(n_ds), 0)
     * b.sched_group_barrier(MFMA,    int(n_mma), 0) */
    rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_READ, n_ds, 0);
    rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, n_mma, 0);
}

/* ===================================================================== *
 *  FRAGMENT LOADERS  (Python lines 1382-1459, 2011-2149)
 * ===================================================================== */

/* _load_frag_iu8_from_lds (Python 1382-1397). */
rocke_value_t* rocke_gfx1151_dfcp_load_frag_iu8_from_lds(rocke_ir_builder_t* b,
                                                         rocke_value_t* smem,
                                                         rocke_value_t* frag_rc,
                                                         rocke_value_t* atom_off,
                                                         rocke_value_t* k_tile_base)
{
    /* row = b.add(atom_off, frag_rc)
     * raw = b.smem_load_vN(smem, row, k_tile_base, dtype=I8, n=_WMMA)  # <16 x i8>
     * return b.bitcast(raw, VectorType(I32, _K_PER_I32)) */
    rocke_value_t* row = rocke_b_add(b, atom_off, frag_rc);
    rocke_value_t* idx[2] = {row, k_tile_base};
    rocke_value_t* raw = rocke_b_smem_load_vN(b, smem, idx, 2, rocke_i8(), DFCP_WMMA);
    return rocke_b_bitcast(b, raw, rocke_vector_type(b, rocke_i32(), DFCP_K_PER_I32));
}

/* _pack_i4_codes_to_i32 (Python 1400-1413). */
rocke_value_t* rocke_gfx1151_dfcp_pack_i4_codes_to_i32(rocke_ir_builder_t* b, rocke_value_t* codes)
{
    /* word = b.const_i32(0); mask = b.const_i32(0xF)
     * for i in range(_I4_PER_I32):
     *     nib = b.land(b.sext(b.vec_extract(codes, i), I32), mask)
     *     if i: nib = b.shl(nib, b.const_i32(4*i))
     *     word = b.lor(word, nib)
     * return word */
    rocke_value_t* word = rocke_b_const_i32(b, 0);
    rocke_value_t* mask = rocke_b_const_i32(b, 0xF);
    for(int i = 0; i < DFCP_I4_PER_I32; ++i)
    {
        rocke_value_t* nib
            = rocke_b_land(b, rocke_b_sext(b, rocke_b_vec_extract(b, codes, i), rocke_i32()), mask);
        if(i)
            nib = rocke_b_shl(b, nib, rocke_b_const_i32(b, 4 * i));
        word = rocke_b_lor(b, word, nib);
    }
    return word;
}

/* _load_frag_iu4_codes_from_lds (Python 1416-1440). */
rocke_value_t* rocke_gfx1151_dfcp_load_frag_iu4_codes_from_lds(rocke_ir_builder_t* b,
                                                               rocke_value_t* smem,
                                                               rocke_value_t* frag_rc,
                                                               rocke_value_t* atom_off,
                                                               rocke_value_t* k_tile_base)
{
    /* row = b.add(atom_off, frag_rc)
     * for slot in range(2):
     *     raw = b.smem_load_vN(smem, row, b.add(k_tile_base, slot*_I4_PER_I32),
     *                          dtype=I8, n=_I4_PER_I32)
     *     words.append(_pack_i4_codes_to_i32(b, raw))
     * return b.vec_pack(words, I32) */
    rocke_value_t* row = rocke_b_add(b, atom_off, frag_rc);
    rocke_value_t* words[2];
    for(int slot = 0; slot < 2; ++slot)
    {
        rocke_value_t* col
            = rocke_b_add(b, k_tile_base, rocke_b_const_i32(b, slot * DFCP_I4_PER_I32));
        rocke_value_t* idx[2] = {row, col};
        rocke_value_t* raw = rocke_b_smem_load_vN(b, smem, idx, 2, rocke_i8(), DFCP_I4_PER_I32);
        words[slot] = rocke_gfx1151_dfcp_pack_i4_codes_to_i32(b, raw);
    }
    return rocke_b_vec_pack(b, words, 2, rocke_i32());
}

/* _load_frag_iu4_packed_from_lds (Python 1443-1459). */
rocke_value_t* rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(rocke_ir_builder_t* b,
                                                                rocke_value_t* smem,
                                                                rocke_value_t* frag_rc,
                                                                rocke_value_t* atom_off,
                                                                rocke_value_t* k_tile_base)
{
    /* row = b.add(atom_off, frag_rc)
     * byte_base = b.div(k_tile_base, b.const_i32(2))
     * raw = b.smem_load_vN(smem, row, byte_base, dtype=I8, n=8)  # <8 x i8>
     * return b.bitcast(raw, VectorType(I32, 2)) */
    rocke_value_t* row = rocke_b_add(b, atom_off, frag_rc);
    rocke_value_t* byte_base = rocke_b_div(b, k_tile_base, rocke_b_const_i32(b, 2));
    rocke_value_t* idx[2] = {row, byte_base};
    rocke_value_t* raw = rocke_b_smem_load_vN(b, smem, idx, 2, rocke_i8(), 8);
    return rocke_b_bitcast(b, raw, rocke_vector_type(b, rocke_i32(), 2));
}

/* _load_conv0_a_frag_from_footprint (Python 2011-2060). */
rocke_value_t*
    rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                        rocke_value_t* inp_smem,
                                                        rocke_value_t* m_row,
                                                        rocke_value_t* k_base,
                                                        int frag_len)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_conv_problem_t* c = ctx->c;

    rocke_value_t* c_ctw = rocke_b_const_i32(b, rocke_gfx1151_dfcp_conv_tile_w(ctx->spec));
    rocke_value_t* c_sc = rocke_b_const_i32(b, c->X * c->C);
    rocke_value_t* c_fw = rocke_b_const_i32(b, rocke_gfx1151_dfcp_foot_w(ctx->spec));
    rocke_value_t* c_kg = rocke_b_const_i32(b, rocke_conv_problem_k_gemm(c));
    rocke_value_t* zero_h = rocke_b_trunc_f32_to_f16(b, rocke_b_const_f32(b, 0.0));

    rocke_value_t* local_oh = rocke_b_div(b, m_row, c_ctw);
    rocke_value_t* local_ow = rocke_b_mod(b, m_row, c_ctw);
    rocke_value_t* oh_base = rocke_b_mul(b, local_oh, rocke_b_const_i32(b, c->sH));
    rocke_value_t* ow_base = rocke_b_mul(b, local_ow, rocke_b_const_i32(b, c->sW));

    bool is_pow2_c = (c->C > 0) && ((c->C & (c->C - 1)) == 0);
    int c_log2 = 0;
    if(is_pow2_c)
    {
        /* (c.C - 1).bit_length() */
        int v = c->C - 1;
        while(v)
        {
            ++c_log2;
            v >>= 1;
        }
    }

    /* elems = [] ; for i in range(frag_len): ... ; return b.vec_pack(elems, elems[0].type) */
    rocke_value_t* elems[ROCKE_GFX1151_DFCP_MAX_ACCS * 4];
    for(int i = 0; i < frag_len; ++i)
    {
        rocke_value_t* kg = rocke_b_add(b, k_base, rocke_b_const_i32(b, i));
        rocke_value_t* kg_ok = rocke_b_cmp_lt(b, kg, c_kg);
        rocke_value_t* r = rocke_b_div(b, kg, c_sc);
        rocke_value_t* rem = rocke_b_mod(b, kg, c_sc);
        rocke_value_t* s_col;
        rocke_value_t* ci;
        if(is_pow2_c)
        {
            s_col = rocke_b_lshr(b, rem, rocke_b_const_i32(b, c_log2));
            ci = rocke_b_land(b, rem, rocke_b_const_i32(b, c->C - 1));
        }
        else
        {
            s_col = rocke_b_div(b, rem, rocke_b_const_i32(b, c->C));
            ci = rocke_b_mod(b, rem, rocke_b_const_i32(b, c->C));
        }
        rocke_value_t* fr = rocke_b_add(b, oh_base, rocke_b_mul(b, r, rocke_b_const_i32(b, c->dH)));
        rocke_value_t* fw
            = rocke_b_add(b, ow_base, rocke_b_mul(b, s_col, rocke_b_const_i32(b, c->dW)));
        rocke_value_t* foot_row = rocke_b_add(b, rocke_b_mul(b, fr, c_fw), fw);
        rocke_value_t* fidx[2] = {foot_row, ci};
        rocke_value_t* raw
            = rocke_b_vec_extract(b, rocke_b_smem_load_vN_f16(b, inp_smem, fidx, 2, 1), 0);
        elems[i] = rocke_b_select(b, kg_ok, raw, zero_h);
    }
    return rocke_b_vec_pack(b, elems, frag_len, rocke_f16());
}

/* _load_conv0_a_frag_from_footprint_iu8 (Python 2063-2109). */
rocke_value_t*
    rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint_iu8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                            rocke_value_t* inp_smem,
                                                            rocke_value_t* m_row,
                                                            rocke_value_t* k_base)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_conv_problem_t* c = ctx->c;

    rocke_value_t* c_ctw = rocke_b_const_i32(b, rocke_gfx1151_dfcp_conv_tile_w(ctx->spec));
    rocke_value_t* c_sc = rocke_b_const_i32(b, c->X * c->C);
    rocke_value_t* c_fw = rocke_b_const_i32(b, rocke_gfx1151_dfcp_foot_w(ctx->spec));
    rocke_value_t* c_kg = rocke_b_const_i32(b, rocke_conv_problem_k_gemm(c));

    rocke_value_t* local_oh = rocke_b_div(b, m_row, c_ctw);
    rocke_value_t* local_ow = rocke_b_mod(b, m_row, c_ctw);
    rocke_value_t* oh_base = rocke_b_mul(b, local_oh, rocke_b_const_i32(b, c->sH));
    rocke_value_t* ow_base = rocke_b_mul(b, local_ow, rocke_b_const_i32(b, c->sW));

    bool is_pow2_c = (c->C > 0) && ((c->C & (c->C - 1)) == 0);
    int c_log2 = 0;
    if(is_pow2_c)
    {
        int v = c->C - 1;
        while(v)
        {
            ++c_log2;
            v >>= 1;
        }
    }
    rocke_value_t* zero_vec = rocke_b_zero_vec(b, rocke_i8(), DFCP_K_PER_I32);

    rocke_value_t* words[DFCP_K_PER_I32];
    for(int slot = 0; slot < DFCP_K_PER_I32; ++slot)
    {
        rocke_value_t* kg = rocke_b_add(b, k_base, rocke_b_const_i32(b, slot * DFCP_K_PER_I32));
        rocke_value_t* kg_ok = rocke_b_cmp_lt(b, kg, c_kg);
        rocke_value_t* r = rocke_b_div(b, kg, c_sc);
        rocke_value_t* rem = rocke_b_mod(b, kg, c_sc);
        rocke_value_t* s_col;
        rocke_value_t* ci;
        if(is_pow2_c)
        {
            s_col = rocke_b_lshr(b, rem, rocke_b_const_i32(b, c_log2));
            ci = rocke_b_land(b, rem, rocke_b_const_i32(b, c->C - 1));
        }
        else
        {
            s_col = rocke_b_div(b, rem, rocke_b_const_i32(b, c->C));
            ci = rocke_b_mod(b, rem, rocke_b_const_i32(b, c->C));
        }
        rocke_value_t* fr = rocke_b_add(b, oh_base, rocke_b_mul(b, r, rocke_b_const_i32(b, c->dH)));
        rocke_value_t* fw
            = rocke_b_add(b, ow_base, rocke_b_mul(b, s_col, rocke_b_const_i32(b, c->dW)));
        rocke_value_t* foot_row = rocke_b_add(b, rocke_b_mul(b, fr, c_fw), fw);
        rocke_value_t* idx[2] = {foot_row, ci};
        rocke_value_t* raw = rocke_b_smem_load_vN(b, inp_smem, idx, 2, rocke_i8(), DFCP_K_PER_I32);
        rocke_value_t* code = rocke_b_vector_select(b, kg_ok, raw, zero_vec);
        words[slot] = rocke_b_bitcast(b, code, rocke_i32());
    }
    return rocke_b_vec_pack(b, words, DFCP_K_PER_I32, rocke_i32());
}

/* _load_conv0_a_frag_from_footprint_iu8_static (Python 2112-2149). */
rocke_value_t* rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint_iu8_static(
    rocke_gfx1151_dfcp_build_ctx_t* ctx, rocke_value_t* inp_smem, rocke_value_t* m_row, int kk)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_conv_problem_t* c = ctx->c;

    rocke_value_t* c_ctw = rocke_b_const_i32(b, rocke_gfx1151_dfcp_conv_tile_w(ctx->spec));
    rocke_value_t* c_fw = rocke_b_const_i32(b, rocke_gfx1151_dfcp_foot_w(ctx->spec));
    rocke_value_t* local_oh = rocke_b_div(b, m_row, c_ctw);
    rocke_value_t* local_ow = rocke_b_mod(b, m_row, c_ctw);
    rocke_value_t* oh_base = rocke_b_mul(b, local_oh, rocke_b_const_i32(b, c->sH));
    rocke_value_t* ow_base = rocke_b_mul(b, local_ow, rocke_b_const_i32(b, c->sW));

    rocke_value_t* words[DFCP_K_PER_I32];
    for(int slot = 0; slot < DFCP_K_PER_I32; ++slot)
    {
        int kg0 = kk * DFCP_WMMA + slot * DFCP_K_PER_I32;
        if(kg0 >= rocke_conv_problem_k_gemm(c))
        {
            words[slot] = rocke_b_const_i32(b, 0);
            continue;
        }
        int r = kg0 / (c->X * c->C);
        int rem = kg0 % (c->X * c->C);
        int s_col = rem / c->C;
        int ci = rem % c->C;
        rocke_value_t* fr = rocke_b_add(b, oh_base, rocke_b_const_i32(b, r * c->dH));
        rocke_value_t* fw = rocke_b_add(b, ow_base, rocke_b_const_i32(b, s_col * c->dW));
        rocke_value_t* foot_row = rocke_b_add(b, rocke_b_mul(b, fr, c_fw), fw);
        rocke_value_t* idx[2] = {foot_row, rocke_b_const_i32(b, ci)};
        rocke_value_t* raw = rocke_b_smem_load_vN(b, inp_smem, idx, 2, rocke_i8(), DFCP_K_PER_I32);
        words[slot] = rocke_b_bitcast(b, raw, rocke_i32());
    }
    return rocke_b_vec_pack(b, words, DFCP_K_PER_I32, rocke_i32());
}

/* ===================================================================== *
 *  WMMA GEMMS  (Python lines 1462-2276)
 * ===================================================================== */

/* _wmma_gemm_from_lds (Python 1959-2008). */
void rocke_gfx1151_dfcp_wmma_gemm_from_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                           const rocke_mma_op_t* op,
                                           rocke_value_t* a_smem,
                                           rocke_value_t* b_smem,
                                           int k_total,
                                           const void* policy,
                                           rocke_value_t** out_accs,
                                           size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = k_total / DFCP_WMMA;
    const rocke_layout_map_t* a_map = rocke_mma_op_a_layout(op, b);
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t *a_row, *a_k, *b_k, *b_col;
    dfcp_layout_coord0(b, a_map, grid->lane, &a_row, &a_k);
    dfcp_layout_coord0(b, b_map, grid->lane, &b_k, &b_col);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int ds_per_frag = (op->a_frag_len + 7) / 8;
    int n_ds = ds_per_frag * (mfmas_m + mfmas_n);

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec_f32(b, op->c_frag_len);

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
        rocke_value_t* a_rows[ROCKE_GFX1151_DFCP_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        for(int mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
            a_rows[mi] = rocke_conv_emit_frag_smem_load(
                b, a_smem, a_row, a_k, atom_row, k_tile_base, op->a_frag_len);
        }
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
            b_cols[ni] = rocke_conv_emit_frag_smem_load(
                b, b_smem, b_col, b_k, atom_row, k_tile_base, op->b_frag_len);
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                out_accs[flat] = dfcp_mma(b, op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n);
    }
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_from_lds_int (Python 1462-1507). */
void rocke_gfx1151_dfcp_wmma_gemm_from_lds_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                               const rocke_mma_op_t* op,
                                               rocke_value_t* a_smem,
                                               rocke_value_t* b_smem,
                                               int k_total,
                                               const void* policy,
                                               rocke_value_t** out_accs,
                                               size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = k_total / DFCP_WMMA;
    const rocke_layout_map_t* a_map = rocke_mma_op_a_layout(op, b);
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t *a_row, *b_col;
    dfcp_layout_coord0(b, a_map, grid->lane, &a_row, NULL); /* a_row = lane % 16 */
    dfcp_layout_coord0(b, b_map, grid->lane, NULL, &b_col); /* b_col = lane % 16 */
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = mfmas_m + mfmas_n; /* one ds_read_b128 per fragment */

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec(b, rocke_i32(), op->c_frag_len);

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
        rocke_value_t* a_rows[ROCKE_GFX1151_DFCP_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        for(int mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* atom_off
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
            a_rows[mi] = rocke_gfx1151_dfcp_load_frag_iu8_from_lds(
                b, a_smem, a_row, atom_off, k_tile_base);
        }
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* atom_off
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
            b_cols[ni] = rocke_gfx1151_dfcp_load_frag_iu8_from_lds(
                b, b_smem, b_col, atom_off, k_tile_base);
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                out_accs[flat] = dfcp_mma(b, op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n);
    }
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_conv0_direct (Python 2152-2204). */
void rocke_gfx1151_dfcp_wmma_gemm_conv0_direct(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                               const rocke_mma_op_t* op,
                                               rocke_value_t* inp_smem,
                                               rocke_value_t* w0_smem,
                                               const void* policy,
                                               rocke_value_t** out_accs,
                                               size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = rocke_gfx1151_dfcp_kpad(ctx->spec) / DFCP_WMMA;
    const rocke_layout_map_t* a_map = rocke_mma_op_a_layout(op, b);
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t *a_row, *a_k, *b_k, *b_col;
    dfcp_layout_coord0(b, a_map, grid->lane, &a_row, &a_k);
    dfcp_layout_coord0(b, b_map, grid->lane, &b_k, &b_col);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = op->a_frag_len * mfmas_m + ((op->b_frag_len + 7) / 8) * mfmas_n;

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec_f32(b, op->c_frag_len);

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
        rocke_value_t* k_base = rocke_b_add(b, k_tile_base, a_k);
        rocke_value_t* a_rows[ROCKE_GFX1151_DFCP_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        for(int mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
            rocke_value_t* m_row = rocke_b_add(b, atom_row, a_row);
            a_rows[mi] = rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint(
                ctx, inp_smem, m_row, k_base, op->a_frag_len);
        }
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
            b_cols[ni] = rocke_conv_emit_frag_smem_load(
                b, w0_smem, b_col, b_k, atom_row, k_tile_base, op->b_frag_len);
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                out_accs[flat] = dfcp_mma(b, op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n);
    }
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_conv0_direct_int (Python 2207-2276). */
void rocke_gfx1151_dfcp_wmma_gemm_conv0_direct_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                   const rocke_mma_op_t* op,
                                                   rocke_value_t* inp_smem,
                                                   rocke_value_t* w0_smem,
                                                   const void* policy,
                                                   bool reorient,
                                                   rocke_value_t** out_accs,
                                                   size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = rocke_gfx1151_dfcp_kpad(ctx->spec) / DFCP_WMMA;
    const rocke_layout_map_t* a_map = rocke_mma_op_a_layout(op, b);
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t *a_row, *a_k, *b_col;
    dfcp_layout_coord0(b, a_map, grid->lane, &a_row, &a_k);
    dfcp_layout_coord0(b, b_map, grid->lane, NULL, &b_col);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = DFCP_K_PER_I32 * mfmas_m + mfmas_n;

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec(b, rocke_i32(), op->c_frag_len);

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
        rocke_value_t* k_base = rocke_b_add(b, k_tile_base, a_k);
        rocke_value_t* a_rows[ROCKE_GFX1151_DFCP_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        for(int mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* atom_row
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
            rocke_value_t* m_row = rocke_b_add(b, atom_row, a_row);
            if(ctx->spec->static_direct_kmap)
                a_rows[mi] = rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint_iu8_static(
                    ctx, inp_smem, m_row, kk);
            else
                a_rows[mi] = rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint_iu8(
                    ctx, inp_smem, m_row, k_base);
        }
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* atom_off
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
            b_cols[ni] = rocke_gfx1151_dfcp_load_frag_iu8_from_lds(
                b, w0_smem, b_col, atom_off, k_tile_base);
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                if(reorient)
                    /* Swap operands: W0 -> A (row=k0), footprint -> B (col=m). */
                    out_accs[flat] = dfcp_mma(b, op, b_cols[ni], a_rows[mi], out_accs[flat]);
                else
                    out_accs[flat] = dfcp_mma(b, op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n);
    }
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_conv1_i4_from_lds (Python 1510-1596). */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                    const rocke_mma_op_t* op,
                                                    rocke_value_t* c0_smem,
                                                    rocke_value_t* w1_smem,
                                                    int k_total,
                                                    const void* policy,
                                                    bool prefetch_k,
                                                    bool sched_fuse,
                                                    rocke_value_t** out_accs,
                                                    size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = k_total / DFCP_WMMA;
    const rocke_layout_map_t* a_map = rocke_mma_op_a_layout(op, b);
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t *a_row, *b_col;
    dfcp_layout_coord0(b, a_map, grid->lane, &a_row, NULL);
    dfcp_layout_coord0(b, b_map, grid->lane, NULL, &b_col);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = 2 * mfmas_m + mfmas_n; /* A: two 8-byte loads + pack; B: one 8-byte load */

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec(b, rocke_i32(), op->c_frag_len);

    const void* per_step = sched_fuse ? NULL : policy;

    /* Prefetch hoists all k-step A/B frags before any MMA; both branches emit the
     * same MMA / acc ops in the same order, only the load placement differs. */
    rocke_value_t* a_all[64 /*k_atoms*/][ROCKE_GFX1151_DFCP_MAX_ACCS];
    rocke_value_t* b_all[64 /*k_atoms*/][ROCKE_GFX1151_DFCP_MAX_ACCS];
    if(prefetch_k)
    {
        for(int kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            for(int mi = 0; mi < mfmas_m; ++mi)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
                a_all[kk][mi] = rocke_gfx1151_dfcp_load_frag_iu4_codes_from_lds(
                    b, c0_smem, a_row, atom_off, k_tile_base);
            }
        }
        for(int kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
                b_all[kk][ni] = rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(
                    b, w1_smem, b_col, atom_off, k_tile_base);
            }
        }
        for(int kk = 0; kk < k_atoms; ++kk)
        {
            int flat = 0;
            for(int mi = 0; mi < mfmas_m; ++mi)
                for(int ni = 0; ni < mfmas_n; ++ni)
                {
                    out_accs[flat] = dfcp_mma(b, op, a_all[kk][mi], b_all[kk][ni], out_accs[flat]);
                    ++flat;
                }
            rocke_gfx1151_dfcp_emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n);
        }
    }
    else
    {
        for(int kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            rocke_value_t* a_rows[ROCKE_GFX1151_DFCP_MAX_ACCS];
            rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
            for(int mi = 0; mi < mfmas_m; ++mi)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
                a_rows[mi] = rocke_gfx1151_dfcp_load_frag_iu4_codes_from_lds(
                    b, c0_smem, a_row, atom_off, k_tile_base);
            }
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
                b_cols[ni] = rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(
                    b, w1_smem, b_col, atom_off, k_tile_base);
            }
            int flat = 0;
            for(int mi = 0; mi < mfmas_m; ++mi)
                for(int ni = 0; ni < mfmas_n; ++ni)
                {
                    out_accs[flat] = dfcp_mma(b, op, a_rows[mi], b_cols[ni], out_accs[flat]);
                    ++flat;
                }
            rocke_gfx1151_dfcp_emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n);
        }
    }

    if(sched_fuse)
        rocke_gfx1151_dfcp_emit_wmma_k_sched(
            b, policy, n_ds * k_atoms, mfmas_m * mfmas_n * k_atoms);
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_conv1_i4_packed_from_lds (Python 1599-1641). */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_packed_from_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                           const rocke_mma_op_t* op,
                                                           rocke_value_t* c0_smem,
                                                           rocke_value_t* w1_smem,
                                                           int k_total,
                                                           const void* policy,
                                                           rocke_value_t** out_accs,
                                                           size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = k_total / DFCP_WMMA;
    const rocke_layout_map_t* a_map = rocke_mma_op_a_layout(op, b);
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t *a_row, *b_col;
    dfcp_layout_coord0(b, a_map, grid->lane, &a_row, NULL);
    dfcp_layout_coord0(b, b_map, grid->lane, NULL, &b_col);
    rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = mfmas_m + mfmas_n;

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec(b, rocke_i32(), op->c_frag_len);

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
        rocke_value_t* a_rows[ROCKE_GFX1151_DFCP_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        for(int mi = 0; mi < mfmas_m; ++mi)
        {
            rocke_value_t* atom_off
                = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * DFCP_WMMA));
            a_rows[mi] = rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(
                b, c0_smem, a_row, atom_off, k_tile_base);
        }
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* atom_off
                = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
            b_cols[ni] = rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(
                b, w1_smem, b_col, atom_off, k_tile_base);
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                out_accs[flat] = dfcp_mma(b, op, a_rows[mi], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n);
    }
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_conv1_i4_from_regs (Python 1710-1765). */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_regs(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                     const rocke_mma_op_t* op,
                                                     rocke_value_t* const* a_frags,
                                                     int a_rows,
                                                     int a_cols,
                                                     rocke_value_t* w1_smem,
                                                     int k_total,
                                                     const void* policy,
                                                     bool prefetch_k,
                                                     bool sched_fuse,
                                                     rocke_value_t** out_accs,
                                                     size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = k_total / DFCP_WMMA;
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t* b_col;
    dfcp_layout_coord0(b, b_map, grid->lane, NULL, &b_col);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = mfmas_n; /* one 8-byte W1 load per n-atom per k-step */
    (void)a_cols;
    (void)a_rows;

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec(b, rocke_i32(), op->c_frag_len);

    const void* per_step = sched_fuse ? NULL : policy;

    rocke_value_t* b_all[64 /*k_atoms*/][ROCKE_GFX1151_DFCP_MAX_ACCS];
    if(prefetch_k)
    {
        for(int kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
                b_all[kk][ni] = rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(
                    b, w1_smem, b_col, atom_off, k_tile_base);
            }
        }
    }

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        if(prefetch_k)
        {
            for(int ni = 0; ni < mfmas_n; ++ni)
                b_cols[ni] = b_all[kk][ni];
        }
        else
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
                b_cols[ni] = rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(
                    b, w1_smem, b_col, atom_off, k_tile_base);
            }
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                /* a_frags[mi][kk] -- row-major flatten (mi*a_cols + kk). */
                out_accs[flat]
                    = dfcp_mma(b, op, a_frags[mi * a_cols + kk], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n);
    }

    if(sched_fuse)
        rocke_gfx1151_dfcp_emit_wmma_k_sched(
            b, policy, n_ds * k_atoms, mfmas_m * mfmas_n * k_atoms);
    *out_count = (size_t)n_acc;
}

/* _wmma_gemm_conv1_i8_from_regs (Python 1768-1818). */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i8_from_regs(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                     const rocke_mma_op_t* op,
                                                     rocke_value_t* const* a_frags,
                                                     int a_rows,
                                                     int a_cols,
                                                     rocke_value_t* w1_smem,
                                                     int k_total,
                                                     const void* policy,
                                                     bool prefetch_k,
                                                     bool sched_fuse,
                                                     rocke_value_t** out_accs,
                                                     size_t* out_count)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    int k_atoms = k_total / DFCP_WMMA;
    const rocke_layout_map_t* b_map = rocke_mma_op_b_layout(op, b);
    rocke_value_t* b_col;
    dfcp_layout_coord0(b, b_map, grid->lane, NULL, &b_col);
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    int n_ds = mfmas_n; /* one 16-byte W1 load (ds_read_b128) per n-atom per k-step */
    (void)a_cols;
    (void)a_rows;

    int n_acc = mfmas_m * mfmas_n;
    for(int i = 0; i < n_acc; ++i)
        out_accs[i] = rocke_b_zero_vec(b, rocke_i32(), op->c_frag_len);

    const void* per_step = sched_fuse ? NULL : policy;

    rocke_value_t* b_all[64 /*k_atoms*/][ROCKE_GFX1151_DFCP_MAX_ACCS];
    if(prefetch_k)
    {
        for(int kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
                b_all[kk][ni] = rocke_gfx1151_dfcp_load_frag_iu8_from_lds(
                    b, w1_smem, b_col, atom_off, k_tile_base);
            }
        }
    }

    for(int kk = 0; kk < k_atoms; ++kk)
    {
        rocke_value_t* b_cols[ROCKE_GFX1151_DFCP_MAX_ACCS];
        if(prefetch_k)
        {
            for(int ni = 0; ni < mfmas_n; ++ni)
                b_cols[ni] = b_all[kk][ni];
        }
        else
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * DFCP_WMMA);
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* atom_off
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * DFCP_WMMA));
                b_cols[ni] = rocke_gfx1151_dfcp_load_frag_iu8_from_lds(
                    b, w1_smem, b_col, atom_off, k_tile_base);
            }
        }
        int flat = 0;
        for(int mi = 0; mi < mfmas_m; ++mi)
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                out_accs[flat]
                    = dfcp_mma(b, op, a_frags[mi * a_cols + kk], b_cols[ni], out_accs[flat]);
                ++flat;
            }
        rocke_gfx1151_dfcp_emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n);
    }

    if(sched_fuse)
        rocke_gfx1151_dfcp_emit_wmma_k_sched(
            b, policy, n_ds * k_atoms, mfmas_m * mfmas_n * k_atoms);
    *out_count = (size_t)n_acc;
}

/* ===================================================================== *
 *  CONV0 -> CONV1 HANDOFF  _fuse_c0_to_conv1_a_regs (Python 1644-1707)
 * ===================================================================== */
void rocke_gfx1151_dfcp_fuse_c0_to_conv1_a_regs(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                const rocke_mma_op_t* op0,
                                                rocke_value_t* const* accs0,
                                                size_t num_accs0,
                                                int code_fn,
                                                bool int8,
                                                rocke_value_t** out_a_frags,
                                                size_t* out_count,
                                                int* out_rows,
                                                int* out_cols)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_warp_grid_t* grid = ctx->grid;
    int mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    int mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    (void)num_accs0;

    /* is_lo = b.cmp_lt(grid.lane, b.const_i32(16))  # half == 0 (wave32) */
    rocke_value_t* is_lo = rocke_b_cmp_lt(b, grid->lane, rocke_b_const_i32(b, 16));

    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        for(int kk = 0; kk < mfmas_n; ++kk)
        {
            rocke_value_t* acc = accs0[mi * mfmas_n + kk];
            /* codes = [code_fn(b.vec_extract(acc, s)) for s in range(op0.c_frag_len)] */
            rocke_value_t* codes[ROCKE_GFX1151_DFCP_MAX_ACCS];
            for(int s = 0; s < op0->c_frag_len; ++s)
                codes[s] = rocke_gfx1151_dfcp_apply_code_fn(
                    ctx, code_fn, rocke_b_vec_extract(b, acc, s));
            /* words = b.bitcast(b.vec_pack(codes, I8), VectorType(I32, 2)) */
            rocke_value_t* packed_i8 = rocke_b_vec_pack(b, codes, op0->c_frag_len, rocke_i8());
            rocke_value_t* words
                = rocke_b_bitcast(b, packed_i8, rocke_vector_type(b, rocke_i32(), 2));
            rocke_value_t* lo = rocke_b_vec_extract(b, words, 0);
            rocke_value_t* hi = rocke_b_vec_extract(b, words, 1);
            rocke_value_t* plo = rocke_b_permlanex16(b, lo);
            rocke_value_t* phi = rocke_b_permlanex16(b, hi);
            rocke_value_t* e_lo = rocke_b_select(b, is_lo, lo, plo);
            rocke_value_t* o_lo = rocke_b_select(b, is_lo, plo, lo);
            rocke_value_t* e_hi = rocke_b_select(b, is_lo, hi, phi);
            rocke_value_t* o_hi = rocke_b_select(b, is_lo, phi, hi);
            /* v_perm sources: 0..3 = arg b LSB-first, 4..7 = arg a LSB-first. */
            rocke_value_t* ord0 = rocke_b_byte_perm(b, o_lo, e_lo, 0x05010400);
            rocke_value_t* ord1 = rocke_b_byte_perm(b, o_lo, e_lo, 0x07030602);
            rocke_value_t* ord2 = rocke_b_byte_perm(b, o_hi, e_hi, 0x05010400);
            rocke_value_t* ord3 = rocke_b_byte_perm(b, o_hi, e_hi, 0x07030602);
            if(int8)
            {
                /* iu8 A-fragment is <4 x i32>; ordN already holds 4 contiguous k0
                 * byte codes for that slot -- hand them through directly. */
                rocke_value_t* four[4] = {ord0, ord1, ord2, ord3};
                out_a_frags[mi * mfmas_n + kk] = rocke_b_vec_pack(b, four, 4, rocke_i32());
                continue;
            }
            /* s0 = _pack_i4_codes_to_i32(b, bitcast(vec_pack([ord0,ord1], I32), <8 x i8>)) */
            rocke_value_t* p01[2] = {ord0, ord1};
            rocke_value_t* s0 = rocke_gfx1151_dfcp_pack_i4_codes_to_i32(
                b,
                rocke_b_bitcast(b,
                                rocke_b_vec_pack(b, p01, 2, rocke_i32()),
                                rocke_vector_type(b, rocke_i8(), 8)));
            rocke_value_t* p23[2] = {ord2, ord3};
            rocke_value_t* s1 = rocke_gfx1151_dfcp_pack_i4_codes_to_i32(
                b,
                rocke_b_bitcast(b,
                                rocke_b_vec_pack(b, p23, 2, rocke_i32()),
                                rocke_vector_type(b, rocke_i8(), 8)));
            rocke_value_t* s01[2] = {s0, s1};
            out_a_frags[mi * mfmas_n + kk] = rocke_b_vec_pack(b, s01, 2, rocke_i32());
        }
    }

    if(out_count)
        *out_count = (size_t)(mfmas_m * mfmas_n);
    if(out_rows)
        *out_rows = mfmas_m;
    if(out_cols)
        *out_cols = mfmas_n; /* conv0 N atoms over k0 == conv1 K atoms */
}
