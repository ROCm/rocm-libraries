// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/wmma_swapqk.py -- the transposed-QK WMMA
 * FMHA-forward inner body. See the header for the symbol map and the
 * byte-fidelity contract. Every rocke_b_* call sequence reproduces the
 * Python builder-call order, operands and compile-time constants op-for-op
 * so the emitted IR is byte-identical to the Python helper's emission.
 *
 * C argument-evaluation order is unspecified (unlike Python's guaranteed
 * left-to-right), so every nested multi-operand rocke_b_* expression from the
 * Python source is hoisted into ordered local temporaries below, matching the
 * value-creation order the Python interpreter would produce.
 */
#include "rocke/helper_rocke.helpers.wmma_swapqk.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* head_size in {64, 128} -> n_dk in {4, 8}. block_n in {32, 64} -> n_kv_sub in
 * {2, 4}. Generous fixed bounds (mirroring the ROCKE_ATTN_MAX_* pattern in
 * mfma_attention.cpp) avoid heap churn for this narrow, fixed-shape kernel. */
#define ROCKE_SWAPQK_MAX_DK 8
#define ROCKE_SWAPQK_MAX_NS 4
#define ROCKE_SWAPQK_MAX_CFRAG 8
#define ROCKE_SWAPQK_MAX_AFRAG 16
#define ROCKE_SWAPQK_MAX_NI32 8
#define ROCKE_SWAPQK_MAX_ALLS (ROCKE_SWAPQK_MAX_NS * ROCKE_SWAPQK_MAX_CFRAG)

extern "C" {
typedef rocke_value_t* (*rocke_swapqk_binop_fn)(rocke_ir_builder_t*,
                                                rocke_value_t*,
                                                rocke_value_t*);
}

/* ---------------------------------------------------------- permx16_f32 *
 *
 * Python:
 *     def permx16_f32(v):
 *         return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)
 */
static rocke_value_t* rocke_swapqk_permx16_f32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* vi = rocke_b_bitcast(b, v, rocke_i32());
    rocke_value_t* p = rocke_b_permlanex16(b, vi);
    return rocke_b_bitcast(b, p, rocke_f32());
}

/* ------------------------------------------------------------- _tree *
 *
 * Python: log-depth pairwise reduction, odd leftover carried unchanged.
 */
static rocke_value_t*
    rocke_swapqk_tree(rocke_ir_builder_t* b, rocke_value_t** vals, int n, rocke_swapqk_binop_fn op)
{
    rocke_value_t* buf[ROCKE_SWAPQK_MAX_ALLS];
    for(int i = 0; i < n; ++i)
        buf[i] = vals[i];
    while(n > 1)
    {
        int half = n / 2;
        for(int i = 0; i < half; ++i)
            buf[i] = op(b, buf[2 * i], buf[2 * i + 1]);
        if(n % 2)
            buf[half] = buf[n - 1];
        n = half + (n % 2);
    }
    return buf[0];
}

/* ------------------------------------------------------ load_wmma_fragment *
 *
 * Python:
 *     if role == "a": lmap = atom.a_layout(arch); lane_idx = lmap.coord(b, lane, 0)[0]
 *     elif role == "b": lmap = atom.b_layout(arch); lane_idx = lmap.coord(b, lane, 0)[1]
 *     idx = list(lead) + [lane_idx, b.const_i32(k_offset)]
 *     return window.load_vec(b, *idx, n=atom.a_per_lane)
 *
 * `role_is_a` selects the A-operand (K, row = lane%16) vs B-operand
 * (Q, col = lane%16) layout map; `lead0` is the single leading index (c0).
 */
static rocke_value_t* rocke_swapqk_load_operand(rocke_ir_builder_t* b,
                                                const rocke_tile_window_t* window,
                                                const rocke_mma_op_t* op,
                                                rocke_value_t* lane,
                                                rocke_value_t* lead0,
                                                bool role_is_a,
                                                int k_offset)
{
    const rocke_layout_map_t* lmap = role_is_a ? op->a_layout : op->b_layout;
    rocke_value_t* out0 = NULL;
    rocke_value_t* out1 = NULL;
    rocke_layout_map_coord(lmap, b, lane, 0, &out0, &out1);
    rocke_value_t* lane_idx = role_is_a ? out0 : out1;
    rocke_value_t* k_off = rocke_b_const_i32(b, k_offset);
    rocke_value_t* idx[3];
    idx[0] = lead0;
    idx[1] = lane_idx;
    idx[2] = k_off;
    return rocke_tile_window_load_vec(b, window, idx, 3, op->a_frag_len);
}

/* --------------------------------------------------------------- k_window *
 *
 * The leading view coordinate is the already-combined batch/head ELEMENT
 * offset. The second coordinate is a logical KV token.
 */
static void rocke_swapqk_k_window(rocke_tile_window_t* out,
                                  const rocke_tensor_view_t* K_view,
                                  rocke_value_t* k_base_elems,
                                  rocke_value_t* k_tile_base,
                                  rocke_value_t* c0,
                                  int hs)
{
    rocke_value_t* origin[3];
    origin[0] = k_base_elems;
    origin[1] = k_tile_base;
    origin[2] = c0;
    int lengths[3] = {1, 16, hs};
    rocke_make_tile_window(out, K_view, lengths, origin, 3);
}

/* --------------------------------------------------------- p_transpose_reg *
 *
 * CK PermuteWarpGemmCToA: C-dist P (c_frag f32 slots) -> operand fragment
 * (a_frag f16). Python (see wmma_swapqk.py::p_transpose_reg).
 */
static rocke_value_t* rocke_swapqk_p_transpose_reg(rocke_ir_builder_t* b,
                                                   rocke_value_t* const* ps,
                                                   int c_frag,
                                                   int a_frag,
                                                   rocke_value_t* c16,
                                                   rocke_value_t* sel0,
                                                   rocke_value_t* sel1)
{
    rocke_value_t* outs[ROCKE_SWAPQK_MAX_CFRAG];
    for(int m = 0; m < c_frag / 2; ++m)
    {
        rocke_value_t* lo_f16 = rocke_b_cast_f32_to(b, ps[2 * m], rocke_f16());
        rocke_value_t* lo_i16 = rocke_b_bitcast(b, lo_f16, rocke_i16());
        rocke_value_t* lo = rocke_b_zext(b, lo_i16, rocke_i32());
        rocke_value_t* hi_f16 = rocke_b_cast_f32_to(b, ps[2 * m + 1], rocke_f16());
        rocke_value_t* hi_i16 = rocke_b_bitcast(b, hi_f16, rocke_i16());
        rocke_value_t* hi = rocke_b_zext(b, hi_i16, rocke_i32());
        rocke_value_t* hi_shl = rocke_b_shl(b, hi, c16);
        rocke_value_t* v = rocke_b_lor(b, lo, hi_shl);
        rocke_value_t* w = rocke_b_permlanex16(b, v);
        outs[2 * m] = rocke_b_perm_b32(b, w, v, sel0); /* {kv 4m,   4m+1} */
        outs[2 * m + 1] = rocke_b_perm_b32(b, w, v, sel1); /* {kv 4m+2, 4m+3} */
    }
    rocke_value_t* packed = rocke_b_vec_pack(b, outs, c_frag, rocke_i32());
    const rocke_type_t* frag_ty = rocke_vector_type(b, rocke_f16(), a_frag);
    return rocke_b_vec_bitcast(b, packed, frag_ty);
}

/* ------------------------------------------------------------ _load_col *
 *
 * Gather V[kv=0..15, d_col] (per-lane d_col) via the buffer-descriptor D16
 * half-return load; row-major [B, S, H, D] V (no host transpose). Python:
 *     elem0 = b.add(b.add(kvh_off, b.mul(k_base, stride_v_token)), d_col)
 *     voff = b.mul(elem0, c2)
 *     v_a = b.undef_vec(F16, a_frag)
 *     for j in range(a_frag): v_a = b.vec_insert(v_a, b.buffer_load_f16_d16(v_rsrc, voff, soff_list[j]), j)
 */
static rocke_value_t* rocke_swapqk_load_col(rocke_ir_builder_t* b,
                                            rocke_value_t* v_rsrc,
                                            rocke_value_t* kvh_off,
                                            rocke_value_t* stride_v_token,
                                            rocke_value_t* c2,
                                            rocke_value_t* const* soff_list,
                                            int a_frag,
                                            rocke_value_t* k_base,
                                            rocke_value_t* d_col)
{
    rocke_value_t* k_mul = rocke_b_mul(b, k_base, stride_v_token);
    rocke_value_t* base_sum = rocke_b_add(b, kvh_off, k_mul);
    rocke_value_t* elem0 = rocke_b_add(b, base_sum, d_col);
    rocke_value_t* voff = rocke_b_mul(b, elem0, c2);
    rocke_value_t* v_a = rocke_b_undef_vec(b, rocke_f16(), a_frag);
    for(int j = 0; j < a_frag; ++j)
    {
        rocke_value_t* loaded = rocke_b_buffer_load_f16_d16(b, v_rsrc, voff, soff_list[j]);
        v_a = rocke_b_vec_insert(b, v_a, loaded, j);
    }
    return v_a;
}

/* ------------------------------------------------------ dual_gather_issue *
 *
 * Python:
 *     d_col = b.add(b.const_i32(d * 16), b.add(b.mul(b.div(lane, c16), c16), col))
 *     return _load_col(k_base, d_col)
 */
static rocke_value_t* rocke_swapqk_dual_gather_issue(rocke_ir_builder_t* b,
                                                     rocke_value_t* v_rsrc,
                                                     rocke_value_t* kvh_off,
                                                     rocke_value_t* stride_v_token,
                                                     rocke_value_t* c2,
                                                     rocke_value_t* const* soff_list,
                                                     int a_frag,
                                                     rocke_value_t* lane,
                                                     rocke_value_t* col,
                                                     rocke_value_t* c16,
                                                     rocke_value_t* k_base,
                                                     int d)
{
    rocke_value_t* d16 = rocke_b_const_i32(b, d * 16);
    rocke_value_t* half = rocke_b_div(b, lane, c16);
    rocke_value_t* half16 = rocke_b_mul(b, half, c16);
    rocke_value_t* half16_col = rocke_b_add(b, half16, col);
    rocke_value_t* d_col = rocke_b_add(b, d16, half16_col);
    return rocke_swapqk_load_col(
        b, v_rsrc, kvh_off, stride_v_token, c2, soff_list, a_frag, k_base, d_col);
}

/* ----------------------------------------------------- dual_gather_finish *
 *
 * permlanex16 + select broadcast each subtile into both lane-halves.
 */
static void rocke_swapqk_dual_gather_finish(rocke_ir_builder_t* b,
                                            rocke_value_t* loaded,
                                            int n_i32,
                                            int a_frag,
                                            rocke_value_t* lane_lt16,
                                            rocke_value_t** out_frag_d,
                                            rocke_value_t** out_frag_d1)
{
    const rocke_type_t* i32n = rocke_vector_type(b, rocke_i32(), n_i32);
    rocke_value_t* li = rocke_b_vec_bitcast(b, loaded, i32n);
    rocke_value_t* fd[ROCKE_SWAPQK_MAX_NI32];
    rocke_value_t* fd1[ROCKE_SWAPQK_MAX_NI32];
    for(int i = 0; i < n_i32; ++i)
    {
        rocke_value_t* e = rocke_b_vec_extract(b, li, i);
        rocke_value_t* p = rocke_b_permlanex16(b, e);
        fd[i] = rocke_b_select(b, lane_lt16, e, p);
        fd1[i] = rocke_b_select(b, lane_lt16, p, e);
    }
    const rocke_type_t* frag_ty = rocke_vector_type(b, rocke_f16(), a_frag);
    rocke_value_t* packed_d = rocke_b_vec_pack(b, fd, n_i32, rocke_i32());
    *out_frag_d = rocke_b_vec_bitcast(b, packed_d, frag_ty);
    rocke_value_t* packed_d1 = rocke_b_vec_pack(b, fd1, n_i32, rocke_i32());
    *out_frag_d1 = rocke_b_vec_bitcast(b, packed_d1, frag_ty);
}

/* -------------------------------------------------------- store epilogue *
 *
 * Python (store_wmma_acc, transform = lambda bld,val,...: bld.fmul(val, inv_l)):
 *     cmap = atom.c_layout(arch); c_off = b.const_i32(col_offset)
 *     for r in range(atom.c_per_lane):
 *         row, col = cmap.coord(b, lane, r)
 *         val = b.vec_extract(acc, r)
 *         val = transform(b, val, r, row, col)
 *         window.store_scalar(b, c0, row, b.add(c_off, col), value=b.cast_f32_to(val, F16), align=2)
 */
static void rocke_swapqk_store_col(rocke_ir_builder_t* b,
                                   const rocke_tile_window_t* owin,
                                   const rocke_mma_op_t* op,
                                   rocke_value_t* lane,
                                   rocke_value_t* acc,
                                   rocke_value_t* c0,
                                   int col_offset,
                                   rocke_value_t* inv_l)
{
    rocke_value_t* c_off = rocke_b_const_i32(b, col_offset);
    for(int r = 0; r < op->c_frag_len; ++r)
    {
        rocke_value_t* row = NULL;
        rocke_value_t* col = NULL;
        rocke_layout_map_coord(op->c_layout, b, lane, r, &row, &col);
        rocke_value_t* val = rocke_b_vec_extract(b, acc, r);
        rocke_value_t* rescaled = rocke_b_fmul(b, val, inv_l);
        rocke_value_t* store_col = rocke_b_add(b, c_off, col);
        rocke_value_t* store_val = rocke_b_cast_f32_to(b, rescaled, rocke_f16());
        rocke_value_t* idx[3];
        idx[0] = c0;
        idx[1] = row;
        idx[2] = store_col;
        rocke_tile_window_store_scalar(b, owin, idx, 3, store_val, 2 /* align */);
    }
}

/* ===================================================== public entry point */

rocke_status_t rocke_wmma_swapqk_fwd_inner_body(rocke_ir_builder_t* b,
                                                const rocke_mfma_attn_params_t* p,
                                                int block_n,
                                                int n_waves)
{
    if(p == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wmma_swapqk: params must be non-NULL");
        return ROCKE_ERR_VALUE;
    }
    const char* arch = (p->arch != NULL) ? p->arch : "gfx1151";
    if(strcmp(arch, "gfx1151") != 0)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "wmma_swapqk is a gfx1151 (RDNA3.5) kernel; got arch=%s", arch);
        return ROCKE_ERR_VALUE;
    }
    int head_size = p->head_size;
    if(head_size != 64 && head_size != 128)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "wmma_swapqk head_size must be 64 or 128 (got %d)", head_size);
        return ROCKE_ERR_VALUE;
    }
    if(block_n != 32 && block_n != 64)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "block_n must be 32 or 64 (got %d)", block_n);
        return ROCKE_ERR_VALUE;
    }
    if(n_waves != 1 && n_waves != 2)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "n_waves must be 1 or 2 (got %d)", n_waves);
        return ROCKE_ERR_VALUE;
    }
    if(p->mask_mode != ROCKE_ATTN_MASK_NONE && p->mask_mode != ROCKE_ATTN_MASK_CAUSAL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wmma_swapqk mask_mode must be none or causal");
        return ROCKE_ERR_VALUE;
    }
    if(p->causal_ctx_offset != NULL && p->mask_mode != ROCKE_ATTN_MASK_CAUSAL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "a causal context offset requires causal masking");
        return ROCKE_ERR_VALUE;
    }

    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wmma_swapqk: no target for arch %s", arch);
        return ROCKE_ERR_VALUE;
    }
    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&target->mma, "wmma_f32_16x16x16_f16");
    if(op == NULL || op->a_layout == NULL || op->b_layout == NULL || op->c_layout == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "wmma_swapqk: wmma_f32_16x16x16_f16 atom/layout absent on %s",
                        arch);
        return ROCKE_ERR_VALUE;
    }
    const int wave = op->wave_size; /* 32 */
    const int c_frag = op->c_frag_len; /* 8 */
    const int a_frag = op->a_frag_len; /* 16 */
    const int hs = head_size;
    const int n_dk = hs / 16;
    const int n_kv_sub = block_n / 16;

    rocke_value_t* Q = p->Q;
    rocke_value_t* K = p->K;
    rocke_value_t* V = p->V;
    rocke_value_t* O = p->O;
    rocke_value_t* seqlen_k = p->seqlen_k;
    rocke_value_t* q_tile_base = p->q_tile_base;
    rocke_value_t* q_pos_base = p->q_pos_base;
    rocke_value_t* head_idx = p->head_idx;
    rocke_value_t* kv_head_idx = p->kv_head_idx;
    rocke_value_t* stride_q_token = p->stride_q_token;
    rocke_value_t* stride_q_head = p->stride_q_head;
    rocke_value_t* stride_k_token = p->stride_k_token;
    rocke_value_t* stride_k_head = p->stride_k_head;
    rocke_value_t* stride_v_token = p->stride_v_token;
    rocke_value_t* stride_v_head = p->stride_v_head;
    rocke_value_t* stride_o_token = p->stride_o_token;
    rocke_value_t* stride_o_head = p->stride_o_head;
    rocke_value_t* scale_log2 = p->scale_log2;
    rocke_value_t* k_token_offset_elems = p->k_token_offset_elems;
    rocke_value_t* v_token_offset_elems = p->v_token_offset_elems;

    /* ---- thread -> (wave, lane, query column) ---- */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* c_wave = rocke_b_const_i32(b, wave);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* wave_id = rocke_b_div(b, tid, c_wave);
    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);
    rocke_value_t* col = rocke_b_mod(b, lane, c16); /* lane % 16 == query row within the 16-tile */
    rocke_value_t* lane_lt16 = rocke_b_cmp_lt(b, lane, c16);

    const bool strict = p->causal_ctx_offset != NULL || p->mask_neg_inf != NULL;
    rocke_value_t* neg_inf = p->mask_neg_inf;
    if(neg_inf == NULL)
        neg_inf = rocke_b_const_f32(b, strict ? -INFINITY : -1e30);
    rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);

    /* Q/K: (head, token, dim), dim contiguous. O^T view: (head, dim, token). */
    int qkv_shape[3] = {1, 1, hs};
    rocke_stride_t q_strides[3] = {
        rocke_stride_value(stride_q_head), rocke_stride_value(stride_q_token), rocke_stride_imm(1)};
    rocke_tensor_view_t Q_view;
    rocke_make_global_view(&Q_view, Q, qkv_shape, 3, rocke_f16(), q_strides);
    rocke_stride_t k_strides[3]
        = {rocke_stride_imm(1), rocke_stride_value(stride_k_token), rocke_stride_imm(1)};
    rocke_tensor_view_t K_view;
    rocke_make_global_view(&K_view, K, qkv_shape, 3, rocke_f16(), k_strides);
    int o_shape[3] = {1, hs, 1};
    rocke_stride_t o_strides[3] = {
        rocke_stride_value(stride_o_head), rocke_stride_imm(1), rocke_stride_value(stride_o_token)};
    rocke_tensor_view_t O_T_view;
    rocke_make_global_view(&O_T_view, O, o_shape, 3, rocke_f16(), o_strides);

    /* This wave's 16-row query tile: sequence-local (masking) and global
     * (addressing) row bases. */
    rocke_value_t* wave_row = rocke_b_mul(b, wave_id, c16);
    rocke_value_t* q_token_base = rocke_b_add(b, q_tile_base, wave_row);
    rocke_value_t* q_pos_local = rocke_b_add(b, q_pos_base, wave_row);
    rocke_tile_window_t qwin;
    {
        int lengths[3] = {1, 16, hs};
        rocke_value_t* origin[3] = {head_idx, q_token_base, c0};
        rocke_make_tile_window(&qwin, &Q_view, lengths, origin, 3);
    }

    /* ---- CK PermuteWarpGemmCToA byte selectors (upper 16 lanes swapped) ---- */
    rocke_value_t* sel0_lo = rocke_b_const_i32(b, 0x05040100);
    rocke_value_t* sel0_hi = rocke_b_const_i32(b, 0x01000504);
    rocke_value_t* sel0 = rocke_b_select(b, lane_lt16, sel0_lo, sel0_hi);
    rocke_value_t* sel1_lo = rocke_b_const_i32(b, 0x07060302);
    rocke_value_t* sel1_hi = rocke_b_const_i32(b, 0x03020706);
    rocke_value_t* sel1 = rocke_b_select(b, lane_lt16, sel1_lo, sel1_hi);

    /* ---- iter-args: m (scalar) | l (scalar) | acc (n_dk O^T f32 tiles) ---- */
    rocke_iter_arg_t iter_args[2 + ROCKE_SWAPQK_MAX_DK];
    char iter_names[2 + ROCKE_SWAPQK_MAX_DK][8];
    int n_ia = 0;
    strcpy(iter_names[n_ia], "m");
    iter_args[n_ia].name = iter_names[n_ia];
    iter_args[n_ia].init = neg_inf;
    ++n_ia;
    strcpy(iter_names[n_ia], "l");
    iter_args[n_ia].name = iter_names[n_ia];
    iter_args[n_ia].init = zero_f;
    ++n_ia;
    for(int d = 0; d < n_dk; ++d)
    {
        snprintf(iter_names[n_ia], sizeof(iter_names[0]), "acc%d", d);
        iter_args[n_ia].name = iter_names[n_ia];
        iter_args[n_ia].init = rocke_b_zero_vec_f32(b, c_frag);
        ++n_ia;
    }

    rocke_value_t* c_block_n = rocke_b_const_i32(b, block_n);
    rocke_value_t* loop_stop = rocke_b_div(b, seqlen_k, c_block_n);
    if(p->mask_mode == ROCKE_ATTN_MASK_CAUSAL)
    {
        rocke_value_t* span = rocke_b_const_i32(b, 16 * n_waves);
        rocke_value_t* max_row = rocke_b_add(b, q_pos_base, span);
        rocke_value_t* causal_stop = NULL;
        if(p->causal_ctx_offset == NULL)
        {
            rocke_value_t* stop_div = rocke_b_div(b, max_row, c_block_n);
            rocke_value_t* one = rocke_b_const_i32(b, 1);
            causal_stop = rocke_b_add(b, stop_div, one);
        }
        else
        {
            max_row = rocke_b_add(b, max_row, p->causal_ctx_offset);
            rocke_value_t* negative = rocke_b_cmp_lt(b, max_row, c0);
            max_row = rocke_b_select(b, negative, c0, max_row);
            rocke_value_t* last = rocke_b_const_i32(b, block_n - 1);
            rocke_value_t* rounded = rocke_b_add(b, max_row, last);
            causal_stop = rocke_b_div(b, rounded, c_block_n);
        }
        rocke_value_t* pick = rocke_b_cmp_lt(b, causal_stop, loop_stop);
        loop_stop = rocke_b_select(b, pick, causal_stop, loop_stop);
    }

    /* ---- buffer-descriptor D16 V-gather (address in the memory unit, no VALU) ---- */
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* buf_max = rocke_b_const_i32(b, 0x7FFFFFFF);
    rocke_value_t* v_rsrc = rocke_b_buffer_rsrc(b, V, buf_max);
    rocke_value_t* sv2
        = rocke_b_mul(b, stride_v_token, c2); /* bytes per kv step (loop-invariant) */
    rocke_value_t* soff_list[ROCKE_SWAPQK_MAX_AFRAG];
    for(int j = 0; j < a_frag; ++j)
    {
        rocke_value_t* cj = rocke_b_const_i32(b, j);
        soff_list[j] = rocke_b_mul(b, cj, sv2);
    }
    rocke_value_t* k_head_elems = rocke_b_mul(b, kv_head_idx, stride_k_head);
    rocke_value_t* k_base_elems = rocke_b_add(b, k_token_offset_elems, k_head_elems);
    rocke_value_t* v_head_elems = rocke_b_mul(b, kv_head_idx, stride_v_head);
    rocke_value_t* kvh_off = rocke_b_add(b, v_token_offset_elems, v_head_elems);

    rocke_value_t* kloop_step = rocke_b_const_i32(b, 1);
    rocke_for_t kloop
        = rocke_b_scf_for_iter(b, c0, loop_stop, kloop_step, iter_args, n_ia, "kt", false, true);
    rocke_b_region_enter(b, kloop.body);
    {
        rocke_value_t* m_i = kloop.iter_vars[0];
        rocke_value_t* l_i = kloop.iter_vars[1];
        rocke_value_t* accs[ROCKE_SWAPQK_MAX_DK];
        for(int d = 0; d < n_dk; ++d)
            accs[d] = kloop.iter_vars[2 + d];
        rocke_value_t* k_block_base = rocke_b_mul(b, kloop.iv, c_block_n);

        /* ---- QK: S^T = K @ Q^T. d-outer / kv-inner: Q[d] is invariant in kv,
         * so the n_kv_sub accumulator chains stay mutually independent (their
         * own ILP) and only n_dk Q loads are issued per K-tile. ---- */
        rocke_b_s_setprio(b, 1);
        rocke_tile_window_t kwins[ROCKE_SWAPQK_MAX_NS];
        for(int ns = 0; ns < n_kv_sub; ++ns)
        {
            rocke_value_t* off = rocke_b_const_i32(b, ns * 16);
            rocke_value_t* k_tile_base = rocke_b_add(b, k_block_base, off);
            rocke_swapqk_k_window(&kwins[ns], &K_view, k_base_elems, k_tile_base, c0, hs);
        }
        rocke_value_t* subs[ROCKE_SWAPQK_MAX_NS];
        for(int ns = 0; ns < n_kv_sub; ++ns)
            subs[ns] = rocke_b_zero_vec_f32(b, c_frag);
        for(int d = 0; d < n_dk; ++d)
        {
            rocke_value_t* q_tile
                = rocke_swapqk_load_operand(b, &qwin, op, lane, c0, false, d * 16);
            for(int ns = 0; ns < n_kv_sub; ++ns)
            {
                rocke_value_t* k_frag
                    = rocke_swapqk_load_operand(b, &kwins[ns], op, lane, c0, true, d * 16);
                subs[ns] = rocke_b_mma(b, op->op_id, k_frag, q_tile, subs[ns], NULL, 0);
            }
        }
        rocke_b_s_setprio(b, 0);
        rocke_b_s_setprio(b,
                          0); /* mirrors the pinned kernel's redundant post-compute_qk setprio(0) */

        rocke_value_t* k_bases[ROCKE_SWAPQK_MAX_NS];
        for(int ns = 0; ns < n_kv_sub; ++ns)
        {
            rocke_value_t* off = rocke_b_const_i32(b, ns * 16);
            k_bases[ns] = rocke_b_add(b, k_block_base, off);
        }

        /* ---- online softmax over ALL block_n keys (n_kv_sub*8 in-lane
         * slots + 1 permlanex16) ---- */
        rocke_value_t* s_sub[ROCKE_SWAPQK_MAX_NS][ROCKE_SWAPQK_MAX_CFRAG];
        for(int ns = 0; ns < n_kv_sub; ++ns)
        {
            rocke_value_t* off = rocke_b_const_i32(b, ns * 16);
            rocke_value_t* kv_base = rocke_b_add(b, k_block_base, off);
            for(int i = 0; i < c_frag; ++i)
            {
                rocke_value_t* kv_rel = NULL;
                rocke_value_t* q_rel = NULL;
                rocke_layout_map_coord(op->c_layout, b, lane, i, &kv_rel, &q_rel);
                rocke_value_t* slot = rocke_b_vec_extract(b, subs[ns], i);
                rocke_value_t* s_i = rocke_b_fmul(b, slot, scale_log2);
                rocke_value_t* k_idx = rocke_b_add(b, kv_base, kv_rel);
                rocke_value_t* query_pos = rocke_b_add(b, q_pos_local, q_rel);
                s_i = rocke_apply_attention_mask(b,
                                                 s_i,
                                                 p->mask_mode,
                                                 k_idx,
                                                 query_pos,
                                                 0,
                                                 p->causal_ctx_offset,
                                                 strict ? neg_inf : NULL);
                s_sub[ns][i] = s_i;
            }
        }

        rocke_value_t* all_s[ROCKE_SWAPQK_MAX_ALLS];
        int n_all = 0;
        for(int ns = 0; ns < n_kv_sub; ++ns)
            for(int i = 0; i < c_frag; ++i)
                all_s[n_all++] = s_sub[ns][i];
        rocke_value_t* local_max = rocke_swapqk_tree(b, all_s, n_all, rocke_b_fmax);
        rocke_value_t* local_max_x = rocke_swapqk_permx16_f32(b, local_max);
        rocke_value_t* tile_max = rocke_b_fmax(b, local_max, local_max_x);
        /* lazy: if every lane's tile_max is within threshold of m_i, don't
         * re-anchor (m_new = m_i -> alpha = 1) and skip the O rescale below. */
        rocke_value_t* delta = rocke_b_fsub(b, tile_max, m_i);
        rocke_value_t* thresh = rocke_b_const_f32(b, 8.0); /* bounds P before its FP16 cast */
        rocke_value_t* within = rocke_b_fcmp(b, "ole", delta, thresh);
        rocke_value_t* c1_i32 = rocke_b_const_i32(b, 1);
        rocke_value_t* below = rocke_b_select(b, within, c1_i32, c0);
        rocke_value_t* below_all = rocke_b_wave_all(b, below);
        rocke_value_t* skip_rescale = rocke_b_cmp_ne(b, below_all, c0); /* wave-uniform i1 */
        rocke_value_t* m_hi = rocke_b_fmax(b, m_i, tile_max);
        rocke_value_t* m_new = rocke_b_select(b, skip_rescale, m_i, m_hi);

        rocke_value_t* shift = m_new;
        if(strict)
        {
            rocke_value_t* empty = rocke_b_fcmp(b, "oeq", m_new, neg_inf);
            shift = rocke_b_select(b, empty, zero_f, m_new);
        }
        rocke_value_t* m_delta = rocke_b_fsub(b, m_i, shift);
        rocke_value_t* alpha = rocke_b_exp2_fast(b, m_delta);
        rocke_value_t* ps_sub[ROCKE_SWAPQK_MAX_NS][ROCKE_SWAPQK_MAX_CFRAG];
        for(int ns = 0; ns < n_kv_sub; ++ns)
            for(int i = 0; i < c_frag; ++i)
            {
                rocke_value_t* d = rocke_b_fsub(b, s_sub[ns][i], shift);
                ps_sub[ns][i] = rocke_b_exp2_fast(b, d);
            }
        rocke_value_t* all_p[ROCKE_SWAPQK_MAX_ALLS];
        int n_allp = 0;
        for(int ns = 0; ns < n_kv_sub; ++ns)
            for(int i = 0; i < c_frag; ++i)
                all_p[n_allp++] = ps_sub[ns][i];
        rocke_value_t* local_sum = rocke_swapqk_tree(b, all_p, n_allp, rocke_b_fadd);
        rocke_value_t* local_sum_x = rocke_swapqk_permx16_f32(b, local_sum);
        rocke_value_t* tile_sum = rocke_b_fadd(b, local_sum, local_sum_x);
        rocke_value_t* l_scaled = rocke_b_fmul(b, l_i, alpha);
        rocke_value_t* l_new = rocke_b_fadd(b, l_scaled, tile_sum);

        /* ---- alpha (rescale factor) + P operand tiles ---- */
        rocke_value_t* alpha_vec = rocke_b_zero_vec_f32(b, c_frag);
        for(int i = 0; i < c_frag; ++i)
            alpha_vec = rocke_b_vec_insert(b, alpha_vec, alpha, i);
        rocke_value_t* p_tiles[ROCKE_SWAPQK_MAX_NS];
        for(int ns = 0; ns < n_kv_sub; ++ns)
            p_tiles[ns]
                = rocke_swapqk_p_transpose_reg(b, ps_sub[ns], c_frag, a_frag, c16, sel0, sel1);

        /* ---- rescale the O^T accumulators by alpha ONCE per block_n keys ---- */
        /* wave-uniform 0/1-trip loop: run the n_dk rescale muls only when the
         * max re-anchored (skip_rescale False) -> 0-trip skips them. */
        rocke_value_t* n_res_one = rocke_b_const_i32(b, 1);
        rocke_value_t* n_res = rocke_b_select(b, skip_rescale, c0, n_res_one);
        rocke_iter_arg_t r_iter_args[ROCKE_SWAPQK_MAX_DK];
        char r_iter_names[ROCKE_SWAPQK_MAX_DK][8];
        for(int d = 0; d < n_dk; ++d)
        {
            snprintf(r_iter_names[d], sizeof(r_iter_names[0]), "ra%d", d);
            r_iter_args[d].name = r_iter_names[d];
            r_iter_args[d].init = accs[d];
        }
        rocke_value_t* rloop_step = rocke_b_const_i32(b, 1);
        rocke_for_t rloop
            = rocke_b_scf_for_iter(b, c0, n_res, rloop_step, r_iter_args, n_dk, "rsc", false, true);
        rocke_b_region_enter(b, rloop.body);
        {
            rocke_value_t* out_vals[ROCKE_SWAPQK_MAX_DK];
            for(int d = 0; d < n_dk; ++d)
                out_vals[d] = rocke_b_vector_mul(b, rloop.iter_vars[d], alpha_vec);
            rocke_b_scf_yield(b, out_vals, n_dk);
        }
        rocke_b_region_leave(b);
        rocke_value_t* new_accs[ROCKE_SWAPQK_MAX_DK];
        for(int d = 0; d < n_dk; ++d)
            new_accs[d] = (rloop.op != NULL) ? rloop.op->results[d] : NULL;

        /* ---- PV: O^T += V @ P per kv sub-tile (register P-transpose, no LDS) ---- */
        rocke_b_s_setprio(b, 1);
        for(int ns = 0; ns < n_kv_sub; ++ns)
        {
            for(int dp = 0; dp < n_dk; dp += 2)
            {
                rocke_value_t* loaded = rocke_swapqk_dual_gather_issue(b,
                                                                       v_rsrc,
                                                                       kvh_off,
                                                                       stride_v_token,
                                                                       c2,
                                                                       soff_list,
                                                                       a_frag,
                                                                       lane,
                                                                       col,
                                                                       c16,
                                                                       k_bases[ns],
                                                                       dp);
                rocke_value_t* frag_d = NULL;
                rocke_value_t* frag_d1 = NULL;
                rocke_swapqk_dual_gather_finish(
                    b, loaded, a_frag / 2, a_frag, lane_lt16, &frag_d, &frag_d1);
                new_accs[dp]
                    = rocke_b_mma(b, op->op_id, frag_d, p_tiles[ns], new_accs[dp], NULL, 0);
                new_accs[dp + 1]
                    = rocke_b_mma(b, op->op_id, frag_d1, p_tiles[ns], new_accs[dp + 1], NULL, 0);
            }
        }
        rocke_b_s_setprio(b, 0);

        rocke_value_t* yields[2 + ROCKE_SWAPQK_MAX_DK];
        yields[0] = m_new;
        yields[1] = l_new;
        for(int d = 0; d < n_dk; ++d)
            yields[2 + d] = new_accs[d];
        rocke_b_scf_yield(b, yields, 2 + n_dk);
    }
    rocke_b_region_leave(b);

    rocke_value_t* l_f = (kloop.op != NULL) ? kloop.op->results[1] : NULL;
    rocke_value_t* accs_f[ROCKE_SWAPQK_MAX_DK];
    for(int d = 0; d < n_dk; ++d)
        accs_f[d] = (kloop.op != NULL) ? kloop.op->results[2 + d] : NULL;

    /* ---- Epilogue: O^T[d, query] -> O[query, d], rescaled by 1/l. ---- */
    rocke_value_t* zmask = rocke_b_fcmp(b, "oeq", l_f, zero_f);
    rocke_value_t* l_rcp = rocke_b_rcp(b, l_f);
    rocke_value_t* inv_l = rocke_b_select(b, zmask, zero_f, l_rcp);

    for(int d = 0; d < n_dk; ++d)
    {
        rocke_tile_window_t owin;
        {
            rocke_value_t* d_off = rocke_b_const_i32(b, d * 16);
            int lengths[3] = {1, 16, 16};
            rocke_value_t* origin[3] = {head_idx, d_off, q_token_base};
            rocke_make_tile_window(&owin, &O_T_view, lengths, origin, 3);
        }
        rocke_swapqk_store_col(b, &owin, op, lane, accs_f[d], c0, 0, inv_l);
    }

    return rocke_ir_builder_status(b);
}
