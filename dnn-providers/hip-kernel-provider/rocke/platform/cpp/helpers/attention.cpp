// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.attention.c -- C99 port of selected symbols from
 * rocke/helpers/attention.py.
 *
 * Each helper reproduces its Python counterpart's rocke_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands). The host-side control
 * structure (the Python ``None`` defaults, the mask_mode dispatch, the
 * fixed-count XOR butterfly loop) is reproduced exactly so the emitted op stream
 * is identical to the Python.
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph.
 */

#include <stdarg.h>
#include <stdio.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Set the builder's sticky error (first failure wins) and return NULL. Mirrors
 * the private set-err but binds only to rocke/ir.h's public struct fields
 * (status + err). */
/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing `return (T*)rocke_attn_set_err(...)`
 * call sites valid -- the cast/return is simply never reached. */
[[noreturn]] static void*
    rocke_attn_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

/* ------------------------------------------------------------------- masks */

rocke_value_t* rocke_causal_mask(rocke_ir_builder_t* b,
                                 rocke_value_t* key_pos,
                                 rocke_value_t* context_len,
                                 rocke_value_t* query_pos)
{
    /* return b.cmp_le(key_pos, b.add(context_len, query_pos)) */
    return rocke_b_cmp_le(b, key_pos, rocke_b_add(b, context_len, query_pos));
}

rocke_value_t* rocke_sliding_window_mask(rocke_ir_builder_t* b,
                                         rocke_value_t* key_pos,
                                         rocke_value_t* context_len,
                                         rocke_value_t* query_pos,
                                         int sliding_window)
{
    /* dist = b.sub(b.add(context_len, query_pos), key_pos)
     * return b.cmp_lt(dist, b.const_i32(sliding_window)) */
    rocke_value_t* dist = rocke_b_sub(b, rocke_b_add(b, context_len, query_pos), key_pos);
    return rocke_b_cmp_lt(b, dist, rocke_b_const_i32(b, (int64_t)sliding_window));
}

rocke_value_t* rocke_apply_attention_mask(rocke_ir_builder_t* b,
                                          rocke_value_t* score_log2,
                                          rocke_attn_mask_mode_t mask_mode,
                                          rocke_value_t* k_idx,
                                          rocke_value_t* query_pos,
                                          int sliding_window,
                                          rocke_value_t* context_len,
                                          rocke_value_t* neg_inf)
{
    rocke_value_t* keep;

    /* if mask_mode == "none": return score_log2 */
    if(mask_mode == ROCKE_ATTN_MASK_NONE)
    {
        return score_log2;
    }
    /* if neg_inf is None: neg_inf = b.const_f32(-1e30) */
    if(neg_inf == NULL)
    {
        neg_inf = rocke_b_const_f32(b, -1e30);
    }
    /* if context_len is None: context_len = b.const_i32(0) */
    if(context_len == NULL)
    {
        context_len = rocke_b_const_i32(b, 0);
    }
    if(mask_mode == ROCKE_ATTN_MASK_CAUSAL)
    {
        /* keep = causal_mask(b, k_idx, context_len, query_pos) */
        keep = rocke_causal_mask(b, k_idx, context_len, query_pos);
    }
    else if(mask_mode == ROCKE_ATTN_MASK_SLIDING_WINDOW)
    {
        /* keep = sliding_window_mask(b, k_idx, context_len, query_pos, sliding_window) */
        keep = rocke_sliding_window_mask(b, k_idx, context_len, query_pos, sliding_window);
    }
    else
    {
        /* raise ValueError(f"unknown mask_mode {mask_mode!r}") */
        return (rocke_value_t*)rocke_attn_set_err(
            b, ROCKE_ERR_VALUE, "unknown mask_mode %d", (int)mask_mode);
    }
    /* return b.select(keep, score_log2, neg_inf) */
    return rocke_b_select(b, keep, score_log2, neg_inf);
}

/* ------------------------------------------------------ online-softmax inv-l */

rocke_value_t* rocke_safe_inv_l(rocke_ir_builder_t* b, rocke_value_t* denom)
{
    /* zero_mask = b.fcmp("oeq", denom, b.const_f32(0.0))
     * inv_l_raw = b.rcp(denom)
     * return b.select(zero_mask, b.const_f32(0.0), inv_l_raw) */
    rocke_value_t* zero_mask = rocke_b_fcmp(b, "oeq", denom, rocke_b_const_f32(b, 0.0));
    rocke_value_t* inv_l_raw = rocke_b_rcp(b, denom);
    return rocke_b_select(b, zero_mask, rocke_b_const_f32(b, 0.0), inv_l_raw);
}

/* ---------------------------------------------- wave row-reduction selector */

int rocke_wave_reduce_stages(rocke_ir_builder_t* b, int wave_size, int lanes_per_row)
{
    int bits;
    int n;

    /* if lanes_per_row <= 0 or (lanes_per_row & (lanes_per_row - 1)) != 0:
     *     raise ValueError(...) */
    if(lanes_per_row <= 0 || (lanes_per_row & (lanes_per_row - 1)) != 0)
    {
        rocke_attn_set_err(
            b, ROCKE_ERR_VALUE, "lanes_per_row must be a power of two, got %d", lanes_per_row);
        return -1;
    }
    /* if lanes_per_row > wave_size: raise ValueError(...) */
    if(lanes_per_row > wave_size)
    {
        rocke_attn_set_err(b,
                           ROCKE_ERR_VALUE,
                           "lanes_per_row (%d) cannot exceed wave_size (%d)",
                           lanes_per_row,
                           wave_size);
        return -1;
    }
    /* return lanes_per_row.bit_length() - 1 */
    bits = 0;
    n = lanes_per_row;
    while(n > 0)
    {
        bits += 1;
        n >>= 1;
    }
    return bits - 1;
}

/* -------------------------------------------------- cross-lane sum reduction */

/* b.fadd as a butterfly_allreduce combine. */
static rocke_value_t*
    rocke_attn_fadd_cb(rocke_ir_builder_t* b, rocke_value_t* own, rocke_value_t* remote, void* user)
{
    (void)user;
    return rocke_b_fadd(b, own, remote);
}

rocke_value_t* rocke_warp_xor_reduce_sum(rocke_ir_builder_t* b, rocke_value_t* v, int stages)
{
    /* butterfly_allreduce(b, v, b.fadd, stages=stages).
     * Mirrors helpers/attention.py::warp_xor_reduce_sum. */
    return rocke_butterfly_allreduce(b, v, rocke_attn_fadd_cb, NULL, stages);
}

/* ------------------------------------------------------ fp8 in-register dequant */

rocke_value_t* rocke_decode_fp8e4m3fn_to_f32(rocke_ir_builder_t* b, rocke_value_t* byte)
{
    rocke_value_t* bits = rocke_b_zext(b, byte, rocke_i32());
    rocke_value_t* magnitude = rocke_b_land(b, bits, rocke_b_const_i32(b, 0x7F));
    rocke_value_t* sign_byte = rocke_b_land(b, bits, rocke_b_const_i32(b, 0x80));
    rocke_value_t* sign = rocke_b_shl(b, sign_byte, rocke_b_const_i32(b, 24));
    rocke_value_t* shifted = rocke_b_shl(b, magnitude, rocke_b_const_i32(b, 20));
    rocke_value_t* normal = rocke_b_add(b, shifted, rocke_b_const_i32(b, 120 << 23));
    rocke_value_t* subnormal_f = rocke_b_sitofp_f32(b, magnitude);
    rocke_value_t* subnormal = rocke_b_fmul(b, subnormal_f, rocke_b_const_f32(b, 0.001953125));
    rocke_value_t* is_subnormal = rocke_b_cmp_lt(b, magnitude, rocke_b_const_i32(b, 8));
    rocke_value_t* result
        = rocke_b_select(b, is_subnormal, rocke_b_bitcast(b, subnormal, rocke_i32()), normal);
    rocke_value_t* is_nan = rocke_b_cmp_eq(b, magnitude, rocke_b_const_i32(b, 0x7F));
    result = rocke_b_select(b, is_nan, rocke_b_const_i32(b, 0x7FC00000), result);
    return rocke_b_bitcast(b, rocke_b_lor(b, result, sign), rocke_f32());
}

rocke_value_t* rocke_dequant_fp8x8_to_dtype(rocke_ir_builder_t* b,
                                            rocke_value_t* fp8_vec,
                                            rocke_value_t* scale,
                                            const rocke_type_t* dtype)
{
    /* FP8E4M3 is the imported singleton scalar type in the Python; bind to the
     * ir.h accessor. */
    const rocke_type_t* fp8e4m3 = rocke_fp8e4m3();
    rocke_value_t* lo_comp[4];
    rocke_value_t* hi_comp[4];
    rocke_value_t* lo_fp8;
    rocke_value_t* hi_fp8;
    rocke_value_t* lo_f32;
    rocke_value_t* hi_f32;
    rocke_value_t* deq[8];
    int i;

    if(dtype == NULL)
    {
        return (rocke_value_t*)rocke_attn_set_err(
            b, ROCKE_ERR_VALUE, "dequant_fp8x8_to_dtype: dtype is NULL");
    }

    /* lo_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4)], FP8E4M3) */
    for(i = 0; i < 4; ++i)
    {
        lo_comp[i] = rocke_b_vec_extract(b, fp8_vec, i);
    }
    lo_fp8 = rocke_b_vec_pack(b, lo_comp, 4, fp8e4m3);

    /* hi_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4, 8)], FP8E4M3) */
    for(i = 0; i < 4; ++i)
    {
        hi_comp[i] = rocke_b_vec_extract(b, fp8_vec, 4 + i);
    }
    hi_fp8 = rocke_b_vec_pack(b, hi_comp, 4, fp8e4m3);

    /* lo_f32 = b.cvt_pk_f32_fp8x4(lo_fp8); hi_f32 = b.cvt_pk_f32_fp8x4(hi_fp8) */
    lo_f32 = rocke_b_cvt_pk_f32_fp8x4(b, lo_fp8);
    hi_f32 = rocke_b_cvt_pk_f32_fp8x4(b, hi_fp8);

    /* deq = [b.cast_f32_to(b.fmul(b.vec_extract(lo_f32, i), scale), dtype) for i in range(4)]
     *     + [b.cast_f32_to(b.fmul(b.vec_extract(hi_f32, i), scale), dtype) for i in range(4)]
     *
     * The list comprehension evaluates the lo loop fully, then the hi loop;
     * within each iteration the order is vec_extract -> fmul -> cast_f32_to. */
    for(i = 0; i < 4; ++i)
    {
        deq[i] = rocke_b_cast_f32_to(
            b, rocke_b_fmul(b, rocke_b_vec_extract(b, lo_f32, i), scale), dtype);
    }
    for(i = 0; i < 4; ++i)
    {
        deq[4 + i] = rocke_b_cast_f32_to(
            b, rocke_b_fmul(b, rocke_b_vec_extract(b, hi_f32, i), scale), dtype);
    }

    /* return b.vec_pack(deq, dtype) */
    return rocke_b_vec_pack(b, deq, 8, dtype);
}

// pv32_v_load_paired: two transpose reads provide one 32x32x16 PV A operand.
rocke_value_t* rocke_pv32_v_load_paired(rocke_ir_builder_t* b,
                                        rocke_value_t* V_lds,
                                        rocke_value_t* v_buf,
                                        int n,
                                        int k,
                                        rocke_value_t* lane_half32,
                                        rocke_value_t* lane_col32,
                                        const rocke_type_t* dtype)
{
    /* col_group16 = (lane_col32 / 16) * 16. Python emits the div (with its const
     * 16) BEFORE the mul's const 16; bind the div first. */
    rocke_value_t* col_div16 = rocke_b_div(b, lane_col32, rocke_b_const_i32(b, 16));
    rocke_value_t* col_group16 = rocke_b_mul(b, col_div16, rocke_b_const_i32(b, 16));
    /* tr_col32 = col_group16 + (lane_col32 % 4) * 4. Bind the mod first. */
    rocke_value_t* tr_col_mod = rocke_b_mod(b, lane_col32, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_col_rhs = rocke_b_mul(b, tr_col_mod, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_col32 = rocke_b_add(b, col_group16, tr_col_rhs);
    /* tr_row_base32 = (k*16 + lane_half32*4) + ((lane_col32 / 4) % 4). Python
     * evaluates the inner add (const(k*16), mul(lane_half32,4)) BEFORE the mod. */
    rocke_value_t* tr_row_k = rocke_b_const_i32(b, (int64_t)(k * 16));
    rocke_value_t* tr_row_inner
        = rocke_b_add(b, tr_row_k, rocke_b_mul(b, lane_half32, rocke_b_const_i32(b, 4)));
    rocke_value_t* tr_row_div4 = rocke_b_div(b, lane_col32, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_row_mod = rocke_b_mod(b, tr_row_div4, rocke_b_const_i32(b, 4));
    rocke_value_t* tr_row_base32 = rocke_b_add(b, tr_row_inner, tr_row_mod);

    /* Python computes col = add(const(n*32), tr_col32) INLINE inside EACH
     * ds_read call (helpers/attention.py 844-846, 851-853), so it is emitted
     * TWICE -- once per read. Mirror that (do NOT hoist) for value numbering. */
    rocke_value_t* col0 = rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n * 32)), tr_col32);
    rocke_value_t* idx0[3] = {v_buf, tr_row_base32, col0};
    rocke_value_t* A_r0 = rocke_b_ds_read_tr16_b64(b, V_lds, idx0, 3, dtype);
    rocke_value_t* row1 = rocke_b_add(b, tr_row_base32, rocke_b_const_i32(b, 8));
    rocke_value_t* col1 = rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n * 32)), tr_col32);
    rocke_value_t* idx1[3] = {v_buf, row1, col1};
    rocke_value_t* A_r1 = rocke_b_ds_read_tr16_b64(b, V_lds, idx1, 3, dtype);
    return rocke_b_vec_concat(b, A_r0, A_r1);
}
