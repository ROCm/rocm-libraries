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

rocke_value_t* rocke_warp_xor_reduce_sum(rocke_ir_builder_t* b, rocke_value_t* v, int stages)
{
    /* cur = v
     * for k in range(stages):
     *     remote = b.warp_shuffle_xor(cur, 1 << k)
     *     cur = b.fadd(cur, remote)
     * return cur */
    rocke_value_t* cur = v;
    int k;
    for(k = 0; k < stages; ++k)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(b, cur, 1 << k);
        cur = rocke_b_fadd(b, cur, remote);
    }
    return cur;
}

/* ------------------------------------------------------ fp8 in-register dequant */

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
