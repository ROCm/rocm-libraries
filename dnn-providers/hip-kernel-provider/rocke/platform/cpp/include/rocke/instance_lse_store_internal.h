// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* LSE store internals. Mirrors library/kernels/common/_lse_store.py
 * (lse_store, lse_row_has_mass, and the dense-attention dense_row_has_key and
 * zero_keyless_row) and natural_lse_from_log2_stats from
 * rocke/helpers/attention_fwd_ext.py, op for op and in the same order, so the
 * two engines emit byte-identical IR. Every builder call is its own statement:
 * C leaves the evaluation order of call arguments unspecified, and the order
 * the ops are created in decides their SSA names. Not a public ABI: a Python
 * ValueError is a thrown ckc::Error (rocke_i_set_err), which the calling
 * builder's extern "C" entry point turns into the sticky builder error.
 * Not ported: validate_dense_lse and validate_unified_lse (host-side torch
 * tensor checks, no C caller). */
#ifndef ROCKE_INSTANCE_LSE_STORE_INTERNAL_H
#define ROCKE_INSTANCE_LSE_STORE_INTERNAL_H

#include <math.h>

#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* LN2 in attention_fwd_ext.py. */
#define ROCKE_LSE_LN2 0.6931471805599453

/* natural_lse_from_log2_stats: LN2 * (m_log2 + log2(row_sum)). */
static inline rocke_value_t* rocke_natural_lse_from_log2_stats(rocke_ir_builder_t* b,
                                                               rocke_value_t* m_log2,
                                                               rocke_value_t* row_sum)
{
    rocke_value_t* lg = rocke_b_log2(b, row_sum);
    rocke_value_t* sum = rocke_b_fadd(b, m_log2, lg);
    rocke_value_t* ln2 = rocke_b_const_f32(b, ROCKE_LSE_LN2);
    return rocke_b_fmul(b, sum, ln2);
}

/* lse_row_has_mass: !(l == 0) with NaN counted as mass, the exact inverse of
 * the tiled 2D O guard fcmp("oeq", l, 0). Built as one OR uno, not
 * lnot(oeq): the HIP lowerer lowers i1 NOT to ~bool, which is always true. */
static inline rocke_value_t* rocke_lse_row_has_mass(rocke_ir_builder_t* b, rocke_value_t* l)
{
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* ne = rocke_b_fcmp(b, "one", l, zero);
    rocke_value_t* nan = rocke_b_fcmp(b, "uno", l, zero);
    return rocke_b_lor(b, ne, nan);
}

/* lse_store: one FP32 store of the natural-log LSE; -inf when the row has no
 * mass. The caller guards row existence and writer ownership. */
static inline void rocke_lse_store(rocke_ir_builder_t* b,
                                   rocke_value_t* dst,
                                   rocke_value_t* element_offset,
                                   rocke_value_t* m_log2,
                                   rocke_value_t* l,
                                   rocke_value_t* row_has_mass)
{
    rocke_value_t* one = rocke_b_const_f32(b, 1.0);
    rocke_value_t* safe_l = rocke_b_select(b, row_has_mass, l, one);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* safe_m = rocke_b_select(b, row_has_mass, m_log2, zero);
    rocke_value_t* value = rocke_natural_lse_from_log2_stats(b, safe_m, safe_l);
    rocke_value_t* neg_inf = rocke_b_const_f32(b, -INFINITY);
    value = rocke_b_select(b, row_has_mass, value, neg_inf);
    rocke_b_global_store(b, dst, element_offset, value, 4);
}

/* dense_row_has_key(b, query_row, kv_length, *, causal, diagonal, window,
 * right_bound=0): i1, true iff the row's visible key range [lower, upper] is
 * non-empty. diagonal / window / right_bound are compile-time ints; 0 skips
 * the corresponding op exactly as the Python truthiness tests do. */
static inline rocke_value_t* rocke_dense_row_has_key(rocke_ir_builder_t* b,
                                                     rocke_value_t* query_row,
                                                     rocke_value_t* kv_length,
                                                     bool causal,
                                                     int diagonal,
                                                     int window,
                                                     int right_bound)
{
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* last_key = rocke_b_sub(b, kv_length, one);
    rocke_value_t* shifted_q = query_row;
    if(diagonal)
    {
        rocke_value_t* diag = rocke_b_const_i32(b, diagonal);
        shifted_q = rocke_b_add(b, query_row, diag);
    }
    rocke_value_t* upper_q = shifted_q;
    if(right_bound)
    {
        rocke_value_t* rb = rocke_b_const_i32(b, right_bound);
        upper_q = rocke_b_add(b, shifted_q, rb);
    }
    rocke_value_t* upper = causal ? rocke_b_smin(b, last_key, upper_q) : last_key;
    rocke_value_t* lower;
    if(window)
    {
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* span = rocke_b_const_i32(b, window - 1);
        rocke_value_t* start = rocke_b_sub(b, shifted_q, span);
        lower = rocke_b_smax(b, zero, start);
    }
    else
    {
        lower = rocke_b_const_i32(b, 0);
    }
    return rocke_b_cmp_le(b, lower, upper);
}

/* zero_keyless_row(b, packed, row_has_key, dtype, width=4): zero the packed
 * <width x dtype> O value for rows without keys with an integer AND on its
 * bits (a float select would let LLVM fuse the convert and change nonempty
 * fp16 O). width must be 1, 2 or 4; anything else raises the Python
 * ValueError "unsupported O store width {width}". */
static inline rocke_value_t* rocke_zero_keyless_row(rocke_ir_builder_t* b,
                                                    rocke_value_t* packed,
                                                    rocke_value_t* row_has_key,
                                                    const rocke_type_t* dtype,
                                                    int width)
{
    rocke_value_t* all_ones = rocke_b_const_i32(b, -1);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* mask = rocke_b_select(b, row_has_key, all_ones, zero);
    const rocke_type_t* lane;
    int lanes;
    if(width == 1)
    {
        lane = rocke_i16();
        lanes = 1;
        mask = rocke_b_trunc(b, mask, rocke_i16());
    }
    else if(width == 2 || width == 4)
    {
        lane = rocke_i32();
        lanes = width / 2;
    }
    else
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unsupported O store width %d", width);
    }
    rocke_value_t* as_ints = rocke_b_bitcast(b, packed, rocke_vector_type(b, lane, lanes));
    rocke_value_t* splat = rocke_b_vector_splat(b, mask, lanes);
    rocke_value_t* bits = rocke_b_vector_and(b, as_ints, splat);
    return rocke_b_bitcast(b, bits, rocke_vector_type(b, dtype, width));
}

#endif /* ROCKE_INSTANCE_LSE_STORE_INTERNAL_H */
