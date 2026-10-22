// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_dense_lse.cpp -- C port of the dense-attention emitters in
 * library/kernels/common/_lse_store.py (see
 * rocke/instance_attention_dense_lse_internal.h).
 *
 * Every builder call is its own statement, in Python's left-to-right evaluation
 * order: C++ leaves argument evaluation order unspecified, and each call
 * consumes an SSA id.
 */
#include <math.h>

#include "rocke/helper_rocke.helpers.attention_fwd_ext.h"
#include "rocke/instance_attention_dense_lse_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

void rocke_attention_dense_lse_store(rocke_ir_builder_t* b,
                                     rocke_value_t* dst,
                                     rocke_value_t* element_offset,
                                     rocke_value_t* m_log2,
                                     rocke_value_t* l,
                                     rocke_value_t* row_has_mass)
{
    if(b == NULL || !rocke_ir_builder_ok(b))
        return;
    rocke_value_t* one = rocke_b_const_f32(b, 1.0);
    rocke_value_t* safe_l = rocke_b_select(b, row_has_mass, l, one);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* safe_m = rocke_b_select(b, row_has_mass, m_log2, zero);
    rocke_value_t* lse = rocke_natural_lse_from_log2_stats(b, safe_m, safe_l);
    rocke_value_t* neg_inf = rocke_b_const_f32(b, -INFINITY);
    rocke_value_t* value = rocke_b_select(b, row_has_mass, lse, neg_inf);
    rocke_b_global_store(b, dst, element_offset, value, 4);
}

rocke_value_t* rocke_attention_dense_row_has_key(rocke_ir_builder_t* b,
                                                 rocke_value_t* query_row,
                                                 rocke_value_t* kv_length,
                                                 bool causal,
                                                 int diagonal,
                                                 int window,
                                                 int right_bound)
{
    if(b == NULL || !rocke_ir_builder_ok(b))
        return NULL;
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

rocke_value_t* rocke_attention_dense_zero_keyless_row(rocke_ir_builder_t* b,
                                                      rocke_value_t* packed,
                                                      rocke_value_t* row_has_key,
                                                      const rocke_type_t* dtype,
                                                      int width)
{
    if(b == NULL || !rocke_ir_builder_ok(b))
        return NULL;
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
