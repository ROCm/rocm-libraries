// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.attention_dropout -- C port of
 * rocke.helpers.attention_dropout (SEM-015 dropout mask words). Every builder
 * call is its own statement, in the Python evaluation order.
 */

#include "rocke/helper_rocke.helpers.attention_dropout.h"

#include "rocke/helper_rocke.helpers.rng.h"

void rocke_b_dropout_row_counter(rocke_ir_builder_t* b,
                                 rocke_value_t* offset_lo,
                                 rocke_value_t* offset_hi,
                                 rocke_value_t* row,
                                 rocke_value_t* max_sk,
                                 rocke_value_t** lo,
                                 rocke_value_t** hi)
{
    rocke_value_t* three = rocke_b_const_i32(b, 3);
    rocke_value_t* sk3 = rocke_b_add(b, max_sk, three);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* per_row = rocke_b_lshr(b, sk3, two);
    rocke_value_t* per_row64 = rocke_b_zext(b, per_row, rocke_i64());
    rocke_value_t* row_base = rocke_b_mul(b, row, per_row64);
    rocke_value_t* rb_lo = rocke_b_trunc(b, row_base, rocke_i32());
    rocke_value_t* sh32 = rocke_b_const_i64(b, 32);
    rocke_value_t* row_base_hi = rocke_b_lshr(b, row_base, sh32);
    rocke_value_t* rb_hi = rocke_b_trunc(b, row_base_hi, rocke_i32());
    rocke_value_t* l = rocke_b_add_u32_wrap(b, offset_lo, rb_lo);
    rocke_value_t* wrapped = rocke_b_cmp_ult(b, l, rb_lo);
    rocke_value_t* carry = rocke_b_zext(b, wrapped, rocke_i32());
    rocke_value_t* h0 = rocke_b_add_u32_wrap(b, offset_hi, rb_hi);
    *lo = l;
    *hi = rocke_b_add_u32_wrap(b, h0, carry);
}

void rocke_b_dropout_words(rocke_ir_builder_t* b,
                           rocke_value_t* seed_lo,
                           rocke_value_t* seed_hi,
                           rocke_value_t* ctr_lo,
                           rocke_value_t* ctr_hi,
                           bool xor3,
                           bool mad64,
                           rocke_value_t* out[4])
{
    rocke_philox_call_t call;
    rocke_philox_call_init(&call, b, seed_lo, seed_hi, ctr_lo, ctr_hi, xor3, mad64);
    rocke_philox_call_finish(&call, out);
}

rocke_value_t*
    rocke_b_dropout_keep(rocke_ir_builder_t* b, rocke_value_t* word, rocke_value_t* threshold)
{
    return rocke_b_cmp_gt(b, word, threshold);
}
