// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.mx_scale.c -- C99 port of
 * rocke.helpers.mx_scale.decode_mx_scale_e8m0 and
 * rocke.helpers.mx_scale.fma_with_mx_scale_pair.
 *
 * Faithful translation of (decoder shown; the pair helper follows the same
 * rules and is quoted step by step below):
 *
 *     def decode_mx_scale_e8m0(b: IRBuilder, e8m0: Value) -> Value:
 *         code, is_nan_code = _read_e8m0_code("decode_mx_scale_e8m0", b, e8m0)
 *         exponent = b.fsub(b.sitofp_f32(code), b.const_f32(127.0))
 *         scale = b.exp2(exponent)
 *         return b.select(is_nan_code, b.const_f32(math.nan), scale)
 *
 * The builder-call sequence is reproduced in the exact same order as the Python
 * so the emitted IR (and SSA value numbering) stays byte-identical.
 */

#include "rocke/helper_rocke.helpers.mx_scale.h"

#include <math.h>
#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

#define E8M0_BIAS 127
#define E8M0_NAN_CODE 255

/* Python: _read_e8m0_code(caller, b, e8m0) -> (code, is_nan).
 *
 * The byte is an unsigned biased exponent, so it is zero-extended to an i32
 * code 0..255; *is_nan is true for the NaN code 255. Returns NULL (with the
 * builder's sticky error set) on a non-i8 input. */
static rocke_value_t* read_e8m0_code(const char* caller,
                                     rocke_ir_builder_t* b,
                                     rocke_value_t* e8m0,
                                     rocke_value_t** is_nan)
{
    rocke_value_t* code;

    *is_nan = NULL;
    if(e8m0 == NULL || e8m0->type == NULL || strcmp(e8m0->type->name, "i8") != 0)
    {
        const char* got = (e8m0 != NULL && e8m0->type != NULL) ? e8m0->type->name : "None";
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "%s expects i8 input, got %s", caller, got);
    }
    /* code = b.zext(e8m0, I32) */
    code = rocke_b_zext(b, e8m0, rocke_i32());
    /* is_nan = b.cmp_eq(code, b.const_i32(255)) */
    *is_nan = rocke_b_cmp_eq(b, code, rocke_b_const_i32(b, E8M0_NAN_CODE));
    return code;
}

rocke_value_t* rocke_decode_mx_scale_e8m0(rocke_ir_builder_t* b, rocke_value_t* e8m0)
{
    rocke_value_t* code;
    rocke_value_t* is_nan_code;
    rocke_value_t* exponent;
    rocke_value_t* scale;
    rocke_value_t* nan_value;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* code, is_nan_code = _read_e8m0_code("decode_mx_scale_e8m0", b, e8m0) */
    code = read_e8m0_code("decode_mx_scale_e8m0", b, e8m0, &is_nan_code);
    if(code == NULL)
    {
        return NULL;
    }
    /* exponent = b.fsub(b.sitofp_f32(code), b.const_f32(127.0))
     *
     * Python evaluates fsub's arguments left-to-right: the sitofp is emitted
     * before the const_f32, and each consumes one SSA value-counter tick.
     * C argument-evaluation order is unspecified, so the two emissions are
     * sequenced explicitly to keep value numbering byte-identical. */
    {
        rocke_value_t* code_f32 = rocke_b_sitofp_f32(b, code);
        rocke_value_t* c127 = rocke_b_const_f32(b, 127.0);
        exponent = rocke_b_fsub(b, code_f32, c127);
    }
    /* scale = b.exp2(exponent) */
    scale = rocke_b_exp2(b, exponent);
    /* return b.select(is_nan_code, b.const_f32(math.nan), scale) */
    nan_value = rocke_b_const_f32(b, (double)NAN);
    return rocke_b_select(b, is_nan_code, nan_value, scale);
}

rocke_value_t* rocke_fma_with_mx_scale_pair(rocke_ir_builder_t* b,
                                            rocke_value_t* value,
                                            rocke_value_t* a_e8m0,
                                            rocke_value_t* b_e8m0,
                                            rocke_value_t* addend)
{
    rocke_value_t* a_code;
    rocke_value_t* a_is_nan;
    rocke_value_t* b_code;
    rocke_value_t* b_is_nan;
    rocke_value_t* exponent;
    rocke_value_t* fused;
    rocke_value_t* pre_scale;
    rocke_value_t* fused_scale;
    rocke_value_t* pre_scaled;
    int lanes;

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: value must be an f32 vector; the addend must match its type. */
    if(value == NULL || value->type == NULL || value->type->kind != ROCKE_TYPE_VECTOR
       || strcmp(value->type->elem->name, "f32") != 0)
    {
        const char* got = (value != NULL && value->type != NULL) ? value->type->name : "None";
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "fma_with_mx_scale_pair expects an f32 vector value, got %s", got);
    }
    if(addend == NULL || addend->type == NULL || strcmp(addend->type->name, value->type->name) != 0)
    {
        const char* got = (addend != NULL && addend->type != NULL) ? addend->type->name : "None";
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "fma_with_mx_scale_pair expects the addend type to "
                                               "match the value type %s, got %s",
                                               value->type->name,
                                               got);
    }

    a_code = read_e8m0_code("fma_with_mx_scale_pair", b, a_e8m0, &a_is_nan);
    if(a_code == NULL)
    {
        return NULL;
    }
    b_code = read_e8m0_code("fma_with_mx_scale_pair", b, b_e8m0, &b_is_nan);
    if(b_code == NULL)
    {
        return NULL;
    }

    /* Every step below is sequenced in Python's left-to-right evaluation
     * order: each emission consumes one SSA value-counter tick, so the order
     * is load-bearing for byte-identical value numbering.
     *
     * exponent = b.sub(b.add(a_code, b_code), b.const_i32(2 * 127)) */
    {
        rocke_value_t* code_sum = rocke_b_add(b, a_code, b_code);
        rocke_value_t* double_bias = rocke_b_const_i32(b, 2 * E8M0_BIAS);
        exponent = rocke_b_sub(b, code_sum, double_bias);
    }
    /* fused = b.smin(b.smax(exponent, b.const_i32(-127)), b.const_i32(127)) */
    {
        rocke_value_t* low = rocke_b_const_i32(b, -E8M0_BIAS);
        rocke_value_t* at_least_low = rocke_b_smax(b, exponent, low);
        rocke_value_t* high = rocke_b_const_i32(b, E8M0_BIAS);
        fused = rocke_b_smin(b, at_least_low, high);
    }
    /* pre_scale = b.exp2(b.sitofp_f32(b.sub(exponent, fused))) */
    pre_scale = rocke_b_exp2(b, rocke_b_sitofp_f32(b, rocke_b_sub(b, exponent, fused)));
    /* fused_scale = b.select(b.lor(a_is_nan, b_is_nan), b.const_f32(math.nan),
     *                        b.exp2(b.sitofp_f32(fused))) */
    {
        rocke_value_t* is_nan = rocke_b_lor(b, a_is_nan, b_is_nan);
        rocke_value_t* nan_value = rocke_b_const_f32(b, (double)NAN);
        rocke_value_t* scale = rocke_b_exp2(b, rocke_b_sitofp_f32(b, fused));
        fused_scale = rocke_b_select(b, is_nan, nan_value, scale);
    }
    /* pre_scaled = b.vector_mul(value, b.vector_splat(pre_scale, lanes)) */
    lanes = value->type->count;
    pre_scaled = rocke_b_vector_mul(b, value, rocke_b_vector_splat(b, pre_scale, lanes));
    /* return b.vector_fma(pre_scaled, b.vector_splat(fused_scale, lanes), addend) */
    return rocke_b_vector_fma(b, pre_scaled, rocke_b_vector_splat(b, fused_scale, lanes), addend);
}
