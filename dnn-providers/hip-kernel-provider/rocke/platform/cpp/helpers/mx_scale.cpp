// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.mx_scale.c -- C99 port of
 * rocke.helpers.mx_scale.decode_mx_scale_e8m0,
 * rocke.helpers.mx_scale.exact_mx_scale_split and
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
#include <stdio.h>
#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

#define E8M0_BIAS 127
#define E8M0_NAN_CODE 255
/* ea + eb for two finite E8M0 codes (0..254) spans -254..254. */
#define SMALLEST_EXPONENT_SUM (-2 * E8M0_BIAS)
#define LARGEST_EXPONENT_SUM (2 * E8M0_BIAS)
/* The f32 powers of two: 2^-149 (the smallest subnormal) .. 2^127. */
#define F32_SMALLEST_EXPONENT (-149)
#define F32_LARGEST_EXPONENT 127
#define F32_SIGNIFICAND_BITS 24

/* Python FP8_E4M3_OCP, BF8_E5M2_OCP, FP8_E4M3_FNUZ, BF8_E5M2_FNUZ. */
const rocke_mx_element_format_t ROCKE_FP8_E4M3_OCP = {"OCP E4M3", -9, 448.0};
const rocke_mx_element_format_t ROCKE_BF8_E5M2_OCP = {"OCP E5M2", -16, 57344.0};
const rocke_mx_element_format_t ROCKE_FP8_E4M3_FNUZ = {"E4M3 FNUZ", -10, 240.0};
const rocke_mx_element_format_t ROCKE_BF8_E5M2_FNUZ = {"E5M2 FNUZ", -17, 57344.0};

/* Python _f32_top_exponent(magnitude): e with 2^e <= y < 2^(e + 1), for y the
 * smallest f32 that is at least `magnitude` (a positive double holding its
 * exact value). */
static int f32_top_exponent(double magnitude)
{
    int exponent;
    int top;

    (void)frexp(magnitude, &exponent); /* magnitude = fraction * 2^exponent */
    top = exponent - 1;
    /* Above the largest 24-bit significand at 2^top, an f32 rounds up to
     * 2^(top + 1). */
    if(magnitude > (2.0 - ldexp(1.0, 1 - F32_SIGNIFICAND_BITS)) * ldexp(1.0, top))
    {
        top += 1;
    }
    return top;
}

static rocke_status_t split_refusal(char* err, size_t err_cap, const char* msg)
{
    if(err != NULL && err_cap > 0)
    {
        snprintf(err, err_cap, "%s", msg);
    }
    return ROCKE_ERR_VALUE;
}

rocke_status_t rocke_exact_mx_scale_split(const rocke_mx_element_format_t* a_format,
                                          const rocke_mx_element_format_t* b_format,
                                          int group_size,
                                          rocke_mx_scale_split_t* out,
                                          char* err,
                                          size_t err_cap)
{
    char pair[192];
    char msg[512];
    int low;
    int top;
    int pre_min;
    int exponent_cap;

    if(a_format == NULL || b_format == NULL || out == NULL)
    {
        return split_refusal(err, err_cap, "exact_mx_scale_split: NULL argument");
    }
    low = a_format->smallest_exponent + b_format->smallest_exponent;
    top = f32_top_exponent((double)group_size * a_format->largest_value * b_format->largest_value);
    snprintf(pair,
             sizeof(pair),
             "%d products of %s and %s values",
             group_size,
             a_format->name,
             b_format->name);
    if(top > F32_LARGEST_EXPONENT)
    {
        snprintf(msg,
                 sizeof(msg),
                 "exact_mx_scale_split: the dot product of %s can reach 2^%d, "
                 "beyond the f32 range",
                 pair,
                 top);
        return split_refusal(err, err_cap, msg);
    }
    pre_min = F32_SMALLEST_EXPONENT - low;
    if(SMALLEST_EXPONENT_SUM - F32_SMALLEST_EXPONENT < pre_min)
    {
        snprintf(msg,
                 sizeof(msg),
                 "exact_mx_scale_split: the dot product of %s can be as small as "
                 "2^%d; at the smallest scales (ea + eb = %d) it needs a factor of "
                 "2^%d before the fma, which would round it",
                 pair,
                 low,
                 SMALLEST_EXPONENT_SUM,
                 SMALLEST_EXPONENT_SUM - F32_SMALLEST_EXPONENT);
        return split_refusal(err, err_cap, msg);
    }
    exponent_cap = 2 * F32_LARGEST_EXPONENT - top;
    if(exponent_cap < LARGEST_EXPONENT_SUM && low + exponent_cap < F32_LARGEST_EXPONENT + 2)
    {
        snprintf(msg,
                 sizeof(msg),
                 "exact_mx_scale_split: the dot product of %s spans 2^%d to 2^%d, "
                 "too wide to saturate ea + eb at %d without changing a result",
                 pair,
                 low,
                 top + 1,
                 exponent_cap);
        return split_refusal(err, err_cap, msg);
    }
    out->exponent_cap = exponent_cap;
    out->fused_min = F32_SMALLEST_EXPONENT;
    out->fused_max = F32_LARGEST_EXPONENT;
    return ROCKE_OK;
}

int rocke_mx_scale_split_pre_max(const rocke_mx_scale_split_t* split)
{
    return split->exponent_cap - split->fused_max;
}

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
                                            rocke_value_t* addend,
                                            const rocke_mx_scale_split_t* split)
{
    rocke_value_t* a_code;
    rocke_value_t* a_is_nan;
    rocke_value_t* b_code;
    rocke_value_t* b_is_nan;
    rocke_value_t* exponent;
    rocke_value_t* fused;
    rocke_value_t* pre;
    rocke_value_t* pre_scale;
    rocke_value_t* fused_scale;
    rocke_value_t* pre_scaled;
    int lanes;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    /* Python: split must be an MxScaleSplit (checked first, as in Python). */
    if(split == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "fma_with_mx_scale_pair expects an MxScaleSplit, got None");
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
    /* fused = b.smin(b.smax(exponent, b.const_i32(split.fused_min)),
     *                b.const_i32(split.fused_max)) */
    {
        rocke_value_t* low = rocke_b_const_i32(b, split->fused_min);
        rocke_value_t* at_least_low = rocke_b_smax(b, exponent, low);
        rocke_value_t* high = rocke_b_const_i32(b, split->fused_max);
        fused = rocke_b_smin(b, at_least_low, high);
    }
    /* pre = b.smin(b.sub(exponent, fused), b.const_i32(split.pre_max)) */
    {
        rocke_value_t* unsaturated = rocke_b_sub(b, exponent, fused);
        rocke_value_t* pre_max = rocke_b_const_i32(b, rocke_mx_scale_split_pre_max(split));
        pre = rocke_b_smin(b, unsaturated, pre_max);
    }
    /* pre_scale = b.exp2(b.sitofp_f32(pre)) */
    pre_scale = rocke_b_exp2(b, rocke_b_sitofp_f32(b, pre));
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
