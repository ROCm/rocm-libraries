// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * mx_scale.cpp -- the C++ engine port of
 * rocke.helpers.mx_scale.decode_mx_scale_e8m0,
 * rocke.helpers.mx_scale.largest_safe_mx_scale_code and
 * rocke.helpers.mx_scale.add_mx_scaled_dot.
 *
 * Faithful translation of (decoder shown; add_mx_scaled_dot follows the same
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

#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

#define E8M0_BIAS 127
#define E8M0_NAN_CODE 255
#define E8M0_LARGEST_FINITE_CODE 254
/* The f32 overflow threshold is 2^128 (less half an ulp); a sum bounded by
 * 2^127 stays a full binade below it. */
#define F32_SAFE_SUM_EXPONENT 127

/* Python FP8_E4M3_OCP, BF8_E5M2_OCP, FP8_E4M3_FNUZ, BF8_E5M2_FNUZ. */
const rocke_mx_element_format_t ROCKE_FP8_E4M3_OCP = {"OCP E4M3", 448.0};
const rocke_mx_element_format_t ROCKE_BF8_E5M2_OCP = {"OCP E5M2", 57344.0};
const rocke_mx_element_format_t ROCKE_FP8_E4M3_FNUZ = {"E4M3 FNUZ", 240.0};
const rocke_mx_element_format_t ROCKE_BF8_E5M2_FNUZ = {"E5M2 FNUZ", 57344.0};

rocke_status_t rocke_largest_safe_mx_scale_code(const rocke_mx_element_format_t* a_format,
                                                const rocke_mx_element_format_t* b_format,
                                                int products,
                                                int* out,
                                                char* err,
                                                size_t err_cap)
{
    double bound;
    int top;
    int code;

    if(a_format == NULL || b_format == NULL || out == NULL || products <= 0)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err,
                     err_cap,
                     "largest_safe_mx_scale_code: products must be positive, got %d",
                     products);
        }
        return ROCKE_ERR_VALUE;
    }
    bound = (double)products * a_format->largest_value * b_format->largest_value;
    top = (int)ceil(log2(bound)); /* bound <= 2^top */
    /* 2c - 2 * bias + top <= 127 (Python floor division of a positive value) */
    code = (F32_SAFE_SUM_EXPONENT + 2 * E8M0_BIAS - top) / 2;
    *out = code < E8M0_LARGEST_FINITE_CODE ? code : E8M0_LARGEST_FINITE_CODE;
    return ROCKE_OK;
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

rocke_value_t* rocke_add_mx_scaled_dot(rocke_ir_builder_t* b,
                                       rocke_value_t* acc,
                                       rocke_value_t* dot,
                                       rocke_value_t* const* a_codes,
                                       size_t n_a_codes,
                                       rocke_value_t* const* b_codes,
                                       size_t n_b_codes)
{
    /* Each distinct code Value is read once (Python's `read` dict keyed by
     * id): (byte, code, is_nan). */
    struct read_code
    {
        rocke_value_t* byte;
        rocke_value_t* code;
        rocke_value_t* is_nan;
    };
    std::vector<read_code> read;
    std::vector<rocke_value_t*> terms;
    rocke_value_t* nan_value;
    rocke_value_t* two_biases;
    int count;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(dot == NULL || dot->type == NULL || dot->type->kind != ROCKE_TYPE_VECTOR
       || strcmp(dot->type->elem->name, "f32") != 0)
    {
        const char* got = (dot != NULL && dot->type != NULL) ? dot->type->name : "None";
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "add_mx_scaled_dot expects an f32 vector dot, got %s", got);
    }
    if(acc == NULL || acc->type == NULL || strcmp(acc->type->name, dot->type->name) != 0)
    {
        const char* got = (acc != NULL && acc->type != NULL) ? acc->type->name : "None";
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "add_mx_scaled_dot expects the accumulator type "
                                               "to match the dot type %s, got %s",
                                               dot->type->name,
                                               got);
    }
    count = dot->type->count;
    if(n_a_codes != (size_t)count || n_b_codes != (size_t)count)
    {
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "add_mx_scaled_dot expects %d A and %d B scale "
                                               "bytes, got %zu and %zu",
                                               count,
                                               count,
                                               n_a_codes,
                                               n_b_codes);
    }

    /* code_of(e8m0): read each distinct byte once. */
    auto code_of = [&](rocke_value_t* byte) -> const read_code* {
        for(const read_code& r : read)
        {
            if(r.byte == byte)
            {
                return &r;
            }
        }
        rocke_value_t* is_nan = NULL;
        rocke_value_t* code = read_e8m0_code("add_mx_scaled_dot", b, byte, &is_nan);
        if(code == NULL)
        {
            return NULL;
        }
        read.push_back({byte, code, is_nan});
        return &read.back();
    };

    /* nan = b.const_f32(math.nan); two_biases = b.const_i32(2 * _E8M0_BIAS) */
    nan_value = rocke_b_const_f32(b, (double)NAN);
    two_biases = rocke_b_const_i32(b, 2 * E8M0_BIAS);
    terms.reserve((size_t)count);
    for(int i = 0; i < count; ++i)
    {
        /* a_code, a_is_nan = code_of(a_codes[i]); then the B code. The
         * pointers are copied out: a later push_back may move `read`. */
        const read_code* a_read = code_of(a_codes[i]);
        if(a_read == NULL)
        {
            return NULL;
        }
        rocke_value_t* a_code = a_read->code;
        rocke_value_t* a_is_nan = a_read->is_nan;
        const read_code* b_read = code_of(b_codes[i]);
        if(b_read == NULL)
        {
            return NULL;
        }
        rocke_value_t* b_code = b_read->code;
        rocke_value_t* b_is_nan = b_read->is_nan;
        /* exponent = b.sub(b.add(a_code, b_code), two_biases) */
        rocke_value_t* exponent = rocke_b_sub(b, rocke_b_add(b, a_code, b_code), two_biases);
        /* term = b.ldexp(b.vec_extract(dot, i), exponent) */
        rocke_value_t* term = rocke_b_ldexp(b, rocke_b_vec_extract(b, dot, i), exponent);
        /* term = b.select(b.lor(a_is_nan, b_is_nan), nan, term) */
        term = rocke_b_select(b, rocke_b_lor(b, a_is_nan, b_is_nan), nan_value, term);
        /* terms.append(b.fadd(b.vec_extract(acc, i), term)) */
        terms.push_back(rocke_b_fadd(b, rocke_b_vec_extract(b, acc, i), term));
    }
    /* return b.vec_pack(terms, F32) */
    return rocke_b_vec_pack(b, terms.data(), (int)terms.size(), rocke_f32());
}
