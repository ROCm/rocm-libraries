// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.mx_scale.c -- C99 port of
 * rocke.helpers.mx_scale.decode_mx_scale_e8m0.
 *
 * Faithful translation of:
 *
 *     def decode_mx_scale_e8m0(b: IRBuilder, e8m0: Value) -> Value:
 *         if e8m0.type.name != "i8":
 *             raise ValueError(
 *                 f"decode_mx_scale_e8m0 expects i8 input, got {e8m0.type.name}")
 *         e_i32 = b.sext(e8m0, I32)
 *         is_zero = b.cmp_eq(e_i32, b.const_i32(0))
 *         is_nan  = b.cmp_eq(e_i32, b.const_i32(255))
 *         e_minus_127_f32 = b.fsub(b.sitofp_f32(e_i32), b.const_f32(127.0))
 *         raw_scale = b.exp2(e_minus_127_f32)
 *         zero = b.const_f32(0.0)
 *         return b.select(b.lor(is_zero, is_nan), zero, raw_scale)
 *
 * The builder-call sequence is reproduced in the exact same order as the Python
 * so the emitted IR (and SSA value numbering) stays byte-identical.
 */

#include "rocke/helper_rocke.helpers.mx_scale.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

rocke_value_t* rocke_decode_mx_scale_e8m0(rocke_ir_builder_t* b, rocke_value_t* e8m0)
{
    rocke_value_t* e_i32;
    rocke_value_t* is_zero;
    rocke_value_t* is_nan;
    rocke_value_t* e_minus_127_f32;
    rocke_value_t* raw_scale;
    rocke_value_t* zero;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* Python: if e8m0.type.name != "i8": raise ValueError(...) */
    if(e8m0 == NULL || e8m0->type == NULL || strcmp(e8m0->type->name, "i8") != 0)
    {
        const char* got = (e8m0 != NULL && e8m0->type != NULL) ? e8m0->type->name : "None";
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "decode_mx_scale_e8m0 expects i8 input, got %s", got);
    }

    /* e_i32 = b.sext(e8m0, I32) */
    e_i32 = rocke_b_sext(b, e8m0, rocke_i32());
    /* is_zero = b.cmp_eq(e_i32, b.const_i32(0)) */
    is_zero = rocke_b_cmp_eq(b, e_i32, rocke_b_const_i32(b, 0));
    /* is_nan = b.cmp_eq(e_i32, b.const_i32(255)) */
    is_nan = rocke_b_cmp_eq(b, e_i32, rocke_b_const_i32(b, 255));
    /* e_minus_127_f32 = b.fsub(b.sitofp_f32(e_i32), b.const_f32(127.0))
     *
     * Python evaluates fsub's arguments left-to-right: the sitofp is emitted
     * before the const_f32, and each consumes one SSA value-counter tick.
     * C argument-evaluation order is unspecified, so the two emissions are
     * sequenced explicitly to keep value numbering byte-identical. */
    {
        rocke_value_t* sitof = rocke_b_sitofp_f32(b, e_i32);
        rocke_value_t* c127 = rocke_b_const_f32(b, 127.0);
        e_minus_127_f32 = rocke_b_fsub(b, sitof, c127);
    }
    /* raw_scale = b.exp2(e_minus_127_f32) */
    raw_scale = rocke_b_exp2(b, e_minus_127_f32);
    /* zero = b.const_f32(0.0) */
    zero = rocke_b_const_f32(b, 0.0);
    /* return b.select(b.lor(is_zero, is_nan), zero, raw_scale) */
    return rocke_b_select(b, rocke_b_lor(b, is_zero, is_nan), zero, raw_scale);
}
