// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of unpack_i4_byte_to_pair_i32 / unpack_i4_byte_to_pair_f32 /
 * unpack_i4_byte_to_pair_f16 / dequant_i4_byte_to_f16_pair from
 * rocke/helpers/i4_dequant.py. See helper_rocke.helpers.i4_dequant.h for the
 * contract.
 *
 * Every symbol emits IR. The rocke_b_* op-emission order below is byte-faithful to
 * the Python source-line evaluation order so the lowered IR is identical.
 */

#include "rocke/helper_rocke.helpers.i4_dequant.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

/* ------------------------------------------------ unpack_i4_byte_to_pair_i32 */

rocke_status_t rocke_unpack_i4_byte_to_pair_i32(rocke_ir_builder_t* b,
                                                rocke_value_t* packed_byte,
                                                rocke_value_t** out_low,
                                                rocke_value_t** out_high)
{
    rocke_value_t* byte_i32;
    rocke_value_t* mask_lo;
    rocke_value_t* c8;
    rocke_value_t* c16;
    rocke_value_t* low_unsigned;
    rocke_value_t* high_unsigned;
    rocke_value_t* low_signed;
    rocke_value_t* high_signed;

    /* Sticky-error model: a failed builder makes every call a no-op. */
    if(!rocke_i_live(b))
    {
        return b != NULL ? b->status : ROCKE_ERR_VALUE;
    }

    /* Python: if packed_byte.type.name != "i8": raise ValueError(...) */
    if(packed_byte == NULL || packed_byte->type == NULL || packed_byte->type->name == NULL
       || strcmp(packed_byte->type->name, "i8") != 0)
    {
        const char* got
            = (packed_byte != NULL && packed_byte->type != NULL && packed_byte->type->name != NULL)
                  ? packed_byte->type->name
                  : "(null)";
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unpack_i4_byte_to_pair_i32 expects i8 input, got %s", got);
        return ROCKE_ERR_VALUE;
    }

    /* byte_i32 = b.sext(packed_byte, I32) */
    byte_i32 = rocke_b_sext(b, packed_byte, rocke_i32());

    /* mask_lo = b.const_i32(0x0F); c8 = b.const_i32(8); c16 = b.const_i32(16).
     * All three literals are bound before either land (matches Python order). */
    mask_lo = rocke_b_const_i32(b, 0x0F);
    c8 = rocke_b_const_i32(b, 8);
    c16 = rocke_b_const_i32(b, 16);

    /* low_unsigned = b.land(byte_i32, mask_lo) */
    low_unsigned = rocke_b_land(b, byte_i32, mask_lo);

    /* high_unsigned = b.land(b.lshr(byte_i32, b.const_i32(4)), mask_lo).
     * The const_i32(4) + lshr are emitted here, AFTER low_unsigned, exactly as
     * the Python evaluates the high_unsigned right-hand side. */
    high_unsigned = rocke_b_land(b, rocke_b_lshr(b, byte_i32, rocke_b_const_i32(b, 4)), mask_lo);

    /* low_signed = b.select(b.cmp_ge(low_unsigned, c8),
     *                       b.sub(low_unsigned, c16), low_unsigned)
     * Python evaluates the select's args left-to-right: the cmp_ge (predicate)
     * is emitted BEFORE the sub (true-value). C leaves function-argument
     * evaluation order unspecified (clang/gcc emit right-to-left), which would
     * emit the sub first and renumber the SSA values. Bind each sub-expression
     * to an explicit temporary in Python order to pin the emission sequence. */
    {
        rocke_value_t* ge_low = rocke_b_cmp_ge(b, low_unsigned, c8);
        rocke_value_t* sub_low = rocke_b_sub(b, low_unsigned, c16);
        low_signed = rocke_b_select(b, ge_low, sub_low, low_unsigned);
    }

    /* high_signed = b.select(b.cmp_ge(high_unsigned, c8),
     *                        b.sub(high_unsigned, c16), high_unsigned)
     * Same left-to-right pinning as low_signed above. */
    {
        rocke_value_t* ge_high = rocke_b_cmp_ge(b, high_unsigned, c8);
        rocke_value_t* sub_high = rocke_b_sub(b, high_unsigned, c16);
        high_signed = rocke_b_select(b, ge_high, sub_high, high_unsigned);
    }

    if(!rocke_i_live(b))
    {
        return b->status; /* a mid-chain op recorded an error */
    }
    if(out_low != NULL)
    {
        *out_low = low_signed;
    }
    if(out_high != NULL)
    {
        *out_high = high_signed;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------ unpack_i4_byte_to_pair_f32 */

rocke_status_t rocke_unpack_i4_byte_to_pair_f32(rocke_ir_builder_t* b,
                                                rocke_value_t* packed_byte,
                                                rocke_value_t** out_low,
                                                rocke_value_t** out_high)
{
    rocke_value_t* low_i32;
    rocke_value_t* high_i32;
    rocke_value_t* low_f32 = NULL;
    rocke_value_t* high_f32 = NULL;
    rocke_status_t st;

    if(!rocke_i_live(b))
    {
        return b != NULL ? b->status : ROCKE_ERR_VALUE;
    }

    /* low_i32, high_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte) */
    st = rocke_unpack_i4_byte_to_pair_i32(b, packed_byte, &low_i32, &high_i32);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* return b.sitofp_f32(low_i32), b.sitofp_f32(high_i32) */
    low_f32 = rocke_b_sitofp_f32(b, low_i32);
    high_f32 = rocke_b_sitofp_f32(b, high_i32);

    if(!rocke_i_live(b))
    {
        return b->status;
    }
    if(out_low != NULL)
    {
        *out_low = low_f32;
    }
    if(out_high != NULL)
    {
        *out_high = high_f32;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------ unpack_i4_byte_to_pair_f16 */

rocke_status_t rocke_unpack_i4_byte_to_pair_f16(rocke_ir_builder_t* b,
                                                rocke_value_t* packed_byte,
                                                rocke_value_t** out_low,
                                                rocke_value_t** out_high)
{
    rocke_value_t* low_f32 = NULL;
    rocke_value_t* high_f32 = NULL;
    rocke_value_t* low_f16;
    rocke_value_t* high_f16;
    rocke_status_t st;

    if(!rocke_i_live(b))
    {
        return b != NULL ? b->status : ROCKE_ERR_VALUE;
    }

    /* low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte) */
    st = rocke_unpack_i4_byte_to_pair_f32(b, packed_byte, &low_f32, &high_f32);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* return b.trunc_f32_to_f16(low_f32), b.trunc_f32_to_f16(high_f32) */
    low_f16 = rocke_b_trunc_f32_to_f16(b, low_f32);
    high_f16 = rocke_b_trunc_f32_to_f16(b, high_f32);

    if(!rocke_i_live(b))
    {
        return b->status;
    }
    if(out_low != NULL)
    {
        *out_low = low_f16;
    }
    if(out_high != NULL)
    {
        *out_high = high_f16;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------ dequant_i4_byte_to_f16_pair */

rocke_status_t rocke_dequant_i4_byte_to_f16_pair(rocke_ir_builder_t* b,
                                                 rocke_value_t* packed_byte,
                                                 rocke_value_t* scale,
                                                 rocke_value_t** out_low,
                                                 rocke_value_t** out_high)
{
    rocke_value_t* low_f32 = NULL;
    rocke_value_t* high_f32 = NULL;
    rocke_value_t* low_f16;
    rocke_value_t* high_f16;
    rocke_status_t st;

    if(!rocke_i_live(b))
    {
        return b != NULL ? b->status : ROCKE_ERR_VALUE;
    }

    /* low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte) */
    st = rocke_unpack_i4_byte_to_pair_f32(b, packed_byte, &low_f32, &high_f32);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* low_f16 = b.trunc_f32_to_f16(b.fmul(low_f32, scale)).
     * The fmul is emitted immediately before its trunc -- Python fully evaluates
     * the low lane before touching the high lane. */
    low_f16 = rocke_b_trunc_f32_to_f16(b, rocke_b_fmul(b, low_f32, scale));

    /* high_f16 = b.trunc_f32_to_f16(b.fmul(high_f32, scale)) */
    high_f16 = rocke_b_trunc_f32_to_f16(b, rocke_b_fmul(b, high_f32, scale));

    if(!rocke_i_live(b))
    {
        return b->status;
    }
    if(out_low != NULL)
    {
        *out_low = low_f16;
    }
    if(out_high != NULL)
    {
        *out_high = high_f16;
    }
    return ROCKE_OK;
}
