// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of unpack_i4_byte_to_pair_i32 / unpack_i4_byte_to_pair_f32 /
 * unpack_i4_byte_to_pair_f16 / dequant_i4_byte_to_f16_pair from
 * ck_dsl/helpers/i4_dequant.py. See helper_ck_dsl.helpers.i4_dequant.h for the
 * contract.
 *
 * Every symbol emits IR. The ckc_b_* op-emission order below is byte-faithful to
 * the Python source-line evaluation order so the lowered IR is identical.
 */

#include "ckc/helper_ck_dsl.helpers.i4_dequant.h"

#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */

/* ------------------------------------------------ unpack_i4_byte_to_pair_i32 */

ckc_status_t ckc_unpack_i4_byte_to_pair_i32(ckc_ir_builder_t* b,
                                            ckc_value_t* packed_byte,
                                            ckc_value_t** out_low,
                                            ckc_value_t** out_high)
{
    ckc_value_t* byte_i32;
    ckc_value_t* mask_lo;
    ckc_value_t* c8;
    ckc_value_t* c16;
    ckc_value_t* low_unsigned;
    ckc_value_t* high_unsigned;
    ckc_value_t* low_signed;
    ckc_value_t* high_signed;

    /* Sticky-error model: a failed builder makes every call a no-op. */
    if(!ckc_i_live(b))
    {
        return b != NULL ? b->status : CKC_ERR_VALUE;
    }

    /* Python: if packed_byte.type.name != "i8": raise ValueError(...) */
    if(packed_byte == NULL || packed_byte->type == NULL || packed_byte->type->name == NULL
       || strcmp(packed_byte->type->name, "i8") != 0)
    {
        const char* got
            = (packed_byte != NULL && packed_byte->type != NULL && packed_byte->type->name != NULL)
                  ? packed_byte->type->name
                  : "(null)";
        ckc_i_set_err(b, CKC_ERR_VALUE, "unpack_i4_byte_to_pair_i32 expects i8 input, got %s", got);
        return CKC_ERR_VALUE;
    }

    /* byte_i32 = b.sext(packed_byte, I32) */
    byte_i32 = ckc_b_sext(b, packed_byte, ckc_i32());

    /* mask_lo = b.const_i32(0x0F); c8 = b.const_i32(8); c16 = b.const_i32(16).
     * All three literals are bound before either land (matches Python order). */
    mask_lo = ckc_b_const_i32(b, 0x0F);
    c8 = ckc_b_const_i32(b, 8);
    c16 = ckc_b_const_i32(b, 16);

    /* low_unsigned = b.land(byte_i32, mask_lo) */
    low_unsigned = ckc_b_land(b, byte_i32, mask_lo);

    /* high_unsigned = b.land(b.lshr(byte_i32, b.const_i32(4)), mask_lo).
     * The const_i32(4) + lshr are emitted here, AFTER low_unsigned, exactly as
     * the Python evaluates the high_unsigned right-hand side. */
    high_unsigned = ckc_b_land(b, ckc_b_lshr(b, byte_i32, ckc_b_const_i32(b, 4)), mask_lo);

    /* low_signed = b.select(b.cmp_ge(low_unsigned, c8),
     *                       b.sub(low_unsigned, c16), low_unsigned)
     * Python evaluates the select's args left-to-right: the cmp_ge (predicate)
     * is emitted BEFORE the sub (true-value). C leaves function-argument
     * evaluation order unspecified (clang/gcc emit right-to-left), which would
     * emit the sub first and renumber the SSA values. Bind each sub-expression
     * to an explicit temporary in Python order to pin the emission sequence. */
    {
        ckc_value_t* ge_low = ckc_b_cmp_ge(b, low_unsigned, c8);
        ckc_value_t* sub_low = ckc_b_sub(b, low_unsigned, c16);
        low_signed = ckc_b_select(b, ge_low, sub_low, low_unsigned);
    }

    /* high_signed = b.select(b.cmp_ge(high_unsigned, c8),
     *                        b.sub(high_unsigned, c16), high_unsigned)
     * Same left-to-right pinning as low_signed above. */
    {
        ckc_value_t* ge_high = ckc_b_cmp_ge(b, high_unsigned, c8);
        ckc_value_t* sub_high = ckc_b_sub(b, high_unsigned, c16);
        high_signed = ckc_b_select(b, ge_high, sub_high, high_unsigned);
    }

    if(!ckc_i_live(b))
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
    return CKC_OK;
}

/* ------------------------------------------------ unpack_i4_byte_to_pair_f32 */

ckc_status_t ckc_unpack_i4_byte_to_pair_f32(ckc_ir_builder_t* b,
                                            ckc_value_t* packed_byte,
                                            ckc_value_t** out_low,
                                            ckc_value_t** out_high)
{
    ckc_value_t* low_i32;
    ckc_value_t* high_i32;
    ckc_value_t* low_f32 = NULL;
    ckc_value_t* high_f32 = NULL;
    ckc_status_t st;

    if(!ckc_i_live(b))
    {
        return b != NULL ? b->status : CKC_ERR_VALUE;
    }

    /* low_i32, high_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte) */
    st = ckc_unpack_i4_byte_to_pair_i32(b, packed_byte, &low_i32, &high_i32);
    if(st != CKC_OK)
    {
        return st;
    }

    /* return b.sitofp_f32(low_i32), b.sitofp_f32(high_i32) */
    low_f32 = ckc_b_sitofp_f32(b, low_i32);
    high_f32 = ckc_b_sitofp_f32(b, high_i32);

    if(!ckc_i_live(b))
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
    return CKC_OK;
}

/* ------------------------------------------------ unpack_i4_byte_to_pair_f16 */

ckc_status_t ckc_unpack_i4_byte_to_pair_f16(ckc_ir_builder_t* b,
                                            ckc_value_t* packed_byte,
                                            ckc_value_t** out_low,
                                            ckc_value_t** out_high)
{
    ckc_value_t* low_f32 = NULL;
    ckc_value_t* high_f32 = NULL;
    ckc_value_t* low_f16;
    ckc_value_t* high_f16;
    ckc_status_t st;

    if(!ckc_i_live(b))
    {
        return b != NULL ? b->status : CKC_ERR_VALUE;
    }

    /* low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte) */
    st = ckc_unpack_i4_byte_to_pair_f32(b, packed_byte, &low_f32, &high_f32);
    if(st != CKC_OK)
    {
        return st;
    }

    /* return b.trunc_f32_to_f16(low_f32), b.trunc_f32_to_f16(high_f32) */
    low_f16 = ckc_b_trunc_f32_to_f16(b, low_f32);
    high_f16 = ckc_b_trunc_f32_to_f16(b, high_f32);

    if(!ckc_i_live(b))
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
    return CKC_OK;
}

/* ------------------------------------------------ dequant_i4_byte_to_f16_pair */

ckc_status_t ckc_dequant_i4_byte_to_f16_pair(ckc_ir_builder_t* b,
                                             ckc_value_t* packed_byte,
                                             ckc_value_t* scale,
                                             ckc_value_t** out_low,
                                             ckc_value_t** out_high)
{
    ckc_value_t* low_f32 = NULL;
    ckc_value_t* high_f32 = NULL;
    ckc_value_t* low_f16;
    ckc_value_t* high_f16;
    ckc_status_t st;

    if(!ckc_i_live(b))
    {
        return b != NULL ? b->status : CKC_ERR_VALUE;
    }

    /* low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte) */
    st = ckc_unpack_i4_byte_to_pair_f32(b, packed_byte, &low_f32, &high_f32);
    if(st != CKC_OK)
    {
        return st;
    }

    /* low_f16 = b.trunc_f32_to_f16(b.fmul(low_f32, scale)).
     * The fmul is emitted immediately before its trunc -- Python fully evaluates
     * the low lane before touching the high lane. */
    low_f16 = ckc_b_trunc_f32_to_f16(b, ckc_b_fmul(b, low_f32, scale));

    /* high_f16 = b.trunc_f32_to_f16(b.fmul(high_f32, scale)) */
    high_f16 = ckc_b_trunc_f32_to_f16(b, ckc_b_fmul(b, high_f32, scale));

    if(!ckc_i_live(b))
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
    return CKC_OK;
}
