// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.helpers.rotary.c -- C99 port of ck_dsl.helpers.rotary
 * (RotarySpec, pair_indices, load_cos_sin, apply_rotary_pair_f32).
 *
 * The builder-call sequences inside load_cos_sin / apply_rotary_pair_f32 are
 * reproduced in the exact same order as the Python so the emitted IR (and SSA
 * value numbering) stays byte-identical. Where a Python expression nests
 * several builder calls, C argument-evaluation order is unspecified, so the
 * sub-emissions are sequenced explicitly to match Python's strict
 * left-to-right evaluation of each call's arguments.
 *
 * See the header for the Python originals reproduced verbatim above each port.
 */

#include "ckc/helper_ck_dsl.helpers.rotary.h"

#include "ckc/ir_internal.h" /* ckc_i_live, ckc_i_set_err, ckc_i_type_is */

/* ------------------------------------------------------------------ *
 * RotaryLayout / RotarySpec
 * ------------------------------------------------------------------ */

const char* ckc_rotary_layout_name(ckc_rotary_layout_t layout)
{
    switch(layout)
    {
    case CKC_ROTARY_INTERLEAVED: return "interleaved";
    case CKC_ROTARY_HALF: return "half";
    default: return NULL;
    }
}

ckc_status_t ckc_rotary_spec_init(ckc_rotary_spec_t* out,
                                  int head_size,
                                  ckc_rotary_layout_t layout,
                                  int table_stride_pos)
{
    if(!out)
    {
        return CKC_ERR_VALUE;
    }

    /* if self.head_size <= 0 or self.head_size % 2 != 0: raise ValueError(...) */
    if(head_size <= 0 || head_size % 2 != 0)
    {
        return CKC_ERR_VALUE;
    }
    /* if self.layout not in ("interleaved", "half"): raise ValueError(...) */
    if(layout != CKC_ROTARY_INTERLEAVED && layout != CKC_ROTARY_HALF)
    {
        return CKC_ERR_VALUE;
    }

    out->head_size        = head_size;
    out->layout           = layout;
    out->table_stride_pos = table_stride_pos;
    return CKC_OK;
}

int ckc_rotary_spec_pair_count(const ckc_rotary_spec_t* spec)
{
    /* return self.head_size // 2 */
    return spec->head_size / 2;
}

int ckc_rotary_spec_stride_pos(const ckc_rotary_spec_t* spec)
{
    /* return self.table_stride_pos or self.pair_count */
    if(spec->table_stride_pos)
    {
        return spec->table_stride_pos;
    }
    return ckc_rotary_spec_pair_count(spec);
}

/* ------------------------------------------------------------------ *
 * pair_indices
 * ------------------------------------------------------------------ */

ckc_status_t
ckc_rotary_pair_indices(const ckc_rotary_spec_t* spec, int pair_idx, int* out_lo, int* out_hi)
{
    int pair_count;

    if(!spec || !out_lo || !out_hi)
    {
        return CKC_ERR_VALUE;
    }

    pair_count = ckc_rotary_spec_pair_count(spec);
    /* if pair_idx < 0 or pair_idx >= spec.pair_count: raise ValueError(...) */
    if(pair_idx < 0 || pair_idx >= pair_count)
    {
        return CKC_ERR_VALUE;
    }

    if(spec->layout == CKC_ROTARY_INTERLEAVED)
    {
        /* return (2 * pair_idx, 2 * pair_idx + 1) */
        *out_lo = 2 * pair_idx;
        *out_hi = 2 * pair_idx + 1;
        return CKC_OK;
    }
    /* return (pair_idx, pair_idx + spec.pair_count) */
    *out_lo = pair_idx;
    *out_hi = pair_idx + pair_count;
    return CKC_OK;
}

/* ------------------------------------------------------------------ *
 * load_cos_sin
 * ------------------------------------------------------------------ */

ckc_status_t ckc_rotary_load_cos_sin(ckc_ir_builder_t* b,
                                     ckc_value_t* cos_table,
                                     ckc_value_t* sin_table,
                                     ckc_value_t* token_pos,
                                     ckc_value_t* pair_idx,
                                     const ckc_rotary_spec_t* spec,
                                     ckc_value_t** out_cos,
                                     ckc_value_t** out_sin)
{
    ckc_value_t* stride_c;
    ckc_value_t* scaled;
    ckc_value_t* offset;
    ckc_value_t* cos_v;
    ckc_value_t* sin_v;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!ckc_i_live(b))
    {
        return ckc_ir_builder_status(b);
    }
    if(!cos_table || !sin_table || !token_pos || !pair_idx || !spec || !out_cos || !out_sin)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "load_cos_sin: null operand");
        return CKC_ERR_VALUE;
    }

    /* if cos_table.type != sin_table.type:
     *     raise ValueError("cos / sin tables must have matching pointer type") */
    if(!ckc_type_eq(cos_table->type, sin_table->type))
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "cos / sin tables must have matching pointer type");
        return CKC_ERR_VALUE;
    }
    /* if not isinstance(cos_table.type, PtrType):
     *     raise ValueError("cos_table must be a pointer") */
    if(!cos_table->type || cos_table->type->kind != CKC_TYPE_PTR)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "cos_table must be a pointer");
        return CKC_ERR_VALUE;
    }
    /* if cos_table.type.pointee != F32:
     *     raise ValueError("rotary tables must be ptr<f32> in v1") */
    if(!ckc_type_eq(cos_table->type->pointee, ckc_f32()))
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "rotary tables must be ptr<f32> in v1");
        return CKC_ERR_VALUE;
    }

    /* offset = b.add(b.mul(token_pos, b.const_i32(spec.stride_pos)), pair_idx)
     *
     * Python evaluates add's first argument (the mul chain) before its second
     * (pair_idx, already bound). Inside the mul, const_i32 is emitted, then the
     * mul, then the add -- sequenced explicitly for byte-identical numbering. */
    stride_c = ckc_b_const_i32(b, (int64_t)ckc_rotary_spec_stride_pos(spec));
    scaled   = ckc_b_mul(b, token_pos, stride_c);
    offset   = ckc_b_add(b, scaled, pair_idx);

    /* cos_v = b.global_load_f32(cos_table, offset)  (Python align default = 4) */
    cos_v = ckc_b_global_load_f32(b, cos_table, offset, /*align=*/0);
    /* sin_v = b.global_load_f32(sin_table, offset) */
    sin_v = ckc_b_global_load_f32(b, sin_table, offset, /*align=*/0);

    /* return cos_v, sin_v */
    *out_cos = cos_v;
    *out_sin = sin_v;
    return ckc_ir_builder_status(b);
}

/* ------------------------------------------------------------------ *
 * apply_rotary_pair_f32
 * ------------------------------------------------------------------ */

ckc_status_t ckc_rotary_apply_pair_f32(ckc_ir_builder_t* b,
                                       ckc_value_t* lo,
                                       ckc_value_t* hi,
                                       ckc_value_t* cos_t,
                                       ckc_value_t* sin_t,
                                       ckc_value_t** out_lo,
                                       ckc_value_t** out_hi)
{
    ckc_value_t* new_lo;
    ckc_value_t* new_hi;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!ckc_i_live(b))
    {
        return ckc_ir_builder_status(b);
    }
    if(!lo || !hi || !cos_t || !sin_t || !out_lo || !out_hi)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_rotary_pair_f32: null operand");
        return CKC_ERR_VALUE;
    }

    /* if lo.type.name != "f32" or hi.type.name != "f32":
     *     raise ValueError("apply_rotary_pair_f32 expects f32 inputs") */
    if(!ckc_i_type_is(lo->type, "f32") || !ckc_i_type_is(hi->type, "f32"))
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_rotary_pair_f32 expects f32 inputs");
        return CKC_ERR_VALUE;
    }
    /* if cos_t.type.name != "f32" or sin_t.type.name != "f32":
     *     raise ValueError("apply_rotary_pair_f32 expects f32 cos / sin") */
    if(!ckc_i_type_is(cos_t->type, "f32") || !ckc_i_type_is(sin_t->type, "f32"))
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_rotary_pair_f32 expects f32 cos / sin");
        return CKC_ERR_VALUE;
    }

    /* new_lo = b.fsub(b.fmul(lo, cos_t), b.fmul(hi, sin_t))
     *
     * Python evaluates fsub's two arguments left-to-right: fmul(lo, cos_t) is
     * emitted before fmul(hi, sin_t), then the fsub. Sequenced to keep value
     * numbering byte-identical. */
    {
        ckc_value_t* lc = ckc_b_fmul(b, lo, cos_t);
        ckc_value_t* hs = ckc_b_fmul(b, hi, sin_t);
        new_lo          = ckc_b_fsub(b, lc, hs);
    }
    /* new_hi = b.fadd(b.fmul(lo, sin_t), b.fmul(hi, cos_t)) */
    {
        ckc_value_t* ls = ckc_b_fmul(b, lo, sin_t);
        ckc_value_t* hc = ckc_b_fmul(b, hi, cos_t);
        new_hi          = ckc_b_fadd(b, ls, hc);
    }

    /* return new_lo, new_hi */
    *out_lo = new_lo;
    *out_hi = new_hi;
    return ckc_ir_builder_status(b);
}
