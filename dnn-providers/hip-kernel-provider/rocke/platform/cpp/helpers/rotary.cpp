// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.rotary.c -- C99 port of rocke.helpers.rotary
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

#include "rocke/helper_rocke.helpers.rotary.h"

#include "rocke/ir_internal.h" /* rocke_i_live, rocke_i_set_err, rocke_i_type_is */

/* ------------------------------------------------------------------ *
 * RotaryLayout / RotarySpec
 * ------------------------------------------------------------------ */

const char* rocke_rotary_layout_name(rocke_rotary_layout_t layout)
{
    switch(layout)
    {
    case ROCKE_ROTARY_INTERLEAVED:
        return "interleaved";
    case ROCKE_ROTARY_HALF:
        return "half";
    default:
        return NULL;
    }
}

rocke_status_t rocke_rotary_spec_init(rocke_rotary_spec_t* out,
                                      int head_size,
                                      rocke_rotary_layout_t layout,
                                      int table_stride_pos)
{
    if(!out)
    {
        return ROCKE_ERR_VALUE;
    }

    /* if self.head_size <= 0 or self.head_size % 2 != 0: raise ValueError(...) */
    if(head_size <= 0 || head_size % 2 != 0)
    {
        return ROCKE_ERR_VALUE;
    }
    /* if self.layout not in ("interleaved", "half"): raise ValueError(...) */
    if(layout != ROCKE_ROTARY_INTERLEAVED && layout != ROCKE_ROTARY_HALF)
    {
        return ROCKE_ERR_VALUE;
    }

    out->head_size = head_size;
    out->layout = layout;
    out->table_stride_pos = table_stride_pos;
    return ROCKE_OK;
}

int rocke_rotary_spec_pair_count(const rocke_rotary_spec_t* spec)
{
    /* return self.head_size // 2 */
    return spec->head_size / 2;
}

int rocke_rotary_spec_stride_pos(const rocke_rotary_spec_t* spec)
{
    /* return self.table_stride_pos or self.pair_count */
    if(spec->table_stride_pos)
    {
        return spec->table_stride_pos;
    }
    return rocke_rotary_spec_pair_count(spec);
}

/* ------------------------------------------------------------------ *
 * pair_indices
 * ------------------------------------------------------------------ */

rocke_status_t rocke_rotary_pair_indices(const rocke_rotary_spec_t* spec,
                                         int pair_idx,
                                         int* out_lo,
                                         int* out_hi)
{
    int pair_count;

    if(!spec || !out_lo || !out_hi)
    {
        return ROCKE_ERR_VALUE;
    }

    pair_count = rocke_rotary_spec_pair_count(spec);
    /* if pair_idx < 0 or pair_idx >= spec.pair_count: raise ValueError(...) */
    if(pair_idx < 0 || pair_idx >= pair_count)
    {
        return ROCKE_ERR_VALUE;
    }

    if(spec->layout == ROCKE_ROTARY_INTERLEAVED)
    {
        /* return (2 * pair_idx, 2 * pair_idx + 1) */
        *out_lo = 2 * pair_idx;
        *out_hi = 2 * pair_idx + 1;
        return ROCKE_OK;
    }
    /* return (pair_idx, pair_idx + spec.pair_count) */
    *out_lo = pair_idx;
    *out_hi = pair_idx + pair_count;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * load_cos_sin
 * ------------------------------------------------------------------ */

rocke_status_t rocke_rotary_load_cos_sin(rocke_ir_builder_t* b,
                                         rocke_value_t* cos_table,
                                         rocke_value_t* sin_table,
                                         rocke_value_t* token_pos,
                                         rocke_value_t* pair_idx,
                                         const rocke_rotary_spec_t* spec,
                                         rocke_value_t** out_cos,
                                         rocke_value_t** out_sin)
{
    rocke_value_t* stride_c;
    rocke_value_t* scaled;
    rocke_value_t* offset;
    rocke_value_t* cos_v;
    rocke_value_t* sin_v;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return rocke_ir_builder_status(b);
    }
    if(!cos_table || !sin_table || !token_pos || !pair_idx || !spec || !out_cos || !out_sin)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "load_cos_sin: null operand");
        return ROCKE_ERR_VALUE;
    }

    /* if cos_table.type != sin_table.type:
     *     raise ValueError("cos / sin tables must have matching pointer type") */
    if(!rocke_type_eq(cos_table->type, sin_table->type))
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "cos / sin tables must have matching pointer type");
        return ROCKE_ERR_VALUE;
    }
    /* if not isinstance(cos_table.type, PtrType):
     *     raise ValueError("cos_table must be a pointer") */
    if(!cos_table->type || cos_table->type->kind != ROCKE_TYPE_PTR)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "cos_table must be a pointer");
        return ROCKE_ERR_VALUE;
    }
    /* if cos_table.type.pointee != F32:
     *     raise ValueError("rotary tables must be ptr<f32> in v1") */
    if(!rocke_type_eq(cos_table->type->pointee, rocke_f32()))
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "rotary tables must be ptr<f32> in v1");
        return ROCKE_ERR_VALUE;
    }

    /* offset = b.add(b.mul(token_pos, b.const_i32(spec.stride_pos)), pair_idx)
     *
     * Python evaluates add's first argument (the mul chain) before its second
     * (pair_idx, already bound). Inside the mul, const_i32 is emitted, then the
     * mul, then the add -- sequenced explicitly for byte-identical numbering. */
    stride_c = rocke_b_const_i32(b, (int64_t)rocke_rotary_spec_stride_pos(spec));
    scaled = rocke_b_mul(b, token_pos, stride_c);
    offset = rocke_b_add(b, scaled, pair_idx);

    /* cos_v = b.global_load_f32(cos_table, offset)  (Python align default = 4) */
    cos_v = rocke_b_global_load_f32(b, cos_table, offset, /*align=*/0);
    /* sin_v = b.global_load_f32(sin_table, offset) */
    sin_v = rocke_b_global_load_f32(b, sin_table, offset, /*align=*/0);

    /* return cos_v, sin_v */
    *out_cos = cos_v;
    *out_sin = sin_v;
    return rocke_ir_builder_status(b);
}

/* ------------------------------------------------------------------ *
 * apply_rotary_pair_f32
 * ------------------------------------------------------------------ */

rocke_status_t rocke_rotary_apply_pair_f32(rocke_ir_builder_t* b,
                                           rocke_value_t* lo,
                                           rocke_value_t* hi,
                                           rocke_value_t* cos_t,
                                           rocke_value_t* sin_t,
                                           rocke_value_t** out_lo,
                                           rocke_value_t** out_hi)
{
    rocke_value_t* new_lo;
    rocke_value_t* new_hi;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!rocke_i_live(b))
    {
        return rocke_ir_builder_status(b);
    }
    if(!lo || !hi || !cos_t || !sin_t || !out_lo || !out_hi)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "apply_rotary_pair_f32: null operand");
        return ROCKE_ERR_VALUE;
    }

    /* if lo.type.name != "f32" or hi.type.name != "f32":
     *     raise ValueError("apply_rotary_pair_f32 expects f32 inputs") */
    if(!rocke_i_type_is(lo->type, "f32") || !rocke_i_type_is(hi->type, "f32"))
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "apply_rotary_pair_f32 expects f32 inputs");
        return ROCKE_ERR_VALUE;
    }
    /* if cos_t.type.name != "f32" or sin_t.type.name != "f32":
     *     raise ValueError("apply_rotary_pair_f32 expects f32 cos / sin") */
    if(!rocke_i_type_is(cos_t->type, "f32") || !rocke_i_type_is(sin_t->type, "f32"))
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "apply_rotary_pair_f32 expects f32 cos / sin");
        return ROCKE_ERR_VALUE;
    }

    /* new_lo = b.fsub(b.fmul(lo, cos_t), b.fmul(hi, sin_t))
     *
     * Python evaluates fsub's two arguments left-to-right: fmul(lo, cos_t) is
     * emitted before fmul(hi, sin_t), then the fsub. Sequenced to keep value
     * numbering byte-identical. */
    {
        rocke_value_t* lc = rocke_b_fmul(b, lo, cos_t);
        rocke_value_t* hs = rocke_b_fmul(b, hi, sin_t);
        new_lo = rocke_b_fsub(b, lc, hs);
    }
    /* new_hi = b.fadd(b.fmul(lo, sin_t), b.fmul(hi, cos_t)) */
    {
        rocke_value_t* ls = rocke_b_fmul(b, lo, sin_t);
        rocke_value_t* hc = rocke_b_fmul(b, hi, cos_t);
        new_hi = rocke_b_fadd(b, ls, hc);
    }

    /* return new_lo, new_hi */
    *out_lo = new_lo;
    *out_hi = new_hi;
    return rocke_ir_builder_status(b);
}
