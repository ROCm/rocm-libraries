// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors global_read.py builder calls in expression order. No IR rewriting. */
#include "rocke/helper_rocke.helpers.global_read.h"
#include "rocke/ir_internal.h"
#include <string.h>

static bool gr_is(const rocke_type_t* type, const char* name)
{
    return type && type->name && strcmp(type->name, name) == 0;
}

static int gr_bytes(const rocke_type_t* type, bool buffer)
{
    if(gr_is(type, "f16") || gr_is(type, "bf16"))
        return 2;
    if(gr_is(type, "f32") || gr_is(type, "i32"))
        return 4;
    if(buffer)
        return 0;
    if(gr_is(type, "f64") || gr_is(type, "i64"))
        return 8;
    if(gr_is(type, "i16"))
        return 2;
    if(gr_is(type, "tf32"))
        return 4;
    if(gr_is(type, "i8") || gr_is(type, "fp8e4m3") || gr_is(type, "bf8e5m2"))
        return 1;
    return 0;
}

static bool gr_uniform(rocke_ir_builder_t* b, const rocke_value_t* value, int depth, int* visits)
{
    static const char* const names[] = {"arith.add",
                                        "arith.sub",
                                        "arith.mul",
                                        "arith.div",
                                        "arith.mod",
                                        "arith.sext",
                                        "arith.zext",
                                        "arith.trunc",
                                        "arith.smin",
                                        "arith.smax",
                                        "arith.select",
                                        "arith.cmp",
                                        "arith.and",
                                        "arith.or",
                                        "arith.not",
                                        "tile.global_ptr_offset",
                                        "tile.global_ptr_add"};
    int i;
    const rocke_op_t* op;
    bool allowed = false;
    if(value == NULL || depth >= 64 || *visits == 0)
        return false;
    --*visits;
    op = value->op;
    if(op == NULL)
    {
        for(i = 0; i < b->num_param_lookup; ++i)
            if(value == b->param_values[i])
                return true;
        return false;
    }
    if(strcmp(op->name, "arith.constant") == 0 || strcmp(op->name, "gpu.block_id") == 0)
        return true;
    for(i = 0; i < (int)(sizeof(names) / sizeof(names[0])); ++i)
        if(strcmp(op->name, names[i]) == 0)
            allowed = true;
    if(!allowed)
        return false;
    for(i = 0; i < op->num_operands; ++i)
        if(!gr_uniform(b, op->operands[i], depth + 1, visits))
            return false;
    return true;
}

static bool gr_proven_uniform(rocke_ir_builder_t* b, const rocke_value_t* value)
{
    int visits = 256;
    return gr_uniform(b, value, 0, &visits);
}

static rocke_value_t* gr_wide(rocke_ir_builder_t* b, rocke_stride_t value)
{
    if(!value.is_value)
        return rocke_b_const_i64(b, value.imm);
    if(value.value && gr_is(value.value->type, "i64"))
        return value.value;
    if(value.value && gr_is(value.value->type, "i32"))
        return rocke_b_sext(b, value.value, rocke_i64());
    rocke_i_set_err(
        b, ROCKE_ERR_TYPE, "global reads require signed i32 or i64 coordinates and metadata");
    return NULL;
}

static bool gr_metadata(rocke_ir_builder_t* b, rocke_stride_t value)
{
    if(!value.is_value)
        return value.imm >= 0;
    return value.value && (gr_is(value.value->type, "i32") || gr_is(value.value->type, "i64"))
           && gr_proven_uniform(b, value.value);
}

static bool gr_can_buffer(rocke_ir_builder_t* b, const rocke_global_read_spec_t* spec)
{
    int i;
    if(!gr_bytes(spec->view->desc.dtype, true) || !gr_proven_uniform(b, spec->view->base))
        return false;
    for(i = 0; i < spec->rank; ++i)
        if(!gr_metadata(b, spec->extents[i]) || !gr_metadata(b, spec->view->desc.strides[i]))
            return false;
    return true;
}

/* select+cmp, not rocke_b_smin/smax: those lower as i32 only. */
static rocke_value_t* gr_smin(rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y)
{
    rocke_value_t* less = rocke_b_cmp_lt(b, x, y);
    return rocke_b_select(b, less, x, y);
}

static rocke_value_t* gr_smax(rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y)
{
    rocke_value_t* greater = rocke_b_cmp_gt(b, x, y);
    return rocke_b_select(b, greater, x, y);
}

/* Only i64 metadata can overflow a 64-bit product; i32 metadata is used as is. */
static rocke_value_t* gr_bounded(rocke_ir_builder_t* b,
                                 rocke_stride_t value,
                                 int64_t cap_int,
                                 rocke_value_t* cap,
                                 rocke_value_t* minus)
{
    if(!value.is_value)
        return rocke_b_const_i64(b, value.imm < cap_int ? value.imm : cap_int);
    rocke_value_t* wide = gr_wide(b, value);
    if(wide && gr_is(value.value->type, "i64"))
    {
        rocke_value_t* low = gr_smin(b, wide, cap);
        return gr_smax(b, low, minus);
    }
    return wide;
}

static void gr_span(rocke_ir_builder_t* b,
                    const rocke_global_read_spec_t* spec,
                    rocke_value_t** out_valid,
                    rocke_value_t** out_bytes)
{
    /* span = 1 + sum((extent - 1) * stride). Operands are i32 or clamped to
     * [-1, cap], so each product fits i64. A term above limit clears valid;
     * when every term is within it, the sum of at most 8 terms cannot wrap.
     * Empty or negative metadata clears valid (the full-width path). */
    int elem_bytes = gr_bytes(spec->view->desc.dtype, true);
    int64_t limit_int = INT32_MAX / elem_bytes;
    rocke_value_t* limit = rocke_b_const_i64(b, limit_int);
    rocke_value_t* cap = rocke_b_const_i64(b, limit_int + 1);
    rocke_value_t* minus = rocke_b_const_i64(b, -1);
    rocke_value_t* zero = rocke_b_const_i64(b, 0);
    rocke_value_t* one = rocke_b_const_i64(b, 1);
    rocke_value_t* span = one;
    rocke_value_t* valid = rocke_b_cmp_eq(b, zero, zero);
    int i;
    for(i = 0; i < spec->rank; ++i)
    {
        rocke_value_t* extent = gr_bounded(b, spec->extents[i], limit_int + 1, cap, minus);
        rocke_value_t* stride
            = gr_bounded(b, spec->view->desc.strides[i], limit_int + 1, cap, minus);
        rocke_value_t* test = rocke_b_cmp_ge(b, extent, one);
        valid = rocke_b_land(b, valid, test);
        test = rocke_b_cmp_ge(b, stride, zero);
        valid = rocke_b_land(b, valid, test);
        rocke_value_t* count = rocke_b_sub(b, extent, one);
        rocke_value_t* term = rocke_b_mul(b, count, stride);
        test = rocke_b_cmp_le(b, term, limit);
        valid = rocke_b_land(b, valid, test);
        span = rocke_b_add(b, span, term);
    }
    rocke_value_t* test = rocke_b_cmp_le(b, span, limit);
    valid = rocke_b_land(b, valid, test);
    rocke_value_t* scale = rocke_b_const_i64(b, elem_bytes);
    *out_valid = valid;
    *out_bytes = rocke_b_mul(b, span, scale);
}

rocke_status_t rocke_emit_global_read_paths(rocke_ir_builder_t* b,
                                            const rocke_global_read_spec_t* specs,
                                            int count,
                                            rocke_global_read_body_fn body_fn,
                                            void* user,
                                            const char* addressing)
{
    rocke_global_read_plan_t plans[ROCKE_GLOBAL_READ_MAX_INPUTS] = {0};
    rocke_value_t* conditions[ROCKE_GLOBAL_READ_MAX_INPUTS];
    rocke_value_t* spans[ROCKE_GLOBAL_READ_MAX_INPUTS];
    bool can_buffer = true;
    int i;
    if(!b)
        return ROCKE_ERR_VALUE;
    if(!addressing)
        addressing = "auto";
    if(strcmp(addressing, "auto") != 0 && strcmp(addressing, "global64") != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "global read addressing must be 'auto' or 'global64'");
        return b->status;
    }
    if(!specs || !body_fn || count < 1 || count > ROCKE_GLOBAL_READ_MAX_INPUTS)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "global reads require between 1 and 8 inputs");
        return b->status;
    }
    for(i = 0; i < count; ++i)
    {
        const rocke_global_read_spec_t* spec = &specs[i];
        if(!spec->view || !spec->extents || spec->view->addr_space != ROCKE_ADDR_GLOBAL
           || spec->rank != spec->view->desc.rank || spec->rank < 1
           || spec->rank > ROCKE_TV_MAX_RANK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "global read extent rank/view mismatch");
            return b->status;
        }
        if(!gr_bytes(spec->view->desc.dtype, false))
        {
            rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "global reads do not support dtype");
            return b->status;
        }
        plans[i].view = spec->view;
        plans[i].addressing = "global64";
        if(!gr_can_buffer(b, spec))
            can_buffer = false;
    }
    if(strcmp(addressing, "global64") == 0 || !can_buffer)
    {
        body_fn(b, plans, count, user);
        return b->status;
    }
    for(i = 0; i < count; ++i)
        gr_span(b, &specs[i], &conditions[i], &spans[i]);
    rocke_value_t* valid = conditions[0];
    for(i = 1; i < count; ++i)
        valid = rocke_b_land(b, valid, conditions[i]);
    rocke_if_else_t branch = rocke_b_scf_if_else(b, valid);
    rocke_b_region_enter(b, branch.then_region);
    for(i = 0; i < count; ++i)
    {
        rocke_value_t* size = rocke_b_trunc(b, spans[i], rocke_i32());
        rocke_tv_make_buffer_resource(b, &plans[i].buffer, specs[i].view->base, size, NULL);
        plans[i].addressing = "buffer";
    }
    body_fn(b, plans, count, user);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, branch.else_region);
    for(i = 0; i < count; ++i)
    {
        plans[i].addressing = "global64";
        memset(&plans[i].buffer, 0, sizeof(plans[i].buffer));
    }
    if(b->status == ROCKE_OK)
        body_fn(b, plans, count, user);
    rocke_b_region_leave(b);
    return b->status;
}

rocke_status_t rocke_global_read_bind(rocke_ir_builder_t* b,
                                      rocke_global_read_cursor_t* out_cursor,
                                      const rocke_global_read_plan_t* plan,
                                      rocke_value_t* const* indices,
                                      int rank)
{
    if(!b)
        return ROCKE_ERR_VALUE;
    if(!out_cursor || !plan || !plan->view)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid global read plan/cursor");
        return b->status;
    }
    rocke_value_t* off = rocke_tensor_descriptor_offset_i64(b, &plan->view->desc, indices, rank);
    rocke_value_t* scale = rocke_b_const_i64(b, gr_bytes(plan->view->desc.dtype, false));
    rocke_value_t* byte_off = rocke_b_mul(b, off, scale);
    out_cursor->plan = plan;
    if(strcmp(plan->addressing, "buffer") == 0)
        out_cursor->base = rocke_b_trunc(b, byte_off, rocke_i32());
    else
        out_cursor->base = rocke_b_global_ptr_offset(b, plan->view->base, byte_off);
    return b->status;
}

static rocke_value_t* gr_narrow(rocke_ir_builder_t* b, rocke_value_t* value)
{
    if(value && gr_is(value->type, "i32"))
        return value;
    if(value && gr_is(value->type, "i64"))
        return rocke_b_trunc(b, value, rocke_i32());
    rocke_i_set_err(b, ROCKE_ERR_TYPE, "global read deltas require signed i32 or i64");
    return NULL;
}

static rocke_value_t* gr_offset(rocke_ir_builder_t* b,
                                const rocke_global_read_cursor_t* cursor,
                                rocke_value_t* const* deltas,
                                int rank)
{
    if(!cursor || !cursor->plan || !cursor->plan->view || !cursor->base || !deltas
       || rank != cursor->plan->view->desc.rank)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "global read delta rank/cursor mismatch");
        return NULL;
    }
    const rocke_tensor_descriptor_t* desc = &cursor->plan->view->desc;
    if(strcmp(cursor->plan->addressing, "buffer") != 0)
        return rocke_tensor_descriptor_offset_i64(b, desc, deltas, rank);
    rocke_value_t* off = NULL;
    int i;
    for(i = 0; i < rank; ++i)
    {
        rocke_value_t* delta = gr_narrow(b, deltas[i]);
        rocke_value_t* term;
        const rocke_stride_t* stride = &desc->strides[i];
        if(stride->is_value)
        {
            rocke_value_t* narrow = gr_narrow(b, stride->value);
            term = rocke_b_mul(b, delta, narrow);
        }
        else if(stride->imm == 1)
            term = delta;
        else
        {
            uint32_t bits = (uint32_t)stride->imm;
            int64_t imm = bits <= INT32_MAX ? (int64_t)bits : (int64_t)bits - INT64_C(4294967296);
            rocke_value_t* constant = rocke_b_const_i32(b, imm);
            term = rocke_b_mul(b, delta, constant);
        }
        off = off == NULL ? term : rocke_b_add(b, off, term);
    }
    rocke_value_t* scale = rocke_b_const_i32(b, gr_bytes(desc->dtype, false));
    rocke_value_t* byte_off = rocke_b_mul(b, off, scale);
    return rocke_b_add(b, cursor->base, byte_off);
}

rocke_value_t* rocke_global_read_load_vec(rocke_ir_builder_t* b,
                                          const rocke_global_read_cursor_t* cursor,
                                          rocke_value_t* const* deltas,
                                          int rank,
                                          int n)
{
    if(!b)
        return NULL;
    rocke_value_t* off = gr_offset(b, cursor, deltas, rank);
    if(!off)
        return NULL;
    const rocke_global_read_plan_t* plan = cursor->plan;
    if(strcmp(plan->addressing, "buffer") == 0)
        return rocke_b_buffer_load_vN(
            b, plan->buffer.rsrc, off, plan->buffer.soffset, plan->view->desc.dtype, n);
    return rocke_b_global_load_vN(b, cursor->base, off, plan->view->desc.dtype, n, 1);
}

rocke_value_t* rocke_global_read_load_scalar(rocke_ir_builder_t* b,
                                             const rocke_global_read_cursor_t* cursor,
                                             rocke_value_t* const* deltas,
                                             int rank)
{
    if(!b)
        return NULL;
    rocke_value_t* off = gr_offset(b, cursor, deltas, rank);
    if(!off)
        return NULL;
    const rocke_global_read_plan_t* plan = cursor->plan;
    if(strcmp(plan->addressing, "buffer") == 0)
        return rocke_b_buffer_load(
            b, plan->buffer.rsrc, off, plan->buffer.soffset, plan->view->desc.dtype);
    return rocke_b_global_load(b, cursor->base, off, plan->view->desc.dtype, 0);
}
