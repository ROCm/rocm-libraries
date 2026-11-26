// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Tiled attention family internals. Mirrors library/kernels/common/
 * _attention_additive_bias.py: the four trailing kernel params and the masked
 * load-and-add of the fp32 bias at
 * seq*batch_stride + q_head*head_stride + q_row*sq_stride + col
 * (a stride of 0 broadcasts that dimension). Not a public caller-owned spec ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_ADDITIVE_BIAS_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_ADDITIVE_BIAS_INTERNAL_H

#include <stdbool.h>
#include <string.h>

#include "rocke/ir.h"

/* BIAS_FLOOR and log2 e of _attention_additive_bias.py. */
#define ROCKE_ADDITIVE_BIAS_FLOOR (-158456325028528675187087900672.0) /* -(2**97) */
#define ROCKE_ADDITIVE_BIAS_LOG2E 1.4426950408889634

/* The BiasFloor constants: floor * log2e / 4, log2e / 4, 4, 0 (make_bias_floor). */
typedef struct rocke_additive_bias_floor
{
    rocke_value_t* floor_q;
    rocke_value_t* rcp_ln2_q;
    rocke_value_t* four;
    rocke_value_t* zero;
} rocke_additive_bias_floor_t;

typedef struct rocke_additive_bias_params
{
    rocke_value_t* ptr;
    rocke_value_t* batch_stride;
    rocke_value_t* head_stride;
    rocke_value_t* sq_stride;
    bool floored;
    rocke_additive_bias_floor_t floor;
} rocke_additive_bias_params_t;

/* Appended after all other params, in this order; must match the host launcher
 * signature (append_additive_bias_signature). ``floored`` also emits the floor
 * constants, as declare_additive_bias_params(floored=True) does. */
static inline rocke_additive_bias_params_t rocke_declare_additive_bias_params(rocke_ir_builder_t* b,
                                                                              bool floored)
{
    rocke_additive_bias_params_t p;
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    p.ptr = rocke_b_param(b, "additive_bias_ptr", rocke_ptr_type(b, rocke_f32(), "global"), &o);
    p.batch_stride = rocke_b_param(b, "additive_bias_batch_stride", rocke_i32(), NULL);
    p.head_stride = rocke_b_param(b, "additive_bias_head_stride", rocke_i32(), NULL);
    p.sq_stride = rocke_b_param(b, "additive_bias_sq_stride", rocke_i32(), NULL);
    p.floored = floored;
    memset(&p.floor, 0, sizeof(p.floor));
    if(floored)
    {
        p.floor.floor_q
            = rocke_b_const_f32(b, ROCKE_ADDITIVE_BIAS_FLOOR * ROCKE_ADDITIVE_BIAS_LOG2E / 4);
        p.floor.rcp_ln2_q = rocke_b_const_f32(b, ROCKE_ADDITIVE_BIAS_LOG2E / 4);
        p.floor.four = rocke_b_const_f32(b, 4.0);
        p.floor.zero = rocke_b_const_f32(b, 0.0);
    }
    return p;
}

/* score + bias[idx] / ln2, floored when ``p`` is (_floored_term: a finite bias is
 * raised to the floor; -inf, +inf and NaN pass through). Where ``mask`` is false
 * (padding rows, tail columns) the lane loads element 0 instead of an out-of-range
 * index and adds 0. */
static inline rocke_value_t* rocke_add_additive_bias(rocke_ir_builder_t* b,
                                                     rocke_value_t* score,
                                                     const rocke_additive_bias_params_t* p,
                                                     rocke_value_t* seq_idx,
                                                     rocke_value_t* q_head,
                                                     rocke_value_t* q_row,
                                                     rocke_value_t* col,
                                                     rocke_value_t* mask,
                                                     rocke_value_t* rcp_ln2)
{
    /* One op per statement: the emission order must match the Python
     * left-to-right evaluation. */
    rocke_value_t* batch_off = rocke_b_mul(b, p->batch_stride, seq_idx);
    rocke_value_t* head_off = rocke_b_mul(b, p->head_stride, q_head);
    rocke_value_t* base = rocke_b_add(b, batch_off, head_off);
    rocke_value_t* row_off = rocke_b_mul(b, p->sq_stride, q_row);
    rocke_value_t* row = rocke_b_add(b, base, row_off);
    rocke_value_t* idx = rocke_b_add(b, row, col);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* v = rocke_b_masked_global_load(b, p->ptr, idx, mask, zero, rocke_f32(), 4);
    if(p->floored)
    {
        rocke_value_t* vq = rocke_b_fmul(b, v, p->floor.rcp_ln2_q);
        rocke_value_t* guard = rocke_b_fma(b, v, p->floor.zero, p->floor.floor_q);
        rocke_value_t* raised = rocke_b_fmax(b, vq, guard);
        return rocke_b_fma(b, raised, p->floor.four, score);
    }
    rocke_value_t* scaled = rocke_b_fmul(b, v, rcp_ln2);
    return rocke_b_fadd(b, score, scaled);
}
#endif
