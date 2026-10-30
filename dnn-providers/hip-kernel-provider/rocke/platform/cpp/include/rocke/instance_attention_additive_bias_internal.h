// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Tiled attention family internals. Mirrors library/kernels/common/
 * _attention_additive_bias.py: the four trailing kernel params and the masked
 * load-and-add of the fp32 bias at
 * seq*batch_stride + q_head*head_stride + q_row*sq_stride + col
 * (a stride of 0 broadcasts that dimension). Not a public caller-owned spec ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_ADDITIVE_BIAS_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_ADDITIVE_BIAS_INTERNAL_H

#include <string.h>

#include "rocke/ir.h"

typedef struct rocke_additive_bias_params
{
    rocke_value_t* ptr;
    rocke_value_t* batch_stride;
    rocke_value_t* head_stride;
    rocke_value_t* sq_stride;
} rocke_additive_bias_params_t;

/* Appended after all other params, in this order; must match the host launcher
 * signature (append_additive_bias_signature). */
static inline rocke_additive_bias_params_t rocke_declare_additive_bias_params(rocke_ir_builder_t* b)
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
    return p;
}

/* score + bias[idx] / ln2. Where ``mask`` is false (padding rows, tail columns)
 * the lane loads element 0 instead of an out-of-range index and adds 0. */
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
    rocke_value_t* scaled = rocke_b_fmul(b, v, rcp_ln2);
    return rocke_b_fadd(b, score, scaled);
}
#endif
