// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of alibi_slope() from library/kernels/common/_attention_alibi.py.
 *
 * See rocke/helper_rocke.instances.common._attention_alibi.h for the mapping.
 */
#include "rocke/helper_rocke.instances.common._attention_alibi.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* _alibi_builtin_slope_exponent(b, head, num_heads) for an int num_heads: log2(slope(head)). */
static rocke_value_t*
    alibi_builtin_slope_exponent(rocke_ir_builder_t* b, rocke_value_t* head, int num_heads)
{
    int n = 1;
    while(n <= num_heads / 2)
        n <<= 1;
    const double step = -8.0 / n; /* a power of two, exact in f32 */
    rocke_value_t* n_v = rocke_b_const_i32(b, n);
    rocke_value_t* step_v = rocke_b_const_f32(b, step);
    rocke_value_t* tail_off = rocke_b_const_f32(b, step * (0.5 - n));
    /* h < n: step * (h + 1); otherwise step * (h - n + 0.5). */
    rocke_value_t* below_n = rocke_b_cmp_lt(b, head, n_v);
    rocke_value_t* off = rocke_b_select(b, below_n, step_v, tail_off);
    rocke_value_t* head_f = rocke_b_sitofp_f32(b, head);
    return rocke_b_fma(b, head_f, step_v, off);
}

rocke_value_t* rocke_attn_alibi_slope(rocke_ir_builder_t* b,
                                      rocke_value_t* head,
                                      int num_heads,
                                      rocke_value_t* slopes)
{
    if(num_heads <= 0)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "num_heads must be positive, got %d", num_heads);
    rocke_value_t* addr = rocke_b_global_ptr_to_i64(b, slopes);
    rocke_value_t* has_slopes = rocke_b_cmp_ne(b, addr, rocke_b_const_i64(b, 0));
    const rocke_type_t* result_types[1] = {rocke_f32()};
    rocke_if_else_t branch = rocke_b_scf_if_else_results(b, has_slopes, result_types, 1);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, branch.then_region);
    {
        rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, slopes, rocke_b_const_i32(b, 4 * num_heads));
        rocke_value_t* voffset = rocke_b_mul(b, head, rocke_b_const_i32(b, 4));
        rocke_value_t* given
            = rocke_b_buffer_load(b, rsrc, voffset, rocke_b_const_i32(b, 0), rocke_f32());
        rocke_b_scf_yield(b, &given, 1);
    }
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, branch.else_region);
    {
        rocke_value_t* builtin = rocke_b_exp2(b, alibi_builtin_slope_exponent(b, head, num_heads));
        rocke_b_scf_yield(b, &builtin, 1);
    }
    rocke_b_region_leave(b);
    return branch.op->results[0];
}
