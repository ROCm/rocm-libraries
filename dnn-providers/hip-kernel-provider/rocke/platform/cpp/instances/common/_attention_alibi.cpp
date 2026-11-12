// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of alibi_slope() from library/kernels/common/_attention_alibi.py.
 *
 * See rocke/helper_rocke.instances.common._attention_alibi.h for the mapping.
 */
#include "rocke/helper_rocke.instances.common._attention_alibi.h"

/* _alibi_builtin_slope_exponent(b, head, num_heads) for a runtime num_heads: log2(slope(head)).
 * floor(log2(H)) is the exponent field of float(H) (exact below 2**24); -8 / n = -(2 ** (3 - e))
 * is rebuilt from the same field. Each op is its own statement so the emission order matches the
 * Python helper (C leaves argument evaluation order unspecified). */
static rocke_value_t* alibi_builtin_slope_exponent(rocke_ir_builder_t* b,
                                                   rocke_value_t* head,
                                                   rocke_value_t* num_heads)
{
    rocke_value_t* heads_f = rocke_b_sitofp_f32(b, num_heads);
    rocke_value_t* heads_bits = rocke_b_bitcast(b, heads_f, rocke_i32());
    rocke_value_t* c23 = rocke_b_const_i32(b, 23);
    rocke_value_t* biased_e = rocke_b_lshr(b, heads_bits, c23);
    rocke_value_t* c127 = rocke_b_const_i32(b, 127);
    rocke_value_t* e = rocke_b_sub(b, biased_e, c127);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* n_v = rocke_b_shl(b, one, e);
    rocke_value_t* c130 = rocke_b_const_i32(b, 127 + 3);
    rocke_value_t* inv8_e = rocke_b_sub(b, c130, e);
    rocke_value_t* c23b = rocke_b_const_i32(b, 23);
    rocke_value_t* inv8_bits = rocke_b_shl(b, inv8_e, c23b);
    rocke_value_t* inv8 = rocke_b_bitcast(b, inv8_bits, rocke_f32());
    rocke_value_t* step_v = rocke_b_fneg(b, inv8);
    /* step * (0.5 - n) = step / 2 + 8: n * step = -8 exactly. */
    rocke_value_t* half = rocke_b_const_f32(b, 0.5);
    rocke_value_t* eight = rocke_b_const_f32(b, 8.0);
    rocke_value_t* tail_off = rocke_b_fma(b, step_v, half, eight);
    /* h < n: step * (h + 1); otherwise step * (h - n + 0.5). */
    rocke_value_t* below_n = rocke_b_cmp_lt(b, head, n_v);
    rocke_value_t* off = rocke_b_select(b, below_n, step_v, tail_off);
    rocke_value_t* head_f = rocke_b_sitofp_f32(b, head);
    return rocke_b_fma(b, head_f, step_v, off);
}

rocke_value_t* rocke_attn_alibi_slope(rocke_ir_builder_t* b,
                                      rocke_value_t* head,
                                      rocke_value_t* num_heads,
                                      rocke_value_t* slopes)
{
    rocke_value_t* addr = rocke_b_global_ptr_to_i64(b, slopes);
    rocke_value_t* has_slopes = rocke_b_cmp_ne(b, addr, rocke_b_const_i64(b, 0));
    const rocke_type_t* result_types[1] = {rocke_f32()};
    rocke_if_else_t branch = rocke_b_scf_if_else_results(b, has_slopes, result_types, 1);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, branch.then_region);
    {
        rocke_value_t* num_bytes = rocke_b_mul(b, num_heads, rocke_b_const_i32(b, 4));
        rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, slopes, num_bytes);
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
