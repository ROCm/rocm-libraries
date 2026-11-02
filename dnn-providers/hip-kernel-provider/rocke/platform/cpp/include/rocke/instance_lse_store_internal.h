// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* LSE store internals. Mirrors library/kernels/common/_lse_store.py
 * (lse_store, lse_row_has_mass) and natural_lse_from_log2_stats from
 * rocke/helpers/attention_fwd_ext.py, op for op and in the same order, so the
 * two engines emit byte-identical IR. Every builder call is its own statement:
 * C leaves the evaluation order of call arguments unspecified, and the order
 * the ops are created in decides their SSA names. Not a public ABI. */
#ifndef ROCKE_INSTANCE_LSE_STORE_INTERNAL_H
#define ROCKE_INSTANCE_LSE_STORE_INTERNAL_H

#include <math.h>

#include "rocke/ir.h"

/* LN2 in attention_fwd_ext.py. */
#define ROCKE_LSE_LN2 0.6931471805599453

/* natural_lse_from_log2_stats: LN2 * (m_log2 + log2(row_sum)). */
static inline rocke_value_t* rocke_natural_lse_from_log2_stats(rocke_ir_builder_t* b,
                                                               rocke_value_t* m_log2,
                                                               rocke_value_t* row_sum)
{
    rocke_value_t* lg = rocke_b_log2(b, row_sum);
    rocke_value_t* sum = rocke_b_fadd(b, m_log2, lg);
    rocke_value_t* ln2 = rocke_b_const_f32(b, ROCKE_LSE_LN2);
    return rocke_b_fmul(b, sum, ln2);
}

/* lse_row_has_mass: !(l == 0) with NaN counted as mass, the exact inverse of
 * the tiled 2D O guard fcmp("oeq", l, 0). Built as one OR uno, not
 * lnot(oeq): the HIP lowerer lowers i1 NOT to ~bool, which is always true. */
static inline rocke_value_t* rocke_lse_row_has_mass(rocke_ir_builder_t* b, rocke_value_t* l)
{
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* ne = rocke_b_fcmp(b, "one", l, zero);
    rocke_value_t* nan = rocke_b_fcmp(b, "uno", l, zero);
    return rocke_b_lor(b, ne, nan);
}

/* lse_store: one FP32 store of the natural-log LSE; -inf when the row has no
 * mass. The caller guards row existence and writer ownership. */
static inline void rocke_lse_store(rocke_ir_builder_t* b,
                                   rocke_value_t* dst,
                                   rocke_value_t* element_offset,
                                   rocke_value_t* m_log2,
                                   rocke_value_t* l,
                                   rocke_value_t* row_has_mass)
{
    rocke_value_t* one = rocke_b_const_f32(b, 1.0);
    rocke_value_t* safe_l = rocke_b_select(b, row_has_mass, l, one);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* safe_m = rocke_b_select(b, row_has_mass, m_log2, zero);
    rocke_value_t* value = rocke_natural_lse_from_log2_stats(b, safe_m, safe_l);
    rocke_value_t* neg_inf = rocke_b_const_f32(b, -INFINITY);
    value = rocke_b_select(b, row_has_mass, value, neg_inf);
    rocke_b_global_store(b, dst, element_offset, value, 4);
}

#endif /* ROCKE_INSTANCE_LSE_STORE_INTERNAL_H */
