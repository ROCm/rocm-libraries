// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of lse_store in library/kernels/common/_lse_store.py. See
 * rocke/helper_rocke.instances.common._lse_store.h for the symbol map.
 */
#include "rocke/helper_rocke.instances.common._lse_store.h"

#include <math.h>

#define ROCKE_LSE_LN2 0.6931471805599453

rocke_value_t* rocke_natural_lse_from_log2_stats(rocke_ir_builder_t* b,
                                                 rocke_value_t* m_log2,
                                                 rocke_value_t* row_sum)
{
    rocke_value_t* sum = rocke_b_fadd(b, m_log2, rocke_b_log2(b, row_sum));
    return rocke_b_fmul(b, sum, rocke_b_const_f32(b, ROCKE_LSE_LN2));
}

void rocke_lse_store(rocke_ir_builder_t* b,
                     rocke_value_t* dst,
                     rocke_value_t* element_offset,
                     rocke_value_t* m_log2,
                     rocke_value_t* l,
                     rocke_value_t* row_has_mass)
{
    rocke_value_t* safe_l = rocke_b_select(b, row_has_mass, l, rocke_b_const_f32(b, 1.0));
    rocke_value_t* safe_m = rocke_b_select(b, row_has_mass, m_log2, rocke_b_const_f32(b, 0.0));
    rocke_value_t* value = rocke_natural_lse_from_log2_stats(b, safe_m, safe_l);
    value = rocke_b_select(b, row_has_mass, value, rocke_b_const_f32(b, -INFINITY));
    rocke_b_global_store(b, dst, element_offset, value, 4);
}
