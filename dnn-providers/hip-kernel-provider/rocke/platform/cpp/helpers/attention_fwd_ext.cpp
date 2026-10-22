// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_fwd_ext.cpp -- C port of the selection of rocke/helpers/
 * attention_fwd_ext.py declared in rocke/helper_rocke.helpers.attention_fwd_ext.h.
 *
 * Operands are bound to temporaries in Python's left-to-right evaluation order:
 * C++ leaves argument evaluation order unspecified, and each builder call
 * consumes an SSA id.
 */
#include "rocke/helper_rocke.helpers.attention_fwd_ext.h"

rocke_value_t* rocke_natural_lse_from_log2_stats(rocke_ir_builder_t* b,
                                                 rocke_value_t* m_log2,
                                                 rocke_value_t* row_sum)
{
    if(b == NULL || !rocke_ir_builder_ok(b))
        return NULL;
    rocke_value_t* log2_sum = rocke_b_log2(b, row_sum);
    rocke_value_t* sum = rocke_b_fadd(b, m_log2, log2_sum);
    rocke_value_t* ln2 = rocke_b_const_f32(b, ROCKE_ATTN_FWD_EXT_LN2);
    return rocke_b_fmul(b, sum, ln2);
}
