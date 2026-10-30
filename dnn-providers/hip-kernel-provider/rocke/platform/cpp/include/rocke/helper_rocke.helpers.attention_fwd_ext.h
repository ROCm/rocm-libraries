/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.attention_fwd_ext.h -- C port of a selection of
 * symbols from rocke/helpers/attention_fwd_ext.py.
 *
 *   Python (rocke/helpers/attention_fwd_ext.py)   C (this header)
 *   ------------------------------------------    ---------------------------------
 *   LN2                                           ROCKE_ATTN_FWD_EXT_LN2
 *   natural_lse_from_log2_stats(b, m_log2, l)     rocke_natural_lse_from_log2_stats
 *
 * The emitter reproduces the Python builder-call sequence op for op, so the IR
 * it builds is byte-identical to the Python engine's. Every node is
 * arena-owned. A dead builder is a no-op that returns NULL.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_FWD_EXT_H
#define ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_FWD_EXT_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* attention_fwd_ext.LN2: ln(2). */
#define ROCKE_ATTN_FWD_EXT_LN2 0.6931471805599453

/* natural_lse_from_log2_stats(b, m_log2, row_sum):
 *
 *   return b.fmul(b.fadd(m_log2, b.log2(row_sum)), b.const_f32(LN2))
 *
 * Natural-log LSE LN2 * (m_log2 + log2(row_sum)) from the log2-domain running
 * max and running sum. Meaningless for invalid rows; the caller selects -inf. */
rocke_value_t* rocke_natural_lse_from_log2_stats(rocke_ir_builder_t* b,
                                                 rocke_value_t* m_log2,
                                                 rocke_value_t* row_sum);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_FWD_EXT_H */
