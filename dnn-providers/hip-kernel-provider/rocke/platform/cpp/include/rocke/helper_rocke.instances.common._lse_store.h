/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._lse_store.h -- C port of the LSE store
 * in library/kernels/common/_lse_store.py and its natural-log helper in
 * rocke/helpers/attention_fwd_ext.py:
 *
 *   Python                                   C (this header)
 *   ---------------------------------------  -------------------------------------
 *   natural_lse_from_log2_stats(b, m, l)     rocke_natural_lse_from_log2_stats()
 *   lse_store(b, dst, off, m, l, row_has_mass=...)
 *                                            rocke_lse_store()
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__LSE_STORE_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__LSE_STORE_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* LN2 * (m_log2 + log2(row_sum)); meaningless for invalid rows. */
rocke_value_t* rocke_natural_lse_from_log2_stats(rocke_ir_builder_t* b,
                                                 rocke_value_t* m_log2,
                                                 rocke_value_t* row_sum);

/* One FP32 store of the natural-log LSE (-inf where row_has_mass is false). The
 * caller guards row existence and writer ownership. */
void rocke_lse_store(rocke_ir_builder_t* b,
                     rocke_value_t* dst,
                     rocke_value_t* element_offset,
                     rocke_value_t* m_log2,
                     rocke_value_t* l,
                     rocke_value_t* row_has_mass);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__LSE_STORE_H */
