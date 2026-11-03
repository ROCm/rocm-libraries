/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_alibi.h -- C port of
 * alibi_slope() from library/kernels/common/_attention_alibi.py, the ALiBi
 * slope of one query head as used by the unified tiled attention kernels.
 *
 *   Python (_attention_alibi.py)                    C (this header)
 *   ----------------------------------------------  ----------------------------
 *   alibi_slope(b, head, num_heads=H, slopes=p)     rocke_attn_alibi_slope(b, head, H, p)
 *
 * Only the baked (int) head count is ported: the dense kernels that pass a
 * runtime head count have no hand-written C++ twin. The emitted op sequence
 * (constants included) matches the Python helper exactly, so callers stay
 * byte-identical with their Python kernels.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_ALIBI_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_ALIBI_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* slope(head) as f32. ``slopes`` is the alibi_slopes_ptr kernel argument: a
 * non-null pointer gives slopes[head] (fp32, one per query head, 0 past
 * ``num_heads``), a null one hipDNN's built-in slope of ``head`` among
 * ``num_heads`` query heads (SdpaSemanticsRegister SEM-021), chosen at run time
 * without a branch. ``num_heads`` must be positive (ROCKE_ERR_VALUE + NULL
 * otherwise). */
rocke_value_t* rocke_attn_alibi_slope(rocke_ir_builder_t* b,
                                      rocke_value_t* head,
                                      int num_heads,
                                      rocke_value_t* slopes);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_ALIBI_H */
