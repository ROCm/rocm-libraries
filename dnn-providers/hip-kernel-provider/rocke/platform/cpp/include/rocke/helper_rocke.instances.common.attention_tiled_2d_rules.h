/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.attention_tiled_2d_rules.h -- C99 port of
 * kernels/common/attention_tiled_2d_rules.py: the grouped-KV2 and early-V rules
 * the gfx950 tiled-2D spec runs (the gfx942 spec refuses these knobs outright).
 *
 *   Python (attention_tiled_2d_rules.py)  C99 (this header)
 *   -----------------------------------   ----------------------------------------------
 *   validate_grouped_kv2_and_early_v      rocke_attention_tiled_2d_validate_grouped_kv2_and_early_v
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_RULES_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_RULES_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h" /* spec struct */

#ifdef __cplusplus
extern "C" {
#endif

/* Python: kernels.common.attention_tiled_2d_rules.validate_grouped_kv2_and_early_v.
 * True when the spec's grouped-KV2 softmax and early-V schedule knobs are
 * supported together; otherwise false, with the Python ValueError text of the
 * first broken rule written to the caller-owned `reason` (rocke_spec_set_reason
 * rules: NUL-terminated, truncated to reason_cap). `spec` must not be NULL. */
bool rocke_attention_tiled_2d_validate_grouped_kv2_and_early_v(
    const rocke_attention_tiled_2d_spec_t* spec, char* reason, size_t reason_cap);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_RULES_H */
