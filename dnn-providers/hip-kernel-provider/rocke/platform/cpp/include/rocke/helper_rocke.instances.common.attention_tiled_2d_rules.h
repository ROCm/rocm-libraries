/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.attention_tiled_2d_rules.h -- C99 port of
 * kernels/common/attention_tiled_2d_rules.py, the spec rules the gfx942 and gfx950
 * tiled-2D attention specs share.
 *
 *   Python (attention_tiled_2d_rules.py) -> C99 (this header)
 *   validate_transposed_sub_knobs
 *       -> rocke_attention_tiled_2d_validate_transposed_sub_knobs
 *   validate_skip_legacy_qreg
 *       -> rocke_attention_tiled_2d_validate_skip_legacy_qreg
 *   validate_transposed_softmax_valu_opts
 *       -> rocke_attention_tiled_2d_validate_transposed_softmax_valu_opts
 *   validate_grouped_kv2_and_early_v
 *       -> rocke_attention_tiled_2d_validate_grouped_kv2_and_early_v
 *
 * Both write the Python ValueError text of the first broken rule to the
 * caller-owned `reason` (rocke_spec_set_reason rules: NUL-terminated, truncated
 * to reason_cap) and return false; they return true when the spec passes.
 * `spec` must not be NULL.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_RULES_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_RULES_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.attention_tiled_2d_spec.h" /* rocke_attention_tiled_2d_spec_t */

#ifdef __cplusplus
extern "C" {
#endif

/* The transposed-softmax sub-knobs (scalar state, invariant hoist, mask-once,
 * half-local PV) each require use_transposed_qk_32x32. */
bool rocke_attention_tiled_2d_validate_transposed_sub_knobs(
    const rocke_attention_tiled_2d_spec_t* spec, char* reason, size_t reason_cap);

/* use_mfma32_skip_legacy_qreg requires use_mfma_32x32. */
bool rocke_attention_tiled_2d_validate_skip_legacy_qreg(const rocke_attention_tiled_2d_spec_t* spec,
                                                        char* reason,
                                                        size_t reason_cap);

/* The transposed-softmax VALU opts (use_transposed_mask_limit) run only on the
 * R4_s1mask path (the transposed QK with the scalar state and mask-once knobs),
 * without a sliding window, softcap, ALiBi, QQ bias or grouped-KV2 softmax. */
bool rocke_attention_tiled_2d_validate_transposed_softmax_valu_opts(
    const rocke_attention_tiled_2d_spec_t* spec, char* reason, size_t reason_cap);

/* Grouped-KV2 softmax and the early-V schedule; the gfx950 spec runs these (the
 * gfx942 spec refuses grouped-KV2 and the fp8 K/V cache outright). */
bool rocke_attention_tiled_2d_validate_grouped_kv2_and_early_v(
    const rocke_attention_tiled_2d_spec_t* spec, char* reason, size_t reason_cap);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_TILED_2D_RULES_H */
