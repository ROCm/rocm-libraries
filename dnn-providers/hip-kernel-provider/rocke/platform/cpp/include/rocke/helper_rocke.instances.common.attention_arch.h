/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.attention_arch.h -- C port of the tiled
 * attention arch gate in library/kernels/common/attention_arch.py:
 *
 *   Python                                   C (this header)
 *   ---------------------------------------  -------------------------------------
 *   validate_tiled_attention_arch(arch)      rocke_validate_tiled_attention_arch()
 *   require_tiled_attention_arch(arch)       rocke_require_tiled_attention_arch()
 *
 * Reason strings are byte-identical to Python's, including str(KeyError) for an
 * unknown arch.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_ARCH_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_ARCH_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_ir_builder_t */

#ifdef __cplusplus
extern "C" {
#endif

/* (ok, reason): returns ok and writes reason ("ok" on accept) into out
 * (NUL-terminated, truncated to out_cap). out may be NULL with out_cap 0. */
bool rocke_validate_tiled_attention_arch(const char* arch, char* out, size_t out_cap);

/* On reject latches ROCKE_ERR_NOTIMPL with the reason on b (unless b already
 * carries an error) and returns false. */
bool rocke_require_tiled_attention_arch(rocke_ir_builder_t* b, const char* arch);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_ATTENTION_ARCH_H */
