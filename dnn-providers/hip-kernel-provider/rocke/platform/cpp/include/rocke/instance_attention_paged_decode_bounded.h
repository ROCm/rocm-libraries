// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_ATTENTION_PAGED_DECODE_BOUNDED_H
#define ROCKE_INSTANCE_ATTENTION_PAGED_DECODE_BOUNDED_H
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Versioned bounded T32 decode builders. Old spec POD layouts are unchanged.
 * Callers provide truthful allocation bounds, matching grids, and stream ordering.
 * Device metadata failures produce NaN for the affected sequence via the paired reducer. */
rocke_kernel_def_t* rocke_build_paged_decode_bounded_v1_gfx942(
    rocke_ir_builder_t*, const rocke_unified_attention_3d_tiled_spec_t*, const char*);
rocke_kernel_def_t* rocke_build_paged_decode_reduce_bounded_v1_gfx942(
    rocke_ir_builder_t*, const rocke_unified_attention_reduce_tiled_spec_t*, const char*);
rocke_kernel_def_t* rocke_build_paged_decode_bounded_v1_gfx950(
    rocke_ir_builder_t*, const rocke_unified_attention_3d_tiled_spec_t*, const char*);
rocke_kernel_def_t* rocke_build_paged_decode_reduce_bounded_v1_gfx950(
    rocke_ir_builder_t*, const rocke_unified_attention_reduce_tiled_spec_t*, const char*);
#ifdef __cplusplus
}
#endif
#endif
