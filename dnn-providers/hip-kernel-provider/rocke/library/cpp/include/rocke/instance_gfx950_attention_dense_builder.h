/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Native mirror of kernels.gfx950.attention_dense.build_attention_dense.
 * Link ckc::rocke_library (which links ckc::rocke_core). No Python runtime.
 */
#ifndef ROCKE_INSTANCE_GFX950_ATTENTION_DENSE_BUILDER_H
#define ROCKE_INSTANCE_GFX950_ATTENTION_DENSE_BUILDER_H

#include "rocke/instance_gfx950_attention_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize fresh builder storage and emit the selected ordinary or persistent
 * kernel. arch == NULL selects gfx950. The returned kernel is owned by b's
 * arena. On refusal returns NULL; rocke_ir_builder_status/error report why.
 * For every non-NULL b, call rocke_ir_builder_free(b), also on failure.
 * Free an existing builder before reusing its storage here. */
rocke_kernel_def_t* rocke_gfx950_build_attention_dense_new(
    rocke_ir_builder_t* b, const rocke_gfx950_attention_dense_spec_t* spec, const char* arch);

#ifdef __cplusplus
}
#endif
#endif
