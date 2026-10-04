// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_TF32_MMA_PROBE_H
#define ROCKE_INSTANCE_TF32_MMA_PROBE_H
#include "rocke/ir.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Initializes b; returned kernel is owned by b. Mirrors Tf32MmaProbeSpec. */
rocke_kernel_def_t*
    rocke_build_tf32_mma_probe(rocke_ir_builder_t* b, int m, const char* preparation);
#ifdef __cplusplus
}
#endif
#endif
