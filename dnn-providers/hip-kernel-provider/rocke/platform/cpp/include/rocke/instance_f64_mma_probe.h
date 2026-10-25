// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_F64_MMA_PROBE_H
#define ROCKE_INSTANCE_F64_MMA_PROBE_H
#include "rocke/ir.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Initializes b; returned kernel is owned by b. Mirrors
 * build_f64_mma_probe(F64MmaProbeSpec(m, k), arch): NULL (with b's error set)
 * when the arch's catalog has no f64 atom of that shape. */
rocke_kernel_def_t*
    rocke_build_f64_mma_probe_arch(rocke_ir_builder_t* b, int m, int k, const char* arch);
/* The gfx942 probe (build_f64_mma_probe's default arch). */
rocke_kernel_def_t* rocke_build_f64_mma_probe(rocke_ir_builder_t* b, int m, int k);
#ifdef __cplusplus
}
#endif
#endif
