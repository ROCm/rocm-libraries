// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_F64_MMA_PROBE_H
#define ROCKE_INSTANCE_F64_MMA_PROBE_H
#include "rocke/ir.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Initializes b; returned kernel is owned by b. Mirrors F64MmaProbeSpec. */
rocke_kernel_def_t* rocke_build_f64_mma_probe(rocke_ir_builder_t* b, int m, int k);
#ifdef __cplusplus
}
#endif
#endif
