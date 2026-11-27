// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_INSTANCE_F64_MMA_PROBE_H
#define ROCKE_INSTANCE_F64_MMA_PROBE_H
#include "rocke/ir.h"
#ifdef __cplusplus
extern "C" {
#endif
/* build_f64_mma_probe(F64MmaProbeSpec(m, k), arch): the probe for the f64 MFMA
 * atom of shape m x m x 4 that arch's MMA catalog lists. There is no default
 * target: a NULL arch is refused like an unknown one. Initializes b; the
 * returned kernel is owned by b. Returns NULL with b's error set
 * (ROCKE_ERR_VALUE) for a spec or a target the probe refuses. */
rocke_kernel_def_t*
    rocke_build_f64_mma_probe(rocke_ir_builder_t* b, int m, int k, const char* arch);
#ifdef __cplusplus
}
#endif
#endif
