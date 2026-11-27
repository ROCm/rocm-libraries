// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Private implementation of the native mirror of build_f64_mma_probe. */
#ifndef ROCKE_F64_MMA_PROBE_INTERNAL_H
#define ROCKE_F64_MMA_PROBE_INTERNAL_H

#include "rocke/arch_target.h"

namespace ckc
{
// A null catalog selects arch's MMA catalog. A non-null catalog replaces it and
// is borrowed for this call only; arch must still be a known target, and names
// it in a refusal. Initializes b, which owns the returned kernel.
rocke_kernel_def_t* build_f64_mma_probe(
    rocke_ir_builder_t* b, int m, int k, const char* arch, const rocke_mma_catalog_t* catalog);
} // namespace ckc
#endif
