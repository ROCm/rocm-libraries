// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_TF32_MMA_PROBE_INTERNAL_H
#define ROCKE_TF32_MMA_PROBE_INTERNAL_H

#include "rocke/arch_target.h"

namespace ckc
{
// Private implementation shared by the public builder and malformed-catalog tests.
// A null catalog selects gfx942. Otherwise borrows it for the duration of the call.
// Initializes b, which owns the kernel.
rocke_kernel_def_t* build_tf32_mma_probe(rocke_ir_builder_t* b,
                                         int m,
                                         const char* preparation,
                                         const rocke_mma_catalog_t* catalog);
}
#endif
