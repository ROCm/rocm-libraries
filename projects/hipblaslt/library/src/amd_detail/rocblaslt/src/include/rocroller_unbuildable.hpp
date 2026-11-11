// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocroller_kernel_match.hpp"

#include <hip/library_types.h>

struct KernelType;

// One rocRoller GEMM configuration with 32-element E8M0 block scales on A and
// B: the transposes, the element types, the workgroup tile, and whether the
// kernel uses workgroup mapping (never with Stream-K) and Stream-K.
struct RocRollerConfigKey
{
    bool        transA;
    bool        transB;
    hipDataType typeA;
    hipDataType typeB;
    hipDataType typeC;
    hipDataType typeD;
    int         tileM;
    int         tileN;
    int         tileK;
    bool        workgroupMapping;
    bool        streamK;
};

// Is this a configuration rocRoller fails to build? Answered from
// tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def,
// which scripts/rocroller_unbuildable_configs.py writes with a rocRoller JIT
// build.
bool rocRollerConfigUnbuildable(const RocRollerConfigKey& config);

// Does unbuildable_configs.def record the configuration rocRoller builds for
// this ranked tile of this kernel type? The table holds plain 32-element E8M0
// block scales on A and B only (not pre-swizzled, pre-tiled or swizzled), so
// a tile of any other kernel type is never recorded.
bool rocRollerTileUnbuildable(const KernelType& kernelType, const RocRollerSelectedTile& tile);
