// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>

#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

// The block and grid a conv-forward launch uses. Both are one-dimensional.
struct ConvFwdLaunchGeometry
{
    unsigned int block = 0;
    unsigned int gridX = 0;
};

// One thread for each output element, in blocks of `blockSize` threads.
inline ConvFwdLaunchGeometry convFwdLaunchGeometry(const ConvFwdLaunchInputs& launch)
{
    const auto& extents = launch.extents;
    const auto blockSize = static_cast<unsigned int>(launch.blockSize);
    const auto p = extents.h - extents.r + 1;
    const auto q = extents.width - extents.s + 1;
    // int64_t: n*k*p*q can exceed 2^31 for shapes this matcher admits. A 32-bit
    // product would wrap, corrupting both the grid size and the kernel's own bounds
    // guard (ConvFwd.cpp).
    const int64_t total = static_cast<int64_t>(extents.n) * extents.k * p * q;
    const auto gridSize = static_cast<unsigned int>((total + static_cast<int64_t>(blockSize) - 1)
                                                    / static_cast<int64_t>(blockSize));
    return ConvFwdLaunchGeometry{blockSize, gridSize};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
