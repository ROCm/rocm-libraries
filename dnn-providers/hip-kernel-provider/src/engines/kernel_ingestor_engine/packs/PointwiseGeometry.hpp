// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/PointwiseLaunchValues.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

// The block and grid a pointwise launch uses. Both are one-dimensional.
struct PointwiseLaunchGeometry
{
    unsigned int block = 0;
    unsigned int gridX = 0;
};

// One block of `blockSize` threads.
inline PointwiseLaunchGeometry pointwiseLaunchGeometry(const PointwiseLaunchInputs& launch)
{
    return PointwiseLaunchGeometry{static_cast<unsigned int>(launch.blockSize), 1};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
