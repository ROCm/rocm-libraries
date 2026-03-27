// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>

namespace hip_kernel_provider
{

namespace batchnorm
{

/**
 * Calculates the vector size for batchnorm forward inference plans based on the layout and channel information.
 */
inline size_t computeVectorSize(bool isLayoutNHWC, int channels, unsigned int inCstride)
{
    if(isLayoutNHWC)
    {
        if(channels % 4 == 0)
        {
            return 4;
        }
        return channels % 2 == 0 ? 2 : 1;
    }

    if(inCstride % 4 == 0)
    {
        return 4;
    }
    return inCstride % 2 == 0 ? 2 : 1;
}

} // namespace batchnorm

} // namespace hip_kernel_provider
