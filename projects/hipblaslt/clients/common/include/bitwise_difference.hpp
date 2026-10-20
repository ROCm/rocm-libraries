/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#pragma once

#include <cstddef>
#include <cstring>

/*! \brief the elements of two buffers whose bits differ: how many, and the index of the first */
struct BitwiseDifference
{
    size_t count = 0;
    size_t first = 0;
};

inline BitwiseDifference
    bitwise_difference(const void* a, const void* b, size_t elements, size_t element_size)
{
    BitwiseDifference difference;
    if(elements == 0 || std::memcmp(a, b, elements * element_size) == 0)
        return difference;

    const auto* pa = static_cast<const unsigned char*>(a);
    const auto* pb = static_cast<const unsigned char*>(b);
    for(size_t i = 0; i < elements; i++)
    {
        if(std::memcmp(pa + i * element_size, pb + i * element_size, element_size) != 0)
        {
            if(difference.count == 0)
                difference.first = i;
            difference.count++;
        }
    }
    return difference;
}
