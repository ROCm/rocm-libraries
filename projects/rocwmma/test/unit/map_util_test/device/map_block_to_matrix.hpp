/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_DEVICE_MAP_BLOCK_TO_MATRIX_HPP
#define ROCWMMA_DEVICE_MAP_BLOCK_TO_MATRIX_HPP

#include "unit_test_traits.hpp"
#include <rocwmma/internal/mapping_util.hpp>
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    __global__ void MapBlockToMatrix(uint32_t     m,
                                     uint32_t     n,
                                     DataT const* in,
                                     DataT*       out,
                                     uint32_t     ld,
                                     DataT        param1,
                                     DataT        param2)
    {
        using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;
        auto aCoord   = Mapping::blockCoord();

        enum : uint32_t
        {
            MajorIndex = std::is_same_v<DataLayout, row_major> ? 0 : 1,
            MinorIndex = std::is_same_v<DataLayout, row_major> ? 1 : 0,
            ldmajor    = std::is_same_v<DataLayout, row_major> ? BlockM : BlockN,
            ldminor    = std::is_same_v<DataLayout, row_major> ? BlockN : BlockM
        };

        auto majCoord = get<MajorIndex>(aCoord) * ldmajor;
        auto minCoord = get<MinorIndex>(aCoord) * ldminor;

        for(int i = 0; i < ldminor; ++i)
        {
            for(int j = 0; j < ldmajor; ++j)
            {
                out[(majCoord + j) * ld + (minCoord + i)]
                    = in[(((majCoord + j) * ld) + (minCoord + i))];
            }
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_MAP_BLOCK_TO_MATRIX_HPP
