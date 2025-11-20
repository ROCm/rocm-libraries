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

#ifndef ROCWMMA_DEVICE_COL_LAYOUT_HPP
#define ROCWMMA_DEVICE_COL_LAYOUT_HPP

#include "unit_test_traits.hpp"
#include <rocwmma/internal/io_layout.hpp>
#include <rocwmma/internal/io_traits.hpp>
#include <rocwmma/internal/mapping_util.hpp>

namespace rocwmma
{
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayoutT>
    __global__ void ColLayout(uint32_t     m,
                              uint32_t     n,
                              DataT const* in,
                              DataT*       out,
                              uint32_t     ld,
                              DataT        param1,
                              DataT        param2)
    {
        if constexpr(FragSize_guard<BlockM,
                                    BlockN,
                                    DataT,
                                    DataLayoutT,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable())
        {
            enum : uint32_t
            {
                BlockHeight = BlockM,
                BlockWidth  = BlockN,

                BlockDim = BlockM,
                BlockK   = BlockN,

                MaxVectorWidth = detail::MaxVWSelector<matrix_a, BlockDim, BlockK, DataT, DataLayoutT>::Result,
                VectorWidth = MaxVectorWidth
            };

            using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;

            using LayoutT = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>;

            using Mapping = MappingUtil<BlockHeight, BlockWidth, DataT, DataLayoutT>;

            constexpr auto ioCount     = IOTraits::IOCount;
            auto baseOffset  = LayoutT::baseOffset();
            auto matrixCoord = Mapping::matrixCoord();

            auto currentOffset = matrixCoord + baseOffset;
            for(auto i = 0u; i < ioCount; ++i)
            {
                for(auto j = 0u; j < VectorWidth; ++j)
                {
                    auto index = Mapping::dataOffset(currentOffset, ld) + j;
                    out[index] = in[index];
                }
                currentOffset += LayoutT::incrementalOffset(i);
            }
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_COL_LAYOUT_HPP
