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

#ifndef ROCWMMA_DEVICE_FILL_FRAGMENT_HPP
#define ROCWMMA_DEVICE_FILL_FRAGMENT_HPP

#include "unit_test_traits.hpp"
#include <rocwmma/internal/mapping_util.hpp>
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    __global__ void fillFragmentA(uint32_t     m,
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
                                    DataLayout,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable())
        {
            using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

            // Create frag and fill
            auto frag = fragment<matrix_a, BlockM, 1, BlockN, DataT, DataLayout>();

            fill_fragment(frag, param1);

            // Map and store
            auto* offset = Mapping::dataCoord(out, ld);
            store_matrix_sync(offset, frag, ld);
        }
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    __global__ void fillFragmentB(uint32_t     m,
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
                                    DataLayout,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable())
        {
            using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

            // Create frag and fill
            auto frag = fragment<matrix_b, 1, BlockN, BlockM, DataT, DataLayout>();

            fill_fragment(frag, param1);

            // Map and store
            auto* offset = Mapping::dataCoord(out, ld);
            store_matrix_sync(offset, frag, ld);
        }
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    __global__ void fillFragmentAcc(uint32_t     m,
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
                                    DataLayout,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable())
        {
            using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

            // Create frag and fill
            auto frag = fragment<accumulator, BlockM, BlockN, 1, DataT, DataLayout>();

            fill_fragment(frag, param1);

            // Map and store
            auto* offset = Mapping::dataCoord(out, ld);
            store_matrix_sync(offset, frag, ld);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_FILL_FRAGMENT_HPP
