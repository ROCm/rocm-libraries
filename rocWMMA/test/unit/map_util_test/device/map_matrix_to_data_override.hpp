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

#ifndef ROCWMMA_DEVICE_MAP_MATRIX_TO_DATA_OVERRIDE_HPP
#define ROCWMMA_DEVICE_MAP_MATRIX_TO_DATA_OVERRIDE_HPP

#include "unit_test_traits.hpp"
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    __global__ void MapMatrixToDataOverrideM(uint32_t     m,
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
            using Frag    = fragment<accumulator, BlockM, BlockN, 1, DataT, DataLayout>;
            using Mapping = GetMappingUtil_t<Frag>;

            // Override read M coord
            auto readCoord
                = Mapping::matrixCoordM(static_cast<uint32_t>(static_cast<float32_t>(param1)));
            auto writeCoord = Mapping::matrixCoord();

            Frag f;
            load_matrix_sync(f, Mapping::dataCoord(in, readCoord, ld), ld);
            store_matrix_sync(Mapping::dataCoord(out, writeCoord, ld), f, ld);
        }
    }

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayout>
    __global__ void MapMatrixToDataOverrideN(uint32_t     m,
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
            using Frag    = fragment<accumulator, BlockM, BlockN, 1, DataT, DataLayout>;
            using Mapping = GetMappingUtil_t<Frag>;

            // Override read N coord
            auto readCoord
                = Mapping::matrixCoordN(static_cast<uint32_t>(static_cast<float32_t>(param1)));
            auto writeCoord = Mapping::matrixCoord();

            Frag f;
            load_matrix_sync(f, Mapping::dataCoord(in, readCoord, ld), ld);
            store_matrix_sync(Mapping::dataCoord(out, writeCoord, ld), f, ld);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_MAP_MATRIX_TO_DATA_OVERRIDE_HPP
