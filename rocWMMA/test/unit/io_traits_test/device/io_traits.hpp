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

#ifndef ROCWMMA_DEVICE_IO_TRAITS_TEST_HPP
#define ROCWMMA_DEVICE_IO_TRAITS_TEST_HPP

#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename DataT, uint32_t BlockDim, uint32_t BlockK, uint32_t VectorWidth>
    __global__ void ioTraitsTest(uint32_t     m,
                                 uint32_t     n,
                                 DataT const* in,
                                 DataT*       out,
                                 uint32_t     ld,
                                 DataT        param1,
                                 DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        using PackTraits = PackTraits<DataT>;

        // Check on pack ratio sizes
        err |= (PackTraits::PackRatio * sizeof(typename PackTraits::UnpackedT)
                != sizeof(typename PackTraits::PackedT));
        err |= (!std::is_same<DataT, typename PackTraits::UnpackedT>::value);

        // Check consistency of packed vs unpacked types
        if(std::is_floating_point<typename PackTraits::UnpackedT>::value)
        {
            err |= (!std::is_floating_point<typename PackTraits::PackedT>::value);
        }
        else if(std::is_integral<typename PackTraits::UnpackedT>::value)
        {
            err |= (!std::is_integral<typename PackTraits::PackedT>::value);
        }

        // Device detected waveSize comes in mParam1
        uint32_t waveSize = static_cast<uint32_t>(param1);

        // VectorWidthTraits, C++ perspective
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;

        err |= (IOTraits::ThreadsPerIO != waveSize);
        err |= (IOTraits::ElementsPerIO != (waveSize * VectorWidth));
        err |= (IOTraits::KPerIO != std::max(1u, (waveSize * VectorWidth / BlockDim)));
        err |= (IOTraits::ElementCount != (BlockDim * BlockK));
        err |= (IOTraits::IOCount != (BlockDim * BlockK / (waveSize * VectorWidth)));
        err |= (IOTraits::UnpackedSize != (BlockDim * BlockK / waveSize));
        err |= (IOTraits::PackedSize != (BlockDim * BlockK / waveSize / PackTraits::PackRatio));

        // Physical hardware perspective
        err |= (IOTraits::UnpackedVRegCount
                != (IOTraits::UnpackedSize
                    * std::max(1u, (uint32_t)sizeof(DataT) / Constants::AMDGCN_DWORD_SIZE_BYTES)));
        err |= (IOTraits::PackedVRegCount
                != (IOTraits::PackedSize
                    * std::max(1u, (uint32_t)sizeof(DataT) / Constants::AMDGCN_DWORD_SIZE_BYTES)));

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_IO_TRAITS_TEST_HPP
