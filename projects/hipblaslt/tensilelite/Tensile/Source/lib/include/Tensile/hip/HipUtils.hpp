/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <sstream>
#include <stdexcept>

#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

#define HIP_CHECK_EXC(expr)                                                                       \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if(e)                                                                                     \
        {                                                                                         \
            const char*        errName = hipGetErrorName(e);                                      \
            const char*        errMsg  = hipGetErrorString(e);                                    \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl;                                                           \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

#define HIP_CHECK_EXC_MESSAGE(expr, message)                                                      \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if(e)                                                                                     \
        {                                                                                         \
            const char*        errName = hipGetErrorName(e);                                      \
            const char*        errMsg  = hipGetErrorString(e);                                    \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl                                                            \
                << (message) << std::endl;                                                        \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

#define HIP_CHECK_RETURN(expr) \
    do                         \
    {                          \
        hipError_t e = (expr); \
        if(e)                  \
            return e;          \
    } while(0)

#define HIP_CHECK_PRINT(expr)                             \
    {                                                     \
        hipError_t e = (expr);                            \
        if(e)                                             \
            std::cout << "Error code " << e << std::endl; \
    }

namespace TensileLite
{
    namespace hip
    {
        inline void CopyTensorVoid(void*                   dst,
                                   void const*             src,
                                   TensorDescriptor const& desc,
                                   hipMemcpyKind           direction,
                                   hipStream_t             stream = 0)
        {
            if(desc.dimensions() == 0 || desc.totalLogicalElements() == 0)
                return;

            auto const&         sizes   = desc.sizes();
            auto const&         strides = desc.strides();
            std::vector<size_t> coord(desc.dimensions(), 0);

            size_t contiguousDimensions = 0;
            size_t expectedStride       = 1;

            // Optimize the number of copy operations by coalescing all the
            // dimensions that are contiguous in memory.
            for(size_t i = 0; i < desc.dimensions(); i++)
            {
                if(strides[i] > expectedStride)
                    break;

                contiguousDimensions = i + 1;

                if(i < desc.dimensions() - 1)
                    expectedStride = strides[i] * sizes[i];
            }

            auto copyCount = CoordCount(sizes.begin() + contiguousDimensions, sizes.end());

            size_t maxStride
                = *std::max_element(strides.begin(), strides.begin() + contiguousDimensions);
            size_t copyBytes = maxStride * sizes.at(contiguousDimensions - 1) * desc.elementBytes();

            for(size_t idx = 0; idx < copyCount; idx++)
            {
                CoordNumbered(idx,
                              coord.begin() + contiguousDimensions,
                              coord.end(),
                              sizes.begin() + contiguousDimensions,
                              sizes.end());

                auto     beginOffset = desc.index(coord);
                auto     bytesOffset = desc.elementBytes() * beginOffset;
                uint8_t* dstBytes    = (uint8_t*)dst + bytesOffset;
                uint8_t* srcBytes    = (uint8_t*)dst + bytesOffset;

                HIP_CHECK_EXC(hipMemcpyAsync(dstBytes, srcBytes, copyBytes, direction, stream));
            }
        }

        template <typename T>
        void CopyTensor(T*                      dst,
                        T const*                src,
                        TensorDescriptor const& desc,
                        hipMemcpyKind           direction,
                        hipStream_t             stream = 0)
        {
            if(desc.dimensions() == 0 || desc.totalLogicalElements() == 0)
                return;

            auto const&         sizes   = desc.sizes();
            auto const&         strides = desc.strides();
            std::vector<size_t> coord(desc.dimensions(), 0);

            size_t contiguousDimensions = 0;
            size_t expectedStride       = 1;

            // Optimize the number of copy operations by coalescing all the
            // dimensions that are contiguous in memory.
            for(size_t i = 0; i < desc.dimensions(); i++)
            {
                if(strides[i] > expectedStride)
                    break;

                contiguousDimensions = i + 1;

                if(i < desc.dimensions() - 1)
                    expectedStride = strides[i] * sizes[i];
            }

            auto copyCount = CoordCount(sizes.begin() + contiguousDimensions, sizes.end());

            size_t maxStride
                = *std::max_element(strides.begin(), strides.begin() + contiguousDimensions);
            size_t copyBytes = maxStride * sizes.at(contiguousDimensions - 1) * sizeof(T);

            for(size_t idx = 0; idx < copyCount; idx++)
            {
                CoordNumbered(idx,
                              coord.begin() + contiguousDimensions,
                              coord.end(),
                              sizes.begin() + contiguousDimensions,
                              sizes.end());

                auto beginOffset = desc.index(coord);

                HIP_CHECK_EXC(hipMemcpyAsync(
                    dst + beginOffset, src + beginOffset, copyBytes, direction, stream));
            }
        }

        template <typename T>
        void CopyBuffer(T*            dst,
                        T const*      src,
                        const size_t  copyBytes,
                        hipMemcpyKind direction,
                        hipStream_t   stream = 0)
        {
            HIP_CHECK_EXC(hipMemcpyAsync(dst, src, copyBytes, direction, stream));
        }
    } // namespace hip
} // namespace TensileLite
