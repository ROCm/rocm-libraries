// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_HIP_HPP_

#include <type_traits>
#include <iterator>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "detail/device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class UnaryFunction
>
__global__
void transform_kernel(InputIterator input,
                      const size_t size,
                      OutputIterator output,
                      UnaryFunction transform_op)
{
    transform_kernel_impl<BlockSize, ItemsPerThread>(
        input, size, output, transform_op
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC(name, size, start) \
    if(debug_synchronous) \
    { \
        std::cout << name << "(" << size << ")"; \
        auto error = hipStreamSynchronize(stream); \
        if(error != hipSuccess) return error; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
        std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
    }

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto error = hipPeekAtLastError(); \
        if(error != hipSuccess) return error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto error = hipStreamSynchronize(stream); \
            if(error != hipSuccess) return error; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

} // end of detail namespace

template<
    class InputIterator,
    class OutputIterator,
    class UnaryFunction
>
hipError_t device_transform(InputIterator input,
                            OutputIterator output,
                            const size_t size,
                            UnaryFunction transform_op,
                            const hipStream_t stream = 0,
                            bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::transform_kernel<
            block_size, items_per_thread,
            InputIterator, OutputIterator, UnaryFunction
        >),
        dim3(number_of_blocks), dim3(block_size), 0, stream,
        input, size, output, transform_op
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("transform_kernel", size, start);

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_TRANSFORM_HIP_HPP_
