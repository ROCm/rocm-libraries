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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_HIP_HPP_

#include <type_traits>
#include <iterator>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "detail/device_reduce.hpp"


BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class T
>
__global__
void block_reduce_kernel(InputIterator input,
                         const size_t size,
                         OutputIterator output,
                         BinaryFunction reduce_op,
                         T init_value)
{
    block_reduce_kernel_impl<BlockSize, ItemsPerThread>(
        input, size, output, reduce_op, init_value
    );
}
    
template<
    unsigned int BlockSize,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction
>
__global__
void final_reduce_kernel(InputIterator input,
                         const size_t size,
                         OutputIterator output,
                         BinaryFunction reduce_op)
{
    final_reduce_kernel_impl<BlockSize>(
        input, size, output, reduce_op
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


template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class T
>
hipError_t device_reduce_impl(void * temporary_storage,
                             size_t& storage_size,
                             InputIterator input,
                             OutputIterator output,
                             const size_t size,
                             BinaryFunction reduce_op,
                             T init_value,
                             const hipStream_t stream,
                             bool debug_synchronous)

{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    #ifdef __cpp_lib_is_invocable
    using result_type = typename std::invoke_result<BinaryFunction, input_type, input_type>::type;
    #else
    using result_type = typename std::result_of<BinaryFunction(input_type, input_type)>::type;
    #endif

    constexpr unsigned int block_size = BlockSize;
    constexpr unsigned int items_per_thread = ItemsPerThread;
    constexpr auto items_per_block = block_size * items_per_thread;

    if(temporary_storage == nullptr)
    {
        storage_size = get_temporary_storage_bytes<result_type>(size, items_per_block);
        // Make sure user won't try to allocate 0 bytes memory
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }
    
    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    if(number_of_blocks > 1)
    {
        const unsigned int grid_size = number_of_blocks;

        // Pointer to array with block_prefixes
        result_type * block_prefixes = static_cast<result_type*>(temporary_storage);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::block_reduce_kernel<
                block_size, items_per_thread,
                InputIterator, OutputIterator, BinaryFunction, T
            >),
            dim3(grid_size), dim3(block_size), 0, stream,
            input, size, block_prefixes, reduce_op, init_value
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", size, start);
        
        const unsigned int block = ::rocprim::detail::next_power_of_two(grid_size);
        
        //for (unsigned int i = 0; i < grid_size; i++)
        //    std::cout << block_prefixes[i] << std::endl;
        
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::final_reduce_kernel<
                block_size,
                OutputIterator, OutputIterator, BinaryFunction
            >),
            dim3(1), dim3(block), 0, stream,
            block_prefixes, grid_size, output, reduce_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("final_reduce_kernel", size, start);

    }
    else
    {
        constexpr unsigned int single_reduce_block_size = BlockSize;
        constexpr unsigned int single_reduce_items_per_thread = ItemsPerThread;

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::block_reduce_kernel<
                single_reduce_block_size, single_reduce_items_per_thread,
                InputIterator, OutputIterator, BinaryFunction, T
            >),
            dim3(1), dim3(single_reduce_block_size), 0, stream,
            input, size, output, reduce_op, init_value
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", size, start);
    }
    
    return hipSuccess;
}
    
#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC
    
} // end of detail namespace

template<
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>,
    class T
>
hipError_t device_reduce(void * temporary_storage,
                         size_t& storage_size,
                         InputIterator input,
                         OutputIterator output,
                         const size_t size,
                         BinaryFunction reduce_op = BinaryFunction(),
                         T init_value = T(),
                         const hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    return detail::device_reduce_impl<block_size, items_per_thread>(
        temporary_storage, storage_size,
        input, output, size,
        reduce_op, init_value, stream, debug_synchronous
    );
}


END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_HIP_HPP_
