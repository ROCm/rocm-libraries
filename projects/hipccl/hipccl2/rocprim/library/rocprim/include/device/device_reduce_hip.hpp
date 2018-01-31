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

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool WithInitialValue,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
__global__
void block_reduce_kernel(InputIterator input,
                         const size_t size,
                         OutputIterator output,
                         InitValueType initial_value,
                         BinaryFunction reduce_op)
{
    block_reduce_kernel_impl<BlockSize, ItemsPerThread, WithInitialValue>(
        input, size, output, initial_value, reduce_op
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
    bool WithInitialValue, // true when inital_value should be used in reduction
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
hipError_t device_reduce_impl(void * temporary_storage,
                              size_t& storage_size,
                              InputIterator input,
                              OutputIterator output,
                              const InitValueType initial_value,
                              const size_t size,
                              BinaryFunction reduce_op,
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
        storage_size = reduce_get_temporary_storage_bytes<result_type>(size, items_per_block);
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
                block_size, items_per_thread, false,
                InputIterator, OutputIterator, InitValueType, BinaryFunction
            >),
            dim3(grid_size), dim3(block_size), 0, stream,
            input, size, block_prefixes, initial_value, reduce_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", size, start);

        void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
        auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(result_type));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        auto error = device_reduce_impl<BlockSize, ItemsPerThread, WithInitialValue>(
            nested_temp_storage,
            nested_temp_storage_size,
            block_prefixes, // input
            output, // output
            initial_value,
            number_of_blocks, // input size
            reduce_op,
            stream,
            debug_synchronous
        );
        if(error != hipSuccess) return error;
        ROCPRIM_DETAIL_HIP_SYNC("nested_device_reduce", number_of_blocks, start);
    }
    else
    {
        constexpr unsigned int single_reduce_block_size = BlockSize;
        constexpr unsigned int single_reduce_items_per_thread = ItemsPerThread;

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::block_reduce_kernel<
                single_reduce_block_size, single_reduce_items_per_thread, WithInitialValue,
                InputIterator, OutputIterator, InitValueType, BinaryFunction
            >),
            dim3(1), dim3(single_reduce_block_size), 0, stream,
            input, size, output, initial_value, reduce_op
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
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
hipError_t device_reduce(void * temporary_storage,
                         size_t& storage_size,
                         InputIterator input,
                         OutputIterator output,
                         const InitValueType initial_value,
                         const size_t size,
                         BinaryFunction reduce_op = BinaryFunction(),
                         const hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    return detail::device_reduce_impl<block_size, items_per_thread, true>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        reduce_op, stream, debug_synchronous
    );
}

template<
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
hipError_t device_reduce(void * temporary_storage,
                         size_t& storage_size,
                         InputIterator input,
                         OutputIterator output,
                         const size_t size,
                         BinaryFunction reduce_op = BinaryFunction(),
                         const hipStream_t stream = 0,
                         bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    return detail::device_reduce_impl<block_size, items_per_thread, false>(
        temporary_storage, storage_size,
        input, output, char(0), size,
        reduce_op, stream, debug_synchronous
    );
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_HIP_HPP_
