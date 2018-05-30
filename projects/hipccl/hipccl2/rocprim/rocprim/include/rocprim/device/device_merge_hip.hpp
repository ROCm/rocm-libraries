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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_HIP_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_merge.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hip
/// @{

namespace detail
{

template<
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class BinaryFunction
>
__global__
void partition_kernel(IndexIterator index,
                      KeysInputIterator1 keys_input1,
                      KeysInputIterator2 keys_input2,
                      const size_t input1_size,
                      const size_t input2_size,
                      const unsigned int spacing,
                      BinaryFunction compare_function)
{
    partition_kernel_impl(
        index, keys_input1, keys_input2, input1_size, input2_size,
        spacing, compare_function
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
__global__
void merge_kernel(IndexIterator index,
                  KeysInputIterator1 keys_input1,
                  KeysInputIterator2 keys_input2,
                  KeysOutputIterator keys_output,
                  ValuesInputIterator1 values_input1,
                  ValuesInputIterator2 values_input2,
                  ValuesOutputIterator values_output,
                  const size_t input1_size,
                  const size_t input2_size,
                  BinaryFunction compare_function)
{
    merge_kernel_impl<BlockSize, ItemsPerThread>(
        index, keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function
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
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
inline
hipError_t merge_impl(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator1 keys_input1,
                      KeysInputIterator2 keys_input2,
                      KeysOutputIterator keys_output,
                      ValuesInputIterator1 values_input1,
                      ValuesInputIterator2 values_input2,
                      ValuesOutputIterator values_output,
                      const size_t input1_size,
                      const size_t input2_size,
                      BinaryFunction compare_function,
                      const hipStream_t stream,
                      bool debug_synchronous)

{
    constexpr unsigned int block_size = BlockSize;
    constexpr unsigned int items_per_thread = ItemsPerThread;
    constexpr auto items_per_block = block_size * items_per_thread;

    const unsigned int partitions = div_up((unsigned int)(input1_size + input2_size), items_per_block) + 1;
    const size_t partition_bytes = partitions * sizeof(unsigned int);

    if(temporary_storage == nullptr)
    {
        storage_size = partition_bytes;
        // Make sure user won't try to allocate 0 bytes memory
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = partitions;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    unsigned int * index = reinterpret_cast<unsigned int *>(temporary_storage);

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::partition_kernel),
        dim3(number_of_blocks), dim3(1), 0, stream,
        index, keys_input1, keys_input2, input1_size, input2_size,
        items_per_block, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", input1_size, start);

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::merge_kernel<block_size, items_per_thread>),
        dim3(number_of_blocks), dim3(block_size), 0, stream,
        index, keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("serial_merge_kernel", input1_size, start);

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

template<
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator1>::value_type>
>
inline
hipError_t merge(void * temporary_storage,
                 size_t& storage_size,
                 KeysInputIterator1 keys_input1,
                 KeysInputIterator2 keys_input2,
                 KeysOutputIterator keys_output,
                 const size_t input1_size,
                 const size_t input2_size,
                 BinaryFunction compare_function = BinaryFunction(),
                 const hipStream_t stream = 0,
                 bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 8;
    empty_type * values = nullptr;
    return detail::merge_impl<block_size, items_per_thread>(
        temporary_storage, storage_size,
        keys_input1, keys_input2, keys_output,
        values, values, values,
        input1_size, input2_size, compare_function,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hip

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_HIP_HPP_
