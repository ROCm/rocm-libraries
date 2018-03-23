// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_HIP_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../iterator/transform_iterator.hpp"

#include "detail/device_select.hpp"
#include "device_scan_hip.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hip
/// @{

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
__global__
void scatter_kernel(InputIterator input,
                    const size_t size,
                    FlagIterator flags,
                    unsigned int * indices,
                    OutputIterator output,
                    SelectedCountOutputIterator selected_count_output)
{
    detail::scatter_kernel_impl<BlockSize, ItemsPerThread>(
        input, size, flags, indices,
        output, selected_count_output
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class SelectOp
>
__global__
void scatter_if_kernel(InputIterator input,
                       const size_t size,
                       unsigned int * indices,
                       OutputIterator output,
                       SelectedCountOutputIterator selected_count_output,
                       SelectOp select_op)
{
    detail::scatter_if_kernel_impl<BlockSize, ItemsPerThread>(
        input, size, indices,
        output, selected_count_output,
        select_op
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputFlagIterator,
    class InequalityOp
>
__global__
void flag_unique_kernel(InputIterator input,
                        const size_t size,
                        OutputFlagIterator flags,
                        InequalityOp inequality_op)
{
    detail::flag_unique_kernel_impl<BlockSize, ItemsPerThread>(
        input, size, flags, inequality_op
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
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

} // end detail namespace

template<
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
hipError_t select(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  FlagIterator flags,
                  OutputIterator output,
                  SelectedCountOutputIterator selected_count_output,
                  const size_t size,
                  const hipStream_t stream = 0,
                  const bool debug_synchronous = false)
{
    // Get temporary storage required by scan operation
    size_t scan_storage_size = 0;
    unsigned int * dummy_ptr = nullptr;
    auto error = ::rocprim::exclusive_scan(
        nullptr, scan_storage_size,
        flags, dummy_ptr, 0U, size, ::rocprim::plus<unsigned int>(),
        stream, debug_synchronous
    );
    if(error != hipSuccess) return error;
    // Align
    scan_storage_size = ::rocprim::detail::align_size(scan_storage_size);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = scan_storage_size;
        // Add storage required for indexes
        storage_size += size * sizeof(unsigned int);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    // Return for empty input
    if(size == 0) return hipSuccess;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    // Calculate output indices to scatter selected values
    auto indices = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(temporary_storage) + scan_storage_size
    );
    error = ::rocprim::exclusive_scan(
        temporary_storage, scan_storage_size,
        flags, indices, 0U, size, ::rocprim::plus<unsigned int>(),
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::exclusive_scan", size, start)

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::scatter_kernel<
            block_size, items_per_thread,
            InputIterator, FlagIterator,
            OutputIterator, SelectedCountOutputIterator
        >),
        dim3(number_of_blocks), dim3(block_size), 0, stream,
        input, size, flags, indices, output, selected_count_output
    );
    error = hipPeekAtLastError();
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scatter_kernel", size, start)

    return hipSuccess;
}

template<
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class SelectOp
>
inline
hipError_t select(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  SelectedCountOutputIterator selected_count_output,
                  const size_t size,
                  SelectOp select_op,
                  const hipStream_t stream = 0,
                  const bool debug_synchronous = false)
{
    // Get temporary storage required by scan operation
    size_t scan_storage_size = 0;
    unsigned char * dummy_in_ptr = nullptr;
    unsigned int *  dummy_out_ptr = nullptr;
    auto error = ::rocprim::exclusive_scan(
        nullptr, scan_storage_size,
        dummy_in_ptr, dummy_out_ptr,
        0U, size, ::rocprim::plus<unsigned int>(),
        stream, debug_synchronous
    );
    // Align
    scan_storage_size = ::rocprim::detail::align_size(scan_storage_size);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = scan_storage_size;
        // Add storage required for indexes
        storage_size += size * sizeof(unsigned int);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    // Return for empty input
    if(size == 0) return hipSuccess;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    // Calculate output indices to scatter selected values
    auto indices = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(temporary_storage) + scan_storage_size
    );
    error = ::rocprim::exclusive_scan(
        temporary_storage, scan_storage_size,
        ::rocprim::make_transform_iterator(input, select_op),
        indices, 0U, size, ::rocprim::plus<unsigned int>(),
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::exclusive_scan", size, start)

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::scatter_if_kernel<
            block_size, items_per_thread,
            InputIterator,
            OutputIterator, SelectedCountOutputIterator,
            SelectOp
        >),
        dim3(number_of_blocks), dim3(block_size), 0, stream,
        input, size, indices, output, selected_count_output, select_op
    );
    error = hipPeekAtLastError();
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scatter_if_kernel", size, start)

    return hipSuccess;
}

template<
    class InputIterator,
    class OutputIterator,
    class UniqueCountOutputIterator,
    class EqualityOp = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t unique(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  UniqueCountOutputIterator unique_count_output,
                  const size_t size,
                  EqualityOp equality_op = EqualityOp(),
                  const hipStream_t stream = 0,
                  const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    // Get temporary storage required by select operation
    size_t select_storage_size = 0;
    unsigned char * dummy_flags_ptr = nullptr;
    auto error = ::rocprim::select(
        static_cast<void*>(nullptr), select_storage_size,
        input, dummy_flags_ptr, output, unique_count_output, size,
        stream, debug_synchronous
    );
    // Align
    select_storage_size = ::rocprim::detail::align_size(select_storage_size);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = select_storage_size;
        // Add storage required for flags
        storage_size += size * sizeof(unsigned char);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    // Return for empty input
    if(size == 0) return hipSuccess;

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto flags = static_cast<unsigned char*>(temporary_storage) + select_storage_size;
    auto inequality_op =
        [equality_op] __device__ (const input_type& a, const input_type& b) -> bool
        {
            return !equality_op(a, b);
        };
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::flag_unique_kernel<
            block_size, items_per_thread,
            InputIterator, decltype(flags), decltype(inequality_op)
        >),
        dim3(number_of_blocks), dim3(block_size), 0, stream,
        input, size, flags, inequality_op
    );
    error = hipPeekAtLastError();
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("flag_unique_kernel", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    // select unique values
    error = ::rocprim::select(
        temporary_storage, select_storage_size,
        input, flags, output, unique_count_output, size,
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::select", size, start)

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

/// @}
// end of group devicemodule_hip

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SELECT_HIP_HPP_
