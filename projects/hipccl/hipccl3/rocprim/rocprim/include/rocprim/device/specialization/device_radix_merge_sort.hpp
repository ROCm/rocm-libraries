// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
#define ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_

#include "../detail/device_radix_sort.hpp"
#include "../specialization/device_radix_single_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    template<
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator,
        class BinaryFunction
    >
    ROCPRIM_KERNEL
   __launch_bounds__(BlockSize)
   void radix_block_merge_kernel(KeysInputIterator   keys_input,
                                KeysOutputIterator   keys_output,
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                const size_t         input_size,
                                const unsigned int   merge_items_per_block_size,
                                BinaryFunction       compare_function)
   {
       radix_block_merge_impl<BlockSize, ItemsPerThread>(
           keys_input, keys_output,
           values_input, values_output,
           input_size, merge_items_per_block_size,
           compare_function
       );
   }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    hipError_t radix_sort_merge(KeysInputIterator keys_input,
                                typename std::iterator_traits<KeysInputIterator>::value_type * keys_buffer,
                                KeysOutputIterator keys_output,
                                ValuesInputIterator values_input,
                                typename std::iterator_traits<ValuesInputIterator>::value_type * values_buffer,
                                ValuesOutputIterator values_output,
                                unsigned int size,
                                unsigned int bit,
                                unsigned int end_bit,
                                hipStream_t stream,
                                bool debug_synchronous)
    {
        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
        using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

        constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

        constexpr unsigned int items_per_thread = Config::sort_merge::items_per_thread;
        constexpr unsigned int block_size = Config::sort_merge::block_size;
        constexpr unsigned int items_per_block = block_size * items_per_thread;

        const unsigned int current_radix_bits = end_bit - bit;
        auto number_of_blocks = (size + items_per_block - 1) / items_per_block;

        std::chrono::high_resolution_clock::time_point start;
        if(debug_synchronous)
        {
            std::cout << "block size " << block_size << '\n';
            std::cout << "items per thread " << items_per_thread << '\n';
            std::cout << "number of blocks " << number_of_blocks << '\n';
            std::cout << "bit " << bit << '\n';
            std::cout << "current_radix_bits " << current_radix_bits << '\n';
        }

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_single_kernel<
                block_size, items_per_thread , Descending
            >),
            dim3(number_of_blocks), dim3(block_size), 0, stream,
            keys_input, keys_buffer, values_input, values_buffer,
            size, bit, current_radix_bits
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_sort_single", size, start)

        bool temporary_store = true;
        for(unsigned int block = items_per_block; block < size; block *= 2)
        {
            temporary_store = !temporary_store;
            if(temporary_store)
            {
                if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
                if( current_radix_bits == sizeof(key_type) * 8 )
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(radix_block_merge_kernel<block_size, items_per_thread>),
                        dim3(number_of_blocks), dim3(block_size), 0, stream,
                        keys_output, keys_buffer, values_output, values_buffer,
                        size, block, radix_merge_compare<Descending, false, key_type>()
                    );
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(radix_block_merge_kernel<block_size, items_per_thread>),
                        dim3(number_of_blocks), dim3(block_size), 0, stream,
                        keys_output, keys_buffer, values_output, values_buffer,
                        size, block, radix_merge_compare<Descending, true, key_type>(bit, current_radix_bits)
                    );
                }
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_block_merge_kernel", size, start);
            }
            else
            {
                if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
                if( current_radix_bits == sizeof(key_type) * 8 )
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(radix_block_merge_kernel<block_size, items_per_thread>),
                        dim3(number_of_blocks), dim3(block_size), 0, stream,
                        keys_buffer, keys_output, values_buffer, values_output,
                        size, block, radix_merge_compare<Descending, false, key_type>()
                    );
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(radix_block_merge_kernel<block_size, items_per_thread>),
                        dim3(number_of_blocks), dim3(block_size), 0, stream,
                        keys_buffer, keys_output, values_buffer, values_output,
                        size, block, radix_merge_compare<Descending, true, key_type>(bit, current_radix_bits)
                    );
                }
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_block_merge_kernel", size, start);
            }
        }

        if(temporary_store)
        {
            hipError_t error = ::rocprim::transform(
                keys_buffer, keys_output, size,
                ::rocprim::identity<key_type>(), stream, debug_synchronous
            );
            if(error != hipSuccess) return error;

            if(with_values)
            {
                hipError_t error = ::rocprim::transform(
                    values_buffer, values_output, size,
                    ::rocprim::identity<value_type>(), stream, debug_synchronous
                );
                if(error != hipSuccess) return error;
            }
        }

        return hipSuccess;
    }
} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_MERGE_SORT_HPP_
