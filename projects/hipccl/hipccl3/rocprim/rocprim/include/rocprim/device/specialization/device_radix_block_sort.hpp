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

#ifndef ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_SINGLE_SORT_HPP_
#define ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_SINGLE_SORT_HPP_

#include "../detail/device_radix_sort.hpp"
#include "../device_radix_sort_config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<Config>().block_size) void radix_sort_block_sort_kernel(
        KeysInputIterator    keys_input,
        KeysOutputIterator   keys_output,
        ValuesInputIterator  values_input,
        ValuesOutputIterator values_output,
        unsigned int         size,
        unsigned int         bit,
        unsigned int         current_radix_bits)
{
    static constexpr kernel_config_params params = device_params<Config>();
    sort_single<params.block_size, params.items_per_thread, Descending>(keys_input,
                                                                        keys_output,
                                                                        values_input,
                                                                        values_output,
                                                                        size,
                                                                        bit,
                                                                        current_radix_bits);
}

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator>
inline hipError_t radix_sort_block_sort(KeysInputIterator    keys_input,
                                        KeysOutputIterator   keys_output,
                                        ValuesInputIterator  values_input,
                                        ValuesOutputIterator values_output,
                                        unsigned int         size,
                                        unsigned int&        sort_items_per_block,
                                        unsigned int         bit,
                                        unsigned int         end_bit,
                                        hipStream_t          stream,
                                        bool                 debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using config = wrapped_radix_sort_block_sort_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const kernel_config_params params = dispatch_target_arch<config>(target_arch);

    sort_items_per_block                     = params.block_size * params.items_per_thread;
    const unsigned int sort_number_of_blocks = ceiling_div(size, sort_items_per_block);
    const unsigned int current_radix_bits    = end_bit - bit;

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << '\n';
        std::cout << "sort_block_size: " << params.block_size << '\n';
        std::cout << "sort_items_per_thread: " << params.items_per_thread << '\n';
        std::cout << "sort_items_per_block: " << sort_items_per_block << '\n';
        std::cout << "sort_number_of_blocks: " << sort_number_of_blocks << '\n';
        std::cout << "current_radix_bit: " << current_radix_bits << '\n';
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
        start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(HIP_KERNEL_NAME(radix_sort_block_sort_kernel<config, Descending>),
                       dim3(sort_number_of_blocks),
                       dim3(params.block_size),
                       0,
                       stream,
                       keys_input,
                       keys_output,
                       values_input,
                       values_output,
                       size,
                       bit,
                       current_radix_bits);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_sort_block_sort_kernel", size, start)
    return hipSuccess;
}

template<class Config, class key_type, class value_type>
hipError_t get_radix_sort_block_sort_config(hipStream_t stream, kernel_config_params& params)
{
    using block_single_config = wrapped_radix_sort_block_sort_config<Config, key_type, value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }

    params = dispatch_target_arch<block_single_config>(target_arch);
    return hipSuccess;
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_SINGLE_SORT_HPP_
