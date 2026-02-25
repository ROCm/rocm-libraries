// Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../common.hpp"
#include "../detail/device_radix_sort.hpp"
#include "../device_radix_sort_config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class Decomposer>
inline hipError_t radix_sort_block_sort(KeysInputIterator    keys_input,
                                        KeysOutputIterator   keys_output,
                                        ValuesInputIterator  values_input,
                                        ValuesOutputIterator values_output,
                                        unsigned int         size,
                                        unsigned int&        sort_items_per_block,
                                        Decomposer           decomposer,
                                        unsigned int         bit,
                                        unsigned int         end_bit,
                                        hipStream_t          stream,
                                        bool                 debug_synchronous)
{
    using key_type   = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using Selector = radix_sort_block_sort_config_selector<key_type, value_type>;

    const target current_target(stream);

    const auto params = get_config<Selector>(Config{}, current_target);

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
    std::chrono::steady_clock::time_point start;
    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
    }

    auto radix_sort_block_sort_kernel = [=](auto target_config)
    {
        using TargetConfig           = decltype(target_config);
        static constexpr auto params = TargetConfig::params;

        sort_single<params.block_size,
                    params.items_per_thread,
                    Descending,
                    TargetConfig::wavefront>(keys_input,
                                             keys_output,
                                             values_input,
                                             values_output,
                                             size,
                                             decomposer,
                                             bit,
                                             current_radix_bits);
    };

    ROCPRIM_RETURN_ON_ERROR(
        execute_launch_plan<Config, Selector, radix_sort_block_sort_config_static_selector>(
            current_target,
            radix_sort_block_sort_kernel,
            dim3(sort_number_of_blocks),
            dim3(params.block_size),
            0,
            stream));

    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_sort_block_sort_kernel", size, start);
    return hipSuccess;
}

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_SINGLE_SORT_HPP_
