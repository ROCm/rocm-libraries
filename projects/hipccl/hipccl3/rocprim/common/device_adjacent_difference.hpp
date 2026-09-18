// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef COMMON_DEVICE_ADJACENT_DIFFERENCE_HPP_
#define COMMON_DEVICE_ADJACENT_DIFFERENCE_HPP_

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_adjacent_difference.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace common
{

enum class api_variant
{
    no_alias,
    alias,
    in_place
};

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(
    std::true_type /*left*/,
    std::integral_constant<api_variant, api_variant::no_alias> /*aliasing*/,
    void* const    temporary_storage,
    std::size_t&   storage_size,
    const InputIt  input,
    const OutputIt output,
    Args&&... args)
{
    return ::rocprim::adjacent_difference<Config>(temporary_storage,
                                                  storage_size,
                                                  input,
                                                  output,
                                                  std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(
    std::false_type /*left*/,
    std::integral_constant<api_variant, api_variant::no_alias> /*aliasing*/,
    void* const    temporary_storage,
    std::size_t&   storage_size,
    const InputIt  input,
    const OutputIt output,
    Args&&... args)
{
    return ::rocprim::adjacent_difference_right<Config>(temporary_storage,
                                                        storage_size,
                                                        input,
                                                        output,
                                                        std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(
    std::true_type /*left*/,
    std::integral_constant<api_variant, api_variant::in_place> /*aliasing*/,
    void* const   temporary_storage,
    std::size_t&  storage_size,
    const InputIt input,
    const OutputIt /*output*/,
    Args&&... args)
{
    return ::rocprim::adjacent_difference_inplace<Config>(temporary_storage,
                                                          storage_size,
                                                          input,
                                                          std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(
    std::false_type /*left*/,
    std::integral_constant<api_variant, api_variant::in_place> /*aliasing*/,
    void* const   temporary_storage,
    std::size_t&  storage_size,
    const InputIt input,
    const OutputIt /*output*/,
    Args&&... args)
{
    return ::rocprim::adjacent_difference_right_inplace<Config>(temporary_storage,
                                                                storage_size,
                                                                input,
                                                                std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(
    std::true_type /*left*/,
    std::integral_constant<api_variant, api_variant::alias> /*aliasing*/,
    void* const    temporary_storage,
    std::size_t&   storage_size,
    const InputIt  input,
    const OutputIt output,
    Args&&... args)
{
    return ::rocprim::adjacent_difference_inplace<Config>(temporary_storage,
                                                          storage_size,
                                                          input,
                                                          output,
                                                          std::forward<Args>(args)...);
}

template<typename Config = rocprim::default_config,
         typename InputIt,
         typename OutputIt,
         typename... Args>
auto dispatch_adjacent_difference(
    std::false_type /*left*/,
    std::integral_constant<api_variant, api_variant::alias> /*aliasing*/,
    void* const    temporary_storage,
    std::size_t&   storage_size,
    const InputIt  input,
    const OutputIt output,
    Args&&... args)
{
    return ::rocprim::adjacent_difference_right_inplace<Config>(temporary_storage,
                                                                storage_size,
                                                                input,
                                                                output,
                                                                std::forward<Args>(args)...);
}

} // namespace common

#endif // COMMON_DEVICE_ADJACENT_DIFFERENCE_HPP_
