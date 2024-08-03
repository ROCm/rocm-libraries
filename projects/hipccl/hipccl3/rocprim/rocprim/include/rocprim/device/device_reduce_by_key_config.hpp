// Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_

#include "detail/config/device_reduce_by_key.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Generic for user-provided config: instantiate user-provided config.
template<typename ReduceByKeyConfig, typename, typename, typename>
struct wrapped_reduce_by_key_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr reduce_by_key_config_params params = ReduceByKeyConfig{};
    };
};

// Generic for default config: instantiate base config.
template<typename KeyType,
         typename AccumulatorType,
         typename BinaryFunction,
         typename Enable = void>
struct wrapped_reduce_by_key_impl
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr reduce_by_key_config_params params =
            typename default_reduce_by_key_config_base<KeyType, AccumulatorType>::type{};
    };
};

// Specialization for default config if types are not custom: instantiate the tuned config.
template<typename KeyType, typename AccumulatorType, typename BinaryFunction>
struct wrapped_reduce_by_key_impl<
    KeyType,
    AccumulatorType,
    BinaryFunction,
    std::enable_if_t<is_arithmetic<KeyType>::value && is_arithmetic<AccumulatorType>::value
                     && is_binary_functional<BinaryFunction>::value>>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr reduce_by_key_config_params params
            = default_reduce_by_key_config<static_cast<unsigned int>(Arch),
                                           KeyType,
                                           AccumulatorType>{};
    };
};

// Specialization for default config.
template<typename KeyType, typename AccumulatorType, typename BinaryFunction>
struct wrapped_reduce_by_key_config<default_config, KeyType, AccumulatorType, BinaryFunction>
    : wrapped_reduce_by_key_impl<KeyType, AccumulatorType, BinaryFunction>
{};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename ReduceByKeyConfig,
         typename KeyType,
         typename AccumulatorType,
         typename BinaryFunction>
template<target_arch Arch>
constexpr reduce_by_key_config_params
    wrapped_reduce_by_key_config<ReduceByKeyConfig, KeyType, AccumulatorType, BinaryFunction>::
        architecture_config<Arch>::params;

template<typename KeyType, typename AccumulatorType, typename BinaryFunction, typename Enable>
template<target_arch Arch>
constexpr reduce_by_key_config_params
    wrapped_reduce_by_key_impl<KeyType, AccumulatorType, BinaryFunction, Enable>::
        architecture_config<Arch>::params;

template<typename KeyType, typename AccumulatorType, typename BinaryFunction>
template<target_arch Arch>
constexpr reduce_by_key_config_params wrapped_reduce_by_key_impl<
    KeyType,
    AccumulatorType,
    BinaryFunction,
    std::enable_if_t<is_arithmetic<KeyType>::value && is_arithmetic<AccumulatorType>::value
                     && is_binary_functional<BinaryFunction>::value>>::architecture_config<Arch>::
    params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
