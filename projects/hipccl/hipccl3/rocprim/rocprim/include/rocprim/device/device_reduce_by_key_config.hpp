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

#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level reduce-by-key operation.
///
/// \tparam ScanConfig - configuration of carry-outs scan kernel. Must be \p kernel_config.
/// \tparam ReduceConfig - configuration of the main reduce-by-key kernel. Must be \p kernel_config.
template<
    class ScanConfig,
    class ReduceConfig
>
struct reduce_by_key_config
{
    /// \brief Configuration of carry-outs scan kernel.
    using scan = ScanConfig;
    /// \brief Configuration of the main reduce-by-key kernel.
    using reduce = ReduceConfig;
};

namespace detail
{

template<class Key, class Value>
struct reduce_by_key_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using scan = kernel_config<256, 4>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) <= 8 && sizeof(Value) <= 8),
            reduce_by_key_config<scan, kernel_config<256, 7> >
        >,
        reduce_by_key_config<scan, kernel_config<limit_block_size<256U, sizeof(Key) + sizeof(Value)>::value, ::rocprim::max(1u, 15u / item_scale)> >
    >;
};

template<class Key, class Value>
struct reduce_by_key_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) <= 8 && sizeof(Value) <= 8),
            reduce_by_key_config<scan, kernel_config<256, 10> >
        >,
        reduce_by_key_config<scan, kernel_config<limit_block_size<256U, sizeof(Key) + sizeof(Value)>::value, ::rocprim::max(1u, 15u / item_scale)> >
    >;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_reduce_by_key_config
    : select_arch<
        TargetArch,
        select_arch_case<803, reduce_by_key_config_803<Key, Value> >,
        select_arch_case<900, reduce_by_key_config_900<Key, Value> >,
        reduce_by_key_config_900<Key, Value>
    > { };

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
