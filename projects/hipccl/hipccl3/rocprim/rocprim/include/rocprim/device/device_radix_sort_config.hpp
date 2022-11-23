// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_

#include "config_types.hpp"
#include "detail/config/device_radix_sort.hpp"
#include "detail/config/device_radix_sort_block_sort.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// sub-algorithm onesweep:
template<typename RadixSortOnesweepConfig, typename, typename>
struct wrapped_radix_sort_onesweep_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr radix_sort_onesweep_config_params params = RadixSortOnesweepConfig();
    };
};

template<typename Key, typename Value>
struct wrapped_radix_sort_onesweep_config<default_config, Key, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr radix_sort_onesweep_config_params params
            = default_radix_sort_onesweep_config<static_cast<unsigned int>(Arch), Key, Value>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename RadixSortOnesweepConfig, typename Key, typename Value>
template<target_arch Arch>
constexpr radix_sort_onesweep_config_params
    wrapped_radix_sort_onesweep_config<RadixSortOnesweepConfig, Key, Value>::architecture_config<
        Arch>::params;

template<typename Key, typename Value>
template<target_arch Arch>
constexpr radix_sort_onesweep_config_params
    wrapped_radix_sort_onesweep_config<default_config, Key, Value>::architecture_config<
        Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

// Sub-algorithm block_sort:
template<typename RadixSortBlockSortConfig, typename, typename>
struct wrapped_radix_sort_block_sort_config
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr kernel_config_params params = RadixSortBlockSortConfig();
    };
};

template<typename Key, typename Value>
struct wrapped_radix_sort_block_sort_config<default_config, Key, Value>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr kernel_config_params params
            = default_radix_sort_block_sort_config<static_cast<unsigned int>(Arch), Key, Value>();
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename RadixSortBlockSortConfig, typename Key, typename Value>
template<target_arch Arch>
constexpr kernel_config_params
    wrapped_radix_sort_block_sort_config<RadixSortBlockSortConfig, Key, Value>::architecture_config<
        Arch>::params;

template<typename Key, typename Value>
template<target_arch Arch>
constexpr kernel_config_params wrapped_radix_sort_block_sort_config<default_config, Key, Value>::
    architecture_config<Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
