// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_

#include <algorithm>
#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"
#include "detail/config/device_segmented_radix_sort.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Selects the appropriate \p WarpSortConfig based on the size of the key type.
///
/// \tparam Key the type of the sorted keys.
/// \tparam MediumWarpSize the logical warp size of the medium segment processing kernel.
template<class Key, unsigned int MediumWarpSize = ROCPRIM_WARP_SIZE_32>
using select_warp_sort_config_t
    = std::conditional_t<sizeof(Key) < 2,
                         DisabledWarpSortConfig,
                         WarpSortConfig<32, //< logical warp size - small kernel
                                        4, //< items per thread - small kernel
                                        256, //< block size - small kernel
                                        3000, //< partitioning threshold
                                        MediumWarpSize, //< logical warp size - medium kernel
                                        4, //< items per thread - medium kernel
                                        256 //< block size - medium kernel
                                        >>;

namespace detail
{

template<class SegmentedRadixSortConfig, typename, typename>
struct wrapped_segmented_radix_sort_config
{
    static_assert(std::is_same<typename SegmentedRadixSortConfig::tag,
                               detail::segmented_radix_sort_config_tag>::value,
                  "Config must be a specialization of struct template segmented_radix_sort_config");

    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr detail::segmented_radix_sort_config_params params
            = SegmentedRadixSortConfig{};
    };
};

template<typename key_type, typename value_type>
struct wrapped_segmented_radix_sort_config<default_config, key_type, value_type>
{
    template<target_arch Arch>
    struct architecture_config
    {
        static constexpr segmented_radix_sort_config_params params
            = detail::default_segmented_radix_sort_config<static_cast<unsigned int>(Arch),
                                                          key_type,
                                                          value_type>{};
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class SegmentedRadixSortConfig, class key_type, class Value>
template<target_arch Arch>
constexpr segmented_radix_sort_config_params
    wrapped_segmented_radix_sort_config<SegmentedRadixSortConfig, key_type, Value>::
        architecture_config<Arch>::params;
template<class key_type, class Value>
template<target_arch Arch>
constexpr segmented_radix_sort_config_params
    wrapped_segmented_radix_sort_config<rocprim::default_config, key_type, Value>::
        architecture_config<Arch>::params;
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_
