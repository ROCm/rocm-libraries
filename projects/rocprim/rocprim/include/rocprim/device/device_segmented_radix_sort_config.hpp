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

template<class Key, class Value>
struct segmented_radix_sort_config_selector
{
    using targets    = segmented_radix_sort_targets;
    using param_type = segmented_radix_sort_config_params;

    param_type params;

    template<class Target>
    constexpr segmented_radix_sort_config_selector(Target)
        : params(segmented_radix_sort_config_picker<Target, Key, Value>())
    {}
};

template<typename Config, class Selector, class Target>
struct segmented_radix_sort_warp_sort_small_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.warp_sort_config.block_size_small;
};

template<typename Config, class Selector, class Target>
struct segmented_radix_sort_warp_sort_medium_config_static_selector
{
    static constexpr auto block_size
        = target_config<Config, Selector, Target>::params.warp_sort_config.block_size_medium;
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_
