// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "detail/config/device_radix_sort_block_sort.hpp"
#include "detail/config/device_radix_sort_onesweep.hpp"
#include "detail/device_config_helper.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Configuration of device-level radix sort operation.
///
/// One of three algorithms is used: single sort (launches only a single block),
/// merge sort, or Onesweep.
///
/// \tparam SortSingleConfig Configuration for the single kernel subalgorithm.
///         must be \p kernel_config or \p default_config.
/// \tparam MergeSortConfig Configuration for the merge sort subalgorithm.
///         must be \p merge_sort_config or \p default_config. If \p merge_sort_config, the sorted
///         items per block must be a power of two.
/// \tparam OnesweepConfig Configuration for the Onesweep subalgorithm.
///         must be \p radix_sort_onesweep_config or \p default_config.
/// \tparam MergeSortLimit The largest number of items for which the merge sort algorithm will be
///         used. Note that below this limit, a different algorithm may be used.
template<class SingleSortConfig = default_config,
         class MergeSortConfig  = default_config,
         class OnesweepConfig   = default_config,
         size_t MergeSortLimit  = 1024 * 1024>
struct radix_sort_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    /// \brief Configuration of radix sort single kernel.
    using single_sort_config = SingleSortConfig;
    /// \brief Configuration of merge sort algorithm.
    using merge_sort_config = MergeSortConfig;
    /// \brief Configuration of radix sort onesweep.
    using onesweep_config = OnesweepConfig;
    /// \brief Maximum number of items to use merge sort algorithm.
    static constexpr size_t merge_sort_limit = MergeSortLimit;
#endif
};

namespace detail
{

template<class Key, class Value>
struct radix_sort_onesweep_config_selector
{
    using targets    = radix_sort_onesweep_targets;
    using param_type = radix_sort_onesweep_config_params;

    param_type params;

    template<class Target>
    constexpr radix_sort_onesweep_config_selector(Target)
        : params(radix_sort_onesweep_config_picker<Target, Key, Value>())
    {}
};

template<class Config, class Selector, class Target>
struct radix_sort_onesweep_histogram_config_static_selector
{
    static constexpr auto block_size = target_config<Config, Selector, Target>::params
                                           .radix_sort_onesweep_config_params::histogram.block_size;
};

template<class Config, class Selector, class Target>
struct radix_sort_onesweep_sort_config_static_selector
{
    static constexpr auto block_size = target_config<Config, Selector, Target>::params
                                           .radix_sort_onesweep_config_params::sort.block_size;
};

template<class Key, class Value>
struct radix_sort_block_sort_config_selector
{
    using targets    = radix_sort_block_sort_targets;
    using param_type = kernel_config_params;

    param_type params;

    template<class Target>
    constexpr radix_sort_block_sort_config_selector(Target)
        : params(radix_sort_block_sort_config_picker<Target, Key, Value>())
    {}
};

template<typename Config, class Selector, class Target>
struct radix_sort_block_sort_config_static_selector
{
    static constexpr auto block_size = target_config<Config, Selector, Target>::params.block_size;
};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
